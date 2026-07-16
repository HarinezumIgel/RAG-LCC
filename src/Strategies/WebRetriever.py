"""Web search retrieval for augmenting local RAG results with internet content.

Performs a web search using a configurable backend (default: DuckDuckGo) and
returns results as LangchainDocuments compatible with the existing RRF pipeline.

Results enter the retrieval pipeline as a 4th leg alongside Vector, BM25, and
Graph — scored by search-result rank position and re-scored by the cross-encoder
reranker alongside local documents.

Supported backends
------------------
  duckduckgo  (default) — no API key required; uses ``ddgs`` package.
                          Install: pip install ddgs
  brave       — requires API key in ``_WEB_SEARCH.api_key`` (stub, not yet implemented)
  tavily      — requires API key in ``_WEB_SEARCH.api_key`` (stub, not yet implemented)
  bing        — requires Azure API key in ``_WEB_SEARCH.api_key`` (stub, not yet implemented)

Safety
------
    * ``WEB_SEARCH_MODE`` (from Config_Internet_Env.py) must be ``"1"`` or no query is
    ever executed (enforced by QueryParts and RAGChatImpl before this class is
    reached, plus a belt-and-suspenders check in ``query()``).
  * ``_sanitize_query()`` runs four lightweight checks in order.  Heavy
    compliance (banned-phrase embedding + LLM guard) is intentionally omitted
    here because the query has already been through those checks during the
    rewrite/normalisation phase upstream (``QueryParts`` / ``RAGChatImpl``).
    Re-running them on the translated/rewritten text causes false positives
    (e.g. language-detection mismatches on already-normalised queries).

    0. **Hard-blocked categories** (CSAM, WMD synthesis, controlled-substance
       synthesis, human trafficking, solicitation of violence) — unconditional,
       cannot be disabled by any configuration flag.
    1. **Injection / attack patterns** — prompt injection, persona override,
       credential exfiltration, SQL destructive commands, shell metacharacters,
       and control characters.  Gated by ``_WEB_SEARCH.block_on_injection``.
    1.5 **Intent classifier** — weighted scoring via
       ``WebSearchFilter.score_query()``.  Baseline is immutable in code;
       ``WEB_SEARCH_INTENT_EXTENSIONS`` in ``Config_WebSearch.py`` may only add
       entity terms or tighten thresholds.
    2. **Length truncation** — queries longer than ``_WEB_SEARCH.max_query_length``
       are trimmed (logged, not blocked).

  * Every attempt — including blocked ones — is appended to the audit log
    configured by ``_QUERY_LOG`` in Config_RAGChat.py / Config_RAGChatService.py.

On any failure the caller receives an empty list; the query pipeline continues
with local results only — the web leg never aborts a query.
"""

import datetime
import os
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, cast
from urllib.parse import urlparse

from langchain_core.documents.base import Document as LangchainDocument

from Config.Config import Config
from Configuration.Config_Banned import HARDBLOCK_PATTERNS, INJECTION_PATTERNS
from Configuration.Config_WebSearch import WEB_SEARCH_INTENT_EXTENSIONS
from Gui.PrettyWriter import PrettyWriter
from Helpers.PerfLogger import PerfLogger
from Strategies.WebSearchFilter import WebSearchFilter

# Hard-blocked content patterns and injection patterns are defined in
# Configuration.Config_Banned (HARDBLOCK_PATTERNS, INJECTION_PATTERNS).
# WEB_SEARCH_INTENT_EXTENSIONS is defined in Configuration.Config_WebSearch.
# All three are imported above.


class WebRetriever:
    """Fetches web search results and wraps them as LangchainDocuments.

    Not a singleton — no persistent state between queries.  Instantiated once
    in RAGChatImpl.__init__ and reused across queries, but each call to
    ``query()`` is fully independent.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self._backend: str = (
            self.cfg.get_str("_WEB_SEARCH.backend") or "duckduckgo"
        ).lower()
        self._api_key: str = self.cfg.get_str("_WEB_SEARCH.api_key") or ""
        self._max_results: int = self.cfg.get_int("_WEB_SEARCH.max_results") or 5
        self._max_query_length: int = (
            self.cfg.get_int("_WEB_SEARCH.max_query_length") or 500
        )
        boi = self.cfg.get("_WEB_SEARCH.block_on_injection")
        self._block_on_injection: bool = bool(boi) if boi is not None else True
        log_val = self.cfg.get(
            "_QUERY_LOG"
        )  # None = key absent → use default; "" = explicitly disabled
        self._log_path: str = (
            os.path.join("logs", "RAGChat", "queries.log")
            if log_val is None
            else str(log_val)
        )
        intent_log = self.cfg.get("_INTENT_FILTER_LOG")
        intent_log_path: str = (
            os.path.join("logs", "RAGChat", "intent_filter.log")
            if intent_log is None
            else str(intent_log)
        )
        self._intent_filter: WebSearchFilter = WebSearchFilter.get_instance(
            extensions_cfg=WEB_SEARCH_INTENT_EXTENSIONS,
            log_path=intent_log_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        fetch_page_content: bool = False,
        *,
        original_query: str = "",
        collection: str = "",
    ) -> List[LangchainDocument]:
        """Search the web for *query_text* and return ranked LangchainDocuments.

        Parameters
        ----------
        query_text:
            The final query string (after translation/rewrite) used for
            retrieval.
        k:
            Maximum number of results to return.  Falls back to
            ``_WEB_SEARCH.max_results`` when *None*.
        fetch_page_content:
            When *True*, the full page text is fetched instead of the snippet.
        original_query:
            The raw user query before any translation / rewriting — used only
            for audit logging.
        collection:
            The active ChromaDB collection name — written to the audit log.

        Returns
        -------
        List of LangchainDocuments with metadata::

            FileName         — domain name extracted from result URL (e.g., "example.com")
            FilePath         — full result URL
            Source           — "Web"
            Title            — page title from the search result
            retriever_sources — "Web"
            chroma_score     — (k - rank) / k  (rank 0 = best ≈ 1.0)
            chroma_sim       — 1.0
            bm25_score       — 0.0
            graph_score      — 0.0
        """
        effective_k = k if k is not None else self._max_results
        truncated = len(query_text) > self._max_query_length
        orig = original_query or query_text

        # Sanitize — may truncate or block the query
        sanitized, blocked, reason, gate = self._sanitize_query(query_text)

        if blocked:
            self._log_web_query(
                status="BLOCKED",
                gate=gate,
                original_query=orig,
                rewritten_query=query_text,
                effective_query=sanitized,
                collection=collection,
                block_reason=reason,
                truncated=truncated,
            )
            self.pretty.write(
                "W",
                "Web",
                f"Web query blocked ({reason}) \u2014 falling back to local only.",
            )
            return []

        self.pretty.write(
            "I",
            "Web",
            f"Internet search started — backend: {self._backend!r}  query: {sanitized!r}",
        )
        PerfLogger().log(
            "WebRetriever.query",
            f"start web query backend={self._backend!r} q={sanitized[:60]!r}",
        )
        _t0 = time.perf_counter()
        try:
            if self._backend == "duckduckgo":
                raw_results = self._search_duckduckgo(sanitized, effective_k)
            elif self._backend == "brave":
                raw_results = self._search_brave(sanitized, effective_k)
            elif self._backend == "tavily":
                raw_results = self._search_tavily(sanitized, effective_k)
            elif self._backend == "bing":
                raw_results = self._search_bing(sanitized, effective_k)
            else:
                self._log_web_query(
                    status="FAILED",
                    gate="NONE",
                    original_query=orig,
                    rewritten_query=query_text,
                    effective_query=sanitized,
                    collection=collection,
                    block_reason=f"unknown backend '{self._backend}'",
                    truncated=truncated,
                )
                self.pretty.write(
                    "W",
                    "Web",
                    f"Unknown backend '{self._backend}' \u2014 falling back to local only",
                )
                return []
        except Exception as exc:  # noqa: BLE001
            # Missing DuckDuckGo client is a hard configuration/runtime error:
            # do not silently degrade to local-only, fail fast with guidance.
            if "DuckDuckGo client is not installed" in str(exc):
                raise RuntimeError(str(exc)) from exc
            self._log_web_query(
                status="FAILED",
                gate="NONE",
                original_query=orig,
                rewritten_query=query_text,
                effective_query=sanitized,
                collection=collection,
                block_reason=str(exc),
                truncated=truncated,
            )
            self.pretty.write(
                "W",
                "Web",
                f"Web search failed \u2014 falling back to local only ({exc})",
            )
            return []

        # Log after execution — we now know the actual result count
        self._log_web_query(
            status="EXECUTED",
            gate="NONE",
            original_query=orig,
            rewritten_query=query_text,
            effective_query=sanitized,
            collection=collection,
            result_count=len(raw_results),
            truncated=truncated,
        )
        self.pretty.write(
            "I",
            "Web",
            f"Internet search completed — {len(raw_results)} result(s) returned.",
        )

        if not raw_results:
            self.pretty.write(
                "W",
                "Web",
                "Web search returned no results \u2014 falling back to local only",
            )
            return []

        docs: List[LangchainDocument] = []
        n = len(raw_results)
        for rank, result in enumerate(raw_results):
            url: str = result.get("href", "") or result.get("url", "")
            title: str = result.get("title", "")
            snippet: str = (
                result.get("body", "")
                or result.get("snippet", "")
                or result.get("description", "")
                or title
            )

            content: str = snippet
            if fetch_page_content and url:
                fetched = self._fetch_page(url)
                if fetched:
                    content = fetched

            # rank 0 = best result \u2192 score closest to 1.0
            score: float = (n - rank) / n

            # Extract domain from URL for FileName
            parsed = urlparse(url)
            domain = parsed.netloc or url

            meta: Dict[str, Any] = {
                "FileName": domain,  # Domain only (e.g., "example.com")
                "FilePath": url,
                "Source": "Web",
                "Title": title,
                "retriever_sources": "Web",
                "chroma_score": score,
                "chroma_sim": 1.0,
                "bm25_score": 0.0,
                "graph_score": 0.0,
                # Always preserve the original search-engine snippet, even when
                # fetch_page_content replaces page_content with the full page body.
                # RAGChatImpl._rerank() uses this for cross-encoder scoring because
                # the full page's first 512 tokens are often navigation headers and
                # boilerplate, not the relevant content.
                "snippet": snippet,
            }

            doc_id = f"web_{rank}_{hash(url) & 0xFFFFFFFF:08x}"
            docs.append(
                LangchainDocument(
                    page_content=content,
                    metadata=meta,
                    id=doc_id,
                )
            )

        PerfLogger().log(
            "WebRetriever.query",
            f"stop  web query n={len(docs)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        return docs

    # ------------------------------------------------------------------
    # Internal — query safety
    # ------------------------------------------------------------------

    def _sanitize_query(self, query: str) -> tuple[str, bool, str, str]:
        """Validate and sanitize *query* before sending it to a search engine.

        Returns ``(sanitized_query, blocked, reason, gate)``.

        * ``blocked=True``  — query must NOT be executed.
        * ``gate``          — name of the check that triggered the block, or
          ``"NONE"`` when the query is clean.  Values: ``HARD_BLOCK``,
          ``INJECTION``, ``INTENT_SCORE``, ``NONE``.

        Heavy compliance checks (ALGO_FILTER, LLM_GUARD) are omitted — the
        query was already vetted upstream during query rewrite/normalisation.
        Re-running them on the translated/rewritten string causes false
        positives (language-detection mismatches, etc.).

        Checks run in this order:

        0. Hard-blocked categories (CSAM, WMD, etc.) — unconditional.
        1. Injection / attack patterns — gated by ``block_on_injection``.
        1.5 Intent classifier — weighted scoring via ``WebSearchFilter``.
        2. Length truncation.
        """
        # 0. Hard-blocked categories — unconditional, no config can disable this.
        for pattern, reason in HARDBLOCK_PATTERNS:
            if pattern.search(query):
                return (query, True, reason, "HARD_BLOCK")

        # 1. Injection / attack pattern check
        if self._block_on_injection:
            for pattern, reason in INJECTION_PATTERNS:
                if pattern.search(query):
                    return (query, True, reason, "INJECTION")

        # 1.5 Intent classifier — weighted scoring; baseline is immutable in code.
        score, outcome, reasons = self._intent_filter.score_query(query, path="web")
        if outcome == "REFUSE":
            reason_str = ", ".join(reasons) if reasons else "intent score"
            return (query, True, f"intent score {score} — {reason_str}", "INTENT_SCORE")
        # ALLOW_WITH_SAFETY_FRAMING: already logged by score_query(); fall through.

        # 2. Length — truncate rather than block; inform the user via PrettyWriter
        out = query
        if len(out) > self._max_query_length:
            self.pretty.write(
                "W",
                "Web",
                f"Query truncated from {len(out)} to {self._max_query_length} characters "
                "before sending to web search.",
            )
            out = out[: self._max_query_length].rstrip()

        return (out, False, "", "NONE")

    # ------------------------------------------------------------------
    # Internal — audit logging
    # ------------------------------------------------------------------

    def _log_web_query(
        self,
        *,
        status: str,
        gate: str,
        original_query: str,
        rewritten_query: str,
        effective_query: str,
        collection: str,
        block_reason: str = "",
        result_count: int | None = None,
        truncated: bool = False,
    ) -> None:
        """Append one structured record to the web-search audit log.

        Each record is a single human-readable, pipe-separated line::

            timestamp | status=... | gate=... | collection=... | backend=...
                      | results=... | truncated=... | original=... | effective=...
                      | reason=...

        ``status`` values
        -----------------
        EXECUTED  \u2014 query passed all checks and was sent to the search engine.
        BLOCKED   \u2014 query was blocked before execution.
        FAILED    \u2014 execution attempted but the backend raised an exception.

        ``gate`` values
        ---------------
        NONE         — query passed all checks.
        HARD_BLOCK   — caught by ``HARDBLOCK_PATTERNS`` (CSAM, WMD, etc.).
        INJECTION    — caught by ``INJECTION_PATTERNS``.
        INTENT_SCORE — caught by ``WebSearchFilter`` weighted intent classifier.
        ALGO_FILTER  — caught by the algorithm filter chain (banned phrases / embeddings).
        LLM_GUARD    — caught by the LLM content guard; ``reason`` contains the LLM explanation.
        """
        if not self._log_path:
            return
        try:
            parent = os.path.dirname(self._log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            ts = datetime.datetime.now().isoformat(timespec="seconds")

            def _esc(s: str) -> str:
                """Escape pipes and collapse newlines so each record stays one line."""
                return (
                    s.replace("|", "\\|")
                    .replace("\r\n", " ")
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

            results_str = str(result_count) if result_count is not None else "N/A"
            line = (
                f"{ts}"
                f" | status={status:<16}"
                f" | gate={gate:<12}"
                f" | collection={_esc(collection)}"
                f" | backend={self._backend}"
                f" | results={results_str}"
                f" | truncated={truncated}"
                f" | original={_esc(original_query)!r}"
                f" | rewritten={_esc(rewritten_query)!r}"
                f" | effective={_esc(effective_query)!r}"
                f" | reason={_esc(block_reason)!r}"
                "\n"
            )
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:  # noqa: BLE001
            self.pretty.write("W", "Web", f"Could not write to web query log: {exc}")

    # ------------------------------------------------------------------
    # Internal — search backends
    # ------------------------------------------------------------------

    def _search_duckduckgo(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search via DuckDuckGo.

        Supports both package names used in the ecosystem:
        - ddgs (preferred/newer)
        - duckduckgo_search (legacy)
        """
        try:
            from ddgs import DDGS  # type: ignore[import-untyped]
        except ImportError:
            try:
                from duckduckgo_search import \
                    DDGS  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "DuckDuckGo client is not installed. "
                    "Install one of: pip install ddgs  OR  pip install duckduckgo_search"
                ) from exc

        DDGS_cls: Any = DDGS  # pyright: ignore[reportUnknownVariableType]
        with DDGS_cls() as ddgs_client:  # pyright: ignore[reportUnknownVariableType]
            raw_results: Any = cast(Any, ddgs_client).text(query, max_results=k)
            results: List[Dict[str, Any]] = list(raw_results)
        return results

    def _search_brave(self, query: str, k: int) -> List[Dict[str, Any]]:  # noqa: ARG002
        raise NotImplementedError(
            "Brave Search backend is not yet implemented. "
            "Set _WEB_SEARCH.backend to 'duckduckgo' or implement _search_brave() "
            "using the Brave Search API (requires _WEB_SEARCH.api_key)."
        )

    def _search_tavily(
        self, query: str, k: int
    ) -> List[Dict[str, Any]]:  # noqa: ARG002
        raise NotImplementedError(
            "Tavily backend is not yet implemented. "
            "Set _WEB_SEARCH.backend to 'duckduckgo' or implement _search_tavily() "
            "using the Tavily API (requires _WEB_SEARCH.api_key)."
        )

    def _search_bing(self, query: str, k: int) -> List[Dict[str, Any]]:  # noqa: ARG002
        raise NotImplementedError(
            "Bing Search backend is not yet implemented. "
            "Set _WEB_SEARCH.backend to 'duckduckgo' or implement _search_bing() "
            "using the Azure Cognitive Services Bing Search API "
            "(requires _WEB_SEARCH.api_key)."
        )

    # ------------------------------------------------------------------
    # Internal — page fetcher
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str) -> str:
        """Fetch *url* and return its plain-text body (up to ~1 500 words).

        Returns an empty string on any failure so that the caller can fall back
        to the search snippet transparently.
        """
        try:
            import httpx  # type: ignore[import-untyped]

            response = httpx.get(
                url,
                timeout=5.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-LCC/1.0)"},
            )
            response.raise_for_status()

            class _TextExtractor(HTMLParser):
                """Minimal HTML-to-text converter using the stdlib parser."""

                def __init__(self) -> None:
                    super().__init__()
                    self.parts: List[str] = []
                    self._skip: bool = False

                def handle_starttag(self, tag: str, attrs: Any) -> None:  # noqa: ARG002
                    if tag in ("script", "style", "nav", "footer", "header", "aside"):
                        self._skip = True

                def handle_endtag(self, tag: str) -> None:
                    if tag in ("script", "style", "nav", "footer", "header", "aside"):
                        self._skip = False

                def handle_data(self, data: str) -> None:
                    if not self._skip:
                        stripped = data.strip()
                        if stripped:
                            self.parts.append(stripped)

            parser = _TextExtractor()
            parser.feed(response.text)
            text = " ".join(parser.parts)

            # Trim to ~1 500 words to avoid overwhelming the context window.
            words = text.split()
            if len(words) > 1500:
                words = words[:1500]
            return " ".join(words)

        except Exception:  # noqa: BLE001
            return ""
