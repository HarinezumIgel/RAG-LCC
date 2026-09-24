from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Tuple, cast

from Globals.Session import Session
from Gui.Colors import BRIGHT_MAGENTA, CYAN, ORANGE
from Helpers.DebugHelper import DebugHelper
from Helpers.LanguageConfig import get_active_language_codes


@dataclass
class _RetrievalPlan:
    user_query_original: str
    final_query: str
    alternate_queries: list[str]
    retrieve_mode: str
    bm25_query: str
    orig_translated_query_en: str


@dataclass
class _LocalDocs:
    vector_docs: list[Any]
    bm25_docs: list[Any]
    graph_docs: list[Any]
    regex_docs: list[Any]


@dataclass(frozen=True)
class _GuardrailInputs:
    primary_query: str
    guardrail_query: str
    use_guardrail: bool


@dataclass(frozen=True)
class _OriginalQueryLeg:
    enabled: bool
    query: str
    language_bucket: str | None
    reason: str


@dataclass(frozen=True)
class _GraphRegexStageIteration:
    language_bucket: str | None
    file_filter: dict[str, Any] | None
    primary_query: str
    guardrail_query: str


@dataclass(frozen=True)
class _Bm25StageInvocation:
    file_filter: dict[str, Any] | None
    primary_query: str
    guardrail_query: str


@dataclass(frozen=True)
class _IndexedStageResult:
    bm25_docs: list[Any]
    graph_docs: list[Any]
    regex_docs: list[Any]
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int


@dataclass(frozen=True)
class _VectorStageResult:
    vector_docs: list[Any]
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int
    alternate_queries: list[str] | None = None


@dataclass(frozen=True)
class _LocalStageInputs:
    vector_enabled: bool
    bm25_enabled: bool
    graph_enabled: bool
    regex_enabled: bool
    primary_query: str
    guardrail_query: str
    use_guardrail: bool
    alternate_queries: list[str]
    file_filter: dict[str, Any] | None
    original_query_leg: _OriginalQueryLeg


@dataclass(frozen=True)
class _LocalStageResult:
    local_docs: _LocalDocs
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int


@dataclass(frozen=True)
class _PostGatePipelineInputs:
    retrieve_mode: str
    bm25_query: str
    alternate_queries: list[str]
    orig_translated_query_en: str
    user_query_original: str
    resolved_guardrail: _GuardrailInputs


@dataclass(frozen=True)
class _PreGatePipelineResult:
    should_abort: bool
    post_gate_inputs: _PostGatePipelineInputs | None


IndexedSpec = tuple[str, bool, float, Any, int, Any]
_GRAPH_REGEX_LANGUAGE_SAMPLE_LIMIT = 5000


class RetrievalOrchestrator:
    """Coordinate retrieval pipeline stages while preserving RAGChatImpl behavior."""

    _TRACE_RETRIEVER_WIDTH = 10
    _TRACE_LANGUAGE_WIDTH = 7

    def __init__(self, host: Any) -> None:
        self._host = host
        self._last_language_status_signature: (
            tuple[
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ]
            | None
        ) = None
        self._active_indexed_stage_iterations: (
            list[_GraphRegexStageIteration] | None
        ) = None
        self._suppress_indexed_retriever_dispatch_trace = False
        self._indexed_stage_shape_queries = True

    def _trace_language_fallback(self, mySession: Session) -> str:
        """Return the best available session language label for trace lines."""
        for candidate in (
            getattr(mySession, "retrieval_language", None),
            getattr(mySession, "current_query_lang", None),
            getattr(mySession, "user_language", None),
            "default",
        ):
            normalized = self._normalize_language_bucket(candidate)
            if normalized:
                return normalized
        return "default"

    def _write_retrieval_orchestration(
        self,
        *,
        severity: str,
        message: str,
        color: str | None = BRIGHT_MAGENTA,
    ) -> None:
        """Write retrieval orchestration lines with PrettyWriter always-on semantics."""
        pretty = getattr(self._host, "pretty", None)
        if pretty is None or not hasattr(pretty, "write"):
            return

        had_always_on = hasattr(pretty, "always_on")
        previous_always_on = getattr(pretty, "always_on", None)
        if had_always_on:
            try:
                setattr(pretty, "always_on", True)
            except Exception:
                had_always_on = False

        try:
            pretty.write(
                severity,
                "Retrieval Orchestration",
                message,
                color=color,
            )
        finally:
            if had_always_on:
                try:
                    setattr(pretty, "always_on", previous_always_on)
                except Exception:
                    pass

    def _write_multiquery_status(
        self,
        *,
        severity: str,
        message: str,
        color: str | None = CYAN,
    ) -> None:
        """Write MultiQuery lines with PrettyWriter always-on semantics."""
        pretty = getattr(self._host, "pretty", None)
        if pretty is None or not hasattr(pretty, "write"):
            return

        had_always_on = hasattr(pretty, "always_on")
        previous_always_on = getattr(pretty, "always_on", None)
        if had_always_on:
            try:
                setattr(pretty, "always_on", True)
            except Exception:
                had_always_on = False

        try:
            pretty.write(
                severity,
                "MultiQuery",
                message,
                color=color,
            )
        finally:
            if had_always_on:
                try:
                    setattr(pretty, "always_on", previous_always_on)
                except Exception:
                    pass

    def _trace(
        self,
        mySession: Session,
        message: str,
        *,
        retriever_scope: str = "orchestrate",
        language_bucket: str | None = None,
        query_text: str | None = None,
        include_language: bool = True,
    ) -> None:
        scope_text = str(retriever_scope or "orchestrate").strip()
        scope_field = f"{scope_text[: self._TRACE_RETRIEVER_WIDTH]:<{self._TRACE_RETRIEVER_WIDTH}}"
        if include_language:
            language_text = self._normalize_language_bucket(language_bucket)
            if not language_text:
                language_text = self._trace_language_fallback(mySession)
            language_field = f"{language_text:<{self._TRACE_LANGUAGE_WIDTH}}"
            if scope_text.lower() in {"orchestrate", "orchestration"}:
                prefix = f"language: {language_field}"
            else:
                prefix = f"retriever: {scope_field} language: {language_field}"
        else:
            if scope_text.lower() in {"orchestrate", "orchestration"}:
                prefix = ""
            else:
                prefix = f"retriever: {scope_field}"
        if query_text:
            if prefix:
                prefix = f"{prefix} query: {query_text!r}"
            else:
                prefix = f"query: {query_text!r}"
        formatted_message = f"{prefix} {message}".strip()
        self._write_retrieval_orchestration(
            severity="I",
            message=formatted_message,
            color=BRIGHT_MAGENTA,
        )

    def _trace_stage(
        self,
        mySession: Session,
        *,
        stage_name: str,
        details: str = "",
        query_text: str | None = None,
    ) -> None:
        """Emit a normalized stage trace message for retrieval orchestration."""
        message = stage_name.strip()
        details_text = details.strip()
        if details_text:
            message = f"{message} {details_text}"
        self._trace(
            mySession,
            message,
            retriever_scope="orchestrate",
            language_bucket=None,
            query_text=query_text,
        )

    def _trace_query_dispatch(
        self,
        mySession: Session,
        *,
        retriever_scope: str,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        include_language: bool = True,
    ) -> None:
        """Emit one compact trace line for a query dispatch."""
        message = "dispatch query"
        if use_guardrail:
            message = f"{message} guardrail_query={guardrail_query!r}"
        self._trace(
            mySession,
            message,
            retriever_scope=retriever_scope,
            language_bucket=language_bucket,
            query_text=primary_query,
            include_language=include_language,
        )

    @staticmethod
    def _indexed_stage_retriever_labels(
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
    ) -> list[str]:
        """Return ordered retriever labels participating in indexed stage."""
        labels: list[str] = []
        if bm25_specs:
            labels.append("BM25")
        if any(spec[0] == "Graph" for spec in graph_regex_specs):
            labels.append("Graph")
        if any(spec[0] == "Regex" for spec in graph_regex_specs):
            labels.append("Regex")
        return labels

    def _trace_indexed_stage_dispatch(
        self,
        mySession: Session,
        *,
        stage_iterations: list[_GraphRegexStageIteration],
        retriever_labels: list[str],
        use_guardrail: bool,
    ) -> None:
        """Emit one indexed dispatch trace per language-stage iteration."""
        if not stage_iterations or not retriever_labels:
            return

        retriever_group = "/".join(retriever_labels)
        for stage_iteration in stage_iterations:
            language_text = self._normalize_language_bucket(
                stage_iteration.language_bucket
            )
            if not language_text:
                language_text = self._trace_language_fallback(mySession)
            language_field = f"{language_text:<{self._TRACE_LANGUAGE_WIDTH}}"
            message = (
                f"retrievers={retriever_group!r} "
                f"language: {language_field} "
                f"query: {stage_iteration.primary_query!r}"
            )
            if use_guardrail:
                message = (
                    f"{message} " f"guardrail_query={stage_iteration.guardrail_query!r}"
                )
            self._write_retrieval_orchestration(
                severity="I",
                message=message,
                color=BRIGHT_MAGENTA,
            )

    def _write_indexed_stage_query_status(
        self,
        *,
        target_language: str | None,
        was_translated: bool,
        primary_query: str,
        stage_primary_query: str,
        guardrail_query: str,
        stage_guardrail_query: str,
    ) -> None:
        """Emit one indexed-stage query-shaping status block per language."""
        self._write_retrieval_orchestration(
            severity="I",
            message=(
                f"idx_stage l={target_language!r} r='B/G/R' "
                f"tr={int(was_translated)} "
                f"\nprimary_query={primary_query!r}"
                f"\nstage_primary_query={stage_primary_query!r}"
                f"\nguardrail_query={guardrail_query!r}"
                f"\nstage_guardrail_query={stage_guardrail_query!r}"
            ),
            color=BRIGHT_MAGENTA,
        )

    @staticmethod
    def _clip_message_text(text: Any, limit: int = 140) -> str:
        """Return a compact, single-line preview for status messages."""
        clipped = str(text or "").strip().replace("\n", " ")
        if len(clipped) <= limit:
            return clipped
        return clipped[: limit - 3] + "..."

    @staticmethod
    def _is_english_language_bucket(language_bucket: str | None) -> bool:
        """Return True when *language_bucket* represents English."""
        text = str(language_bucket or "").strip().lower()
        if not text:
            return False
        return text == "en" or text.startswith("en-") or text == "english"

    def _resolve_original_query_leg(
        self,
        mySession: Session,
        *,
        primary_query: str,
        guardrail_query: str,
    ) -> _OriginalQueryLeg:
        """Decide whether to run a native-language vector retrieval leg."""
        original_query = str(
            getattr(mySession, "user_query_original", None)
            or getattr(mySession, "query", None)
            or ""
        ).strip()
        if not original_query:
            return _OriginalQueryLeg(
                enabled=False,
                query="",
                language_bucket=None,
                reason="missing original query",
            )

        language_bucket = self._normalize_language_bucket(
            getattr(mySession, "user_language", None)
        )
        if not language_bucket:
            return _OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=None,
                reason="missing user-language detection",
            )

        if self._is_english_language_bucket(language_bucket):
            return _OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="user query already in English",
            )

        lower_original = original_query.lower()
        if lower_original == (primary_query or "").strip().lower():
            return _OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="original query equals primary retrieval query",
            )
        if guardrail_query and lower_original == guardrail_query.strip().lower():
            return _OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="original query equals guardrail retrieval query",
            )

        return _OriginalQueryLeg(
            enabled=True,
            query=original_query,
            language_bucket=language_bucket,
            reason="non-English original query differs from English retrieval legs",
        )

    def _emit_original_query_leg_status(
        self,
        *,
        original_query_leg: _OriginalQueryLeg,
    ) -> None:
        """Emit one status line describing original-language leg planning."""
        status = "on" if original_query_leg.enabled else "off"
        lang = original_query_leg.language_bucket or "unknown"
        query_preview = self._clip_message_text(original_query_leg.query)
        self._write_retrieval_orchestration(
            severity="I",
            message=(
                "original-language vector leg: "
                f"{status} lang={lang} "
                f"query={query_preview!r} "
                f"reason={original_query_leg.reason}"
            ),
            color=BRIGHT_MAGENTA,
        )

    @staticmethod
    def _doc_identity(doc: Any) -> str:
        """Return a stable best-effort identity key for deduping documents."""
        metadata_obj = getattr(doc, "metadata", None)
        metadata = (
            cast(dict[str, Any], metadata_obj) if isinstance(metadata_obj, dict) else {}
        )
        doc_id = metadata.get("id")
        if doc_id is not None:
            doc_id_text = str(doc_id).strip()
            if doc_id_text:
                return f"id:{doc_id_text}"

        page_content = getattr(doc, "page_content", None)
        if isinstance(page_content, str) and page_content:
            return f"text:{page_content}"

        return f"obj:{id(doc)}"

    @staticmethod
    def _append_retriever_source_marker(doc: Any, marker: str) -> None:
        """Append *marker* to doc.metadata['retriever_sources'] when possible."""
        if not marker:
            return
        metadata_obj = getattr(doc, "metadata", None)
        if not isinstance(metadata_obj, dict):
            return

        metadata = cast(dict[str, Any], metadata_obj)
        raw_sources = str(metadata.get("retriever_sources", "") or "").strip()
        if not raw_sources:
            metadata["retriever_sources"] = marker
            return

        source_parts = [part.strip() for part in raw_sources.split(",") if part.strip()]
        if marker in source_parts:
            return
        metadata["retriever_sources"] = ",".join(source_parts + [marker])

    def _merge_docs_with_native_leg(
        self,
        *,
        base_docs: list[Any],
        native_docs: list[Any],
        source_marker: str,
    ) -> tuple[list[Any], int, int]:
        """Merge native-leg docs into *base_docs* with stable de-duplication."""
        merged_docs = list(base_docs)
        existing_by_identity: dict[str, Any] = {
            self._doc_identity(doc): doc for doc in merged_docs
        }

        native_added = 0
        native_overlap = 0
        for doc in native_docs:
            identity = self._doc_identity(doc)
            existing_doc = existing_by_identity.get(identity)
            if existing_doc is not None:
                native_overlap += 1
                self._append_retriever_source_marker(existing_doc, source_marker)
                continue

            self._append_retriever_source_marker(doc, source_marker)
            merged_docs.append(doc)
            existing_by_identity[identity] = doc
            native_added += 1

        return merged_docs, native_added, native_overlap

    def _emit_vector_query_language_flow(self, mySession: Session) -> None:
        """Emit language-flow status only for Vector execution."""
        user_language = (
            self._normalize_language_bucket(getattr(mySession, "user_language", None))
            or "unknown"
        )
        current_query_language = (
            self._normalize_language_bucket(
                getattr(mySession, "current_query_lang", None)
            )
            or "unknown"
        )
        rewrite_language = (
            self._normalize_language_bucket(
                getattr(mySession, "rewrite_language", None)
            )
            or "unknown"
        )
        retrieval_language = (
            self._normalize_language_bucket(
                getattr(mySession, "retrieval_language", None)
            )
            or "unknown"
        )
        self._write_retrieval_orchestration(
            severity="I",
            message=(
                "query language flow: "
                f"user={user_language} "
                f"current={current_query_language} "
                f"rewrite={rewrite_language} "
                f"retrieval={retrieval_language}"
            ),
            color=BRIGHT_MAGENTA,
        )

    def _emit_vector_alternate_queries_status(
        self,
        *,
        alternate_queries: list[str],
    ) -> None:
        """Emit vector alternate-query fanout details as a multiline block."""
        if not alternate_queries:
            return

        query_lines = [
            f"{index}: {query!r}"
            for index, query in enumerate(alternate_queries, start=1)
        ]
        message = f"{len(alternate_queries)} Alternate queries for vector retrieval:"
        if query_lines:
            message = message + "\n" + "\n".join(query_lines)

        self._write_multiquery_status(
            severity="I",
            message=message,
            color=CYAN,
        )

    def _emit_query_language_rewrite_status(
        self,
        mySession: Session,
        *,
        plan: _RetrievalPlan,
    ) -> None:
        """Emit per-turn query-rewrite status block."""

        seed_query_en = self._clip_message_text(plan.orig_translated_query_en)
        rewritten_query = self._clip_message_text(
            getattr(mySession, "rewritten_query", None)
        )
        final_query_en = self._clip_message_text(
            getattr(mySession, "post_rewrite_query_en", None) or plan.final_query
        )

        self._write_retrieval_orchestration(
            severity="I",
            message=(
                "query rewrite flow:"
                f"\nseed_en={seed_query_en!r}"
                f"\nrewritten={rewritten_query!r}"
                f"\nfinal_en={final_query_en!r}"
            ),
            color=BRIGHT_MAGENTA,
        )

    def _prepare_session(self, mySession: Session) -> str:
        return self._host._prepare_session(mySession)

    def _normalize_query(
        self,
        mySession: Session,
        user_query_original: str,
    ) -> tuple[str, list[str]]:
        return self._host._normalize_query(mySession, user_query_original)

    def _check_gates(self, mySession: Session, final_query: str) -> bool:
        return self._host._check_gates(mySession, final_query)

    def _resolve_plan(self, mySession: Session) -> _RetrievalPlan:
        user_query_original = self._prepare_session(mySession)
        final_query, alternate_queries = self._normalize_query(
            mySession,
            user_query_original,
        )

        retrieve_mode: str = (mySession.retrieve_mode or "VECTOR").upper()
        bm25_query: str = mySession.query or ""
        orig_translated_query_en: str = (
            getattr(mySession, "orig_translated_query_en", None)
            or getattr(mySession, "seed_retrieval_query", None)
            or getattr(mySession, "t1_query", None)
            or ""
        )
        return _RetrievalPlan(
            user_query_original=user_query_original,
            final_query=final_query,
            alternate_queries=alternate_queries,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            orig_translated_query_en=orig_translated_query_en,
        )

    def run_web_retriever(
        self,
        mySession: Session,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Run web retrieval and optional BM25/cosine pre-filters."""
        web_docs: list[Any] = []
        web_triggered: bool = mySession.web_search or retrieve_mode == "WEB"
        if not web_triggered:
            return web_docs

        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        if web_mode == "0":
            if retrieve_mode == "WEB":
                self._host.pretty.write(
                    "W",
                    "Web",
                    'retrieve_mode=WEB requested but WEB_SEARCH_MODE="0" '
                    "— no results will be returned.",
                )
            else:
                self._host.pretty.write(
                    "W",
                    "Web",
                    'Web search blocked by administrator (WEB_SEARCH_MODE = "0") — skipping',
                )
            return web_docs
        if web_mode != "1":
            self._host.pretty.write(
                "W",
                "Web",
                f"Web search blocked (WEB_SEARCH_MODE = {web_mode!r}) — skipping",
            )
            return web_docs

        label = (
            "retrieve_mode=WEB — querying web only..."
            if retrieve_mode == "WEB"
            else "Querying web search..."
        )
        self._host.pretty.write("I", "Web", label)
        self._host.perf_logger.log("RAGChatImpl._retrieve", "chat", "start web block")
        _t_web = time.perf_counter()
        web_docs = self._host.web_retriever.query(
            mySession.query or "",
            k=self._host.cfg.get_int("_WEB_SEARCH.max_results") or 5,
            fetch_page_content=bool(mySession.fetch_page_content),
            original_query=user_query_original,
            collection=mySession.collection_name or "",
        )
        self._host.pretty.write(
            "O",
            "Web",
            f"Web search returned {len(web_docs)} results",
            color=CYAN,
        )
        self._host.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  web block n={len(web_docs)} elapsed={time.perf_counter() - _t_web:.3f}s",
        )
        if DebugHelper.check_session(mySession, 10):
            self._host._print_web_debug(web_docs)

        # Apply optional pre-filters to trim noisy web snippets before merge.
        pre_bm25: float = self._host.cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        pre_cosine: float = (
            self._host.cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        )
        if web_docs and (pre_bm25 > 0.0 or pre_cosine > 0.0):
            if DebugHelper.check_session(mySession, 30):
                self._host.pretty.write(
                    "D",
                    "WebPreFilter",
                    f"Pre-filtering {len(web_docs)} web result(s) — "
                    f"bm25_pre_filter={pre_bm25:.3f}, "
                    f"cosine_pre_filter={pre_cosine:.3f}",
                    color=CYAN,
                )
            before_pre = len(web_docs)
            if pre_bm25 > 0.0:
                docs_before_bm25 = web_docs
                web_docs = self._host.web_pre_filter.bm25_prefilter(
                    web_docs,
                    mySession.query or "",
                )
                if DebugHelper.check_session(mySession, 30):
                    self._host.pretty.write(
                        "D",
                        "WebPreFilter",
                        f"After BM25 pre-filter: {len(web_docs)}/{before_pre} kept",
                        color=CYAN,
                    )
                    kept_set = set(id(d) for d in web_docs)
                    dropped_bm25 = [
                        d for d in docs_before_bm25 if id(d) not in kept_set
                    ]
                    self._host._print_web_prefilter_debug(
                        web_docs,
                        dropped_bm25,
                        "BM25",
                        pre_bm25,
                    )
            if pre_cosine > 0.0 and web_docs:
                pre_query_vec: list[float] = self._host.embedder.embed_query(
                    mySession.query or ""
                )
                docs_before_cosine = web_docs
                web_docs = self._host.web_pre_filter.cosine_prefilter(
                    web_docs,
                    mySession.query or "",
                    query_vec=pre_query_vec,
                )
                if DebugHelper.check_session(mySession, 30):
                    self._host.pretty.write(
                        "D",
                        "WebPreFilter",
                        f"After cosine pre-filter: {len(web_docs)}/{before_pre} kept",
                        color=CYAN,
                    )
                    kept_set = set(id(d) for d in web_docs)
                    dropped_cosine = [
                        d for d in docs_before_cosine if id(d) not in kept_set
                    ]
                    self._host._print_web_prefilter_debug(
                        web_docs,
                        dropped_cosine,
                        "Cosine",
                        pre_cosine,
                    )

        return web_docs

    def _merge_and_build_context(
        self,
        mySession: Session,
        local_docs: _LocalDocs,
        web_docs: list[Any],
    ) -> Tuple[str, int]:
        chosen = self._host._merge_and_select(
            mySession,
            local_docs.vector_docs,
            local_docs.bm25_docs,
            local_docs.graph_docs,
            local_docs.regex_docs,
            web_docs,
        )
        return self._host._build_context(mySession, chosen)

    def _run_merge_and_context_stage(
        self,
        mySession: Session,
        local_docs: _LocalDocs,
        web_docs: list[Any],
    ) -> Tuple[str, int]:
        """Merge docs and build final retrieval context."""
        self._trace_stage(
            mySession,
            stage_name="merge candidates and build context",
        )
        context, count = self._merge_and_build_context(mySession, local_docs, web_docs)
        self._trace(
            mySession,
            f"retrieval completed chosen={count} context_chars={len(context)}",
        )
        return context, count

    def _resolve_guardrail_inputs(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> _GuardrailInputs:
        """Resolve guardrail query inputs using host helper semantics."""
        (
            primary_query,
            guardrail_query,
            use_guardrail,
        ) = self._host._resolve_guardrail_queries(
            bm25_query=bm25_query,
            orig_translated_query_en=orig_translated_query_en,
        )
        return _GuardrailInputs(
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
        )

    @staticmethod
    def _is_web_only_mode(retrieve_mode: str) -> bool:
        """Return True when retrieval mode requests only the web leg."""
        return str(retrieve_mode or "").strip().upper() == "WEB"

    @staticmethod
    def _empty_local_docs() -> _LocalDocs:
        """Return an empty local-doc container for web-only retrieval."""
        return _LocalDocs(
            vector_docs=[],
            bm25_docs=[],
            graph_docs=[],
            regex_docs=[],
        )

    @staticmethod
    def _partition_indexed_specs(
        indexed_specs: list[IndexedSpec],
    ) -> tuple[list[IndexedSpec], list[IndexedSpec]]:
        """Split indexed specs into BM25 and Graph/Regex execution stages."""
        active_specs = [spec for spec in indexed_specs if spec[1] and spec[2] != 0.0]
        bm25_specs = [spec for spec in active_specs if spec[0] == "BM25"]
        graph_regex_specs = [
            spec for spec in active_specs if spec[0] in ("Graph", "Regex")
        ]
        return bm25_specs, graph_regex_specs

    def _run_indexed_specs_stage(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run one indexed-spec stage through the host retriever executor."""
        empty_docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        if not indexed_specs:
            return empty_docs, 0, 0

        stage_docs, top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._host._run_indexed_local_retrievers(
                mySession,
                indexed_specs=indexed_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
        )

        normalized_docs: dict[str, list[Any]] = {
            "BM25": list(stage_docs.get("BM25", [])),
            "Graph": list(stage_docs.get("Graph", [])),
            "Regex": list(stage_docs.get("Regex", [])),
        }
        return normalized_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _stage_invocation_values(
        self,
        stage_invocation: Any,
    ) -> tuple[str | None, dict[str, Any] | None, str, str]:
        """Extract stage language/filter/query values from an invocation object."""
        stage_file_filter_obj = getattr(stage_invocation, "file_filter", None)
        stage_file_filter: dict[str, Any] | None = (
            cast(dict[str, Any], stage_file_filter_obj)
            if isinstance(stage_file_filter_obj, dict)
            else None
        )
        stage_primary_query = str(getattr(stage_invocation, "primary_query", "") or "")
        stage_guardrail_query = str(
            getattr(stage_invocation, "guardrail_query", "") or ""
        )
        stage_language_bucket = self._normalize_language_bucket(
            getattr(stage_invocation, "language_bucket", None)
        )
        if stage_language_bucket is None:
            stage_language_bucket = self._language_bucket_from_filter(stage_file_filter)
        return (
            stage_language_bucket,
            stage_file_filter,
            stage_primary_query,
            stage_guardrail_query,
        )

    def _run_indexed_stage_group(
        self,
        mySession: Session,
        *,
        stage_invocations: list[Any],
        specs: list[IndexedSpec],
        use_guardrail: bool,
        dispatch_per_spec: bool,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run indexed stage invocations for one spec group.

        ``dispatch_per_spec=False`` executes all ``specs`` together per stage
        invocation (BM25 behavior). ``dispatch_per_spec=True`` executes one spec
        at a time per stage invocation (Graph/Regex behavior).
        """
        merged_docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        if not specs or not stage_invocations:
            return merged_docs, 0, 0

        total_top_k_orig_query_en = 0
        total_top_k_post_rewrite_query_en = 0
        for stage_invocation in stage_invocations:
            (
                stage_language_bucket,
                stage_file_filter,
                stage_primary_query,
                stage_guardrail_query,
            ) = self._stage_invocation_values(stage_invocation)
            spec_groups: list[list[IndexedSpec]] = (
                [[spec] for spec in specs] if dispatch_per_spec else [specs]
            )
            for stage_specs in spec_groups:
                if not stage_specs:
                    continue
                dispatch_scope = (
                    str(stage_specs[0][0])
                    if len(stage_specs) == 1
                    else "/".join(str(spec[0]) for spec in stage_specs)
                )
                if not self._suppress_indexed_retriever_dispatch_trace:
                    self._trace_query_dispatch(
                        mySession,
                        retriever_scope=dispatch_scope,
                        language_bucket=stage_language_bucket,
                        primary_query=stage_primary_query,
                        guardrail_query=stage_guardrail_query,
                        use_guardrail=use_guardrail,
                    )

                (
                    stage_docs,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                ) = self._run_indexed_specs_stage(
                    mySession,
                    indexed_specs=stage_specs,
                    primary_query=stage_primary_query,
                    guardrail_query=stage_guardrail_query,
                    use_guardrail=use_guardrail,
                    file_filter=stage_file_filter,
                )
                merged_docs = self._merge_indexed_stage_docs(merged_docs, stage_docs)
                (
                    total_top_k_orig_query_en,
                    total_top_k_post_rewrite_query_en,
                ) = self._host._accumulate_topk_counts(
                    total_top_k_orig_query_en,
                    total_top_k_post_rewrite_query_en,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                )

        return (
            merged_docs,
            total_top_k_orig_query_en,
            total_top_k_post_rewrite_query_en,
        )

    def _bm25_stage_invocations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[_Bm25StageInvocation]:
        """Plan BM25 stage invocations using language-stage iterations.

        Legacy default behavior is preserved when no explicit language filter is
        set and the detected stage set collapses to a single English/default
        bucket: BM25 runs once with the base filter and unshaped query.
        """
        stage_iterations = self._shared_graph_regex_stage_iterations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )

        explicit_language = self._language_bucket_from_filter(file_filter)
        if (
            stage_iterations
            and explicit_language is None
            and len(stage_iterations) == 1
        ):
            only_language = self._normalize_language_bucket(
                stage_iterations[0].language_bucket
            )
            if (
                only_language is None
                or only_language == "en"
                or only_language.startswith("en-")
            ):
                return [
                    self._bm25_stage_invocation(
                        mySession,
                        file_filter=file_filter,
                        primary_query=primary_query,
                        guardrail_query=guardrail_query,
                    )
                ]

        if stage_iterations:
            return [
                _Bm25StageInvocation(
                    file_filter=stage_iteration.file_filter,
                    primary_query=stage_iteration.primary_query,
                    guardrail_query=stage_iteration.guardrail_query,
                )
                for stage_iteration in stage_iterations
            ]

        return [
            self._bm25_stage_invocation(
                mySession,
                file_filter=file_filter,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
            )
        ]

    def _bm25_stage_invocation(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> _Bm25StageInvocation:
        """Plan the single BM25 stage invocation.

        This seam keeps BM25 stage execution symmetric with Graph/Regex stage
        iteration planning while preserving current behavior.
        """
        _ = mySession
        return _Bm25StageInvocation(
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )

    def _normalize_language_bucket(self, language: Any) -> str | None:
        """Normalize a language label into a lower-case bucket code."""
        host_normalizer = getattr(self._host, "_normalize_language_bucket", None)
        if callable(host_normalizer):
            try:
                normalized_obj = host_normalizer(language)
            except Exception:
                normalized_obj = None
            normalized_text = str(normalized_obj or "").strip().lower()
            if normalized_text:
                return normalized_text

        text = str(language or "").strip().lower()
        if not text:
            return None

        shared = getattr(self._host, "_shared", None)
        mapping_obj = getattr(shared, "lang_name_to_code", None)
        if isinstance(mapping_obj, dict):
            mapping = cast(dict[str, Any], mapping_obj)
            mapped = mapping.get(text)
            if mapped is not None:
                mapped_text = str(mapped).strip().lower()
                if mapped_text:
                    return mapped_text

        return text

    def _language_bucket_from_filter(
        self,
        file_filter: dict[str, Any] | None,
    ) -> str | None:
        """Extract an explicit Language filter bucket if one is set."""
        if not isinstance(file_filter, dict) or "Language" not in file_filter:
            return None
        language_value = file_filter.get("Language")
        if isinstance(language_value, dict):
            language_dict = cast(dict[str, Any], language_value)
            eq_value = language_dict.get("$eq")
            return self._normalize_language_bucket(eq_value)
        return self._normalize_language_bucket(language_value)

    @staticmethod
    def _remove_language_from_filter(
        file_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Drop Language from a flat filter so discovery sees all languages."""
        if not isinstance(file_filter, dict):
            return None
        without_language = {k: v for k, v in file_filter.items() if k != "Language"}
        return without_language or None

    def _discover_graph_regex_languages(
        self,
        file_filter: dict[str, Any] | None,
    ) -> list[str]:
        """Discover normalized language values from collection metadata."""
        collection = getattr(self._host, "collection", None)
        if collection is None or not hasattr(collection, "get"):
            return []

        where_filter = self._remove_language_from_filter(file_filter)
        try:
            result = collection.get(
                include=["metadatas"],
                where=where_filter,
                limit=_GRAPH_REGEX_LANGUAGE_SAMPLE_LIMIT,
            )
        except Exception:
            return []

        result_dict: dict[str, Any] | None = (
            cast(dict[str, Any], result) if isinstance(result, dict) else None
        )
        metadatas_obj = (
            result_dict.get("metadatas") if result_dict is not None else None
        )
        if not isinstance(metadatas_obj, list):
            return []
        metadatas = cast(list[Any], metadatas_obj)

        languages: set[str] = set()
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            meta_dict = cast(dict[str, Any], meta)
            normalized = self._normalize_language_bucket(meta_dict.get("Language"))
            if normalized:
                languages.add(normalized)

        return sorted(languages)

    def _active_configured_language_codes(self) -> set[str]:
        """Return active language codes from shared ACTIVE_LANGUAGES config."""
        cfg = getattr(self._host, "cfg", None)
        if cfg is None:
            return {"en"}

        configured_codes = get_active_language_codes(cfg)
        active_codes: set[str] = set()
        for code in configured_codes:
            normalized = self._normalize_language_bucket(code)
            if normalized:
                active_codes.add(normalized)

        return active_codes or {"en"}

    def _emit_vector_store_language_status(self, mySession: Session) -> None:
        """Report whether vector-store languages are fully covered by config."""
        vector_store_languages = self._discover_graph_regex_languages(None)
        if not vector_store_languages:
            return

        active_codes = sorted(self._active_configured_language_codes())
        undefined_languages = sorted(
            lang for lang in vector_store_languages if lang not in set(active_codes)
        )

        collection_name = str(
            getattr(self._host, "collection_name", None)
            or getattr(mySession, "collection_name", None)
            or ""
        )
        signature = (
            collection_name,
            tuple(vector_store_languages),
            tuple(active_codes),
            tuple(undefined_languages),
        )
        if signature == self._last_language_status_signature:
            return
        self._last_language_status_signature = signature

        languages_label = ", ".join(vector_store_languages)
        active_label = ", ".join(active_codes)
        self._write_retrieval_orchestration(
            severity="I",
            message=(
                "discovered corpus language buckets: "
                f"[{languages_label}] active config buckets: [{active_label}]"
            ),
            color=BRIGHT_MAGENTA,
        )

        if undefined_languages:
            pretty = getattr(self._host, "pretty", None)
            if pretty is None or not hasattr(pretty, "write"):
                return
            undefined_label = ", ".join(undefined_languages)
            pretty.write(
                "W",
                "LanguageConfig",
                "Vector-store document languages "
                f"[{languages_label}] include codes not active in "
                "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES: "
                f"{undefined_label}",
                color=ORANGE,
            )
            return

        return

    def _relevant_graph_regex_languages(
        self,
        mySession: Session,
        available_languages: list[str],
    ) -> list[str]:
        """Select query-relevant languages from the discovered set."""
        available = {lang for lang in available_languages if lang}
        if not available:
            return []

        current_query_lang = cast(
            str | None, getattr(mySession, "current_query_lang", None)
        )
        user_language = cast(str | None, getattr(mySession, "user_language", None))
        retrieval_language = cast(
            str | None, getattr(mySession, "retrieval_language", None)
        )
        candidates: list[str | None] = [
            current_query_lang,
            user_language,
            retrieval_language,
            "english",
            "en",
        ]

        selected: list[str] = []
        for candidate in candidates:
            normalized = self._normalize_language_bucket(candidate)
            if normalized and normalized in available and normalized not in selected:
                selected.append(normalized)
        return selected

    def _plan_graph_regex_language_buckets(
        self,
        mySession: Session,
        file_filter: dict[str, Any] | None,
    ) -> list[str]:
        """Plan language buckets for Graph/Regex stage iterations."""
        self._emit_vector_store_language_status(mySession)

        explicit_language = self._language_bucket_from_filter(file_filter)
        available_languages = self._discover_graph_regex_languages(file_filter)
        active_languages = self._active_configured_language_codes()
        allowed_languages = sorted(
            lang for lang in available_languages if lang in active_languages
        )

        if explicit_language:
            return [explicit_language] if explicit_language in allowed_languages else []

        relevant_languages = self._relevant_graph_regex_languages(
            mySession,
            allowed_languages,
        )
        if relevant_languages:
            # Keep query-relevant buckets first, then run remaining active/present
            # buckets so multilingual corpora are still fully covered.
            relevant = [
                lang for lang in relevant_languages if lang in allowed_languages
            ]
            relevant_set = set(relevant)
            remaining = [lang for lang in allowed_languages if lang not in relevant_set]
            return relevant + remaining

        # Loop only over languages active in config and present in the vector store.
        return [lang for lang in allowed_languages]

    @staticmethod
    def _build_graph_regex_stage_filter(
        base_file_filter: dict[str, Any] | None,
        language_bucket: str | None,
    ) -> dict[str, Any] | None:
        """Build the file filter for one Graph/Regex language bucket."""
        staged_filter = dict(base_file_filter or {})
        if language_bucket:
            staged_filter["Language"] = language_bucket
        return staged_filter or None

    def _graph_regex_stage_filters(
        self,
        mySession: Session,
        file_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any] | None]:
        """Return Graph/Regex stage filters for each execution iteration."""
        stage_filters: list[dict[str, Any] | None] = []
        for language_bucket in self._plan_graph_regex_language_buckets(
            mySession,
            file_filter,
        ):
            stage_filters.append(
                self._build_graph_regex_stage_filter(file_filter, language_bucket)
            )
        return stage_filters

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
    ) -> tuple[str, str]:
        """Return per-language query forms for one Graph/Regex iteration.

        Delegates query-shaping policy to the host seam when available.
        This keeps translation/backend policy centralized in RAGChatImpl.
        """
        target_language = self._normalize_language_bucket(language_bucket)
        stage_primary_query = primary_query
        stage_guardrail_query = guardrail_query

        host_shaper = getattr(self._host, "_shape_graph_regex_stage_queries", None)
        if callable(host_shaper):
            try:
                shaped_obj = host_shaper(
                    mySession,
                    language_bucket=target_language,
                    primary_query=primary_query,
                    guardrail_query=guardrail_query,
                )
                if isinstance(shaped_obj, tuple):
                    shaped_tuple = cast(tuple[Any, ...], shaped_obj)
                    if len(shaped_tuple) == 2:
                        host_primary_obj, host_guardrail_obj = shaped_tuple
                        stage_primary_query = (
                            str(host_primary_obj or "").strip() or primary_query
                        )
                        stage_guardrail_query = (
                            str(host_guardrail_obj or "").strip() or guardrail_query
                        )
            except Exception:
                stage_primary_query = primary_query
                stage_guardrail_query = guardrail_query

        was_translated = (
            stage_primary_query != primary_query
            or stage_guardrail_query != guardrail_query
        )
        self._write_indexed_stage_query_status(
            target_language=target_language,
            was_translated=was_translated,
            primary_query=primary_query,
            stage_primary_query=stage_primary_query,
            guardrail_query=guardrail_query,
            stage_guardrail_query=stage_guardrail_query,
        )
        return stage_primary_query, stage_guardrail_query

    def _graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[_GraphRegexStageIteration]:
        """Plan Graph/Regex iterations with staged filters and shaped queries."""
        iterations: list[_GraphRegexStageIteration] = []
        shape_queries = bool(self._indexed_stage_shape_queries)
        for stage_file_filter in self._graph_regex_stage_filters(
            mySession, file_filter
        ):
            stage_language_bucket = self._language_bucket_from_filter(stage_file_filter)
            if shape_queries:
                stage_primary_query, stage_guardrail_query = (
                    self._shape_graph_regex_stage_queries(
                        mySession,
                        language_bucket=stage_language_bucket,
                        primary_query=primary_query,
                        guardrail_query=guardrail_query,
                    )
                )
            else:
                stage_primary_query = primary_query
                stage_guardrail_query = guardrail_query
                self._write_indexed_stage_query_status(
                    target_language=self._normalize_language_bucket(
                        stage_language_bucket
                    ),
                    was_translated=False,
                    primary_query=primary_query,
                    stage_primary_query=stage_primary_query,
                    guardrail_query=guardrail_query,
                    stage_guardrail_query=stage_guardrail_query,
                )
            iterations.append(
                _GraphRegexStageIteration(
                    language_bucket=stage_language_bucket,
                    file_filter=stage_file_filter,
                    primary_query=stage_primary_query,
                    guardrail_query=stage_guardrail_query,
                )
            )
        return iterations

    def _shared_graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[_GraphRegexStageIteration]:
        """Return precomputed language-stage iterations for indexed retrievers."""
        active_iterations = self._active_indexed_stage_iterations
        if active_iterations is not None:
            return active_iterations
        return self._graph_regex_stage_iterations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )

    @staticmethod
    def _merge_indexed_stage_docs(
        first_stage_docs: dict[str, list[Any]],
        second_stage_docs: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """Merge two indexed-doc maps with stable BM25/Graph/Regex keys."""
        return {
            "BM25": list(first_stage_docs.get("BM25", []))
            + list(second_stage_docs.get("BM25", [])),
            "Graph": list(first_stage_docs.get("Graph", []))
            + list(second_stage_docs.get("Graph", [])),
            "Regex": list(first_stage_docs.get("Regex", []))
            + list(second_stage_docs.get("Regex", [])),
        }

    def _run_indexed_retrieval_stages(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> _IndexedStageResult:
        """Run BM25 and Graph/Regex indexed stages and merge their outputs."""
        bm25_specs, graph_regex_specs = self._partition_indexed_specs(indexed_specs)

        indexed_docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        top_k_orig_query_en = 0
        top_k_post_rewrite_query_en = 0

        shared_stage_iterations: list[_GraphRegexStageIteration] | None = None
        if bm25_specs or graph_regex_specs:
            shared_stage_iterations = self._graph_regex_stage_iterations(
                mySession,
                file_filter=file_filter,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
            )

        indexed_retriever_labels = self._indexed_stage_retriever_labels(
            bm25_specs,
            graph_regex_specs,
        )
        suppress_dispatch_trace = bool(
            shared_stage_iterations and indexed_retriever_labels
        )
        if suppress_dispatch_trace:
            self._trace_indexed_stage_dispatch(
                mySession,
                stage_iterations=list(shared_stage_iterations or []),
                retriever_labels=indexed_retriever_labels,
                use_guardrail=use_guardrail,
            )

        previous_stage_iterations = self._active_indexed_stage_iterations
        previous_suppress_dispatch_trace = (
            self._suppress_indexed_retriever_dispatch_trace
        )
        self._active_indexed_stage_iterations = shared_stage_iterations
        self._suppress_indexed_retriever_dispatch_trace = suppress_dispatch_trace
        try:
            stage_groups: list[tuple[list[Any], list[IndexedSpec], bool]] = [
                (
                    cast(
                        list[Any],
                        self._bm25_stage_invocations(
                            mySession,
                            file_filter=file_filter,
                            primary_query=primary_query,
                            guardrail_query=guardrail_query,
                        ),
                    ),
                    bm25_specs,
                    False,
                ),
                (
                    cast(
                        list[Any],
                        self._shared_graph_regex_stage_iterations(
                            mySession,
                            file_filter=file_filter,
                            primary_query=primary_query,
                            guardrail_query=guardrail_query,
                        ),
                    ),
                    graph_regex_specs,
                    True,
                ),
            ]
            for stage_invocations, specs, dispatch_per_spec in stage_groups:
                (
                    stage_docs,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                ) = self._run_indexed_stage_group(
                    mySession,
                    stage_invocations=stage_invocations,
                    specs=specs,
                    use_guardrail=use_guardrail,
                    dispatch_per_spec=dispatch_per_spec,
                )
                indexed_docs = self._merge_indexed_stage_docs(indexed_docs, stage_docs)
                (
                    top_k_orig_query_en,
                    top_k_post_rewrite_query_en,
                ) = self._host._accumulate_topk_counts(
                    top_k_orig_query_en,
                    top_k_post_rewrite_query_en,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                )
        finally:
            self._active_indexed_stage_iterations = previous_stage_iterations
            self._suppress_indexed_retriever_dispatch_trace = (
                previous_suppress_dispatch_trace
            )

        return _IndexedStageResult(
            bm25_docs=indexed_docs["BM25"],
            graph_docs=indexed_docs["Graph"],
            regex_docs=indexed_docs["Regex"],
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _run_indexed_retrieval_with_shape_mode(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
        shape_queries: bool,
    ) -> _IndexedStageResult:
        """Run indexed retrieval while temporarily toggling query shaping."""
        previous_shape_queries = self._indexed_stage_shape_queries
        self._indexed_stage_shape_queries = bool(shape_queries)
        try:
            return self._run_indexed_retrieval_stages(
                mySession,
                indexed_specs=indexed_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
        finally:
            self._indexed_stage_shape_queries = previous_shape_queries

    def _run_vector_stage(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> _VectorStageResult:
        """Run vector retrieval stage and return docs plus top-k counters."""
        vector_weight = float(
            mySession.vector_weight if mySession.vector_weight is not None else 1.0
        )
        if not vector_enabled or vector_weight == 0.0:
            return _VectorStageResult(
                vector_docs=[],
                top_k_orig_query_en=0,
                top_k_post_rewrite_query_en=0,
                alternate_queries=[],
            )

        self._emit_vector_query_language_flow(mySession)

        self._trace_query_dispatch(
            mySession,
            retriever_scope="Vector",
            language_bucket=None,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
            include_language=False,
        )

        (
            vector_docs,
            vector_top_k_orig_query_en,
            vector_top_k_post_rewrite_query_en,
        ) = self._host._run_vector_retriever_with_guardrail(
            mySession=mySession,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
            alternate_queries=alternate_queries,
        )
        return _VectorStageResult(
            vector_docs=vector_docs,
            top_k_orig_query_en=vector_top_k_orig_query_en,
            top_k_post_rewrite_query_en=vector_top_k_post_rewrite_query_en,
            alternate_queries=list(alternate_queries),
        )

    def _store_original_query_leg_stats(
        self,
        mySession: Session,
        *,
        original_query_leg: _OriginalQueryLeg,
        vector_hits: int,
        vector_added: int,
        vector_overlap: int,
        bm25_hits: int = 0,
        bm25_added: int = 0,
        bm25_overlap: int = 0,
        graph_hits: int = 0,
        graph_added: int = 0,
        graph_overlap: int = 0,
        regex_hits: int = 0,
        regex_added: int = 0,
        regex_overlap: int = 0,
    ) -> None:
        """Persist original-language leg diagnostics on the session object."""
        mySession.original_query_leg_enabled = bool(original_query_leg.enabled)  # type: ignore[attr-defined]
        mySession.original_query_leg_query = original_query_leg.query  # type: ignore[attr-defined]
        mySession.original_query_leg_language = original_query_leg.language_bucket  # type: ignore[attr-defined]
        mySession.original_query_leg_reason = original_query_leg.reason  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_hits = int(vector_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_added = int(vector_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_overlap = int(vector_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_hits = int(bm25_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_added = int(bm25_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_overlap = int(bm25_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_hits = int(graph_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_added = int(graph_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_overlap = int(graph_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_hits = int(regex_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_added = int(regex_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_overlap = int(regex_overlap)  # type: ignore[attr-defined]

    def _run_original_query_vector_leg(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        original_query_leg: _OriginalQueryLeg,
    ) -> list[Any]:
        """Run native-language vector retrieval leg and return native docs."""
        if not original_query_leg.enabled:
            return []
        if not vector_enabled:
            self._write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language vector leg skipped "
                    "(vector retriever disabled by mode or weight)"
                ),
                color=BRIGHT_MAGENTA,
            )
            return []

        self._trace_query_dispatch(
            mySession,
            retriever_scope="VectorRaw",
            language_bucket=original_query_leg.language_bucket,
            primary_query=original_query_leg.query,
            guardrail_query="",
            use_guardrail=False,
            include_language=True,
        )

        (
            native_docs,
            _native_top_k_orig,
            _native_top_k_post,
        ) = self._host._run_vector_retriever_with_guardrail(
            mySession=mySession,
            primary_query=original_query_leg.query,
            guardrail_query="",
            use_guardrail=False,
            alternate_queries=[],
        )
        _ = (_native_top_k_orig, _native_top_k_post)

        self._write_retrieval_orchestration(
            severity="I",
            message=(
                "original-language vector leg completed "
                f"lang={original_query_leg.language_bucket or 'unknown'} "
                f"hits={len(native_docs)}"
            ),
            color=BRIGHT_MAGENTA,
        )

        return native_docs

    def _resolve_local_stage_inputs(
        self,
        mySession: Session,
        *,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: _GuardrailInputs | None,
    ) -> _LocalStageInputs:
        """Resolve local stage flags, queries, and shared filters once."""
        (
            vector_enabled,
            bm25_enabled,
            graph_enabled,
            regex_enabled,
        ) = self._host._mode_flags_for_local_retrievers(retrieve_mode)

        guardrail_inputs = resolved_guardrail
        if guardrail_inputs is None:
            guardrail_inputs = self._resolve_guardrail_inputs(
                bm25_query=bm25_query,
                orig_translated_query_en=orig_translated_query_en,
            )

        original_query_leg = self._resolve_original_query_leg(
            mySession,
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
        )

        return _LocalStageInputs(
            vector_enabled=vector_enabled,
            bm25_enabled=bm25_enabled,
            graph_enabled=graph_enabled,
            regex_enabled=regex_enabled,
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
            use_guardrail=guardrail_inputs.use_guardrail,
            alternate_queries=list(alternate_queries),
            file_filter=self._host._resolve_file_filter(mySession),
            original_query_leg=original_query_leg,
        )

    def _run_local_retrieval_stages(
        self,
        mySession: Session,
        *,
        local_inputs: _LocalStageInputs,
    ) -> _LocalStageResult:
        """Run local retrieval stages and accumulate guardrail top-k counters."""
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = 0

        vector_stage_result = self._run_vector_stage(
            mySession,
            vector_enabled=local_inputs.vector_enabled,
            primary_query=local_inputs.primary_query,
            guardrail_query=local_inputs.guardrail_query,
            use_guardrail=local_inputs.use_guardrail,
            alternate_queries=local_inputs.alternate_queries,
        )
        alternate_queries_used_obj = getattr(
            vector_stage_result,
            "alternate_queries",
            [],
        )
        alternate_queries_used = cast(
            list[str],
            alternate_queries_used_obj if alternate_queries_used_obj else [],
        )
        self._emit_vector_alternate_queries_status(
            alternate_queries=alternate_queries_used,
        )
        top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._host._accumulate_topk_counts(
                top_k_orig_query_en,
                top_k_post_rewrite_query_en,
                vector_stage_result.top_k_orig_query_en,
                vector_stage_result.top_k_post_rewrite_query_en,
            )
        )

        native_vector_docs = self._run_original_query_vector_leg(
            mySession,
            vector_enabled=local_inputs.vector_enabled,
            original_query_leg=local_inputs.original_query_leg,
        )
        merged_vector_docs, native_added, native_overlap = (
            self._merge_docs_with_native_leg(
                base_docs=vector_stage_result.vector_docs,
                native_docs=native_vector_docs,
                source_marker="VectorRaw",
            )
        )
        if local_inputs.original_query_leg.enabled:
            self._write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language vector merge "
                    f"added={native_added} overlap={native_overlap} "
                    f"total_vector={len(merged_vector_docs)}"
                ),
                color=BRIGHT_MAGENTA,
            )

        indexed_specs = self._host._indexed_retriever_specs(
            mySession,
            bm25_enabled=local_inputs.bm25_enabled,
            graph_enabled=local_inputs.graph_enabled,
            regex_enabled=local_inputs.regex_enabled,
        )
        indexed_stage_result = self._run_indexed_retrieval_with_shape_mode(
            mySession,
            indexed_specs=indexed_specs,
            primary_query=local_inputs.primary_query,
            guardrail_query=local_inputs.guardrail_query,
            use_guardrail=local_inputs.use_guardrail,
            file_filter=local_inputs.file_filter,
            shape_queries=True,
        )
        top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._host._accumulate_topk_counts(
                top_k_orig_query_en,
                top_k_post_rewrite_query_en,
                indexed_stage_result.top_k_orig_query_en,
                indexed_stage_result.top_k_post_rewrite_query_en,
            )
        )

        merged_bm25_docs = list(indexed_stage_result.bm25_docs)
        merged_graph_docs = list(indexed_stage_result.graph_docs)
        merged_regex_docs = list(indexed_stage_result.regex_docs)

        native_bm25_hits = 0
        native_bm25_added = 0
        native_bm25_overlap = 0
        native_graph_hits = 0
        native_graph_added = 0
        native_graph_overlap = 0
        native_regex_hits = 0
        native_regex_added = 0
        native_regex_overlap = 0

        indexed_leg_enabled = bool(
            any(spec[1] and spec[2] != 0.0 for spec in indexed_specs)
        )
        if local_inputs.original_query_leg.enabled and indexed_leg_enabled:
            native_file_filter = self._build_graph_regex_stage_filter(
                local_inputs.file_filter,
                local_inputs.original_query_leg.language_bucket,
            )
            native_indexed_stage_result = self._run_indexed_retrieval_with_shape_mode(
                mySession,
                indexed_specs=indexed_specs,
                primary_query=local_inputs.original_query_leg.query,
                guardrail_query="",
                use_guardrail=False,
                file_filter=native_file_filter,
                shape_queries=False,
            )

            native_bm25_hits = len(native_indexed_stage_result.bm25_docs)
            native_graph_hits = len(native_indexed_stage_result.graph_docs)
            native_regex_hits = len(native_indexed_stage_result.regex_docs)

            merged_bm25_docs, native_bm25_added, native_bm25_overlap = (
                self._merge_docs_with_native_leg(
                    base_docs=merged_bm25_docs,
                    native_docs=native_indexed_stage_result.bm25_docs,
                    source_marker="BM25Raw",
                )
            )
            merged_graph_docs, native_graph_added, native_graph_overlap = (
                self._merge_docs_with_native_leg(
                    base_docs=merged_graph_docs,
                    native_docs=native_indexed_stage_result.graph_docs,
                    source_marker="GraphRaw",
                )
            )
            merged_regex_docs, native_regex_added, native_regex_overlap = (
                self._merge_docs_with_native_leg(
                    base_docs=merged_regex_docs,
                    native_docs=native_indexed_stage_result.regex_docs,
                    source_marker="RegexRaw",
                )
            )

            self._write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language indexed merge "
                    f"bm25(h={native_bm25_hits},a={native_bm25_added},o={native_bm25_overlap}) "
                    f"graph(h={native_graph_hits},a={native_graph_added},o={native_graph_overlap}) "
                    f"regex(h={native_regex_hits},a={native_regex_added},o={native_regex_overlap})"
                ),
                color=BRIGHT_MAGENTA,
            )
        elif local_inputs.original_query_leg.enabled:
            self._write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language indexed leg skipped "
                    "(no indexed retriever enabled by mode or weight)"
                ),
                color=BRIGHT_MAGENTA,
            )

        self._store_original_query_leg_stats(
            mySession,
            original_query_leg=local_inputs.original_query_leg,
            vector_hits=len(native_vector_docs),
            vector_added=native_added,
            vector_overlap=native_overlap,
            bm25_hits=native_bm25_hits,
            bm25_added=native_bm25_added,
            bm25_overlap=native_bm25_overlap,
            graph_hits=native_graph_hits,
            graph_added=native_graph_added,
            graph_overlap=native_graph_overlap,
            regex_hits=native_regex_hits,
            regex_added=native_regex_added,
            regex_overlap=native_regex_overlap,
        )

        return _LocalStageResult(
            local_docs=_LocalDocs(
                vector_docs=merged_vector_docs,
                bm25_docs=merged_bm25_docs,
                graph_docs=merged_graph_docs,
                regex_docs=merged_regex_docs,
            ),
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _store_guardrail_topk_if_enabled(
        self,
        mySession: Session,
        *,
        use_guardrail: bool,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None:
        """Persist guardrail top-k counters only when guardrail execution is active."""
        if not use_guardrail:
            return
        self._host._store_guardrail_retrieval_topk(
            mySession,
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _run_local_docs_stage(
        self,
        mySession: Session,
        *,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: _GuardrailInputs | None,
    ) -> _LocalDocs:
        """Run local stages end-to-end and return assembled local docs."""
        local_inputs = self._resolve_local_stage_inputs(
            mySession,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            alternate_queries=alternate_queries,
            orig_translated_query_en=orig_translated_query_en,
            resolved_guardrail=resolved_guardrail,
        )
        local_stage_result = self._run_local_retrieval_stages(
            mySession,
            local_inputs=local_inputs,
        )

        self._store_guardrail_topk_if_enabled(
            mySession,
            use_guardrail=local_inputs.use_guardrail,
            top_k_orig_query_en=local_stage_result.top_k_orig_query_en,
            top_k_post_rewrite_query_en=local_stage_result.top_k_post_rewrite_query_en,
        )

        return local_stage_result.local_docs

    def _run_post_gate_pipeline(
        self,
        mySession: Session,
        *,
        pipeline_inputs: _PostGatePipelineInputs,
    ) -> Tuple[str, int]:
        """Run local/web retrieval and context assembly after gate checks pass."""
        local_docs = self._empty_local_docs()
        if not self._is_web_only_mode(pipeline_inputs.retrieve_mode):
            self._trace_stage(
                mySession,
                stage_name="retrieve local candidates",
                details=(f"mode={pipeline_inputs.retrieve_mode}"),
            )
            local_docs = self._run_local_docs_stage(
                mySession,
                retrieve_mode=pipeline_inputs.retrieve_mode,
                bm25_query=pipeline_inputs.bm25_query,
                alternate_queries=pipeline_inputs.alternate_queries,
                orig_translated_query_en=pipeline_inputs.orig_translated_query_en,
                resolved_guardrail=pipeline_inputs.resolved_guardrail,
            )
            self._trace(
                mySession,
                "local docs fetched "
                f"vector={len(local_docs.vector_docs)} "
                f"bm25={len(local_docs.bm25_docs)} "
                f"graph={len(local_docs.graph_docs)} regex={len(local_docs.regex_docs)}",
            )

        self._trace_stage(
            mySession,
            stage_name="retrieve web candidates",
        )
        web_docs = self.run_web_retriever(
            mySession,
            retrieve_mode=pipeline_inputs.retrieve_mode,
            user_query_original=pipeline_inputs.user_query_original,
        )
        self._trace(mySession, f"web docs fetched count={len(web_docs)}")

        return self._run_merge_and_context_stage(mySession, local_docs, web_docs)

    def _run_pre_gate_pipeline(self, mySession: Session) -> _PreGatePipelineResult:
        """Prepare query inputs and resolve post-gate pipeline inputs."""
        self._trace_stage(mySession, stage_name="prepare session context")
        self._trace_stage(mySession, stage_name="normalize user query")
        plan = self._resolve_plan(mySession)
        self._emit_query_language_rewrite_status(mySession, plan=plan)
        self._trace(
            mySession,
            "normalized query "
            f"final={plan.final_query!r} alternate_count={len(plan.alternate_queries)}",
        )

        self._trace_stage(mySession, stage_name="apply retrieval gates")
        if self._check_gates(mySession, plan.final_query):
            self._trace(
                mySession, "gated: retrieval aborted by policy or retrieval gate"
            )
            return _PreGatePipelineResult(
                should_abort=True,
                post_gate_inputs=None,
            )

        # Guardrail enables a second local retrieval leg on the pre-rewrite
        # English seed retrieval query to reduce misses when rewrite/
        # normalization drifts.
        guardrail_inputs = self._resolve_guardrail_inputs(
            bm25_query=plan.bm25_query,
            orig_translated_query_en=plan.orig_translated_query_en,
        )
        original_query_leg = self._resolve_original_query_leg(
            mySession,
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
        )
        self._emit_original_query_leg_status(
            original_query_leg=original_query_leg,
        )
        self._trace(
            mySession,
            "stage plan "
            f"mode={plan.retrieve_mode} "
            f"guardrail={'on' if guardrail_inputs.use_guardrail else 'off'} "
            f"original_leg={'on' if original_query_leg.enabled else 'off'}",
        )

        pipeline_inputs = _PostGatePipelineInputs(
            retrieve_mode=plan.retrieve_mode,
            bm25_query=plan.bm25_query,
            alternate_queries=list(plan.alternate_queries),
            orig_translated_query_en=plan.orig_translated_query_en,
            user_query_original=plan.user_query_original,
            resolved_guardrail=guardrail_inputs,
        )
        return _PreGatePipelineResult(
            should_abort=False,
            post_gate_inputs=pipeline_inputs,
        )

    def run_local_retrievers(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
        resolved_guardrail: _GuardrailInputs | None = None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Run local retrievers and guardrail fusion using host retriever helpers.

        This method centralizes local retrieval orchestration while keeping the
        concrete retriever implementations on the host object.
        """
        local_docs = self._run_local_docs_stage(
            mySession,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            alternate_queries=alternate_queries,
            orig_translated_query_en=orig_translated_query_en,
            resolved_guardrail=resolved_guardrail,
        )
        return (
            local_docs.vector_docs,
            local_docs.bm25_docs,
            local_docs.graph_docs,
            local_docs.regex_docs,
        )

    def run(self, mySession: Session) -> Tuple[str, int]:
        """Run one retrieval turn and return (context, chunk_count)."""

        self._trace(mySession, "start retrieval turn")

        if not self._host._set_vector_store(mySession):
            self._trace(mySession, "vector store setup failed; aborting retrieval")
            return "", 0

        self._host.perf_logger.log("RAGChatImpl._retrieve", "chat", "start retrieve")

        pre_gate_result = self._run_pre_gate_pipeline(mySession)
        if pre_gate_result.should_abort:
            return "", 0

        pipeline_inputs = pre_gate_result.post_gate_inputs
        if pipeline_inputs is None:
            self._trace(mySession, "pre-gate pipeline produced no post-gate inputs")
            return "", 0

        return self._run_post_gate_pipeline(
            mySession,
            pipeline_inputs=pipeline_inputs,
        )
