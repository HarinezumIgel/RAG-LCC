"""Pre-filtering and re-scoring of web search results before they enter the
rerank pool.

Two independent filters can be applied in sequence after WebRetriever.query():

BM25 pre-filter
  Scores each web snippet against the query using an in-memory mini BM25 corpus
  built from the web snippets themselves.  Does not require the collection BM25
  index — purely query-time, no persistence.  Cheap (pure tokenisation + math).

Cosine pre-filter
  Embeds the query and each snippet using the loaded HuggingFace embedder and
  computes cosine similarity.  Drops snippets below the configured threshold.
  Runs after the BM25 filter when both are enabled (BM25 narrows the pool first,
  saving embedding calls).

Both filters update ``chroma_score`` on surviving documents so the reranker
pool sees a meaningful relevance score instead of the raw rank-based value
produced by WebRetriever.query().  Documents that fail a filter are dropped
and logged.

Configuration (``_WEB_SEARCH`` in Config_WebSearch.py):
  bm25_pre_filter   — minimum BM25 score to keep (0.0 = disabled)
  cosine_pre_filter — minimum cosine similarity to keep (0.0 = disabled)
"""

import math
from collections import Counter
from typing import Any, List

from langchain_core.documents.base import Document as LangchainDocument

from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter

# ---------------------------------------------------------------------------
# Pure helper functions (module-level, no state)
# ---------------------------------------------------------------------------


def _idf(df: int, N: int) -> float:
    """Robertson-Spärck Jones IDF with +1 smoothing (always non-negative)."""
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    df_counts: "Counter[str]",
    N: int,
    avgdl: float,
    k1: float,
    b: float,
) -> float:
    """Okapi BM25 score for *doc_tokens* against *query_tokens*.

    Parameters
    ----------
    query_tokens:
        Tokenised query.
    doc_tokens:
        Tokenised document text.
    df_counts:
        Document-frequency counts over the mini-corpus.
    N:
        Total number of documents in the mini-corpus.
    avgdl:
        Average document length (tokens) in the mini-corpus.
    k1, b:
        Standard BM25 hyper-parameters (k1≈1.5, b≈0.75).
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    tf_counts: Counter[str] = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf = tf_counts.get(token, 0)
        if tf == 0:
            continue
        idf = _idf(df_counts.get(token, 0), N)
        norm_tf = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0)))
        score += idf * norm_tf
    return score


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# WebPreFilter
# ---------------------------------------------------------------------------


class WebPreFilter:
    """Scores and optionally drops web results before they enter the rerank pool.

    Not a singleton — created once by RAGChatImpl and reused across queries.

    Parameters
    ----------
    cfg:
        Config singleton.  Reads ``_WEB_SEARCH.bm25_pre_filter``,
        ``_WEB_SEARCH.cosine_pre_filter``, ``_BM25_INDEX.k1``,
        ``_BM25_INDEX.b``.
    embedder:
        HuggingFace embeddings model (``RAGChatImpl.embedder``).  Only
        required when ``cosine_pre_filter > 0``.
    pretty:
        PrettyWriter for progress messages.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        embedder: Any = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        self._cfg: Config = cfg or Config()
        self._embedder: Any = embedder
        self._pretty: PrettyWriter = pretty or PrettyWriter()
        self._shared: SharedHelpers = SharedHelpers()

        # Reuse the same BM25 hyper-parameters as the collection index.
        self._k1: float = self._cfg.get_float("_BM25_INDEX.k1") or 1.5
        self._b: float = self._cfg.get_float("_BM25_INDEX.b") or 0.75

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bm25_prefilter(
        self, docs: List[LangchainDocument], query: str
    ) -> List[LangchainDocument]:
        """Score *docs* against *query* using a mini BM25 corpus built from
        the docs themselves.  Drop docs whose score is below
        ``_WEB_SEARCH.bm25_pre_filter``.  Updates ``chroma_score`` on
        survivors with the BM25 score.

        When the threshold is 0.0 (default), returns *docs* unchanged.
        """
        threshold: float = self._cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        if threshold <= 0.0 or not docs:
            return docs

        query_tokens = self._shared.tokenize(query)
        if not query_tokens:
            return docs

        all_tokens = [self._shared.tokenize(d.page_content) for d in docs]
        N = len(docs)
        avgdl = sum(len(t) for t in all_tokens) / max(N, 1)

        df_counts: Counter[str] = Counter()
        for tokens in all_tokens:
            for tok in set(tokens):
                df_counts[tok] += 1

        survivors: List[LangchainDocument] = []
        for doc, tokens in zip(docs, all_tokens):
            score = _bm25_score(
                query_tokens, tokens, df_counts, N, avgdl, self._k1, self._b
            )
            if score >= threshold:
                doc.metadata["chroma_score"] = score  # type: ignore[index]
                survivors.append(doc)

        dropped = len(docs) - len(survivors)
        if dropped:
            self._pretty.write(
                "I",
                "WebPreFilter",
                f"BM25 pre-filter dropped {dropped}/{len(docs)} web result(s) "
                f"(threshold={threshold:.3f})",
            )
        return survivors

    def cosine_prefilter(
        self,
        docs: List[LangchainDocument],
        query: str,
        *,
        query_vec: "List[float] | None" = None,
    ) -> List[LangchainDocument]:
        """Embed *query* and each doc's scoring text (snippet when available,
        otherwise page_content); drop docs whose cosine similarity to the
        query is below ``_WEB_SEARCH.cosine_pre_filter``.  Updates
        ``chroma_score`` on survivors with the cosine score.

        Parameters
        ----------
        docs:
            Web result documents to filter.
        query:
            Query string.  Only used when *query_vec* is ``None``.
        query_vec:
            Pre-computed query embedding.  Pass this when the caller has
            already embedded the query (e.g. the vector retriever already
            ran) to avoid a redundant ``embed_query`` call.

        When the threshold is 0.0 (default), or when no embedder is
        available, returns *docs* unchanged.
        """
        threshold: float = self._cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        if threshold <= 0.0 or not docs:
            return docs

        if self._embedder is None:
            self._pretty.write(
                "W",
                "WebPreFilter",
                "cosine_pre_filter is set but embedder is unavailable — skipping.",
            )
            return docs

        # Use the stored snippet for scoring (same text the cross-encoder uses),
        # falling back to page_content when snippet is absent.
        texts = [
            str(d.metadata.get("snippet") or d.page_content)  # type: ignore[union-attr]
            for d in docs
        ]

        # Reuse a pre-computed query vector when provided (avoids a redundant
        # embed_query call when vector retrieval already ran this turn).
        effective_query_vec: List[float] = (
            query_vec if query_vec is not None else self._embedder.embed_query(query)
        )
        doc_vecs: List[List[float]] = self._embedder.embed_documents(texts)

        survivors: List[LangchainDocument] = []
        for doc, vec in zip(docs, doc_vecs):
            sim = _cosine(effective_query_vec, vec)
            if sim >= threshold:
                doc.metadata["chroma_score"] = sim  # type: ignore[index]
                survivors.append(doc)

        dropped = len(docs) - len(survivors)
        if dropped:
            self._pretty.write(
                "I",
                "WebPreFilter",
                f"Cosine pre-filter dropped {dropped}/{len(docs)} web result(s) "
                f"(threshold={threshold:.3f})",
            )
        return survivors
