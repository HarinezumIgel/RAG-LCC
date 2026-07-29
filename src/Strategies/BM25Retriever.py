"""BM25 retrieval index for hybrid search.

Builds an Okapi-BM25 index over chunk texts stored in a ChromaDB collection.
The index can be:
  - Persisted to disk during RAGLoad (``build_and_persist``)
  - Loaded from disk during RAGChat  (``load_or_rebuild``)
  - Updated incrementally per file   (``remove_by_filepath`` / ``add_chunks``)

Scoring at query time returns LangChain Documents with ``bm25_score`` in
their metadata, compatible with the existing rerank / chunk-selection
pipeline.
"""

import gzip
import hashlib
import math
import os
import pickle
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, cast

from langchain_core.documents.base import Document as LangchainDocument

from Commons.SingletonMixin import SingletonMixin
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.PerfLogger import PerfLogger


# ---------------------------------------------------------------------------
# Persisted data structure
# ---------------------------------------------------------------------------
class _BM25IndexData:
    """Serialisable container for the BM25 index state."""

    __slots__ = (
        "chunk_ids",
        "chunk_tokens",
        "chunk_metas",
        "chunk_texts",
        "df",
        "N",  # pyright: ignore[reportConstantRedefinition]  — standard BM25 notation
        "avg_dl",
        "idf",
        "collection_name",
        "doc_count_at_build",
    )

    def __init__(self) -> None:
        self.chunk_ids: List[str] = []
        self.chunk_tokens: List[List[str]] = []
        self.chunk_metas: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []
        self.df: Counter[str] = Counter()
        self.N: int = 0  # pyright: ignore[reportConstantRedefinition]
        self.avg_dl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.collection_name: str = ""
        self.doc_count_at_build: int = 0


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------
class BM25Retriever(SingletonMixin):
    """Singleton that manages a per-collection BM25 index.

    The index is **self-contained in memory** — ``_BM25IndexData`` stores the
    full chunk corpus as parallel arrays (``chunk_texts``, ``chunk_metas``,
    ``chunk_tokens``) alongside the scoring data (``df``, ``idf``, ``avg_dl``).
    It is effectively a duplicate of the ChromaDB text corpus held in RAM, not
    a pointer-into-Chroma design.

    Lifecycle
    ---------
    1. **Build / load** — ``load_or_rebuild`` populates ``_BM25IndexData`` from
       a persisted ``.pkl.gz`` file (fast path) or by reading all chunks from the
       ChromaDB collection (rebuild path).  ``add_chunks`` / ``remove_by_filepath``
       update the in-memory index incrementally during ingestion.

    2. **Query** — ``query()`` scores every chunk at position ``i`` via
       ``_score_doc()`` (uses pre-tokenised ``chunk_tokens[i]`` + ``idf`` table),
       then constructs a fully populated ``LangchainDocument`` from
       ``chunk_texts[i]`` and ``chunk_metas[i]`` for the top-k hits.  Documents
       are complete objects before they leave ``query()`` — no back-reference to
       the index is needed afterwards.

    3. **Merge** — ``reciprocal_rank_fusion`` receives fully populated
       ``LangchainDocument`` lists from all three retrievers (Vector, BM25,
       Graph).  It uses ``metadata["chunk_id"]`` (with MD5 fallback) solely for
       deduplication / score fusion identity; it never reads back into the index.
    """

    INDEX_FILENAME = "bm25_index.pkl.gz"

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self._shared: SharedHelpers = SharedHelpers()
        self._data: _BM25IndexData = _BM25IndexData()

        # BM25 hyper-parameters — read from _BM25_INDEX config slot
        self._k1: float = self.cfg.get_float("_BM25_INDEX.k1")
        self._b: float = self.cfg.get_float("_BM25_INDEX.b")
        self._rrf_k: int = self.cfg.get_int("_BM25_INDEX.rrf_k")
        self.perf_logger: PerfLogger = PerfLogger()

    def get_bm25_dir(self, collection_name: str) -> str:
        """Return the BM25 index directory for *collection_name*.

        Path: ``<_BM25_INDEX.BM25_INDEX_DIR>/<collection_name>``
        """
        return os.path.join(
            self.cfg.get_str("_BM25_INDEX.BM25_INDEX_DIR"), collection_name
        )

    @property
    def rrf_k(self) -> int:
        """Reciprocal Rank Fusion constant (public read-only access)."""
        return self._rrf_k

    # ------------------------------------------------------------------
    # Public API — index lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if an index is currently in memory (any collection)."""
        return self._data.N > 0

    def is_loaded_for(self, collection_name: str) -> bool:
        """True if the in-memory index belongs to *collection_name*."""
        return self._data.collection_name == collection_name and self._data.N > 0

    def load_or_rebuild(
        self,
        bm25_directory: str,
        collection_name: str,
        collection: Any,
        *,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load a persisted index or rebuild from the ChromaDB collection."""
        # Short-circuit: in-memory index is already current (avoids disk I/O
        # on every query when the collection has not changed).
        if (
            self._data.collection_name == collection_name
            and self._data.N > 0
            and self._data.doc_count_at_build == collection.count()
        ):
            return

        self.perf_logger.log(
            "BM25Retriever.load_or_rebuild",
            "retriever",
            f"start load collection={collection_name}",
        )

        idx_path = self._index_path(bm25_directory)

        if os.path.isfile(idx_path):
            self._load(idx_path)
            # Staleness check
            if (
                self._data.collection_name == collection_name
                and self._data.doc_count_at_build == collection.count()
            ):
                self.pretty.write(
                    "O",
                    "BM25",
                    f"Loaded persisted BM25 index ({self._data.N} chunks, "
                    f"{len(self._data.idf)} terms)",
                )
                self.perf_logger.log(
                    "BM25Retriever.load_or_rebuild",
                    "retriever",
                    f"stop  load (persisted) collection={collection_name} n={self._data.N}",
                )
                return
            self.pretty.write(
                "I",
                "BM25",
                "Persisted BM25 index is stale — rebuilding from collection",
            )

        # Rebuild from ChromaDB and immediately persist so the next call does
        # not see a stale file and repeat the rebuild.
        self._rebuild_from_collection(collection_name, collection, file_filter)
        self._persist(idx_path)
        self.perf_logger.log(
            "BM25Retriever.load_or_rebuild",
            "retriever",
            f"stop  load (rebuilt) collection={collection_name} n={self._data.N}",
        )

    def build_and_persist(
        self,
        bm25_directory: str,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Full rebuild from collection + write to disk.  Called from RAGLoad."""
        self._rebuild_from_collection(collection_name, collection)
        self._persist(self._index_path(bm25_directory))

    def remove_by_filepath(self, file_path: str) -> None:
        """Remove all chunks belonging to *file_path* and update corpus stats."""
        keep_ids: List[int] = []
        remove_ids: List[int] = []

        for i, meta in enumerate(self._data.chunk_metas):
            if meta.get("FilePath") == file_path:
                remove_ids.append(i)
            else:
                keep_ids.append(i)

        if not remove_ids:
            return

        # Decrement DF for removed chunks
        for i in remove_ids:
            for tok in set(self._data.chunk_tokens[i]):
                self._data.df[tok] -= 1
                if self._data.df[tok] <= 0:
                    del self._data.df[tok]

        # Compact arrays
        self._data.chunk_ids = [self._data.chunk_ids[i] for i in keep_ids]
        self._data.chunk_tokens = [self._data.chunk_tokens[i] for i in keep_ids]
        self._data.chunk_metas = [self._data.chunk_metas[i] for i in keep_ids]
        self._data.chunk_texts = [self._data.chunk_texts[i] for i in keep_ids]
        self._data.N = len(self._data.chunk_ids)
        self._recompute_avg_dl()
        self._recompute_idf()

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Add new chunks and update corpus stats incrementally."""
        for chunk_id, text, meta in zip(ids, texts, metas):
            tokens = self._shared.tokenize(text)
            self._data.chunk_ids.append(chunk_id)
            self._data.chunk_tokens.append(tokens)
            self._data.chunk_metas.append(dict(meta))
            self._data.chunk_texts.append(text)

            for tok in set(tokens):
                self._data.df[tok] += 1

        self._data.N = len(self._data.chunk_ids)
        self._recompute_avg_dl()
        self._recompute_idf()

    def persist(self, bm25_directory: str) -> None:
        """Write current index state to disk."""
        self._persist(self._index_path(bm25_directory))

    def ingest_file(
        self,
        file_path: str,
        collection_name: str,
        collection: Any,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Incrementally update the BM25 index for a single file.

        Loads the index from disk if not already in memory, removes any
        existing chunks for *file_path*, adds the new *ids/texts/metas*,
        then persists the updated index.
        """
        bm25_dir = self.get_bm25_dir(collection_name)
        if not self.is_loaded_for(collection_name):
            self.load_or_rebuild(bm25_dir, collection_name, collection)
        self.remove_by_filepath(file_path)
        if ids:
            self.add_chunks(ids, texts, metas)
        self.persist(bm25_dir)

    # ------------------------------------------------------------------
    # Public API — query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 100,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """Score *query_text* against the index and return top-*k* Documents.

        Each returned Document has ``bm25_score`` and ``chroma_score``
        (set to 0.0) in its metadata so it is compatible with the
        downstream rerank / chunk-selection pipeline.
        """
        if self._data.N == 0:
            return []

        query_tokens: List[str] = self._shared.tokenize(query_text)
        if not query_tokens:
            return []

        self.perf_logger.log(
            "BM25Retriever.query",
            "retriever",
            f"start bm25 query q={query_text[:60]!r}",
        )
        _t0 = time.perf_counter()
        scored: List[Tuple[int, float]] = []
        for idx in range(self._data.N):
            # Apply file filter if provided
            if file_filter:
                meta = self._data.chunk_metas[idx]
                if not self._matches_filter(meta, file_filter):
                    continue

            score = self._score_doc(query_tokens, idx)
            if score > 0.0:
                scored.append((idx, score))

        # Sort descending by score, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        docs: List[LangchainDocument] = []
        for idx, bm25_score in top:
            meta = dict(self._data.chunk_metas[idx])
            meta["bm25_score"] = bm25_score
            meta["chroma_score"] = 0.0  # placeholder for pipeline compat
            meta["chroma_sim"] = 1.0
            doc = LangchainDocument(
                page_content=self._data.chunk_texts[idx],
                metadata=meta,
                id=self._data.chunk_ids[idx],
            )
            docs.append(doc)
        self.perf_logger.log(
            "BM25Retriever.query",
            "retriever",
            f"stop  bm25 query n={len(docs)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        return docs

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion (used by RAGChatImpl for HYBRID mode)
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        *ranked_lists: List[LangchainDocument],
        k: int = 60,
        labels: List[str] | None = None,
        weights: List[float] | None = None,
    ) -> List[LangchainDocument]:
        """Merge multiple ranked Document lists using Reciprocal Rank Fusion.

        Returns a single list sorted by descending RRF score.  Each
        Document's metadata receives an ``rrf_score`` key and, when
        *labels* are provided, a ``retriever_sources`` key listing the
        comma-separated names of every retriever that contributed that
        document (e.g. ``"Vector,BM25"``).

        How it works
        ------------
        Each document at position *rank* in a list receives a score of
        ``1 / (k + rank)``.  If the same document appears in multiple
        lists its scores are **summed**, so documents found by *both*
        retrievers naturally float to the top.

        Example (k=60, two lists of 5 documents each):

            Vector list                BM25 list
            ───────────                ─────────
            1  Hedgehogs (A)           1  Fish (X)
            2  Cats      (B)           2  Hedgehogs (A)  ← same chunk
            3  Fish      (X) ← same    3  Dogs (Y)
            4  Lions     (C)           4  Cats (D)       ← different chunk
            5  Apes      (E)           5  Lions (C)      ← same chunk

        Scoring per list:  1/(60+rank)
            A: vector 1/61=0.01639  +  BM25 1/62=0.01613  =  0.03252
            X: vector 1/63=0.01587  +  BM25 1/61=0.01639  =  0.03226
            C: vector 1/64=0.01563  +  BM25 1/65=0.01538  =  0.03101
            B: vector 1/62=0.01613  only                   =  0.01613
            Y:                         BM25 1/63=0.01587   =  0.01587

        Result (sorted):  A → X → C → B → Y → ...
        Documents in both lists rank highest; single-list documents rank lower.

        What *k* controls
        -----------------
        - **Large k** (e.g. 60): scores compress — rank differences are
          dampened, favouring *agreement between lists* over individual
          rank position.
        - **Small k** (e.g. 1): scores spread — a #1 hit scores 3× more
          than a #5 hit, favouring *top-ranked items* from individual
          lists regardless of agreement.

        k=60 is the standard value from the original RRF paper and
        balances both signals well for most retrieval tasks.
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, LangchainDocument] = {}
        sources_map: Dict[str, list[str]] = {}

        for list_idx, ranked in enumerate(ranked_lists):
            label = (
                labels[list_idx] if labels and list_idx < len(labels) else str(list_idx)
            )
            weight = weights[list_idx] if weights and list_idx < len(weights) else 1.0
            for rank, doc in enumerate(ranked, start=1):
                # Ensure chunks with the same content are actually merged across
                # vector + BM25 streams. Layered fallback:
                #   1. metadata["chunk_id"] — explicit, set during ingestion
                #   2. doc.id              — LangChain Document id (often None for Chroma)
                #   3. md5(page_content)   — last-resort content hash
                # Relying purely on doc.id or id(doc) (memory address) would cause
                # Vector and BM25 objects to look distinct and fail to fuse.
                meta: Dict[str, Any] = cast(Dict[str, Any], doc.metadata)  # type: ignore[reportUnknownMemberType]
                doc_id: Any = (
                    meta.get("chunk_id")
                    or getattr(doc, "id", None)
                    or hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                )
                key: str = str(doc_id)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (k + rank)
                if key not in doc_map:
                    doc_map[key] = doc
                if label not in sources_map.get(key, []):
                    sources_map.setdefault(key, []).append(label)

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        result: List[LangchainDocument] = []
        for key in sorted_keys:
            doc = doc_map[key]
            meta: Dict[str, Any] = doc.metadata  # type: ignore[assignment]
            meta["rrf_score"] = rrf_scores[key]
            # Use rrf_score as the primary score for downstream pipeline
            meta["chroma_score"] = rrf_scores[key]
            if labels is not None:
                meta["retriever_sources"] = ",".join(sources_map.get(key, []))
            result.append(doc)
        return result

    # ------------------------------------------------------------------
    # Internal — BM25 scoring
    # ------------------------------------------------------------------

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document against query tokens."""
        doc_tokens = self._data.chunk_tokens[doc_idx]
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            freq = tf[term]
            term_idf = self._data.idf.get(term, 0.0)
            denom = freq + self._k1 * (
                1.0 - self._b + self._b * (doc_len / (self._data.avg_dl + 1e-12))
            )
            score += term_idf * (freq * (self._k1 + 1.0)) / (denom + 1e-12)
        return score

    # ------------------------------------------------------------------
    # Internal — corpus stats
    # ------------------------------------------------------------------

    def _recompute_idf(self) -> None:
        """Recompute IDF dict from maintained df counter and N."""
        N = self._data.N
        idf: Dict[str, float] = {}
        for term, freq in self._data.df.items():
            idf[term] = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))
        self._data.idf = idf

    def _recompute_avg_dl(self) -> None:
        """Recompute average document length from token lists."""
        if self._data.N == 0:
            self._data.avg_dl = 0.0
            return
        total = sum(len(toks) for toks in self._data.chunk_tokens)
        self._data.avg_dl = total / self._data.N

    # ------------------------------------------------------------------
    # Internal — rebuild from collection
    # ------------------------------------------------------------------

    def _rebuild_from_collection(
        self,
        collection_name: str,
        collection: Any,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fetch all chunks from ChromaDB and build the BM25 index."""
        self.perf_logger.log(
            "BM25Retriever._rebuild_from_collection",
            "retriever",
            f"start rebuild collection={collection_name}",
        )
        _t0 = time.perf_counter()
        self.pretty.write(
            "I", "BM25", f"Building BM25 index from collection '{collection_name}'..."
        )

        kwargs: Dict[str, Any] = {"include": ["documents", "metadatas"]}
        if file_filter:
            kwargs["where"] = file_filter

        result = collection.get(**kwargs)
        ids: List[str] = result.get("ids", []) or []
        documents: List[str] = result.get("documents", []) or []
        metadatas: List[Dict[str, Any]] = result.get("metadatas", []) or []

        # Reset state
        data = _BM25IndexData()
        data.collection_name = collection_name
        data.doc_count_at_build = collection.count()

        df: Counter[str] = Counter()
        total_len: int = 0

        for chunk_id, text, meta in zip(ids, documents, metadatas):
            tokens = self._shared.tokenize(text or "")
            data.chunk_ids.append(chunk_id)
            data.chunk_tokens.append(tokens)
            data.chunk_metas.append(dict(meta) if meta else {})
            data.chunk_texts.append(text or "")

            for tok in set(tokens):
                df[tok] += 1
            total_len += len(tokens)

        data.df = df
        data.N = len(data.chunk_ids)
        data.avg_dl = (total_len / data.N) if data.N > 0 else 0.0

        # Compute IDF
        idf: Dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log(1.0 + (data.N - freq + 0.5) / (freq + 0.5))
        data.idf = idf

        self._data = data
        self.pretty.write(
            "O",
            "BM25",
            f"Built BM25 index: {data.N} chunks, {len(data.idf)} unique terms, "
            f"avg_dl={data.avg_dl:.1f}",
        )
        self.perf_logger.log(
            "BM25Retriever._rebuild_from_collection",
            "retriever",
            f"stop  rebuild collection={collection_name} n={data.N} elapsed={time.perf_counter() - _t0:.3f}s",
        )

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _index_path(self, bm25_directory: str) -> str:
        return os.path.join(bm25_directory, self.INDEX_FILENAME)

    def _persist(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.pretty.write(
            "O",
            "BM25",
            f"Persisted BM25 index to {path} ({self._data.N} chunks)",
        )

    def _load(self, path: str) -> None:
        with gzip.open(path, "rb") as f:
            self._data = pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    # Internal — filter matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filter(meta: Dict[str, Any], filt: Dict[str, Any]) -> bool:
        """Check if a metadata dict matches a ChromaDB-style where filter."""
        for key, condition in filt.items():
            if isinstance(condition, dict):
                for op, val in cast(Dict[str, Any], condition).items():
                    if op == "$eq" and meta.get(key) != val:
                        return False
            else:
                if meta.get(key) != condition:
                    return False
        return True
