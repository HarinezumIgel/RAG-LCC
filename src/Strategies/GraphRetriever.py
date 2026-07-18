"""Graph-based retrieval index for semantic graph search.

Builds an entity co-occurrence graph over chunk texts stored in a ChromaDB
collection using spaCy NER.  The graph can be:
  - Persisted to disk during RAGLoad   (``build_and_persist``)
  - Loaded from disk during RAGChat    (``load_or_rebuild``)
  - Updated incrementally per file     (``remove_by_filepath`` / ``add_chunks``)

Scoring at query time: NER the query → BFS hop traversal → score candidate
chunks by sum of co-occurrence edge weights → return LangChain Documents with
``graph_score`` and ``bm25_score=0.0`` in metadata (the latter is the existing
pipeline signal that triggers rerank-only blending in ``_rerank``).
"""

import gzip
import os
import pickle
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, cast

from langchain_core.documents.base import Document as LangchainDocument

from Commons.Exceptions import ModelLoadError
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.PerfLogger import PerfLogger


# ---------------------------------------------------------------------------
# Persisted data structure
# ---------------------------------------------------------------------------
class _GraphIndexData:
    """Serialisable container for the graph index state.

    Ground-truth fields (set during ``add_chunks``, never derived):
        chunk_entities  — chunk_id  → list of entity strings
        chunk_metas     — chunk_id  → metadata dict
        chunk_texts     — chunk_id  → raw chunk text (for page_content)

    Derived fields (rebuilt from chunk_entities via ``_rebuild_derived``):
        adjacency        — entity → {neighbour_entity: co-occurrence weight}
        entity_to_chunks — entity → [chunk_ids]

    Rebuilding derived fields from ground truth makes ``remove_by_filepath``
    safe: delete the affected chunk_ids, call ``_rebuild_derived``, done.
    This mirrors the BM25Retriever's ``df``-rebuild pattern exactly.
    """

    __slots__ = (
        "chunk_entities",
        "chunk_metas",
        "chunk_texts",
        "adjacency",
        "entity_to_chunks",
        "collection_name",
        "doc_count_at_build",
    )

    def __init__(self) -> None:
        self.chunk_entities: Dict[str, List[str]] = {}
        self.chunk_metas: Dict[str, Dict[str, Any]] = {}
        self.chunk_texts: Dict[str, str] = {}
        self.adjacency: Dict[str, Dict[str, int]] = {}
        self.entity_to_chunks: Dict[str, List[str]] = {}
        self.collection_name: str = ""
        self.doc_count_at_build: int = 0


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------
class GraphRetriever(SingletonMixin):
    """Singleton that manages a per-collection entity co-occurrence graph index."""

    INDEX_FILENAME = "graph_index.pkl.gz"

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        nlp: Any = None,  # injectable spaCy model for tests
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self._data: _GraphIndexData = _GraphIndexData()
        self.perf_logger: PerfLogger = PerfLogger()

        # Graph hyper-parameters from _GRAPH_INDEX config slot
        self._entity_types: List[str] = self.cfg.get_list("_GRAPH_INDEX.entity_types")
        self._max_hops: int = self.cfg.get_int("_GRAPH_INDEX.max_hops")
        self._max_candidates: int = self.cfg.get_int("_GRAPH_INDEX.max_candidates")
        self._min_edge_weight: int = self.cfg.get_int("_GRAPH_INDEX.min_edge_weight")
        self._spacy_model: str = self.cfg.get_str("_GRAPH_INDEX.spacy_model")
        self._noun_chunk_min_chars: int = self.cfg.get_int(
            "_GRAPH_INDEX.noun_chunk_min_chars", 3
        )
        self._noun_chunk_drop_leading: str = self.cfg.get_str(
            "_GRAPH_INDEX.noun_chunk_drop_leading", "[({<"
        )

        if nlp is not None:
            # Injected in tests — skip real spaCy load
            self._nlp = nlp
        else:
            try:
                import spacy  # type: ignore[import-untyped]

                self._nlp = spacy.load(self._spacy_model)
            except OSError as exc:
                raise ModelLoadError(
                    f"spaCy model '{self._spacy_model}' not found. "
                    f"Run: python -m spacy download {self._spacy_model}"
                ) from exc

    # ------------------------------------------------------------------
    # Public API — directory helpers
    # ------------------------------------------------------------------

    def get_graph_dir(self, collection_name: str) -> str:
        """Return the graph index directory for *collection_name*.

        Path: ``<_GRAPH_INDEX.GRAPH_INDEX_DIR>/<collection_name>``
        """
        return os.path.join(
            self.cfg.get_str("_GRAPH_INDEX.GRAPH_INDEX_DIR"), collection_name
        )

    # ------------------------------------------------------------------
    # Public API — index state queries
    # ------------------------------------------------------------------

    def is_loaded_for(self, collection_name: str) -> bool:
        """True if the in-memory index belongs to *collection_name*."""
        return self._data.collection_name == collection_name and bool(
            self._data.chunk_entities
        )

    # ------------------------------------------------------------------
    # Public API — index lifecycle
    # ------------------------------------------------------------------

    def load_or_rebuild(
        self,
        graph_directory: str,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Load a persisted index or rebuild from the ChromaDB collection."""
        # Short-circuit: in-memory index is already current (avoids disk I/O
        # on every query when the collection has not changed).
        if (
            self._data.collection_name == collection_name
            and bool(self._data.chunk_entities)
            and self._data.doc_count_at_build == collection.count()
        ):
            return

        idx_path = self._index_path(graph_directory)

        if os.path.isfile(idx_path):
            self._load(idx_path)
            # Staleness check
            if (
                self._data.collection_name == collection_name
                and self._data.doc_count_at_build == collection.count()
            ):
                self.pretty.write(
                    "O",
                    "Graph",
                    f"Loaded persisted graph index ({len(self._data.chunk_entities)} chunks, "
                    f"{len(self._data.adjacency)} entities)",
                )
                return
            self.pretty.write(
                "I",
                "Graph",
                "Persisted graph index is stale — rebuilding from collection",
            )

        # Rebuild from ChromaDB and immediately persist so the next call does
        # not see a stale file and repeat the rebuild.
        self._rebuild_from_collection(collection_name, collection)
        self._persist(idx_path)

    def build_and_persist(
        self,
        graph_directory: str,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Full rebuild from collection + write to disk.  Called from RAGLoad."""
        self._rebuild_from_collection(collection_name, collection)
        self._persist(self._index_path(graph_directory))

    def remove_by_filepath(self, file_path: str) -> None:
        """Remove all chunks belonging to *file_path* and rebuild derived data."""
        chunk_ids_to_remove = [
            cid
            for cid, meta in self._data.chunk_metas.items()
            if meta.get("FilePath") == file_path
        ]
        if not chunk_ids_to_remove:
            return

        for cid in chunk_ids_to_remove:
            self._data.chunk_entities.pop(cid, None)
            self._data.chunk_metas.pop(cid, None)
            self._data.chunk_texts.pop(cid, None)

        self._rebuild_derived()

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Extract entities from chunks, add to ground-truth, rebuild derived data."""
        for chunk_id, text, meta in zip(ids, texts, metas):
            entities = self._extract_entities(text)
            self._data.chunk_entities[chunk_id] = entities
            self._data.chunk_metas[chunk_id] = dict(meta)
            self._data.chunk_texts[chunk_id] = text

        self._rebuild_derived()

    def persist(self, graph_directory: str) -> None:
        """Write current index state to disk."""
        self._persist(self._index_path(graph_directory))

    def ingest_file(
        self,
        file_path: str,
        collection_name: str,
        collection: Any,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Incrementally update the graph index for a single file.

        Loads the index from disk if not already in memory, removes any
        existing chunks for *file_path*, adds the new *ids/texts/metas*,
        then persists the updated index.
        """
        graph_dir = self.get_graph_dir(collection_name)
        if not self.is_loaded_for(collection_name):
            self.load_or_rebuild(graph_dir, collection_name, collection)
        self.remove_by_filepath(file_path)
        if ids:
            self.add_chunks(ids, texts, metas)
        self.persist(graph_dir)

    # ------------------------------------------------------------------
    # Public API — query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 100,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """Score *query_text* against the graph and return top-*k* Documents.

        Algorithm:
          1. NER on query_text → seed entities
          2. BFS up to max_hops, pruning edges below min_edge_weight
          3. Collect candidate chunk_ids from visited entity nodes
          4. Score each chunk: sum of edge weights from connected visited entities
          5. Apply file_filter, sort descending, return top-k LangchainDocuments

        Each returned Document has ``graph_score``, ``bm25_score=0.0``, and
        ``chroma_score=<graph_score>`` in its metadata so it is compatible with
        the downstream rerank / chunk-selection pipeline.
        """
        if not self._data.chunk_entities:
            return []

        self.perf_logger.log(
            "GraphRetriever.query", f"start graph query q={query_text[:60]!r}"
        )
        _t0 = time.perf_counter()
        seed_entities = [
            e
            for e in self._extract_entities(query_text)
            if e in self._data.entity_to_chunks
        ]
        if not seed_entities:
            return []

        # BFS traversal
        visited: Dict[str, int] = {}  # entity -> total accumulated weight
        queue: deque[tuple[str, int]] = deque()  # (entity, hops_remaining)
        for ent in seed_entities:
            if ent not in visited:
                visited[ent] = 0
                queue.append((ent, self._max_hops))

        while queue:
            current, hops_left = queue.popleft()
            for neighbour, weight in self._data.adjacency.get(current, {}).items():
                if weight < self._min_edge_weight:
                    continue
                if neighbour not in visited:
                    visited[neighbour] = weight
                    if hops_left > 1:
                        queue.append((neighbour, hops_left - 1))
                else:
                    visited[neighbour] += weight

        # Collect and score candidate chunks
        chunk_scores: Dict[str, float] = defaultdict(float)
        for entity, accumulated_weight in visited.items():
            for cid in self._data.entity_to_chunks.get(entity, []):
                chunk_scores[cid] += accumulated_weight

        if not chunk_scores:
            return []

        # Apply file filter
        if file_filter:
            chunk_scores = {
                cid: score
                for cid, score in chunk_scores.items()
                if self._matches_filter(
                    self._data.chunk_metas.get(cid, {}), file_filter
                )
            }

        # Sort by score descending, take top max_candidates then cap at k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[
            : self._max_candidates
        ][:k]

        docs: List[LangchainDocument] = []
        for cid, score in sorted_chunks:
            meta = dict(self._data.chunk_metas.get(cid, {}))
            meta["graph_score"] = score
            meta["bm25_score"] = 0.0  # signals rerank-only blending downstream
            meta["chroma_score"] = score  # unified score key for the pipeline
            meta["chroma_sim"] = 1.0
            docs.append(
                LangchainDocument(
                    page_content=self._data.chunk_texts.get(cid, ""),
                    metadata=meta,
                    id=cid,
                )
            )
        self.perf_logger.log(
            "GraphRetriever.query",
            f"stop  graph query n={len(docs)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        return docs

    # ------------------------------------------------------------------
    # Internal — entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> List[str]:
        """Run spaCy NER (and optionally noun-chunk extraction) on *text*.

        If ``"NOUN_CHUNK"`` is present in *entity_types*, all noun phrases
        produced by the spaCy dependency parser are included in addition to
        any named entities whose label matches the remaining entries.
        Named-entity labels and ``"NOUN_CHUNK"`` can be combined freely.
        """
        doc = self._nlp(text)
        seen: set[str] = set()
        entities: List[str] = []

        use_noun_chunks: bool = "NOUN_CHUNK" in self._entity_types
        ner_types: List[str] = [t for t in self._entity_types if t != "NOUN_CHUNK"]

        # --- Named entities ---
        for ent in doc.ents:
            if ner_types and ent.label_ not in ner_types:
                continue
            normalised = ent.text.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                entities.append(normalised)

        # --- Noun chunks (syntactic noun phrases from the dependency parser) ---
        if use_noun_chunks:
            for chunk in doc.noun_chunks:
                normalised = chunk.text.strip().lower()
                # Configurable noise filter: too short, starts with a drop-leading
                # character, or contains no alphabetic characters at all.
                if (
                    not normalised
                    or len(normalised) < self._noun_chunk_min_chars
                    or (
                        self._noun_chunk_drop_leading
                        and normalised[0] in self._noun_chunk_drop_leading
                    )
                    or not any(c.isalpha() for c in normalised)
                ):
                    continue
                if normalised not in seen:
                    seen.add(normalised)
                    entities.append(normalised)

        return entities

    # ------------------------------------------------------------------
    # Internal — derived structure rebuild
    # ------------------------------------------------------------------

    def _rebuild_derived(self) -> None:
        """Rebuild adjacency and entity_to_chunks from chunk_entities (ground truth).

        Called after every add_chunks or remove_by_filepath to keep derived
        structures consistent without partial-update logic.
        """
        adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        entity_to_chunks: Dict[str, List[str]] = defaultdict(list)

        for chunk_id, entities in self._data.chunk_entities.items():
            unique = list(dict.fromkeys(entities))  # preserve order, deduplicate
            for ent in unique:
                entity_to_chunks[ent].append(chunk_id)
            # Build co-occurrence edges for every entity pair in this chunk
            for i, ent_a in enumerate(unique):
                for ent_b in unique[i + 1 :]:
                    adjacency[ent_a][ent_b] += 1
                    adjacency[ent_b][ent_a] += 1

        # Convert defaultdicts to plain dicts for pickle safety
        self._data.adjacency = {k: dict(v) for k, v in adjacency.items()}
        self._data.entity_to_chunks = dict(entity_to_chunks)

    # ------------------------------------------------------------------
    # Internal — rebuild from ChromaDB collection
    # ------------------------------------------------------------------

    def _rebuild_from_collection(
        self,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Fetch all chunks from ChromaDB and build the graph index from scratch."""
        self.pretty.write(
            "I", "Graph", f"Building graph index from collection '{collection_name}'..."
        )

        result = collection.get(include=["documents", "metadatas"])
        ids: List[str] = result.get("ids", []) or []
        documents: List[str] = result.get("documents", []) or []
        metadatas: List[Dict[str, Any]] = result.get("metadatas", []) or []

        data = _GraphIndexData()
        data.collection_name = collection_name
        data.doc_count_at_build = collection.count()

        for chunk_id, text, meta in zip(ids, documents, metadatas):
            text = text or ""
            entities = self._extract_entities(text)
            data.chunk_entities[chunk_id] = entities
            data.chunk_metas[chunk_id] = dict(meta) if meta else {}
            data.chunk_texts[chunk_id] = text

        self._data = data
        self._rebuild_derived()

        self.pretty.write(
            "O",
            "Graph",
            f"Built graph index: {len(self._data.chunk_entities)} chunks, "
            f"{len(self._data.adjacency)} entities",
        )

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _index_path(self, graph_directory: str) -> str:
        return os.path.join(graph_directory, self.INDEX_FILENAME)

    def _persist(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.pretty.write(
            "O",
            "Graph",
            f"Persisted graph index to {path} ({len(self._data.chunk_entities)} chunks)",
        )

    def _load(self, path: str) -> None:
        with gzip.open(path, "rb") as f:
            self._data = pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    # Internal — filter matching (mirrors BM25Retriever._matches_filter)
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
