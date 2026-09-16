"""Structural protocol for local retriever implementations.

This protocol captures the shared lifecycle/query surface used by the local
retrieval orchestrator. It is intentionally behavior-free so existing retrievers
(BM25, Graph, Regex) can adopt it without changing scoring logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Shared contract for persistent local retrieval indexes."""

    def get_index_dir(self, collection_name: str) -> str:
        """Return the index directory for a collection."""

    def is_loaded_for(self, collection_name: str) -> bool:
        """Return True when in-memory state matches the collection."""

    def load_or_rebuild(
        self,
        index_directory: str,
        collection_name: str,
        collection: Any,
        /,
    ) -> None:
        """Load persisted index state or rebuild from the collection."""

    def build_and_persist(
        self,
        index_directory: str,
        collection_name: str,
        collection: Any,
        /,
    ) -> None:
        """Force full rebuild and persist to disk."""

    def persist(self, index_directory: str, /) -> None:
        """Persist current in-memory state to disk."""

    def ingest_file(
        self,
        file_path: str,
        collection_name: str,
        collection: Any,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Apply incremental update for a single source file."""

    def query(
        self,
        query_text: str,
        k: int = 100,
        file_filter: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return ranked retrieval results as document-like objects."""
