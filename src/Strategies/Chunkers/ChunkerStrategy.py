from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents.base import Document as langchainDoc

# Return type for chunk(): a list of documents and an optional list of
# pre-computed embeddings (one per doc).  Entries may be ``None`` when the
# embedding could not be derived cheaply (e.g. oversized splits).
# Chunkers that do not produce embeddings return ``None`` for the whole list.
ChunkResult = tuple[list[langchainDoc], list[list[float] | None] | None]


class ChunkerStrategy(ABC):
    """Base class for all chunking strategies."""

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...

    @abstractmethod
    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult: ...
