from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.documents.base import Document as langchainDoc

from Config.Config import Config

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from Strategies.Chunkers.TableChunker import DocumentBlock

# Return type for chunk(): a list of documents and an optional list of
# pre-computed embeddings (one per doc).  Entries may be ``None`` when the
# embedding could not be derived cheaply (e.g. oversized splits).
# Chunkers that do not produce embeddings return ``None`` for the whole list.
ChunkResult = tuple[list[langchainDoc], list[list[float] | None] | None]


def format_pdf_page_reference(number: int, label: str) -> str:
    return f"Page {label}" if label else f"PDF page {number}"


class ChunkerStrategy(ABC):
    """Base class for all chunking strategies."""

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...

    @abstractmethod
    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult: ...

    def chunk_prose(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        """Chunk an already-extracted prose block without reopening its source."""
        return self.chunk(content, metadata)

    def chunk_document(
        self,
        content: str,
        metadata: dict[str, Any],
        blocks: list["DocumentBlock"],
        tokenizer: "PreTrainedTokenizerBase | None" = None,
    ) -> ChunkResult:
        """Delegate tables without exposing table assembly to ingestion.

        All strategies share this entry point. Heading/page strategies override
        prose handling so normalized blocks never trigger original-file re-reads.
        Documents without structured tables retain their existing chunk() path.

        Detection happens during loading: TableChunker.extract_word() walks
        native paragraph/table elements; extract_pdf() uses pdfplumber's
        find_tables(). The loader passes ordered, normalized/masked blocks here.
        This method consumes that structure; it does not detect tables in text.
        """
        from Strategies.Chunkers.TableChunker import TableChunker

        if not blocks:
            return self.chunk(content, metadata)
        if tokenizer is None:
            from Commons.Exceptions import DataProcessingError

            raise DataProcessingError(
                "Structured table chunking requires the embedding tokenizer."
            )
        tables = TableChunker(
            tokenizer, Config().get_int("_CHUNKERS.TABLE.MAX_CHUNK_TOKENS")
        )
        docs: list[langchainDoc] = []
        embeddings: list[list[float] | None] = []
        # Preserve source order: prose -> table -> following prose. TableChunker
        # is a delegate, not a replacement for the configured document chunker.
        for block in blocks:
            meta = dict(metadata)
            meta.update(block.metadata())
            # A positive document-local table index marks a detected table.
            # Delegate only that block so each fragment retains its identity
            # and headers; subsequent prose resumes the selected strategy.
            if block.table_index:
                chunks = tables.chunk_table(block, meta)
                vectors = None
            else:
                chunks, vectors = self.chunk_prose(block.text, meta)
            docs.extend(chunks)
            # Table vectors are computed later from the final serialized text;
            # keep any prose vectors aligned with the combined chunk sequence.
            embeddings.extend(vectors if vectors is not None else [None] * len(chunks))
        for i, doc in enumerate(docs):
            doc.metadata["MyChunk"] = i
        return docs, embeddings
