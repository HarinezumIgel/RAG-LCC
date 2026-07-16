import uuid
from typing import Any

from langchain_core.documents.base import Document as langchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.Config import Config
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy, ChunkResult
from Strategies.Chunkers.SentenceSplitter import SentenceSplitter


class SentenceWindowChunker(ChunkerStrategy):
    """Chunks documents by grouping consecutive sentences up to a word budget.

    Splits text into sentences using punctuation/newline boundaries, then
    greedily packs sentences into chunks without exceeding
    ``MAX_CHUNK_SIZE`` words.  No embeddings or GPU required — a fast,
    boundary-aware alternative to fixed-size windowing.

    Oversized individual sentences (longer than ``MAX_CHUNK_SIZE``) are
    split with ``RecursiveCharacterTextSplitter`` as a safety fallback.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        helpers: "Helpers | None" = None,
        file_utils: "FileUtils | None" = None,
        chunker_name: str | None = None,
    ) -> None:
        self._cfg: Config = cfg or Config()
        self._helpers: Helpers = helpers or Helpers()
        self._fileUtils: FileUtils = file_utils or FileUtils()

        chunker_slot: str = (
            f"_CHUNKERS.{chunker_name}"
            if chunker_name
            else self._helpers.get_chunker_config_slot()
        )
        self._max_chunk_size: int = self._cfg.get_int(f"{chunker_slot}.MAX_CHUNK_SIZE")
        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")

    # -- ChunkerStrategy interface ------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._max_chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        sentences: list[str] = self._split_sentences(content)

        if not sentences:
            return [], None

        groups: list[str] = self._pack_sentences(sentences)

        # Safety cap: split oversized individual sentences
        final_texts: list[str] = []
        for group in groups:
            if self._fileUtils.count_words(group) > self._max_chunk_size:
                final_texts.extend(self._split_oversized(group))
            else:
                final_texts.append(group)

        # No pre-computed embeddings — caller will embed all chunks
        return self._to_docs(final_texts, metadata), None

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using shared boundary detection."""
        return SentenceSplitter.split_sentences(text)

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        """Greedily pack sentences into chunks up to ``_max_chunk_size`` words."""
        groups: list[str] = []
        current: list[str] = []
        current_words: int = 0

        for sentence in sentences:
            sent_words: int = self._fileUtils.count_words(sentence)

            if current and current_words + sent_words > self._max_chunk_size:
                groups.append(" ".join(current))
                current = []
                current_words = 0

            current.append(sentence)
            current_words += sent_words

        if current:
            groups.append(" ".join(current))

        return groups

    def _split_oversized(self, text: str) -> list[str]:
        """Fall back to RecursiveCharacterTextSplitter for oversized sentences."""
        separators_str: list[str] = [str(s) for s in self._separators]
        splitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self._fileUtils.count_words,
            chunk_size=self._max_chunk_size,
            chunk_overlap=0,
        )
        return [d.page_content for d in splitter.create_documents([text])]

    @staticmethod
    def _to_docs(texts: list[str], metadata: dict[str, Any]) -> list[langchainDoc]:
        """Wrap text segments into langchainDocs with UUIDs and MyChunk index."""
        docs: list[langchainDoc] = []
        for i, text in enumerate(texts):
            meta = dict(metadata)
            meta["MyChunk"] = i
            docs.append(
                langchainDoc(page_content=text, metadata=meta, id=str(uuid.uuid4()))
            )
        return docs
