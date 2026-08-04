import time
import uuid
from typing import Any

from langchain_core.documents.base import Document as langchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.Config import Config
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy, ChunkResult
from Strategies.Chunkers.SentenceSplitter import SentenceSplitter


class SlidingWindowChunker(ChunkerStrategy):
    """Chunks documents with a sliding window of sentences and configurable overlap.

    Like ``SentenceWindowChunker`` but keeps ``OVERLAP_SENTENCES`` trailing
    sentences from the previous chunk at the start of the next one.  This
    ensures that information near chunk boundaries appears in at least two
    chunks, significantly improving retrieval for queries that fall on a
    boundary.

    No embeddings or GPU required.
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
        self.perf_logger: PerfLogger = PerfLogger()

        chunker_slot: str = (
            f"_CHUNKERS.{chunker_name}"
            if chunker_name
            else self._helpers.get_chunker_config_slot()
        )
        self._max_chunk_size: int = self._cfg.get_int(f"{chunker_slot}.MAX_CHUNK_SIZE")
        self._overlap_sentences: int = self._cfg.get_int(
            f"{chunker_slot}.OVERLAP_SENTENCES", 2
        )
        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")

    # -- ChunkerStrategy interface ------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._max_chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        self.perf_logger.log(
            "SlidingWindowChunker.chunk",
            "chunker",
            f"start content_len={len(content)}",
        )
        _t0 = time.perf_counter()

        sentences: list[str] = SentenceSplitter().split_sentences(content)

        if not sentences:
            return [], None

        groups: list[str] = self._slide_sentences(sentences)

        # Safety cap: split oversized individual sentences
        final_texts: list[str] = []
        for group in groups:
            if self._fileUtils.count_words(group) > self._max_chunk_size:
                final_texts.extend(self._split_oversized(group))
            else:
                final_texts.append(group)

        # No pre-computed embeddings — caller will embed all chunks
        result = self._to_docs(final_texts, metadata), None

        elapsed = time.perf_counter() - _t0
        self.perf_logger.log(
            "SlidingWindowChunker.chunk",
            "chunker",
            f"stop n={len(result[0])} elapsed={elapsed:.3f}s",
        )
        return result

    # -- Internal helpers ---------------------------------------------------

    def _slide_sentences(self, sentences: list[str]) -> list[str]:
        """Pack sentences into chunks with sliding overlap.

        When a chunk fills up, the next chunk starts ``_overlap_sentences``
        sentences back from the end of the previous chunk.
        """
        groups: list[str] = []
        start: int = 0

        while start < len(sentences):
            current: list[str] = []
            current_words: int = 0

            i = start
            while i < len(sentences):
                sent_words: int = self._fileUtils.count_words(sentences[i])

                if current and current_words + sent_words > self._max_chunk_size:
                    break

                current.append(sentences[i])
                current_words += sent_words
                i += 1

            if current:
                groups.append(" ".join(current))

            # Advance: move start forward, but keep overlap_sentences of context
            new_sentences_added = len(current)
            advance = max(1, new_sentences_added - self._overlap_sentences)
            start += advance

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

    def _to_docs(
        self, texts: list[str], metadata: dict[str, Any]
    ) -> list[langchainDoc]:
        """Wrap text segments into langchainDocs with UUIDs and MyChunk index."""
        docs: list[langchainDoc] = []
        for i, text in enumerate(texts):
            meta = dict(metadata)
            meta["MyChunk"] = i
            docs.append(
                langchainDoc(page_content=text, metadata=meta, id=str(uuid.uuid4()))
            )
        return docs
