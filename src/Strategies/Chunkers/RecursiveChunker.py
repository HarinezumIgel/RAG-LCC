import uuid
from typing import Any

from langchain_core.documents.base import Document as langchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.Config import Config
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy, ChunkResult


class RecursiveChunker(ChunkerStrategy):
    """Chunks documents using RecursiveCharacterTextSplitter.

    Tries a hierarchy of separators (paragraph breaks, newlines, spaces,
    then character-by-character) and only falls back to the next level when
    a chunk still exceeds the configured word-count limit.  Overlap between
    consecutive chunks is configurable."""

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

        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")
        chunker_slot: str = (
            f"_CHUNKERS.{chunker_name}"
            if chunker_name
            else self._helpers.get_chunker_config_slot()
        )
        self._chunk_size: int = self._cfg.get_int(f"{chunker_slot}.CHUNK_SIZE")
        self._overlap: int = self._cfg.get_int(f"{chunker_slot}.CHUNK_OVERLAP", 0)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        stop_words: list[str] = self._fileUtils.get_stopwords(content)
        if stop_words:
            cleaned: str | None = self._fileUtils.removeStopwords(
                content, set(stop_words)
            )
        else:
            cleaned = content

        separators_str: list[str] = [str(s) for s in self._separators]
        splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self._fileUtils.count_words,
            chunk_size=self._chunk_size,
            chunk_overlap=self._overlap,
        )

        base_doc: langchainDoc = langchainDoc(
            page_content=cleaned or "", metadata=metadata
        )
        slices: list[langchainDoc] = splitter.split_documents([base_doc])

        ids: list[str] = [str(uuid.uuid4()) for _ in slices]
        chunks: list[langchainDoc] = []
        for i, s in enumerate(slices):
            meta = dict(metadata)
            meta["MyChunk"] = i
            chunks.append(
                langchainDoc(page_content=s.page_content, metadata=meta, id=ids[i])
            )
        # No pre-computed embeddings — caller will embed all chunks
        return chunks, None
