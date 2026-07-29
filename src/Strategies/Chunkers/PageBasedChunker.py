import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents.base import Document as langchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy, ChunkResult

# (page/slide number, label/title, body text)
PageData = tuple[int, str, str]

# (chunk text, per-chunk extra metadata)
_PageText = tuple[str, dict[str, Any]]


class PageBasedChunker(ChunkerStrategy, ABC):
    """Abstract base for chunkers that split a document into discrete pages or
    slides, each of which becomes one chunk (or a handful of sub-chunks if
    the page exceeds ``MAX_CHUNK_SIZE`` words).

    Subclasses must implement:

    * ``_parse_pages(file_type, file_path, content)`` — extract structured
      pages from the source file (or fall back to flat content).
    * ``_format_prefix(num, title)`` — build the per-chunk prefix string,
      e.g. ``"Slide 3: My Title"`` or ``"Page 6"``.

    Optional hook:

    * ``_extra_meta_for_page(num, title)`` — extra metadata dict to merge
      into each chunk that belongs to this page/slide (default: empty).
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
        self._pretty: PrettyWriter = PrettyWriter()

        chunker_slot: str = (
            f"_CHUNKERS.{chunker_name}"
            if chunker_name
            else self._helpers.get_chunker_config_slot()
        )
        self._max_chunk_size: int = self._cfg.get_int(
            f"{chunker_slot}.MAX_CHUNK_SIZE", 256
        )
        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")

    # -- ChunkerStrategy interface ------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._max_chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        self.perf_logger.log(
            "PageBasedChunker.chunk",
            "chunker",
            f"start content_len={len(content)}",
        )
        _t0 = time.perf_counter()

        file_type: str = str(metadata.get("FileType", "")).lower()
        file_path: str = str(metadata.get("FilePath", ""))

        pages: list[PageData] = self._parse_pages(file_type, file_path, content)
        if not pages:
            result = [], None
            elapsed = time.perf_counter() - _t0
            self.perf_logger.log(
                "PageBasedChunker.chunk",
                "chunker",
                f"stop n=0 elapsed={elapsed:.3f}s",
            )
            return result

        page_texts: list[_PageText] = self._pages_to_texts(pages)
        result = self._to_docs(page_texts, metadata), None
        elapsed = time.perf_counter() - _t0
        self.perf_logger.log(
            "PageBasedChunker.chunk",
            "chunker",
            f"stop n={len(result[0])} elapsed={elapsed:.3f}s",
        )
        return result

    # -- Subclass interface -------------------------------------------------

    @abstractmethod
    def _parse_pages(
        self, file_type: str, file_path: str, content: str
    ) -> list[PageData]:
        """Return a list of ``(number, title, body)`` triples.

        Implementations should fall back to ``[(1, "", content.strip())]``
        when structured parsing is unavailable.
        """

    def _format_prefix(self, num: int, title: str) -> str:
        """Build the prefix line prepended to each chunk."""
        return f"Page {num}: {title}" if title else f"Page {num}"

    def _extra_meta_for_page(self, num: int, title: str) -> dict[str, Any]:
        """Extra metadata merged into every chunk belonging to this page.

        Override to attach e.g. ``{"PageNumber": num}`` or
        ``{"SlideNumber": num}``.
        """
        return {}

    # -- Chunk assembly -----------------------------------------------------

    def _pages_to_texts(self, pages: list[PageData]) -> list[_PageText]:
        """Convert parsed pages into ``(text, extra_meta)`` pairs.

        Each page becomes one chunk unless its word count exceeds
        ``MAX_CHUNK_SIZE``, in which case it is split with
        ``RecursiveCharacterTextSplitter`` and each sub-chunk carries the
        same prefix and extra metadata.
        """
        result: list[_PageText] = []

        for num, title, body in pages:
            prefix: str = self._format_prefix(num, title)
            extra: dict[str, Any] = self._extra_meta_for_page(num, title)

            if not body:
                # Label/title-only page — still worth embedding
                if prefix:
                    result.append((prefix, extra))
                continue

            chunk_text: str = f"{prefix}\n\n{body}" if prefix else body

            if self._fileUtils.count_words(chunk_text) <= self._max_chunk_size:
                result.append((chunk_text, extra))
            else:
                for sub in self._split_oversized(chunk_text, prefix):
                    result.append((sub, extra))

        return result

    def _split_oversized(self, text: str, prefix: str) -> list[str]:
        """Split an oversized page, re-prepending ``prefix`` to each sub-chunk."""
        separators_str: list[str] = [str(s) for s in self._separators]
        prefix_words: int = self._fileUtils.count_words(prefix) + 1 if prefix else 0
        body_budget: int = max(50, self._max_chunk_size - prefix_words)

        body: str = text[len(prefix) :].lstrip("\n") if prefix else text

        splitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self._fileUtils.count_words,
            chunk_size=body_budget,
            chunk_overlap=0,
        )
        parts: list[str] = [d.page_content for d in splitter.create_documents([body])]

        if prefix:
            return [f"{prefix}\n\n{part}" for part in parts]
        return parts

    # -- Doc assembly -------------------------------------------------------

    @staticmethod
    def _to_docs(
        page_texts: list[_PageText], metadata: dict[str, Any]
    ) -> list[langchainDoc]:
        docs: list[langchainDoc] = []
        for i, (text, extra) in enumerate(page_texts):
            meta: dict[str, Any] = dict(metadata)
            meta["MyChunk"] = i
            meta.update(extra)
            docs.append(
                langchainDoc(
                    page_content=text,
                    metadata=meta,
                    id=str(uuid.uuid4()),
                )
            )
        return docs
