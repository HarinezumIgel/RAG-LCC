import os
import re
import time
import uuid
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

# Heading level → heading text → body text under it
_Section = tuple[int, str, str]

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class HeadingChunker(ChunkerStrategy):
    """Structure-aware chunker that splits on document headings.

    Supported formats:
    * **Markdown** (``.md``): detects ``# … ######`` heading markers in content.
    * **DOCX / DOC**: re-reads the file via *python-docx* to inspect paragraph
      styles (``Heading 1`` … ``Heading 9``).

    For any other file type the chunker falls back to flat sentence-based
    splitting identical to ``SlidingWindowChunker``.

    Each chunk is prefixed with its heading breadcrumb so that retrieval
    embeddings capture the section context even after the chunk is detached
    from the full document.
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
        mode: str = self._cfg.get_str(
            f"{chunker_slot}.BREADCRUMB_MODE", "suffix", silent=True
        ).lower()
        if mode not in ("prefix", "suffix", "off"):
            self._pretty.write(
                "W",
                "HeadingChunker",
                f"Unknown BREADCRUMB_MODE '{mode}' — falling back to 'suffix'",
                color=ORANGE,
            )
            mode = "suffix"
        self._breadcrumb_mode: str = mode
        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")

    # -- ChunkerStrategy interface ------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._max_chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        self.perf_logger.log(
            "HeadingChunker.chunk",
            "chunker",
            f"start content_len={len(content)}",
        )
        _t0 = time.perf_counter()

        file_type: str = str(metadata.get("FileType", "")).lower()
        file_path: str = str(metadata.get("FilePath", ""))

        if file_type == "md":
            sections = self._parse_md(content)
        elif file_type == "docx" and file_path and os.path.isfile(file_path):
            sections = self._parse_docx(file_path)
        else:
            # Flat fallback — heading structure is unavailable
            self._pretty.write(
                "W",
                "HeadingChunker",
                f"No heading structure available for .{file_type} — falling back to flat splitting. File: {file_path}",
                color=ORANGE,
            )
            sections = self._parse_flat(content)

        if not sections:
            result = [], None
            elapsed = time.perf_counter() - _t0
            self.perf_logger.log(
                "HeadingChunker.chunk",
                "chunker",
                f"stop n=0 elapsed={elapsed:.3f}s",
            )
            return result

        pairs: list[tuple[str, str]] = self._sections_to_texts(sections)
        # No pre-computed embeddings — caller will embed all chunks
        result = self._to_docs(pairs, metadata), None
        elapsed = time.perf_counter() - _t0
        self.perf_logger.log(
            "HeadingChunker.chunk",
            "chunker",
            f"stop n={len(result[0])} elapsed={elapsed:.3f}s",
        )
        return result

    # -- Parsers ------------------------------------------------------------

    @staticmethod
    def _parse_md(content: str) -> list[_Section]:
        """Split Markdown content on ``#``-style headings."""
        sections: list[_Section] = []
        lines = content.split("\n")
        current_level: int = 0
        current_heading: str = ""
        body_lines: list[str] = []

        for line in lines:
            m = _MD_HEADING_RE.match(line)
            if m:
                # Flush previous section
                body = "\n".join(body_lines).strip()
                if current_heading or body:
                    sections.append((current_level, current_heading, body))
                current_level = len(m.group(1))
                current_heading = m.group(2).strip()
                body_lines = []
            else:
                body_lines.append(line)

        # Flush last section
        body = "\n".join(body_lines).strip()
        if current_heading or body:
            sections.append((current_level, current_heading, body))

        return sections

    @staticmethod
    def _parse_docx(file_path: str) -> list[_Section]:
        """Re-read a DOCX file and split on Heading styles."""
        from docx import \
            Document as DocxDocument  # type: ignore[import-untyped]

        doc = DocxDocument(file_path)
        sections: list[_Section] = []
        current_level: int = 0
        current_heading: str = ""
        body_parts: list[str] = []

        for para in doc.paragraphs:
            style_name: str = (para.style.name or "") if para.style else ""
            heading_level = _docx_heading_level(style_name)

            if heading_level > 0:
                # Flush previous section
                body = "\n".join(body_parts).strip()
                if current_heading or body:
                    sections.append((current_level, current_heading, body))
                current_level = heading_level
                current_heading = para.text.strip()
                body_parts = []
            else:
                text = para.text.strip()
                if text:
                    body_parts.append(text)

        # Flush last section
        body = "\n".join(body_parts).strip()
        if current_heading or body:
            sections.append((current_level, current_heading, body))

        return sections

    def _parse_flat(self, content: str) -> list[_Section]:
        """Fallback: treat entire content as a single level-0 section."""
        text = content.strip()
        if not text:
            return []
        return [(0, "", text)]

    # -- Chunk assembly -----------------------------------------------------

    def _sections_to_texts(self, sections: list[_Section]) -> list[tuple[str, str]]:
        """Convert parsed sections into ``(chunk_text, breadcrumb)`` pairs.

        The breadcrumb (``H1 > H2 > H3``) is always returned so downstream code
        can persist it into metadata.  Its placement inside ``chunk_text`` is
        controlled by ``BREADCRUMB_MODE``:

        * ``prefix`` — prepend to the body (legacy behaviour).
        * ``suffix`` — append after the body (default; keeps leading tokens
          focused on actual content while retaining section context in the
          embedding).
        * ``off``    — omit from the chunk text entirely.

        Oversized sections are split; every sub-chunk carries the same
        breadcrumb placement.
        """
        # Build breadcrumb trail: track active heading per level
        trail: dict[int, str] = {}
        pairs: list[tuple[str, str]] = []

        for level, heading, body in sections:
            if heading:
                trail[level] = heading
                # Clear deeper levels
                for deeper in [k for k in trail if k > level]:
                    del trail[deeper]

            breadcrumb = " > ".join(trail[k] for k in sorted(trail) if trail[k])

            if not body:
                continue

            chunk_text = self._format_chunk(body, breadcrumb)

            if self._fileUtils.count_words(chunk_text) <= self._max_chunk_size:
                pairs.append((chunk_text, breadcrumb))
            else:
                for part in self._split_oversized(body, breadcrumb):
                    pairs.append((part, breadcrumb))

        return pairs

    def _format_chunk(self, body: str, breadcrumb: str) -> str:
        """Combine body and breadcrumb according to ``BREADCRUMB_MODE``."""
        if not breadcrumb or self._breadcrumb_mode == "off":
            return body
        if self._breadcrumb_mode == "prefix":
            return f"{breadcrumb}\n\n{body}"
        # suffix
        return f"{body}\n\n[section: {breadcrumb}]"

    def _split_oversized(self, body: str, breadcrumb: str) -> list[str]:
        """Split an oversized section body, re-applying the breadcrumb to each
        resulting sub-chunk per ``BREADCRUMB_MODE``."""
        separators_str: list[str] = [str(s) for s in self._separators]
        # Reserve a few words of budget for the breadcrumb when it is embedded
        # in chunk text; none when it lives only in metadata.
        reserve = 0
        if breadcrumb and self._breadcrumb_mode != "off":
            reserve = self._fileUtils.count_words(breadcrumb) + 3
        body_budget = max(50, self._max_chunk_size - reserve)

        splitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self._fileUtils.count_words,
            chunk_size=body_budget,
            chunk_overlap=0,
        )
        parts = [d.page_content for d in splitter.create_documents([body])]
        return [self._format_chunk(p, breadcrumb) for p in parts]

    # -- Doc assembly -------------------------------------------------------

    @staticmethod
    def _to_docs(
        pairs: list[tuple[str, str]], metadata: dict[str, Any]
    ) -> list[langchainDoc]:
        docs: list[langchainDoc] = []
        for i, (text, breadcrumb) in enumerate(pairs):
            meta = dict(metadata)
            meta["MyChunk"] = i
            if breadcrumb:
                meta["HeadingPath"] = breadcrumb
            docs.append(
                langchainDoc(page_content=text, metadata=meta, id=str(uuid.uuid4()))
            )
        return docs


def _docx_heading_level(style_name: str) -> int:
    """Extract heading level from a python-docx style name.

    Returns 1-9 for 'Heading 1' … 'Heading 9', 0 for non-heading styles.
    """
    m = re.match(r"Heading\s+(\d)", style_name)
    return int(m.group(1)) if m else 0
