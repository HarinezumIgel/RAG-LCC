import os
import time
from typing import Any

from Config.Config import Config
from Gui.Colors import ORANGE
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.Chunkers.PageBasedChunker import PageBasedChunker, PageData


class PdfPageChunker(PageBasedChunker):
    """Page-aware chunker for PDF files.

    Re-reads the PDF via *pypdf* to recover hard page boundaries that are
    lost during the flat text extraction in the loading pipeline.  Each
    PDF page becomes one chunk (prefixed ``"Page N"``).  Pages whose word
    count exceeds ``MAX_CHUNK_SIZE`` are split with
    ``RecursiveCharacterTextSplitter`` and each sub-chunk keeps the prefix.

    An integer ``PageNumber`` field is added to every chunk's metadata so
    callers can trace a chunk back to its source page.

    For non-PDF file types (or when the file is inaccessible / password-
    protected) the chunker logs a warning and falls back to treating the
    entire pre-extracted content as a single chunk.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        helpers: "Helpers | None" = None,
        file_utils: "FileUtils | None" = None,
        chunker_name: str | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            helpers=helpers,
            file_utils=file_utils,
            chunker_name=chunker_name,
        )
        self.perf_logger: PerfLogger = PerfLogger()

    # -- PageBasedChunker interface -----------------------------------------

    def _parse_pages(
        self, file_type: str, file_path: str, content: str
    ) -> list[PageData]:
        if file_type == "pdf" and file_path and os.path.isfile(file_path):
            try:
                return self._parse_pdf(file_path)
            except Exception as exc:
                self._pretty.write(
                    "W",
                    "PdfPageChunker",
                    f"PDF parsing failed ({exc}) — falling back to flat content."
                    f" File: {file_path}",
                    color=ORANGE,
                )
        elif file_type != "pdf":
            self._pretty.write(
                "W",
                "PdfPageChunker",
                f"No PDF structure available for .{file_type}"
                f" — falling back to flat content. File: {file_path}",
                color=ORANGE,
            )
        else:
            self._pretty.write(
                "W",
                "PdfPageChunker",
                f"PDF file not accessible — falling back to flat content."
                f" File: {file_path}",
                color=ORANGE,
            )

        text = content.strip()
        return [(1, "", text)] if text else []

    def _format_prefix(self, num: int, title: str) -> str:
        return f"Page {num}"

    def _extra_meta_for_page(self, num: int, title: str) -> dict[str, Any]:
        return {"PageNumber": num}

    # -- PDF parser ---------------------------------------------------------

    @staticmethod
    def _parse_pdf(file_path: str) -> list[PageData]:
        """Extract per-page text from a PDF using *pypdf*.

        Returns a list of ``(page_number, "", page_text)`` triples — one per
        page that yields non-empty extractable text.  Entirely blank pages
        (e.g. image-only pages with no text layer) are skipped.
        """
        from pypdf import PdfReader  # pypdf is a declared project dependency

        reader = PdfReader(file_path)
        pages: list[PageData] = []

        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, "", text))

        return pages
