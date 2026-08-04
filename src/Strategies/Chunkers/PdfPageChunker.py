import os
import re
from typing import Any

from Config.Config import Config
from Gui.Colors import ORANGE
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.Chunkers.PageBasedChunker import PageBasedChunker, PageData

# Strict roman-numeral validator (non-empty), case-insensitive.
_ROMAN_RE = re.compile(
    r"^(?=[mdclxvi])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)
# Page number at the very END of a running footer, e.g.
# "© Copyright Lenovo 2020, 2021 iii" or "Chapter 1 . Meet your computer 3".
# Language-agnostic (position-based).
_PAGE_TRAILING_RE = re.compile(r"\S\s+([0-9]{1,4}|[ivxlcdm]+)\s*$", re.IGNORECASE)
# Page number at the very START of a running header, e.g. "ii P620 User Guide".
# The mandatory whitespace after the token rejects list labels like "7." .
_PAGE_LEADING_RE = re.compile(r"^([0-9]{1,4}|[ivxlcdm]+)\s+\S", re.IGNORECASE)


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
        # Field name under which the printed page label (e.g. "iii", "1") is
        # stored. Empty string disables label capture (legacy "Page N" only).
        enabled: bool = self._cfg.get_bool(
            "_METADATA_EXTRACTION.ENABLED", False, silent=True
        )
        self._page_label_field: str = (
            self._cfg.get_str(
                "_METADATA_EXTRACTION.PDF_PAGE_LABEL_FIELD", "", silent=True
            )
            if enabled
            else ""
        )
        # Best-effort: recover the *printed* page number (e.g. roman front
        # matter) from the page's footer/header text when the PDF's own
        # /PageLabels metadata does not declare it.
        self._detect_labels: bool = self._cfg.get_bool(
            "_CHUNKERS.PDF_PAGE.DETECT_PRINTED_LABEL", True, silent=True
        )
        # Per-document "page N" / "page N of M" matchers; rebuilt in
        # _parse_pdf with the document's language (default English here).
        self._page_token_re, self._page_of_re = self._build_page_patterns("")

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
        # ``title`` carries the printed page label when capture is enabled.
        label: str = title if (self._page_label_field and title) else str(num)
        return f"Page {label}"

    def _extra_meta_for_page(self, num: int, title: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"PageNumber": num}
        if self._page_label_field and title:
            meta[self._page_label_field] = title
        return meta

    # -- PDF parser ---------------------------------------------------------

    def _parse_pdf(self, file_path: str) -> list[PageData]:
        """Extract per-page text from a PDF using *pypdf*.

        Returns a list of ``(physical_page_number, page_label, page_text)``
        triples — one per page that yields non-empty extractable text.  The
        physical page number is the 1-based sequential index; the page label
        is the document's printed page label (e.g. ``"iii"``, ``"1"``) when
        available, else the physical number as a string.  Entirely blank
        pages (e.g. image-only pages with no text layer) are skipped.
        """
        from pypdf import PdfReader  # pypdf is a declared project dependency

        # Resolve the language-specific "page"/"of" matchers once per document.
        self._page_token_re, self._page_of_re = self._build_page_patterns(
            self._doc_language
        )
        reader = PdfReader(file_path)
        labels: list[str] = self._page_labels(reader)
        pages: list[PageData] = []

        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                raw_label: str = labels[i - 1] if i - 1 < len(labels) else ""
                printed: str = (
                    self._detect_printed_label(text) if self._detect_labels else ""
                )
                # Prefer the printed page number recovered from the page text;
                # otherwise use the PDF's declared label only when it differs
                # from the physical index (a real label, not a duplicate).
                if printed:
                    label = printed
                elif raw_label and raw_label != str(i):
                    label = raw_label
                else:
                    label = ""
                pages.append((i, label, text))

        return pages

    def _detect_printed_label(self, text: str) -> str:
        """Recover a printed page number from a page's footer/header lines."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Footer first (most common), then header; check the two outermost
        # lines on each side to tolerate a trailing separator line.
        candidates = [lines[-1], lines[0]]
        if len(lines) > 1:
            candidates += [lines[-2], lines[1]]
        for line in candidates:
            label = self._match_page_token(line)
            if label:
                return label
        return ""

    def _match_page_token(self, line: str) -> str:
        """Return the page label if *line* is (essentially) a page number."""
        m = (
            self._page_of_re.search(line)
            or self._page_token_re.match(line)
            or _PAGE_TRAILING_RE.search(line)
            or _PAGE_LEADING_RE.match(line)
        )
        if not m:
            return ""
        token = m.group(1)
        if token.isdigit():
            return token
        return token.lower() if _ROMAN_RE.match(token) else ""

    def _build_page_patterns(
        self, lang: str
    ) -> "tuple[re.Pattern[str], re.Pattern[str]]":
        """Build the "page N" / "page N of M" matchers for the document language.

        The English keywords are always included; for non-English documents the
        translated "page"/"of" words are added using the shared, cached banned-
        word translator (translation only — nothing is added to any banlist).
        """
        page_words: set[str] = {"page"}
        of_words: set[str] = {"of"}
        lang = (lang or "").lower()
        if lang and not lang.startswith("en"):
            try:
                from Compliance.SharedHelpers import SharedHelpers

                shared = SharedHelpers()
                for word, bucket in (("page", page_words), ("of", of_words)):
                    translated = shared.translate_text(
                        word, target_lang=lang, source_lang="en"
                    )
                    if translated and translated.strip():
                        bucket.add(translated.strip().lower())
            except Exception:
                pass
        page_alt = "|".join(
            re.escape(w) for w in sorted(page_words, key=len, reverse=True)
        )
        of_alt = "|".join(re.escape(w) for w in sorted(of_words, key=len, reverse=True))
        page_token_re = re.compile(
            rf"^\s*(?:(?:{page_alt})\s+)?[\-\u2013\u2014.\s]*"
            rf"([0-9]{{1,4}}|[ivxlcdm]+)[\-\u2013\u2014.\s]*$",
            re.IGNORECASE,
        )
        page_of_re = re.compile(
            rf"\b(?:{page_alt})\s+([0-9]{{1,4}}|[ivxlcdm]+)\s+"
            rf"(?:{of_alt})\s+[0-9]{{1,4}}\b",
            re.IGNORECASE,
        )
        return page_token_re, page_of_re

    def _page_labels(self, reader: Any) -> list[str]:
        """Return the PDF's printed page labels as strings, or ``[]`` if absent."""
        try:
            raw = list(getattr(reader, "page_labels", []) or [])
        except Exception:
            return []
        return [str(x) for x in raw]
