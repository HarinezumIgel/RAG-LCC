# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""pdfplumber + pypdf PDF chunk highlighter (MIT / BSD-3-Clause).

Strategy
--------
* ``pdfplumber`` (MIT) extracts word-level bounding boxes using a
  top-left-origin coordinate system (``x0``, ``top``, ``x1``, ``bottom``).
* ``pypdf`` (BSD-3) writes ``/Highlight`` annotations at those positions
  and serialises the result to bytes — entirely in memory, no disk writes.

Coordinate conversion
---------------------
pdfplumber uses a top-left origin; PDF annotations use a bottom-left
origin.  The conversion for a word ``w`` on a page of height ``H`` is::

    pdf_y0 = H - w["bottom"]   # bottom of the word in PDF coords
    pdf_y1 = H - w["top"]      # top    of the word in PDF coords
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from VisualMarkers.VisualMarker import ChunkSnippet, VisualMarker

# Strip a leading "Page N" header inserted by PdfPageChunker before
# searching the actual page text.
_PAGE_PREFIX_RX = re.compile(r"^\s*Page\s+\d+\s*\n+", re.IGNORECASE)
# Minimum length of a fallback line/sentence to bother searching for.
_MIN_FRAGMENT_LEN = 12
# Used to normalise text before token comparison.
_PUNCT_RX = re.compile(r"[^\w\s]")

# Named highlight colours → 6-char hex strings (no '#') as accepted by
# pypdf's Highlight.highlight_color parameter.
_NAMED_COLORS: dict[str, str] = {
    "yellow": "ffff00",
    "green": "00ff00",
    "cyan": "00ffff",
    "magenta": "ff00ff",
    "pink": "ffbfcc",
    "red": "ff0000",
    "blue": "0000ff",
    "orange": "ffa600",
    "lime": "80ff00",
}


def _parse_color(color: str | None) -> str:
    """Convert a CSS color name or ``#RRGGBB`` hex string to a 6-char hex string
    (no ``#`` prefix) as expected by ``pypdf`` ``Highlight.highlight_color``.

    Missing or invalid values fall back to a safe default highlight color.
    """
    if not isinstance(color, str) or not color.strip():
        return "ffff00"

    c = color.strip().lower()
    if c in _NAMED_COLORS:
        return _NAMED_COLORS[c]
    if c.startswith("#"):
        hex_part = c[1:]
        try:
            if len(hex_part) == 6:
                int(hex_part, 16)  # validate
                return hex_part
            if len(hex_part) == 3:
                return hex_part[0] * 2 + hex_part[1] * 2 + hex_part[2] * 2
        except ValueError:
            pass
    return "ffff00"  # fallback yellow


class PdfVisualMarker(VisualMarker):
    """Highlight retrieved chunks in a PDF using pdfplumber + pypdf."""

    def mark_to_bytes(
        self,
        source_path: Path,
        snippets: Sequence[ChunkSnippet],
        *,
        highlight_color: str = "yellow",
    ) -> bytes:
        # Lazy imports so missing packages only fail when mark_text=True.
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "pdfplumber is required for PDF visual marking. "
                "Install with: pip install pdfplumber pypdf"
            ) from exc
        try:
            from pypdf import PdfReader, PdfWriter
            from pypdf.annotations import Highlight
            from pypdf.generic import ArrayObject, FloatObject
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for PDF visual marking. "
                "Install with: pip install pdfplumber pypdf"
            ) from exc

        pdf_bytes = source_path.read_bytes()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        _pypdf_logger = logging.getLogger("pypdf")
        _prior_level = _pypdf_logger.level
        _pypdf_logger.setLevel(logging.ERROR)
        try:
            writer.append(reader)
        finally:
            _pypdf_logger.setLevel(_prior_level)
        hex_color = _parse_color(highlight_color)

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
            num_pages = len(plumber_doc.pages)

            for snippet in snippets:
                text = _PAGE_PREFIX_RX.sub("", snippet.text or "").strip()
                if not text:
                    continue

                # Use per-snippet color if provided, otherwise use default
                snippet_color = (
                    _parse_color(snippet.color) if snippet.color else hex_color
                )

                if (
                    snippet.page_number is not None
                    and 1 <= snippet.page_number <= num_pages
                ):
                    page_indices = [snippet.page_number - 1]
                else:
                    page_indices = list(range(num_pages))

                for page_idx in page_indices:
                    plumber_page = plumber_doc.pages[page_idx]
                    page_height = float(plumber_page.height)
                    words = plumber_page.extract_words(
                        keep_blank_chars=False,
                        use_text_flow=True,
                    )
                    rects = _find_rects(words, page_height, text)
                    if rects:
                        for x0, y0, x1, y1 in rects:
                            # QuadPoints per PDF spec: UL, UR, LL, LR
                            quad = ArrayObject(
                                [
                                    FloatObject(x0),
                                    FloatObject(y1),  # upper-left
                                    FloatObject(x1),
                                    FloatObject(y1),  # upper-right
                                    FloatObject(x0),
                                    FloatObject(y0),  # lower-left
                                    FloatObject(x1),
                                    FloatObject(y0),  # lower-right
                                ]
                            )
                            writer.add_annotation(
                                page_number=page_idx,
                                annotation=Highlight(
                                    rect=(x0, y0, x1, y1),
                                    quad_points=quad,
                                    highlight_color=snippet_color,
                                ),
                            )
                        break  # found on this page; stop scanning
                else:
                    # Exhausted all pages without a match — log so grounding debug can diagnose it.
                    logging.getLogger(__name__).debug(
                        "PDF text not found in %s: %r", source_path.name, text[:120]
                    )

        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Text-location helpers
# ---------------------------------------------------------------------------


def _find_rects(
    words: list[dict[str, Any]],
    page_height: float,
    text: str,
) -> list[tuple[float, float, float, float]]:
    """Return PDF-coordinate rects for all matches of *text* on the page."""
    rects = _match_word_sequence(words, page_height, text)
    if rects:
        return rects
    for fragment in _iter_fragments(text):
        rects.extend(_match_word_sequence(words, page_height, fragment))
    return rects


def _match_word_sequence(
    words: list[dict[str, Any]],
    page_height: float,
    text: str,
) -> list[tuple[float, float, float, float]]:
    """Sliding-window token search; returns merged bboxes in PDF coords."""
    target = _tokenize(text)
    if not target:
        return []

    # Flatten page words into (token, word_index) pairs.
    flat: list[tuple[str, int]] = []
    for wi, w in enumerate(words):
        for tok in _tokenize(w["text"]):
            flat.append((tok, wi))

    n = len(target)
    rects: list[tuple[float, float, float, float]] = []
    i = 0
    while i <= len(flat) - n:
        if [t for t, _ in flat[i : i + n]] == target:
            involved = sorted({wi for _, wi in flat[i : i + n]})
            ws = [words[wi] for wi in involved]
            x0 = min(w["x0"] for w in ws)
            x1 = max(w["x1"] for w in ws)
            # Convert pdfplumber top-left origin → PDF bottom-left origin.
            y0 = page_height - max(w["bottom"] for w in ws)
            y1 = page_height - min(w["top"] for w in ws)
            rects.append((x0, y0, x1, y1))
            i += n
        else:
            i += 1
    return rects


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split into non-empty tokens."""
    return _PUNCT_RX.sub(" ", text.lower()).split()


def _iter_fragments(text: str):
    """Yield non-trivial substrings of *text* for the fragment fallback."""
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= _MIN_FRAGMENT_LEN and stripped not in seen:
            seen.add(stripped)
            yield stripped
    for sent in re.split(r"(?<=[\.\!\?])\s+", text):
        stripped = sent.strip()
        if len(stripped) >= _MIN_FRAGMENT_LEN and stripped not in seen:
            seen.add(stripped)
            yield stripped
