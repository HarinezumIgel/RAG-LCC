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
# searching the actual page text. The label may be numeric or roman ("iii").
_PAGE_PREFIX_RX = re.compile(r"^\s*Page\s+\S+\s*\n+", re.IGNORECASE)
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

            # Cache each touched page's extracted words + token index so that
            # extract_words() runs at most once per page regardless of how many
            # snippets (and fragments) target it. Dominant cost on large docs.
            page_cache: dict[
                int,
                tuple[
                    float,
                    list[dict[str, Any]],
                    list[tuple[str, int]],
                    dict[str, list[int]],
                ],
            ] = {}

            def _page_data(idx: int):
                data = page_cache.get(idx)
                if data is None:
                    plumber_page = plumber_doc.pages[idx]
                    page_height = float(plumber_page.height)
                    words = plumber_page.extract_words(
                        keep_blank_chars=False,
                        use_text_flow=True,
                    )
                    flat = _flatten(words)
                    data = (page_height, words, flat, _build_first_index(flat))
                    page_cache[idx] = data
                return data

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
                    page_height, words, flat, first_index = _page_data(page_idx)
                    rects = _find_rects(words, page_height, text, flat, first_index)
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
    flat: "list[tuple[str, int]] | None" = None,
    first_index: "dict[str, list[int]] | None" = None,
) -> list[tuple[float, float, float, float]]:
    """Return PDF-coordinate rects for all matches of *text* on the page.

    ``flat`` and ``first_index`` may be supplied by the caller (see the per-page
    cache in ``mark_to_bytes``) to avoid recomputing them for every snippet and
    every fragment on the same page.
    """
    if flat is None:
        flat = _flatten(words)
    if first_index is None:
        first_index = _build_first_index(flat)
    rects = _match_word_sequence(words, page_height, text, flat, first_index)
    if rects:
        return rects
    for fragment in _iter_fragments(text):
        rects.extend(
            _match_word_sequence(words, page_height, fragment, flat, first_index)
        )
    if rects:
        return rects
    # Last resort for tables/blocks with neither newlines nor sentence
    # punctuation (e.g. numbered connector legends): match short fixed-size
    # token windows so the source region is still marked even when the full
    # token order differs from the page's reading order.
    return _match_token_windows(words, page_height, text, flat, first_index)


def _flatten(words: list[dict[str, Any]]) -> "list[tuple[str, int]]":
    """Flatten page words into ``(token, word_index)`` pairs (one per token)."""
    flat: list[tuple[str, int]] = []
    for wi, w in enumerate(words):
        for tok in _tokenize(w["text"]):
            flat.append((tok, wi))
    return flat


def _build_first_index(flat: "list[tuple[str, int]]") -> "dict[str, list[int]]":
    """Map each token to the ascending flat-positions where it occurs.

    Used to jump straight to candidate match starts instead of scanning every
    position — a large win on long pages with many short fragment/window probes.
    """
    idx: dict[str, list[int]] = {}
    for i, (tok, _) in enumerate(flat):
        idx.setdefault(tok, []).append(i)
    return idx


def _match_token_windows(
    words: list[dict[str, Any]],
    page_height: float,
    text: str,
    flat: "list[tuple[str, int]]",
    first_index: "dict[str, list[int]]",
    window: int = 4,
) -> list[tuple[float, float, float, float]]:
    """Match consecutive ``window``-token slices of *text*, deduping rects."""
    tokens = _tokenize(text)
    if len(tokens) <= window:
        return []
    seen: set[tuple[float, float, float, float]] = set()
    out: list[tuple[float, float, float, float]] = []
    for start in range(0, len(tokens) - window + 1, window):
        fragment = " ".join(tokens[start : start + window])
        for rect in _match_word_sequence(
            words, page_height, fragment, flat, first_index
        ):
            if rect not in seen:
                seen.add(rect)
                out.append(rect)
    return out


def _match_word_sequence(
    words: list[dict[str, Any]],
    page_height: float,
    text: str,
    flat: "list[tuple[str, int]] | None" = None,
    first_index: "dict[str, list[int]] | None" = None,
) -> list[tuple[float, float, float, float]]:
    """Token-sequence search; returns merged bboxes in PDF coords.

    Candidate start positions are looked up via ``first_index`` (positions of the
    first target token) so only plausible windows are compared.
    """
    target = _tokenize(text)
    if not target:
        return []
    if flat is None:
        flat = _flatten(words)
    if first_index is None:
        first_index = _build_first_index(flat)

    n = len(target)
    flat_len = len(flat)
    rects: list[tuple[float, float, float, float]] = []
    last_end = -1
    for start in first_index.get(target[0], ()):
        if start < last_end:  # keep matches non-overlapping (as before)
            continue
        if start + n > flat_len:  # positions are ascending → no later fit either
            break
        if [t for t, _ in flat[start : start + n]] == target:
            involved = sorted({wi for _, wi in flat[start : start + n]})
            ws = [words[wi] for wi in involved]
            x0 = min(w["x0"] for w in ws)
            x1 = max(w["x1"] for w in ws)
            # Convert pdfplumber top-left origin → PDF bottom-left origin.
            y0 = page_height - max(w["bottom"] for w in ws)
            y1 = page_height - min(w["top"] for w in ws)
            rects.append((x0, y0, x1, y1))
            last_end = start + n
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
