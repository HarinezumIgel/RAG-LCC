"""Plain-text (.txt) and Markdown (.md) chunk highlighter.

Strategy
--------
The file is read as UTF-8 text and split on newlines.  For each retrieved
snippet every line is tested with the same bidirectional token-containment
check used by ``DocxVisualMarker``.  Matching lines are wrapped with
``<mark style="background: COLOR">…</mark>`` (inline HTML; widely supported by
Markdown renderers such as GitHub, VS Code Preview, and most static-site generators).

Both ``.md`` and ``.txt`` files produce identical ``<mark>``-annotated output
with per-snippet colors (yellow for chunks, orange for grounded sentences).
``.txt`` files are saved as ``.txt.md`` so they open in a Markdown viewer where
the color highlights are rendered.

For Markdown files, common structural prefixes (headings, lists, blockquotes)
are preserved outside the ``<mark>`` wrapper so the document structure remains
intact when rendered.

The modified content is returned as UTF-8-encoded bytes; the source file is
never touched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from VisualMarkers.VisualMarker import ChunkSnippet, VisualMarker

# Strip known chunker prefixes before token comparison.
_PAGE_PREFIX_RX = re.compile(r"^\s*Page\s+\d+\s*\n+", re.IGNORECASE)
_SLIDE_PREFIX_RX = re.compile(r"^\s*Slide\s+\d+(?::[^\n]*)?\n+", re.IGNORECASE)

_MIN_PARA_TOKENS = 4
_MIN_FRAGMENT_LEN = 12
_PUNCT_RX = re.compile(r"[^\w\s]")

# Markdown structural prefix: headings, list items, blockquotes, code-fences.
# Only the first (outermost) structural token is stripped so that e.g.
# "## 1. Introduction" is treated as heading prefix "## " + body "1. Introduction"
# rather than stripping both "## " and "1. " as nested prefixes.
_MD_STRUCT_PREFIX_RE = re.compile(r"^(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s*)")


class PlainTextVisualMarker(VisualMarker):
    """Highlight retrieved chunks in plain-text or Markdown files."""

    def mark_to_bytes(
        self,
        source_path: Path,
        snippets: Sequence[ChunkSnippet],
        *,
        highlight_color: str = "yellow",
    ) -> bytes:
        text = source_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        # Collect indices of lines that match any snippet, along with the snippet's color.
        # Maps line_index -> color (use per-snippet color, or default highlight_color).
        to_highlight: dict[int, str] = {}
        for snippet in snippets:
            raw = (snippet.text or "").strip()
            raw = _PAGE_PREFIX_RX.sub("", raw)
            raw = _SLIDE_PREFIX_RX.sub("", raw).strip()
            if not raw:
                continue
            snippet_tokens = _tokenize(raw)
            if not snippet_tokens:
                continue
            snippet_color = snippet.color if snippet.color else highlight_color
            for i, line in enumerate(lines):
                content = line.rstrip("\r\n")
                if not content.strip():
                    continue
                if _para_matches(_tokenize(content), snippet_tokens, raw):
                    # If line already marked, keep first color (yellow chunks before orange grounding)
                    if i not in to_highlight:
                        to_highlight[i] = snippet_color

        if not to_highlight:
            return text.encode("utf-8")

        # Both .txt and .md files use <mark> tags without inline styles so that
        # DOMPurify (OpenWebUI) does not strip the visual marker.
        # Grounded/orange sentences are wrapped with <mark><strong> to distinguish
        # them from plain chunk highlights (<mark> only).
        # .txt files are saved as .txt.md by the viewer so the tags render properly.
        result: list[str] = []
        for i, line in enumerate(lines):
            if i not in to_highlight:
                result.append(line)
                continue
            eol = _eol(line)
            content = line.rstrip("\r\n")
            color = to_highlight[i]
            result.append(_wrap_md_line(content, color) + eol)

        return "".join(result).encode("utf-8")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _wrap_md_line(line: str, color: str = "yellow") -> str:
    """Wrap a line's text in ``<mark>…</mark>`` (or ``<mark><strong>…</strong></mark>``
    for non-yellow colors), preserving any leading structural prefix.

    Inline ``style=`` attributes are intentionally omitted so that DOMPurify
    (used by OpenWebUI) does not strip the highlight.  Grounded sentences
    (non-yellow) get an additional ``<strong>`` wrapper so they remain visually
    distinct from plain chunk marks even without colour support.

    Used for both .md and .txt files (txt files are saved as .txt.md to enable rendering).
    """
    grounded = color != "yellow"

    def _wrap(body: str) -> str:
        inner = f"<strong>{body}</strong>" if grounded else body
        return f"<mark>{inner}</mark>"

    m = _MD_STRUCT_PREFIX_RE.match(line)
    if m:
        prefix = m.group(0)
        body = line[m.end() :]
        if body.strip():
            return f"{prefix}{_wrap(body)}"
        return line  # prefix only (e.g. blank list item) — nothing to mark
    return _wrap(line)


def _eol(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


# ---------------------------------------------------------------------------
# Matching helpers (shared pattern with DocxVisualMarker / PptxVisualMarker)
# ---------------------------------------------------------------------------


def _para_matches(
    para_tokens: list[str],
    snippet_tokens: list[str],
    snippet_text: str,
) -> bool:
    if len(para_tokens) >= _MIN_PARA_TOKENS and _contains_sequence(
        snippet_tokens, para_tokens
    ):
        return True
    for frag in _iter_fragments(snippet_text):
        frag_tokens = _tokenize(frag)
        if frag_tokens and _contains_sequence(para_tokens, frag_tokens):
            return True
    return False


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RX.sub(" ", text.lower()).split()


def _iter_fragments(text: str):
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
