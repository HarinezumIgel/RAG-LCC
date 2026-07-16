"""Abstract base class for visual chunk markers.

A ``VisualMarker`` takes a source document and a list of chunk snippets
(with optional page numbers) and returns the document with the snippets
highlighted, as in-memory bytes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ChunkSnippet:
    """A retrieved chunk to be highlighted in its source document.

    Attributes
    ----------
    text:
        The chunk text as stored in the vector DB. May contain a leading
        page-prefix (e.g. ``"Page 3\\n\\n..."``); marker implementations
        should strip such prefixes before searching.
    page_number:
        1-based page number when known (PDFs, slides). ``None`` for
        formats without an intrinsic page concept.
    color:
        CSS color name or hex string for this snippet's highlight.
        ``None`` uses the default color from mark_to_bytes().
    """

    text: str
    page_number: int | None = None
    color: str | None = None


class VisualMarker:
    """Abstract base for document-specific chunk highlighters."""

    def mark_to_bytes(
        self,
        source_path: Path,
        snippets: Sequence[ChunkSnippet],
        *,
        highlight_color: str = "yellow",
    ) -> bytes:
        """Return *source_path* with *snippets* highlighted, as bytes.

        Implementations must not modify the original file on disk.
        ``highlight_color`` is a CSS-style name (e.g. ``"yellow"``) or
        hex string (e.g. ``"#FFFF00"``).
        """
        raise NotImplementedError
