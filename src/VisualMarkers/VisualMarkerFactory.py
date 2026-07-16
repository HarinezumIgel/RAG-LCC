"""Dispatch source files to the appropriate ``VisualMarker``."""

from __future__ import annotations

from pathlib import Path

from VisualMarkers.DocxVisualMarker import DocxVisualMarker
from VisualMarkers.PdfVisualMarker import PdfVisualMarker
from VisualMarkers.PlainTextVisualMarker import PlainTextVisualMarker
from VisualMarkers.PptxVisualMarker import PptxVisualMarker
from VisualMarkers.VisualMarker import VisualMarker


class VisualMarkerFactory:
    """Returns a marker for the given source file, or ``None``."""

    @staticmethod
    def for_path(source_path: Path | str) -> VisualMarker | None:
        suffix = Path(source_path).suffix.lower()
        if suffix == ".pdf":
            return PdfVisualMarker()
        if suffix == ".docx":
            return DocxVisualMarker()
        if suffix in (".pptx", ".ppt"):
            return PptxVisualMarker()
        if suffix in (".md", ".txt"):
            return PlainTextVisualMarker()
        return None
