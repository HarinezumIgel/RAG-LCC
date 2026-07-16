"""Visual markers for highlighting retrieved chunks in source documents.

Currently supports PDF (via pdfplumber + pypdf). Additional document types
can be added by subclassing ``VisualMarker`` and registering the
implementation in ``VisualMarkerFactory``.
"""

from VisualMarkers.VisualMarker import ChunkSnippet, VisualMarker
from VisualMarkers.VisualMarkerFactory import VisualMarkerFactory

__all__ = ["ChunkSnippet", "VisualMarker", "VisualMarkerFactory"]
