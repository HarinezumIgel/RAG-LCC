# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for AUTO_CHUNK per-file-type chunker routing in DocumentIngestionStrategy."""

import sys
import os
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.DocumentIngestionStrategy import DocumentIngestionStrategy
from Strategies.Chunkers.RecursiveChunker import RecursiveChunker
from Strategies.Chunkers.HeadingChunker import HeadingChunker
from Strategies.Chunkers.SemanticChunker import SemanticChunker
from Strategies.Chunkers.SentenceWindowChunker import SentenceWindowChunker
from Strategies.Chunkers.SlideChunker import SlideChunker
from Strategies.Chunkers.SlidingWindowChunker import SlidingWindowChunker


def _mock_cfg():
    cfg = MagicMock()
    cfg.get_int.return_value = 256
    cfg.get_list.return_value = ["\n", " ", "."]
    cfg.get_str.return_value = "SEMANTIC"
    cfg.get_float.return_value = 0.0
    return cfg


def _mock_helpers():
    h = MagicMock()
    h.get_chunker_config_slot.return_value = "_CHUNKERS.SEMANTIC"
    return h


def _mock_file_utils():
    fu = MagicMock()
    fu.count_words.side_effect = lambda t: len(t.split())
    return fu


def _patched_make_chunker(name):
    """Instantiate chunker with mocked dependencies."""
    kwargs = dict(
        cfg=_mock_cfg(),
        helpers=_mock_helpers(),
        file_utils=_mock_file_utils(),
        chunker_name=name,
    )
    if name == "SEMANTIC":
        return SemanticChunker(embedder=MagicMock(), **kwargs)
    if name == "SENTENCE_WINDOW":
        return SentenceWindowChunker(**kwargs)
    if name == "SLIDING_WINDOW":
        return SlidingWindowChunker(**kwargs)
    if name == "HEADING":
        return HeadingChunker(**kwargs)
    if name == "SLIDE":
        return SlideChunker(**kwargs)
    return RecursiveChunker(**kwargs)


class TestMakeChunker:
    """Test the _make_chunker factory returns correct types."""

    def test_semantic(self):
        c = _patched_make_chunker("SEMANTIC")
        assert isinstance(c, SemanticChunker)

    def test_sentence_window(self):
        c = _patched_make_chunker("SENTENCE_WINDOW")
        assert isinstance(c, SentenceWindowChunker)

    def test_sliding_window(self):
        c = _patched_make_chunker("SLIDING_WINDOW")
        assert isinstance(c, SlidingWindowChunker)

    def test_heading(self):
        c = _patched_make_chunker("HEADING")
        assert isinstance(c, HeadingChunker)

    def test_slide(self):
        c = _patched_make_chunker("SLIDE")
        assert isinstance(c, SlideChunker)

    def test_recursive(self):
        c = _patched_make_chunker("RECURSIVE")
        assert isinstance(c, RecursiveChunker)

    def test_unknown_falls_back_to_fixed(self):
        c = _patched_make_chunker("NONEXISTENT")
        assert isinstance(c, RecursiveChunker)


class TestResolveChunkerForFile:
    """Test _resolve_chunker_for_file picks the right chunker per file type."""

    def _make_strategy_stub(self, chunk_map: dict[str, str]):
        """Build a minimal DocumentIngestionStrategy-like object with chunk-map wiring."""
        obj = object.__new__(DocumentIngestionStrategy)
        obj._chunk_map = chunk_map
        obj.chunker = MagicMock()  # dummy initial
        return obj

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_pdf_routes_to_semantic(self, _mock):
        s = self._make_strategy_stub({"pdf": "SEMANTIC", "DEFAULT": "RECURSIVE"})
        s.fileType = "pdf"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SemanticChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_csv_routes_to_fixed(self, _mock):
        s = self._make_strategy_stub({"csv": "RECURSIVE", "DEFAULT": "SEMANTIC"})
        s.fileType = "csv"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, RecursiveChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_pptx_routes_to_slide(self, _mock):
        s = self._make_strategy_stub({"pptx": "SLIDE", "DEFAULT": "SEMANTIC"})
        s.fileType = "pptx"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SlideChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_docx_routes_to_heading(self, _mock):
        s = self._make_strategy_stub({"docx": "HEADING", "DEFAULT": "SEMANTIC"})
        s.fileType = "docx"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, HeadingChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_unknown_type_uses_default(self, _mock):
        s = self._make_strategy_stub({"pdf": "SEMANTIC", "DEFAULT": "SLIDING_WINDOW"})
        s.fileType = "xyz"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SlidingWindowChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_none_filetype_uses_default(self, _mock):
        s = self._make_strategy_stub({"DEFAULT": "SENTENCE_WINDOW"})
        s.fileType = None
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SentenceWindowChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_case_insensitive_filetype(self, _mock):
        s = self._make_strategy_stub({"pdf": "SEMANTIC", "DEFAULT": "RECURSIVE"})
        s.fileType = "PDF"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SemanticChunker)

    @patch.object(
        DocumentIngestionStrategy, "_make_chunker", side_effect=_patched_make_chunker
    )
    def test_default_fallback_when_no_default_key(self, _mock):
        """If chunk map has no DEFAULT key, falls back to SEMANTIC."""
        s = self._make_strategy_stub({"pdf": "RECURSIVE"})
        s.fileType = "xyz"
        s._resolve_chunker_for_file()
        assert isinstance(s.chunker, SemanticChunker)
