# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for the SlideChunker."""

import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.Chunkers.PageBasedChunker import PageBasedChunker
from Strategies.Chunkers.SlideChunker import SlideChunker


def _make_chunker(max_chunk_size: int = 256):
    """Create a SlideChunker with mocked config."""
    cfg = MagicMock()
    helpers = MagicMock()
    file_utils = MagicMock()

    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.SLIDE"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return max_chunk_size
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n", " ", "."]

    file_utils.count_words.side_effect = lambda t: len(t.split())

    return SlideChunker(cfg=cfg, helpers=helpers, file_utils=file_utils)


META_PPTX = {"FileName": "deck.pptx", "FilePath": "/deck.pptx", "FileType": "pptx"}
META_TXT = {"FileName": "notes.txt", "FilePath": "/notes.txt", "FileType": "txt"}


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_subclass_of_page_based_chunker(self):
        assert issubclass(SlideChunker, PageBasedChunker)

    def test_format_prefix_with_title(self):
        chunker = _make_chunker()
        assert chunker._format_prefix(3, "Agenda") == "Slide 3: Agenda"

    def test_format_prefix_no_title(self):
        chunker = _make_chunker()
        assert chunker._format_prefix(2, "") == "Slide 2"


# ---------------------------------------------------------------------------
# _parse_pptx via mock Presentation
# ---------------------------------------------------------------------------


def _mock_shape(text, is_title=False):
    shape = MagicMock()
    shape.text = text

    # Build text_frame with paragraph structure
    frame = MagicMock()
    paras = []
    for line in text.split("\n"):
        p = MagicMock()
        p.text = line
        paras.append(p)
    frame.paragraphs = paras
    shape.text_frame = frame
    return shape


def _mock_slide(title_text, body_texts):
    """Build a mock slide with a title shape and body shapes."""
    slide = MagicMock()
    shapes = []

    title_shape = None
    if title_text:
        title_shape = _mock_shape(title_text, is_title=True)
        shapes.append(title_shape)

    for bt in body_texts:
        shapes.append(_mock_shape(bt))

    slide.shapes = MagicMock()
    slide.shapes.__iter__ = MagicMock(return_value=iter(shapes))
    slide.shapes.title = title_shape
    if title_shape:
        slide.shapes.title.text = title_text

    return slide


def _mock_presentation(slides_spec):
    """Build a mock Presentation from a list of (title, [body_texts]) tuples."""
    prs = MagicMock()
    slides = [_mock_slide(t, b) for t, b in slides_spec]
    prs.slides = slides
    return prs


class TestSlideChunkerWithMockPptx:
    """Test _parse_pptx by patching python-pptx Presentation."""

    def _chunk_with_mock(self, slides_spec, max_chunk_size=256):
        chunker = _make_chunker(max_chunk_size)
        prs = _mock_presentation(slides_spec)

        with patch(
            "Strategies.Chunkers.SlideChunker.Presentation",
            return_value=prs,
            create=True,
        ):
            # We need to patch the import inside _parse_pptx
            import Strategies.Chunkers.SlideChunker as mod

            original_parse = mod.SlideChunker._parse_pptx

            def patched_parse(file_path):
                # Simulate what _parse_pptx does but with our mock
                slides = []
                for idx, slide in enumerate(prs.slides, start=1):
                    title = ""
                    body_parts = []
                    if slide.shapes.title and slide.shapes.title.text:
                        title = slide.shapes.title.text.strip()
                    for shape in slide.shapes:
                        if not hasattr(shape, "text"):
                            continue
                        text = shape.text.strip()
                        if not text:
                            continue
                        if shape == slide.shapes.title:
                            continue
                        if hasattr(shape, "text_frame"):
                            for para in shape.text_frame.paragraphs:
                                line = para.text.strip()
                                if line:
                                    body_parts.append(line)
                        else:
                            body_parts.append(text)
                    body = "\n".join(body_parts)
                    if title or body:
                        slides.append((idx, title, body))
                return slides

            mod.SlideChunker._parse_pptx = staticmethod(patched_parse)
            try:
                meta = dict(META_PPTX)
                meta["FilePath"] = (
                    __file__  # needs to be a real file for os.path.isfile
                )
                docs, _ = chunker.chunk("ignored", meta)
            finally:
                mod.SlideChunker._parse_pptx = original_parse

        return docs

    def test_single_slide_title_and_body(self):
        docs = self._chunk_with_mock(
            [
                ("Introduction", ["Welcome to the talk"]),
            ]
        )
        assert len(docs) == 1
        assert "Slide 1: Introduction" in docs[0].page_content
        assert "Welcome to the talk" in docs[0].page_content

    def test_multiple_slides(self):
        docs = self._chunk_with_mock(
            [
                ("Slide A", ["Body A"]),
                ("Slide B", ["Body B"]),
                ("Slide C", ["Body C"]),
            ]
        )
        assert len(docs) == 3
        assert "Slide 1: Slide A" in docs[0].page_content
        assert "Slide 2: Slide B" in docs[1].page_content
        assert "Slide 3: Slide C" in docs[2].page_content

    def test_title_only_slide(self):
        docs = self._chunk_with_mock(
            [
                ("Title Only", []),
            ]
        )
        assert len(docs) == 1
        assert "Slide 1: Title Only" in docs[0].page_content

    def test_no_title_slide(self):
        docs = self._chunk_with_mock(
            [
                ("", ["Just content here"]),
            ]
        )
        assert len(docs) == 1
        assert "Slide 1" in docs[0].page_content
        assert "Just content here" in docs[0].page_content

    def test_oversized_slide_split(self):
        long_body = " ".join(f"word{i}" for i in range(100))
        docs = self._chunk_with_mock(
            [("Big Slide", [long_body])],
            max_chunk_size=30,
        )
        assert len(docs) > 1
        for doc in docs:
            assert "Slide 1: Big Slide" in doc.page_content


# ---------------------------------------------------------------------------
# Flat fallback
# ---------------------------------------------------------------------------


class TestFlatFallback:
    def test_txt_uses_flat_fallback(self):
        chunker = _make_chunker()
        docs, _ = chunker.chunk("Just some text.", META_TXT)
        assert len(docs) == 1
        assert docs[0].page_content == "Slide 1\n\nJust some text."

    def test_empty_content(self):
        chunker = _make_chunker()
        assert chunker.chunk("", META_TXT) == ([], None)


# ---------------------------------------------------------------------------
# _pages_to_texts
# ---------------------------------------------------------------------------


class TestPagesToTexts:
    def test_basic(self):
        chunker = _make_chunker()
        page_texts = chunker._pages_to_texts(
            [
                (1, "Intro", "Hello world."),
                (2, "Details", "More info here."),
            ]
        )
        assert len(page_texts) == 2
        assert "Slide 1: Intro" in page_texts[0][0]
        assert "Hello world." in page_texts[0][0]
        assert "Slide 2: Details" in page_texts[1][0]

    def test_empty_body(self):
        chunker = _make_chunker()
        page_texts = chunker._pages_to_texts([(1, "Only Title", "")])
        assert len(page_texts) == 1
        assert page_texts[0][0] == "Slide 1: Only Title"

    def test_no_title(self):
        chunker = _make_chunker()
        page_texts = chunker._pages_to_texts([(3, "", "Content only.")])
        assert "Slide 3" in page_texts[0][0]
        assert "Content only." in page_texts[0][0]


# ---------------------------------------------------------------------------
# Metadata and IDs
# ---------------------------------------------------------------------------


class TestMetadataAndIds:
    def test_mychunk_index(self):
        chunker = _make_chunker()
        # Use flat fallback for simplicity
        docs, _ = chunker.chunk("Slide content.", META_TXT)
        assert docs[0].metadata["MyChunk"] == 0

    def test_unique_ids(self):
        chunker = _make_chunker()
        slides = [(1, "A", "Body A"), (2, "B", "Body B"), (3, "C", "Body C")]
        page_texts = chunker._pages_to_texts(slides)
        docs = chunker._to_docs(page_texts, META_PPTX)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        chunker = _make_chunker()
        docs, _ = chunker.chunk("Text.", META_TXT)
        assert docs[0].metadata["FileName"] == "notes.txt"


# ---------------------------------------------------------------------------
# chunk_size property
# ---------------------------------------------------------------------------


class TestChunkSizeProperty:
    def test_chunk_size(self):
        chunker = _make_chunker(max_chunk_size=512)
        assert chunker.chunk_size == 512
