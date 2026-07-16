# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for the HeadingChunker."""

import sys
import os
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.Chunkers.HeadingChunker import HeadingChunker, _docx_heading_level


def _make_chunker(max_chunk_size: int = 256):
    """Create a HeadingChunker with mocked config."""
    cfg = MagicMock()
    helpers = MagicMock()
    file_utils = MagicMock()

    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.HEADING"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return max_chunk_size
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n", " ", "."]

    file_utils.count_words.side_effect = lambda t: len(t.split())

    return HeadingChunker(cfg=cfg, helpers=helpers, file_utils=file_utils)


META_MD = {"FileName": "readme.md", "FilePath": "/readme.md", "FileType": "md"}
META_TXT = {"FileName": "notes.txt", "FilePath": "/notes.txt", "FileType": "txt"}


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


class TestMarkdownParsing:
    def test_empty_content(self):
        chunker = _make_chunker()
        assert chunker.chunk("", META_MD) == ([], None)

    def test_no_headings(self):
        chunker = _make_chunker()
        docs, _ = chunker.chunk("Just some plain text.\nAnother line.", META_MD)
        assert len(docs) == 1
        assert "Just some plain text." in docs[0].page_content

    def test_single_heading_with_body(self):
        content = "# Introduction\n\nThis is the intro."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        assert len(docs) == 1
        assert "Introduction" in docs[0].page_content
        assert "This is the intro." in docs[0].page_content

    def test_multiple_h1_sections(self):
        content = "# Section One\n\nBody one.\n\n# Section Two\n\nBody two."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        assert len(docs) == 2
        assert "Section One" in docs[0].page_content
        assert "Body one." in docs[0].page_content
        assert "Section Two" in docs[1].page_content
        assert "Body two." in docs[1].page_content

    def test_nested_headings_breadcrumb(self):
        content = "# Chapter\n\n## Section\n\nBody under section."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        # Chapter has no body → no chunk for it
        # Section body includes breadcrumb
        section_doc = [d for d in docs if "Body under section." in d.page_content][0]
        assert "Chapter > Section" in section_doc.page_content

    def test_heading_trail_resets_on_same_level(self):
        content = "# Ch1\n\n## A\n\nBody A.\n\n## B\n\nBody B."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        doc_b = [d for d in docs if "Body B." in d.page_content][0]
        assert "Ch1 > B" in doc_b.page_content
        assert "A" not in doc_b.page_content.split("Body B.")[0].split("> B")[0]

    def test_deeper_level_cleared(self):
        content = (
            "# Top\n\n## Sub\n\n### Deep\n\nDeep body.\n\n## Another\n\nAnother body."
        )
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        doc_another = [d for d in docs if "Another body." in d.page_content][0]
        # Breadcrumb should be "Top > Another", NOT "Top > Another > Deep"
        assert "Top > Another" in doc_another.page_content
        assert "Deep" not in doc_another.page_content


class TestMarkdownEdgeCases:
    def test_heading_without_body_skipped(self):
        content = "# Empty Section\n\n# Next\n\nHas body."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        # Only the section with body should produce a chunk
        assert len(docs) == 1
        assert "Has body." in docs[0].page_content

    def test_body_before_first_heading(self):
        content = "Preamble text.\n\n# Heading\n\nBody."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        assert len(docs) == 2
        assert "Preamble text." in docs[0].page_content
        assert "Body." in docs[1].page_content

    def test_h6_heading(self):
        content = "###### Deep Heading\n\nDeep body."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        assert len(docs) == 1
        assert "Deep Heading" in docs[0].page_content


# ---------------------------------------------------------------------------
# Oversized section splitting
# ---------------------------------------------------------------------------


class TestOversizedSections:
    def test_oversized_section_split(self):
        body = " ".join(f"word{i}" for i in range(100))
        content = f"# Big Section\n\n{body}"
        chunker = _make_chunker(max_chunk_size=30)
        docs, _ = chunker.chunk(content, META_MD)
        # Should produce multiple chunks, each with heading breadcrumb
        assert len(docs) > 1
        for doc in docs:
            assert "Big Section" in doc.page_content


# ---------------------------------------------------------------------------
# Flat fallback (non-MD, non-DOCX)
# ---------------------------------------------------------------------------


class TestFlatFallback:
    def test_txt_uses_flat_fallback(self):
        chunker = _make_chunker()
        docs, _ = chunker.chunk("Just some text content.", META_TXT)
        assert len(docs) == 1
        assert docs[0].page_content == "Just some text content."

    def test_empty_flat(self):
        chunker = _make_chunker()
        assert chunker.chunk("", META_TXT) == ([], None)


# ---------------------------------------------------------------------------
# Metadata and IDs
# ---------------------------------------------------------------------------


class TestMetadataAndIds:
    def test_mychunk_index(self):
        content = "# A\n\nBody A.\n\n# B\n\nBody B."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        indices = [d.metadata["MyChunk"] for d in docs]
        assert indices == list(range(len(docs)))

    def test_unique_ids(self):
        content = "# A\n\nBody A.\n\n# B\n\nBody B.\n\n# C\n\nBody C."
        chunker = _make_chunker()
        docs, _ = chunker.chunk(content, META_MD)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        chunker = _make_chunker()
        docs, _ = chunker.chunk("# H\n\nBody.", META_MD)
        assert docs[0].metadata["FileName"] == "readme.md"
        assert docs[0].metadata["FileType"] == "md"


# ---------------------------------------------------------------------------
# _docx_heading_level helper
# ---------------------------------------------------------------------------


class TestDocxHeadingLevel:
    def test_heading_1(self):
        assert _docx_heading_level("Heading 1") == 1

    def test_heading_9(self):
        assert _docx_heading_level("Heading 9") == 9

    def test_normal_style(self):
        assert _docx_heading_level("Normal") == 0

    def test_body_text(self):
        assert _docx_heading_level("Body Text") == 0

    def test_empty(self):
        assert _docx_heading_level("") == 0


# ---------------------------------------------------------------------------
# chunk_size property
# ---------------------------------------------------------------------------


class TestChunkSizeProperty:
    def test_chunk_size(self):
        chunker = _make_chunker(max_chunk_size=512)
        assert chunker.chunk_size == 512
