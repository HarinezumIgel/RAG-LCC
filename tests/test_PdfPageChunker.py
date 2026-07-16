# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for PdfPageChunker."""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from langchain_core.documents.base import Document as langchainDoc

from Strategies.Chunkers.PageBasedChunker import PageBasedChunker
from Strategies.Chunkers.PdfPageChunker import PdfPageChunker

# Real test PDF shipped with the repository
_HEDGEHOGS_PDF = os.path.join(ROOT, "TestDocs", "Hedgehogs.pdf")


# ---------------------------------------------------------------------------
# Stubs & factory
# ---------------------------------------------------------------------------


def _make_chunker(max_chunk_size: int = 100) -> PdfPageChunker:
    cfg = MagicMock()
    helpers = MagicMock()
    file_utils = MagicMock()

    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.PDF_PAGE"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return max_chunk_size
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n\n", "\n", " "]
    file_utils.count_words.side_effect = lambda t: len(t.split())

    return PdfPageChunker(cfg=cfg, helpers=helpers, file_utils=file_utils)


PDF_META = {
    "FileName": "doc.pdf",
    "FilePath": "/docs/doc.pdf",
    "FileType": "pdf",
    "FileHash": "def456",
}
TXT_META = {
    "FileName": "doc.txt",
    "FilePath": "/docs/doc.txt",
    "FileType": "txt",
    "FileHash": "jkl000",
}


def _meta_with_path(path: str) -> dict:
    return {**PDF_META, "FilePath": path}


# ---------------------------------------------------------------------------
# Inheritance & interface
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_subclass_of_page_based_chunker(self):
        assert issubclass(PdfPageChunker, PageBasedChunker)

    def test_chunk_size_property(self):
        assert _make_chunker(max_chunk_size=400).chunk_size == 400

    def test_embeddings_always_none(self):
        c = _make_chunker()
        meta = {**PDF_META, "FilePath": "/nonexistent.pdf"}
        _, emb = c.chunk("fallback content", meta)
        assert emb is None

    def test_returns_langchain_docs(self):
        c = _make_chunker()
        meta = {**PDF_META, "FilePath": "/nonexistent.pdf"}
        docs, _ = c.chunk("some text", meta)
        assert all(isinstance(d, langchainDoc) for d in docs)


# ---------------------------------------------------------------------------
# Prefix format
# ---------------------------------------------------------------------------


class TestPrefixFormat:
    def test_format_prefix_uses_page_number_only(self):
        c = _make_chunker()
        assert c._format_prefix(7, "Some Title") == "Page 7"

    def test_format_prefix_ignores_title(self):
        c = _make_chunker()
        assert c._format_prefix(1, "") == "Page 1"
        assert c._format_prefix(99, "Long Title That Should Be Ignored") == "Page 99"


# ---------------------------------------------------------------------------
# Flat fallback — wrong file type
# ---------------------------------------------------------------------------


class TestFlatFallbackWrongType:
    def test_wrong_type_returns_single_chunk(self):
        c = _make_chunker()
        docs, _ = c.chunk("hello world", TXT_META)
        assert len(docs) == 1

    def test_wrong_type_content_in_chunk(self):
        c = _make_chunker()
        docs, _ = c.chunk("hello world", TXT_META)
        assert "hello world" in docs[0].page_content

    def test_wrong_type_page_number_is_one(self):
        c = _make_chunker()
        docs, _ = c.chunk("content", TXT_META)
        assert docs[0].metadata["PageNumber"] == 1


# ---------------------------------------------------------------------------
# Flat fallback — inaccessible / non-existent PDF
# ---------------------------------------------------------------------------


class TestFlatFallbackMissingFile:
    def test_nonexistent_pdf_returns_single_chunk(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("fallback content", meta)
        assert len(docs) == 1

    def test_nonexistent_pdf_content_preserved(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("fallback content", meta)
        assert "fallback content" in docs[0].page_content

    def test_nonexistent_pdf_page_number_is_one(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("some text", meta)
        assert docs[0].metadata["PageNumber"] == 1

    def test_empty_content_returns_empty(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("   ", meta)
        assert docs == []


# ---------------------------------------------------------------------------
# Metadata integrity (flat fallback)
# ---------------------------------------------------------------------------


class TestMetadataFlatFallback:
    def test_mychunk_zero_for_single_chunk(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("some content here", meta)
        assert docs[0].metadata["MyChunk"] == 0

    def test_original_fields_preserved(self):
        c = _make_chunker()
        meta = _meta_with_path("/nonexistent/doc.pdf")
        docs, _ = c.chunk("text", meta)
        assert docs[0].metadata["FileName"] == "doc.pdf"
        assert docs[0].metadata["FileHash"] == "def456"

    def test_original_dict_not_mutated(self):
        c = _make_chunker()
        meta = dict(PDF_META)
        meta["FilePath"] = "/nonexistent/doc.pdf"
        c.chunk("text", meta)
        assert "MyChunk" not in meta
        assert "PageNumber" not in meta


# ---------------------------------------------------------------------------
# PDF structured parsing (via _parse_pdf mock)
# ---------------------------------------------------------------------------


def _make_mock_pdf_pages(page_texts: list[str]) -> list[MagicMock]:
    pages = []
    for text in page_texts:
        p = MagicMock()
        p.extract_text.return_value = text
        pages.append(p)
    return pages


class TestPdfStructuredParsing:
    """Tests that use patch.object on _parse_pdf to avoid real I/O."""

    def test_two_pages_produce_two_chunks(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker(max_chunk_size=200)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker,
            "_parse_pdf",
            return_value=[
                (1, "", "First page content"),
                (2, "", "Second page content"),
            ],
        ):
            docs, _ = c.chunk("", meta)

        assert len(docs) == 2

    def test_page_numbers_in_metadata(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker(max_chunk_size=200)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker,
            "_parse_pdf",
            return_value=[
                (1, "", "Page one"),
                (2, "", "Page two"),
                (3, "", "Page three"),
            ],
        ):
            docs, _ = c.chunk("", meta)

        assert [d.metadata["PageNumber"] for d in docs] == [1, 2, 3]

    def test_chunk_text_starts_with_page_prefix(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker(max_chunk_size=200)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker,
            "_parse_pdf",
            return_value=[(5, "", "Some body text")],
        ):
            docs, _ = c.chunk("", meta)

        assert docs[0].page_content.startswith("Page 5")

    def test_mychunk_index_sequential_across_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker(max_chunk_size=200)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker,
            "_parse_pdf",
            return_value=[(1, "", "body 1"), (2, "", "body 2"), (3, "", "body 3")],
        ):
            docs, _ = c.chunk("", meta)

        assert [d.metadata["MyChunk"] for d in docs] == list(range(len(docs)))

    def test_unique_ids_across_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker(max_chunk_size=200)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker,
            "_parse_pdf",
            return_value=[(i, "", f"body {i}") for i in range(1, 8)],
        ):
            docs, _ = c.chunk("", meta)

        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_oversized_page_splits_into_multiple_chunks(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        # body_budget = max(50, chunk_size - prefix_words); need body > 50 words
        long_body = " ".join(f"w{i}" for i in range(60))  # 60 words
        c = _make_chunker(max_chunk_size=8)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker, "_parse_pdf", return_value=[(1, "", long_body)]
        ):
            docs, _ = c.chunk("", meta)

        assert len(docs) > 1

    def test_oversized_sub_chunks_share_page_number(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        long_body = " ".join(f"w{i}" for i in range(60))
        c = _make_chunker(max_chunk_size=8)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker, "_parse_pdf", return_value=[(1, "", long_body)]
        ):
            docs, _ = c.chunk("", meta)

        for d in docs:
            assert d.metadata["PageNumber"] == 1

    def test_oversized_sub_chunks_carry_prefix(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        long_body = " ".join(f"w{i}" for i in range(60))
        c = _make_chunker(max_chunk_size=8)
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker, "_parse_pdf", return_value=[(1, "", long_body)]
        ):
            docs, _ = c.chunk("", meta)

        for d in docs:
            assert d.page_content.startswith("Page 1")

    def test_parse_pdf_exception_falls_back(self, tmp_path):
        """If _parse_pdf raises, chunk() must fall back to flat content."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker()
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker, "_parse_pdf", side_effect=Exception("bad pdf")
        ):
            docs, _ = c.chunk("fallback text here", meta)

        assert len(docs) == 1
        assert "fallback text here" in docs[0].page_content

    def test_parse_pdf_called_with_correct_path(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"placeholder")

        c = _make_chunker()
        meta = _meta_with_path(str(pdf_file))

        with patch.object(
            PdfPageChunker, "_parse_pdf", return_value=[(1, "", "text")]
        ) as mock_parse:
            c.chunk("", meta)

        mock_parse.assert_called_once_with(str(pdf_file))


# ---------------------------------------------------------------------------
# Integration test — real PDF from TestDocs/
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isfile(_HEDGEHOGS_PDF),
    reason="TestDocs/Hedgehogs.pdf not present",
)
class TestRealPdfIntegration:
    def test_hedgehogs_pdf_produces_chunks(self):
        c = _make_chunker(max_chunk_size=400)
        meta = {**PDF_META, "FilePath": _HEDGEHOGS_PDF}
        docs, _ = c.chunk("", meta)
        assert len(docs) > 0

    def test_hedgehogs_page_numbers_are_ints(self):
        c = _make_chunker(max_chunk_size=400)
        meta = {**PDF_META, "FilePath": _HEDGEHOGS_PDF}
        docs, _ = c.chunk("", meta)
        assert all(isinstance(d.metadata.get("PageNumber"), int) for d in docs)

    def test_hedgehogs_page_numbers_start_at_one(self):
        c = _make_chunker(max_chunk_size=400)
        meta = {**PDF_META, "FilePath": _HEDGEHOGS_PDF}
        docs, _ = c.chunk("", meta)
        page_nums = sorted({d.metadata["PageNumber"] for d in docs})
        assert page_nums[0] == 1

    def test_hedgehogs_all_chunks_have_page_prefix(self):
        c = _make_chunker(max_chunk_size=400)
        meta = {**PDF_META, "FilePath": _HEDGEHOGS_PDF}
        docs, _ = c.chunk("", meta)
        for d in docs:
            assert d.page_content.startswith("Page ")

    def test_hedgehogs_mychunk_sequential(self):
        c = _make_chunker(max_chunk_size=400)
        meta = {**PDF_META, "FilePath": _HEDGEHOGS_PDF}
        docs, _ = c.chunk("", meta)
        assert [d.metadata["MyChunk"] for d in docs] == list(range(len(docs)))
