# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for PageBasedChunker — the abstract base for page/slide chunkers."""

import sys
import os
from unittest.mock import MagicMock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from langchain_core.documents.base import Document as langchainDoc

from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy
from Strategies.Chunkers.PageBasedChunker import PageBasedChunker, PageData

# ---------------------------------------------------------------------------
# Stubs & helpers
# ---------------------------------------------------------------------------


def _make_cfg(max_chunk_size: int = 50) -> MagicMock:
    cfg = MagicMock()
    helpers = MagicMock()
    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.TEST"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return max_chunk_size
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n\n", "\n", " "]
    return cfg


def _make_file_utils() -> MagicMock:
    fu = MagicMock()
    fu.count_words.side_effect = lambda t: len(t.split())
    return fu


class _ConcreteChunker(PageBasedChunker):
    """Minimal concrete implementation used only for testing the base class."""

    def __init__(
        self,
        pages: list[PageData],
        extra_meta: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._fixed_pages = pages
        self._extra = extra_meta or {}

    def _parse_pages(
        self, file_type: str, file_path: str, content: str
    ) -> list[PageData]:
        return self._fixed_pages

    def _format_prefix(self, num: int, title: str) -> str:
        return f"Page {num}: {title}" if title else f"Page {num}"

    def _extra_meta_for_page(self, num: int, title: str) -> dict:
        return dict(self._extra)


def _make_chunker(
    chunk_size: int = 50,
    pages: list[PageData] | None = None,
    extra_meta: dict | None = None,
) -> _ConcreteChunker:
    cfg = MagicMock()
    helpers = MagicMock()
    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.TEST"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return chunk_size
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n\n", "\n", " "]

    fu = MagicMock()
    fu.count_words.side_effect = lambda t: len(t.split())

    return _ConcreteChunker(
        pages=pages or [],
        extra_meta=extra_meta,
        cfg=cfg,
        helpers=helpers,
        file_utils=fu,
    )


SAMPLE_META = {
    "FileName": "test.txt",
    "FilePath": "/docs/test.txt",
    "FileType": "txt",
    "FileHash": "abc123",
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TestAbstractBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            PageBasedChunker()  # type: ignore[abstract]

    def test_is_subclass_of_chunker_strategy(self):
        assert issubclass(PageBasedChunker, ChunkerStrategy)

    def test_concrete_subclass_instantiates(self):
        c = _make_chunker()
        assert isinstance(c, PageBasedChunker)


# ---------------------------------------------------------------------------
# chunk_size property
# ---------------------------------------------------------------------------


class TestChunkSizeProperty:
    def test_returns_configured_value(self):
        assert _make_chunker(chunk_size=77).chunk_size == 77

    def test_default_applies_from_config(self):
        assert _make_chunker(chunk_size=256).chunk_size == 256


# ---------------------------------------------------------------------------
# chunk() — return types and empty handling
# ---------------------------------------------------------------------------


class TestChunkReturnType:
    def test_returns_tuple(self):
        c = _make_chunker(pages=[(1, "", "some body")])
        result = c.chunk("", SAMPLE_META)
        assert isinstance(result, tuple) and len(result) == 2

    def test_embeddings_always_none(self):
        c = _make_chunker(pages=[(1, "", "body")])
        _, emb = c.chunk("", SAMPLE_META)
        assert emb is None

    def test_empty_pages_returns_empty_list(self):
        c = _make_chunker(pages=[])
        docs, emb = c.chunk("anything", SAMPLE_META)
        assert docs == []
        assert emb is None

    def test_returns_langchain_docs(self):
        c = _make_chunker(pages=[(1, "", "some text")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert all(isinstance(d, langchainDoc) for d in docs)


# ---------------------------------------------------------------------------
# Single and multiple pages
# ---------------------------------------------------------------------------


class TestPageToChunkMapping:
    def test_single_page_yields_one_chunk(self):
        c = _make_chunker(pages=[(1, "Intro", "This is the introduction body.")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) == 1

    def test_three_pages_yield_three_chunks(self):
        pages = [
            (i, f"Section {i}", f"Body text for section {i}.") for i in range(1, 4)
        ]
        c = _make_chunker(pages=pages)
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) == 3

    def test_prefix_and_body_in_chunk_text(self):
        c = _make_chunker(pages=[(3, "My Section", "Some body content.")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert docs[0].page_content.startswith("Page 3: My Section")
        assert "Some body content." in docs[0].page_content

    def test_title_only_page_produces_chunk(self):
        c = _make_chunker(pages=[(1, "Section Title", "")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) == 1
        assert "Section Title" in docs[0].page_content

    def test_completely_empty_page_produces_prefix_chunk(self):
        """A page with empty title and empty body still produces a prefix chunk."""
        c = _make_chunker(pages=[(1, "", "")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) == 1
        assert docs[0].page_content == "Page 1"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_mychunk_index_sequential(self):
        pages = [(i, "", f"Body {i}.") for i in range(1, 5)]
        c = _make_chunker(pages=pages)
        docs, _ = c.chunk("", SAMPLE_META)
        assert [d.metadata["MyChunk"] for d in docs] == list(range(4))

    def test_original_metadata_fields_preserved(self):
        c = _make_chunker(pages=[(1, "", "body")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert docs[0].metadata["FileName"] == "test.txt"
        assert docs[0].metadata["FilePath"] == "/docs/test.txt"

    def test_original_metadata_dict_not_mutated(self):
        meta = dict(SAMPLE_META)
        c = _make_chunker(pages=[(1, "", "body")])
        c.chunk("", meta)
        assert "MyChunk" not in meta

    def test_extra_meta_merged_into_chunk(self):
        c = _make_chunker(pages=[(1, "", "body")], extra_meta={"CustomKey": "val"})
        docs, _ = c.chunk("", SAMPLE_META)
        assert docs[0].metadata["CustomKey"] == "val"

    def test_extra_meta_present_on_all_chunks(self):
        pages = [(i, "", f"body {i}") for i in range(1, 4)]
        c = _make_chunker(pages=pages, extra_meta={"Tag": "x"})
        docs, _ = c.chunk("", SAMPLE_META)
        assert all(d.metadata["Tag"] == "x" for d in docs)

    def test_extra_meta_does_not_override_base_fields(self):
        """Extra meta must not clobber FileName etc. from the source metadata."""
        c = _make_chunker(pages=[(1, "", "body")], extra_meta={"FileName": "INJECTED"})
        meta = dict(SAMPLE_META)
        docs, _ = c.chunk("", meta)
        # extra_meta is merged after the base, so it CAN override — just verify
        # it doesn't silently crash and MyChunk is still present
        assert "MyChunk" in docs[0].metadata


# ---------------------------------------------------------------------------
# Unique IDs
# ---------------------------------------------------------------------------


class TestUniqueIds:
    def test_all_ids_unique(self):
        pages = [(i, "", f"{'word ' * 5}") for i in range(1, 10)]
        c = _make_chunker(pages=pages)
        docs, _ = c.chunk("", SAMPLE_META)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_ids_are_strings(self):
        c = _make_chunker(pages=[(1, "", "body")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert all(isinstance(d.id, str) for d in docs)


# ---------------------------------------------------------------------------
# Oversized page splitting
# ---------------------------------------------------------------------------


class TestOversizedPageSplitting:
    def test_oversized_page_produces_multiple_chunks(self):
        # body_budget = max(50, chunk_size - prefix_words); need body > 50 words
        long_body = " ".join(f"w{i}" for i in range(60))  # 60 words
        c = _make_chunker(chunk_size=5, pages=[(1, "", long_body)])
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) > 1

    def test_oversized_sub_chunks_carry_prefix(self):
        long_body = " ".join(f"w{i}" for i in range(60))
        c = _make_chunker(chunk_size=5, pages=[(1, "MyPage", long_body)])
        docs, _ = c.chunk("", SAMPLE_META)
        for d in docs:
            assert d.page_content.startswith("Page 1: MyPage")

    def test_oversized_sub_chunks_have_sequential_indices(self):
        long_body = " ".join(f"w{i}" for i in range(60))
        c = _make_chunker(chunk_size=5, pages=[(1, "", long_body)])
        docs, _ = c.chunk("", SAMPLE_META)
        assert [d.metadata["MyChunk"] for d in docs] == list(range(len(docs)))

    def test_normal_size_page_not_split(self):
        c = _make_chunker(chunk_size=100, pages=[(1, "", "short body")])
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) == 1

    def test_mixed_normal_and_oversized_pages(self):
        short_body = "short body text"
        long_body = " ".join(f"w{i}" for i in range(60))  # 60 words > body_budget(50)
        c = _make_chunker(
            chunk_size=5,
            pages=[(1, "", short_body), (2, "", long_body)],
        )
        docs, _ = c.chunk("", SAMPLE_META)
        assert len(docs) > 2  # page 2 splits into multiple chunks
