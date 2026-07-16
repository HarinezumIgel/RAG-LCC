# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for ChunkerStrategy / RecursiveChunker."""

import sys
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from langchain_core.documents.base import Document as langchainDoc

from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy
from Strategies.Chunkers.RecursiveChunker import RecursiveChunker

# ── Stubs ─────────────────────────────────────────────────────────────────


class StubCfg:
    """Minimal Config stand-in."""

    def __init__(self, chunk_size: int = 10, overlap: int = 2):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def get_list(self, key: str):
        if key == "_SEPARATORS":
            return ["\n\n", "\n", " "]
        return []

    def get_int(self, key: str, default: int = 0) -> int:
        if "CHUNK_SIZE" in key:
            return self._chunk_size
        if "CHUNK_OVERLAP" in key:
            return self._overlap
        return default

    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default


class StubHelpers:
    def get_chunker_config_slot(self) -> str:
        return "_CHUNKERS.RECURSIVE"


class StubFileUtils:
    def count_words(self, text: str) -> int:
        return len(text.split())

    def get_stopwords(self, text: str) -> list[str]:
        return []

    def removeStopwords(self, text: str, stop_set: set[str]) -> str:
        return " ".join(w for w in text.split() if w.lower() not in stop_set)


class StubFileUtilsWithStopwords(StubFileUtils):
    """Returns a fixed set of stop-words so the removal path is exercised."""

    def get_stopwords(self, text: str) -> list[str]:
        return ["the", "a", "is"]


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_chunker(
    chunk_size: int = 10, overlap: int = 0, with_stopwords: bool = False
) -> RecursiveChunker:
    fu = StubFileUtilsWithStopwords() if with_stopwords else StubFileUtils()
    return RecursiveChunker(
        cfg=StubCfg(chunk_size=chunk_size, overlap=overlap),  # type: ignore[arg-type]
        helpers=StubHelpers(),  # type: ignore[arg-type]
        file_utils=fu,  # type: ignore[arg-type]
    )


SAMPLE_META = {
    "FileName": "test.txt",
    "FilePath": "/docs/test.txt",
    "FileHash": "abc123",
}


# ── Tests ─────────────────────────────────────────────────────────────────


class TestChunkerStrategyABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ChunkerStrategy()  # type: ignore[abstract]

    def test_recursive_is_subclass(self):
        assert issubclass(RecursiveChunker, ChunkerStrategy)


class TestRecursiveChunker:

    def test_chunk_size_property(self):
        chunker = _make_chunker(chunk_size=42)
        assert chunker.chunk_size == 42

    def test_single_chunk_for_short_text(self):
        chunker = _make_chunker(chunk_size=50)
        result, _ = chunker.chunk("hello world", SAMPLE_META)
        assert len(result) == 1
        assert result[0].page_content == "hello world"

    def test_returns_langchain_docs(self):
        chunker = _make_chunker(chunk_size=50)
        result, _ = chunker.chunk("some text", SAMPLE_META)
        assert all(isinstance(d, langchainDoc) for d in result)

    def test_multiple_chunks_for_long_text(self):
        chunker = _make_chunker(chunk_size=5, overlap=0)
        text = " ".join(f"word{i}" for i in range(20))
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) > 1

    def test_chunks_have_unique_ids(self):
        chunker = _make_chunker(chunk_size=5)
        text = " ".join(f"w{i}" for i in range(30))
        result, _ = chunker.chunk(text, SAMPLE_META)
        ids = [d.id for d in result]
        assert len(ids) == len(set(ids))

    def test_metadata_has_mychunk_index(self):
        chunker = _make_chunker(chunk_size=5)
        text = " ".join(f"w{i}" for i in range(20))
        result, _ = chunker.chunk(text, SAMPLE_META)
        indices = [d.metadata["MyChunk"] for d in result]
        assert indices == list(range(len(result)))

    def test_original_metadata_preserved(self):
        chunker = _make_chunker(chunk_size=50)
        result, _ = chunker.chunk("some text", SAMPLE_META)
        for d in result:
            assert d.metadata["FileName"] == "test.txt"
            assert d.metadata["FilePath"] == "/docs/test.txt"

    def test_original_metadata_not_mutated(self):
        meta = dict(SAMPLE_META)
        chunker = _make_chunker(chunk_size=50)
        chunker.chunk("some text", meta)
        assert "MyChunk" not in meta

    def test_stopword_removal_path(self):
        chunker = _make_chunker(chunk_size=50, with_stopwords=True)
        text = "the cat is a great animal"
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) == 1
        # "the", "is", "a" should have been stripped
        assert "the" not in result[0].page_content.split()
        assert "is" not in result[0].page_content.split()

    def test_overlap_produces_overlapping_content(self):
        chunker = _make_chunker(chunk_size=5, overlap=2)
        text = " ".join(f"w{i}" for i in range(20))
        result, _ = chunker.chunk(text, SAMPLE_META)
        # With overlap, later chunks should share some words with previous chunks
        if len(result) >= 2:
            words_0 = set(result[0].page_content.split())
            words_1 = set(result[1].page_content.split())
            assert words_0 & words_1  # non-empty intersection

    def test_empty_content(self):
        chunker = _make_chunker(chunk_size=50)
        result, _ = chunker.chunk("", SAMPLE_META)
        # RecursiveCharacterTextSplitter returns one empty doc or nothing
        assert len(result) <= 1
