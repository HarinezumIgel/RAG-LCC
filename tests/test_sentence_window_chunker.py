# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for SentenceWindowChunker."""

import sys
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from langchain_core.documents.base import Document as langchainDoc

from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy
from Strategies.Chunkers.SentenceWindowChunker import SentenceWindowChunker

# ── Stubs ─────────────────────────────────────────────────────────────────


class StubCfg:
    def __init__(self, max_chunk_size: int = 50):
        self._max_chunk_size = max_chunk_size

    def get_list(self, key: str):
        if key == "_SEPARATORS":
            return ["\n\n", "\n", " "]
        return []

    def get_int(self, key: str, default: int = 0) -> int:
        if "MAX_CHUNK_SIZE" in key:
            return self._max_chunk_size
        return default

    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default


class StubHelpers:
    def get_chunker_config_slot(self) -> str:
        return "_CHUNKERS.SENTENCE_WINDOW"


class StubFileUtils:
    def count_words(self, text: str) -> int:
        return len(text.split())


# ── Helpers ───────────────────────────────────────────────────────────────


SAMPLE_META = {
    "FileName": "test.txt",
    "FilePath": "/docs/test.txt",
    "FileHash": "abc123",
}


def _make_chunker(max_chunk_size: int = 50) -> SentenceWindowChunker:
    return SentenceWindowChunker(
        cfg=StubCfg(max_chunk_size=max_chunk_size),  # type: ignore[arg-type]
        helpers=StubHelpers(),  # type: ignore[arg-type]
        file_utils=StubFileUtils(),  # type: ignore[arg-type]
    )


# ── ABC contract ──────────────────────────────────────────────────────────


class TestABCContract:
    def test_is_subclass_of_chunker_strategy(self):
        assert issubclass(SentenceWindowChunker, ChunkerStrategy)

    def test_chunk_size_property(self):
        chunker = _make_chunker(max_chunk_size=100)
        assert chunker.chunk_size == 100


# ── Sentence splitting ────────────────────────────────────────────────────


class TestSplitSentences:
    def test_split_on_period(self):
        result = SentenceWindowChunker._split_sentences("Hello world. Foo bar.")
        assert result == ["Hello world.", "Foo bar."]

    def test_split_on_question_mark(self):
        result = SentenceWindowChunker._split_sentences("What? Why?")
        assert result == ["What?", "Why?"]

    def test_split_on_exclamation(self):
        result = SentenceWindowChunker._split_sentences("Wow! Amazing!")
        assert result == ["Wow!", "Amazing!"]

    def test_split_on_newlines(self):
        result = SentenceWindowChunker._split_sentences(
            "Line one\nLine two\nLine three"
        )
        assert result == ["Line one", "Line two", "Line three"]

    def test_empty_text(self):
        result = SentenceWindowChunker._split_sentences("")
        assert result == []

    def test_no_boundaries_returns_single(self):
        result = SentenceWindowChunker._split_sentences(
            "just one long sentence without punctuation"
        )
        assert result == ["just one long sentence without punctuation"]


# ── Sentence packing ─────────────────────────────────────────────────────


class TestPackSentences:
    def test_all_fit_in_one_chunk(self):
        chunker = _make_chunker(max_chunk_size=20)
        sentences = ["Hello world.", "Foo bar."]
        groups = chunker._pack_sentences(sentences)
        assert len(groups) == 1
        assert groups[0] == "Hello world. Foo bar."

    def test_splits_when_budget_exceeded(self):
        # 3-word budget: each sentence is 2 words
        chunker = _make_chunker(max_chunk_size=3)
        sentences = ["one two.", "three four.", "five six."]
        groups = chunker._pack_sentences(sentences)
        # first: "one two." (2), can't add "three four." (2+2=4 > 3)
        # second: "three four." (2), can't add "five six." (2+2=4 > 3)
        # third: "five six." (2)
        assert len(groups) == 3

    def test_packs_multiple_small_sentences(self):
        chunker = _make_chunker(max_chunk_size=10)
        sentences = ["A.", "B.", "C.", "D.", "E."]  # 1 word each
        groups = chunker._pack_sentences(sentences)
        # All 5 words fit in one chunk of size 10
        assert len(groups) == 1
        assert groups[0] == "A. B. C. D. E."

    def test_empty_list_returns_empty(self):
        chunker = _make_chunker(max_chunk_size=10)
        assert chunker._pack_sentences([]) == []

    def test_single_sentence(self):
        chunker = _make_chunker(max_chunk_size=10)
        groups = chunker._pack_sentences(["Hello world."])
        assert groups == ["Hello world."]

    def test_exact_budget_boundary(self):
        chunker = _make_chunker(max_chunk_size=4)
        sentences = ["one two.", "three four."]  # 2 words each, total 4 = MAX
        groups = chunker._pack_sentences(sentences)
        assert len(groups) == 1
        assert groups[0] == "one two. three four."


# ── End-to-end chunk() ────────────────────────────────────────────────────


class TestChunkEndToEnd:
    def test_returns_langchain_docs(self):
        chunker = _make_chunker(max_chunk_size=50)
        result, _ = chunker.chunk("Hello world. Foo bar.", SAMPLE_META)
        assert all(isinstance(d, langchainDoc) for d in result)

    def test_short_text_single_chunk(self):
        chunker = _make_chunker(max_chunk_size=50)
        result, _ = chunker.chunk("Hello world. Foo bar.", SAMPLE_META)
        assert len(result) == 1
        assert result[0].page_content == "Hello world. Foo bar."

    def test_multiple_chunks_long_text(self):
        # 5-word budget, 3 sentences of ~3 words each
        chunker = _make_chunker(max_chunk_size=5)
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) == 3

    def test_metadata_has_mychunk_index(self):
        chunker = _make_chunker(max_chunk_size=5)
        text = "One two three. Four five six. Seven eight nine."
        result, _ = chunker.chunk(text, SAMPLE_META)
        indices = [d.metadata["MyChunk"] for d in result]
        assert indices == list(range(len(result)))

    def test_original_metadata_preserved(self):
        chunker = _make_chunker(max_chunk_size=50)
        result, _ = chunker.chunk("Hello.", SAMPLE_META)
        assert result[0].metadata["FileName"] == "test.txt"
        assert result[0].metadata["FilePath"] == "/docs/test.txt"

    def test_original_metadata_not_mutated(self):
        meta = dict(SAMPLE_META)
        chunker = _make_chunker(max_chunk_size=50)
        chunker.chunk("Hello.", meta)
        assert "MyChunk" not in meta

    def test_unique_ids(self):
        chunker = _make_chunker(max_chunk_size=5)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result, _ = chunker.chunk(text, SAMPLE_META)
        ids = [d.id for d in result]
        assert len(ids) == len(set(ids))

    def test_empty_content_returns_empty(self):
        chunker = _make_chunker(max_chunk_size=50)
        result, _ = chunker.chunk("", SAMPLE_META)
        assert result == []

    def test_whitespace_only_returns_empty(self):
        chunker = _make_chunker(max_chunk_size=50)
        result, _ = chunker.chunk("   \n\n   ", SAMPLE_META)
        assert result == []


# ── Oversized sentence fallback ──────────────────────────────────────────


class TestOversizedFallback:
    def test_oversized_sentence_gets_split(self):
        chunker = _make_chunker(max_chunk_size=5)
        # Single sentence with 12 words — no punctuation boundary
        text = "one two three four five six seven eight nine ten eleven twelve"
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) > 1
        for doc in result:
            word_count = len(doc.page_content.split())
            assert word_count <= 5

    def test_mixed_normal_and_oversized(self):
        chunker = _make_chunker(max_chunk_size=5)
        # Short sentence + oversized sentence
        text = "Short one. alpha bravo charlie delta echo foxtrot golf hotel india"
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) >= 2
        assert result[0].page_content == "Short one."
