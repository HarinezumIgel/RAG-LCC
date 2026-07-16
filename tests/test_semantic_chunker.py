# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for SemanticChunker."""

import sys
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from langchain_core.documents.base import Document as langchainDoc

from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy
from Strategies.Chunkers.SemanticChunker import SemanticChunker

# ── Stubs ─────────────────────────────────────────────────────────────────


class StubCfg:
    def __init__(self, max_chunk_size: int = 50, percentile: int = 10):
        self._max_chunk_size = max_chunk_size
        self._percentile = percentile

    def get_list(self, key: str):
        if key == "_SEPARATORS":
            return ["\n\n", "\n", " "]
        return []

    def get_int(self, key: str, default: int = 0) -> int:
        if "MAX_CHUNK_SIZE" in key:
            return self._max_chunk_size
        if "BREAKPOINT_PERCENTILE" in key:
            return self._percentile
        if "EMBED_BATCH_SIZE" in key:
            return 64
        if "MIN_SENTENCE_WORDS" in key:
            return 0  # disable consolidation by default in tests
        return default

    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default


class StubHelpers:
    def get_chunker_config_slot(self) -> str:
        return "_CHUNKERS.SEMANTIC"


class StubFileUtils:
    def count_words(self, text: str) -> int:
        return len(text.split())


class StubEmbedder:
    """Returns deterministic embeddings: the vector for sentence i is a one-hot
    at dimension i (mod dim). Consecutive sentences with different indices will
    have cosine similarity = 0, making every boundary a breakpoint."""

    def __init__(self, dim: int = 16):
        self._dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i, _ in enumerate(texts):
            vec = [0.0] * self._dim
            vec[i % self._dim] = 1.0
            embeddings.append(vec)
        return embeddings


class SimilarEmbedder:
    """Returns the same embedding for every sentence — cosine similarity = 1.0
    everywhere, so no breakpoints should be created."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0]] * len(texts)


class TwoTopicEmbedder:
    """First N sentences get one vector, remaining get another.
    There should be exactly one breakpoint at position N."""

    def __init__(self, split_at: int = 3):
        self._split_at = split_at

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs: list[list[float]] = []
        for i in range(len(texts)):
            if i < self._split_at:
                embs.append([1.0, 0.0, 0.0, 0.0])
            else:
                embs.append([0.0, 1.0, 0.0, 0.0])
        return embs


# ── Helpers ───────────────────────────────────────────────────────────────


SAMPLE_META = {"FileName": "doc.txt", "FilePath": "/docs/doc.txt", "FileHash": "abc"}


def _make_chunker(
    max_chunk_size: int = 200,
    percentile: int = 10,
    embedder=None,
) -> SemanticChunker:
    return SemanticChunker(
        cfg=StubCfg(max_chunk_size=max_chunk_size, percentile=percentile),  # type: ignore[arg-type]
        helpers=StubHelpers(),  # type: ignore[arg-type]
        file_utils=StubFileUtils(),  # type: ignore[arg-type]
        embedder=embedder or StubEmbedder(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSemanticChunkerABC:
    def test_is_subclass_of_strategy(self):
        assert issubclass(SemanticChunker, ChunkerStrategy)

    def test_chunk_size_returns_max(self):
        chunker = _make_chunker(max_chunk_size=128)
        assert chunker.chunk_size == 128


class TestSentenceSplitting:
    def test_splits_on_period(self):
        sents = SemanticChunker._split_sentences("Hello world. How are you? Fine!")
        assert len(sents) == 3

    def test_splits_on_newline(self):
        sents = SemanticChunker._split_sentences("Line one\nLine two\nLine three")
        assert len(sents) == 3

    def test_empty_string(self):
        sents = SemanticChunker._split_sentences("")
        assert sents == []

    def test_single_sentence(self):
        sents = SemanticChunker._split_sentences("Just one sentence")
        assert len(sents) == 1


class TestCosineSimilarities:
    def test_identical_vectors_give_one(self):
        embs = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        sims = SemanticChunker._cosine_similarities(embs)
        assert len(sims) == 2
        assert all(abs(s - 1.0) < 1e-6 for s in sims)

    def test_orthogonal_vectors_give_zero(self):
        embs = [[1.0, 0.0], [0.0, 1.0]]
        sims = SemanticChunker._cosine_similarities(embs)
        assert len(sims) == 1
        assert abs(sims[0]) < 1e-6

    def test_length_is_n_minus_one(self):
        embs = [[1.0, 0.0]] * 5
        sims = SemanticChunker._cosine_similarities(embs)
        assert len(sims) == 4


class TestBreakpoints:
    def test_no_breakpoints_for_uniform_similarity(self):
        chunker = _make_chunker(percentile=10, embedder=SimilarEmbedder())
        # All similarities are 1.0, percentile threshold ≈ 1.0, nothing below it
        sims = [1.0, 1.0, 1.0, 1.0]
        bps = chunker._find_breakpoints(sims)
        assert bps == []

    def test_breakpoint_at_low_similarity(self):
        chunker = _make_chunker(percentile=50)
        sims = [0.9, 0.1, 0.9, 0.9]
        bps = chunker._find_breakpoints(sims)
        assert 2 in bps  # after the 0.1 drop

    def test_empty_similarities(self):
        chunker = _make_chunker()
        assert chunker._find_breakpoints([]) == []


class TestGrouping:
    def test_groups_sentences_at_breakpoints(self):
        sentences = ["A", "B", "C", "D", "E"]
        groups = SemanticChunker._group_sentences(sentences, [2, 4])
        assert len(groups) == 3
        assert groups[0] == "A B"
        assert groups[1] == "C D"
        assert groups[2] == "E"

    def test_no_breakpoints_single_group(self):
        sentences = ["A", "B", "C"]
        groups = SemanticChunker._group_sentences(sentences, [])
        assert len(groups) == 1
        assert groups[0] == "A B C"


class TestSemanticChunkerEndToEnd:
    def test_returns_langchain_docs(self):
        chunker = _make_chunker()
        text = "Topic one sentence. Topic two sentence. Topic three sentence."
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert all(isinstance(d, langchainDoc) for d in result)

    def test_unique_ids(self):
        chunker = _make_chunker()
        text = "First topic here. Second topic here. Third topic here."
        result, _ = chunker.chunk(text, SAMPLE_META)
        ids = [d.id for d in result]
        assert len(ids) == len(set(ids))

    def test_mychunk_indices(self):
        chunker = _make_chunker()
        text = "Sentence one. Sentence two. Sentence three."
        result, _ = chunker.chunk(text, SAMPLE_META)
        indices = [d.metadata["MyChunk"] for d in result]
        assert indices == list(range(len(result)))

    def test_metadata_preserved(self):
        chunker = _make_chunker()
        text = "Hello world. Goodbye world."
        result, _ = chunker.chunk(text, SAMPLE_META)
        for d in result:
            assert d.metadata["FileName"] == "doc.txt"

    def test_metadata_not_mutated(self):
        meta = dict(SAMPLE_META)
        chunker = _make_chunker()
        chunker.chunk("Some text here.", meta)
        assert "MyChunk" not in meta

    def test_similar_content_stays_together(self):
        chunker = _make_chunker(embedder=SimilarEmbedder())
        text = "Cat is nice. Cat is lovely. Cat is great. Cat is wonderful."
        result, _ = chunker.chunk(text, SAMPLE_META)
        # All sentences are identical embeddings → no breakpoints → single chunk
        assert len(result) == 1

    def test_two_topics_split_correctly(self):
        chunker = _make_chunker(percentile=50, embedder=TwoTopicEmbedder(split_at=3))
        text = "Dogs are great. Dogs are loyal. Dogs are fun. Cats sleep. Cats purr. Cats nap."
        result, _ = chunker.chunk(text, SAMPLE_META)
        assert len(result) >= 2
        # First chunk should contain dog sentences
        assert "Dogs" in result[0].page_content

    def test_single_sentence_no_embedding(self):
        chunker = _make_chunker()
        result, pre_emb = chunker.chunk("Only one sentence", SAMPLE_META)
        assert len(result) == 1
        assert result[0].page_content == "Only one sentence"
        # Single sentence → no sentence embeddings available
        assert pre_emb is None

    def test_empty_content(self):
        chunker = _make_chunker()
        result, _ = chunker.chunk("", SAMPLE_META)
        assert len(result) == 0

    def test_oversized_segment_gets_split(self):
        # max_chunk_size=5 words, but a semantic group could be larger
        chunker = _make_chunker(max_chunk_size=5, embedder=SimilarEmbedder())
        text = "word " * 20 + "." + " more " * 20
        # All similarities = 1.0 → one group → exceeds 5 words → fallback split
        result, pre_emb = chunker.chunk(text.strip(), SAMPLE_META)
        assert len(result) > 1
        for d in result:
            assert StubFileUtils().count_words(d.page_content) <= 5
        # Single sentence → early return path → no pre-computed embeddings
        assert pre_emb is None


class TestConsolidation:
    """Test the _consolidate_short pre-processing step."""

    def _make_consolidating_chunker(
        self, min_sentence_words: int = 15, max_chunk_size: int = 200
    ) -> SemanticChunker:
        """Create a chunker with consolidation enabled."""

        class ConsolidationCfg(StubCfg):
            def __init__(self, mcs: int, pct: int, msw: int):
                super().__init__(max_chunk_size=mcs, percentile=pct)
                self._msw = msw

            def get_int(self, key: str, default: int = 0) -> int:
                if "MIN_SENTENCE_WORDS" in key:
                    return self._msw
                return super().get_int(key, default)

        return SemanticChunker(
            cfg=ConsolidationCfg(max_chunk_size, 10, min_sentence_words),  # type: ignore[arg-type]
            helpers=StubHelpers(),  # type: ignore[arg-type]
            file_utils=StubFileUtils(),  # type: ignore[arg-type]
            embedder=SimilarEmbedder(),
        )

    def test_short_fragments_merged(self):
        chunker = self._make_consolidating_chunker(min_sentence_words=10)
        fragments = ["a b c", "d e f", "g h i j"]  # 3, 3, 4 words
        result = chunker._consolidate_short(fragments)
        assert len(result) == 1
        assert result[0] == "a b c d e f g h i j"

    def test_long_sentence_not_merged(self):
        chunker = self._make_consolidating_chunker(min_sentence_words=5)
        sentences = ["short", "This is a long enough sentence to stand alone", "tiny"]
        result = chunker._consolidate_short(sentences)
        # "short" is buffered, then the long sentence flushes it and stands alone, "tiny" remains
        assert len(result) == 3
        assert result[0] == "short"
        assert "long enough" in result[1]
        assert result[2] == "tiny"

    def test_threshold_zero_disables(self):
        chunker = self._make_consolidating_chunker(min_sentence_words=0)
        fragments = ["a", "b", "c"]
        result = chunker._consolidate_short(fragments)
        # Every sentence >= 0 words → all "long" → no merging
        assert result == ["a", "b", "c"]

    def test_pcie_slot_fragments_consolidated(self):
        chunker = self._make_consolidating_chunker(min_sentence_words=15)
        fragments = [
            "39  PCIe 4.0 x8 card slot 6",  # 7w
            "41  PCIe 4.0 x16 card slot 4",  # 7w
            "40  PCIe 4.0 x16 card slot 5",  # 7w
            "42  PCIe 4.0 x16 card slot 3",  # 7w
            "44  PCIe 4.0 x8 card slot 2",  # 7w
            "45  PCIe 4.0 x 16 card slot 1",  # 8w
        ]
        result = chunker._consolidate_short(fragments)
        # 6 fragments × ~7 words → should merge into ~3 units
        assert len(result) < len(fragments)
        # All PCIe slot info is preserved
        joined = " ".join(result)
        for frag in fragments:
            assert frag in joined
