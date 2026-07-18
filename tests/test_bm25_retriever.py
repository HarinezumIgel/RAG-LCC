# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for Strategies.BM25Retriever — index build, incremental update,
BM25 scoring, persistence, RRF fusion, and filter matching.
"""

import gzip
import math
import os
import pickle
import sys
from collections import Counter
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents.base import Document as LangchainDocument

from Strategies.BM25Retriever import BM25Retriever, _BM25IndexData

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    """Minimal Config stub returning BM25 defaults."""

    _DEFAULTS: Dict[str, Any] = {
        "_BM25_INDEX.k1": 1.2,
        "_BM25_INDEX.b": 0.75,
        "_BM25_INDEX.rrf_k": 60,
    }

    def get(self, key, default=None):
        return self._DEFAULTS.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self._DEFAULTS.get(key, default))

    def get_str(self, key: str, default: str = "") -> str:
        return str(self._DEFAULTS.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._DEFAULTS.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self._DEFAULTS.get(key, default))

    def get_list(self, key: str, default=None) -> list:
        return default if default is not None else []

    def get_dict(self, key: str, default=None) -> dict:
        return default if default is not None else {}


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


class StubSharedHelpers:
    """Lightweight tokenizer matching SharedHelpers.tokenize()."""

    import re as _re

    _TOKEN_RE = _re.compile(r"[A-Za-z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [t.lower() for t in self._TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
    yield
    BM25Retriever._reset()  # type: ignore[reportPrivateUsage]


def _make_retriever(**overrides: Any) -> BM25Retriever:
    """Create a BM25Retriever with stub deps (bypass __init__)."""
    r = BM25Retriever.__new__(BM25Retriever)
    r._initialized = True
    r.cfg = overrides.get("cfg", StubConfig())
    r.pretty = overrides.get("pretty", StubPrettyWriter())
    r._shared = overrides.get("shared", StubSharedHelpers())
    r._data = _BM25IndexData()
    r._k1 = overrides.get("k1", 1.2)
    r._b = overrides.get("b", 0.75)
    r._rrf_k = overrides.get("rrf_k", 60)
    r.perf_logger = MagicMock()
    # Register as the singleton instance so is_loaded_for works
    BM25Retriever._instance = r  # type: ignore[reportPrivateUsage]
    return r


def _build_index(r: BM25Retriever, chunks: List[Dict[str, Any]]) -> None:
    """Populate the retriever index from a list of {id, text, meta} dicts."""
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]
    r.add_chunks(ids, texts, metas)
    r._data.collection_name = "test_coll"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

CHUNKS = [
    {
        "id": "c1",
        "text": "The quick brown fox jumps over the lazy dog",
        "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
    },
    {
        "id": "c2",
        "text": "A brown dog plays in the park with another dog",
        "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
    },
    {
        "id": "c3",
        "text": "Python programming language is widely used",
        "meta": {"FilePath": "tech.txt", "FileName": "tech.txt"},
    },
    {
        "id": "c4",
        "text": "Machine learning models require training data",
        "meta": {"FilePath": "tech.txt", "FileName": "tech.txt"},
    },
    {
        "id": "c5",
        "text": "The fox and the dog became friends",
        "meta": {"FilePath": "stories.txt", "FileName": "stories.txt"},
    },
]


# ===========================================================================
# _BM25IndexData
# ===========================================================================


class TestBM25IndexData:
    def test_initial_state(self):
        data = _BM25IndexData()
        assert data.N == 0
        assert data.avg_dl == 0.0
        assert data.chunk_ids == []
        assert data.idf == {}
        assert data.collection_name == ""


# ===========================================================================
# Singleton & properties
# ===========================================================================


class TestSingleton:
    def test_rrf_k_property(self):
        r = _make_retriever(rrf_k=42)
        assert r.rrf_k == 42

    def test_is_loaded_empty(self):
        r = _make_retriever()
        assert r.is_loaded is False

    def test_is_loaded_after_add(self):
        r = _make_retriever()
        _build_index(r, CHUNKS[:1])
        assert r.is_loaded is True

    def test_is_loaded_for_correct_collection(self):
        r = _make_retriever()
        _build_index(r, CHUNKS[:1])
        assert r.is_loaded_for("test_coll") is True
        assert r.is_loaded_for("other_coll") is False


# ===========================================================================
# add_chunks & corpus stats
# ===========================================================================


class TestAddChunks:
    def test_chunk_count(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        assert r._data.N == 5

    def test_avg_dl_positive(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        assert r._data.avg_dl > 0.0

    def test_df_counts(self):
        """'the' appears in chunks c1, c2, c5 → df['the'] == 3."""
        r = _make_retriever()
        _build_index(r, CHUNKS)
        assert r._data.df["the"] == 3

    def test_idf_computed(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        # Every term that appears should have an IDF entry
        assert len(r._data.idf) == len(r._data.df)
        # IDF for 'the' (df=3, N=5) should be positive
        assert r._data.idf["the"] > 0.0

    def test_idf_formula(self):
        """Verify IDF = log(1 + (N - df + 0.5) / (df + 0.5))."""
        r = _make_retriever()
        _build_index(r, CHUNKS)
        N = r._data.N
        for term, freq in r._data.df.items():
            expected = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))
            assert (
                abs(r._data.idf[term] - expected) < 1e-9
            ), f"IDF mismatch for '{term}'"

    def test_chunk_texts_stored(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        assert r._data.chunk_texts[0] == CHUNKS[0]["text"]

    def test_chunk_metas_stored(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        assert r._data.chunk_metas[2]["FilePath"] == "tech.txt"

    def test_incremental_add(self):
        """Adding chunks in two batches should equal adding all at once."""
        r1 = _make_retriever()
        _build_index(r1, CHUNKS)

        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever()
        r2.add_chunks(
            [c["id"] for c in CHUNKS[:3]],
            [c["text"] for c in CHUNKS[:3]],
            [c.get("meta", {}) for c in CHUNKS[:3]],
        )
        r2.add_chunks(
            [c["id"] for c in CHUNKS[3:]],
            [c["text"] for c in CHUNKS[3:]],
            [c.get("meta", {}) for c in CHUNKS[3:]],
        )

        assert r2._data.N == r1._data.N
        assert r2._data.df == r1._data.df
        assert abs(r2._data.avg_dl - r1._data.avg_dl) < 1e-9


# ===========================================================================
# remove_by_filepath
# ===========================================================================


class TestRemoveByFilepath:
    def test_removes_correct_chunks(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        r.remove_by_filepath("animals.txt")
        assert r._data.N == 3
        remaining_paths = {m["FilePath"] for m in r._data.chunk_metas}
        assert "animals.txt" not in remaining_paths

    def test_df_decremented(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        # 'dog' appears in c1, c2, c5 → df=3. Removing animals.txt drops c1, c2 → df=1
        r.remove_by_filepath("animals.txt")
        assert r._data.df.get("dog", 0) == 1

    def test_term_removed_when_df_zero(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        # 'park' only in c2 (animals.txt). After removal, it should be gone from df.
        r.remove_by_filepath("animals.txt")
        assert "park" not in r._data.df

    def test_noop_when_filepath_missing(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        n_before = r._data.N
        r.remove_by_filepath("nonexistent.txt")
        assert r._data.N == n_before

    def test_idf_recomputed_after_removal(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        r.remove_by_filepath("tech.txt")
        N = r._data.N
        for term, freq in r._data.df.items():
            expected = math.log(1.0 + (N - freq + 0.5) / (freq + 0.5))
            assert abs(r._data.idf[term] - expected) < 1e-9

    def test_remove_then_add(self):
        """Simulates RAGLoad re-ingestion: remove old chunks, add updated ones."""
        r = _make_retriever()
        _build_index(r, CHUNKS)

        r.remove_by_filepath("animals.txt")
        assert r._data.N == 3

        new_chunks = [
            {
                "id": "c1_v2",
                "text": "The quick brown fox",
                "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
            },
        ]
        r.add_chunks(
            [c["id"] for c in new_chunks],
            [c["text"] for c in new_chunks],
            [c.get("meta", {}) for c in new_chunks],
        )
        assert r._data.N == 4
        assert "c1_v2" in r._data.chunk_ids


# ===========================================================================
# BM25 scoring (_score_doc)
# ===========================================================================


class TestScoring:
    def test_matching_query_scores_positive(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        # "brown fox" should score c1 and c5 positively
        docs = r.query("brown fox", k=10)
        assert len(docs) > 0
        assert all(d.metadata["bm25_score"] > 0.0 for d in docs)

    def test_no_match_returns_empty(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("xyznonexistent", k=10)
        assert docs == []

    def test_empty_query_returns_empty(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("", k=10)
        assert docs == []

    def test_empty_index_returns_empty(self):
        r = _make_retriever()
        docs = r.query("anything", k=10)
        assert docs == []

    def test_top_k_limits_results(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("the dog fox", k=2)
        assert len(docs) <= 2

    def test_results_sorted_descending(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("brown dog fox", k=10)
        scores = [d.metadata["bm25_score"] for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_chroma_score_placeholder(self):
        """BM25-only docs should have chroma_score=0.0 for pipeline compat."""
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("fox", k=5)
        assert all(d.metadata["chroma_score"] == 0.0 for d in docs)

    def test_chroma_sim_set(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("fox", k=5)
        assert all(d.metadata["chroma_sim"] == 1.0 for d in docs)

    def test_document_id_set(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("python programming", k=5)
        assert any(d.id == "c3" for d in docs)

    def test_k1_effect(self):
        """Higher k1 should increase the spread of scores for high-TF docs."""
        r_low = _make_retriever(k1=0.5)
        _build_index(r_low, CHUNKS)
        docs_low = r_low.query("dog", k=10)

        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r_high = _make_retriever(k1=2.0)
        _build_index(r_high, CHUNKS)
        docs_high = r_high.query("dog", k=10)

        # c2 has 'dog' twice → higher k1 should give it a higher relative score
        score_c2_low = next(
            (d.metadata["bm25_score"] for d in docs_low if d.id == "c2"), 0
        )
        score_c2_high = next(
            (d.metadata["bm25_score"] for d in docs_high if d.id == "c2"), 0
        )
        assert score_c2_high > score_c2_low

    def test_b_zero_ignores_length(self):
        """With b=0, document length should not affect normalisation."""
        r = _make_retriever(b=0.0)
        _build_index(r, CHUNKS)
        docs = r.query("dog", k=10)
        # Should still return results (scoring still works)
        assert len(docs) > 0


# ===========================================================================
# File filter
# ===========================================================================


class TestFileFilter:
    def test_simple_eq_filter(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("the", k=20, file_filter={"FileName": "tech.txt"})
        assert all(d.metadata["FileName"] == "tech.txt" for d in docs)

    def test_dollar_eq_filter(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("the", k=20, file_filter={"FileName": {"$eq": "animals.txt"}})
        assert all(d.metadata["FileName"] == "animals.txt" for d in docs)

    def test_filter_no_match(self):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        docs = r.query("dog", k=20, file_filter={"FileName": "nonexistent.txt"})
        assert docs == []


# ===========================================================================
# _matches_filter (static)
# ===========================================================================


class TestMatchesFilter:
    def test_simple_match(self):
        assert BM25Retriever._matches_filter({"a": 1}, {"a": 1}) is True

    def test_simple_no_match(self):
        assert BM25Retriever._matches_filter({"a": 1}, {"a": 2}) is False

    def test_dollar_eq_match(self):
        assert BM25Retriever._matches_filter({"a": "x"}, {"a": {"$eq": "x"}}) is True

    def test_dollar_eq_no_match(self):
        assert BM25Retriever._matches_filter({"a": "x"}, {"a": {"$eq": "y"}}) is False

    def test_missing_key(self):
        assert BM25Retriever._matches_filter({}, {"a": 1}) is False

    def test_multi_key_all_match(self):
        meta = {"a": 1, "b": 2}
        assert BM25Retriever._matches_filter(meta, {"a": 1, "b": 2}) is True

    def test_multi_key_partial_mismatch(self):
        meta = {"a": 1, "b": 2}
        assert BM25Retriever._matches_filter(meta, {"a": 1, "b": 99}) is False


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================


def _make_doc(doc_id: str, score: float, source: str = "v") -> LangchainDocument:
    return LangchainDocument(
        page_content=f"content of {doc_id}",
        metadata={"chroma_score": score, "source": source},
        id=doc_id,
    )


class TestRRF:
    def test_single_list(self):
        docs = [_make_doc("a", 0.9), _make_doc("b", 0.8)]
        result = BM25Retriever.reciprocal_rank_fusion(docs, k=60)
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[0].metadata["rrf_score"] > result[1].metadata["rrf_score"]

    def test_two_lists_merge(self):
        list1 = [_make_doc("a", 0.9), _make_doc("b", 0.8)]
        list2 = [_make_doc("b", 0.7), _make_doc("c", 0.6)]
        result = BM25Retriever.reciprocal_rank_fusion(list1, list2, k=60)
        # All three docs present
        ids = {d.id for d in result}
        assert ids == {"a", "b", "c"}

    def test_shared_doc_boosted(self):
        """A doc in both lists should score higher than one in only one list."""
        list1 = [_make_doc("a", 0.9), _make_doc("shared", 0.8)]
        list2 = [_make_doc("shared", 0.7), _make_doc("b", 0.6)]
        result = BM25Retriever.reciprocal_rank_fusion(list1, list2, k=60)
        shared_score = next(d.metadata["rrf_score"] for d in result if d.id == "shared")
        a_score = next(d.metadata["rrf_score"] for d in result if d.id == "a")
        assert shared_score > a_score

    def test_rrf_score_formula(self):
        """Verify RRF(d) = sum(1 / (k + rank)) for a single list."""
        docs = [_make_doc("a", 0.9)]
        result = BM25Retriever.reciprocal_rank_fusion(docs, k=60)
        expected = 1.0 / (60 + 1)
        assert abs(result[0].metadata["rrf_score"] - expected) < 1e-9

    def test_custom_k(self):
        docs = [_make_doc("a", 0.9)]
        result = BM25Retriever.reciprocal_rank_fusion(docs, k=10)
        expected = 1.0 / (10 + 1)
        assert abs(result[0].metadata["rrf_score"] - expected) < 1e-9

    def test_chroma_score_overwritten_with_rrf(self):
        """RRF should overwrite chroma_score with the RRF score."""
        docs = [_make_doc("a", 0.9)]
        result = BM25Retriever.reciprocal_rank_fusion(docs, k=60)
        assert result[0].metadata["chroma_score"] == result[0].metadata["rrf_score"]

    def test_empty_lists(self):
        result = BM25Retriever.reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_result_sorted_descending(self):
        list1 = [_make_doc("a", 0.9), _make_doc("b", 0.8), _make_doc("c", 0.7)]
        list2 = [_make_doc("c", 0.9), _make_doc("b", 0.8), _make_doc("a", 0.7)]
        result = BM25Retriever.reciprocal_rank_fusion(list1, list2, k=60)
        scores = [d.metadata["rrf_score"] for d in result]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# Persistence (round-trip)
# ===========================================================================


class TestPersistence:
    def test_persist_and_load(self, tmp_path):
        r = _make_retriever()
        _build_index(r, CHUNKS)
        r._data.collection_name = "test_coll"
        r._data.doc_count_at_build = 5

        r.persist(str(tmp_path))

        idx_path = os.path.join(str(tmp_path), BM25Retriever.INDEX_FILENAME)
        assert os.path.isfile(idx_path)

        # Load into a fresh retriever
        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever()
        r2._load(idx_path)

        assert r2._data.N == r._data.N
        assert r2._data.collection_name == "test_coll"
        assert r2._data.doc_count_at_build == 5
        assert r2._data.chunk_ids == r._data.chunk_ids
        assert r2._data.df == r._data.df
        assert abs(r2._data.avg_dl - r._data.avg_dl) < 1e-9

    def test_file_is_gzipped(self, tmp_path):
        r = _make_retriever()
        _build_index(r, CHUNKS[:1])
        r.persist(str(tmp_path))
        idx_path = os.path.join(str(tmp_path), BM25Retriever.INDEX_FILENAME)
        # gzip magic number
        with open(idx_path, "rb") as f:
            assert f.read(2) == b"\x1f\x8b"

    def test_load_or_rebuild_uses_persisted(self, tmp_path):
        """load_or_rebuild should load from disk when count matches."""
        r = _make_retriever()
        _build_index(r, CHUNKS)
        r._data.collection_name = "my_coll"
        r._data.doc_count_at_build = 42
        r.persist(str(tmp_path))

        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever()
        coll = MagicMock()
        coll.count.return_value = 42
        r2.load_or_rebuild(str(tmp_path), "my_coll", coll)

        assert r2._data.N == 5
        assert r2._data.collection_name == "my_coll"

    def test_load_or_rebuild_stale_triggers_rebuild(self, tmp_path):
        """If collection count differs, rebuild from collection."""
        r = _make_retriever()
        _build_index(r, CHUNKS[:2])
        r._data.collection_name = "my_coll"
        r._data.doc_count_at_build = 2
        r.persist(str(tmp_path))

        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever()
        coll = MagicMock()
        coll.count.return_value = 999  # differs → stale

        # The rebuild will call collection.get()
        coll.get.return_value = {
            "ids": ["x1"],
            "documents": ["rebuilt content"],
            "metadatas": [{"FilePath": "new.txt"}],
        }
        r2.load_or_rebuild(str(tmp_path), "my_coll", coll)

        assert r2._data.N == 1
        assert r2._data.chunk_texts == ["rebuilt content"]

    def test_load_or_rebuild_no_file_triggers_rebuild(self, tmp_path):
        """When no persisted file exists, rebuild from collection."""
        r = _make_retriever()
        coll = MagicMock()
        coll.count.return_value = 2
        coll.get.return_value = {
            "ids": ["a1", "a2"],
            "documents": ["hello world", "foo bar"],
            "metadatas": [{"FilePath": "f.txt"}, {"FilePath": "f.txt"}],
        }
        r.load_or_rebuild(str(tmp_path), "coll_x", coll)

        assert r._data.N == 2
        assert r._data.collection_name == "coll_x"


# ===========================================================================
# build_and_persist (full rebuild path)
# ===========================================================================


class TestBuildAndPersist:
    def test_full_rebuild(self, tmp_path):
        r = _make_retriever()
        coll = MagicMock()
        coll.count.return_value = 3
        coll.get.return_value = {
            "ids": ["b1", "b2", "b3"],
            "documents": ["alpha beta", "beta gamma", "gamma delta"],
            "metadatas": [
                {"FilePath": "a.txt"},
                {"FilePath": "a.txt"},
                {"FilePath": "b.txt"},
            ],
        }
        r.build_and_persist(str(tmp_path), "full_coll", coll)

        assert r._data.N == 3
        idx_path = os.path.join(str(tmp_path), BM25Retriever.INDEX_FILENAME)
        assert os.path.isfile(idx_path)

        # Verify persisted data can be loaded back
        BM25Retriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever()
        r2._load(idx_path)
        assert r2._data.N == 3
        assert r2._data.collection_name == "full_coll"


# ===========================================================================
# Config-driven hyper-parameters
# ===========================================================================


class TestConfigValues:
    def test_k1_from_config(self):
        cfg = StubConfig()
        cfg._DEFAULTS["_BM25_INDEX.k1"] = 1.5
        r = BM25Retriever.__new__(BM25Retriever)
        r._initialized = True
        r.cfg = cfg
        r.pretty = StubPrettyWriter()
        r._shared = StubSharedHelpers()
        r._data = _BM25IndexData()
        r._k1 = r.cfg.get_float("_BM25_INDEX.k1")
        r._b = r.cfg.get_float("_BM25_INDEX.b")
        r._rrf_k = r.cfg.get_int("_BM25_INDEX.rrf_k")
        r.perf_logger = MagicMock()
        assert r._k1 == 1.5

    def test_b_from_config(self):
        cfg = StubConfig()
        cfg._DEFAULTS["_BM25_INDEX.b"] = 0.5
        r = _make_retriever(cfg=cfg, b=cfg.get_float("_BM25_INDEX.b"))
        assert r._b == 0.5

    def test_rrf_k_from_config(self):
        cfg = StubConfig()
        cfg._DEFAULTS["_BM25_INDEX.rrf_k"] = 100
        r = _make_retriever(cfg=cfg, rrf_k=cfg.get_int("_BM25_INDEX.rrf_k"))
        assert r.rrf_k == 100
