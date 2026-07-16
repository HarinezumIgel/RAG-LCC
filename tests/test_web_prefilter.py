# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportUnusedVariable=false
"""Tests for Strategies.WebPreFilter.

Covers:
- _idf()        — Robertson-Spärck Jones IDF helper
- _bm25_score() — Okapi BM25 document scorer
- _cosine()     — cosine similarity helper
- WebPreFilter.bm25_prefilter()   — mini-corpus BM25 pre-filter
- WebPreFilter.cosine_prefilter() — embedding-based cosine pre-filter

All tests run completely offline.  No embedder, config file, or network call
is required.
"""

import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents.base import Document as LangchainDocument

from Strategies.WebPreFilter import WebPreFilter, _bm25_score, _cosine, _idf

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    _DEFAULTS: Dict[str, Any] = {
        "_WEB_SEARCH.bm25_pre_filter": 0.0,
        "_WEB_SEARCH.cosine_pre_filter": 0.0,
        "_BM25_INDEX.k1": 1.5,
        "_BM25_INDEX.b": 0.75,
    }

    def __init__(self, overrides: Dict[str, Any] | None = None) -> None:
        self._data = {**self._DEFAULTS, **(overrides or {})}

    def get_float(self, key: str, default: float = 0.0) -> float:
        v = self._data.get(key, default)
        return float(v) if v is not None else default

    def get_str(self, key: str, default: str = "") -> str:
        v = self._data.get(key, default)
        return str(v) if v is not None else default


class StubPretty:
    def __init__(self) -> None:
        self.messages: List[tuple] = []

    def write(self, severity, label, msg, **kwargs) -> None:
        self.messages.append((severity, label, msg))


def _make_filter(**cfg_overrides) -> WebPreFilter:
    return WebPreFilter(
        cfg=StubConfig(cfg_overrides),
        embedder=None,
        pretty=StubPretty(),
    )


def _doc(text: str, *, snippet: str | None = None) -> LangchainDocument:
    meta: Dict[str, Any] = {"Source": "Web", "chroma_score": 0.5}
    if snippet is not None:
        meta["snippet"] = snippet
    return LangchainDocument(page_content=text, metadata=meta)


# ===========================================================================
# _idf
# ===========================================================================


class TestIdf:

    def test_zero_df_gives_highest_value(self):
        """A term that never appeared has the highest IDF."""
        assert _idf(0, 10) > _idf(5, 10)

    def test_always_non_negative(self):
        for df in range(0, 11):
            assert _idf(df, 10) >= 0.0

    def test_known_value(self):
        # df=0, N=1 → log((1-0+0.5)/(0+0.5) + 1) = log(4) ≈ 1.386
        assert _idf(0, 1) == pytest.approx(math.log(4.0), rel=1e-9)

    def test_higher_df_lower_idf(self):
        assert _idf(1, 10) > _idf(5, 10) > _idf(9, 10)

    def test_all_docs_contain_term_still_non_negative(self):
        assert _idf(10, 10) >= 0.0


# ===========================================================================
# _bm25_score
# ===========================================================================


class TestBm25Score:
    # Mini corpus: 3 docs, pre-computed df counts
    _DF: Dict[str, int] = {"whale": 2, "krill": 1, "fish": 1}
    _N = 3
    _AVGDL = 5.0
    _K1 = 1.5
    _B = 0.75

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        return _bm25_score(
            query_tokens,
            doc_tokens,
            Counter(self._DF),
            self._N,
            self._AVGDL,
            self._K1,
            self._B,
        )

    def test_empty_query_returns_zero(self):
        assert self._score([], ["whale", "krill"]) == 0.0

    def test_empty_doc_returns_zero(self):
        assert self._score(["whale"], []) == 0.0

    def test_both_empty_returns_zero(self):
        assert self._score([], []) == 0.0

    def test_no_overlap_returns_zero(self):
        assert self._score(["whale"], ["cat", "dog", "sofa"]) == 0.0

    def test_overlap_gives_positive_score(self):
        assert self._score(["whale", "krill"], ["whale", "eats", "krill"]) > 0.0

    def test_more_overlap_higher_score(self):
        two_match = self._score(["whale", "krill"], ["whale", "krill", "fish"])
        one_match = self._score(["whale", "krill"], ["whale", "cat", "dog"])
        assert two_match > one_match

    def test_higher_tf_higher_score(self):
        single_tf = self._score(["whale"], ["whale", "cat"])
        double_tf = self._score(["whale"], ["whale", "whale", "cat"])
        assert double_tf > single_tf

    def test_rare_term_scores_higher_than_common(self):
        # "krill" (df=1) is rarer than "whale" (df=2) — matching krill should score higher
        krill_score = self._score(["krill"], ["krill"])
        whale_score = self._score(["whale"], ["whale"])
        assert krill_score > whale_score


# ===========================================================================
# _cosine
# ===========================================================================


class TestCosine:

    def test_identical_vectors_returns_one(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0, rel=1e-9)

    def test_orthogonal_vectors_returns_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_a_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self):
        assert _cosine([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors_returns_negative_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, rel=1e-9)

    def test_known_similarity(self):
        # [1,1] vs [1,0]: dot=1, |a|=√2, |b|=1 → 1/√2
        assert _cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(
            1.0 / math.sqrt(2), rel=1e-9
        )


# ===========================================================================
# WebPreFilter.bm25_prefilter
# ===========================================================================


class TestBm25Prefilter:

    def test_disabled_threshold_returns_same_list_object(self):
        f = _make_filter()  # bm25_pre_filter defaults to 0.0
        docs = [_doc("whale eats krill"), _doc("cat sat on mat")]
        result = f.bm25_prefilter(docs, "whale krill")
        assert result is docs

    def test_empty_docs_returns_empty(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.5})
        assert f.bm25_prefilter([], "whale") == []

    def test_empty_query_returns_same_list_object(self):
        """Empty query → no tokens → skip filtering entirely."""
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.5})
        docs = [_doc("whale eats krill")]
        result = f.bm25_prefilter(docs, "")
        assert result is docs

    def test_relevant_doc_survives(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.01})
        relevant = _doc(
            "whales consume enormous quantities of krill daily in the ocean"
        )
        result = f.bm25_prefilter([relevant], "whale krill")
        assert relevant in result

    def test_irrelevant_doc_dropped_by_high_threshold(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 999.0})
        irrelevant = _doc("the French Revolution began in 1789")
        result = f.bm25_prefilter([irrelevant], "whale krill ocean")
        assert irrelevant not in result

    def test_survivor_chroma_score_updated(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.01})
        doc = _doc("whales eat krill every day")
        doc.metadata["chroma_score"] = 0.0
        result = f.bm25_prefilter([doc], "whale krill")
        assert result[0].metadata["chroma_score"] > 0.0

    def test_drop_logs_pretty_message(self):
        pretty = StubPretty()
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.bm25_pre_filter": 999.0}),
            embedder=None,
            pretty=pretty,
        )
        docs = [_doc("the French Revolution"), _doc("Napoleon was exiled")]
        f.bm25_prefilter(docs, "whale krill ocean mammal")
        assert any("BM25 pre-filter dropped" in m for _, _, m in pretty.messages)

    def test_no_drop_no_log(self):
        pretty = StubPretty()
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.bm25_pre_filter": 0.01}),
            embedder=None,
            pretty=pretty,
        )
        f.bm25_prefilter([_doc("whales eat krill every day")], "whale krill")
        assert not any("dropped" in m for _, _, m in pretty.messages)

    def test_mixed_docs_partial_survival(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.1})
        relevant = _doc("whales consume enormous quantities of krill daily")
        irrelevant = _doc("the stock market closed lower on Tuesday")
        result = f.bm25_prefilter([relevant, irrelevant], "whale krill")
        assert relevant in result
        assert irrelevant not in result

    def test_all_pass_low_threshold(self):
        f = _make_filter(**{"_WEB_SEARCH.bm25_pre_filter": 0.001})
        docs = [_doc("whale"), _doc("krill food"), _doc("ocean whale mammal")]
        result = f.bm25_prefilter(docs, "whale krill")
        assert len(result) == 3


# ===========================================================================
# WebPreFilter.cosine_prefilter
# ===========================================================================


class TestCosinePrefilter:

    def test_disabled_threshold_returns_same_list_object(self):
        f = _make_filter()  # cosine_pre_filter defaults to 0.0
        docs = [_doc("whale text"), _doc("unrelated text")]
        result = f.cosine_prefilter(docs, "whale krill")
        assert result is docs

    def test_empty_docs_returns_empty(self):
        f = _make_filter(**{"_WEB_SEARCH.cosine_pre_filter": 0.5})
        assert f.cosine_prefilter([], "whale") == []

    def test_no_embedder_returns_same_list_object_and_logs_warning(self):
        pretty = StubPretty()
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=None,
            pretty=pretty,
        )
        docs = [_doc("whale eats krill")]
        result = f.cosine_prefilter(docs, "whale krill")
        assert result is docs
        assert any(sev == "W" for sev, _, _ in pretty.messages)

    def test_precomputed_query_vec_skips_embed_query(self):
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0]]
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        f.cosine_prefilter([_doc("whale text")], "whale", query_vec=[1.0, 0.0])
        embedder.embed_query.assert_not_called()

    def test_without_precomputed_vec_calls_embed_query(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [1.0, 0.0]
        embedder.embed_documents.return_value = [[1.0, 0.0]]
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        f.cosine_prefilter([_doc("whale text")], "whale krill")
        embedder.embed_query.assert_called_once_with("whale krill")

    def test_doc_above_threshold_kept(self):
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0]]  # cosine([1,0],[1,0])=1.0
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.8}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        doc = _doc("whale text")
        result = f.cosine_prefilter([doc], "whale", query_vec=[1.0, 0.0])
        assert doc in result

    def test_doc_below_threshold_dropped(self):
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[0.0, 1.0]]  # cosine([1,0],[0,1])=0.0
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        doc = _doc("unrelated content")
        result = f.cosine_prefilter([doc], "whale", query_vec=[1.0, 0.0])
        assert doc not in result

    def test_survivor_chroma_score_set_to_cosine_sim(self):
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0]]
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        doc = _doc("whale text")
        doc.metadata["chroma_score"] = 0.0
        result = f.cosine_prefilter([doc], "whale", query_vec=[1.0, 0.0])
        assert result[0].metadata["chroma_score"] == pytest.approx(1.0, rel=1e-9)

    def test_drop_logs_pretty_message(self):
        pretty = StubPretty()
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[0.0, 1.0]]  # orthogonal → cosine=0
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.9}),
            embedder=embedder,
            pretty=pretty,
        )
        f.cosine_prefilter([_doc("irrelevant")], "whale", query_vec=[1.0, 0.0])
        assert any("Cosine pre-filter dropped" in m for _, _, m in pretty.messages)

    def test_snippet_preferred_over_page_content_for_embedding(self):
        """Snippet text from metadata must be sent to embed_documents, not page_content."""
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0]]
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        doc = _doc("full page body text", snippet="short snippet")
        f.cosine_prefilter([doc], "query", query_vec=[1.0, 0.0])
        embedder.embed_documents.assert_called_once_with(["short snippet"])

    def test_page_content_used_when_snippet_absent(self):
        embedder = MagicMock()
        embedder.embed_documents.return_value = [[1.0, 0.0]]
        f = WebPreFilter(
            cfg=StubConfig({"_WEB_SEARCH.cosine_pre_filter": 0.5}),
            embedder=embedder,
            pretty=StubPretty(),
        )
        doc = _doc("full page body text")
        f.cosine_prefilter([doc], "query", query_vec=[1.0, 0.0])
        embedder.embed_documents.assert_called_once_with(["full page body text"])
