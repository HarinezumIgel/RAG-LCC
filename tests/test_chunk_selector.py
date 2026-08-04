# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false
"""Tests for Strategies.HomeBrewChunkSelector.

Covers:
  * ChunkSelector.__init__ — single_chunk_boost read from config
  * filter_threshold        — boost applied only when file_counts[path] == 1
  * filter_threshold        — web chunks use web_rerank_threshold (no boost)
  * filter_threshold        — boost == 1.0 disables promotion
  * filter_threshold        — boosted chunk in hits list (original score preserved)
  * filter_threshold        — non-boosted multi-chunk file stays below threshold
  * filter_threshold        — rerank-skip fallback when pool max < threshold
  * _print_final_score      — called when debug_level >= 10 (smoke test, no crash)
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Stubs — avoid real Session / PrettyWriter / Config instantiation
# ---------------------------------------------------------------------------


class _StubPretty:
    """Captures write() calls for assertions."""

    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write(self, *a, **k) -> None:
        self.calls.append((a, k))

    def last_message(self) -> str:
        if not self.calls:
            return ""
        return str(self.calls[-1][0])


class _StubConfig:
    def __init__(self, overrides: dict[str, Any] | None = None):
        # Note: SINGLE_CHUNK_SCORE_BOOST is intentionally absent from the base
        # defaults so tests can verify the "key missing → 1.0" fallback by simply
        # not including it in overrides.
        self._data: dict[str, Any] = {
            "_WEB_SEARCH.rerank_threshold": 0.40,
            "_DEFAULT_CHAT_NAME": "test",
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key: str, default: Any = None, **kw: Any) -> Any:
        return self._data.get(key, default)

    def get_float(self, key: str, default: float = 0.0, **kw: Any) -> float:
        val = self._data.get(key, default)
        return float(val) if val is not None else default

    def get_str(self, key: str, default: str = "", **kw: Any) -> str:
        val = self._data.get(key, default)
        return str(val) if val is not None else default

    def get_bool(self, key: str, default: bool = False, **kw: Any) -> bool:
        val = self._data.get(key, default)
        return bool(val) if val is not None else default


class _StubSession:
    """Minimal session with the fields ChunkSelector.__init__ reads."""

    def __init__(
        self,
        threshold: float = 0.35,
        web_threshold: float | None = None,
        boost: float = 1.25,
        debug_level: int = 0,
    ):
        self.cfg = _StubConfig(
            {
                "SINGLE_CHUNK_SCORE_BOOST": boost,
                "_WEB_SEARCH.rerank_threshold": (
                    web_threshold if web_threshold is not None else 0.40
                ),
            }
        )
        self.pretty = _StubPretty()
        self.chroma_threshold: float = threshold
        self.web_rerank_threshold: float | None = web_threshold
        self.per_file_limit: int = 10
        self.strategy: str = "DEFAULT"
        self.final_chunks_to_llm: int = 50
        self.collection_name: str = "test_col"
        self.chat_name: str = "test_chat"
        self.debug_level: int = debug_level
        self.debug_mode: str = "ge"


def _make_chunk(
    score: float,
    path: str,
    sources: str = "Vector",
    content: str = "some text",
    raw_score: float | None = None,
) -> Any:
    """Return a minimal document-like object ChunkSelector can process."""
    chunk = types.SimpleNamespace()
    chunk.score = score
    chunk.page_content = content
    chunk.metadata = {
        "FilePath": path,
        "FileName": os.path.basename(path),
        "retriever_sources": sources,
        "rerank_score": score,
        "raw_rerank_score": raw_score if raw_score is not None else score,
    }
    return chunk


# ---------------------------------------------------------------------------
# Import the concrete selector (ScoreRankedSelector) after stubs are ready
# ---------------------------------------------------------------------------

from Strategies.HomeBrewChunkSelector import (
    ChunkSelector,
    ScoreRankedSelector,
)  # noqa: E402


def _make_selector(session: _StubSession) -> ScoreRankedSelector:
    return ScoreRankedSelector(session)  # type: ignore[arg-type]


# ===========================================================================
# __init__ — boost attribute
# ===========================================================================


class TestSingleChunkBoostInit:
    def test_boost_read_from_config(self):
        sel = _make_selector(_StubSession(boost=1.5))
        assert sel.single_chunk_boost == pytest.approx(1.5)

    def test_boost_defaults_to_one_when_key_missing(self):
        session = _StubSession()
        session.cfg = _StubConfig({"_WEB_SEARCH.rerank_threshold": 0.4})  # no boost key
        sel = _make_selector(session)
        assert sel.single_chunk_boost == pytest.approx(1.0)

    def test_boost_one_point_zero_stored(self):
        sel = _make_selector(_StubSession(boost=1.0))
        assert sel.single_chunk_boost == pytest.approx(1.0)


# ===========================================================================
# filter_threshold — core boost logic
# ===========================================================================


class TestFilterThresholdBoost:
    """Boost is applied only to files that appear exactly once in the pool."""

    def test_single_chunk_file_is_boosted_past_threshold(self):
        """A file with 1 chunk whose sigmoid(raw + ln(boost)) >= threshold should pass."""
        # threshold=0.59, boost=1.25 → sigmoid(0.15 + ln(1.25)) = sigmoid(0.373) ≈ 0.592 ≥ 0.59 → hit
        # sentinel keeps pool max sigmoid above threshold so absolute path is used
        session = _StubSession(threshold=0.59, boost=1.25)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.15, path="/docs/lonely.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk in result

    def test_single_chunk_file_still_misses_if_boost_insufficient(self):
        """Even with boost, if sigmoid(raw + ln(boost)) < threshold the chunk is dropped."""
        # threshold=0.59, boost=1.25 → sigmoid(0.10 + ln(1.25)) = sigmoid(0.323) ≈ 0.580 < 0.59 → miss
        # sentinel keeps pool max sigmoid above threshold so the absolute path is used
        session = _StubSession(threshold=0.59, boost=1.25)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.10, path="/docs/lonely.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk not in result

    def test_multi_chunk_file_not_boosted(self):
        """Files with >1 chunk in the pool must not receive the boost."""
        # threshold=0.59, boost=1.25.  Both chunks from the same file with
        # raw score 0.29 — sigmoid(0.29) ≈ 0.572 < 0.59 → both miss.
        # A sentinel at 0.60 keeps pool sigmoid ≥ threshold (absolute path, no fallback).
        session = _StubSession(threshold=0.59, boost=1.25)
        sel = _make_selector(session)
        c1 = _make_chunk(score=0.29, path="/docs/big.txt")
        c2 = _make_chunk(score=0.29, path="/docs/big.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([c1, c2, sentinel])
        assert c1 not in result
        assert c2 not in result

    def test_no_boost_when_factor_is_one(self):
        """With boost=1.0 a single-chunk file below threshold must not pass."""
        # A sentinel at 0.60 keeps pool sigmoid ≥ threshold (absolute path, no fallback).
        session = _StubSession(threshold=0.59, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.29, path="/docs/lonely.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk not in result

    def test_original_score_preserved_on_hit(self):
        """The raw score on the document object is not mutated by the boost."""
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.29, path="/docs/lonely.txt")
        result = sel.filter_threshold([chunk])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.29)

    def test_mixed_pool_single_and_multi(self):
        """Single-chunk file is boosted; sibling chunks from large file are not."""
        session = _StubSession(threshold=0.59, boost=1.25)
        sel = _make_selector(session)
        lonely = _make_chunk(score=0.29, path="/docs/lonely.txt")
        big1 = _make_chunk(
            score=0.50, path="/docs/big.txt"
        )  # sigmoid(0.50)≈0.622 ≥ 0.59 naturally
        big2 = _make_chunk(
            score=0.29, path="/docs/big.txt"
        )  # sigmoid(0.29)≈0.572 < 0.59, no boost
        result = sel.filter_threshold([lonely, big1, big2])
        assert lonely in result  # sigmoid(0.29+ln(1.25)=0.513)≈0.626 ≥ 0.59
        assert big1 in result  # sigmoid(0.50)≈0.622 ≥ 0.59 naturally
        assert (
            big2 not in result
        )  # sigmoid(0.29)≈0.572 < 0.59, no boost (multi-chunk file)


# ===========================================================================
# filter_threshold — web chunks use web_rerank_threshold
# ===========================================================================


class TestFilterThresholdWebChunks:
    def test_web_chunk_uses_web_threshold(self):
        """Chunks flagged 'Web' in retriever_sources use web_rerank_threshold."""
        session = _StubSession(threshold=0.35, web_threshold=0.20, boost=1.0)
        sel = _make_selector(session)
        web_chunk = _make_chunk(score=0.25, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([web_chunk])
        assert web_chunk in result  # 0.25 >= web threshold 0.20

    def test_web_chunk_below_web_threshold_missed(self):
        session = _StubSession(threshold=0.59, web_threshold=0.60, boost=1.0)
        sel = _make_selector(session)
        web_chunk = _make_chunk(score=0.25, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([web_chunk])
        assert web_chunk not in result  # sigmoid(0.25)≈0.562 < 0.60

    def test_web_chunk_not_boosted(self):
        """Web chunks must not receive the single-chunk logit boost."""
        # web_threshold=0.60; sigmoid(0.29)≈0.572 < 0.60, no logit boost on web → miss
        session = _StubSession(threshold=0.59, web_threshold=0.60, boost=1.25)
        sel = _make_selector(session)
        web_chunk = _make_chunk(score=0.29, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([web_chunk])
        assert web_chunk not in result


# ===========================================================================
# filter_threshold — progress message
# ===========================================================================


class TestFilterThresholdMessage:
    def test_boost_info_in_message_when_active(self):
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        sel.filter_threshold([_make_chunk(score=0.50, path="/docs/a.txt")])
        messages = " ".join(str(c[0]) for c in session.pretty.calls)
        assert "single-chunk boost" in messages
        assert "1.25" in messages

    def test_no_boost_info_when_boost_is_one(self):
        session = _StubSession(threshold=0.35, boost=1.0)
        sel = _make_selector(session)
        sel.filter_threshold([_make_chunk(score=0.50, path="/docs/a.txt")])
        messages = " ".join(str(c[0]) for c in session.pretty.calls)
        assert "single-chunk boost" not in messages


# ===========================================================================
# _print_final_score — smoke test (debug path, no crash)
# ===========================================================================


class TestPrintFinalScore:
    def test_smoke_no_crash_with_boosted_and_plain_chunks(self):
        session = _StubSession(threshold=0.59, boost=1.25, debug_level=10)
        sel = _make_selector(session)
        # Mix: boosted hit, plain hit, plain miss
        lonely_hit = _make_chunk(score=0.29, path="/docs/lonely.txt")
        plain_hit = _make_chunk(score=0.50, path="/docs/plain.txt")
        # Two chunks from same file — no boost, sigmoid(0.10)≈0.525 < 0.59 → misses
        miss1 = _make_chunk(score=0.10, path="/docs/big.txt")
        miss2 = _make_chunk(score=0.10, path="/docs/big.txt")

        # Should not raise
        result = sel.filter_threshold([lonely_hit, plain_hit, miss1, miss2])
        assert lonely_hit in result
        assert plain_hit in result
        assert miss1 not in result
        assert miss2 not in result


# ===========================================================================
# filter_threshold — relative-band fallback (all-negative cross-encoder pool)
# ===========================================================================


class TestFilterThresholdRerankSkip:
    """When the pool's best local raw logit is below the threshold the
    cross-encoder is unconfident about the whole pool.  In that case reranking
    is skipped: every chunk is kept (``filter_threshold`` returns all of them)
    and the selector orders them by retrieval score instead of the logit.  An
    orange ``Rerank skipped`` message is emitted."""

    # threshold=0.60, pool max=0.047 → sigmoid(0.047)≈0.512 < 0.60 → skip fires
    _THRESHOLD = 0.60
    _POOL_MAX = 0.047

    def test_all_chunks_kept_when_pool_below_threshold(self):
        """When nothing clears the threshold, every chunk survives."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        best = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        worst = _make_chunk(score=0.010, path="/docs/b.pdf")
        result = sel.filter_threshold([best, worst])
        assert best in result
        assert worst in result

    def test_rerank_skipped_flag_set(self):
        """The _rerank_skipped flag is set when the fallback fires."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        sel.filter_threshold([chunk])
        assert sel._rerank_skipped is True

    def test_low_scoring_chunk_kept_in_skip_mode(self):
        """A chunk far below the pool max is still kept once rerank is skipped."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        low = _make_chunk(score=-5.0, path="/docs/a.pdf")
        pool_max = _make_chunk(score=self._POOL_MAX, path="/docs/b.pdf")
        result = sel.filter_threshold([low, pool_max])
        assert low in result

    def test_empty_pool_no_crash(self):
        """An empty pool must not raise and does not trigger the skip."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        result = sel.filter_threshold([])
        assert result == []
        assert sel._rerank_skipped is False

    def test_absolute_threshold_used_when_pool_max_exceeds_threshold(self):
        """When pool max sigmoid >= threshold the normal absolute path must be used."""
        session = _StubSession(threshold=0.60, boost=1.0)
        sel = _make_selector(session)
        above = _make_chunk(
            score=0.60, path="/docs/a.pdf"
        )  # sigmoid(0.60)≈0.645 ≥ 0.60
        below = _make_chunk(
            score=0.20, path="/docs/b.pdf"
        )  # sigmoid(0.20)≈0.550 < 0.60
        result = sel.filter_threshold([above, below])
        assert above in result
        assert below not in result
        assert sel._rerank_skipped is False

    def test_orange_skip_message_emitted(self):
        """An orange 'Rerank skipped' message is emitted when the fallback fires."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        sel.filter_threshold([chunk])
        messages = " ".join(str(c[0]) for c in session.pretty.calls).lower()
        assert "rerank skipped" in messages

    def test_no_skip_message_when_confident(self):
        """No skip message appears when the pool clears the threshold."""
        session = _StubSession(threshold=0.55, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(
            score=0.60, path="/docs/a.pdf"
        )  # sigmoid(0.60)≈0.645 ≥ 0.55
        sel.filter_threshold([chunk])
        messages = " ".join(str(c[0]) for c in session.pretty.calls).lower()
        assert "rerank skipped" not in messages

    def test_web_chunks_excluded_from_skip_calculation(self):
        """Web chunks must not influence the local-max used to decide the skip."""
        # Local chunk below threshold triggers the skip; both chunks are then kept.
        session = _StubSession(threshold=0.60, web_threshold=0.10, boost=1.0)
        sel = _make_selector(session)
        local = _make_chunk(score=0.040, path="/docs/a.pdf", sources="Vector")
        web = _make_chunk(score=0.90, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([local, web])
        assert local in result
        assert web in result
        assert sel._rerank_skipped is True

    def test_retrieval_order_used_in_skip_mode(self):
        """In skip mode the selector orders by retrieval (RRF) score, not logit."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        # Low logit but high RRF → must rank first once rerank is skipped.
        winner = _make_chunk(score=-2.0, path="/docs/winner.pdf")
        winner.metadata["rrf_score"] = 0.99
        loser = _make_chunk(score=self._POOL_MAX, path="/docs/loser.pdf")
        loser.metadata["rrf_score"] = 0.01
        selected = sel.select([winner, loser])
        assert selected[0] is winner


# ===========================================================================
# filter_threshold — raw logit gate vs normalized rerank_score
# ===========================================================================


class TestRawScoreFallback:
    """Threshold gate uses raw_rerank_score when present; falls back to rerank_score."""

    def test_falls_back_to_rerank_score_when_raw_absent(self):
        """When raw_rerank_score is absent, rerank_score is used for the gate."""
        session = _StubSession(threshold=0.35, boost=1.0)
        sel = _make_selector(session)
        chunk = types.SimpleNamespace()
        chunk.score = 0.50
        chunk.page_content = "text"
        chunk.metadata = {
            "FilePath": "/docs/a.txt",
            "FileName": "a.txt",
            "retriever_sources": "Vector",
            "rerank_score": 0.50,
            # raw_rerank_score deliberately absent — fallback path
        }
        sentinel = _make_chunk(score=0.60, path="/docs/s.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk in result  # 0.50 >= 0.35 via fallback

    def test_raw_logit_below_threshold_drops_chunk_despite_high_normalized_score(self):
        """High rerank_score must not override a low raw logit at the gate."""
        # rerank_score=0.80 would pass a permissive threshold,
        # but sigmoid(raw_rerank_score=0.20)=0.550 < 0.60 → dropped.
        session = _StubSession(threshold=0.60, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.80, path="/docs/a.txt", raw_score=0.20)
        sentinel = _make_chunk(score=0.60, path="/docs/s.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk not in result  # sigmoid(0.20)≈0.550 < threshold 0.60

    def test_high_raw_logit_passes_despite_low_normalized_score(self):
        """A high raw logit passes even when normalized rerank_score is very low."""
        # rerank_score=0.10 would fail the old normalized threshold,
        # but raw_rerank_score=0.80 >= 0.35 → kept under the new gate.
        session = _StubSession(threshold=0.35, boost=1.0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.10, path="/docs/a.txt", raw_score=0.80)
        sentinel = _make_chunk(score=0.60, path="/docs/s.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk in result  # raw 0.80 >= threshold 0.35
