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
  * filter_threshold        — relative-band fallback when pool max < threshold
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
        """A file with 1 chunk whose raw score × boost >= threshold should pass."""
        # threshold=0.35, boost=1.25 → 0.29 * 1.25 = 0.3625 >= 0.35 → hit
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.29, path="/docs/lonely.txt")
        result = sel.filter_threshold([chunk])
        assert chunk in result

    def test_single_chunk_file_still_misses_if_boost_insufficient(self):
        """Even with boost, if effective score < threshold the chunk is dropped."""
        # threshold=0.35, boost=1.25 → 0.20 * 1.25 = 0.25 < 0.35 → miss.
        # A sentinel at 0.60 keeps pool max >= threshold so the absolute path
        # is used (no relative-band fallback).
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        chunk = _make_chunk(score=0.20, path="/docs/lonely.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([chunk, sentinel])
        assert chunk not in result

    def test_multi_chunk_file_not_boosted(self):
        """Files with >1 chunk in the pool must not receive the boost."""
        # threshold=0.35, boost=1.25.  Both chunks from the same file with
        # raw score 0.29 — eff stays 0.29 → both miss.
        # A sentinel at 0.60 keeps pool max >= threshold so the absolute path
        # is used (no relative-band fallback).
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        c1 = _make_chunk(score=0.29, path="/docs/big.txt")
        c2 = _make_chunk(score=0.29, path="/docs/big.txt")
        sentinel = _make_chunk(score=0.60, path="/docs/sentinel.txt")
        result = sel.filter_threshold([c1, c2, sentinel])
        assert c1 not in result
        assert c2 not in result

    def test_no_boost_when_factor_is_one(self):
        """With boost=1.0 a single-chunk file below threshold must not pass."""
        # A sentinel at 0.60 keeps pool max >= threshold so the absolute path
        # is used (no relative-band fallback).
        session = _StubSession(threshold=0.35, boost=1.0)
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
        session = _StubSession(threshold=0.35, boost=1.25)
        sel = _make_selector(session)
        lonely = _make_chunk(score=0.29, path="/docs/lonely.txt")
        big1 = _make_chunk(
            score=0.50, path="/docs/big.txt"
        )  # above threshold naturally
        big2 = _make_chunk(
            score=0.29, path="/docs/big.txt"
        )  # below threshold, not boosted
        result = sel.filter_threshold([lonely, big1, big2])
        assert lonely in result  # boosted 0.29→0.3625 >= 0.35
        assert big1 in result  # naturally above 0.35
        assert big2 not in result  # 0.29 < 0.35, no boost


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
        session = _StubSession(threshold=0.35, web_threshold=0.30, boost=1.0)
        sel = _make_selector(session)
        web_chunk = _make_chunk(score=0.25, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([web_chunk])
        assert web_chunk not in result

    def test_web_chunk_not_boosted(self):
        """Web chunks should never receive the single-chunk boost."""
        # web threshold=0.40, boost=1.25: 0.29*1.25=0.3625 still < 0.40
        session = _StubSession(threshold=0.35, web_threshold=0.40, boost=1.25)
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
        session = _StubSession(threshold=0.35, boost=1.25, debug_level=10)
        sel = _make_selector(session)
        # Mix: boosted hit, plain hit, plain miss
        lonely_hit = _make_chunk(score=0.29, path="/docs/lonely.txt")
        plain_hit = _make_chunk(score=0.50, path="/docs/plain.txt")
        # Two chunks from same file — no boost, low score → misses
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


class TestFilterThresholdRelativeBand:
    """When the pool's best local score is below the absolute threshold the
    selector falls back to a relative band: accept chunks whose score is within
    _RELATIVE_THRESHOLD_FACTOR (0.75) of the pool best."""

    # Convenience: threshold=0.35, pool max=0.047 → relative thr = 0.047*0.75 ≈ 0.035
    _THRESHOLD = 0.35
    _POOL_MAX = 0.047
    _FACTOR = ScoreRankedSelector._RELATIVE_THRESHOLD_FACTOR  # 0.75

    def _relative_thr(self) -> float:
        return self._POOL_MAX * self._FACTOR

    def test_top_ranked_chunk_passes_relative_band(self):
        """The best chunk in a below-threshold pool must be kept."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        best = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        worst = _make_chunk(score=0.010, path="/docs/b.pdf")
        result = sel.filter_threshold([best, worst])
        assert best in result

    def test_chunk_just_above_relative_thr_passes(self):
        """A chunk above the relative threshold should pass."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        thr = self._relative_thr()
        chunk = _make_chunk(score=thr + 0.001, path="/docs/a.pdf")
        pool_max = _make_chunk(score=self._POOL_MAX, path="/docs/b.pdf")
        result = sel.filter_threshold([chunk, pool_max])
        assert chunk in result

    def test_chunk_below_relative_thr_misses(self):
        """A chunk below the relative threshold must still be dropped."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        thr = self._relative_thr()
        low = _make_chunk(score=thr - 0.002, path="/docs/a.pdf")
        pool_max = _make_chunk(score=self._POOL_MAX, path="/docs/b.pdf")
        result = sel.filter_threshold([low, pool_max])
        assert low not in result

    def test_empty_pool_no_crash(self):
        """An empty pool must not raise."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0)
        sel = _make_selector(session)
        result = sel.filter_threshold([])
        assert result == []

    def test_absolute_threshold_used_when_pool_max_exceeds_threshold(self):
        """When pool max >= threshold the normal absolute path must be used."""
        session = _StubSession(threshold=0.35, boost=1.0)
        sel = _make_selector(session)
        above = _make_chunk(score=0.60, path="/docs/a.pdf")
        below = _make_chunk(score=0.20, path="/docs/b.pdf")
        result = sel.filter_threshold([above, below])
        assert above in result
        assert below not in result  # 0.20 < 0.35 absolute threshold

    def test_debug_message_emitted_when_fallback_active(self):
        """At debug_level >= 10, a message about the relative band should appear."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0, debug_level=10)
        sel = _make_selector(session)
        chunk = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        sel.filter_threshold([chunk])
        messages = " ".join(str(c[0]) for c in session.pretty.calls)
        assert "relative band" in messages.lower() or "relative" in messages.lower()

    def test_no_debug_message_below_debug_level(self):
        """At debug_level=0 no relative-band message should appear."""
        session = _StubSession(threshold=self._THRESHOLD, boost=1.0, debug_level=0)
        sel = _make_selector(session)
        chunk = _make_chunk(score=self._POOL_MAX, path="/docs/a.pdf")
        sel.filter_threshold([chunk])
        messages = " ".join(str(c[0]) for c in session.pretty.calls)
        assert "relative band" not in messages.lower()

    def test_web_chunks_excluded_from_relative_band_calculation(self):
        """Web chunks must not influence max_local_score used in the fallback."""
        # Local chunks all below threshold; web chunk has high score — but
        # local_scores should still trigger the fallback.
        session = _StubSession(threshold=0.35, web_threshold=0.10, boost=1.0)
        sel = _make_selector(session)
        local = _make_chunk(score=0.040, path="/docs/a.pdf", sources="Vector")
        web = _make_chunk(score=0.90, path="/web/page.html", sources="Web")
        result = sel.filter_threshold([local, web])
        # local: relative thr = 0.040*0.75=0.030 → passes; web: 0.90 >= 0.10 → passes
        assert local in result
        assert web in result
