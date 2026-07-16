# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for BM25Scorer.return_algo_result — pure scoring logic with pre-set state.
"""

import pytest
import sys, os
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.BM25Scorer import BM25Scorer
from Algos.ComplianceAlgoResult import ComplianceAlgoResult

# ---------------------------------------------------------------------------
# Build a BM25Scorer with stub deps (bypass __init__ entirely)
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        return default

    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        return default

    def get_list(self, key: str, default: list[object] | None = None) -> list[object]:
        return default if default is not None else []

    def get_dict(
        self, key: str, default: dict[str, object] | None = None
    ) -> dict[str, object]:
        return default if default is not None else {}


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


@pytest.fixture(autouse=True)
def reset():
    BM25Scorer._reset()  # type: ignore[reportPrivateUsage]
    yield
    BM25Scorer._reset()  # type: ignore[reportPrivateUsage]


@pytest.fixture
def scorer():
    """Create a minimal BM25Scorer and set only the fields return_algo_result reads."""
    s = BM25Scorer.__new__(BM25Scorer)
    s.algo = "BM25"
    s.threshold = 0.5
    s.threshold_min = 0.0
    s.scores_norm = []
    return s


# ---------------------------------------------------------------------------
# return_algo_result
# ---------------------------------------------------------------------------


class TestReturnAlgoResult:
    def test_empty_scores(self, scorer):
        assert scorer.return_algo_result() == []

    def test_single_result_above_threshold_min(self, scorer):
        scorer.scores_norm = [("password", 1.23, 0.75)]
        results: list[ComplianceAlgoResult] = cast(
            list[ComplianceAlgoResult], scorer.return_algo_result()
        )
        assert len(results) == 1
        assert results[0].algo == "BM25"
        assert results[0].phrase == "password"
        assert results[0].score == 0.75
        assert results[0].threshold == 0.5

    def test_filters_below_threshold_min(self, scorer):
        scorer.threshold_min = 0.5
        scorer.scores_norm = [
            ("password", 1.0, 0.6),  # above
            ("secret", 0.5, 0.3),  # below
        ]
        results: list[ComplianceAlgoResult] = cast(
            list[ComplianceAlgoResult], scorer.return_algo_result()
        )
        assert len(results) == 1
        assert results[0].phrase == "password"

    def test_all_below_threshold_min(self, scorer):
        scorer.threshold_min = 0.9
        scorer.scores_norm = [("password", 1.0, 0.5)]
        results = scorer.return_algo_result()
        assert results == []

    def test_detail_contains_raw_and_norm(self, scorer):
        scorer.scores_norm = [("phrase1", 2.5, 0.8)]
        results = scorer.return_algo_result()
        assert "bm25_raw=2.500000" in results[0].detail
        assert "bm25_norm=0.800000" in results[0].detail
