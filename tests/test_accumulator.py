# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportUnusedVariable=false
"""
Tests for Accumulator.add_results — evaluation of human-review logic
using DI-injected Config, AIHelpers, Helpers, and PrettyWriter.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.Accumulator import Accumulator
from Algos.ComplianceAlgoResult import ComplianceAlgoResult

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_REQUIRED_DEPTH = 2  # need >= 2 algos above threshold at chunk level
_REQUIRED_BREADTH = 2  # need >= 2 algos with any score for a phrase


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_KEYBERT": "KEYBERT",
            "_JACCARD": "JACCARD",
            "_REGEX": "REGEX",
            "_COSINE": "COSINE",
            "_LEVENSHTEIN": "LEVENSHTEIN",
            "_DEFAULT_ALGOS": ["REGEX", "JACCARD", "BM25", "COSINE", "KEYBERT"],
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD": _REQUIRED_DEPTH,
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": _REQUIRED_BREADTH,
        }
        if key in mapping:
            return mapping[key]
        if default is not None:
            return default
        return None

    def get_int(self, key, default=0):
        return int(self.get(key, default))

    def get_str(self, key, default=""):
        return str(self.get(key, default))

    def get_bool(self, key, default=False):
        return bool(self.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.get(key, default))

    def get_list(self, key, default=None):
        return self.get(key, default if default is not None else [])

    def get_dict(self, key, default=None):
        return self.get(key, default if default is not None else {})


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


class StubHelpers:
    def require_set(self, **kw):
        for k, v in kw.items():
            if v is None:
                raise ValueError(f"{k} is not set")

    def get_compliance_config_slot(self, stage: str) -> str:
        return f"_BANNED_DETECT.TEST.SITE.{stage}"


class StubAIHelpers:
    def get_compliance_config_slot(self, stage):
        return f"_BANNED_DETECT.TEST.SITE.{stage}"

    def get_banned_phrases_config_slot(self):
        return ["password", "secret"]


@pytest.fixture(autouse=True)
def reset():
    Accumulator._reset()
    yield
    Accumulator._reset()


@pytest.fixture
def acc():
    """Create an Accumulator with all deps stubbed, bypassing heavy __init__."""
    from collections import defaultdict

    a = Accumulator.__new__(Accumulator)
    a._initialized = True
    a.raw_results = []
    a.per_chunk_decisions = []
    a.per_chunk_filtered = []
    a.per_chunk_phrase_hits = []
    a.cfg = StubConfig()
    a.aiHelpers = StubAIHelpers()
    a.helpers = StubHelpers()
    a.pretty = StubPrettyWriter()
    a.default_algos = ["REGEX", "JACCARD", "BM25", "COSINE", "KEYBERT"]
    a.keybert = "KEYBERT"
    a.jaccard = "JACCARD"
    a.regex = "REGEX"
    a.cosine = "COSINE"
    a.levenshtein = "LEVENSHTEIN"
    a.debug_level = 0
    return a


# ---------------------------------------------------------------------------
# add_results
# ---------------------------------------------------------------------------


class TestAddResults:
    def test_empty_results_no_review(self, acc):
        human_review, filtered = acc.add_results([], "PIPELINE_CHECK")
        assert human_review is False
        assert filtered == []

    def test_single_algo_below_depth_threshold(self, acc):
        """One algo above its threshold, but depth requires 2 -> no review."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9, threshold=0.5
            ),
        ]
        human_review, filtered = acc.add_results(results, "PIPELINE_CHECK")
        # Only 1 algo passes threshold, need 2 -> no review (unless breadth triggers)
        # Breadth: password seen by 1 algo -> need 2 -> no breadth trigger
        assert human_review is False

    def test_two_algos_above_threshold_triggers_review(self, acc):
        """Two different algos above threshold -> depth met -> human review."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.7, threshold=0.5
            ),
        ]
        human_review, filtered = acc.add_results(results, "PIPELINE_CHECK")
        assert human_review is True
        assert len(filtered) == 2

    def test_breadth_triggers_review(self, acc):
        """Two algos score >0 for same phrase (but below threshold) -> breadth met -> review."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.3, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.2, threshold=0.5
            ),
        ]
        human_review, filtered = acc.add_results(results, "PIPELINE_CHECK")
        # Depth: 0 algos pass threshold -> no depth trigger
        # Breadth: "password" seen by 2 algos with score >0 -> >= required_breadth(2) -> trigger
        assert human_review is True

    def test_internal_state_accumulates(self, acc):
        """Multiple add_results calls accumulate internal state."""
        results1 = [
            ComplianceAlgoResult(algo="REGEX", phrase="pw", score=0.1, threshold=0.5),
        ]
        results2 = [
            ComplianceAlgoResult(algo="BM25", phrase="pw", score=0.2, threshold=0.5),
        ]
        acc.add_results(results1, "PIPELINE_CHECK")
        acc.add_results(results2, "PIPELINE_CHECK")

        assert len(acc.raw_results) == 2
        assert len(acc.per_chunk_decisions) == 2
