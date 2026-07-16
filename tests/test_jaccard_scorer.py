# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for JaccardScorer.return_algo_result — pure scoring logic with pre-set state.
Uses SharedHelpers.jaccard / containment (static, pure) under the hood.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.JaccardScorer import JaccardScorer
from Compliance.SharedHelpers import SharedHelpers

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_JACCARD": "JACCARD",
            "_LEET_MAP": {},
            "_CONFUSABLES": {},
        }
        return mapping.get(key, default)

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
        pass


class StubAIHelpers:
    def get_banned_phrases_config_slot(self):
        return ["password", "secret key"]

    def get_compliance_config_slot(self, stage):
        return f"_BANNED_DETECT.TEST.SITE.{stage}"


@pytest.fixture(autouse=True)
def reset():
    SharedHelpers._reset()
    yield
    SharedHelpers._reset()


@pytest.fixture
def scorer():
    """Build JaccardScorer with pre-set state (bypass heavy __init__)."""
    s = JaccardScorer.__new__(JaccardScorer)
    sh = SharedHelpers(cfg=StubConfig(), pretty=StubPrettyWriter())
    s.sharedHelpers = sh
    s.cfg = StubConfig()
    s.pretty = StubPrettyWriter()
    s.helpers = StubHelpers()
    s.algo = "JACCARD"
    s.threshold = 0.5
    s.threshold_min = 0.0
    return s


# ---------------------------------------------------------------------------
# return_algo_result — pure scoring on pre-set state
# ---------------------------------------------------------------------------


class TestReturnAlgoResult:
    def test_exact_match_gives_score_1(self, scorer):
        scorer.prepared_banlist = [
            {
                "phrase": "password",
                "toks": ["password"],
                "char_grams": ["pa", "as", "ss", "sw", "wo", "or", "rd"],
            },
        ]
        scorer.toks_text = ["password"]
        scorer.char_grams_text = ["pa", "as", "ss", "sw", "wo", "or", "rd"]

        results = scorer.return_algo_result()
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].algo == "JACCARD"

    def test_no_overlap_gives_zero(self, scorer):
        scorer.prepared_banlist = [
            {"phrase": "password", "toks": ["password"], "char_grams": ["pa", "as"]},
        ]
        scorer.toks_text = ["hello", "world"]
        scorer.char_grams_text = ["he", "el", "ll", "lo"]

        results = scorer.return_algo_result()
        # score is 0.0 which equals threshold_min (0.0), so it IS included
        assert len(results) == 1
        assert results[0].score == 0.0

    def test_filters_below_threshold_min(self, scorer):
        scorer.threshold_min = 0.3
        scorer.prepared_banlist = [
            {"phrase": "password", "toks": ["password"], "char_grams": ["pa", "as"]},
        ]
        scorer.toks_text = ["xyz"]
        scorer.char_grams_text = ["xy", "yz"]

        results = scorer.return_algo_result()
        assert results == []

    def test_partial_overlap(self, scorer):
        scorer.threshold_min = 0.0
        # Token overlap: text has ["my", "password"], phrase has ["password"]
        # token jaccard = {password} & {password} / {my, password} | {password} = 1/2 = 0.5
        # containment = 1/1 = 1.0
        # score = max(0.5, char_jacc, 1.0) = 1.0
        scorer.prepared_banlist = [
            {"phrase": "password", "toks": ["password"], "char_grams": ["pa"]},
        ]
        scorer.toks_text = ["my", "password"]
        scorer.char_grams_text = ["my", "pa"]

        results = scorer.return_algo_result()
        assert len(results) == 1
        assert results[0].score == 1.0  # containment dominates

    def test_detail_string_format(self, scorer):
        scorer.prepared_banlist = [
            {"phrase": "test", "toks": ["test"], "char_grams": ["te", "es", "st"]},
        ]
        scorer.toks_text = ["test"]
        scorer.char_grams_text = ["te", "es", "st"]

        results = scorer.return_algo_result()
        assert "token=" in results[0].detail
        assert "char=" in results[0].detail
        assert "contain=" in results[0].detail
