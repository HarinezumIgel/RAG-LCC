# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for LevenshteinScorer: pure _distance algorithm and return_algo_result logic.
Constructs scorer directly with pre-set state (no real Config/model needed).
"""

import pytest
import sys, os
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.ComplianceAlgoResult import ComplianceAlgoResult
from Algos.LevenshteinScorer import LevenshteinScorer

# ---------------------------------------------------------------------------
# Helpers — lightweight stubs so the scorer can be constructed
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_LEVENSHTEIN": "Levenshtein",
            # SharedHelpers deps
            "_LEET_MAP": {},
            "_CONFUSABLES": {},
        }
        return mapping.get(key, default)  # type: ignore[reportUnknownArgumentType]

    def get_int(self, key, default=0) -> int:
        return int(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_str(self, key, default="") -> str:
        return str(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_bool(self, key, default=False) -> bool:
        return bool(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_float(self, key, default=0.0) -> float:
        return float(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_list(self, key, default=None) -> list[object]:
        return self.get(key, default if default is not None else [])  # type: ignore[reportReturnType]

    def get_dict(self, key, default=None) -> dict[str, object]:
        return self.get(key, default if default is not None else {})  # type: ignore[reportReturnType]


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


class StubHelpers:
    def require_set(self, **kw):
        for k, v in kw.items():
            if v is None:
                raise ValueError(f"{k} is not set")


class StubAIHelpers:
    def get_compliance_config_slot(self, stage):
        return f"_BANNED_DETECT.TEST.SITE.{stage}"

    def get_banned_phrases_config_slot(self):
        return ["password", "secret"]


@pytest.fixture
def scorer(monkeypatch):
    """Build a LevenshteinScorer with stub deps injected after construction."""
    # We need the singleton singletons to be stubs during construction
    # Since LevenshteinScorer is NOT a singleton, we monkeypatch the imports
    import Config.Config as cfg_mod
    import Gui.PrettyWriter as pw_mod

    monkeypatch.setattr(cfg_mod.Config, "get", lambda self, key, *a, **kw: StubConfig().get(key, a[0] if a else None), raising=False)  # type: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(pw_mod, "PrettyWriter", StubPrettyWriter)

    s = LevenshteinScorer.__new__(LevenshteinScorer)
    s.cfg = StubConfig()  # type: ignore[reportAttributeAccessIssue]
    s.pretty = StubPrettyWriter()  # type: ignore[reportAttributeAccessIssue]
    s.aiHelpers = StubAIHelpers()  # type: ignore[reportAttributeAccessIssue]
    s.helpers = StubHelpers()  # type: ignore[reportAttributeAccessIssue]
    s.algo = "Levenshtein"
    s.threshold = 0.5
    s.compliance_results = []
    return s


# ---------------------------------------------------------------------------
# _distance — pure Wagner-Fischer
# ---------------------------------------------------------------------------


class TestDistance:
    def test_identical_strings(self, scorer):
        assert scorer._distance("hello", "hello") == 0

    def test_single_insertion(self, scorer):
        assert scorer._distance("abc", "abcd") == 1

    def test_single_deletion(self, scorer):
        assert scorer._distance("abcd", "abc") == 1

    def test_single_substitution(self, scorer):
        assert scorer._distance("cat", "bat") == 1

    def test_empty_first(self, scorer):
        assert scorer._distance("", "abc") == 3

    def test_empty_both(self, scorer):
        assert scorer._distance("", "") == 0

    def test_completely_different(self, scorer):
        assert scorer._distance("abc", "xyz") == 3

    def test_none_handling(self, scorer):
        assert scorer._distance(None, "abc") == 3
        assert scorer._distance("abc", None) == 3


# ---------------------------------------------------------------------------
# return_algo_result — uses pre-set compliance_results
# ---------------------------------------------------------------------------


class TestReturnAlgoResult:
    def test_empty_results(self, scorer):
        scorer.compliance_results = []
        assert scorer.return_algo_result() == []

    def test_zero_score_skipped(self, scorer):
        scorer.compliance_results = [
            ComplianceAlgoResult(
                algo="Regex",
                phrase="password",
                score=0.0,
                threshold=0.5,
                detail="password",
            )
        ]
        assert scorer.return_algo_result() == []

    def test_exact_match_gives_score_1(self, scorer):
        scorer.threshold = 0.5
        scorer.compliance_results = [
            ComplianceAlgoResult(
                algo="Regex",
                phrase="password",
                score=1.0,
                threshold=0.5,
                detail="password",
            )
        ]
        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].algo == "Levenshtein"

    def test_similar_match_gives_high_score(self, scorer):
        scorer.threshold = 0.5
        scorer.compliance_results = [
            ComplianceAlgoResult(
                algo="Regex",
                phrase="password",
                score=1.0,
                threshold=0.5,
                detail="p@ssword",
            )
        ]
        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        # distance("p@ssword", "password") = 1, max_len = 8, score = 1 - 1/8 = 0.875
        assert results[0].score is not None and abs(results[0].score - 0.875) < 1e-3

    def test_dissimilar_strings(self, scorer):
        scorer.threshold = 0.5
        scorer.compliance_results = [
            ComplianceAlgoResult(
                algo="Regex", phrase="password", score=1.0, threshold=0.5, detail="xyz"
            )
        ]
        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        # distance("xyz", "password") = 8, max_len = 8, score = 0.0
        assert results[0].score == 0.0
