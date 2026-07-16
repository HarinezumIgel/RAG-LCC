# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for RegexScorer.return_algo_result — pure regex matching with pre-set patterns/text.
"""

import re
import pytest
import sys, os
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.RegexScorer import RegexScorer
from Algos.ComplianceAlgoResult import ComplianceAlgoResult

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_REGEX": "REGEX",
        }
        return mapping.get(key, default)

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


@pytest.fixture
def scorer():
    """Create a RegexScorer with pre-set state (bypass heavy __init__)."""
    s = RegexScorer.__new__(RegexScorer)
    s.cfg = StubConfig()  # type: ignore[reportAttributeAccessIssue]
    s.pretty = StubPrettyWriter()  # type: ignore[reportAttributeAccessIssue]
    s.algo = "REGEX"
    s.soft_score_hard = 1.0
    s.soft_score_fuzzy = 0.75
    s.fuzzy_eval_after_hard = True
    s.threshold = 0.5
    s.threshold_min = 0.0
    s.text_norm = ""
    s.patterns = []
    return s


# ---------------------------------------------------------------------------
# return_algo_result
# ---------------------------------------------------------------------------


class TestReturnAlgoResult:
    def test_empty_patterns(self, scorer):
        scorer.patterns = []
        assert scorer.return_algo_result() == []

    def test_match_found(self, scorer):
        pattern = re.compile(r"\bpassword\b", re.IGNORECASE)
        scorer.patterns = [
            {
                "phrase": "password",
                "pattern": pattern,
                "strict_compiled": re.compile(r"\bpassword\b", re.IGNORECASE),
                "pattern_text": r"\bpassword\b",
                "meta": {
                    "type": "single_token",
                    "strict_pattern_text": r"\bpassword\b",
                },
            }
        ]
        scorer.text_norm = "my password is secret"

        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        assert results[0].algo == "REGEX"
        assert results[0].phrase == "password"
        assert results[0].score == 1.0
        assert results[0].detail == "password"

    def test_no_match_gives_zero_score(self, scorer):
        pattern = re.compile(r"\bpassword\b", re.IGNORECASE)
        scorer.patterns = [
            {
                "phrase": "password",
                "pattern": pattern,
                "pattern_text": r"\bpassword\b",
                "meta": {"type": "single_token"},
            }
        ]
        scorer.text_norm = "nothing here"

        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].detail is None

    def test_multiple_patterns_mixed_hits(self, scorer):
        scorer.patterns = [
            {
                "phrase": "password",
                "pattern": re.compile(r"\bpassword\b", re.IGNORECASE),
                "strict_compiled": re.compile(r"\bpassword\b", re.IGNORECASE),
                "pattern_text": r"\bpassword\b",
                "meta": {"type": "single_token"},
            },
            {
                "phrase": "secret",
                "pattern": re.compile(r"\bsecret\b", re.IGNORECASE),
                "strict_compiled": re.compile(r"\bsecret\b", re.IGNORECASE),
                "pattern_text": r"\bsecret\b",
                "meta": {"type": "single_token"},
            },
        ]
        scorer.text_norm = "my password only"

        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 2
        matched = {r.phrase: r.score for r in results}
        assert matched["password"] == 1.0
        assert matched["secret"] == 0.0

    def test_fuzzy_only_match_uses_soft_score_fuzzy(self, scorer):
        """When strict misses but fuzzy hits, score should be soft_score_fuzzy."""
        scorer.patterns = [
            {
                "phrase": "password",
                # fuzzy pattern: matches "passXword" (no word boundary)
                "pattern": re.compile(r"pass.{0,3}word", re.IGNORECASE),
                # strict pattern: requires exact word boundary
                "strict_compiled": re.compile(r"\bpassword\b", re.IGNORECASE),
                "pattern_text": r"pass.{0,3}word",
                "meta": {"type": "single_token"},
            },
        ]
        scorer.text_norm = "my passXword is here"

        results = cast(list[ComplianceAlgoResult], scorer.return_algo_result())
        assert len(results) == 1
        assert results[0].score == 0.75
        assert results[0].detail == "passXword"

    def test_meta_propagated(self, scorer):
        scorer.patterns = [
            {
                "phrase": "test",
                "pattern": re.compile(r"test"),
                "pattern_text": "test",
                "meta": {
                    "type": "single_token",
                    "strict_pattern_text": r"\btest\b",
                    "fuzzy_pattern_text": "test",
                },
            }
        ]
        scorer.text_norm = "this is a test"

        results = scorer.return_algo_result()
        assert results[0].meta["type"] == "single_token"
        assert "strict_pattern_text" in results[0].meta
