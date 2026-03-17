# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for AIHelpers.run_ensemble_checks and merge_algo_results — full ensemble
orchestration with all scorers, Accumulator and Helpers stubbed via DI + attribute injection.
"""

import pytest
import sys, os
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Any
from Algos.ComplianceAlgoResult import ComplianceAlgoResult
from AI.AIHelpers import AIHelpers
from Helpers.Accumulator import Accumulator
from AI.ModelsCache import ModelsCache

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_ALGOS = OrderedDict(
    [
        ("REGEX", True),
        ("JACCARD", True),
        ("BM25", True),
    ]
)


class StubConfig:
    """Provides all config keys the ensemble + accumulator need."""

    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_KEYBERT": "KEYBERT",
            "_JACCARD": "JACCARD",
            "_REGEX": "REGEX",
            "_BM25": "BM25",
            "_COSINE": "COSINE",
            "_LEVENSHTEIN": "LEVENSHTEIN",
            "_DETECTION_CONFIG": "TEST",
            "_FRIENDLY_NAME": "SITE",
            "_DEFAULT_ALGOS": ["REGEX", "JACCARD", "BM25"],
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.ALGOS_TO_PROCESS": _ALGOS,
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 2,
        }
        if key in mapping:
            return mapping[key]
        if default is not None:
            return default
        return None

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_str(self, key: str, default: str = "") -> str:
        return str(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_list(self, key: str, default: list[object] | None = None) -> list[object]:
        return self.get(key, default if default is not None else [])  # type: ignore[return-value]

    def get_dict(
        self, key: str, default: dict[str, object] | None = None
    ) -> dict[str, object]:
        return self.get(key, default if default is not None else {})  # type: ignore[return-value]


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


class StubHelpers:
    def require_set(self, **kw):
        pass

    def make_ordered_dict(self, raw):
        if isinstance(raw, dict):
            return OrderedDict((str(k), bool(v)) for k, v in raw.items())  # type: ignore[arg-type]
        return OrderedDict()

    def get_compliance_config_slot(self, stage: str) -> str:
        return f"_BANNED_DETECT.TEST.SITE.{stage}"


class StubScorer:
    """A scorer that returns canned results."""

    def __init__(self, results=None):
        self._results = results or []

    def verify(self, *a, **kw):
        return self._results


class StubAccumulator:
    """Accumulator that tracks calls and returns canned results."""

    def __init__(self, human_review=False, filtered=None, phrase_table=None):
        self._human_review = human_review
        self._filtered = filtered or []
        self._phrase_table = phrase_table or []
        self.add_results_calls: list[tuple[list[Any], str]] = []
        self.show_accumulated_calls: list[str] = []

    def add_results(self, results: list[Any], stage: str) -> tuple[bool, list[Any]]:
        self.add_results_calls.append((results, stage))
        return self._human_review, self._filtered

    def show_accumulated(self, stage: str) -> tuple[bool, list[Any]]:
        self.show_accumulated_calls.append(stage)
        return self._human_review, self._phrase_table


@pytest.fixture(autouse=True)
def reset():
    AIHelpers._reset()  # type: ignore[reportPrivateUsage]
    Accumulator._reset()  # type: ignore[reportPrivateUsage]
    ModelsCache._reset()  # type: ignore[reportPrivateUsage]
    yield
    AIHelpers._reset()  # type: ignore[reportPrivateUsage]
    Accumulator._reset()  # type: ignore[reportPrivateUsage]
    ModelsCache._reset()  # type: ignore[reportPrivateUsage]


def _build_ai_helpers(
    regex_results=None,
    jaccard_results=None,
    bm25_results=None,
    levenshtein_results=None,
    accumulator=None,
):
    """Build an AIHelpers singleton with all deps stubbed."""
    ah = AIHelpers.__new__(AIHelpers)
    ah._initialized = True  # type: ignore[reportPrivateUsage]

    ah.cfg = StubConfig()  # type: ignore[reportAttributeAccessIssue]
    ah.pretty = StubPrettyWriter()  # type: ignore[reportAttributeAccessIssue]
    ah.helpers = StubHelpers()  # type: ignore[reportAttributeAccessIssue]
    ah.accumulator = accumulator or StubAccumulator()  # type: ignore[reportAttributeAccessIssue]

    # Algo name bindings
    ah.regex = "REGEX"
    ah.jaccard = "JACCARD"
    ah.bm25 = "BM25"
    ah.cosine = "COSINE"
    ah.keybert = "KEYBERT"

    # Scorer stubs
    ah.regexDetector = StubScorer(regex_results or [])  # type: ignore[reportAttributeAccessIssue]
    ah.jaccardScorer = StubScorer(jaccard_results or [])  # type: ignore[reportAttributeAccessIssue]
    ah.bm25Scorer = StubScorer(bm25_results or [])  # type: ignore[reportAttributeAccessIssue]
    ah.cosineScorer = StubScorer([])  # type: ignore[reportAttributeAccessIssue]
    ah.keyBertWordDetect = StubScorer([])  # type: ignore[reportAttributeAccessIssue]
    ah.levenshteinScorer = StubScorer(levenshtein_results or [])  # type: ignore[reportAttributeAccessIssue]
    ah.classifyHelper = None  # type: ignore[reportAttributeAccessIssue]
    ah.embedder = None  # type: ignore[reportAttributeAccessIssue]
    ah.tensor_helpers = None  # type: ignore[reportAttributeAccessIssue]
    ah.models_cache = None  # type: ignore[reportAttributeAccessIssue]

    return ah


# ---------------------------------------------------------------------------
# run_ensemble_checks
# ---------------------------------------------------------------------------


class TestRunEnsembleChecks:
    def test_no_results_no_review(self):
        ah = _build_ai_helpers()
        human_review, _kw_emb, _phrase_table = ah.run_ensemble_checks(
            "clean text", "en", "PIPELINE_CHECK"
        )
        assert human_review is False

    def test_results_forwarded_to_accumulator(self):
        regex_hits = [
            ComplianceAlgoResult(
                algo="REGEX",
                phrase="password",
                score=1.0,
                threshold=0.5,
                detail="password",
            ),
        ]
        jaccard_hits = [
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.8, threshold=0.5
            ),
        ]
        acc = StubAccumulator(human_review=True, phrase_table=[{"phrase": "password"}])
        ah = _build_ai_helpers(
            regex_results=regex_hits,
            jaccard_results=jaccard_hits,
            accumulator=acc,
        )
        human_review, _kw_emb, _phrase_table = ah.run_ensemble_checks(
            "my password is here", "en", "PIPELINE_CHECK"
        )
        assert human_review is True
        # Accumulator.add_results must have been called once
        assert len(acc.add_results_calls) == 1
        # The results passed should contain both jaccard and merged regex+levenshtein
        all_results = acc.add_results_calls[0][0]
        algos_in_results = {r.algo for r in all_results}
        assert "JACCARD" in algos_in_results

    def test_accumulate_mode_skips_show(self):
        """When accumulate=True, show_accumulated is NOT called."""
        acc = StubAccumulator()
        ah = _build_ai_helpers(accumulator=acc)
        ah.run_ensemble_checks("text", "en", "PIPELINE_CHECK", accumulate=True)
        assert len(acc.show_accumulated_calls) == 0

    def test_non_accumulate_calls_show(self):
        """When accumulate=False (default), show_accumulated IS called."""
        acc = StubAccumulator()
        ah = _build_ai_helpers(accumulator=acc)
        ah.run_ensemble_checks("text", "en", "PIPELINE_CHECK", accumulate=False)
        assert len(acc.show_accumulated_calls) == 1


# ---------------------------------------------------------------------------
# merge_algo_results
# ---------------------------------------------------------------------------


class TestMergeAlgoResults:
    def test_empty_inputs(self):
        ah = _build_ai_helpers()
        merged = ah.merge_algo_results([], [], merged_algo_name="REGEX")
        assert merged == []

    def test_regex_only(self):
        ah = _build_ai_helpers()
        regex = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="pw", score=0.8, threshold=0.5, detail="pw_match"
            ),
        ]
        merged = ah.merge_algo_results(regex, [], merged_algo_name="REGEX")
        assert len(merged) == 1
        assert merged[0].score == 0.8
        assert "regex:pw_match" in (merged[0].detail or "")

    def test_both_merge_scores(self):
        ah = _build_ai_helpers()
        regex = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="pw", score=0.8, threshold=0.5, detail="pw_match"
            ),
        ]
        lev = [
            ComplianceAlgoResult(
                algo="Levenshtein",
                phrase="pw",
                score=0.9,
                threshold=0.6,
                detail="lev_detail",
            ),
        ]
        merged = ah.merge_algo_results(regex, lev, merged_algo_name="REGEX")
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(1.7)
        assert merged[0].threshold == pytest.approx(1.1)
        assert "regex:pw_match" in (merged[0].detail or "")
        assert "lev:lev_detail" in (merged[0].detail or "")

    def test_different_phrases_stay_separate(self):
        ah = _build_ai_helpers()
        regex = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.8, threshold=0.5, detail="d1"
            ),
        ]
        lev = [
            ComplianceAlgoResult(
                algo="Levenshtein",
                phrase="secret",
                score=0.7,
                threshold=0.5,
                detail="d2",
            ),
        ]
        merged = ah.merge_algo_results(regex, lev, merged_algo_name="REGEX")
        assert len(merged) == 2
        phrases = {m.phrase for m in merged}
        assert phrases == {"password", "secret"}
