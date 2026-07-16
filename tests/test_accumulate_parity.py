# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportUnusedVariable=false
"""
Parity tests: accumulate=True + show_accumulated vs. accumulate=False
on a single-chunk input with the same config must yield identical results.

Background
----------
- DocClassify feeds KeyBERT n-grams; RAGLoad feeds document-text chunks.
- run_ensemble_checks(accumulate=False)  →  add_results + show_accumulated
  immediately (single-chunk, presentation-ready ResultsForPrint).
- run_ensemble_checks(accumulate=True)   →  add_results only; caller invokes
  show_accumulated after the loop.
  For a *single* chunk the two paths must produce the same human_review
  boolean and the same set of phrases / algo scores in the output.

Why they *can* diverge on multi-chunk documents
------------------------------------------------
- show_accumulated merges across all accumulated chunks: best score per algo
  is picked across chunks, breadth counts are unioned.  A phrase that fails
  threshold in every individual chunk may appear as passing in the merged view
  if different algos pass in different chunks.
- In the non-accumulate path each call is isolated (buffers cleared after each
  show_accumulated call), so cross-chunk aggregation never happens.
"""

import pytest
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Any
from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ResultsForPrint
from Helpers.Accumulator import Accumulator

# ---------------------------------------------------------------------------
# Stubs — mirrors from test_accumulator.py / test_ensemble_di.py
# ---------------------------------------------------------------------------

_ALGOS = OrderedDict(
    [
        ("REGEX", True),
        ("JACCARD", True),
        ("BM25", True),
    ]
)


class StubConfig:
    """Config stub that serves both Accumulator and show_accumulated paths."""

    def __init__(self, required_depth: int = 2, required_breadth: int = 2):
        self._depth = required_depth
        self._breadth = required_breadth

    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_KEYBERT": "KEYBERT",
            "_JACCARD": "JACCARD",
            "_REGEX": "REGEX",
            "_BM25": "BM25",
            "_COSINE": "COSINE",
            "_LEVENSHTEIN": "LEVENSHTEIN",
            "_DEFAULT_ALGOS": ["REGEX", "JACCARD", "BM25"],
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.ALGOS_TO_PROCESS": _ALGOS,
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD": self._depth,
            "_BANNED_DETECT.TEST.SITE.PIPELINE_CHECK.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": self._breadth,
        }
        if key in mapping:
            return mapping[key]
        return default

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

    def make_ordered_dict(self, raw):
        if isinstance(raw, dict):
            return OrderedDict((str(k), bool(v)) for k, v in raw.items())
        return OrderedDict()

    def get_compliance_config_slot(self, stage: str) -> str:
        return f"_BANNED_DETECT.TEST.SITE.{stage}"

    def get_label_alias(self, algo: str) -> str:
        return algo


class StubAIHelpers:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset():
    Accumulator._reset()
    yield
    Accumulator._reset()


def _make_accumulator(
    required_depth: int = 2, required_breadth: int = 2
) -> Accumulator:
    """Build a real Accumulator with stubbed deps (no heavy __init__)."""
    a = Accumulator.__new__(Accumulator)
    a._initialized = True
    a.raw_results = []
    a.per_chunk_decisions = []
    a.per_chunk_filtered = []
    a.per_chunk_phrase_hits = []
    a.cfg = StubConfig(required_depth, required_breadth)
    a.helpers = StubHelpers()
    a.pretty = StubPrettyWriter()
    a.default_algos = ["REGEX", "JACCARD", "BM25"]
    a.keybert = "KEYBERT"
    a.jaccard = "JACCARD"
    a.regex = "REGEX"
    a.cosine = "COSINE"
    a.levenshtein = "LEVENSHTEIN"
    return a


def _extract_phrase_algo_scores(
    results: list[ResultsForPrint],
) -> dict[tuple[str, str], float | None]:
    """Build a {(phrase, algo): score} dict for comparison."""
    return {(r.phrase, r.algo): r.score for r in results if r.algo}


# ---------------------------------------------------------------------------
# Helpers: run the two paths and return comparable outputs
# ---------------------------------------------------------------------------


def _run_non_accumulate(
    results: list[ComplianceAlgoResult],
    stage: str = "PIPELINE_CHECK",
    required_depth: int = 2,
    required_breadth: int = 2,
) -> tuple[bool, list[ResultsForPrint]]:
    """Simulate the DocClassify path: accumulate=False (add_results + show_accumulated)."""
    acc = _make_accumulator(required_depth, required_breadth)
    acc.add_results(results, stage)
    human_review, phrase_table = acc.show_accumulated(stage)
    return human_review, phrase_table


def _run_accumulate_then_show(
    results: list[ComplianceAlgoResult],
    stage: str = "PIPELINE_CHECK",
    required_depth: int = 2,
    required_breadth: int = 2,
) -> tuple[bool, list[ResultsForPrint]]:
    """Simulate the RAGLoad path on a single chunk: accumulate=True, then show_accumulated."""
    acc = _make_accumulator(required_depth, required_breadth)
    acc.add_results(results, stage)
    human_review, phrase_table = acc.show_accumulated(stage)
    return human_review, phrase_table


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleChunkParity:
    """For a single chunk the two paths must produce identical output."""

    def test_empty_results(self):
        """Both paths agree on empty input."""
        hr_a, pt_a = _run_accumulate_then_show([])
        hr_b, pt_b = _run_non_accumulate([])
        assert hr_a == hr_b
        assert pt_a == pt_b

    def test_depth_trigger_parity(self):
        """Two algos above threshold -> both paths agree on human_review and phrases."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.7, threshold=0.5
            ),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert (
            hr_a == hr_b
        ), f"human_review mismatch: accumulate={hr_a}, non-accumulate={hr_b}"
        assert hr_a is True

        scores_a = _extract_phrase_algo_scores(pt_a)
        scores_b = _extract_phrase_algo_scores(pt_b)
        assert (
            scores_a == scores_b
        ), f"score mismatch:\n  accumulate:     {scores_a}\n  non-accumulate: {scores_b}"

    def test_breadth_trigger_parity(self):
        """Two algos score >0 for same phrase (below threshold) -> breadth triggers."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.3, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.2, threshold=0.5
            ),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert (
            hr_a == hr_b
        ), f"human_review mismatch: accumulate={hr_a}, non-accumulate={hr_b}"

        phrases_a = {r.phrase for r in pt_a}
        phrases_b = {r.phrase for r in pt_b}
        assert (
            phrases_a == phrases_b
        ), f"phrase mismatch:\n  accumulate:     {phrases_a}\n  non-accumulate: {phrases_b}"

    def test_no_trigger_parity(self):
        """Single algo, no breadth, no depth -> both paths agree no review."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9, threshold=0.5
            ),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert hr_a == hr_b
        assert hr_a is False

    def test_mixed_phrases_parity(self):
        """Multiple phrases, some triggering and some not."""
        results = [
            # "password": 2 algos above threshold -> depth trigger
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="BM25", phrase="password", score=0.8, threshold=0.5
            ),
            # "secret": 1 algo above threshold, 1 below -> breadth trigger only
            ComplianceAlgoResult(
                algo="JACCARD", phrase="secret", score=0.6, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="REGEX", phrase="secret", score=0.3, threshold=0.5
            ),
            # "benign": only 1 algo with a score -> no trigger
            ComplianceAlgoResult(
                algo="BM25", phrase="benign", score=0.1, threshold=0.5
            ),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert (
            hr_a == hr_b
        ), f"human_review mismatch: accumulate={hr_a}, non-accumulate={hr_b}"

        scores_a = _extract_phrase_algo_scores(pt_a)
        scores_b = _extract_phrase_algo_scores(pt_b)
        assert (
            scores_a == scores_b
        ), f"score mismatch:\n  accumulate:     {scores_a}\n  non-accumulate: {scores_b}"

    def test_all_below_threshold_breadth_only_parity(self):
        """All scores below threshold but 3 algos score >0 for same phrase -> breadth only."""
        results = [
            ComplianceAlgoResult(algo="REGEX", phrase="ssn", score=0.1, threshold=0.5),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="ssn", score=0.2, threshold=0.5
            ),
            ComplianceAlgoResult(algo="BM25", phrase="ssn", score=0.15, threshold=0.5),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert hr_a == hr_b
        phrases_a = {r.phrase for r in pt_a}
        phrases_b = {r.phrase for r in pt_b}
        assert phrases_a == phrases_b

    def test_score_values_match_exactly(self):
        """Verify that individual score values are identical, not just phrase sets."""
        results = [
            ComplianceAlgoResult(
                algo="REGEX", phrase="password", score=0.9123, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="JACCARD", phrase="password", score=0.7456, threshold=0.5
            ),
            ComplianceAlgoResult(
                algo="BM25", phrase="password", score=0.3, threshold=0.5
            ),
        ]
        hr_a, pt_a = _run_accumulate_then_show(results)
        hr_b, pt_b = _run_non_accumulate(results)

        assert hr_a == hr_b

        for r_a in pt_a:
            matching = [
                r_b for r_b in pt_b if r_b.phrase == r_a.phrase and r_b.algo == r_a.algo
            ]
            assert (
                len(matching) == 1
            ), f"No match for ({r_a.phrase}, {r_a.algo}) in non-accumulate"
            r_b = matching[0]
            assert r_a.score == r_b.score, (
                f"Score mismatch for ({r_a.phrase}, {r_a.algo}): "
                f"accumulate={r_a.score}, non-accumulate={r_b.score}"
            )
            assert r_a.threshold == r_b.threshold, (
                f"Threshold mismatch for ({r_a.phrase}, {r_a.algo}): "
                f"accumulate={r_a.threshold}, non-accumulate={r_b.threshold}"
            )
