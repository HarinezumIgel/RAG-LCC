# pyright: reportPrivateUsage=false
"""Regression tests for collector alias and HUMAN_REVIEW schema behavior."""

from typing import Any

from Algos.ComplianceAlgoResult import ResultsForPrint
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Configuration import Config_Banned_Detection as banned_cfg
from Configuration import Config_Banned_HumanReview_CSV as csv_cfg
from Helpers.CSVWriter import build_human_review_csv_keys


class _StubHelpers:
    def __init__(self) -> None:
        self.alias_map: dict[str, str] = dict(csv_cfg._LABEL_ALIAS)

    def replace_keys_with_aliases(self, input: dict[str, Any]) -> dict[str, Any]:
        return {self.alias_map.get(k, k): v for k, v in input.items()}


def _build_collector(default_algos: list[str]) -> BannedPhraseCollector:
    expected_keys = build_human_review_csv_keys(
        list(csv_cfg._HUMAN_REVIEW_PREFIX_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_ALGO_BASE_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_SUFFIX_COLUMNS),
    )
    collector = object.__new__(BannedPhraseCollector)
    collector.helpers = _StubHelpers()
    collector.default_algos = list(default_algos)
    collector.conf_default_metadata_keys = expected_keys
    return collector


def _mk_result(
    *,
    algo: str,
    phrase: str,
    score: float,
    threshold: float,
    detail: str,
    matched_algos_count: int,
    algos_matched: str,
) -> ResultsForPrint:
    return ResultsForPrint(
        algo=algo,
        phrase=phrase,
        score=score,
        score_str=f"{score:.4f}/{threshold:.4f}",
        threshold=threshold,
        detail=detail,
        matched_algos_count=matched_algos_count,
        algos_matched=algos_matched,
    )


def test_prepare_for_csv_print_applies_regex_alias_keys() -> None:
    collector = _build_collector([banned_cfg._REGEX, banned_cfg._JACCARD])
    rows = [
        _mk_result(
            algo=banned_cfg._REGEX,
            phrase="api key",
            score=1.0,
            threshold=0.5,
            detail="regex hit",
            matched_algos_count=1,
            algos_matched="1/2 1/2",
        )
    ]

    out = collector.prepare_for_csv_print(rows, {"Status": "NOT_OK"})
    row = out[0]

    assert "Regex+Levenshtein" in row
    assert "Score Regex+Levenshtein" in row
    assert "Threshold Regex+Levenshtein" in row
    assert "Regex" not in row
    assert "Score Regex" not in row
    assert "Threshold Regex" not in row


def test_prepare_for_csv_print_keeps_metadata_projection_compatible() -> None:
    expected_keys = build_human_review_csv_keys(
        list(csv_cfg._HUMAN_REVIEW_PREFIX_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_ALGO_BASE_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_SUFFIX_COLUMNS),
    )

    collector = _build_collector([banned_cfg._REGEX, banned_cfg._BM25])
    rows = [
        _mk_result(
            algo=banned_cfg._REGEX,
            phrase="secret",
            score=0.9,
            threshold=0.5,
            detail="regex hit",
            matched_algos_count=1,
            algos_matched="1/2 1/2",
        )
    ]

    out = collector.prepare_for_csv_print(
        rows,
        {
            "Status": "NOT_OK",
            "Stage": "PIPELINE_CHECK",
            "Time": "2026-09-25T12:00:00",
            "FilePath": "D:/tmp/file.txt",
        },
    )
    row = out[0]

    projected = {key: row.get(key, "") for key in expected_keys}

    assert list(projected.keys()) == expected_keys
    assert projected["Status"] == "NOT_OK"
    assert projected["Stage"] == "PIPELINE_CHECK"
    assert projected["Phrase"] == "secret"
    assert projected["Regex+Levenshtein"] == "DEPTH"
    assert projected["Score Regex+Levenshtein"] == "0.9000"


def test_prepare_for_csv_print_preserves_blank_cells_for_missing_algo_hits() -> None:
    collector = _build_collector([banned_cfg._REGEX, banned_cfg._BM25])
    rows = [
        _mk_result(
            algo=banned_cfg._REGEX,
            phrase="token",
            score=0.4,
            threshold=0.5,
            detail="regex partial",
            matched_algos_count=1,
            algos_matched="0/2 1/2",
        )
    ]

    out = collector.prepare_for_csv_print(rows)
    row = out[0]

    assert row["Regex+Levenshtein"] == "BREADTH"
    assert row["BM25"] == ""
    assert row["Score BM25"] == ""
    assert row["Threshold BM25"] == ""
