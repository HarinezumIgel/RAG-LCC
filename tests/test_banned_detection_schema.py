# pyright: reportPrivateUsage=false
"""Regression tests for banned-detection schema constants."""

from Configuration import Config_Banned_Detection as banned_cfg
from Configuration import Config_Banned_HumanReview_CSV as csv_cfg
from Helpers.CSVWriter import build_human_review_csv_keys


def test_human_review_keys_expand_to_expected_order_and_names() -> None:
    expanded = build_human_review_csv_keys(
        list(csv_cfg._HUMAN_REVIEW_PREFIX_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_ALGO_BASE_COLUMNS),
        list(csv_cfg._HUMAN_REVIEW_SUFFIX_COLUMNS),
    )

    assert expanded == [
        "Status",
        "Time",
        "Stage",
        "Skip Status",
        "Skipped Chunks",
        "Inserted Chunks",
        "Phrase",
        "Max Score",
        "Matched Algos Count",
        "Algos Matched",
        "Jaccard",
        "Score Jaccard",
        "Threshold Jaccard",
        "Regex+Levenshtein",
        "Score Regex+Levenshtein",
        "Threshold Regex+Levenshtein",
        "BM25",
        "Score BM25",
        "Threshold BM25",
        "Keybert",
        "Score Keybert",
        "Threshold Keybert",
        "WordCount",
        "Temperature",
        "FilePath",
        "FileType",
        "Language",
        "CreationDate",
        "Chunk",
        "FileHash",
    ]

    assert csv_cfg._KEYS_FOR_HUMAN_REVIEW_CSV == []


def test_label_alias_regex_family_is_stable() -> None:
    assert csv_cfg._LABEL_ALIAS["Regex"] == "Regex+Levenshtein"
    assert csv_cfg._LABEL_ALIAS["Score Regex"] == "Score Regex+Levenshtein"
    assert csv_cfg._LABEL_ALIAS["Threshold Regex"] == "Threshold Regex+Levenshtein"
    assert csv_cfg._LABEL_ALIAS["Detail Regex"] == "Details Regex+Levenshtein"


def test_default_algos_order_is_stable() -> None:
    assert banned_cfg._DEFAULT_ALGOS == ["Jaccard", "BM25", "Regex", "Keybert"]
