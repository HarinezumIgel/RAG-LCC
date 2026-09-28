# pyright: reportUnusedVariable=false, reportUnknownVariableType=false
# -------------------------------------------------------------------------
# HUMAN_REVIEW CSV presentation config.
#
# This module intentionally holds CSV/output-oriented constants only so
# detection algorithm/pipeline config in Config_Banned_Detection.py stays
# focused on detection behavior.
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Human-friendly alias mapping used for CSV headers and CLI summaries.
# This ensures consistent column names when Regex and Levenshtein are
# reported together as a single combined label.
# ---------------------------------------------------------------------
_LABEL_ALIAS = {
    "Regex": "Regex+Levenshtein",
    "Score Regex": "Score Regex+Levenshtein",
    "Threshold Regex": "Threshold Regex+Levenshtein",
    "Detail Regex": "Details Regex+Levenshtein",
}

# ---------------------------------------------------------------------
# Base HUMAN_REVIEW CSV schema pieces.
# Runtime code expands algo base labels into [algo, score, threshold]
# triplets to build final HUMAN_REVIEW field order.
# ---------------------------------------------------------------------
_HUMAN_REVIEW_PREFIX_COLUMNS = [
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
]

_HUMAN_REVIEW_ALGO_BASE_COLUMNS = [
    "Jaccard",
    "Regex+Levenshtein",
    "BM25",
    "Keybert",
    # "Cosine",
]

_HUMAN_REVIEW_SUFFIX_COLUMNS = [
    "WordCount",
    "Temperature",
    "FilePath",
    "FileType",
    "Language",
    "CreationDate",
    "Chunk",
    "FileHash",
]

# Optional explicit override slot.
# Keep empty to let runtime derive full key order from base lists above.
_KEYS_FOR_HUMAN_REVIEW_CSV: list[str] = []
