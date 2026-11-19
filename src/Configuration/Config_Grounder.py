"""
Configuration for answer grounding, source-document marking, and summary budgets.
"""

from typing import Any

# Maximum requested words per summary call. Intermediate applies to both chunk
# batches and recursive reductions; final applies only to the last document merge.
SUMMARY_WORD_LIMITS: dict[str, dict[str, int]] = {
    "short": {"intermediate": 150, "final": 400},
    "medium": {"intermediate": 300, "final": 1000},
    "long": {"intermediate": 600, "final": 2000},
    "very_long": {"intermediate": 1000, "final": 4000},
}

# -----------------------------------------------------------------------------
# Answer grounding sensitivity (applies to both RAGChat and RAGChatService)
#
# Validation tries direct overlap first, then an internal answer translation
# into selected evidence languages. The displayed answer language is unchanged.
# Retrieval language is used only for evidence lacking language metadata.
# Web evidence without language metadata is language-detected first.
# Every prose claim must pass; lists and Mermaid need one supported claim.
# Short prose/list claims are validated even below the highlighting threshold.
# Debug level 30 shows the pre-validation answer and translation routes;
# debug level 33 shows per-chunk matches and grounding/marking timings.
# Source-document highlights are prepared only when the original answer has
# matching spans in local evidence; translated validation alone is insufficient.
#
# Grounding marks answer sentences as "effective" only when they overlap
# retrieved chunk text. Tuning these values changes strictness:
# - Lower min_sentence_tokens: more short sentences can be marked.
# - Lower min_fragment_len: shorter chunk lines can anchor a match.
# - Lower min_overlap_window: requires shorter contiguous overlap for paraphrases.
#
# The relaxed_* values control the second-pass evidence check for paraphrased
# or translated answers (unique tokens of at least relaxed_min_token_len chars):
# a sentence with at least relaxed_min_sentence_tokens tokens is supported when
# it shares relaxed_min_shared_tokens tokens with one evidence text, one of them
# is at least relaxed_substantive_token_len chars long, and the shared tokens
# cover relaxed_min_containment of the smaller token set.
#
# index_cache_max_tokens bounds the per-chunk token/n-gram index cache that lets
# repeated grounding passes in one turn (validation, CLI marks, source marks)
# reuse tokenization. Larger values keep more chunks of big summaries cached.
# -----------------------------------------------------------------------------
_GROUNDING: dict[str, Any] = {
    "min_sentence_tokens": 5,
    "min_fragment_len": 12,
    "min_overlap_window": 5,  # Contiguous token window for paraphrase grounding
    "relaxed_min_sentence_tokens": 4,
    "relaxed_min_shared_tokens": 3,
    "relaxed_min_token_len": 3,
    "relaxed_substantive_token_len": 5,
    "relaxed_min_containment": 0.60,
    "index_cache_max_tokens": 500_000,
}

# -----------------------------------------------------------------------------
# Source-document marking
#
# PDF marking is CPU-bound pure Python (pdfplumber/pypdf), so several PDFs are
# marked in parallel in separate lightweight worker processes. A worker only
# imports VisualMarkers, but still costs roughly a second to start, so it is
# used only when at least two PDFs are marked and the estimated number of pages
# to scan reaches min_scan_pages (a snippet without page number counts as a
# full-document scan). max_workers <= 1 disables parallel marking.
# A failed worker falls back to in-process marking for that file.
# -----------------------------------------------------------------------------
_GROUNDING_MARKING: dict[str, Any] = {
    # Summary paraphrases must overlap a bounded source passage, not a whole chunk.
    "passage_tokens": 48,
    "relaxed_min_shared_tokens": 4,
    "relaxed_min_answer_coverage": 0.60,
    # Summaries: mark only matching source passages, not every input chunk.
    # False restores yellow input-chunk highlights plus orange support passages.
    # Normal question-answer highlighting is unaffected.
    "summary_support_only": True,
    "max_workers": 4,
    "min_scan_pages": 12,
    "worker_timeout_s": 300,
}
