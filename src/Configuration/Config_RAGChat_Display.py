from typing import Any

from Gui.Colors import \
    MARKED_DOCS_ANSWER_ANSI_COLOR as _DEFAULT_ANSWER_ANSI_COLOR
from Gui.Colors import \
    MARKED_DOCS_ANSWER_MARK_COLOR as _DEFAULT_ANSWER_MARK_COLOR
from Gui.Colors import MARKED_DOCS_HIGHLIGHT_COLOR as _DEFAULT_HIGHLIGHT_COLOR

TERMINAL_LINE_SIZE = {
    "debug": 180,
    "no_debug": 100,
}

# Multiplicative score boost applied during threshold filtering to chunks whose
# source file appears only once in the reranked candidate pool.  Such files
# have a single shot at clearing the threshold while large documents may
# contribute dozens of chunks; this levels the playing field.
# Set to 1.0 (or remove) to disable.  Default: 1.25
SINGLE_CHUNK_SCORE_BOOST = 1.25

# -----------------------------------------------------------------------------
# Visual-marker colours
# Canonical defaults live in Gui/Colors.py.
#
# _MARKED_DOCS_COLORS drives all visual-marker colours in one place:
# - highlight: source-document chunk highlight colour
# - answer_mark: grounded/effective answer spans in Markdown / HTML
# - answer_ansi: grounded/effective answer spans in CLI (ANSI SGR parameters)
# -----------------------------------------------------------------------------
_MARKED_DOCS_COLORS: dict[str, str] = {
    "highlight": _DEFAULT_HIGHLIGHT_COLOR,
    "answer_mark": _DEFAULT_ANSWER_MARK_COLOR,
    "answer_ansi": _DEFAULT_ANSWER_ANSI_COLOR,
}

# -----------------------------------------------------------------------------
# Answer grounding sensitivity (applies to both RAGChat and RAGChatService)
#
# Grounding marks answer sentences as "effective" only when they overlap
# retrieved chunk text. Tuning these values changes strictness:
# - Lower min_sentence_tokens: more short sentences can be marked.
# - Lower min_fragment_len: shorter chunk lines can anchor a match.
# - Lower min_overlap_window: requires shorter contiguous overlap for paraphrases.
# -----------------------------------------------------------------------------
_MARKED_DOCS_GROUNDING: dict[str, Any] = {
    "min_sentence_tokens": 5,
    "min_fragment_len": 12,
    "min_overlap_window": 5,  # Contiguous token window for paraphrase grounding
}
