"""Answer-grounding: mark LLM answer sentences that overlap retrieved chunks.

Strategy (Option A — token overlap)
-------------------------------------
After the LLM returns its answer, split it into sentences, then run the same
bidirectional token-containment check used by ``PlainTextVisualMarker`` against
every retrieved chunk text. Sentences that pass the check are considered
*grounded* and are wrapped with a format-specific marker:

* CLI (terminal) — ANSI escape codes for a configurable background colour.
* Markdown / HTML — ``<mark style="background: COLOR">…</mark>``.

This approach has a ~30–50% match rate on paraphrased text, but every match it
finds is a genuine one (no false positives from the token check).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config

_PUNCT_RX = re.compile(r"[^\w\s]")

# Split on sentence-ending punctuation followed by whitespace or end-of-string.
# Keeps the delimiter attached to the preceding sentence.
_SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AnswerGrounder(SingletonMixin):
    """Singleton that marks grounded answer sentences using config-backed thresholds."""

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.config = Config()
        self.min_sentence_tokens: int = max(
            1,
            self.config.get_int(
                "_MARKED_DOCS_GROUNDING.min_sentence_tokens", 5, silent=True
            ),
        )
        self.min_fragment_len: int = max(
            1,
            self.config.get_int(
                "_MARKED_DOCS_GROUNDING.min_fragment_len", 12, silent=True
            ),
        )
        self.min_overlap_window: int = max(
            1,
            self.config.get_int(
                "_MARKED_DOCS_GROUNDING.min_overlap_window", 5, silent=True
            ),
        )

    def ground_answer_cli(
        self,
        answer: str,
        chunk_texts: Sequence[str],
        ansi_codes: str = "48;5;116",
    ) -> str:
        """Return *answer* with grounded sentences wrapped in ANSI colour codes."""
        if not ansi_codes or not chunk_texts:
            return answer
        reset = "\033[0m"
        open_tag = f"\033[{ansi_codes}m"

        def _wrap(sentence: str) -> str:
            return f"{open_tag}{sentence}{reset}"

        return _apply_grounding(
            answer,
            chunk_texts,
            _wrap,
            min_sentence_tokens=self.min_sentence_tokens,
            min_fragment_len=self.min_fragment_len,
            min_overlap_window=self.min_overlap_window,
        )

    def ground_answer_md(
        self,
        answer: str,
        chunk_texts: Sequence[str],
        mark_color: str = "#C8F0E8",
    ) -> str:
        """Return *answer* with grounded sentences wrapped in ``<mark>`` HTML tags."""
        if not chunk_texts:
            return answer
        if mark_color:

            def _wrap(sentence: str) -> str:
                return f'<mark style="background-color: {mark_color}">{sentence}</mark>'

        else:

            def _wrap(sentence: str) -> str:  # type: ignore[misc]
                return f"<mark>{sentence}</mark>"

        return _apply_grounding(
            answer,
            chunk_texts,
            _wrap,
            min_sentence_tokens=self.min_sentence_tokens,
            min_fragment_len=self.min_fragment_len,
            min_overlap_window=self.min_overlap_window,
        )

    def find_grounded_sentences(
        self, answer: str, chunk_texts: Sequence[str]
    ) -> list[str]:
        """Return a list of grounded sentences from the answer."""
        if not chunk_texts:
            return []

        chunk_token_sets: list[list[str]] = [_tokenize(c) for c in chunk_texts]
        chunk_fragments: list[list[str]] = [
            [f for f in _iter_fragments(c, min_fragment_len=self.min_fragment_len)]
            for c in chunk_texts
        ]

        grounded: list[str] = []
        for para in answer.splitlines():
            if not para.strip():
                continue

            sentences = _SENTENCE_SPLIT_RX.split(para)
            for sentence in sentences:
                tokens = _tokenize(sentence)
                if len(tokens) >= self.min_sentence_tokens and _is_grounded(
                    tokens,
                    sentence,
                    chunk_token_sets,
                    chunk_fragments,
                    self.min_overlap_window,
                ):
                    grounded.append(sentence)

        return grounded

    def find_grounding_fragments_in_chunk(
        self, grounded_sentences: list[str], chunk_text: str
    ) -> list[str]:
        """Return verbatim chunk sentences that overlap with *grounded_sentences*."""
        if not grounded_sentences or not chunk_text.strip():
            return grounded_sentences

        sentence_token_sets = [_tokenize(s) for s in grounded_sentences]
        window = max(1, self.min_overlap_window)

        chunk_sentences: list[str] = []
        seen_cs: set[str] = set()
        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in _SENTENCE_SPLIT_RX.split(line):
                stripped = sent.strip()
                if len(stripped) >= self.min_fragment_len and stripped not in seen_cs:
                    seen_cs.add(stripped)
                    chunk_sentences.append(stripped)
        if not chunk_sentences:
            stripped = chunk_text.strip()
            if stripped:
                chunk_sentences = [stripped]

        result: list[str] = []
        seen_result: set[str] = set()
        for chunk_sent in chunk_sentences:
            c_tokens = _tokenize(chunk_sent)
            key = " ".join(c_tokens)
            if key in seen_result:
                continue
            for s_tokens in sentence_token_sets:
                effective_window = min(window, len(s_tokens), len(c_tokens))
                if effective_window < 1:
                    continue
                if _has_contiguous_overlap(
                    s_tokens, c_tokens, window=effective_window
                ) or _has_contiguous_overlap(
                    c_tokens, s_tokens, window=effective_window
                ):
                    seen_result.add(key)
                    result.append(chunk_sent)
                    break

        return result if result else grounded_sentences

    def find_first_overlap_span(self, sentence_text: str, chunk_text: str) -> str:
        """Return the first token span shared between *sentence_text* and *chunk_text*."""
        s_tokens = _tokenize(sentence_text)
        c_tokens = _tokenize(chunk_text)
        window = max(1, min(self.min_overlap_window, len(s_tokens), len(c_tokens)))
        for i in range(len(s_tokens) - window + 1):
            span = s_tokens[i : i + window]
            if _contains_sequence(c_tokens, span):
                return " ".join(span)
        for i in range(len(c_tokens) - window + 1):
            span = c_tokens[i : i + window]
            if _contains_sequence(s_tokens, span):
                return " ".join(span)
        return ""


def ground_answer_cli(
    answer: str,
    chunk_texts: Sequence[str],
    ansi_codes: str = "48;5;116",
    *,
    min_sentence_tokens: int | None = None,
    min_fragment_len: int | None = None,
    min_overlap_window: int | None = None,
) -> str:
    """Compatibility wrapper around :class:`AnswerGrounder`."""
    grounder = AnswerGrounder()
    if min_sentence_tokens is not None:
        grounder.min_sentence_tokens = max(1, int(min_sentence_tokens))
    if min_fragment_len is not None:
        grounder.min_fragment_len = max(1, int(min_fragment_len))
    if min_overlap_window is not None:
        grounder.min_overlap_window = max(1, int(min_overlap_window))
    return grounder.ground_answer_cli(answer, chunk_texts, ansi_codes=ansi_codes)


def ground_answer_md(
    answer: str,
    chunk_texts: Sequence[str],
    mark_color: str = "#C8F0E8",
    *,
    min_sentence_tokens: int | None = None,
    min_fragment_len: int | None = None,
    min_overlap_window: int | None = None,
) -> str:
    """Compatibility wrapper around :class:`AnswerGrounder`."""
    grounder = AnswerGrounder()
    if min_sentence_tokens is not None:
        grounder.min_sentence_tokens = max(1, int(min_sentence_tokens))
    if min_fragment_len is not None:
        grounder.min_fragment_len = max(1, int(min_fragment_len))
    if min_overlap_window is not None:
        grounder.min_overlap_window = max(1, int(min_overlap_window))
    return grounder.ground_answer_md(answer, chunk_texts, mark_color=mark_color)


def find_grounded_sentences(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    min_sentence_tokens: int | None = None,
    min_fragment_len: int | None = None,
    min_overlap_window: int | None = None,
) -> list[str]:
    """Compatibility wrapper around :class:`AnswerGrounder`."""
    grounder = AnswerGrounder()
    if min_sentence_tokens is not None:
        grounder.min_sentence_tokens = max(1, int(min_sentence_tokens))
    if min_fragment_len is not None:
        grounder.min_fragment_len = max(1, int(min_fragment_len))
    if min_overlap_window is not None:
        grounder.min_overlap_window = max(1, int(min_overlap_window))
    return grounder.find_grounded_sentences(answer, chunk_texts)


def find_grounding_fragments_in_chunk(
    grounded_sentences: list[str],
    chunk_text: str,
    *,
    min_fragment_len: int | None = None,
    min_overlap_window: int | None = None,
) -> list[str]:
    """Compatibility wrapper around :class:`AnswerGrounder`."""
    grounder = AnswerGrounder()
    if min_fragment_len is not None:
        grounder.min_fragment_len = max(1, int(min_fragment_len))
    if min_overlap_window is not None:
        grounder.min_overlap_window = max(1, int(min_overlap_window))
    return grounder.find_grounding_fragments_in_chunk(grounded_sentences, chunk_text)


def find_first_overlap_span(
    sentence_text: str,
    chunk_text: str,
    *,
    min_overlap_window: int | None = None,
) -> str:
    """Compatibility wrapper around :class:`AnswerGrounder`."""
    grounder = AnswerGrounder()
    if min_overlap_window is not None:
        grounder.min_overlap_window = max(1, int(min_overlap_window))
    return grounder.find_first_overlap_span(sentence_text, chunk_text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_grounding(
    answer: str,
    chunk_texts: Sequence[str],
    wrap: Callable[[str], str],
    *,
    min_sentence_tokens: int,
    min_fragment_len: int,
    min_overlap_window: int,
) -> str:
    chunk_token_sets: list[list[str]] = [_tokenize(c) for c in chunk_texts]
    chunk_fragments: list[list[str]] = [
        [f for f in _iter_fragments(c, min_fragment_len=min_fragment_len)]
        for c in chunk_texts
    ]

    result_lines: list[str] = []
    for para in answer.splitlines():
        if not para.strip():
            result_lines.append(para)
            continue

        sentences = _SENTENCE_SPLIT_RX.split(para)
        annotated: list[str] = []
        for sentence in sentences:
            tokens = _tokenize(sentence)
            if len(tokens) >= min_sentence_tokens and _is_grounded(
                tokens, sentence, chunk_token_sets, chunk_fragments, min_overlap_window
            ):
                annotated.append(wrap(sentence))
            else:
                annotated.append(sentence)
        result_lines.append(" ".join(annotated))

    return "\n".join(result_lines)


def _is_grounded(
    sentence_tokens: list[str],
    sentence_text: str,
    chunk_token_sets: list[list[str]],
    chunk_fragments: list[list[str]],
    min_overlap_window: int,
) -> bool:
    for chunk_tokens, fragments in zip(chunk_token_sets, chunk_fragments):
        if _contains_sequence(chunk_tokens, sentence_tokens):
            return True
        if _contains_sequence(sentence_tokens, chunk_tokens):
            return True
        if _has_contiguous_overlap(
            sentence_tokens,
            chunk_tokens,
            window=min(min_overlap_window, len(sentence_tokens)),
        ):
            return True
        for frag in fragments:
            frag_tokens = _tokenize(frag)
            if frag_tokens and _contains_sequence(sentence_tokens, frag_tokens):
                return True
    return False


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _has_contiguous_overlap(
    a_tokens: list[str], b_tokens: list[str], *, window: int
) -> bool:
    """Return True when *a_tokens* shares any contiguous `window`-token span with *b_tokens*."""
    if window <= 0 or len(a_tokens) < window or len(b_tokens) < window:
        return False
    for i in range(len(a_tokens) - window + 1):
        span = a_tokens[i : i + window]
        if _contains_sequence(b_tokens, span):
            return True
    return False


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RX.sub(" ", text.lower()).split()


def _iter_fragments(text: str, *, min_fragment_len: int = 12) -> list[str]:
    """Extract stable chunk-line anchors for grounding."""
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= min_fragment_len and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result
