"""Evidence-based answer grounding: decide which sentences are supported.

Strategy (token overlap)
------------------------
Split the answer into sentences and check each sentence against every
evidence text (retrieved chunks, web snippets). A sentence is *grounded* in a
chunk when

* it shares a contiguous ``min_overlap_window``-token span with the chunk
  (shorter sentences use their full length as the window), or
* the whole chunk, or one of its lines of at least ``min_fragment_len``
  characters, occurs as a contiguous token sequence inside the sentence.

Every match is a genuine lexical overlap (no false positives from the token
check). Paraphrased or translated answers get a second, relaxed token
containment check and an optional translated *proxy* of the answer.

Speed
-----
Each chunk is tokenized once into an index of hashed n-grams and contained
token sequences. Indexes are cached (bounded by ``index_cache_max_tokens``),
so evidence validation, CLI marks and source-document marks in one turn reuse
the same tokenization. Matching uses set lookups instead of nested scans and
produces exactly the same results as a brute-force contiguous-sequence search.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Grounding.SourceReferences import SummaryStatement
from Gui.Colors import CYAN, ORANGE
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.DebugHelper import DebugHelper

if TYPE_CHECKING:
    from Globals.Session import Session

_PUNCT_RX = re.compile(r"[^\w\s]")
_WORD_RX = re.compile(r"\w+", re.UNICODE)
_LANE_WIDTH_RX = re.compile(r"\bx(?=\d+\b)", re.IGNORECASE)

# Split on sentence-ending punctuation followed by whitespace or end-of-string.
# Keeps the delimiter attached to the preceding sentence.
_SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")

Gram = tuple[str, ...]
# (text, native_lang, stage_label) -> detected language name or code.
LanguageDetector = Callable[[str, str, str], str]


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph exactly like the grounding checks do."""
    return _SENTENCE_SPLIT_RX.split(paragraph)


def tokenize(text: str) -> list[str]:
    # PDF extraction may render the same lane width as either "x16" or "x 16".
    normalized = _LANE_WIDTH_RX.sub("x ", text.lower())
    return _PUNCT_RX.sub(" ", normalized).split()


def _grams(tokens: Sequence[str], n: int) -> set[Gram]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _iter_fragments(text: str, *, min_fragment_len: int) -> list[str]:
    """Extract stable chunk-line anchors for grounding."""
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= min_fragment_len and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


class _ChunkIndex:
    """Tokens, lazily built n-gram sets and token sequences that must occur inside a sentence."""

    __slots__ = ("tokens", "contained", "_grams")

    def __init__(self, text: str, min_fragment_len: int) -> None:
        self.tokens: Gram = tuple(sys.intern(t) for t in tokenize(text))
        contained: dict[int, set[Gram]] = {}
        if self.tokens:
            contained.setdefault(len(self.tokens), set()).add(self.tokens)
        for fragment in _iter_fragments(text, min_fragment_len=min_fragment_len):
            fragment_tokens = tuple(sys.intern(t) for t in tokenize(fragment))
            if fragment_tokens:
                contained.setdefault(len(fragment_tokens), set()).add(fragment_tokens)
        self.contained = contained
        self._grams: dict[int, frozenset[Gram]] = {}

    def grams(self, n: int) -> frozenset[Gram]:
        cached = self._grams.get(n)
        if cached is None:
            cached = frozenset(_grams(self.tokens, n))
            self._grams[n] = cached
        return cached


class _SentenceProbe:
    """Keep sentence tokens and lazily cache their contiguous n-grams.

    Example: ``_SentenceProbe(tokenize("PCIe slots support expansion cards."))``
    stores ``("pcie", "slots", "support", "expansion", "cards")``.
    Calling ``grams(3)`` returns ``{("pcie", "slots", "support"),
    ("slots", "support", "expansion"), ("support", "expansion", "cards")}``;
    later calls with the same size reuse that cached set.
    """

    __slots__ = ("tokens", "_grams")

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tuple(tokens)
        self._grams: dict[int, set[Gram]] = {}

    def grams(self, n: int) -> set[Gram]:
        cached = self._grams.get(n)
        if cached is None:
            cached = _grams(self.tokens, n)
            self._grams[n] = cached
        return cached


def _supports(
    probe: _SentenceProbe, chunk: _ChunkIndex, min_overlap_window: int
) -> bool:
    """Return True when *chunk* grounds the sentence behind *probe*.

    Equivalent to: sentence inside chunk, chunk inside sentence, a shared
    contiguous window of ``min(min_overlap_window, len(sentence))`` tokens, or
    a chunk line (fragment) inside the sentence. "Sentence inside chunk" is
    implied by the shared window because the window never exceeds the sentence.
    """
    size = len(probe.tokens)
    if size == 0:
        return False
    window = min(min_overlap_window, size)
    if len(chunk.tokens) >= window and not probe.grams(window).isdisjoint(
        chunk.grams(window)
    ):
        return True
    for length, sequences in chunk.contained.items():
        if length <= size and not probe.grams(length).isdisjoint(sequences):
            return True
    return False


class _IndexCache:
    """LRU cache of chunk indexes bounded by the total number of cached tokens."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._entries: OrderedDict[tuple[int, str], _ChunkIndex] = OrderedDict()
        self._tokens = 0
        self._lock = threading.Lock()

    def get(self, text: str, min_fragment_len: int) -> _ChunkIndex:
        key = (min_fragment_len, text)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
        entry = _ChunkIndex(text, min_fragment_len)
        with self._lock:
            if key not in self._entries:
                self._entries[key] = entry
                self._tokens += max(1, len(entry.tokens))
                while self._tokens > self.max_tokens and len(self._entries) > 1:
                    _, evicted = self._entries.popitem(last=False)
                    self._tokens -= max(1, len(evicted.tokens))
        return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._tokens = 0


class Grounder(SingletonMixin):
    """Singleton for answer grounding, evidence validation and grounded source snippets."""

    def __init__(self, cfg: Config | None = None) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg = cfg or Config()
        self._index_cache = _IndexCache(
            self.cfg.get_int("_GROUNDING.index_cache_max_tokens")
        )
        self.pretty = PrettyWriter()
        self.pretty_always = PrettyWriter(always_on=True)

    # ------------------------------------------------------------------
    # Sentence matching
    # ------------------------------------------------------------------

    def _indexes(self, chunk_texts: Sequence[str]) -> list[_ChunkIndex]:
        min_fragment_len = self.cfg.get_int("_GROUNDING.min_fragment_len")
        return [self._index_cache.get(text, min_fragment_len) for text in chunk_texts]

    def clear_cache(self) -> None:
        self._index_cache.clear()

    def summary_support(
        self,
        answer: str,
        chunk_texts: Sequence[str],
        chunk_ids: Sequence[int],
        statements: Sequence[SummaryStatement] = (),
    ) -> dict[str, list[tuple[int, str]]]:
        """Locate verified, bounded source passages; context chunk IDs only narrow the search."""
        limit = self.cfg.get_int("_GROUNDING_MARKING.passage_tokens")
        if limit < 2:
            raise ValueError("_GROUNDING_MARKING.passage_tokens must be at least 2.")
        min_shared = self.cfg.get_int("_GROUNDING_MARKING.relaxed_min_shared_tokens")
        coverage = self.cfg.get_float("_GROUNDING_MARKING.relaxed_min_answer_coverage")
        window = self.cfg.get_int("_GROUNDING.min_overlap_window")
        min_fragment = self.cfg.get_int("_GROUNDING.min_fragment_len")
        min_sentence = self.cfg.get_int("_GROUNDING.min_sentence_tokens")
        generic = {
            "about",
            "after",
            "before",
            "being",
            "chapter",
            "describes",
            "discusses",
            "example",
            "examples",
            "provides",
            "their",
            "there",
            "these",
            "those",
            "using",
            "which",
            "would",
            "this",
            "that",
            "with",
            "from",
            "have",
            "into",
            "information",
            "also",
            "some",
            "more",
            "most",
            "only",
        }

        def substantive(tokens: Sequence[str]) -> set[str]:
            return {
                t
                for t in tokens
                if len(t) >= 4
                and any(char.isalpha() for char in t)
                and t not in generic
            }

        passages: dict[int, list[tuple[str, _ChunkIndex]]] = {}

        def source_passages(position: int) -> list[tuple[str, _ChunkIndex]]:
            if position in passages:
                return passages[position]
            text = re.sub(r"^\s*Page\s+\S+\s*", "", chunk_texts[position])
            candidates: list[tuple[str, _ChunkIndex]] = []
            for line in text.splitlines():
                for sentence in split_sentences(line):
                    words = list(re.finditer(r"\S+", sentence))
                    starts = list(
                        range(0, max(1, len(words) - limit + 1), max(1, limit // 2))
                    )
                    if len(words) > limit:
                        starts.append(len(words) - limit)
                    for start in dict.fromkeys(starts):
                        if not words:
                            continue
                        end = min(start + limit, len(words))
                        passage = sentence[words[start].start() : words[end - 1].end()]
                        candidates.append(
                            (passage, self._index_cache.get(passage, min_fragment))
                        )
            passages[position] = candidates
            return candidates

        result: dict[str, list[tuple[int, str]]] = {}
        references_by_text: dict[str, set[int]] = {}
        for statement in statements:
            for line in statement.text.splitlines():
                for sentence in split_sentences(line):
                    references_by_text.setdefault(sentence, set()).update(
                        statement.context_chunk_ids
                    )
        for sentence, _ in self._iter_candidate_sentences(answer):
            references = references_by_text.get(sentence, set())
            preferred = [
                i for i, chunk_id in enumerate(chunk_ids) if chunk_id in references
            ]
            # Compound summary sentences often combine unrelated facts. If the
            # sentence fails, verify and mark only independently matching clauses.
            clauses = [part.strip() for part in re.split(r"[,;:]\s+", sentence)]
            candidates = [sentence] + (clauses if len(clauses) > 1 else [])
            for candidate in candidates:
                probe = _SentenceProbe(tokenize(candidate))
                if len(probe.tokens) < min_sentence:
                    continue
                answer_tokens = {t for t in probe.tokens if len(t) >= 3}
                informative = substantive(probe.tokens)
                numbers = set(re.findall(r"\b\d+(?:[.-]\d+)*\b", candidate))
                negated = bool(set(probe.tokens) & {"not", "never", "without", "no"})

                def search(positions: Sequence[int]) -> list[tuple[int, str]]:
                    found: list[tuple[int, str]] = []
                    for position in positions:
                        best: tuple[float, str] | None = None
                        for passage, index in source_passages(position):
                            shared = answer_tokens & set(index.tokens)
                            if len(informative & substantive(index.tokens)) < 2:
                                continue
                            if not numbers <= set(
                                re.findall(r"\b\d+(?:[.-]\d+)*\b", passage)
                            ) or negated != bool(
                                set(index.tokens) & {"not", "never", "without", "no"}
                            ):
                                continue
                            effective_window = min(window, len(probe.tokens))
                            strict = any(
                                len(substantive(span)) >= 2
                                for span in probe.grams(effective_window)
                                & index.grams(effective_window)
                            )
                            ratio = len(shared) / max(1, len(answer_tokens))
                            relaxed = len(shared) >= min_shared and ratio >= coverage
                            if not (strict or relaxed):
                                continue
                            rank = ratio + int(strict)
                            if best is None or rank > best[0]:
                                best = (rank, passage)
                        if best is not None:
                            found.append((position, best[1]))
                    return found

                matches = search(preferred) if preferred else []
                if not matches:
                    preferred_set = set(preferred)
                    matches = search(
                        [i for i in range(len(chunk_texts)) if i not in preferred_set]
                    )
                if matches:
                    result[candidate] = matches
                    if candidate == sentence:
                        break
        return result

    @staticmethod
    def _sentence_probe(sentence: str, min_tokens: int) -> _SentenceProbe | None:
        tokens = tokenize(sentence)
        return _SentenceProbe(tokens) if len(tokens) >= min_tokens else None

    def _iter_candidate_sentences(self, answer: str):
        min_tokens = self.cfg.get_int("_GROUNDING.min_sentence_tokens")
        for para in answer.splitlines():
            if not para.strip():
                continue
            for sentence in split_sentences(para):
                probe = self._sentence_probe(sentence, min_tokens)
                if probe is not None:
                    yield sentence, probe

    def find_grounded_sentences(
        self, answer: str, chunk_texts: Sequence[str]
    ) -> list[str]:
        """Return the answer sentences grounded in any of *chunk_texts*."""
        if not chunk_texts:
            return []
        indexes = self._indexes(chunk_texts)
        window = self.cfg.get_int("_GROUNDING.min_overlap_window")
        return [
            sentence
            for sentence, probe in self._iter_candidate_sentences(answer)
            if any(_supports(probe, index, window) for index in indexes)
        ]

    def find_grounded_sentences_by_chunk(
        self, answer: str, chunk_texts: Sequence[str]
    ) -> list[list[str]]:
        """Return, per chunk, the answer sentences grounded in that chunk alone."""
        indexes = self._indexes(chunk_texts)
        result: list[list[str]] = [[] for _ in chunk_texts]
        window = self.cfg.get_int("_GROUNDING.min_overlap_window")
        for sentence, probe in self._iter_candidate_sentences(answer):
            for position, index in enumerate(indexes):
                if _supports(probe, index, window):
                    result[position].append(sentence)
        return result

    def ground_answer_cli(
        self,
        answer: str,
        chunk_texts: Sequence[str],
        ansi_codes: str = "48;5;116",
        sentence_translations: Mapping[str, str] | None = None,
        text_transform: Callable[[str], str] | None = None,
    ) -> str:
        """Return *answer* with grounded sentences wrapped in ANSI colour codes.

        *sentence_translations* maps stripped answer sentences to translated text;
        the translated sentence is checked, the transformed display text is wrapped.
        """

        def transform(text: str) -> str:
            return text_transform(text) if text_transform is not None else text

        if not ansi_codes or not chunk_texts:
            return transform(answer)
        open_tag = f"\033[{ansi_codes}m"
        reset = "\033[0m"
        indexes = self._indexes(chunk_texts)
        window = self.cfg.get_int("_GROUNDING.min_overlap_window")
        min_tokens = self.cfg.get_int("_GROUNDING.min_sentence_tokens")

        def _grounded(probe_text: str) -> bool:
            probe = self._sentence_probe(probe_text, min_tokens)
            return probe is not None and any(
                _supports(probe, index, window) for index in indexes
            )

        result_lines: list[str] = []
        for para in answer.splitlines():
            if not para.strip():
                result_lines.append(para)
                continue
            annotated: list[str] = []
            for sentence in split_sentences(para):
                probes = [sentence]
                translated = (sentence_translations or {}).get(sentence.strip())
                if translated and translated != sentence:
                    probes.insert(0, translated)
                if any(_grounded(probe) for probe in probes):
                    annotated.append(f"{open_tag}{transform(sentence)}{reset}")
                else:
                    annotated.append(transform(sentence))
            result_lines.append(" ".join(annotated))
        return "\n".join(result_lines)

    def find_grounding_fragments_in_chunk(
        self, grounded_sentences: list[str], chunk_text: str
    ) -> list[str]:
        """Return verbatim chunk sentences that overlap with *grounded_sentences*."""
        fragments, _ = self._grounding_fragments(grounded_sentences, chunk_text)
        return fragments

    def _grounding_fragments(
        self, grounded_sentences: list[str], chunk_text: str
    ) -> tuple[list[str], list[list[str]]]:
        """Return aggregate fragments and aligned per-answer-sentence matches."""
        if not grounded_sentences or not chunk_text.strip():
            return grounded_sentences, [[] for _ in grounded_sentences]

        sentence_probes = [_SentenceProbe(tokenize(s)) for s in grounded_sentences]
        window = max(1, self.cfg.get_int("_GROUNDING.min_overlap_window"))
        min_fragment_len = self.cfg.get_int("_GROUNDING.min_fragment_len")

        chunk_sentences: list[str] = []
        seen_chunk_sentences: set[str] = set()
        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in split_sentences(line):
                stripped = sent.strip()
                if (
                    len(stripped) >= min_fragment_len
                    and stripped not in seen_chunk_sentences
                ):
                    seen_chunk_sentences.add(stripped)
                    chunk_sentences.append(stripped)
        if not chunk_sentences:
            stripped = chunk_text.strip()
            if stripped:
                chunk_sentences = [stripped]

        result: list[str] = []
        seen_result: set[str] = set()
        by_sentence: list[list[str]] = [[] for _ in grounded_sentences]
        seen_by_sentence: list[set[str]] = [set() for _ in grounded_sentences]
        for chunk_sent in chunk_sentences:
            chunk_probe = _SentenceProbe(tokenize(chunk_sent))
            key = " ".join(chunk_probe.tokens)
            for position, sentence_probe in enumerate(sentence_probes):
                effective_window = min(
                    window, len(sentence_probe.tokens), len(chunk_probe.tokens)
                )
                if effective_window < 1:
                    continue
                # A shared contiguous window is symmetric, so one check covers both directions.
                if not sentence_probe.grams(effective_window).isdisjoint(
                    chunk_probe.grams(effective_window)
                ):
                    if key not in seen_by_sentence[position]:
                        seen_by_sentence[position].add(key)
                        by_sentence[position].append(chunk_sent)
                    if key not in seen_result:
                        seen_result.add(key)
                        result.append(chunk_sent)

        return (result if result else grounded_sentences), by_sentence

    def find_first_overlap_span(self, sentence_text: str, chunk_text: str) -> str:
        """Return the first sentence token span shared with *chunk_text*."""
        s_tokens = tokenize(sentence_text)
        c_tokens = tokenize(chunk_text)
        window = max(
            1,
            min(
                self.cfg.get_int("_GROUNDING.min_overlap_window"),
                len(s_tokens),
                len(c_tokens),
            ),
        )
        chunk_grams = _grams(c_tokens, window)
        for i in range(len(s_tokens) - window + 1):
            span = tuple(s_tokens[i : i + window])
            if span in chunk_grams:
                return " ".join(span)
        return ""

    # ------------------------------------------------------------------
    # Answer-level checks
    # ------------------------------------------------------------------

    @staticmethod
    def extract_answer_section(text: str) -> str:
        """Return answer text without the optional source-metadata footer."""
        if not text:
            return ""
        return re.split(
            r"(?im)^[ \t]*(?:#{1,6}[ \t]+sources[ \t]*:?"
            r"|(?:#{1,6}[ \t]+)?metadata used:)[ \t]*$",
            text,
            maxsplit=1,
        )[0].strip()

    @staticmethod
    def no_evidence_message() -> str:
        """Return the standard fallback when no evidence is found in context."""
        return (
            "I couldn't find evidence in the retrieved context to answer your query.\n\n"
            "Try starting a new topic with `new:`, switching to a different collection, "
            "or adjusting your session parameters."
        )

    @staticmethod
    def short_answer_notice() -> str:
        """Return notice prepended when grounding is skipped for short answers."""
        return (
            f"{Symbols.sym_warning()} Grounding skipped: answer is too short for reliable "
            "sentence-level grounding.\n\n"
        )

    def is_answer_too_short(self, answer_text: str) -> bool:
        """Return True when answer text is too brief for reliable grounding.

        Grounding gets noisy on terse list-style answers (e.g. short bullet lists),
        so we require both:
        - at least one sentence at/above min_sentence_tokens, and
        - a minimum total token volume across the answer body.
        """
        answer_body = self.extract_answer_section(answer_text)
        if not answer_body:
            return True

        min_tokens = self.cfg.get_int("_GROUNDING.min_sentence_tokens")
        min_total_tokens = max(min_tokens * 2, 10)
        token_counts: list[int] = []
        for para in answer_body.splitlines():
            paragraph = para.strip()
            if not paragraph:
                continue
            for sentence in split_sentences(paragraph):
                token_count = len(_WORD_RX.findall(sentence))
                if token_count > 0:
                    token_counts.append(token_count)

        if not token_counts:
            return True
        if sum(token_counts) < min_total_tokens:
            return True
        return not any(count >= min_tokens for count in token_counts)

    @staticmethod
    def collect_web_texts(session: Any) -> list[str]:
        """Collect deduplicated web snippets/page text from retrieved chunks."""
        chosen = list(getattr(session, "last_chosen_chunks", []) or [])
        web_texts: list[str] = []
        seen_norm: set[str] = set()
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() != "web":
                continue
            snippet = str(meta.get("snippet", "") or "").strip()
            page_text = str(getattr(doc, "page_content", "") or "").strip()
            for candidate in (snippet, page_text):
                if not candidate:
                    continue
                norm = " ".join(candidate.split()).lower()
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                web_texts.append(candidate)
        return web_texts

    def relaxed_overlap_supported(
        self, answer_candidate: str, evidence: Sequence[str]
    ) -> bool:
        """Second-pass grounding for paraphrased/translated answers.

        Uses sentence-vs-evidence token containment with conservative
        thresholds to reduce false negatives while still requiring
        substantial lexical overlap.
        """
        if not answer_candidate.strip() or not evidence:
            return False
        cfg = self.cfg
        min_token_len = cfg.get_int("_GROUNDING.relaxed_min_token_len")
        min_sentence_tokens = cfg.get_int("_GROUNDING.relaxed_min_sentence_tokens")
        min_shared_tokens = cfg.get_int("_GROUNDING.relaxed_min_shared_tokens")
        substantive_len = cfg.get_int("_GROUNDING.relaxed_substantive_token_len")
        min_containment = cfg.get_float("_GROUNDING.relaxed_min_containment")

        def _tokset(text: str) -> set[str]:
            return {
                tok
                for tok in _WORD_RX.findall(text.lower())
                if len(tok) >= min_token_len
            }

        evidence_token_sets = [
            _tokset(chunk) for chunk in evidence if chunk and chunk.strip()
        ]
        evidence_token_sets = [s for s in evidence_token_sets if s]
        if not evidence_token_sets:
            return False

        for para in answer_candidate.splitlines():
            paragraph = para.strip()
            if not paragraph:
                continue
            for sentence in split_sentences(paragraph):
                s_tokens = _tokset(sentence.strip())
                if len(s_tokens) < min_sentence_tokens:
                    continue
                for e_tokens in evidence_token_sets:
                    overlap = s_tokens & e_tokens
                    if len(overlap) < min_shared_tokens:
                        continue
                    # Require at least one substantive token to avoid
                    # stopword-only matches (e.g. "and", "with", "from").
                    if not any(len(tok) >= substantive_len for tok in overlap):
                        continue
                    containment = len(overlap) / max(
                        1, min(len(s_tokens), len(e_tokens))
                    )
                    if containment >= min_containment:
                        return True
        return False

    def has_grounded_evidence(
        self,
        session: Session,
        answer_text: str,
        chunk_texts: list[str],
        response_language: str,
        detect_language: LanguageDetector,
    ) -> bool:
        """Return True when answer_text can be grounded in retrieved evidence.

        Grounding is checked against the answer section (excluding an optional
        sources block). Evidence includes local chunk text and retrieved web
        snippets/page text. When the answer language matches no evidence
        language, the answer is translated sentence by sentence into the
        evidence languages first (direct overlap is impossible then);
        otherwise the translated proxy is only tried after direct grounding
        fails. Retrieval language is a fallback for evidence without language
        metadata. A grounded proxy is stored on the session
        (``grounding_proxy_text`` / ``grounding_proxy_sentences``) so CLI and
        PDF highlights use it; the displayed answer is never replaced.
        """
        started = time.perf_counter()
        try:
            return self._has_grounded_evidence(
                session, answer_text, chunk_texts, response_language, detect_language
            )
        finally:
            self._debug_timing(session, "evidence validation", started)

    def _has_grounded_evidence(
        self,
        session: Session,
        answer_text: str,
        chunk_texts: list[str],
        response_language: str,
        detect_language: LanguageDetector,
    ) -> bool:
        if not answer_text.strip():
            return True

        try:
            web_chunk_texts = [
                item
                for item in cast(list[object], self.collect_web_texts(session))
                if isinstance(item, str) and item.strip()
            ]
        except Exception:
            web_chunk_texts = []

        evidence_texts: list[str] = [c for c in chunk_texts if c and c.strip()]
        evidence_texts.extend(web_chunk_texts)
        if not evidence_texts:
            return True

        answer_body = self.extract_answer_section(answer_text)
        if not answer_body:
            return False

        def _directly_grounded(text: str) -> bool:
            return bool(
                self.find_grounded_sentences(text, evidence_texts)
                or self.relaxed_overlap_supported(text, evidence_texts)
            )

        from Helpers.LanguageConfig import get_lang_name_to_code

        language_codes = get_lang_name_to_code(self.cfg)

        def _language_code(value: Any) -> str:
            language = str(value or "").strip().lower()
            if language in {"", "auto", "unknown", "n/a", "default"}:
                return ""
            return language_codes.get(language, language)

        evidence_languages: list[str] = []
        chosen_docs = list(getattr(session, "last_chosen_chunks", []) or [])
        has_unknown_language = not chosen_docs
        for doc in chosen_docs:
            metadata = getattr(doc, "metadata", {}) or {}
            language = _language_code(metadata.get("Language"))
            if (
                not language
                and str(metadata.get("Source", "")).strip().lower() == "web"
            ):
                evidence_text = str(
                    metadata.get("snippet", "")
                    or getattr(doc, "page_content", "")
                    or ""
                ).strip()
                if evidence_text:
                    language = _language_code(
                        detect_language(
                            evidence_text, response_language, "web_grounding_language"
                        )
                    )
            if not language:
                has_unknown_language = True
            elif language not in evidence_languages:
                evidence_languages.append(language)
        if has_unknown_language:
            retrieval_language = _language_code(
                getattr(session, "retrieval_language", "")
            )
            if retrieval_language and retrieval_language not in evidence_languages:
                evidence_languages.append(retrieval_language)

        answer_lang = _language_code(
            str(
                detect_language(
                    answer_body, response_language, "answer_grounding_language"
                )
                or ""
            )
            .strip()
            .lower()
        )
        translation_targets = [
            language for language in evidence_languages if language != answer_lang
        ]
        # A known answer language that matches no evidence language cannot
        # overlap lexically, so the translated proxy is checked first.
        translate_first = bool(
            answer_lang and evidence_languages and answer_lang not in evidence_languages
        )
        if DebugHelper.check_session(session, 33):
            if answer_lang and answer_lang in evidence_languages:
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"Answer already in document language ({answer_lang}); "
                    "no translation required for grounding against this language.",
                    color=CYAN,
                )
            elif not evidence_languages:
                self.pretty.write(
                    "D",
                    "Grounding",
                    "Document language unavailable; checking the original answer "
                    "without a grounding translation.",
                    color=CYAN,
                )
        if not translate_first and _directly_grounded(answer_body):
            return True
        if not translation_targets:
            return False

        if self._check_answer_in_document_languages(
            session, answer_body, answer_lang, translation_targets, _directly_grounded
        ):
            return True
        return translate_first and _directly_grounded(answer_body)

    def _check_answer_in_document_languages(
        self,
        session: Session,
        answer_body: str,
        answer_language: str,
        document_languages: Sequence[str],
        is_grounded: Callable[[str], bool],
    ) -> bool:
        """Translate and check the answer without changing its displayed text.

        Save a supported translation and its sentence mapping for later answer
        and document highlighting. Failed or unchanged translations are logged
        and skipped so another document language can still be checked.
        """
        try:
            from Compliance.SharedHelpers import SharedHelpers

            translator = SharedHelpers()
        except Exception as exc:
            self.pretty.write(
                "W",
                "Grounding",
                f"Cannot initialize answer-proxy translation: {exc}",
                color=ORANGE,
            )
            return False

        for target in document_languages:
            source = answer_language or "auto"
            if DebugHelper.check_session(session, 30):
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"Checking translated answer proxy {source}->{target} "
                    "against retrieved evidence; displayed answer is unchanged.",
                    color=CYAN,
                )
            # Translate per sentence so every displayed sentence maps to its
            # proxy for highlighting.
            proxy_sentences: dict[str, str] = {}
            proxy_lines: list[str] = []
            try:
                for para in answer_body.splitlines():
                    parts: list[str] = []
                    for sentence in split_sentences(para):
                        original = sentence.strip()
                        if not original:
                            continue
                        translated = translator.translate_text(
                            original,
                            target_lang=target,
                            source_lang=source,
                        )
                        translated = (translated or "").strip() or original
                        proxy_sentences[original] = translated
                        parts.append(translated)
                    proxy_lines.append(" ".join(parts))
            except Exception as exc:
                self.pretty.write(
                    "W",
                    "Grounding",
                    f"Answer-proxy translation {source}->{target} failed: {exc}",
                    color=ORANGE,
                )
                continue

            if all(k == v for k, v in proxy_sentences.items()):
                self.pretty.write(
                    "W",
                    "Grounding",
                    f"Answer-proxy translation {source}->{target} returned "
                    "empty or unchanged text; could not check this language.",
                    color=ORANGE,
                )
                continue

            proxy_text = "\n".join(proxy_lines)
            if DebugHelper.check_session(session, 33):
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"Translated answer for grounding ({source}->{target}, "
                    f"document language {target}):\n{proxy_text}",
                    color=CYAN,
                )
            if is_grounded(proxy_text):
                session.grounding_proxy_text = proxy_text
                session.grounding_proxy_sentences = proxy_sentences
                self.pretty_always.write(
                    "I",
                    "Grounding",
                    f"Answer grounded via translation {source}->{target}; "
                    "highlights use the translation, the displayed answer "
                    "is unchanged.",
                    color=CYAN,
                )
                return True

        return False

    @staticmethod
    def _summary_statements(
        session: Any, proxies: Mapping[str, str] | None = None
    ) -> list[SummaryStatement]:
        statements: list[SummaryStatement] = list(
            getattr(session, "summary_statements", []) or []
        )
        if not proxies:
            return statements
        return [
            SummaryStatement(
                proxies.get(sentence.strip(), sentence), statement.context_chunk_ids
            )
            for statement in statements
            for line in statement.text.splitlines()
            for sentence in split_sentences(line)
        ]

    def _session_summary_support(
        self,
        session: Any,
        answer: str,
        chunk_texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        proxies: Mapping[str, str] | None = None,
    ) -> dict[str, list[tuple[int, str]]]:
        ids = [
            int(meta.get("SummaryChunkId", position))
            for position, meta in enumerate(metadatas, 1)
        ]
        return self.summary_support(
            answer, chunk_texts, ids, self._summary_statements(session, proxies)
        )

    def has_local_spans(self, session: Any) -> bool:
        """Only prepare source highlights when the answer has local matching spans.

        Uses the translated grounding proxy when one was accepted.
        """
        answer_body = getattr(
            session, "grounding_proxy_text", ""
        ) or self.extract_answer_section(
            getattr(session, "last_answer_content", "") or ""
        )
        local_texts = list(getattr(session, "chunk_texts_for_grounding", []) or [])
        if getattr(session, "summary_grounding", False):
            docs = list(getattr(session, "last_chosen_chunks", []) or [])
            return bool(
                self._session_summary_support(
                    session,
                    answer_body,
                    local_texts,
                    [doc.metadata for doc in docs],
                    getattr(session, "grounding_proxy_sentences", {}),
                )
            )
        return bool(
            answer_body
            and local_texts
            and self.find_grounded_sentences(answer_body, local_texts)
        )

    def mark_answer_cli(
        self,
        session: Any,
        answer_text: str,
        evidence_texts: Sequence[str],
        sentence_translations: Mapping[str, str] | None = None,
        text_transform: Callable[[str], str] | None = None,
    ) -> str:
        """Return the ANSI-coloured CLI rendering of *answer_text*.

        *sentence_translations* maps displayed sentences to their translations so
        cross-language answers are highlighted in the displayed language. An
        optional transform changes rendered text only, not the grounding probes.
        """

        def transform(text: str) -> str:
            return text_transform(text) if text_transform is not None else text

        started = time.perf_counter()
        ansi = self.cfg.get_str("_MARKED_DOCS_COLORS.answer_ansi") or ""
        try:
            if getattr(session, "summary_grounding", False):
                docs = list(getattr(session, "last_chosen_chunks", []) or [])
                probes = "\n".join(
                    (sentence_translations or {}).get(sentence.strip(), sentence)
                    for line in answer_text.splitlines()
                    for sentence in split_sentences(line)
                )
                support = self._session_summary_support(
                    session,
                    probes,
                    evidence_texts,
                    [doc.metadata for doc in docs],
                    sentence_translations,
                )

                def mark(sentence: str) -> str:
                    translated = (sentence_translations or {}).get(
                        sentence.strip(), sentence
                    )
                    if not ansi:
                        return transform(sentence)
                    if translated in support:
                        return f"\033[{ansi}m{transform(sentence)}\033[0m"
                    if translated != sentence:
                        return transform(sentence)
                    spans = sorted(
                        (start, start + len(part))
                        for part in support
                        if (start := sentence.find(part)) >= 0
                    )
                    marked: list[str] = []
                    end = 0
                    for start, stop in spans:
                        if start < end:
                            continue
                        marked.extend(
                            (
                                transform(sentence[end:start]),
                                f"\033[{ansi}m{transform(sentence[start:stop])}\033[0m",
                            )
                        )
                        end = stop
                    return "".join(marked) + transform(sentence[end:])

                return "\n".join(
                    " ".join(mark(sentence) for sentence in split_sentences(line))
                    for line in answer_text.splitlines()
                )
            return self.ground_answer_cli(
                answer_text,
                evidence_texts,
                ansi_codes=ansi,
                sentence_translations=sentence_translations,
                text_transform=text_transform,
            )
        finally:
            self._debug_timing(session, "CLI answer marks", started)

    # ------------------------------------------------------------------
    # Source-document snippets
    # ------------------------------------------------------------------

    def build_grounded_snippets(
        self,
        session: Any,
        chosen: Sequence[Any],
        answer_mark_color: str,
    ) -> dict[str, list[Any]]:
        """Map answer evidence to source-text snippets, grouped by local file.

        Summary grounding supplies verified passages directly; ordinary grounding
        first identifies supported answer sentences, then locates source fragments.
        Per-sentence attribution is retained for diagnostics without rematching.
        Snippets retain physical page indexes for the document marker. This method
        prepares marking instructions; it does not modify or open source files.
        """
        import os

        from VisualMarkers import ChunkSnippet

        started = time.perf_counter()
        debug = DebugHelper.check(self.cfg, 33)
        grounded: dict[str, list[Any]] = {}
        chunk_texts: list[str] = list(
            getattr(session, "chunk_texts_for_grounding", []) or []
        )
        if debug:
            self.pretty.write(
                "D",
                "Grounding",
                f"chunk_texts_for_grounding={len(chunk_texts)}  "
                f"last_answer_content={bool(getattr(session, 'last_answer_content', ''))}  "
                f"answer_mark_color={answer_mark_color!r}",
                color=CYAN,
            )

        if not chunk_texts:
            return grounded

        # A translated grounding proxy matches the evidence language.
        answer_text = getattr(session, "grounding_proxy_text", "") or getattr(
            session, "last_answer_content", ""
        )
        if not answer_text:
            return grounded

        # Only existing local files can be marked. Positions below refer to this
        # filtered list, not to the original chosen chunks.
        local_docs: list[tuple[Any, str, str]] = []
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() == "web":
                continue
            file_path = str(meta.get("FilePath", "")).strip()
            if not file_path or not os.path.isfile(file_path):
                if debug:
                    self.pretty.write(
                        "D",
                        "Grounding",
                        f"skip chunk: FilePath={file_path!r}  "
                        f"exists={os.path.isfile(file_path) if file_path else False}",
                        color=CYAN,
                    )
                continue
            chunk_text = getattr(doc, "page_content", "") or ""
            if chunk_text.strip():
                local_docs.append((meta, file_path, chunk_text))

        summary_passages: dict[int, list[str]] = {}
        summary_fragments_by_chunk: dict[int, dict[str, list[str]]] = {}
        if getattr(session, "summary_grounding", False):
            support = self._session_summary_support(
                session,
                answer_text,
                [text for _, _, text in local_docs],
                [meta for meta, _, _ in local_docs],
                getattr(session, "grounding_proxy_sentences", {}),
            )
            matches_by_chunk: list[list[str]] = [[] for _ in local_docs]
            # Invert sentence -> supporting chunks into chunk -> answer sentences
            # and preserve the verified source passages for highlighting.
            for sentence, matches in support.items():
                for position, passage in matches:
                    matches_by_chunk[position].append(sentence)
                    summary_passages.setdefault(position, []).append(passage)
                    by_sentence = summary_fragments_by_chunk.setdefault(position, {})
                    by_sentence.setdefault(sentence, []).append(passage)
        else:
            matches_by_chunk = self.find_grounded_sentences_by_chunk(
                answer_text, [chunk_text for _, _, chunk_text in local_docs]
            )
        seen_by_file: dict[str, set[str]] = {}
        for position, ((meta, file_path, chunk_text), matched) in enumerate(
            zip(local_docs, matches_by_chunk)
        ):
            if debug:
                fn = meta.get("FileName", os.path.basename(file_path))
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"chunk {fn!r}  matched={len(matched)}  "
                    + (repr(matched) if matched else "(none)"),
                    color=CYAN,
                )
            if not matched:
                continue

            # Prefer verified summary passages. Ordinary answers need source-text
            # fragments because paraphrased answer text may not occur in the file.
            if position in summary_passages:
                pdf_texts = summary_passages[position]
                fragments_by_sentence = [
                    summary_fragments_by_chunk[position][sentence]
                    for sentence in matched
                ]
            else:
                pdf_texts, fragments_by_sentence = self._grounding_fragments(
                    matched, chunk_text
                )
            if debug:
                self._debug_chunk_fragments(
                    meta, file_path, matched, pdf_texts, fragments_by_sentence
                )

            seen = seen_by_file.setdefault(file_path, set())
            # Confine the answer highlight to the source chunk's physical page so a
            # fragment that also occurs earlier (TOC/front matter) is not marked there.
            try:
                page_value: Any = meta.get("PageNumber")
                page_int: int | None = (
                    int(page_value) if page_value is not None else None
                )
            except (TypeError, ValueError):
                page_int = None
            for text in pdf_texts:
                # Overlapping chunks can repeat evidence. Keep one normalized
                # fragment per file/page, but retain repeats on different pages.
                key = f"{page_int}:" + " ".join(text.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                grounded.setdefault(file_path, []).append(
                    ChunkSnippet(
                        text=text, page_number=page_int, color=answer_mark_color
                    )
                )
        self._debug_timing(session, "grounded source snippets", started)
        return grounded

    def _debug_chunk_fragments(
        self,
        meta: Mapping[str, Any],
        file_path: str,
        matched: list[str],
        pdf_texts: list[str],
        fragments_by_sentence: list[list[str]],
    ) -> None:
        import os

        fn = meta.get("FileName", os.path.basename(file_path))
        status = (
            "FALLBACK — no overlap match"
            if pdf_texts == matched
            else "chunk sentences matched"
        )
        self.pretty.write(
            "D",
            "Grounding",
            f"chunk {fn!r}  pdf_snippets={len(pdf_texts)}  ({status})",
            color=CYAN,
        )
        for i, (sentence, hits) in enumerate(zip(matched, fragments_by_sentence)):
            hit_strs = hits if hits != [sentence] else []
            self.pretty.write(
                "D", "Grounding", f"  answer[{i}]: {sentence.strip()!r}", color=CYAN
            )
            if not hit_strs:
                self.pretty.write(
                    "D", "Grounding", "    → NO MATCH in chunk", color=CYAN
                )
                continue
            for hit in hit_strs:
                span = self.find_first_overlap_span(sentence, hit)
                self.pretty.write("D", "Grounding", f"    → pdf:   {hit!r}", color=CYAN)
                if span:
                    self.pretty.write(
                        "D", "Grounding", f"    → match: >>>{span}<<<", color=CYAN
                    )

    def _debug_timing(self, session: Any, stage: str, started: float) -> None:
        enabled = (
            DebugHelper.check_session(session, 33)
            if session is not None and hasattr(session, "debug_level")
            else DebugHelper.check(self.cfg, 33)
        )
        if enabled:
            self.pretty.write(
                "D",
                "Grounding",
                f"Timing {stage}: {(time.perf_counter() - started) * 1000:.1f} ms",
                color=CYAN,
            )
