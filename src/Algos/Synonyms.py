"""
WordNet-based synonym expander for banned-word lists.

Expands each English banned phrase with direct synonyms from NLTK WordNet,
respecting configurable depth, POS filter, max-per-phrase cap, and a
stoplist to suppress overly generic results.

Intended consumers: RegexScorer, JaccardScorer, BM25Scorer.
KeyBertScorer should NOT use expanded lists (embeddings already capture
semantic neighbours).
"""

from typing import Any, Dict, List, Set, Tuple, cast

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter

# ---------------------------------------------------------------------------
# Lazy import flag — avoids startup crash when NLTK / WordNet not installed.
# ---------------------------------------------------------------------------
_wordnet_available: bool | None = None
_wn = None  # module reference set by _ensure_wordnet()


def _ensure_wordnet() -> bool:
    """Import nltk.corpus.wordnet on first use.  Returns True if available."""
    global _wordnet_available, _wn
    if _wordnet_available is not None:
        return _wordnet_available
    try:
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        # Force a quick lookup to verify the corpus data is downloaded
        wn.synsets("test")  # type: ignore[no-untyped-call]
        _wn = wn
        _wordnet_available = True
    except Exception:
        _wordnet_available = False
    return _wordnet_available


class Synonyms(SingletonMixin):
    """
    Singleton that expands a list of English phrases with WordNet synonyms.

    Usage::

        syn = Synonyms()
        expanded = syn.expand(["password", "credit card number"])
        # → ["password", "watchword", "countersign", "credit card number", ...]
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()

        # Read configuration from _WORDNET block
        self.enabled: bool = self.cfg.get_bool("_WORDNET.ENABLED")
        self.depth: int = self.cfg.get_int("_WORDNET.DEPTH")
        self.max_per_phrase: int = self.cfg.get_int("_WORDNET.MAX_SYNONYMS_PER_PHRASE")
        pos_raw = cast(List[str], self.cfg.get_list("_WORDNET.POS_FILTER"))
        self.pos_filter: Set[str] = {p.lower() for p in pos_raw} if pos_raw else set()
        stoplist_raw = cast(List[str], self.cfg.get_list("_WORDNET.STOPLIST"))
        self.stoplist: Set[str] = {w.lower() for w in stoplist_raw}
        self.debug_level: int = self.cfg.get_int("DEBUG_LEVEL")

        # Cache: tuple(original_list) → expanded list
        self._cache: Dict[Tuple[str, ...], List[str]] = {}

        # Availability flag — set on first expand() call
        self._available: bool | None = None
        self._warned: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, phrases: List[str]) -> List[str]:
        """Return *phrases* expanded with WordNet synonyms.

        If WordNet is unavailable or expansion is disabled, the original
        list is returned unchanged.
        """
        if not self.enabled:
            if self.debug_level >= 1 and not self._warned:
                self._warned = True
                self.pretty.write(
                    "D",
                    "WordNet Synonyms",
                    "Synonym expansion disabled in config (_WORDNET.ENABLED=False)",
                )
            return phrases

        # Lazy availability check
        if self._available is None:
            self._available = _ensure_wordnet()
            if not self._available:
                self.pretty.write(
                    "W",
                    "WordNet Synonyms",
                    "NLTK WordNet not installed — synonym expansion skipped, "
                    "proceeding with original banned-word list. "
                    "Install with: pip install nltk && python -c "
                    "\"import nltk; nltk.download('wordnet')\"",
                    color=ORANGE,
                )
                self._warned = True
                return phrases

        if not self._available:
            return phrases

        cache_key: Tuple[str, ...] = tuple(phrases)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        expanded: List[str] = self._do_expand(phrases)
        self._cache[cache_key] = expanded
        return expanded

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_expand(self, phrases: List[str]) -> List[str]:
        """Core expansion logic."""
        result: List[str] = []
        seen: Set[str] = set()
        total_added: int = 0

        for phrase in phrases:
            low = phrase.lower()
            if low not in seen:
                result.append(phrase)
                seen.add(low)

            synonyms = self._synonyms_for_phrase(phrase, depth=self.depth)
            added_for_phrase = 0

            for syn in synonyms:
                syn_low = syn.lower()
                if syn_low in seen or syn_low in self.stoplist:
                    continue
                if added_for_phrase >= self.max_per_phrase:
                    break
                result.append(syn)
                seen.add(syn_low)
                added_for_phrase += 1
                total_added += 1

                if self.debug_level >= 10:
                    self.pretty.write(
                        "D",
                        "WordNet Synonyms",
                        f"  +synonym: {phrase!r} → {syn!r}",
                    )

        if self.debug_level >= 1:
            self.pretty.write(
                "D",
                "WordNet Synonyms",
                f"Expanded {len(phrases)} phrases → {len(result)} "
                f"(+{total_added} synonyms, depth={self.depth}, "
                f"max/phrase={self.max_per_phrase})",
            )
        return result

    def _synonyms_for_phrase(self, phrase: str, depth: int) -> List[str]:
        """Collect synonyms for a phrase up to *depth* hops.

        Multi-word phrases: looks up synsets for the full phrase first
        (underscore-joined, as WordNet stores them), then falls back to
        individual tokens if no synsets are found.
        """
        assert _wn is not None
        # WordNet uses underscores for multi-word lemmas
        wn_key = phrase.replace(" ", "_").lower()
        all_synonyms: Set[str] = set()
        phrase_low = phrase.lower()

        # Depth-1: direct synonyms
        self._collect_lemmas(wn_key, all_synonyms)

        # Depth-2+: synonyms of synonyms (breadth-first)
        if depth >= 2:
            frontier: Set[str] = set(all_synonyms)
            for _ in range(depth - 1):
                next_frontier: Set[str] = set()
                for syn in frontier:
                    new: Set[str] = set()
                    self._collect_lemmas(syn.replace(" ", "_").lower(), new)
                    next_frontier |= new - all_synonyms
                all_synonyms |= next_frontier
                frontier = next_frontier

        # If multi-word phrase yielded nothing, try individual tokens
        if not all_synonyms and " " in phrase:
            for token in phrase.lower().split():
                self._collect_lemmas(token, all_synonyms)

        # Remove the original phrase itself
        all_synonyms.discard(phrase_low)

        # Sort for deterministic ordering
        return sorted(all_synonyms)

    def _collect_lemmas(self, wn_key: str, out: Set[str]) -> None:
        """Add lemma names from matching synsets into *out*."""
        assert _wn is not None
        synsets: list[Any] = _wn.synsets(wn_key)  # type: ignore[no-untyped-call]
        for synset in synsets:
            if synset is None:
                continue
            # POS filter
            if self.pos_filter and synset.pos() not in self.pos_filter:
                continue
            for lemma in synset.lemmas():
                name: str = lemma.name().replace("_", " ").lower()
                out.add(name)
