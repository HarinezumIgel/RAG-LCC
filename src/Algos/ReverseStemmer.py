"""
ReverseStemmer – maps stemmed tokens back to the best (highest-weight) original word.

Build it alongside stem_keywords_with_weights by calling update() for every
stem/original/weight triple.  The representative for a stem is always the
original word whose weight was highest among all words that share that stem.
Weights are NOT stored permanently in the map – one stem can come from many
words, so keeping a single weight value per stem would be ambiguous once the
best word is chosen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Gui.PrettyWriter import PrettyWriter


class ReverseStemmer:
    """
    Lightweight reverse-stemming lookup table.

    - Only the best-weight original word is kept as the representative for each stem.
    - No weights are retained after construction.
    """

    def __init__(self, pretty: "PrettyWriter | None" = None) -> None:
        from Gui.PrettyWriter import PrettyWriter as _PW

        self.pretty: _PW = pretty or _PW()
        self.stem_to_word: dict[str, str] = {}  # stem → best original word
        self.stem_to_weight: dict[str, float] = {}  # temporary: weight of current best
        self.pretty.write("I", "ReverseStemmer", "Reverse stemmer initialized")

    # ------------------------------------------------------------------
    # Building the map
    # ------------------------------------------------------------------

    def update(self, stem: str, original: str, weight: float) -> None:
        """Record / replace the stem → original mapping if *weight* beats the current best."""
        if stem not in self.stem_to_weight or weight > self.stem_to_weight[stem]:
            self.stem_to_word[stem] = original
            self.stem_to_weight[stem] = weight

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def reverse(self, token: str) -> str:
        """Return the best original word for *token*; if unknown, return *token* unchanged."""
        return self.stem_to_word.get(token, token)

    def reverse_text(self, text: str) -> str:
        """Replace each whitespace-delimited token in *text* with its original word where known."""
        if not text:
            return text
        return " ".join(self.reverse(t) for t in text.split())

    def apply_to_meta(self, meta: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        """
        Return a shallow copy of *meta* with reverse-stemmed string values for
        every key in *keys* that is present in *meta*.
        """
        result = dict(meta)
        for key in keys:
            v = result.get(key)
            if isinstance(v, str):
                result[key] = self.reverse_text(v)
        return result

    # ------------------------------------------------------------------
    # Python data-model helpers
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.stem_to_word)

    def __len__(self) -> int:
        return len(self.stem_to_word)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ReverseStemmer({len(self)} entries)"
