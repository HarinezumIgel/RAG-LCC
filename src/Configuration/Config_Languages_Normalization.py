"""WordNet and text-normalization maps used by banned-word processing."""

_WORDNET: dict[str, int | bool | list[str]] = {
    # --- expansion control ---------------------------------------------------
    # Enable / disable synonym expansion globally
    "ENABLED": True,
    # WordNet lookup depth (1 = direct synonyms only, 2 = synonyms of synonyms)
    "DEPTH": 1,
    # Maximum number of synonyms to add per original banned phrase
    "MAX_SYNONYMS_PER_PHRASE": 1,
    # --- POS filtering -------------------------------------------------------
    # Restrict to these WordNet POS tags.  Allowed: "n" (noun), "v" (verb),
    # "a" (adjective), "r" (adverb), "s" (adjective satellite).
    # Empty list = no POS filter (accept all).
    "POS_FILTER": ["n", "v"],
    # --- stoplist ------------------------------------------------------------
    # Generic words that appear as WordNet synonyms but are too broad to be
    # useful as banned-word expansions.  Case-insensitive comparison.
    "STOPLIST": [
        "word",
        "number",
        "figure",
        "item",
        "thing",
        "part",
        "piece",
        "set",
        "group",
        "kind",
        "type",
        "form",
        "point",
        "line",
        "way",
        "case",
        "level",
        "area",
        "place",
        "make",
        "give",
        "18",  # synonym of "under 18"; leet-decode normalises 1→i 8→b → "ib", causing false positives
    ],
}

_LEET_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
}
_CONFUSABLES = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "в": "b",
    "к": "k",
    "м": "m",
    "н": "h",
    "т": "t",
    "İ": "i",
    "ı": "i",
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
}
