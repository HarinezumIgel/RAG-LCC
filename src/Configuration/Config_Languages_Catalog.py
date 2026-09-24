"""Language catalog, activation list, and Argos translation pairs."""

from typing import Any

_LANG_CODE_TO_NAME: dict[str, str] = {
    "ar": "arabic",
    "bn": "bengali",
    "ca": "catalan",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "he": "hebrew",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "ne": "nepali",
    "nl": "dutch",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sl": "slovene",
    "sq": "albanian",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
}

# -----------------------------------------------------------------
# Active languages used by ALL language-aware subsystems.
# Entries may be ISO codes ("de") or language names ("german").
# IMPORTANT - KEEP THESE THREE SLOTS IN SYNC:
# If you add a language to _ACTIVE_LANGUAGES, you must also update:
# 1) _ARGOS_DEFINITIONS["ARGOS_LANGUAGES"]
#    - add required Argos language pairs for that language.
#    - usually both X->EN (query normalization) and EN->X
#      (banned-word localization).
# 2) _SPACY_MODELS_BY_ACTIVE_LANGUAGE
#    - add a spaCy model package for the same language code.
# Missing either update causes incomplete translation/retrieval coverage.
# -----------------------------------------------------------------
_ACTIVE_LANGUAGES: list[str] = [
    "en",
    "de",
    "es",
    "fr",
    "it",
]

_ARGOS_DEFINITIONS: dict[str, Any] = {
    # Indirect aliases: resolved via Config.indirect_* accessors.
    "ACTIVE_LANGUAGES": "$_ACTIVE_LANGUAGES",
    "LANG_CODE_TO_NAME": "$_LANG_CODE_TO_NAME",
    # Argos Translate language pairs.
    # Each tuple is (from_code, to_code).
    # NOTE: Argos packages are unidirectional. Keep EN->X pairs for banned-word
    # localization and X->EN pairs for query normalization.
    "ARGOS_LANGUAGES": [
        # ("ar", "en"),  # Arabic -> English
        # ("en", "ar"),  # English -> Arabic
        # ("az", "en"),  # Azerbaijani -> English
        # ("en", "az"),  # English -> Azerbaijani
        # ("bg", "en"),  # Bulgarian -> English
        # ("en", "bg"),  # English -> Bulgarian
        # ("bn", "en"),  # Bengali -> English
        # ("en", "bn"),  # English -> Bengali
        # ("ca", "en"),  # Catalan -> English
        # ("en", "ca"),  # English -> Catalan
        # ("cs", "en"),  # Czech -> English
        # ("en", "cs"),  # English -> Czech
        # ("da", "en"),  # Danish -> English
        # ("en", "da"),  # English -> Danish
        ("de", "en"),  # German -> English
        ("en", "de"),  # English -> German
        # ("el", "en"),  # Greek -> English
        # ("en", "el"),  # English -> Greek
        # ("eo", "en"),  # Esperanto -> English
        # ("en", "eo"),  # English -> Esperanto
        ("es", "en"),  # Spanish -> English
        ("en", "es"),  # English -> Spanish
        # ("et", "en"),  # Estonian -> English
        # ("en", "et"),  # English -> Estonian
        # ("eu", "en"),  # Basque -> English
        # ("en", "eu"),  # English -> Basque
        # ("fa", "en"),  # Persian -> English
        # ("en", "fa"),  # English -> Persian
        # ("fi", "en"),  # Finnish -> English
        # ("en", "fi"),  # English -> Finnish
        ("fr", "en"),  # French -> English
        ("en", "fr"),  # English -> French
        # ("ga", "en"),  # Irish -> English
        # ("en", "ga"),  # English -> Irish
        # ("gl", "en"),  # Galician -> English
        # ("en", "gl"),  # English -> Galician
        # ("he", "en"),  # Hebrew -> English
        # ("en", "he"),  # English -> Hebrew
        # ("hi", "en"),  # Hindi -> English
        # ("en", "hi"),  # English -> Hindi
        # ("hu", "en"),  # Hungarian -> English
        # ("en", "hu"),  # English -> Hungarian
        # ("id", "en"),  # Indonesian -> English
        # ("en", "id"),  # English -> Indonesian
        ("it", "en"),  # Italian -> English
        ("en", "it"),  # English -> Italian
        # ("ja", "en"),  # Japanese -> English
        # ("en", "ja"),  # English -> Japanese
        # ("ko", "en"),  # Korean -> English
        # ("en", "ko"),  # English -> Korean
        # ("ky", "en"),  # Kyrgyz -> English
        # ("en", "ky"),  # English -> Kyrgyz
        # ("lt", "en"),  # Lithuanian -> English
        # ("en", "lt"),  # English -> Lithuanian
        # ("lv", "en"),  # Latvian -> English
        # ("en", "lv"),  # English -> Latvian
        # ("ms", "en"),  # Malay -> English
        # ("en", "ms"),  # English -> Malay
        # ("nb", "en"),  # Norwegian -> English
        # ("en", "nb"),  # English -> Norwegian
        # ("nl", "en"),  # Dutch -> English
        # ("en", "nl"),  # English -> Dutch
        # ("pb", "en"),  # Portuguese (Brazil) -> English
        # ("en", "pb"),  # English -> Portuguese (Brazil)
        # ("pl", "en"),  # Polish -> English
        # ("en", "pl"),  # English -> Polish
        # ("pt", "en"),  # Portuguese -> English
        # ("en", "pt"),  # English -> Portuguese
        # ("ro", "en"),  # Romanian -> English
        # ("en", "ro"),  # English -> Romanian
        # ("ru", "en"),  # Russian -> English
        # ("en", "ru"),  # English -> Russian
        # ("sk", "en"),  # Slovak -> English
        # ("en", "sk"),  # English -> Slovak
        # ("sl", "en"),  # Slovenian -> English
        # ("en", "sl"),  # English -> Slovenian
        # ("sq", "en"),  # Albanian -> English
        # ("en", "sq"),  # English -> Albanian
        # ("sv", "en"),  # Swedish -> English
        # ("en", "sv"),  # English -> Swedish
        # ("th", "en"),  # Thai -> English
        # ("en", "th"),  # English -> Thai
        # ("tl", "en"),  # Tagalog -> English
        # ("en", "tl"),  # English -> Tagalog
        # ("tr", "en"),  # Turkish -> English
        # ("en", "tr"),  # English -> Turkish
        # ("uk", "en"),  # Ukrainian -> English
        # ("en", "uk"),  # English -> Ukrainian
        # ("ur", "en"),  # Urdu -> English
        # ("en", "ur"),  # English -> Urdu
        # ("vi", "en"),  # Vietnamese -> English
        # ("en", "vi"),  # English -> Vietnamese
        # ("zh", "en"),  # Chinese -> English
        # ("en", "zh"),  # English -> Chinese
        # ("zt", "en"),  # Chinese (traditional) -> English
        # ("en", "zt"),  # English -> Chinese (traditional)
    ],
}

# =============================================================================
# WordNet synonym expansion for banned-word lists.
# Expands banned phrases with English synonyms (NLTK WordNet) before
# translation / detection.  Only feeds Regex, Jaccard, BM25 — KeyBERT
# already captures semantic neighbours via embeddings.
# =============================================================================
