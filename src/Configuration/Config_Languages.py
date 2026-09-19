"""Language activation, Argos pairs, and shared spaCy role settings."""

import os
from typing import Any

import Configuration.Config_Global as Config_Global

_global_absolute_path = str(getattr(Config_Global, "_ABSOLUTE_PATH", "") or "").strip()

if not _global_absolute_path:
    raise RuntimeError(
        "Config_Global._ABSOLUTE_PATH is required by Config_Languages and "
        "must be defined."
    )
if not os.path.isabs(_global_absolute_path):
    raise RuntimeError(
        "Config_Global._ABSOLUTE_PATH must be an absolute path, got: "
        f"'{_global_absolute_path}'"
    )

# -----------------------------------------------------------------
# Language code <-> name mapping (single source of truth)
# ISO-639-1 code -> NLTK / human-readable name.
# Used by FileUtils (code->name) and SharedHelpers (name->code, reversed).
# Covers the intersection of langdetect output and NLTK stopword names.
# This map does NOT activate a language; it only normalizes labels.
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# spaCy model packages for the active languages above.
# REQUIRED: every active language code in _ACTIVE_LANGUAGES must have
# exactly one entry here.
# -----------------------------------------------------------------------------
_SPACY_MODELS_BY_ACTIVE_LANGUAGE: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
}

# -----------------------------------------------------------------------------
# Shared spaCy settings by retriever role
# -----------------------------------------------------------------------------
_SPACY: dict[str, Any] = {
    "BM25": {
        # Optional default spaCy model for BM25 lemmatization.
        # Leave empty to keep the current regex-tokenizer fallback.
        "spacy_model": "",
        # Optional per-language model overrides for BM25 lemmatization.
        # Keys accept language codes or names (for example: "de" or "german").
        # Values are spaCy package names that must be installed.
        # Example: {"de": "de_core_news_sm", "fr": "fr_core_news_sm"}
        "spacy_models_by_language": "$_SPACY_MODELS_BY_ACTIVE_LANGUAGE",
    },
    "GRAPH": {
        # spaCy model used for NER during both RAGLoad (indexing) and RAGChat (query).
        "spacy_model": "$_SPACY_MODELS_BY_ACTIVE_LANGUAGE.en",
        # Optional per-language model overrides for graph retrieval.
        # Keys accept language codes or names (for example: "de" or "german").
        # Values are spaCy package names that must be installed.
        # Example: {"de": "de_core_news_sm", "fr": "fr_core_news_sm"}
        "spacy_models_by_language": "$_SPACY_MODELS_BY_ACTIVE_LANGUAGE",
    },
    "REGEX": {
        # spaCy model used for POS/lemma extraction during both RAGLoad and RAGChat.
        "spacy_model": "$_SPACY_MODELS_BY_ACTIVE_LANGUAGE.en",
        # Optional per-language model overrides for regex retrieval.
        # Keys accept language codes or names (for example: "de" or "german").
        # Values are spaCy package names that must be installed.
        # Example: {"de": "de_core_news_sm", "it": "it_core_news_sm"}
        "spacy_models_by_language": "$_SPACY_MODELS_BY_ACTIVE_LANGUAGE",
    },
}

# -----------------------------------------------------------------------------
# BM25 index - Okapi BM25 hyper-parameters and RRF fusion constant
# -----------------------------------------------------------------------------
_BM25_INDEX: dict[str, Any] = {
    "BM25_INDEX_DIR": os.path.join(_global_absolute_path, "chromadb", "bm25"),
    "k1": 1.2,  # Term-frequency saturation. Higher -> raw TF matters more
    "b": 0.75,  # Length normalization. 0 = off, 1 = fully length-normalized
    "rrf_k": 60.0,  # Reciprocal Rank Fusion constant (ALL / *_GRAPH modes)
    # Indirect aliases resolved via Config.indirect_* helpers.
    "spacy_model": "$_SPACY.BM25.spacy_model",
    "spacy_models_by_language": "$_SPACY.BM25.spacy_models_by_language",
}

# -----------------------------------------------------------------------------
# Graph index - entity co-occurrence graph for semantic graph search
# Run once before using graph modes:
#   python src/Scripts/SpacyLanguageModels.py install
# -----------------------------------------------------------------------------
_GRAPH_INDEX: dict[str, Any] = {
    "GRAPH_INDEX_DIR": os.path.join(_global_absolute_path, "chromadb", "graph"),
    # spaCy NER label filter - only entities with these labels are indexed.
    # Full list: https://spacy.io/api/annotation#named-entities
    # Special sentinel: "NOUN_CHUNK" enables noun-phrase extraction via the
    # spaCy dependency parser (no extra model needed) - catches domain terms
    # like animals, plants, technical concepts that NER does not tag.
    "entity_types": [
        "PERSON",
        "ORG",
        "GPE",
        "PRODUCT",
        "WORK_OF_ART",
        "LAW",
        "NOUN_CHUNK",
    ],
    # Maximum graph hops from seed entities when expanding the candidate set.
    # 1 = direct co-occurrence only; 2 = two-hop neighborhood.
    "max_hops": 2,
    # Maximum candidate chunks returned by graph query (before top-k cap).
    "max_candidates": 50,
    # Minimum co-occurrence edge weight to follow during BFS traversal.
    # Set to 2+ to require entities to co-occur in at least N chunks.
    "min_edge_weight": 1,
    # Indirect aliases resolved via Config.indirect_* helpers.
    "spacy_model": "$_SPACY.GRAPH.spacy_model",
    "spacy_models_by_language": "$_SPACY.GRAPH.spacy_models_by_language",
    # Noun-chunk noise filter (only applied when NOUN_CHUNK is in entity_types).
    "noun_chunk_min_chars": 3,
    # Discard noun chunks whose first character is one of these.
    "noun_chunk_drop_leading": "[({<",
}

# -----------------------------------------------------------------------------
# Regex index - content-verb/content-noun retrieval
# -----------------------------------------------------------------------------
_REGEX_INDEX: dict[str, Any] = {
    "REGEX_INDEX_DIR": os.path.join(_global_absolute_path, "chromadb", "regex"),
    # Indirect aliases resolved via Config.indirect_* helpers.
    "spacy_model": "$_SPACY.REGEX.spacy_model",
    "spacy_models_by_language": "$_SPACY.REGEX.spacy_models_by_language",
    # Maximum number of candidates kept before final top-k cap.
    "max_candidates": 50,
    # Ignore tiny lemmas/noise (applies to both verbs and nouns).
    "min_token_chars": 3,
    # Keep nouns and proper nouns as the noun channel.
    "noun_pos_tags": ["NOUN", "PROPN"],
    # Exclude auxiliary verbs from the verb channel (be/do/have/etc.).
    "exclude_auxiliaries": True,
    "aux_lemmas": [
        "be",
        "do",
        "have",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    ],
    # Strict gate policy when both query channels are present.
    "require_both_when_available": True,
    "min_verb_hits": 1,
    "min_noun_hits": 1,
    # Fallback policy: if strict gate finds no chunk, allow verb-only matches.
    "fallback_to_verb_only": True,
    "fallback_min_verb_hits": 1,
    # Coverage score weights.
    "verb_weight": 2.0,
    "noun_weight": 1.0,
    "both_match_bonus": 0.25,
    # Damp fallback scores so strict matches stay ahead when both exist.
    "fallback_score_scale": 0.6,
}

# NOTE: ARGOS_LANGUAGES may contain a broader catalog than active runtime use.
# Runtime code filters this list by _ACTIVE_LANGUAGES before consent checks.
# IMPORTANT: when adding to _ACTIVE_LANGUAGES, add matching pairs here too.
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
