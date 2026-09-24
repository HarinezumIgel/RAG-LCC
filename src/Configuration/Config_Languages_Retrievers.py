"""Shared spaCy settings and retriever index paths."""

# pylint: disable=invalid-name

import os
from typing import Any

from Configuration import Config_Global

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
    "BM25_INDEX_DIR": os.path.join(
        Config_Global.require_absolute_path("Config_Languages"),
        "chromadb",
        "bm25",
    ),
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
    "GRAPH_INDEX_DIR": os.path.join(
        Config_Global.require_absolute_path("Config_Languages"),
        "chromadb",
        "graph",
    ),
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
    "REGEX_INDEX_DIR": os.path.join(
        Config_Global.require_absolute_path("Config_Languages"),
        "chromadb",
        "regex",
    ),
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
