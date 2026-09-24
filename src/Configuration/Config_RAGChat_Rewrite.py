from typing import Any

_QUERY_REWRITE: dict[str, Any] = {
    "enabled": True,
    "topic_confidence_threshold": 0.5,  # Minimum LLM confidence (0.0-1.0) to use the contextual
    # rewrite (depends_on_previous_turn=True path). Below this
    # threshold standalone_rewrite is used instead.
    "TOPIC_SUMMARY_MODE": "last",  # Controls rolling_topic_summary passed to the topic-detect LLM.
    # "last" = ASSISTANT block from the most recent history turn only.
    # "all"  = ASSISTANT blocks from all turns in the history window.
    # Retrieval gate: noun heads that signal an attribute/property query without
    # a concrete entity anchor.  If the rewritten query's root NP head matches one
    # of these and contains no named entity or proper noun, retrieval is blocked
    # and a clarification message is returned instead.  The list is intentionally
    # small and domain-agnostic — do not add specific attribute names here.
    "meta_descriptors": [
        "specifications",
        "specification",
        "specs",
        "spec",
        "details",
        "detail",
        "features",
        "feature",
        "properties",
        "property",
        "characteristics",
        "characteristic",
        "capabilities",
        "capability",
        "parameters",
        "parameter",
        "configuration",
        "settings",
        "setting",
        "requirements",
        "requirement",
        "information",
        "info",
        "overview",
    ],
    # Normalise the user query to English BEFORE query rewriting and retrieval.
    # Applies to both vector and BM25 paths so HYBRID fusion stays consistent.
    # Also helps the rewriter LLM resolve English pronouns reliably and prevents
    # zero cross-lingual token overlap in BM25.
    #
    # TRANSLATION_BACKEND selects the engine:
    #   "argos"  — offline Argos Translate (OPUS-MT based; lighter but lower quality
    #              on short/colloquial sentences). Requires the language pair to be
    #              installed (see _ARGOS_DEFINITIONS.ARGOS_LANGUAGES).
    #   "off"    — no translation, query sent as-is.
    "TRANSLATION_BACKEND": "argos",
    "LLM_PARAM": {
        "temperature": 0.05,
        "top_k": 10,
        "top_p": 0.9,
        "num_predict": 256,
        "use_ollama_gpu": True,
        "streaming": False,
    },
}

# ── Multi-Query Expansion ──────────────────────────────────────────────────────

_PROMPT_QUERY_EXPAND = """You are a query expansion assistant for a retrieval system.

Given the retrieval query below, generate {num_variants} alternative phrasings.
The goal is maximum RETRIEVAL DIVERSITY: each variant should use different vocabulary
so that together they cover synonyms, technical equivalents, and domain-specific
terminology that a relevant document might use instead of the original wording.

Rules:
- Prefer domain-specific terms a technical document would actually contain
  (e.g. "thermal conditions" → also try "operating temperature", "operating environment",
  "temperature range", "environmental specifications", "ambient temperature limits").
- Use different wording, synonyms, and sentence structure across variants.
- Each variant must be a complete, self-contained retrieval query.
- Do NOT add new facts, entities, or assumptions not present in the original query.
- Do NOT produce meta-descriptor queries (e.g. "What are the characteristics of X?").
- Keep all variants in English retrieval language.

Output ONLY a JSON array of strings. No commentary, no markdown, no preamble.
Example for num_variants=3: ["variant one", "variant two", "variant three"]

Query: {query}
"""

_MULTI_QUERY: dict[str, Any] = {
    "enabled": True,
    "num_variants": 3,  # Number of alternate queries to generate per turn
    "LLM_PARAM": {
        "temperature": 0.5,  # Higher than rewrite — diversity matters here
        "top_k": 40,
        "top_p": 0.95,
        "num_predict": 256,
        "use_ollama_gpu": True,
        "streaming": False,
    },
}

# ── Chunk Near-Duplicate Removal ───────────────────────────────────────────────

_CHUNK_DEDUP: dict[str, Any] = {
    "enabled": True,
    # Jaccard similarity threshold: chunks sharing >= this fraction of word
    # tokens are considered near-duplicates; the lower-ranked one is dropped.
    "threshold": 0.85,
}

# -----------------------------------------------------------------------------
# Terminal line size — RAGChat-specific override.
# Expressed as a dict so the width switches automatically based on the live
# session debug_level (resolved by QueryParts._resolve_terminal_line_size and
# PrettyWriter.terminal_line_size — both check this key at use time).
#   debug:    wide to accommodate algo tables when debug_level > 0
#   no_debug: normal conversation width
# RAGChatService inherits this via its `from Config_RAGChat import *`.
# All other apps receive the flat baseline from Config_Global.py.
# -----------------------------------------------------------------------------
