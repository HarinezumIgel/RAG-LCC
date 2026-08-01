"""Web search configuration — shared across RAGChat and RAGChatService.

All web-search-specific settings live here so they can be hash-checked as a
unit by ``Compliance._check_models_config_hash()``.  The corresponding hash
slot is ``_WEB_SEARCH_CONFIG_HASH`` in ``Config_Global.py``.

- Keep this file in sync with Config_Global.py's _WEB_SEARCH_CONFIG_HASH.
- Per-app audit log paths (_QUERY_LOG, _INTENT_FILTER_LOG) are NOT here;
  they live in Config_RAGChat.py / Config_RAGChatService.py because each app
  writes to its own log subdirectory.
- _OPENWEB_UI_WEBSEARCH lives here (same file) — it is a web-search switch
  and belongs alongside the other web-search controls.
"""

from typing import Any

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Web-search switch reference                                                │
# │                                                                             │
# │  Switch                  Where              Effect                          │
# │  ─────────────────────── ────────────────── ─────────────────────────────   │
# │  WEB_SEARCH_MODE (env)   Config_Internet_Env MASTER SWITCH. Overrides all   │
# │                                             other web-search settings.      │
# │  _OPENWEB_UI_WEBSEARCH   Config_WebSearch   Default web-search state for    │
# │                                             new OpenWebUI sessions. Has no  │
# │                                             effect when WEB_SEARCH_MODE is  │
# │                                             not "1".                        │
# │  web_search (session)    per-request        Caller opt-in. Ignored when     │
# │                                             WEB_SEARCH_MODE is "0".          │
# │                                                                             │
# │  WEB_SEARCH_MODE values:                                                    │
# │                                                                             │
# │   Value       Result                                                        │
# │  ──────────   ───────────────────────────────────────────────────────────   │
# │   "0"         Web queries blocked.                                          │
# │   "1"         Full production path — queries pass compliance then go live.  │
# └─────────────────────────────────────────────────────────────────────────────┘

# Master switch source of truth:
#   os.environ["WEB_SEARCH_MODE"] in Config_Internet_Env.py
# Allowed values are: "0" | "1"
# This file intentionally defines only web-search behavior settings.

# -----------------------------------------------------------------------------
# OpenWebUI web-search default
#
# When True, web_search is enabled for every OpenWebUI request that does not
# carry an explicit web_search parameter — so users never need to add an
# Advanced Parameter manually.
# This setting has no effect unless WEB_SEARCH_MODE = "1".
# A startup warning is printed when this is True but the master switch is not "1".
# -------------------------------------
_OPENWEB_UI_WEBSEARCH: bool = False

# -----------------------------------------------------------------------------
# Web search backend — optional internet retrieval leg added to the RRF
# pipeline at query time when WEB_SEARCH_MODE = "1".
#
# backend:          "duckduckgo" (default, no key needed) — only implemented backend.
#                   "brave" | "tavily" | "bing" are recognised but raise NotImplementedError.
# api_key:          Reserved for future brave / tavily / bing backends.  Leave empty for duckduckgo.
# max_results:      Maximum web results fetched per query (used when fetch_k not set).
# max_query_length: Queries longer than this are truncated before sending.
# block_on_injection: Block queries that match prompt-injection / attack patterns.
# default_web_weight: Default RRF weight for web results relative to local retrievers
#   (Vector/BM25/Graph = 1.0).  0.5 means every local result naturally outranks any
#   web result; raise to 1.0+ for equal or higher web influence.  Overridable
#   per-session with web_weight=<value>.
# bm25_pre_filter:  Minimum BM25 score (against the query) a web result must reach to
#   survive before entering the rerank pool.  0.0 = disabled (all results pass).
#   Typical useful range: 0.05–0.30.  Only active when retrieve_mode includes web
#   results (web_search=True or retrieve_mode=WEB).
# cosine_pre_filter: Minimum cosine similarity (query embedding vs. snippet embedding)
#   a web result must reach to survive.  0.0 = disabled.  Runs after bm25_pre_filter
#   when both are set.  Typical useful range: 0.20–0.50.  Requires the embedding
#   model to be loaded (always true in RAGChat).
# -----------------------------------------------------------------------------
_WEB_SEARCH: dict[str, Any] = {
    "backend": "duckduckgo",
    "api_key": "",
    "max_results": 10,
    "max_query_length": 500,
    "block_on_injection": True,
    "default_web_weight": 0.5,
    "bm25_pre_filter": 0.10,
    "cosine_pre_filter": 0.30,
    # rerank_threshold: After reranking, web chunks whose sigmoid(raw_rerank_score)
    #   falls below this probability are dropped.  0.50 = sigmoid(0) = neutral logit
    #   (50 % relevance probability); the cosine_pre_filter (0.30) is the primary
    #   admission gate so this default is intentionally permissive.  Lower toward
    #   0.0 to keep weak-scoring web results; raise toward 1.0 for stricter filtering.
    "rerank_threshold": 0.50,
}

# -----------------------------------------------------------------------------
# Intent-filter extensions — additive-only extensions to the baseline
# WebSearchFilter entity catalogue and thresholds.
#
# Allowed keys:
#   entity_extensions        — add entity terms to existing baseline categories
#   entity_categories_extra  — add entirely new entity categories (may NOT
#                              shadow baseline category names)
#   threshold_overrides      — may only LOWER (tighten) the baseline
#                              refuse/warn thresholds; higher values are
#                              silently ignored.
#
# Uncomment and populate entries as needed for your deployment.
# ---------------------------------------------------------------------------

WEB_SEARCH_INTENT_EXTENSIONS: dict[str, Any] = {
    "entity_extensions": {
        # "illicit_substances": ["new_scheduled_compound"],
        # "chemical_weapons":   ["agent_x"],
    },
    "entity_categories_extra": {
        # "your_category": {"weight": 25, "entities": ["term_a", "term_b"]},

        # ── DEMO ── fictional category to demonstrate intent-filter blocking.
        # "zeta-9" is a made-up term that the LLM won't flag, but the scoring
        # pipeline refuses on entity weight alone (no suspicious verb needed):
        #   entity "zeta-9" (weight 70) >= refuse threshold (60) → REFUSE
        # Try the query:  tell me about zeta-9
        # Remove this block before production deployment.
        "demo_watchlist": {
            "weight": 70,
            "entities": ["zeta-9"],
        },
    },
    "threshold_overrides": {
        # "refuse": 45,  # stricter than baseline 60
        # "warn":   20,  # stricter than baseline 30
    },
}
