"""Retrieval orchestration flow presets.

Users choose one of the predefined top-level flow names during a session.
Flows are configured here; the GUI picker does not construct flows.
"""

# -----------------------------------------------------------------------------
# Orchestration-flow selector
# -----------------------------------------------------------------------------
_ALLOWED_ORCHESTRATION_FLOWS = [
    # Empty means: expose all keys from _ORCHESTRATION_FLOWS.
]

_ACTIVE_ORCHESTRATION_FLOW = "THOROUGH_QUERY_REWRITE"

_ALLOWED_ORCHESTRATION_QUERY_SOURCES = [
    "FINAL_QUERY",
    "TRANSLATED_QUERY",
    "ORIGINAL_QUERY",
]

# -----------------------------------------------------------------------------
# Flow profiles
# -----------------------------------------------------------------------------
# Keys:
# - force_retrieve_mode: optional mode override (VECTOR/BM25/GRAPH/REGEX/.../ALL/WEB)
# - main_query_source: FINAL_QUERY | TRANSLATED_QUERY | ORIGINAL_QUERY
# - secondary_query_source: FINAL_QUERY | TRANSLATED_QUERY | ORIGINAL_QUERY
# - use_secondary_query: run a second local retrieval leg with the secondary query
# - use_original_language_vector: run original-language vector leg
# - shape_indexed_queries: per-language query shaping for BM25/Graph/Regex stage
# - use_vector_alternates: enable MultiQuery alternate fanout for vector stage
# - use_query_rewrite: enable/disable query rewriting for this flow
# - use_pronoun_substitution: enable/disable pronoun referent substitution in rewrite
# - run_local_stage: allow local retrieval stage
# - run_web_stage: allow web retrieval stage
# - run_vector: enable vector local retriever leg
# - run_bm25: enable BM25 local retriever leg
# - run_graph: enable graph local retriever leg
# - run_regex: enable regex local retriever leg
# - run_rerank: enable reranking before chunk selection
# - run_low_score_fallback: allow low-score rerank fallback to retrieval order
# - run_low_recall_rescue: allow query-overlap local rescue when rerank hits are sparse
# - run_grounding: enable answer grounding / visual-marker stage

_ORCHESTRATION_FLOWS: dict[str, dict[str, str | bool]] = {
    "THOROUGH_QUERY_REWRITE": {
        "force_retrieve_mode": "",
        "use_secondary_query": True,
        "use_original_language_vector": True,
        "shape_indexed_queries": True,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_pronoun_substitution": True,
        "run_local_stage": True,
        "run_web_stage": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
    },
    "TRANSLATION_FOCUSED": {
        "force_retrieve_mode": "",
        "main_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "use_vector_alternates": False,
        "use_query_rewrite": True,
        "use_pronoun_substitution": True,
        "run_local_stage": True,
        "run_web_stage": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
    },
    "ORIGINAL_LANGUAGE_VECTOR_ONLY": {
        "force_retrieve_mode": "VECTOR",
        "main_query_source": "ORIGINAL_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_pronoun_substitution": True,
        "run_local_stage": True,
        "run_web_stage": False,
        "run_vector": True,
        "run_bm25": False,
        "run_graph": False,
        "run_regex": False,
        "run_rerank": True,
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
    },
    "ORIGINAL_LANGUAGE_ALL_RETRIEVERS": {
        "force_retrieve_mode": "VECTOR",
        "main_query_source": "ORIGINAL_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_pronoun_substitution": True,
        "run_local_stage": True,
        "run_web_stage": False,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
    },
}
