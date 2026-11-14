"""
Configuration for Chat.ConfidenceEvaluator (retrieval evidence confidence).
"""

from typing import Any

# -----------------------------------------------------------------------------
# Heuristic answer confidence from the selected evidence chunks.
#
# Per-chunk confidence is sigmoid(raw_rerank_score), else rerank_score or
# chroma_score clamped to [0, 1]. Chunks scored only on unknown scales
# (RRF, BM25, graph, regex) get unknown_scale_score instead of 0.
#
# C_final = weight_top      * top_score       (best selected chunk)
#         + weight_mean_top * mean_top3       (mean of the top 3 chunks)
#         + weight_coverage * coverage        (selected / final_chunks_to_llm)
#         + weight_local    * local_share     (share of non-web chunks)
#         - fallback_penalty                  (only if the low-confidence
#                                              rerank fallback was triggered)
# clamped to [0, 1]. Keep the four weights summing to 1.0 so C_final stays a
# comparable 0..1 score.
#
# Levels: HIGH when C_final >= level_high, MEDIUM when >= level_medium,
# otherwise LOW.
# -----------------------------------------------------------------------------
_CONFIDENCE_EVALUATOR: dict[str, Any] = {
    "weight_top": 0.50,
    "weight_mean_top": 0.30,
    "weight_coverage": 0.15,
    "weight_local": 0.05,
    "fallback_penalty": 0.15,
    "level_high": 0.78,
    "level_medium": 0.56,
    "unknown_scale_score": 0.5,
}
