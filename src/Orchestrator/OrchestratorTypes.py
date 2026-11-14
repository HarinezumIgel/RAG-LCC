from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Chat.KeywordActionSplitter import KeywordActionSplit


@dataclass
class RetrievalPlan:
    user_query_original: str
    final_query: str
    alternate_queries: list[str]
    retrieve_mode: str
    bm25_query: str
    orig_translated_query_en: str
    keyword_action_split: KeywordActionSplit | None = None


@dataclass
class LocalDocs:
    vector_docs: list[Any]
    bm25_docs: list[Any]
    graph_docs: list[Any]
    regex_docs: list[Any]


@dataclass(frozen=True)
class GuardrailInputs:
    primary_query: str
    guardrail_query: str
    use_guardrail: bool


@dataclass(frozen=True)
class OriginalQueryLeg:
    enabled: bool
    query: str
    language_bucket: str | None
    reason: str


@dataclass(frozen=True)
class GraphRegexStageIteration:
    language_bucket: str | None
    file_filter: dict[str, Any] | None
    primary_query: str
    guardrail_query: str
    is_alternate: bool = False


@dataclass(frozen=True)
class Bm25StageInvocation:
    file_filter: dict[str, Any] | None
    primary_query: str
    guardrail_query: str
    is_alternate: bool = False


@dataclass(frozen=True)
class IndexedStageResult:
    bm25_docs: list[Any]
    graph_docs: list[Any]
    regex_docs: list[Any]
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int


@dataclass(frozen=True)
class VectorStageResult:
    vector_docs: list[Any]
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int
    alternate_queries: list[str] | None = None


@dataclass(frozen=True)
class LocalStageInputs:
    vector_enabled: bool
    bm25_enabled: bool
    graph_enabled: bool
    regex_enabled: bool
    indexed_query_shaping: bool
    primary_query: str
    guardrail_query: str
    use_guardrail: bool
    alternate_queries: list[str]
    file_filter: dict[str, Any] | None
    original_query_leg: OriginalQueryLeg


@dataclass(frozen=True)
class LocalStageResult:
    local_docs: LocalDocs
    top_k_orig_query_en: int
    top_k_post_rewrite_query_en: int


@dataclass(frozen=True)
class PostGatePipelineInputs:
    flow_name: str
    primary_query_source: str
    guardrail_query_source: str
    enable_guardrail_leg: bool
    retrieve_mode: str
    bm25_query: str
    alternate_queries: list[str]
    orig_translated_query_en: str
    user_query_original: str
    resolved_guardrail: GuardrailInputs
    enable_indexed_query_shaping: bool
    enable_indexed_query_translation: bool
    enable_indexed_query_wordnet_expansion: bool
    enable_original_language_vector_leg: bool
    enable_vector_retriever: bool
    enable_bm25_retriever: bool
    enable_graph_retriever: bool
    enable_regex_retriever: bool


@dataclass(frozen=True)
class PreGatePipelineResult:
    should_abort: bool
    post_gate_inputs: PostGatePipelineInputs | None


@dataclass(frozen=True)
class StageExecutionGates:
    web_mode: str
    local_stage_enabled: bool
    web_stage_enabled: bool


@dataclass(frozen=True)
class OrchestrationFlowProfile:
    name: str
    retrieve_mode_override: str
    primary_query_source: str
    guardrail_query_source: str
    enable_guardrail_leg: bool
    enable_original_language_vector_leg: bool
    enable_indexed_query_shaping: bool
    enable_indexed_query_translation: bool
    enable_indexed_query_wordnet_expansion: bool
    enable_vector_alternate_queries: bool
    enable_query_rewrite: bool
    enable_pronoun_substitution: bool
    enable_vector_retriever: bool
    enable_bm25_retriever: bool
    enable_graph_retriever: bool
    enable_regex_retriever: bool
    enable_rerank: bool
    enable_low_score_fallback: bool
    enable_low_recall_rescue: bool
    enable_grounding: bool
    confidence_evaluation: str = "both"
    # Query the cross-encoder scores against. REWRITTEN_QUERY keeps the legacy
    # behaviour (mySession.query, i.e. the normalized/rewritten query).
    rerank_query_source: str = "REWRITTEN_QUERY"
    # ENGLISH: normalize the query to English (legacy). NATIVE: keep the query
    # in the user's language through rewrite, retrieval and rerank.
    retrieval_language_mode: str = "ENGLISH"
    enable_keyword_action_splitter: bool = False


IndexedSpec = tuple[str, bool, float, Any, int, Any]
INDEXED_RETRIEVER_LABELS = ("BM25", "Graph", "Regex")
