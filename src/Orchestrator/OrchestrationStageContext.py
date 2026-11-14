"""Typed shared-state contract for Orchestrator responsibility classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from Chat.RagRetrievalContext import RagRetrievalContext
    from Globals.Session import Session
    from Orchestrator.OrchestratorHelper import OrchestratorHelper
    from Orchestrator.OrchestratorMessages import OrchestratorMessages
    from Orchestrator.OrchestratorOverrides import OrchestratorOverrides
    from Orchestrator.OrchestratorTypes import (Bm25StageInvocation,
                                                GraphRegexStageIteration,
                                                GuardrailInputs, IndexedSpec,
                                                IndexedStageResult, LocalDocs,
                                                LocalStageInputs,
                                                LocalStageResult,
                                                OrchestrationFlowProfile,
                                                OriginalQueryLeg,
                                                PostGatePipelineInputs,
                                                PreGatePipelineResult,
                                                RetrievalPlan,
                                                StageExecutionGates,
                                                VectorStageResult)


class OrchestrationStageContext(Protocol):
    _active_indexed_stage_iterations: list[GraphRegexStageIteration] | None

    _flow_enable_bm25_retriever: bool

    _flow_enable_graph_retriever: bool

    _flow_enable_indexed_query_shaping: bool

    _flow_enable_indexed_query_translation: bool

    _flow_enable_indexed_query_wordnet_expansion: bool

    _flow_enable_original_language_vector_leg: bool

    _flow_enable_regex_retriever: bool

    _flow_enable_vector_retriever: bool

    _indexed_alternate_queries: list[str]

    _indexed_flow_name: str

    _indexed_guardrail_enabled: bool

    _indexed_guardrail_query_source: str

    _indexed_primary_query_source: str

    _indexed_stage_shape_queries: bool

    _prepared_indexed_stage_iterations: list[GraphRegexStageIteration] | None

    _rag_chat_impl: RagRetrievalContext

    _rerank_language_queries: list[dict[str, Any]]

    _respect_legacy_retrieve_mode: bool

    _suppress_indexed_retriever_dispatch_trace: bool

    orchestratorHelper: OrchestratorHelper

    orchestratorMessages: OrchestratorMessages

    orchestratorOverrides: OrchestratorOverrides

    @staticmethod
    def _is_guardrail_query_active(
        primary_query: str, guardrail_query: str
    ) -> bool: ...

    def _resolve_guardrail_inputs_for_flow(
        self,
        mySession: Session,
        *,
        plan: RetrievalPlan,
        flow_profile: OrchestrationFlowProfile,
    ) -> GuardrailInputs: ...

    @staticmethod
    def _apply_original_leg_flow_gate(
        original_query_leg: OriginalQueryLeg,
        *,
        enable_original_language_vector_leg: bool,
    ) -> OriginalQueryLeg: ...

    @staticmethod
    def _is_english_language_bucket(language_bucket: str | None) -> bool: ...

    def _resolve_original_query_leg(
        self, mySession: Session, *, primary_query: str, guardrail_query: str
    ) -> OriginalQueryLeg: ...

    def _prepare_session(self, mySession: Session) -> str: ...

    def _normalize_query(
        self, mySession: Session, user_query_original: str
    ) -> tuple[str, list[str]]: ...

    def _check_gates(self, mySession: Session, final_query: str) -> bool: ...

    def _resolve_plan(self, mySession: Session) -> RetrievalPlan: ...

    def run_web_retriever(
        self, mySession: Session, retrieve_mode: str, user_query_original: str
    ) -> list[Any]: ...

    def _is_web_search_allowed(self, retrieve_mode: str) -> bool: ...

    def _fetch_web_docs(
        self, mySession: Session, *, retrieve_mode: str, user_query_original: str
    ) -> list[Any]: ...

    def _apply_web_prefilters(
        self, mySession: Session, web_docs: list[Any]
    ) -> list[Any]: ...

    def _run_web_prefilter(
        self,
        mySession: Session,
        *,
        docs_before: list[Any],
        docs_after: list[Any],
        before_pre: int,
        label: str,
        threshold: float,
    ) -> list[Any]: ...

    def _merge_and_build_context(
        self, mySession: Session, local_docs: LocalDocs, web_docs: list[Any]
    ) -> Tuple[str, int]: ...

    def _run_merge_and_context_stage(
        self, mySession: Session, local_docs: LocalDocs, web_docs: list[Any]
    ) -> Tuple[str, int]: ...

    def _resolve_guardrail_inputs(
        self, *, bm25_query: str, orig_translated_query_en: str
    ) -> GuardrailInputs: ...

    def _resolve_stage_execution_gates(
        self, mySession: Session, *, retrieve_mode: str
    ) -> StageExecutionGates: ...

    @staticmethod
    def _empty_local_docs() -> LocalDocs: ...

    @staticmethod
    def _partition_indexed_specs(
        indexed_specs: list[IndexedSpec],
    ) -> tuple[list[IndexedSpec], list[IndexedSpec]]: ...

    def _run_indexed_specs_stage(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]: ...

    def _stage_invocation_values(
        self, stage_invocation: Any
    ) -> tuple[str | None, dict[str, Any] | None, str, str]: ...

    def _run_indexed_stage_group(
        self,
        mySession: Session,
        *,
        stage_invocations: list[Any],
        specs: list[IndexedSpec],
        use_guardrail: bool,
        dispatch_per_spec: bool,
    ) -> tuple[dict[str, list[Any]], int, int]: ...

    def _dispatch_indexed_stage_specs(
        self,
        mySession: Session,
        *,
        stage_specs: list[IndexedSpec],
        language_bucket: str | None,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
    ) -> tuple[dict[str, list[Any]], int, int]: ...

    def _merge_alternate_indexed_variants(
        self,
        mySession: Session,
        ranked_results: dict[str, list[tuple[list[Any], float]]],
    ) -> dict[str, list[Any]]: ...

    def _bm25_stage_invocations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[Bm25StageInvocation]: ...

    def _normalize_language_bucket(self, language: Any) -> str | None: ...

    def _language_bucket_from_filter(
        self, file_filter: dict[str, Any] | None
    ) -> str | None: ...

    @staticmethod
    def _remove_language_from_filter(
        file_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None: ...

    def _indexed_query_settings(self) -> dict[str, Any]: ...

    def _discover_graph_regex_languages(
        self, file_filter: dict[str, Any] | None
    ) -> list[str]: ...

    def _active_configured_language_codes(self) -> set[str]: ...

    def _relevant_graph_regex_languages(
        self, mySession: Session, available_languages: list[str]
    ) -> list[str]: ...

    def _plan_graph_regex_language_buckets(
        self, mySession: Session, file_filter: dict[str, Any] | None
    ) -> list[str]: ...

    @staticmethod
    def _build_graph_regex_stage_filter(
        base_file_filter: dict[str, Any] | None, language_bucket: str | None
    ) -> dict[str, Any] | None: ...

    def _graph_regex_stage_filters(
        self, mySession: Session, file_filter: dict[str, Any] | None
    ) -> list[dict[str, Any] | None]: ...

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
    ) -> tuple[str, str]: ...

    def _shape_primary_indexed_query(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        query: str,
        failure_policy: str,
    ) -> str | None: ...

    def _shape_optional_indexed_query(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        query: str,
        category: Literal["guardrail", "alternative"],
    ) -> str | None: ...

    def _shape_indexed_query_alternatives(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        alternatives: list[str],
    ) -> list[tuple[str, str]]: ...

    def _graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[GraphRegexStageIteration]: ...

    def _shape_stage_iteration_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        alternatives: list[str],
        failure_policy: str,
    ) -> tuple[str, str, list[tuple[str, str]]] | None: ...

    @staticmethod
    def _expand_stage_iterations(
        *,
        language_bucket: str | None,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
        stage_alternatives: list[tuple[str, str]],
    ) -> list[GraphRegexStageIteration]: ...

    def _indexed_guardrail_disabled_reason(
        self, *, primary_query: str, guardrail_query: str
    ) -> str | None: ...

    def _shared_graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[GraphRegexStageIteration]: ...

    def _run_indexed_retrieval_stages(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> IndexedStageResult: ...

    def _announce_indexed_stage_dispatch(
        self,
        mySession: Session,
        *,
        stage_iterations: list[GraphRegexStageIteration],
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
        use_guardrail: bool,
    ) -> bool: ...

    def _indexed_stage_groups(
        self,
        mySession: Session,
        *,
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[tuple[list[Any], list[IndexedSpec], bool]]: ...

    def _run_indexed_retrieval_with_shape_mode(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
        shape_queries: bool,
    ) -> IndexedStageResult: ...

    def _run_vector_stage(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> VectorStageResult: ...

    def _merge_native_vector_leg(
        self,
        mySession: Session,
        *,
        vector_docs: list[Any],
        local_inputs: LocalStageInputs,
    ) -> tuple[list[Any], dict[str, int]]: ...

    def _merge_native_indexed_leg(
        self,
        indexed_stage_result: IndexedStageResult,
        native_stage_result: IndexedStageResult | None,
    ) -> tuple[dict[str, list[Any]], dict[str, int]]: ...

    def _store_original_query_leg_stats(
        self,
        mySession: Session,
        *,
        original_query_leg: OriginalQueryLeg,
        vector_hits: int,
        vector_added: int,
        vector_overlap: int,
        bm25_hits: int = 0,
        bm25_added: int = 0,
        bm25_overlap: int = 0,
        graph_hits: int = 0,
        graph_added: int = 0,
        graph_overlap: int = 0,
        regex_hits: int = 0,
        regex_added: int = 0,
        regex_overlap: int = 0,
    ) -> None: ...

    def _run_original_query_vector_leg(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        original_query_leg: OriginalQueryLeg,
    ) -> list[Any]: ...

    def _run_original_query_indexed_leg(
        self,
        mySession: Session,
        *,
        original_query_leg: OriginalQueryLeg,
        indexed_specs: list[IndexedSpec],
        file_filter: dict[str, Any] | None,
    ) -> IndexedStageResult | None: ...

    def _resolve_local_stage_inputs(
        self,
        mySession: Session,
        *,
        retrieve_mode: str | None,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: GuardrailInputs | None,
    ) -> LocalStageInputs: ...

    def _run_local_retrieval_stages(
        self, mySession: Session, *, local_inputs: LocalStageInputs
    ) -> LocalStageResult: ...

    def _execute_local_retrieval_stages(
        self, mySession: Session, *, local_inputs: LocalStageInputs
    ) -> LocalStageResult: ...

    def _store_guardrail_topk_if_enabled(
        self,
        mySession: Session,
        *,
        use_guardrail: bool,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None: ...

    def _run_local_docs_stage(
        self,
        mySession: Session,
        *,
        retrieve_mode: str | None,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: GuardrailInputs | None,
    ) -> LocalDocs: ...

    def _run_split_local_stages(
        self, mySession: Session, local_inputs: LocalStageInputs
    ) -> LocalStageResult: ...

    def _run_post_gate_pipeline(
        self, mySession: Session, *, pipeline_inputs: PostGatePipelineInputs
    ) -> Tuple[str, int]: ...

    def _configure_post_gate_flow(
        self, pipeline_inputs: PostGatePipelineInputs
    ) -> None: ...

    def _run_local_pipeline_stage(
        self,
        mySession: Session,
        *,
        pipeline_inputs: PostGatePipelineInputs,
        stage_gates: StageExecutionGates,
    ) -> LocalDocs: ...

    def _run_web_pipeline_stage(
        self,
        mySession: Session,
        *,
        pipeline_inputs: PostGatePipelineInputs,
        stage_gates: StageExecutionGates,
    ) -> list[Any]: ...

    def _run_pre_gate_pipeline(
        self,
        mySession: Session,
        *,
        flow_profile: OrchestrationFlowProfile | None = None,
    ) -> PreGatePipelineResult: ...

    def _prepare_session_flow_profile(
        self, mySession: Session, flow_profile: OrchestrationFlowProfile | None
    ) -> OrchestrationFlowProfile: ...

    def _normalize_user_query(self, mySession: Session) -> RetrievalPlan: ...

    def _is_retrieval_gated(self, mySession: Session, plan: RetrievalPlan) -> bool: ...

    @staticmethod
    def _build_post_gate_inputs(
        *,
        plan: RetrievalPlan,
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str,
        guardrail_inputs: GuardrailInputs,
        alternate_queries: list[str],
    ) -> PostGatePipelineInputs: ...

    def run_local_retrievers(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
        resolved_guardrail: GuardrailInputs | None = None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]: ...

    def run(self, mySession: Session) -> Tuple[str, int]: ...
