from __future__ import annotations

import os
import time
from typing import Any, Tuple

from Chat.KeywordActionSplitter import KeywordActionSplitter
from Globals.Session import Session
from Gui.Colors import CYAN, ORANGE
from Helpers.DebugHelper import DebugHelper
from Orchestrator.IndexedRetrievalExecutor import IndexedRetrievalExecutor
from Orchestrator.IndexedRetrievalPlanner import IndexedRetrievalPlanner
from Orchestrator.LocalRetrievalPipeline import LocalRetrievalPipeline
from Orchestrator.OrchestratorHelper import OrchestratorHelper
from Orchestrator.OrchestratorMessages import OrchestratorMessages
from Orchestrator.OrchestratorOverrides import OrchestratorOverrides
from Orchestrator.OrchestratorTypes import (GraphRegexStageIteration,
                                            GuardrailInputs, LocalDocs,
                                            OrchestrationFlowProfile,
                                            OriginalQueryLeg,
                                            PostGatePipelineInputs,
                                            PreGatePipelineResult,
                                            RetrievalPlan, StageExecutionGates)
from Orchestrator.OriginalQueryLegExecutor import OriginalQueryLegExecutor


class Orchestrator(
    IndexedRetrievalPlanner,
    IndexedRetrievalExecutor,
    OriginalQueryLegExecutor,
    LocalRetrievalPipeline,
):
    """Coordinate retrieval pipeline stages while preserving RAGChatImpl behavior."""

    def __init__(self, rag_chat_impl: Any) -> None:
        self._rag_chat_impl = rag_chat_impl
        self.orchestratorHelper = OrchestratorHelper()
        self.orchestratorOverrides = OrchestratorOverrides(rag_chat_impl)
        self.orchestratorMessages = OrchestratorMessages(
            rag_chat_impl,
            overrides=self.orchestratorOverrides,
            normalize_language_bucket=self._normalize_language_bucket,
        )
        self._active_indexed_stage_iterations: list[GraphRegexStageIteration] | None = (
            None
        )
        self._suppress_indexed_retriever_dispatch_trace = False
        self._indexed_stage_shape_queries = True
        self._indexed_alternate_queries: list[str] = []
        self._prepared_indexed_stage_iterations: (
            list[GraphRegexStageIteration] | None
        ) = None
        # Turn-scoped flow toggles; defaults preserve current behavior.
        self._flow_enable_indexed_query_shaping = True
        self._flow_enable_indexed_query_translation = True
        self._flow_enable_indexed_query_wordnet_expansion = True
        self._indexed_flow_name = "N/A"
        self._indexed_primary_query_source = "REWRITTEN_QUERY"
        self._indexed_guardrail_query_source = "TRANSLATED_QUERY"
        self._indexed_guardrail_enabled = True
        self._rerank_language_queries: list[dict[str, Any]] = []
        self._flow_enable_original_language_vector_leg = True
        self._flow_enable_vector_retriever = True
        self._flow_enable_bm25_retriever = True
        self._flow_enable_graph_retriever = True
        self._flow_enable_regex_retriever = True
        self._respect_legacy_retrieve_mode = True

    @staticmethod
    def _is_guardrail_query_active(primary_query: str, guardrail_query: str) -> bool:
        """Return True when guardrail query differs from primary query."""
        return bool(
            guardrail_query
            and guardrail_query.lower() != str(primary_query or "").strip().lower()
        )

    def _resolve_guardrail_inputs_for_flow(
        self,
        mySession: Session,
        *,
        plan: RetrievalPlan,
        flow_profile: OrchestrationFlowProfile,
    ) -> GuardrailInputs:
        """Resolve primary/guardrail retrieval queries using flow selectors."""
        primary_query = self.orchestratorOverrides.query_from_source(
            mySession,
            plan=plan,
            source_name=flow_profile.primary_query_source,
        )
        guardrail_query = self.orchestratorOverrides.query_from_source(
            mySession,
            plan=plan,
            source_name=flow_profile.guardrail_query_source,
        )

        if not flow_profile.enable_guardrail_leg:
            return GuardrailInputs(
                primary_query=primary_query,
                guardrail_query="",
                use_guardrail=False,
            )

        return GuardrailInputs(
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=self._is_guardrail_query_active(
                primary_query,
                guardrail_query,
            ),
        )

    @staticmethod
    def _apply_original_leg_flow_gate(
        original_query_leg: OriginalQueryLeg,
        *,
        enable_original_language_vector_leg: bool,
    ) -> OriginalQueryLeg:
        """Apply flow-level on/off gate to original-language leg."""
        if enable_original_language_vector_leg:
            return original_query_leg
        return OriginalQueryLeg(
            enabled=False,
            query=original_query_leg.query,
            language_bucket=original_query_leg.language_bucket,
            reason="disabled by orchestration flow",
        )

    @staticmethod
    def _is_english_language_bucket(language_bucket: str | None) -> bool:
        """Return True when *language_bucket* represents English."""
        text = str(language_bucket or "").strip().lower()
        if not text:
            return False
        return text == "en" or text.startswith("en-") or text == "english"

    def _resolve_original_query_leg(
        self,
        mySession: Session,
        *,
        primary_query: str,
        guardrail_query: str,
    ) -> OriginalQueryLeg:
        """Decide whether to run a native-language vector retrieval leg."""
        original_query = str(
            getattr(mySession, "user_query_original", None)
            or getattr(mySession, "query", None)
            or ""
        ).strip()
        if not original_query:
            return OriginalQueryLeg(
                enabled=False,
                query="",
                language_bucket=None,
                reason="missing original query",
            )

        language_bucket = self._normalize_language_bucket(
            getattr(mySession, "user_language", None)
        )
        if not language_bucket:
            return OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=None,
                reason="missing user-language detection",
            )

        if self._is_english_language_bucket(language_bucket):
            return OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="user query already in English",
            )

        lower_original = original_query.lower()
        if lower_original == (primary_query or "").strip().lower():
            return OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="original query equals primary retrieval query",
            )
        if guardrail_query and lower_original == guardrail_query.strip().lower():
            return OriginalQueryLeg(
                enabled=False,
                query=original_query,
                language_bucket=language_bucket,
                reason="original query equals guardrail retrieval query",
            )

        return OriginalQueryLeg(
            enabled=True,
            query=original_query,
            language_bucket=language_bucket,
            reason="non-English original query differs from English retrieval legs",
        )

    def _prepare_session(self, mySession: Session) -> str:
        return self._rag_chat_impl._prepare_session(
            mySession
        )  # pyright: ignore[reportPrivateUsage]

    def _normalize_query(
        self,
        mySession: Session,
        user_query_original: str,
    ) -> tuple[str, list[str]]:
        return self._rag_chat_impl._normalize_query(
            mySession, user_query_original
        )  # pyright: ignore[reportPrivateUsage]

    def _check_gates(self, mySession: Session, final_query: str) -> bool:
        return self._rag_chat_impl._check_gates(
            mySession, final_query
        )  # pyright: ignore[reportPrivateUsage]

    def _resolve_plan(self, mySession: Session) -> RetrievalPlan:
        user_query_original = self._prepare_session(mySession)
        final_query, alternate_queries = self._normalize_query(
            mySession,
            user_query_original,
        )

        retrieve_mode: str = (mySession.retrieve_mode or "VECTOR").upper()
        bm25_query: str = mySession.query or ""
        orig_translated_query_en: str = (
            getattr(mySession, "orig_translated_query_en", None)
            or getattr(mySession, "seed_retrieval_query", None)
            or getattr(mySession, "t1_query", None)
            or ""
        )
        return RetrievalPlan(
            user_query_original=user_query_original,
            final_query=final_query,
            alternate_queries=alternate_queries,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            orig_translated_query_en=orig_translated_query_en,
        )

    def run_web_retriever(
        self,
        mySession: Session,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Run web retrieval and optional BM25/cosine pre-filters."""
        web_triggered: bool = mySession.web_search or retrieve_mode == "WEB"
        if not web_triggered or not self._is_web_search_allowed(retrieve_mode):
            return []
        web_docs = self._fetch_web_docs(
            mySession,
            retrieve_mode=retrieve_mode,
            user_query_original=user_query_original,
        )
        return self._apply_web_prefilters(mySession, web_docs)

    def _is_web_search_allowed(self, retrieve_mode: str) -> bool:
        """Return whether WEB_SEARCH_MODE permits web retrieval; warn when blocked."""
        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        if web_mode == "1":
            return True
        if web_mode != "0":
            message = f"Web search blocked (WEB_SEARCH_MODE = {web_mode!r}) — skipping"
        elif retrieve_mode == "WEB":
            message = (
                'retrieve_mode=WEB requested but WEB_SEARCH_MODE="0" '
                "— no results will be returned."
            )
        else:
            message = (
                'Web search blocked by administrator (WEB_SEARCH_MODE = "0") — skipping'
            )
        self.orchestratorMessages.write("W", "Web", message)
        return False

    def _fetch_web_docs(
        self,
        mySession: Session,
        *,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Query the web retriever and emit status/perf/debug output."""
        label = (
            "retrieve_mode=WEB — querying web only..."
            if retrieve_mode == "WEB"
            else "Querying web search..."
        )
        self.orchestratorMessages.write("I", "Web", label)
        self._rag_chat_impl.perf_logger.log(
            "RAGChatImpl._retrieve", "chat", "start web block"
        )
        _t_web = time.perf_counter()
        web_fetch_k = getattr(mySession, "web_fetch_k", None)
        if web_fetch_k is None:
            web_fetch_k = (
                self._rag_chat_impl.cfg.get_int("_WEB_SEARCH.max_results") or 5
            )
        web_docs = self._rag_chat_impl.web_retriever.query(
            mySession.query or "",
            k=web_fetch_k,
            fetch_page_content=bool(mySession.fetch_page_content),
            original_query=user_query_original,
            collection=mySession.collection_name or "",
        )
        self.orchestratorMessages.write(
            "O",
            "Web",
            f"Web search returned {len(web_docs)} results",
            color=CYAN,
        )
        self._rag_chat_impl.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  web block n={len(web_docs)} elapsed={time.perf_counter() - _t_web:.3f}s",
        )
        if DebugHelper.check_session(mySession, 10):
            self._rag_chat_impl._print_web_debug(
                web_docs
            )  # pyright: ignore[reportPrivateUsage]
        return web_docs

    def _apply_web_prefilters(
        self, mySession: Session, web_docs: list[Any]
    ) -> list[Any]:
        """Apply optional BM25/cosine pre-filters to trim noisy web snippets before merge."""
        pre_bm25: float = (
            self._rag_chat_impl.cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        )
        pre_cosine: float = (
            self._rag_chat_impl.cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        )
        if not web_docs or (pre_bm25 <= 0.0 and pre_cosine <= 0.0):
            return web_docs
        if DebugHelper.check_session(mySession, 30):
            self.orchestratorMessages.write(
                "D",
                "WebPreFilter",
                f"Pre-filtering {len(web_docs)} web result(s) — "
                f"bm25_pre_filter={pre_bm25:.3f}, "
                f"cosine_pre_filter={pre_cosine:.3f}",
                color=CYAN,
            )
        before_pre = len(web_docs)
        query = mySession.query or ""
        web_pre_filter = self._rag_chat_impl.web_pre_filter
        if pre_bm25 > 0.0:
            web_docs = self._run_web_prefilter(
                mySession,
                docs_before=web_docs,
                docs_after=web_pre_filter.bm25_prefilter(web_docs, query),
                before_pre=before_pre,
                label="BM25",
                threshold=pre_bm25,
            )
        if pre_cosine > 0.0 and web_docs:
            query_vec: list[float] = self._rag_chat_impl.embedder.embed_query(query)
            web_docs = self._run_web_prefilter(
                mySession,
                docs_before=web_docs,
                docs_after=web_pre_filter.cosine_prefilter(
                    web_docs, query, query_vec=query_vec
                ),
                before_pre=before_pre,
                label="Cosine",
                threshold=pre_cosine,
            )
        return web_docs

    def _run_web_prefilter(
        self,
        mySession: Session,
        *,
        docs_before: list[Any],
        docs_after: list[Any],
        before_pre: int,
        label: str,
        threshold: float,
    ) -> list[Any]:
        """Emit debug output for one pre-filter pass and return its kept docs."""
        if DebugHelper.check_session(mySession, 30):
            display_label = "BM25" if label == "BM25" else label.lower()
            self.orchestratorMessages.write(
                "D",
                "WebPreFilter",
                f"After {display_label} pre-filter: {len(docs_after)}/{before_pre} kept",
                color=CYAN,
            )
            kept_ids = {id(d) for d in docs_after}
            dropped = [d for d in docs_before if id(d) not in kept_ids]
            self._rag_chat_impl._print_web_prefilter_debug(  # pyright: ignore[reportPrivateUsage]
                docs_after,
                dropped,
                label,
                threshold,
            )
        return docs_after

    def _merge_and_build_context(
        self,
        mySession: Session,
        local_docs: LocalDocs,
        web_docs: list[Any],
    ) -> Tuple[str, int]:
        chosen = self._rag_chat_impl._merge_and_select(  # pyright: ignore[reportPrivateUsage]
            mySession,
            local_docs.vector_docs,
            local_docs.bm25_docs,
            local_docs.graph_docs,
            local_docs.regex_docs,
            web_docs,
        )
        return self._rag_chat_impl._build_context(
            mySession, chosen
        )  # pyright: ignore[reportPrivateUsage]

    def _run_merge_and_context_stage(
        self,
        mySession: Session,
        local_docs: LocalDocs,
        web_docs: list[Any],
    ) -> Tuple[str, int]:
        """Merge docs and build final retrieval context."""
        self.orchestratorMessages.trace_stage(
            mySession,
            stage_name="merge candidates and build context",
        )
        context, count = self._merge_and_build_context(mySession, local_docs, web_docs)
        self.orchestratorMessages.trace(
            mySession,
            f"retrieval completed chosen={count} context_chars={len(context)}",
        )

        self.orchestratorMessages.log_step(
            mySession,
            step_name="merge candidates and build context",
            status="executed",
            confidence_level="HIGH" if count > 0 else "LOW",
            confidence_score=(1.0 if count > 0 else 0.0),
            detail=f"chosen_chunks={count} context_chars={len(context)}",
            source="orchestrator",
        )
        return context, count

    def _resolve_guardrail_inputs(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> GuardrailInputs:
        """Resolve guardrail query inputs using RAGChatImpl helper semantics."""
        (
            primary_query,
            guardrail_query,
            use_guardrail,
        ) = self._rag_chat_impl._resolve_guardrail_queries(  # pyright: ignore[reportPrivateUsage]
            bm25_query=bm25_query,
            orig_translated_query_en=orig_translated_query_en,
        )
        return GuardrailInputs(
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
        )

    def _resolve_stage_execution_gates(
        self,
        mySession: Session,
        *,
        retrieve_mode: str,
    ) -> StageExecutionGates:
        """Resolve which retrieval stages run for this turn."""
        web_mode = self.orchestratorOverrides.resolve_web_stage_mode(
            mySession, retrieve_mode
        )
        return StageExecutionGates(
            web_mode=web_mode,
            local_stage_enabled=web_mode != "web_only",
            web_stage_enabled=web_mode in ("web_only", "local_and_web"),
        )

    @staticmethod
    def _empty_local_docs() -> LocalDocs:
        """Return an empty local-doc container for web-only retrieval."""
        return LocalDocs(
            vector_docs=[],
            bm25_docs=[],
            graph_docs=[],
            regex_docs=[],
        )

    def _run_post_gate_pipeline(
        self,
        mySession: Session,
        *,
        pipeline_inputs: PostGatePipelineInputs,
    ) -> Tuple[str, int]:
        """Run local/web retrieval and context assembly after gate checks pass."""
        stage_gates = self._resolve_stage_execution_gates(
            mySession,
            retrieve_mode=pipeline_inputs.retrieve_mode,
        )
        self.orchestratorMessages.trace_effective_stage_gates(
            mySession,
            flow_name=pipeline_inputs.flow_name,
            retrieve_mode=pipeline_inputs.retrieve_mode,
            gates=stage_gates,
        )
        self._configure_post_gate_flow(pipeline_inputs)
        local_docs = self._run_local_pipeline_stage(
            mySession,
            pipeline_inputs=pipeline_inputs,
            stage_gates=stage_gates,
        )
        web_docs = self._run_web_pipeline_stage(
            mySession,
            pipeline_inputs=pipeline_inputs,
            stage_gates=stage_gates,
        )
        return self._run_merge_and_context_stage(mySession, local_docs, web_docs)

    def _configure_post_gate_flow(
        self, pipeline_inputs: PostGatePipelineInputs
    ) -> None:
        """Set the turn's resolved flow controls before executing retrieval."""
        self._indexed_flow_name = pipeline_inputs.flow_name
        self._indexed_primary_query_source = pipeline_inputs.primary_query_source
        self._indexed_guardrail_query_source = pipeline_inputs.guardrail_query_source
        self._indexed_guardrail_enabled = pipeline_inputs.enable_guardrail_leg
        self._flow_enable_indexed_query_shaping = bool(
            pipeline_inputs.enable_indexed_query_shaping
        )
        self._flow_enable_indexed_query_translation = bool(
            pipeline_inputs.enable_indexed_query_translation
        )
        self._flow_enable_indexed_query_wordnet_expansion = bool(
            pipeline_inputs.enable_indexed_query_wordnet_expansion
        )
        self._flow_enable_original_language_vector_leg = bool(
            pipeline_inputs.enable_original_language_vector_leg
        )
        self._flow_enable_vector_retriever = bool(
            pipeline_inputs.enable_vector_retriever
        )
        self._flow_enable_bm25_retriever = bool(pipeline_inputs.enable_bm25_retriever)
        self._flow_enable_graph_retriever = bool(pipeline_inputs.enable_graph_retriever)
        self._flow_enable_regex_retriever = bool(pipeline_inputs.enable_regex_retriever)

    def _run_web_pipeline_stage(
        self,
        mySession: Session,
        *,
        pipeline_inputs: PostGatePipelineInputs,
        stage_gates: StageExecutionGates,
    ) -> list[Any]:
        """Execute or skip web retrieval and report its confidence event."""
        web_docs: list[Any] = []
        if stage_gates.web_stage_enabled:
            self.orchestratorMessages.trace_stage(
                mySession,
                stage_name="retrieve web candidates",
            )
            web_docs = self.run_web_retriever(
                mySession,
                retrieve_mode=pipeline_inputs.retrieve_mode,
                user_query_original=pipeline_inputs.user_query_original,
            )
            self.orchestratorMessages.trace(
                mySession, f"web docs fetched count={len(web_docs)}"
            )

            self.orchestratorMessages.log_step(
                mySession,
                step_name="retrieve web candidates",
                status="executed",
                confidence_level=("HIGH" if len(web_docs) > 0 else "LOW"),
                confidence_score=(1.0 if len(web_docs) > 0 else 0.0),
                detail=f"web_docs={len(web_docs)}",
                source="orchestrator",
            )
        else:
            self.orchestratorMessages.trace(
                mySession,
                "web stage skipped " f"(web_mode={stage_gates.web_mode})",
            )

            self.orchestratorMessages.log_step(
                mySession,
                step_name="retrieve web candidates",
                status="skipped",
                confidence_level="N/A",
                detail=(f"web_mode={stage_gates.web_mode}"),
                source="orchestrator",
            )

        return web_docs

    def _run_pre_gate_pipeline(
        self,
        mySession: Session,
        *,
        flow_profile: OrchestrationFlowProfile | None = None,
    ) -> PreGatePipelineResult:
        """Prepare query inputs and resolve post-gate pipeline inputs."""
        flow_profile = self._prepare_session_flow_profile(mySession, flow_profile)
        plan = self._normalize_user_query(mySession)
        if self._is_retrieval_gated(mySession, plan):
            return PreGatePipelineResult(
                should_abort=True,
                post_gate_inputs=None,
            )
        if getattr(mySession, "enable_keyword_action_splitter", False):
            splitter = KeywordActionSplitter(
                cfg=self._rag_chat_impl.cfg,
                pretty=self._rag_chat_impl.pretty,
                helpers=self._rag_chat_impl.helperInstance,
            )
            plan.keyword_action_split = splitter.split(mySession)
            mySession.keyword_action_split = plan.keyword_action_split
            mySession.keyword_action_status = (
                "split" if plan.keyword_action_split is not None else "fallback"
            )
            if plan.keyword_action_split is None:
                plan.alternate_queries = self._rag_chat_impl._generate_alternate_queries(  # pyright: ignore[reportPrivateUsage]
                    plan.final_query, mySession
                )
            elif not plan.keyword_action_split.retrieval_queries:
                mySession.keyword_action_status = "unsupported_no_retrieval"
                mySession.clarification_response = (
                    "This evidence-only pipeline requires a document or information topic "
                    "to retrieve. Standalone actions without retrieval are not supported. "
                    "For a complete indexed document summary, use summarize: followed by a filename."
                )
                self.orchestratorMessages.write(
                    "W", "KeywordActionSplitter", mySession.clarification_response
                )
                return PreGatePipelineResult(should_abort=True, post_gate_inputs=None)
        elif DebugHelper.check_session(mySession, 30):
            self.orchestratorMessages.write(
                "D",
                "KeywordActionSplitter",
                "Disabled; using existing retrieval and answer pipeline.",
            )
        retrieve_mode = self.orchestratorOverrides.resolve_retrieve_mode_for_flow(
            default_retrieve_mode=plan.retrieve_mode,
            flow_profile=flow_profile,
        )
        # Not persisted on the session: clearing force_retrieve_mode restores run_*.
        flow_profile = self.orchestratorOverrides.narrow_retrievers_for_mode(
            flow_profile,
            retrieve_mode,
        )
        self.orchestratorMessages.emit_flow_knob_stage_statuses(
            mySession=mySession,
            flow_profile=flow_profile,
            retrieve_mode=retrieve_mode,
        )

        guardrail_inputs = self._resolve_guardrail_inputs_for_flow(
            mySession,
            plan=plan,
            flow_profile=flow_profile,
        )
        self.orchestratorOverrides.apply_rerank_query_source(
            mySession,
            plan=plan,
            flow_profile=flow_profile,
        )
        original_query_leg = self._resolve_original_query_leg(
            mySession,
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
        )
        original_query_leg = self._apply_original_leg_flow_gate(
            original_query_leg,
            enable_original_language_vector_leg=flow_profile.enable_original_language_vector_leg,
        )
        self.orchestratorMessages.emit_original_query_leg_status(
            original_query_leg=original_query_leg,
        )

        alternate_queries = (
            list(plan.alternate_queries)
            if flow_profile.enable_vector_alternate_queries
            else []
        )
        self._indexed_alternate_queries = (
            list(plan.alternate_queries)
            if guardrail_inputs.primary_query == plan.final_query
            else []
        )

        self.orchestratorMessages.trace_stage_plan(
            mySession,
            flow_profile=flow_profile,
            retrieve_mode=retrieve_mode,
            use_guardrail=guardrail_inputs.use_guardrail,
            original_leg_enabled=original_query_leg.enabled,
            alternate_query_count=len(alternate_queries),
        )
        return PreGatePipelineResult(
            should_abort=False,
            post_gate_inputs=self._build_post_gate_inputs(
                plan=plan,
                flow_profile=flow_profile,
                retrieve_mode=retrieve_mode,
                guardrail_inputs=guardrail_inputs,
                alternate_queries=alternate_queries,
            ),
        )

    def _prepare_session_flow_profile(
        self,
        mySession: Session,
        flow_profile: OrchestrationFlowProfile | None,
    ) -> OrchestrationFlowProfile:
        """Resolve the effective flow profile and apply its query-rewrite controls."""
        self.orchestratorMessages.trace_stage(
            mySession, stage_name="prepare session context"
        )
        if flow_profile is None:
            flow_profile = (
                self.orchestratorOverrides.resolve_orchestration_flow_profile(mySession)
            )
        flow_profile = (
            self.orchestratorOverrides.resolve_effective_flow_profile_for_session(
                mySession,
                flow_profile,
            )
        )
        self.orchestratorOverrides.apply_query_rewrite_flow_controls(
            mySession,
            flow_profile=flow_profile,
        )
        return flow_profile

    def _normalize_user_query(self, mySession: Session) -> RetrievalPlan:
        """Build the retrieval plan (final + alternate queries) and report it."""
        self.orchestratorMessages.trace_stage(
            mySession, stage_name="normalize user query"
        )
        plan = self._resolve_plan(mySession)
        self.orchestratorMessages.emit_query_language_rewrite_status(
            mySession, plan=plan
        )
        self.orchestratorMessages.trace(
            mySession,
            "normalized query "
            f"final={plan.final_query!r} alternate_count={len(plan.alternate_queries)}",
        )
        return plan

    def _is_retrieval_gated(self, mySession: Session, plan: RetrievalPlan) -> bool:
        """Apply policy/retrieval gates; log the skipped post-gate step when blocked."""
        self.orchestratorMessages.trace_stage(
            mySession, stage_name="apply retrieval gates"
        )
        if not self._check_gates(mySession, plan.final_query):
            return False
        self.orchestratorMessages.trace(
            mySession, "gated: retrieval aborted by policy or retrieval gate"
        )
        self.orchestratorMessages.log_step(
            mySession,
            step_name="post-gate retrieval pipeline",
            status="skipped",
            confidence_level="N/A",
            detail="aborted by retrieval gate",
            source="orchestrator",
        )
        return True

    @staticmethod
    def _build_post_gate_inputs(
        *,
        plan: RetrievalPlan,
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str,
        guardrail_inputs: GuardrailInputs,
        alternate_queries: list[str],
    ) -> PostGatePipelineInputs:
        """Bundle the resolved plan and flow knobs for the post-gate pipeline."""
        return PostGatePipelineInputs(
            flow_name=flow_profile.name,
            primary_query_source=flow_profile.primary_query_source,
            guardrail_query_source=flow_profile.guardrail_query_source,
            enable_guardrail_leg=flow_profile.enable_guardrail_leg,
            retrieve_mode=retrieve_mode,
            bm25_query=guardrail_inputs.primary_query,
            alternate_queries=alternate_queries,
            orig_translated_query_en=plan.orig_translated_query_en,
            user_query_original=plan.user_query_original,
            resolved_guardrail=guardrail_inputs,
            enable_indexed_query_shaping=flow_profile.enable_indexed_query_shaping,
            enable_indexed_query_translation=flow_profile.enable_indexed_query_translation,
            enable_indexed_query_wordnet_expansion=(
                flow_profile.enable_indexed_query_wordnet_expansion
            ),
            enable_original_language_vector_leg=flow_profile.enable_original_language_vector_leg,
            enable_vector_retriever=flow_profile.enable_vector_retriever,
            enable_bm25_retriever=flow_profile.enable_bm25_retriever,
            enable_graph_retriever=flow_profile.enable_graph_retriever,
            enable_regex_retriever=flow_profile.enable_regex_retriever,
        )

    def run_local_retrievers(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
        resolved_guardrail: GuardrailInputs | None = None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Run local retrievers and guardrail fusion using RAGChatImpl retriever helpers.

        This method centralizes local retrieval orchestration while keeping the
        concrete retriever implementations on the RAGChatImpl object.
        """
        self._flow_enable_indexed_query_shaping = True
        self._flow_enable_indexed_query_translation = True
        self._flow_enable_indexed_query_wordnet_expansion = True
        self._flow_enable_original_language_vector_leg = True
        self._flow_enable_vector_retriever = True
        self._flow_enable_bm25_retriever = True
        self._flow_enable_graph_retriever = True
        self._flow_enable_regex_retriever = True
        self._rerank_language_queries = []
        self._indexed_flow_name = str(
            getattr(mySession, "orchestrator_flow", None) or "DIRECT"
        )
        self._indexed_primary_query_source = "REWRITTEN_QUERY"
        self._indexed_guardrail_query_source = "TRANSLATED_QUERY"
        self._indexed_guardrail_enabled = bool(
            resolved_guardrail and resolved_guardrail.use_guardrail
        )

        self._indexed_alternate_queries = list(alternate_queries)
        mySession.indexed_query_notices = []
        mySession.indexed_query_warning_keys = set()
        local_docs = self._run_local_docs_stage(
            mySession,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            alternate_queries=alternate_queries,
            orig_translated_query_en=orig_translated_query_en,
            resolved_guardrail=resolved_guardrail,
        )
        return (
            local_docs.vector_docs,
            local_docs.bm25_docs,
            local_docs.graph_docs,
            local_docs.regex_docs,
        )

    def run(self, mySession: Session) -> Tuple[str, int]:
        """Run one retrieval turn and return (context, chunk_count)."""

        flow_profile = self.orchestratorOverrides.resolve_orchestration_flow_profile(
            mySession
        )
        self.orchestratorMessages.write_retrieval_orchestration(
            severity="I",
            message=f"Chosen orchestration flow: {flow_profile.name}",
            color=ORANGE,
        )
        self.orchestratorMessages.emit_session_overrides(mySession)
        self.orchestratorMessages.trace(mySession, "start retrieval turn")

        if not self._rag_chat_impl._set_vector_store(
            mySession
        ):  # pyright: ignore[reportPrivateUsage]
            self.orchestratorMessages.trace(
                mySession, "vector store setup failed; aborting retrieval"
            )
            self.orchestratorMessages.log_step(
                mySession,
                step_name="set vector store",
                status="skipped",
                confidence_level="N/A",
                detail="vector store setup failed",
                source="orchestrator",
            )
            return "", 0

        self._rag_chat_impl.perf_logger.log(
            "RAGChatImpl._retrieve", "chat", "start retrieve"
        )

        pre_gate_result = self._run_pre_gate_pipeline(
            mySession,
            flow_profile=flow_profile,
        )
        if pre_gate_result.should_abort:
            return "", 0

        pipeline_inputs = pre_gate_result.post_gate_inputs
        if pipeline_inputs is None:
            self.orchestratorMessages.trace(
                mySession, "pre-gate pipeline produced no post-gate inputs"
            )
            self.orchestratorMessages.log_step(
                mySession,
                step_name="post-gate retrieval pipeline",
                status="skipped",
                confidence_level="N/A",
                detail="no post-gate inputs produced",
                source="orchestrator",
            )
            return "", 0

        return self._run_post_gate_pipeline(
            mySession,
            pipeline_inputs=pipeline_inputs,
        )
