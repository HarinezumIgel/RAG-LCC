"""Extracted LocalRetrievalPipeline behavior; the coordinating facade owns shared state."""

from __future__ import annotations

from typing import Any, cast

from Chat.KeywordActionSplitter import KeywordActionSplitter
from Commons.Exceptions import QueryTranslationError
from Globals.Session import Session
from Helpers.DebugHelper import DebugHelper
from Orchestrator.OrchestrationStageContext import OrchestrationStageContext
from Orchestrator.OrchestratorTypes import (GuardrailInputs, LocalDocs,
                                            LocalStageInputs, LocalStageResult,
                                            PostGatePipelineInputs,
                                            StageExecutionGates,
                                            VectorStageResult)


class LocalRetrievalPipeline(OrchestrationStageContext):
    def _run_vector_stage(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> VectorStageResult:
        """Run vector retrieval stage and return docs plus top-k counters."""
        vector_weight = float(
            mySession.vector_weight if mySession.vector_weight is not None else 1.0
        )
        if not vector_enabled or vector_weight == 0.0:
            return VectorStageResult(
                vector_docs=[],
                top_k_orig_query_en=0,
                top_k_post_rewrite_query_en=0,
                alternate_queries=[],
            )

        self.orchestratorMessages.emit_vector_query_language_flow(mySession)

        self.orchestratorMessages.trace_query_dispatch(
            mySession,
            retriever_scope="Vector",
            language_bucket=None,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
            include_language=False,
        )

        (
            vector_docs,
            vector_top_k_orig_query_en,
            vector_top_k_post_rewrite_query_en,
        ) = self._rag_chat_impl._run_vector_retriever_with_guardrail(
            mySession=mySession,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
            use_guardrail=use_guardrail,
            alternate_queries=alternate_queries,
        )
        return VectorStageResult(
            vector_docs=vector_docs,
            top_k_orig_query_en=vector_top_k_orig_query_en,
            top_k_post_rewrite_query_en=vector_top_k_post_rewrite_query_en,
            alternate_queries=list(alternate_queries),
        )

    def _resolve_local_stage_inputs(
        self,
        mySession: Session,
        *,
        retrieve_mode: str | None,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: GuardrailInputs | None,
    ) -> LocalStageInputs:
        """Resolve local stage flags, queries, and shared filters once."""
        vector_enabled = bool(self._flow_enable_vector_retriever)
        bm25_enabled = bool(self._flow_enable_bm25_retriever)
        graph_enabled = bool(self._flow_enable_graph_retriever)
        regex_enabled = bool(self._flow_enable_regex_retriever)

        # Backward compatibility: legacy seams may still pass retrieve_mode.
        mode_text = str(retrieve_mode or "").strip().upper()
        if mode_text and bool(getattr(self, "_respect_legacy_retrieve_mode", True)):
            (
                vector_mode_enabled,
                bm25_mode_enabled,
                graph_mode_enabled,
                regex_mode_enabled,
            ) = self._rag_chat_impl._mode_flags_for_local_retrievers(mode_text)
            vector_enabled = bool(vector_enabled and vector_mode_enabled)
            bm25_enabled = bool(bm25_enabled and bm25_mode_enabled)
            graph_enabled = bool(graph_enabled and graph_mode_enabled)
            regex_enabled = bool(regex_enabled and regex_mode_enabled)

        guardrail_inputs = resolved_guardrail
        if guardrail_inputs is None:
            guardrail_inputs = self._resolve_guardrail_inputs(
                bm25_query=bm25_query,
                orig_translated_query_en=orig_translated_query_en,
            )

        original_query_leg = self._resolve_original_query_leg(
            mySession,
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
        )
        original_query_leg = self._apply_original_leg_flow_gate(
            original_query_leg,
            enable_original_language_vector_leg=self._flow_enable_original_language_vector_leg,
        )

        return LocalStageInputs(
            vector_enabled=vector_enabled,
            bm25_enabled=bm25_enabled,
            graph_enabled=graph_enabled,
            regex_enabled=regex_enabled,
            indexed_query_shaping=bool(self._flow_enable_indexed_query_shaping),
            primary_query=guardrail_inputs.primary_query,
            guardrail_query=guardrail_inputs.guardrail_query,
            use_guardrail=guardrail_inputs.use_guardrail,
            alternate_queries=list(alternate_queries),
            file_filter=self._rag_chat_impl._resolve_file_filter(mySession),
            original_query_leg=original_query_leg,
        )

    def _run_local_retrieval_stages(
        self,
        mySession: Session,
        *,
        local_inputs: LocalStageInputs,
    ) -> LocalStageResult:
        """Validate the shared query matrix before executing any local searches."""
        previous = self._prepared_indexed_stage_iterations
        previous_shape = self._indexed_stage_shape_queries
        self._rerank_language_queries = []
        with self.orchestratorMessages.deferred_indexed_status():
            try:
                indexed_enabled = any(
                    enabled and getattr(mySession, weight_field, None) != 0.0
                    for enabled, weight_field in [
                        (local_inputs.bm25_enabled, "bm25_weight"),
                        (local_inputs.graph_enabled, "graph_weight"),
                        (local_inputs.regex_enabled, "regex_weight"),
                    ]
                )
                if indexed_enabled:
                    self._indexed_stage_shape_queries = (
                        local_inputs.indexed_query_shaping
                    )
                    self._prepared_indexed_stage_iterations = (
                        self._graph_regex_stage_iterations(
                            mySession,
                            file_filter=local_inputs.file_filter,
                            primary_query=local_inputs.primary_query,
                            guardrail_query=(
                                local_inputs.guardrail_query
                                if local_inputs.use_guardrail
                                else ""
                            ),
                        )
                    )
                    self._rerank_language_queries = [
                        {
                            "language": self._normalize_language_bucket(
                                iteration.language_bucket
                            )
                            or "N/A",
                            "query": iteration.primary_query,
                        }
                        for iteration in self._prepared_indexed_stage_iterations
                        if iteration.primary_query.strip()
                        and not iteration.is_alternate
                    ]
                    if (
                        not self._prepared_indexed_stage_iterations
                        and not local_inputs.vector_enabled
                        and not getattr(mySession, "web_search", False)
                    ):
                        message = "No usable indexed language queries or vector/web retrieval paths remain."
                        self.orchestratorMessages.write_retrieval_orchestration(
                            severity="E", message=message
                        )
                        raise QueryTranslationError(message)
                return self._execute_local_retrieval_stages(
                    mySession, local_inputs=local_inputs
                )
            finally:
                self._prepared_indexed_stage_iterations = previous
                self._indexed_stage_shape_queries = previous_shape

    def _execute_local_retrieval_stages(
        self,
        mySession: Session,
        *,
        local_inputs: LocalStageInputs,
    ) -> LocalStageResult:
        """Run local retrieval stages and accumulate guardrail top-k counters."""
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = 0

        vector_stage_result = self._run_vector_stage(
            mySession,
            vector_enabled=local_inputs.vector_enabled,
            primary_query=local_inputs.primary_query,
            guardrail_query=local_inputs.guardrail_query,
            use_guardrail=local_inputs.use_guardrail,
            alternate_queries=local_inputs.alternate_queries,
        )
        alternate_queries_used = cast(
            list[str],
            getattr(vector_stage_result, "alternate_queries", None) or [],
        )
        self.orchestratorMessages.emit_vector_alternate_queries_status(
            alternate_queries=alternate_queries_used,
        )
        top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._rag_chat_impl._accumulate_topk_counts(
                top_k_orig_query_en,
                top_k_post_rewrite_query_en,
                vector_stage_result.top_k_orig_query_en,
                vector_stage_result.top_k_post_rewrite_query_en,
            )
        )
        merged_vector_docs, native_vector_stats = self._merge_native_vector_leg(
            mySession,
            vector_docs=vector_stage_result.vector_docs,
            local_inputs=local_inputs,
        )

        indexed_specs = self._rag_chat_impl._indexed_retriever_specs(
            mySession,
            bm25_enabled=local_inputs.bm25_enabled,
            graph_enabled=local_inputs.graph_enabled,
            regex_enabled=local_inputs.regex_enabled,
        )
        indexed_stage_result = self._run_indexed_retrieval_with_shape_mode(
            mySession,
            indexed_specs=indexed_specs,
            primary_query=local_inputs.primary_query,
            guardrail_query=local_inputs.guardrail_query,
            use_guardrail=local_inputs.use_guardrail,
            file_filter=local_inputs.file_filter,
            shape_queries=local_inputs.indexed_query_shaping,
        )
        top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._rag_chat_impl._accumulate_topk_counts(
                top_k_orig_query_en,
                top_k_post_rewrite_query_en,
                indexed_stage_result.top_k_orig_query_en,
                indexed_stage_result.top_k_post_rewrite_query_en,
            )
        )

        native_indexed_stage_result = self._run_original_query_indexed_leg(
            mySession,
            original_query_leg=local_inputs.original_query_leg,
            indexed_specs=indexed_specs,
            file_filter=local_inputs.file_filter,
        )
        merged_indexed_docs, native_indexed_stats = self._merge_native_indexed_leg(
            indexed_stage_result,
            native_indexed_stage_result,
        )
        self._store_original_query_leg_stats(
            mySession,
            original_query_leg=local_inputs.original_query_leg,
            **native_vector_stats,
            **native_indexed_stats,
        )

        return LocalStageResult(
            local_docs=LocalDocs(
                vector_docs=merged_vector_docs,
                bm25_docs=merged_indexed_docs["bm25"],
                graph_docs=merged_indexed_docs["graph"],
                regex_docs=merged_indexed_docs["regex"],
            ),
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _store_guardrail_topk_if_enabled(
        self,
        mySession: Session,
        *,
        use_guardrail: bool,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None:
        """Persist guardrail top-k counters only when guardrail execution is active."""
        if not use_guardrail:
            return
        self._rag_chat_impl._store_guardrail_retrieval_topk(
            mySession,
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _run_local_docs_stage(
        self,
        mySession: Session,
        *,
        retrieve_mode: str | None,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
        resolved_guardrail: GuardrailInputs | None,
    ) -> LocalDocs:
        """Run local stages end-to-end and return assembled local docs."""
        local_inputs = self._resolve_local_stage_inputs(
            mySession,
            retrieve_mode=retrieve_mode,
            bm25_query=bm25_query,
            alternate_queries=alternate_queries,
            orig_translated_query_en=orig_translated_query_en,
            resolved_guardrail=resolved_guardrail,
        )
        local_stage_result = (
            self._run_split_local_stages(mySession, local_inputs)
            if getattr(mySession, "keyword_action_split", None) is not None
            else self._run_local_retrieval_stages(mySession, local_inputs=local_inputs)
        )

        self._store_guardrail_topk_if_enabled(
            mySession,
            use_guardrail=local_inputs.use_guardrail,
            top_k_orig_query_en=local_stage_result.top_k_orig_query_en,
            top_k_post_rewrite_query_en=local_stage_result.top_k_post_rewrite_query_en,
        )

        return local_stage_result.local_docs

    def _run_split_local_stages(
        self, mySession: Session, local_inputs: LocalStageInputs
    ) -> LocalStageResult:
        """Reuse the existing local stages for each need; run safety-net legs once."""
        from dataclasses import replace

        split = mySession.keyword_action_split
        if split is None:
            raise ValueError("Split local retrieval requires validated decomposition")
        ranked: dict[str, list[tuple[list[Any], float]]] = {
            label: [] for label in ("vector", "bm25", "graph", "regex")
        }
        total_seed = 0
        total_final = 0
        language_queries: list[dict[str, Any]] = []
        original_leg_stats: dict[str, Any] = {}
        previous_alternates = self._indexed_alternate_queries
        self._indexed_alternate_queries = []
        try:
            for query_id, query in enumerate(split.retrieval_queries):
                # Fan out only the main local query. Guardrail and native legs
                # still belong to the original turn and must run only once.
                inputs = replace(
                    local_inputs,
                    primary_query=query,
                    alternate_queries=[],
                    use_guardrail=local_inputs.use_guardrail and query_id == 0,
                    guardrail_query=(
                        local_inputs.guardrail_query if query_id == 0 else ""
                    ),
                    original_query_leg=(
                        local_inputs.original_query_leg
                        if query_id == 0
                        else replace(
                            local_inputs.original_query_leg,
                            enabled=False,
                            reason="already run for this turn",
                        )
                    ),
                )
                if DebugHelper.check_session(mySession, 30):
                    self.orchestratorMessages.write(
                        "D",
                        "KeywordActionSplitter",
                        f"Local query_id={query_id}: {query!r}",
                    )
                result = self._run_local_retrieval_stages(
                    mySession, local_inputs=inputs
                )
                mySession.keyword_action_local_needs += 1
                if query_id == 0:
                    original_leg_stats = {
                        key: value
                        for key, value in vars(mySession).items()
                        if key.startswith("original_query_leg_")
                    }
                language_queries.extend(self._rerank_language_queries)
                mySession.keyword_action_language_queries[query_id] = {
                    str(item["language"]): str(item["query"])
                    for item in self._rerank_language_queries
                }
                counts: dict[str, int] = {}
                for label in ranked:
                    docs: list[Any] = getattr(result.local_docs, f"{label}_docs")
                    counts[label] = len(docs)
                    ranked[label].append((docs, 1.0))
                    for doc in docs:
                        # Track associations outside document metadata so they
                        # remain turn-local and survive fusion of document copies.
                        identity = self.orchestratorHelper.doc_identity(doc)
                        ids = mySession.keyword_action_query_ids.setdefault(
                            identity, []
                        )
                        if query_id not in ids:
                            ids.append(query_id)
                total_seed += result.top_k_orig_query_en
                total_final += result.top_k_post_rewrite_query_en
                if DebugHelper.check_session(mySession, 30):
                    self.orchestratorMessages.write(
                        "D",
                        "KeywordActionSplitter",
                        f"query_id={query_id} hits: {counts}",
                    )

            if local_inputs.bm25_enabled and mySession.bm25_weight != 0.0:
                # The combined keyword leg has no single associated need.
                # Its additional chunks are scored against all needs later.
                keyword_query = " ".join(split.key_words)
                if DebugHelper.check_session(mySession, 30):
                    self.orchestratorMessages.write(
                        "D",
                        "KeywordActionSplitter",
                        f"BM25 keyword query: {keyword_query!r}",
                    )
                keyword_inputs = replace(
                    local_inputs,
                    vector_enabled=False,
                    graph_enabled=False,
                    regex_enabled=False,
                    primary_query=keyword_query,
                    guardrail_query="",
                    use_guardrail=False,
                    alternate_queries=[],
                    original_query_leg=replace(
                        local_inputs.original_query_leg,
                        enabled=False,
                        reason="keyword leg",
                    ),
                )
                keywords = self._run_local_retrieval_stages(
                    mySession, local_inputs=keyword_inputs
                )
                mySession.keyword_action_keyword_leg = True
                ranked["bm25"].append((keywords.local_docs.bm25_docs, 1.0))
                total_final += keywords.top_k_post_rewrite_query_en
        finally:
            self._indexed_alternate_queries = previous_alternates
            self._rerank_language_queries = language_queries
            for key, value in original_leg_stats.items():
                setattr(mySession, key, value)
        rrf_k = self._rag_chat_impl.bm25_retriever.rrf_k
        merged = {
            label: self.orchestratorHelper.merge_ranked_variants(lists, rrf_k=rrf_k)
            for label, lists in ranked.items()
        }
        return LocalStageResult(
            LocalDocs(
                merged["vector"], merged["bm25"], merged["graph"], merged["regex"]
            ),
            total_seed,
            total_final,
        )

    def _run_local_pipeline_stage(
        self,
        mySession: Session,
        *,
        pipeline_inputs: PostGatePipelineInputs,
        stage_gates: StageExecutionGates,
    ) -> LocalDocs:
        """Execute or skip local retrieval and report its confidence event."""
        local_docs = self._empty_local_docs()
        if stage_gates.local_stage_enabled:
            self.orchestratorMessages.trace_stage(
                mySession,
                stage_name="retrieve local candidates",
                details=(f"mode={pipeline_inputs.retrieve_mode}"),
            )
            previous_mode_override_flag = self._respect_legacy_retrieve_mode
            self._respect_legacy_retrieve_mode = False
            try:
                local_docs = self._run_local_docs_stage(
                    mySession,
                    retrieve_mode=pipeline_inputs.retrieve_mode,
                    bm25_query=pipeline_inputs.bm25_query,
                    alternate_queries=pipeline_inputs.alternate_queries,
                    orig_translated_query_en=pipeline_inputs.orig_translated_query_en,
                    resolved_guardrail=pipeline_inputs.resolved_guardrail,
                )
            finally:
                self._respect_legacy_retrieve_mode = previous_mode_override_flag
            self.orchestratorMessages.trace(
                mySession,
                "local docs fetched "
                f"vector={len(local_docs.vector_docs)} "
                f"bm25={len(local_docs.bm25_docs)} "
                f"graph={len(local_docs.graph_docs)} regex={len(local_docs.regex_docs)}",
            )

            total_local_docs = (
                len(local_docs.vector_docs)
                + len(local_docs.bm25_docs)
                + len(local_docs.graph_docs)
                + len(local_docs.regex_docs)
            )
            self.orchestratorMessages.log_step(
                mySession,
                step_name="retrieve local candidates",
                status="executed",
                confidence_level=("HIGH" if total_local_docs > 0 else "LOW"),
                confidence_score=(1.0 if total_local_docs > 0 else 0.0),
                detail=(
                    f"vector={len(local_docs.vector_docs)} "
                    f"bm25={len(local_docs.bm25_docs)} "
                    f"graph={len(local_docs.graph_docs)} "
                    f"regex={len(local_docs.regex_docs)}"
                ),
                source="orchestrator",
            )
        else:
            self.orchestratorMessages.trace(
                mySession,
                "local stage skipped " f"(web_mode={stage_gates.web_mode})",
            )

            self.orchestratorMessages.log_step(
                mySession,
                step_name="retrieve local candidates",
                status="skipped",
                confidence_level="N/A",
                detail=(f"web_mode={stage_gates.web_mode}"),
                source="orchestrator",
            )
        return local_docs
