"""Extracted IndexedRetrievalExecutor behavior; the coordinating facade owns shared state."""

from __future__ import annotations

import time
from typing import Any, cast

from Globals.Session import Session
from Orchestrator.OrchestrationStageContext import OrchestrationStageContext
from Orchestrator.OrchestratorTypes import (INDEXED_RETRIEVER_LABELS,
                                            Bm25StageInvocation,
                                            GraphRegexStageIteration,
                                            IndexedSpec, IndexedStageResult)


class IndexedRetrievalExecutor(OrchestrationStageContext):
    @staticmethod
    def _partition_indexed_specs(
        indexed_specs: list[IndexedSpec],
    ) -> tuple[list[IndexedSpec], list[IndexedSpec]]:
        """Split indexed specs into BM25 and Graph/Regex execution stages."""
        active_specs = [spec for spec in indexed_specs if spec[1] and spec[2] != 0.0]
        bm25_specs = [spec for spec in active_specs if spec[0] == "BM25"]
        graph_regex_specs = [
            spec for spec in active_specs if spec[0] in ("Graph", "Regex")
        ]
        return bm25_specs, graph_regex_specs

    def _run_indexed_specs_stage(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run one indexed-spec stage through the RAGChatImpl retriever executor."""
        if not indexed_specs:
            empty_docs: dict[str, list[Any]] = {
                label: [] for label in INDEXED_RETRIEVER_LABELS
            }
            return empty_docs, 0, 0

        stage_docs, top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._rag_chat_impl._run_indexed_local_retrievers(
                mySession,
                indexed_specs=indexed_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
        )

        normalized_docs: dict[str, list[Any]] = {
            label: list(stage_docs.get(label, [])) for label in INDEXED_RETRIEVER_LABELS
        }
        return normalized_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _stage_invocation_values(
        self,
        stage_invocation: Any,
    ) -> tuple[str | None, dict[str, Any] | None, str, str]:
        """Extract stage language/filter/query values from an invocation object."""
        stage_file_filter_obj = getattr(stage_invocation, "file_filter", None)
        stage_file_filter: dict[str, Any] | None = (
            cast(dict[str, Any], stage_file_filter_obj)
            if isinstance(stage_file_filter_obj, dict)
            else None
        )
        stage_primary_query = str(getattr(stage_invocation, "primary_query", "") or "")
        stage_guardrail_query = str(
            getattr(stage_invocation, "guardrail_query", "") or ""
        )
        stage_language_bucket = self._normalize_language_bucket(
            getattr(stage_invocation, "language_bucket", None)
        )
        if stage_language_bucket is None:
            stage_language_bucket = self._language_bucket_from_filter(stage_file_filter)
        return (
            stage_language_bucket,
            stage_file_filter,
            stage_primary_query,
            stage_guardrail_query,
        )

    def _run_indexed_stage_group(
        self,
        mySession: Session,
        *,
        stage_invocations: list[Any],
        specs: list[IndexedSpec],
        use_guardrail: bool,
        dispatch_per_spec: bool,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run indexed stage invocations for one spec group.

        ``dispatch_per_spec=False`` executes all ``specs`` together per stage
        invocation (BM25 behavior). ``dispatch_per_spec=True`` executes one spec
        at a time per stage invocation (Graph/Regex behavior).
        """
        merged_docs: dict[str, list[Any]] = {
            label: [] for label in INDEXED_RETRIEVER_LABELS
        }
        if not specs or not stage_invocations:
            return merged_docs, 0, 0

        total_top_k_orig_query_en = 0
        total_top_k_post_rewrite_query_en = 0
        ranked_results: dict[str, list[tuple[list[Any], float]]] = {
            label: [] for label in INDEXED_RETRIEVER_LABELS
        }
        for stage_invocation in stage_invocations:
            (
                stage_language_bucket,
                stage_file_filter,
                stage_primary_query,
                stage_guardrail_query,
            ) = self._stage_invocation_values(stage_invocation)
            spec_groups: list[list[IndexedSpec]] = (
                [[spec] for spec in specs] if dispatch_per_spec else [specs]
            )
            weight = 0.9 if getattr(stage_invocation, "is_alternate", False) else 1.0
            for stage_specs in spec_groups:
                if not stage_specs:
                    continue
                (
                    stage_docs,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                ) = self._dispatch_indexed_stage_specs(
                    mySession,
                    stage_specs=stage_specs,
                    language_bucket=stage_language_bucket,
                    file_filter=stage_file_filter,
                    primary_query=stage_primary_query,
                    guardrail_query=stage_guardrail_query,
                    use_guardrail=use_guardrail,
                )
                for label, docs in stage_docs.items():
                    ranked_results[label].append((docs, weight))
                merged_docs = self.orchestratorHelper.merge_indexed_stage_docs(
                    merged_docs, stage_docs
                )
                (
                    total_top_k_orig_query_en,
                    total_top_k_post_rewrite_query_en,
                ) = self._rag_chat_impl._accumulate_topk_counts(
                    total_top_k_orig_query_en,
                    total_top_k_post_rewrite_query_en,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                )

        if any(
            getattr(invocation, "is_alternate", False)
            for invocation in stage_invocations
        ):
            merged_docs = self._merge_alternate_indexed_variants(
                mySession, ranked_results
            )
        return (
            merged_docs,
            total_top_k_orig_query_en,
            total_top_k_post_rewrite_query_en,
        )

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
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run one indexed query against one spec group and trace the dispatch."""
        dispatch_scope = "/".join(str(spec[0]) for spec in stage_specs)
        if not self._suppress_indexed_retriever_dispatch_trace:
            self.orchestratorMessages.trace_query_dispatch(
                mySession,
                retriever_scope=dispatch_scope,
                language_bucket=language_bucket,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
            )
        stage_docs, top_k_orig_query_en, top_k_post_rewrite_query_en = (
            self._run_indexed_specs_stage(
                mySession,
                indexed_specs=stage_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail and bool(guardrail_query),
                file_filter=file_filter,
            )
        )
        self.orchestratorMessages.debug_query_step(
            mySession,
            f"Step 6: retriever={dispatch_scope}; language={language_bucket}; "
            f"query={primary_query!r}; "
            f"hits={ {label: len(docs) for label, docs in stage_docs.items()} }",
        )
        return stage_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _merge_alternate_indexed_variants(
        self,
        mySession: Session,
        ranked_results: dict[str, list[tuple[list[Any], float]]],
    ) -> dict[str, list[Any]]:
        """RRF-merge original (weight 1.0) and alternate (0.9) query hits per retriever."""
        rrf_k = float(
            getattr(getattr(self._rag_chat_impl, "bm25_retriever", None), "rrf_k", 60)
        )
        merged_docs = {
            label: self.orchestratorHelper.merge_ranked_variants(
                ranked_lists, rrf_k=rrf_k
            )
            for label, ranked_lists in ranked_results.items()
        }
        self.orchestratorMessages.debug_query_step(
            mySession,
            "Step 7: merged indexed variants by best weighted reciprocal rank "
            f"(original=1.0, alternate=0.9); unique hits="
            f"{ {label: len(docs) for label, docs in merged_docs.items()} }",
        )
        return merged_docs

    def _bm25_stage_invocations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[Bm25StageInvocation]:
        """Use the same language-filtered query matrix for every indexed retriever."""
        stage_iterations = self._shared_graph_regex_stage_iterations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )

        return [
            Bm25StageInvocation(
                file_filter=stage_iteration.file_filter,
                primary_query=stage_iteration.primary_query,
                guardrail_query=stage_iteration.guardrail_query,
                is_alternate=stage_iteration.is_alternate,
            )
            for stage_iteration in stage_iterations
        ]

    def _run_indexed_retrieval_stages(
        self,
        mySession: Session,
        *,
        indexed_specs: list[IndexedSpec],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> IndexedStageResult:
        """Run BM25 and Graph/Regex indexed stages and merge their outputs."""
        bm25_specs, graph_regex_specs = self._partition_indexed_specs(indexed_specs)

        indexed_docs: dict[str, list[Any]] = {
            label: [] for label in INDEXED_RETRIEVER_LABELS
        }
        top_k_orig_query_en = 0
        top_k_post_rewrite_query_en = 0
        if not bm25_specs and not graph_regex_specs:
            return IndexedStageResult(
                bm25_docs=[],
                graph_docs=[],
                regex_docs=[],
                top_k_orig_query_en=0,
                top_k_post_rewrite_query_en=0,
            )

        if (
            self._indexed_stage_shape_queries
            and self._prepared_indexed_stage_iterations is not None
        ):
            shared_stage_iterations = self._prepared_indexed_stage_iterations
        else:
            shared_stage_iterations = self._graph_regex_stage_iterations(
                mySession,
                file_filter=file_filter,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
            )
        suppress_dispatch_trace = self._announce_indexed_stage_dispatch(
            mySession,
            stage_iterations=shared_stage_iterations,
            bm25_specs=bm25_specs,
            graph_regex_specs=graph_regex_specs,
            use_guardrail=use_guardrail,
        )

        previous_stage_iterations = self._active_indexed_stage_iterations
        previous_suppress_dispatch_trace = (
            self._suppress_indexed_retriever_dispatch_trace
        )
        self._active_indexed_stage_iterations = shared_stage_iterations
        self._suppress_indexed_retriever_dispatch_trace = suppress_dispatch_trace
        try:
            stage_groups = self._indexed_stage_groups(
                mySession,
                bm25_specs=bm25_specs,
                graph_regex_specs=graph_regex_specs,
                file_filter=file_filter,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
            )
            for stage_invocations, specs, dispatch_per_spec in stage_groups:
                (
                    stage_docs,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                ) = self._run_indexed_stage_group(
                    mySession,
                    stage_invocations=stage_invocations,
                    specs=specs,
                    use_guardrail=use_guardrail,
                    dispatch_per_spec=dispatch_per_spec,
                )
                indexed_docs = self.orchestratorHelper.merge_indexed_stage_docs(
                    indexed_docs, stage_docs
                )
                (
                    top_k_orig_query_en,
                    top_k_post_rewrite_query_en,
                ) = self._rag_chat_impl._accumulate_topk_counts(
                    top_k_orig_query_en,
                    top_k_post_rewrite_query_en,
                    stage_top_k_orig_query_en,
                    stage_top_k_post_rewrite_query_en,
                )
        finally:
            self._active_indexed_stage_iterations = previous_stage_iterations
            self._suppress_indexed_retriever_dispatch_trace = (
                previous_suppress_dispatch_trace
            )

        return IndexedStageResult(
            bm25_docs=indexed_docs["BM25"],
            graph_docs=indexed_docs["Graph"],
            regex_docs=indexed_docs["Regex"],
            top_k_orig_query_en=top_k_orig_query_en,
            top_k_post_rewrite_query_en=top_k_post_rewrite_query_en,
        )

    def _announce_indexed_stage_dispatch(
        self,
        mySession: Session,
        *,
        stage_iterations: list[GraphRegexStageIteration],
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
        use_guardrail: bool,
    ) -> bool:
        """Flush buffered per-language query status and trace one combined dispatch.

        Returns ``True`` when the combined trace was written, so the per-retriever
        dispatch traces should be suppressed.
        """
        retriever_labels = self.orchestratorMessages.indexed_stage_retriever_labels(
            bm25_specs,
            graph_regex_specs,
        )
        combined_trace = bool(stage_iterations and retriever_labels)
        self.orchestratorMessages.flush_indexed_stage_query_status()
        if combined_trace:
            self.orchestratorMessages.trace_indexed_stage_dispatch(
                mySession,
                stage_iterations=list(stage_iterations),
                retriever_labels=retriever_labels,
                use_guardrail=use_guardrail,
            )
        return combined_trace

    def _indexed_stage_groups(
        self,
        mySession: Session,
        *,
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[tuple[list[Any], list[IndexedSpec], bool]]:
        """Pair each indexed spec group with its stage invocations and dispatch mode.

        BM25 runs all its specs together per invocation; Graph/Regex run one spec
        at a time over the shared per-language iterations.
        """
        bm25_invocations = self._bm25_stage_invocations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )
        graph_regex_iterations = self._shared_graph_regex_stage_iterations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )
        return [
            (cast(list[Any], bm25_invocations), bm25_specs, False),
            (cast(list[Any], graph_regex_iterations), graph_regex_specs, True),
        ]

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
    ) -> IndexedStageResult:
        """Run indexed retrieval while temporarily toggling query shaping."""
        previous_shape_queries = self._indexed_stage_shape_queries
        self._indexed_stage_shape_queries = bool(shape_queries)
        try:
            return self._run_indexed_retrieval_stages(
                mySession,
                indexed_specs=indexed_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
        finally:
            self._indexed_stage_shape_queries = previous_shape_queries
