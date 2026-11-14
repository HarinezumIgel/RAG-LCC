"""Extracted OriginalQueryLegExecutor behavior; the coordinating facade owns shared state."""

from __future__ import annotations

from typing import Any

from Globals.Session import Session
from Gui.Colors import BRIGHT_MAGENTA
from Orchestrator.OrchestrationStageContext import OrchestrationStageContext
from Orchestrator.OrchestratorTypes import (IndexedSpec, IndexedStageResult,
                                            LocalStageInputs, OriginalQueryLeg)


class OriginalQueryLegExecutor(OrchestrationStageContext):
    def _merge_native_vector_leg(
        self,
        mySession: Session,
        *,
        vector_docs: list[Any],
        local_inputs: LocalStageInputs,
    ) -> tuple[list[Any], dict[str, int]]:
        """Run the original-language vector leg and merge its hits into ``vector_docs``.

        Returns the merged docs and the ``vector_hits/_added/_overlap`` counters
        expected by ``_store_original_query_leg_stats``.
        """
        native_vector_docs = self._run_original_query_vector_leg(
            mySession,
            vector_enabled=local_inputs.vector_enabled,
            original_query_leg=local_inputs.original_query_leg,
        )
        merged_vector_docs, added, overlap = (
            self.orchestratorHelper.merge_docs_with_native_leg(
                base_docs=vector_docs,
                native_docs=native_vector_docs,
                source_marker="VectorRaw",
            )
        )
        if local_inputs.original_query_leg.enabled:
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language vector merge "
                    f"added={added} overlap={overlap} "
                    f"total_vector={len(merged_vector_docs)}"
                ),
                color=BRIGHT_MAGENTA,
            )
        return merged_vector_docs, {
            "vector_hits": len(native_vector_docs),
            "vector_added": added,
            "vector_overlap": overlap,
        }

    def _merge_native_indexed_leg(
        self,
        indexed_stage_result: IndexedStageResult,
        native_stage_result: IndexedStageResult | None,
    ) -> tuple[dict[str, list[Any]], dict[str, int]]:
        """Merge native-language BM25/Graph/Regex hits into the indexed results.

        Returns the merged docs keyed by ``bm25``/``graph``/``regex`` and the
        ``<name>_hits/_added/_overlap`` counters expected by
        ``_store_original_query_leg_stats``.
        """
        merged: dict[str, list[Any]] = {}
        stats: dict[str, int] = {}
        for name, marker in (
            ("bm25", "BM25Raw"),
            ("graph", "GraphRaw"),
            ("regex", "RegexRaw"),
        ):
            base_docs = list(getattr(indexed_stage_result, f"{name}_docs"))
            if native_stage_result is None:
                merged[name] = base_docs
                stats.update(
                    {f"{name}_hits": 0, f"{name}_added": 0, f"{name}_overlap": 0}
                )
                continue
            native_docs = getattr(native_stage_result, f"{name}_docs")
            merged[name], added, overlap = (
                self.orchestratorHelper.merge_docs_with_native_leg(
                    base_docs=base_docs,
                    native_docs=native_docs,
                    source_marker=marker,
                )
            )
            stats.update(
                {
                    f"{name}_hits": len(native_docs),
                    f"{name}_added": added,
                    f"{name}_overlap": overlap,
                }
            )

        if native_stage_result is not None:
            summary = " ".join(
                f"{name}(h={stats[f'{name}_hits']},a={stats[f'{name}_added']},"
                f"o={stats[f'{name}_overlap']})"
                for name in ("bm25", "graph", "regex")
            )
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="I",
                message=f"original-language indexed merge {summary}",
                color=BRIGHT_MAGENTA,
            )
        return merged, stats

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
    ) -> None:
        """Persist original-language leg diagnostics on the session object."""
        mySession.original_query_leg_enabled = bool(original_query_leg.enabled)  # type: ignore[attr-defined]
        mySession.original_query_leg_query = original_query_leg.query  # type: ignore[attr-defined]
        mySession.original_query_leg_language = original_query_leg.language_bucket  # type: ignore[attr-defined]
        mySession.original_query_leg_reason = original_query_leg.reason  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_hits = int(vector_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_added = int(vector_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_vector_overlap = int(vector_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_hits = int(bm25_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_added = int(bm25_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_bm25_overlap = int(bm25_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_hits = int(graph_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_added = int(graph_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_graph_overlap = int(graph_overlap)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_hits = int(regex_hits)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_added = int(regex_added)  # type: ignore[attr-defined]
        mySession.original_query_leg_regex_overlap = int(regex_overlap)  # type: ignore[attr-defined]

    def _run_original_query_vector_leg(
        self,
        mySession: Session,
        *,
        vector_enabled: bool,
        original_query_leg: OriginalQueryLeg,
    ) -> list[Any]:
        """Run native-language vector retrieval leg and return native docs."""
        if not original_query_leg.enabled:
            return []
        if not vector_enabled:
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language vector leg skipped "
                    "(vector retriever disabled by flow, mode, or weight)"
                ),
                color=BRIGHT_MAGENTA,
            )
            return []

        self.orchestratorMessages.trace_query_dispatch(
            mySession,
            retriever_scope="VectorRaw",
            language_bucket=original_query_leg.language_bucket,
            primary_query=original_query_leg.query,
            guardrail_query="",
            use_guardrail=False,
            include_language=True,
        )

        (
            native_docs,
            _native_top_k_orig,
            _native_top_k_post,
        ) = self._rag_chat_impl._run_vector_retriever_with_guardrail(
            mySession=mySession,
            primary_query=original_query_leg.query,
            guardrail_query="",
            use_guardrail=False,
            alternate_queries=[],
        )
        _ = (_native_top_k_orig, _native_top_k_post)

        self.orchestratorMessages.write_retrieval_orchestration(
            severity="I",
            message=(
                "original-language vector leg completed "
                f"lang={original_query_leg.language_bucket or 'unknown'} "
                f"hits={len(native_docs)}"
            ),
            color=BRIGHT_MAGENTA,
        )

        return native_docs

    def _run_original_query_indexed_leg(
        self,
        mySession: Session,
        *,
        original_query_leg: OriginalQueryLeg,
        indexed_specs: list[IndexedSpec],
        file_filter: dict[str, Any] | None,
    ) -> IndexedStageResult | None:
        """Run the native query unchanged, or report why its indexed leg is off."""
        if not original_query_leg.enabled:
            return None
        if not any(spec[1] and spec[2] != 0.0 for spec in indexed_specs):
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="I",
                message=(
                    "original-language indexed leg skipped "
                    "(no indexed retriever enabled by flow, mode, or weight)"
                ),
                color=BRIGHT_MAGENTA,
            )
            return None

        native_file_filter = self._build_graph_regex_stage_filter(
            file_filter,
            original_query_leg.language_bucket,
        )
        return self._run_indexed_retrieval_with_shape_mode(
            mySession,
            indexed_specs=indexed_specs,
            primary_query=original_query_leg.query,
            guardrail_query="",
            use_guardrail=False,
            file_filter=native_file_filter,
            shape_queries=False,
        )
