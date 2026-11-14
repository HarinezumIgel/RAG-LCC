from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from Globals.Session import Session
from Gui.Colors import BRIGHT_BLACK, BRIGHT_GREEN, BRIGHT_MAGENTA, CYAN, ORANGE
from Helpers.DebugHelper import DebugHelper
from Orchestrator import FlowConfig
from Orchestrator.OrchestratorOverrides import OrchestratorOverrides
from Orchestrator.OrchestratorTypes import (GraphRegexStageIteration,
                                            IndexedSpec,
                                            OrchestrationFlowProfile,
                                            OriginalQueryLeg, RetrievalPlan,
                                            StageExecutionGates)


class OrchestratorMessages:
    """Own orchestration traces, diagnostic notices, and message buffering."""

    _TRACE_RETRIEVER_WIDTH = 10
    _TRACE_LANGUAGE_WIDTH = 7
    _FLOW_STAGE_WIDTH = 34
    _FLOW_STATUS_WIDTH = 38
    _FLOW_KNOB_WIDTH = 28
    _FLOW_VALUE_WIDTH = 18

    def __init__(
        self,
        rag_chat_impl: Any,
        *,
        overrides: OrchestratorOverrides,
        normalize_language_bucket: Callable[[Any], str | None],
    ) -> None:
        self._rag_chat_impl = rag_chat_impl
        self.orchestratorOverrides = overrides
        self._normalize_language_bucket = normalize_language_bucket
        self._last_language_status_signature: (
            tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
        ) = None
        self._defer_indexed_stage_status = False
        self._pending_indexed_stage_status: list[str] = []

    def write(
        self,
        severity: str,
        stage: str,
        message: str,
        *,
        color: str | None = None,
    ) -> None:
        if color is None:
            self._rag_chat_impl.pretty.write(severity, stage, message)
        else:
            self._rag_chat_impl.pretty.write(severity, stage, message, color=color)

    def log_step(
        self,
        mySession: Session,
        *,
        step_name: str,
        status: str,
        confidence_level: str,
        confidence_score: float | None = None,
        detail: str = "",
        source: str = "orchestrator",
    ) -> None:
        """Forward execution confidence events when a logger is available."""
        logger = getattr(self._rag_chat_impl, "confidence_logger", None)
        if logger is not None and hasattr(logger, "log_step"):
            logger.log_step(
                mySession,
                step_name=step_name,
                status=status,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                detail=detail,
                source=source,
            )

    @contextmanager
    def deferred_indexed_status(self) -> Iterator[None]:
        """Keep preflight messages buffered until dispatch, restoring outer state."""
        previous_defer = self._defer_indexed_stage_status
        previous_pending = self._pending_indexed_stage_status
        self._defer_indexed_stage_status = True
        self._pending_indexed_stage_status = []
        try:
            yield
        finally:
            self._defer_indexed_stage_status = previous_defer
            self._pending_indexed_stage_status = previous_pending

    def trace_language_fallback(self, mySession: Session) -> str | None:
        """Return best available session language label, or None if unavailable."""
        for candidate in (
            getattr(mySession, "retrieval_language", None),
            getattr(mySession, "current_query_lang", None),
            getattr(mySession, "user_language", None),
        ):
            normalized = self._normalize_language_bucket(candidate)
            if normalized:
                return normalized
        return None

    def _write_always_on(
        self,
        *,
        severity: str,
        stage: str,
        message: str,
        color: str | None,
    ) -> None:
        pretty = getattr(self._rag_chat_impl, "pretty", None)
        if pretty is None or not hasattr(pretty, "write"):
            return

        had_always_on = hasattr(pretty, "always_on")
        previous_always_on = getattr(pretty, "always_on", None)
        if had_always_on:
            try:
                setattr(pretty, "always_on", True)
            except Exception:
                had_always_on = False

        try:
            pretty.write(
                severity,
                stage,
                message,
                color=color,
            )
        finally:
            if had_always_on:
                try:
                    setattr(pretty, "always_on", previous_always_on)
                except Exception:
                    pass

    def write_retrieval_orchestration(
        self,
        *,
        severity: str,
        message: str,
        color: str | None = BRIGHT_MAGENTA,
    ) -> None:
        """Write retrieval orchestration lines with PrettyWriter always-on semantics."""
        self._write_always_on(
            severity=severity,
            stage="Retrieval Orchestration",
            message=message,
            color=color,
        )

    def write_multiquery_status(
        self,
        *,
        severity: str,
        message: str,
        color: str | None = CYAN,
    ) -> None:
        """Write MultiQuery lines with PrettyWriter always-on semantics."""
        self._write_always_on(
            severity=severity,
            stage="MultiQuery",
            message=message,
            color=color,
        )

    def collect_session_override_tokens(self, mySession: Session) -> list[str]:
        """Return compact override tokens to show in retrieval orchestration logs."""
        tokens: list[str] = []

        max_out_override = getattr(mySession, "max_output_tokens_override", None)
        if max_out_override is not None:
            tokens.append(f"max_output_tokens={max_out_override}")

        ctx_override = getattr(mySession, "context_size_override", None)
        if ctx_override is not None:
            tokens.append(f"context_size={ctx_override}")

        return tokens

    def emit_session_overrides(self, mySession: Session) -> None:
        """Emit one orchestration line with user/session overrides when present."""
        tokens = self.collect_session_override_tokens(mySession)
        if not tokens:
            return
        self.write_retrieval_orchestration(
            severity="I",
            message=f"session overrides: {'  '.join(tokens)}",
            color=ORANGE,
        )

    def trace(
        self,
        mySession: Session,
        message: str,
        *,
        retriever_scope: str = "orchestrate",
        language_bucket: str | None = None,
        query_text: str | None = None,
        include_language: bool = True,
    ) -> None:
        scope_text = str(retriever_scope or "orchestrate").strip()
        scope_field = f"{scope_text[: self._TRACE_RETRIEVER_WIDTH]:<{self._TRACE_RETRIEVER_WIDTH}}"
        scope_prefix = (
            ""
            if scope_text.lower() in {"orchestrate", "orchestration"}
            else f"retriever: {scope_field}"
        )
        prefix = scope_prefix
        if include_language:
            language_text = self._normalize_language_bucket(language_bucket)
            if not language_text:
                language_text = self.trace_language_fallback(mySession)
            if language_text:
                language_field = f"{language_text:<{self._TRACE_LANGUAGE_WIDTH}}"
                prefix = (
                    f"{scope_prefix} language: {language_field}"
                    if scope_prefix
                    else f"language: {language_field}"
                )
        if query_text:
            if prefix:
                prefix = f"{prefix} query: {query_text!r}"
            else:
                prefix = f"query: {query_text!r}"
        formatted_message = f"{prefix} {message}".strip()
        self.write_retrieval_orchestration(
            severity="I",
            message=formatted_message,
            color=BRIGHT_MAGENTA,
        )

    def trace_stage(
        self,
        mySession: Session,
        *,
        stage_name: str,
        details: str = "",
        query_text: str | None = None,
    ) -> None:
        """Emit a normalized stage trace message for retrieval orchestration."""
        message = stage_name.strip()
        details_text = details.strip()
        if details_text:
            message = f"{message} {details_text}"
        self.trace(
            mySession,
            message,
            retriever_scope="orchestrate",
            language_bucket=None,
            query_text=query_text,
        )

    def trace_stage_plan(
        self,
        mySession: Session,
        *,
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str,
        use_guardrail: bool,
        original_leg_enabled: bool,
        alternate_query_count: int,
    ) -> None:
        """Trace the resolved flow knobs and retriever switches for this turn."""

        def on_off(flag: bool) -> str:
            return "on" if flag else "off"

        fp = flow_profile
        self.trace(
            mySession,
            "stage plan "
            f"flow={fp.name} "
            f"mode={retrieve_mode} "
            f"guardrail={on_off(use_guardrail)} "
            f"original_leg={on_off(original_leg_enabled)} "
            f"rewrite={on_off(fp.enable_query_rewrite)} "
            f"pronouns={on_off(fp.enable_pronoun_substitution)} "
            f"rerank={on_off(fp.enable_rerank)} "
            f"low_score_fallback={on_off(fp.enable_low_score_fallback)} "
            f"low_recall_rescue={on_off(fp.enable_low_recall_rescue)} "
            f"grounding={on_off(fp.enable_grounding)} "
            f"indexed_shape={on_off(fp.enable_indexed_query_shaping)} "
            f"indexed_translate={on_off(fp.enable_indexed_query_translation)} "
            f"indexed_wordnet={on_off(fp.enable_indexed_query_wordnet_expansion)} "
            f"alt_queries={alternate_query_count} "
            f"retrievers="
            f"v={on_off(fp.enable_vector_retriever)},"
            f"b={on_off(fp.enable_bm25_retriever)},"
            f"g={on_off(fp.enable_graph_retriever)},"
            f"r={on_off(fp.enable_regex_retriever)}",
        )

    def emit_flow_knob_stage_status(
        self,
        *,
        mySession: Session | None,
        stage_name: str,
        knob_name: str,
        enabled: bool,
        configured_enabled: bool | None = None,
        action_override: str | None = None,
        cli_knob: str | None = None,
        effective_knob_value: bool | str | None = None,
    ) -> None:
        """Emit one orchestration line explaining stage activation by flow knobs."""
        action = action_override or ("Activated" if enabled else "Not activated")
        config_state = enabled if configured_enabled is None else configured_enabled
        knob_state = "on" if config_state else "off"
        if isinstance(effective_knob_value, bool):
            effective_text = "on" if effective_knob_value else "off"
        elif effective_knob_value is None:
            effective_text = "on" if enabled else "off"
        else:
            effective_text = str(effective_knob_value).strip().replace("_", " ")
        # This is a pre-execution planning line, not evidence confidence.
        step_confidence = "PENDING" if enabled else "N/A"

        message = (
            f"{stage_name:<{self._FLOW_STAGE_WIDTH}} "
            f"{action:<{self._FLOW_STATUS_WIDTH}} "
            f"knob={knob_name:<{self._FLOW_KNOB_WIDTH}} "
            f"value={effective_text:<{self._FLOW_VALUE_WIDTH}}"
        )
        if cli_knob:
            message = f"{message} cli_knob={cli_knob}"

        self.write_retrieval_orchestration(
            severity="I",
            message=message,
            color=CYAN,
        )

        if mySession is not None:
            confidence_obj = getattr(mySession, "orchestration_step_confidence", None)
            confidence_map: dict[str, str]
            if isinstance(confidence_obj, dict):
                confidence_map = cast(dict[str, str], confidence_obj)
            else:
                confidence_map = {}
            confidence_map[stage_name] = step_confidence
            mySession.orchestration_step_confidence = confidence_map  # type: ignore[attr-defined]

            self.log_step(
                mySession,
                step_name=stage_name,
                status="planned" if enabled else "skipped",
                confidence_level=step_confidence,
                detail=(
                    f"knob={knob_name}:{knob_state}; "
                    f"effective={effective_text}; "
                    f"action={action}" + (f"; cli_knob={cli_knob}" if cli_knob else "")
                ),
                source="orchestrator_flow",
            )

    def resolve_mode_stage_flags(
        self,
        retrieve_mode: str | None,
    ) -> tuple[str, bool, bool, bool, bool, bool]:
        """Return mode text plus local/retriever enable flags for planning logs."""
        mode_text = str(retrieve_mode or "").strip().upper()
        # Retriever selection is applied to the flow profile (run_* switches,
        # narrowed by an explicit force_retrieve_mode); retrieve_mode here only
        # keeps WEB-only compatibility for local/web stage gating.
        return (
            mode_text,
            True,
            True,
            True,
            True,
            (
                not self.orchestratorOverrides.is_web_only_mode(mode_text)
                if mode_text
                else True
            ),
        )

    def emit_primary_flow_stage_statuses(
        self,
        *,
        mySession: Session,
        flow_profile: OrchestrationFlowProfile,
    ) -> None:
        """Emit flow-native stage statuses and grounding CLI contradiction."""
        statuses: list[tuple[str, str, bool]] = [
            (
                "Query rewrite stage",
                "use_query_rewrite",
                flow_profile.enable_query_rewrite,
            ),
            (
                "Pronoun substitution stage",
                "use_pronoun_substitution",
                flow_profile.enable_pronoun_substitution,
            ),
            (
                "Secondary query stage",
                "use_secondary_query",
                flow_profile.enable_guardrail_leg,
            ),
            (
                "Original-language vector stage",
                "use_original_language_vector",
                flow_profile.enable_original_language_vector_leg,
            ),
            (
                "Indexed query shaping stage",
                "shape_indexed_queries",
                flow_profile.enable_indexed_query_shaping,
            ),
            (
                "Indexed query translation stage",
                "translate_indexed_queries",
                flow_profile.enable_indexed_query_translation,
            ),
            (
                "Indexed WordNet expansion stage",
                "expand_indexed_queries_with_wordnet",
                flow_profile.enable_indexed_query_wordnet_expansion,
            ),
            (
                "Vector alternates stage",
                "use_vector_alternates",
                flow_profile.enable_vector_alternate_queries,
            ),
        ]

        for stage_name, knob_name, enabled in statuses:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name=stage_name,
                knob_name=knob_name,
                enabled=enabled,
                configured_enabled=enabled,
            )

        (
            grounding_conflict,
            grounding_action,
            grounding_cli_knob,
        ) = self.orchestratorOverrides.resolve_grounding_cli_contradiction(
            mySession,
            flow_enabled=flow_profile.enable_grounding,
        )
        if grounding_conflict:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Grounding stage",
                knob_name="run_grounding",
                enabled=False,
                configured_enabled=True,
                action_override=grounding_action,
                cli_knob=grounding_cli_knob,
            )
        else:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Grounding stage",
                knob_name="run_grounding",
                enabled=flow_profile.enable_grounding,
                configured_enabled=flow_profile.enable_grounding,
            )

        (
            rerank_conflict,
            rerank_effective_enabled,
            rerank_action,
            rerank_cli_knob,
        ) = self.orchestratorOverrides.resolve_rerank_cli_contradiction(
            mySession,
            flow_enabled=flow_profile.enable_rerank,
        )
        if rerank_conflict:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Rerank stage",
                knob_name="run_rerank",
                enabled=rerank_effective_enabled,
                configured_enabled=flow_profile.enable_rerank,
                action_override=rerank_action,
                cli_knob=rerank_cli_knob,
            )
        else:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Rerank stage",
                knob_name="run_rerank",
                enabled=rerank_effective_enabled,
                configured_enabled=flow_profile.enable_rerank,
            )

        for stage_name, knob_name, configured in (
            (
                "Low-score fallback action",
                "run_low_score_fallback",
                flow_profile.enable_low_score_fallback,
            ),
            (
                "Low-recall rescue action",
                "run_low_recall_rescue",
                flow_profile.enable_low_recall_rescue,
            ),
        ):
            if configured and not rerank_effective_enabled:
                action = (
                    "Deactivated in session CLI"
                    if rerank_cli_knob
                    else "Deactivated because rerank stage is off"
                )
                self.emit_flow_knob_stage_status(
                    mySession=mySession,
                    stage_name=stage_name,
                    knob_name=knob_name,
                    enabled=False,
                    configured_enabled=True,
                    action_override=action,
                    cli_knob=rerank_cli_knob,
                )
            else:
                self.emit_flow_knob_stage_status(
                    mySession=mySession,
                    stage_name=stage_name,
                    knob_name=knob_name,
                    enabled=bool(configured and rerank_effective_enabled),
                    configured_enabled=configured,
                )

    def emit_local_web_stage_statuses(
        self,
        *,
        mySession: Session,
        mode_text: str,
        local_mode_enabled: bool,
    ) -> None:
        """Emit local/web stage statuses driven by web mode resolution."""
        web_mode = self.orchestratorOverrides.resolve_web_stage_mode(
            mySession, mode_text
        )
        web_stage_enabled = web_mode in ("local_and_web", "web_only")
        web_knob_name = (
            "retrieve_mode"
            if mode_text == "WEB" and not bool(getattr(mySession, "web_search", False))
            else "web_search"
        )
        web_mode_display = web_mode.replace("_", " ")
        knob_effective_value = (
            mode_text if web_knob_name == "retrieve_mode" else web_mode_display
        )

        if (not local_mode_enabled) and mode_text:
            mode_knob = (
                "web_search=web_only"
                if bool(getattr(mySession, "web_search", False)) and mode_text == "WEB"
                else f"retrieve_mode={mode_text}"
            )
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Local stage",
                knob_name=web_knob_name,
                enabled=False,
                configured_enabled=True,
                action_override="Deactivated in session CLI",
                cli_knob=mode_knob,
                effective_knob_value=knob_effective_value,
            )
        else:
            self.emit_flow_knob_stage_status(
                mySession=mySession,
                stage_name="Local stage",
                knob_name=web_knob_name,
                enabled=bool(local_mode_enabled),
                configured_enabled=True,
                effective_knob_value=knob_effective_value,
            )

        self.emit_flow_knob_stage_status(
            mySession=mySession,
            stage_name="Web stage",
            knob_name=web_knob_name,
            enabled=web_stage_enabled,
            configured_enabled=web_stage_enabled,
            effective_knob_value=knob_effective_value,
        )

    def emit_retriever_stage_status(
        self,
        *,
        mySession: Session,
        stage_name: str,
        knob_name: str,
        flow_enabled: bool,
        cli_knob: str | None = None,
    ) -> None:
        """Emit one retriever stage status line driven by flow controls."""
        self.emit_flow_knob_stage_status(
            mySession=mySession,
            stage_name=stage_name,
            knob_name=knob_name,
            enabled=bool(flow_enabled),
            configured_enabled=flow_enabled,
            cli_knob=cli_knob,
        )

    def emit_retriever_stage_statuses(
        self,
        *,
        mySession: Session,
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str | None = None,
    ) -> None:
        """Emit vector/BM25/graph/regex stage statuses."""
        forced_mode = (
            FlowConfig.normalize_name(retrieve_mode)
            if flow_profile.retrieve_mode_override
            and FlowConfig.narrows_local_retrievers(retrieve_mode)
            else ""
        )
        mode_flags = FlowConfig.mode_flags_for_local_retrievers(forced_mode or "ALL")
        for (stage_name, knob_name, enabled), mode_enabled in zip(
            (
                (
                    "Vector retriever stage",
                    "run_vector",
                    flow_profile.enable_vector_retriever,
                ),
                (
                    "BM25 retriever stage",
                    "run_bm25",
                    flow_profile.enable_bm25_retriever,
                ),
                (
                    "Graph retriever stage",
                    "run_graph",
                    flow_profile.enable_graph_retriever,
                ),
                (
                    "Regex retriever stage",
                    "run_regex",
                    flow_profile.enable_regex_retriever,
                ),
            ),
            mode_flags,
        ):
            self.emit_retriever_stage_status(
                mySession=mySession,
                stage_name=stage_name,
                knob_name=knob_name,
                flow_enabled=enabled,
                cli_knob=(
                    f"force_retrieve_mode={forced_mode}"
                    if forced_mode and not mode_enabled
                    else None
                ),
            )

    def emit_flow_knob_stage_statuses(
        self,
        *,
        mySession: Session,
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str | None = None,
    ) -> None:
        """Emit flow stage lines and annotate CLI/session contradictions."""
        mode_flags = self.resolve_mode_stage_flags(retrieve_mode)
        mode_text = mode_flags[0]
        local_mode_enabled = mode_flags[5]

        self.emit_primary_flow_stage_statuses(
            mySession=mySession,
            flow_profile=flow_profile,
        )
        self.emit_flow_knob_stage_status(
            mySession=mySession,
            stage_name="Confidence evaluation",
            knob_name="confidence_evaluation",
            enabled=flow_profile.confidence_evaluation != "off",
            effective_knob_value=flow_profile.confidence_evaluation,
        )
        self.emit_local_web_stage_statuses(
            mySession=mySession,
            mode_text=mode_text,
            local_mode_enabled=local_mode_enabled,
        )
        self.emit_retriever_stage_statuses(
            mySession=mySession,
            flow_profile=flow_profile,
            retrieve_mode=mode_text,
        )

    def trace_query_dispatch(
        self,
        mySession: Session,
        *,
        retriever_scope: str,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        include_language: bool = True,
    ) -> None:
        """Emit one compact trace line for a query dispatch."""
        message = "dispatch query"
        if use_guardrail:
            message = f"{message} guardrail_query={guardrail_query!r}"
        self.trace(
            mySession,
            message,
            retriever_scope=retriever_scope,
            language_bucket=language_bucket,
            query_text=primary_query,
            include_language=include_language,
        )

    @staticmethod
    def indexed_stage_retriever_labels(
        bm25_specs: list[IndexedSpec],
        graph_regex_specs: list[IndexedSpec],
    ) -> list[str]:
        """Return ordered retriever labels participating in indexed stage."""
        labels: list[str] = []
        if bm25_specs:
            labels.append("BM25")
        if any(spec[0] == "Graph" for spec in graph_regex_specs):
            labels.append("Graph")
        if any(spec[0] == "Regex" for spec in graph_regex_specs):
            labels.append("Regex")
        return labels

    def trace_indexed_stage_dispatch(
        self,
        mySession: Session,
        *,
        stage_iterations: list[GraphRegexStageIteration],
        retriever_labels: list[str],
        use_guardrail: bool,
    ) -> None:
        """Emit one indexed dispatch trace per language-stage iteration."""
        if not stage_iterations or not retriever_labels:
            return

        retriever_group = "/".join(retriever_labels)
        for stage_iteration in stage_iterations:
            language_text = self._normalize_language_bucket(
                stage_iteration.language_bucket
            )
            if not language_text:
                language_text = self.trace_language_fallback(mySession)
            if language_text:
                language_field = f"{language_text:<{self._TRACE_LANGUAGE_WIDTH}}"
                message = (
                    f"retrievers={retriever_group!r} "
                    f"language: {language_field} "
                    f"query: {stage_iteration.primary_query!r}"
                )
            else:
                message = (
                    f"retrievers={retriever_group!r} "
                    f"query: {stage_iteration.primary_query!r}"
                )
            if use_guardrail:
                message = (
                    f"{message} " f"guardrail_query={stage_iteration.guardrail_query!r}"
                )
            self.write_retrieval_orchestration(
                severity="I",
                message=message,
                color=BRIGHT_MAGENTA,
            )

    def write_indexed_stage_query_status(
        self,
        *,
        flow_name: str,
        target_language: str | None,
        primary_query_source: str,
        primary_source_language: str,
        was_translated: bool,
        primary_query: str,
        stage_primary_query: str,
        guardrail_query_source: str,
        guardrail_source_language: str,
        guardrail_query: str,
        stage_guardrail_query: str,
        guardrail_disabled_reason: str | None,
        indexed_shaping_enabled: bool,
        stage_alternatives: list[tuple[str, str]] | None = None,
    ) -> None:
        """Emit one indexed-stage query-shaping status block per language."""
        alternative_lines = "".join(
            f"\nalternative_query={original!r}" f"\nstage_alternative_query={shaped!r}"
            for original, shaped in (stage_alternatives or [])
        )
        language_route = f"{primary_source_language}->{target_language or 'N/A'}"
        primary_lines = (
            f"\nprimary_query source={primary_query_source} lang={language_route}"
            f"\n  original={primary_query!r}"
            f"\n  staged={stage_primary_query!r}"
        )
        if guardrail_disabled_reason == "disabled_by_flow":
            guardrail_lines = (
                f"\nguardrail_query source={guardrail_query_source} "
                f"lang={guardrail_source_language}->{target_language or 'N/A'}"
                f"\n  value=N/A (disabled by flow {flow_name}: "
                "guardrail leg is off)"
            )
        elif guardrail_disabled_reason:
            guardrail_lines = (
                f"\nguardrail_query source={guardrail_query_source} "
                f"lang={guardrail_source_language}->{target_language or 'N/A'}"
                f"\n  value=N/A ({guardrail_disabled_reason})"
            )
        elif not guardrail_query:
            guardrail_lines = (
                f"\nguardrail_query source={guardrail_query_source} "
                f"lang={guardrail_source_language}->{target_language or 'N/A'}"
                "\n  value=N/A (no query value)"
            )
        else:
            guardrail_lines = (
                f"\nguardrail_query source={guardrail_query_source} "
                f"lang={guardrail_source_language}->{target_language or 'N/A'}"
                f"\n  original={guardrail_query!r}"
                f"\n  staged={stage_guardrail_query!r}"
            )
        alternatives_header = (
            f"\nalternatives source={primary_query_source} lang={language_route} "
            f"count={len(stage_alternatives or [])}"
        )
        message = (
            f"idx_stage flow={flow_name} language={target_language or 'N/A'} "
            "retrievers=BM25/Graph/Regex "
            f"shaping={'on' if indexed_shaping_enabled else 'off'} "
            f"changed={int(was_translated)}"
            f"{primary_lines}{guardrail_lines}"
            f"{alternatives_header}{alternative_lines}"
        )
        if self._defer_indexed_stage_status:
            self._pending_indexed_stage_status.append(message)
            return
        self.write_retrieval_orchestration(
            severity="I",
            message=message,
            color=BRIGHT_BLACK,
        )

    def flush_indexed_stage_query_status(self) -> None:
        """Emit any preflight query reports immediately before indexed retrieval."""
        messages = self._pending_indexed_stage_status
        self._pending_indexed_stage_status = []
        for message in messages:
            self.write_retrieval_orchestration(
                severity="I",
                message=message,
                color=BRIGHT_BLACK,
            )

    def indexed_query_source_language(
        self,
        mySession: Session,
        source_name: str,
    ) -> str:
        source = str(source_name or "").strip().upper()
        if source == "TRANSLATED_QUERY":
            return "en"
        if source == "ORIGINAL_QUERY":
            return (
                self._normalize_language_bucket(
                    getattr(mySession, "user_language", None)
                )
                or "N/A"
            )
        retrieval_mode = (
            str(getattr(mySession, "retrieval_language_mode", "ENGLISH") or "ENGLISH")
            .strip()
            .upper()
        )
        if retrieval_mode == "NATIVE":
            return (
                self._normalize_language_bucket(
                    getattr(mySession, "retrieval_language", None)
                )
                or self._normalize_language_bucket(
                    getattr(mySession, "user_language", None)
                )
                or "N/A"
            )
        return "en"

    @staticmethod
    def clip_message_text(text: Any, limit: int = 140) -> str:
        """Return a compact, single-line preview for status messages."""
        clipped = str(text or "").strip().replace("\n", " ")
        if len(clipped) <= limit:
            return clipped
        return clipped[: limit - 3] + "..."

    def emit_original_query_leg_status(
        self,
        *,
        original_query_leg: OriginalQueryLeg,
    ) -> None:
        """Emit one status line describing original-language leg planning."""
        status = "on" if original_query_leg.enabled else "off"
        lang = original_query_leg.language_bucket or "unknown"
        query_preview = self.clip_message_text(original_query_leg.query)
        self.write_retrieval_orchestration(
            severity="I",
            message=(
                "original-language vector leg: "
                f"{status} lang={lang} "
                f"query={query_preview!r} "
                f"reason={original_query_leg.reason}"
            ),
            color=BRIGHT_MAGENTA,
        )

    def emit_vector_query_language_flow(self, mySession: Session) -> None:
        """Emit language-flow status only for Vector execution."""
        user_language = (
            self._normalize_language_bucket(getattr(mySession, "user_language", None))
            or "unknown"
        )
        current_query_language = (
            self._normalize_language_bucket(
                getattr(mySession, "current_query_lang", None)
            )
            or "unknown"
        )
        rewrite_language = (
            self._normalize_language_bucket(
                getattr(mySession, "rewrite_language", None)
            )
            or "unknown"
        )
        retrieval_language = (
            self._normalize_language_bucket(
                getattr(mySession, "retrieval_language", None)
            )
            or "unknown"
        )
        self.write_retrieval_orchestration(
            severity="I",
            message=(
                "query language flow: "
                f"user={user_language} "
                f"current={current_query_language} "
                f"rewrite={rewrite_language} "
                f"retrieval={retrieval_language}"
            ),
            color=BRIGHT_MAGENTA,
        )

    def emit_vector_alternate_queries_status(
        self,
        *,
        alternate_queries: list[str],
    ) -> None:
        """Emit vector alternate-query fanout details as a multiline block."""
        if not alternate_queries:
            return

        query_lines = [
            f"{index}: {query!r}"
            for index, query in enumerate(alternate_queries, start=1)
        ]
        message = f"{len(alternate_queries)} Alternate queries for vector retrieval:"
        if query_lines:
            message = message + "\n" + "\n".join(query_lines)

        self.write_multiquery_status(
            severity="I",
            message=message,
            color=BRIGHT_BLACK,
        )

    def emit_query_language_rewrite_status(
        self,
        mySession: Session,
        *,
        plan: RetrievalPlan,
    ) -> None:
        """Emit per-turn query-rewrite status block."""

        seed_query_en = self.clip_message_text(plan.orig_translated_query_en)
        rewritten_query = self.clip_message_text(
            getattr(mySession, "rewritten_query", None)
        )
        final_query_en = self.clip_message_text(
            getattr(mySession, "post_rewrite_query_en", None) or plan.final_query
        )

        self.write_retrieval_orchestration(
            severity="I",
            message=(
                "query rewrite flow:"
                f"\nseed_en={seed_query_en!r}"
                f"\nrewritten={rewritten_query!r}"
                f"\nfinal_en={final_query_en!r}"
            ),
            color=BRIGHT_MAGENTA,
        )

    def trace_effective_stage_gates(
        self,
        mySession: Session,
        *,
        flow_name: str,
        retrieve_mode: str,
        gates: StageExecutionGates,
    ) -> None:
        """Emit one trace line showing effective retrieval-stage gates."""
        self.write_retrieval_orchestration(
            severity="I",
            message=(
                "effective stage gates: "
                f"flow={flow_name} "
                f"mode={retrieve_mode} "
                f"web_mode={gates.web_mode} "
                f"local_stage={'on' if gates.local_stage_enabled else 'off'} "
                f"web_stage={'on' if gates.web_stage_enabled else 'off'}"
            ),
            color=BRIGHT_GREEN,
        )

    def debug_query_step(self, mySession: Session, message: str) -> None:
        if DebugHelper.check_session(mySession, 30):
            self.write("D", "QueryPlan", message, color=CYAN)

    @staticmethod
    def record_indexed_query_notice(mySession: Session, message: str) -> None:
        notices = list(getattr(mySession, "indexed_query_notices", []))
        if message not in notices:
            notices.append(message)
        mySession.indexed_query_notices = notices

    def warn_indexed_translation_once(
        self,
        mySession: Session,
        *,
        language: str | None,
        category: str,
        message: str,
        query: str,
        severity: str = "W",
    ) -> None:
        keys = set(getattr(mySession, "indexed_query_warning_keys", set()))
        key = (category, language)
        if key not in keys:
            keys.add(key)
            self.write_retrieval_orchestration(severity=severity, message=message)
        mySession.indexed_query_warning_keys = keys
        self.debug_query_step(
            mySession,
            f"Step 5: translation failure; language={language}; category={category}; "
            f"query={query!r}; {message}",
        )

    def emit_vector_store_language_status(
        self,
        mySession: Session,
        *,
        vector_store_languages: list[str],
        active_codes: list[str],
    ) -> None:
        """Report whether vector-store languages are fully covered by config."""
        if not vector_store_languages:
            return

        active_code_set = set(active_codes)
        undefined_languages = sorted(
            lang for lang in vector_store_languages if lang not in active_code_set
        )

        collection_name = str(
            getattr(self._rag_chat_impl, "collection_name", None)
            or getattr(mySession, "collection_name", None)
            or ""
        )
        signature = (
            collection_name,
            tuple(vector_store_languages),
            tuple(active_codes),
            tuple(undefined_languages),
        )
        if signature == self._last_language_status_signature:
            return
        self._last_language_status_signature = signature

        languages_label = ", ".join(vector_store_languages)
        active_label = ", ".join(active_codes)
        self.write_retrieval_orchestration(
            severity="I",
            message=(
                "discovered corpus language buckets: "
                f"[{languages_label}] active config buckets: [{active_label}]"
            ),
            color=BRIGHT_MAGENTA,
        )

        if undefined_languages:
            pretty = getattr(self._rag_chat_impl, "pretty", None)
            if pretty is None or not hasattr(pretty, "write"):
                return
            undefined_label = ", ".join(undefined_languages)
            pretty.write(
                "W",
                "LanguageConfig",
                "Vector-store document languages "
                f"[{languages_label}] include codes not active in "
                "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES: "
                f"{undefined_label}",
                color=ORANGE,
            )
            return

        return
