from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from Globals.Session import Session
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Orchestrator import FlowConfig
from Orchestrator.OrchestratorTypes import (OrchestrationFlowProfile,
                                            RetrievalPlan)


class OrchestratorOverrides:
    """Resolve flow baselines and apply session/CLI override precedence."""

    _DEFAULT_QUERY_SOURCE = FlowConfig.DEFAULT_MAIN_QUERY_SOURCE

    # Profile field -> flow config key (derived from the shared mapping).
    _FLOW_PROFILE_CONFIG_KEYS: dict[str, str] = {
        field_name: config_key
        for config_key, field_name in FlowConfig.FLOW_SWITCH_KEY_TO_FIELD.items()
    }

    _FLOW_SWITCH_FIELDS: tuple[str, ...] = tuple(
        field_name
        for field_name in OrchestrationFlowProfile.__dataclass_fields__
        if field_name.startswith("enable_")
    )

    def __init__(self, rag_chat_impl: Any) -> None:
        self._rag_chat_impl = rag_chat_impl

    def cfg_get_list(self, key: str, default: list[Any]) -> list[Any]:
        cfg = getattr(self._rag_chat_impl, "cfg", None)
        if cfg is None or not hasattr(cfg, "get_list"):
            return list(default)
        try:
            return cast(list[Any], cfg.get_list(key, default))
        except Exception:
            return list(default)

    def cfg_get_dict(self, key: str, default: dict[str, Any]) -> dict[str, Any]:
        cfg = getattr(self._rag_chat_impl, "cfg", None)
        if cfg is None or not hasattr(cfg, "get_dict"):
            return dict(default)
        try:
            return cast(dict[str, Any], cfg.get_dict(key, default))
        except Exception:
            return dict(default)

    def cfg_get_str(self, key: str, default: str) -> str:
        cfg = getattr(self._rag_chat_impl, "cfg", None)
        if cfg is None or not hasattr(cfg, "get_str"):
            return default
        try:
            value = cfg.get_str(key, default)
        except Exception:
            return default
        return str(value)

    @property
    def _cfg(self) -> Any:
        return getattr(self._rag_chat_impl, "cfg", None)

    def cfg_get_allowed_retrieve_modes(self) -> set[str]:
        return FlowConfig.allowed_retrieve_modes(self._cfg)

    def cfg_get_allowed_query_sources(self) -> list[str]:
        return FlowConfig.allowed_query_sources(self._cfg)

    def resolve_orchestration_flow_profile(
        self,
        mySession: Session,
    ) -> OrchestrationFlowProfile:
        """Resolve active flow profile from session + config defaults."""
        cfg = self._cfg
        flow_map = FlowConfig.load_flow_map(cfg)
        selected = FlowConfig.select_flow_name(
            cfg, getattr(mySession, "orchestrator_flow", "")
        )
        profile_raw = flow_map.get(selected, {})

        retrieve_mode_override = FlowConfig.normalize_name(
            profile_raw.get("force_retrieve_mode", "")
        )
        allowed_modes = self.cfg_get_allowed_retrieve_modes()
        if (
            retrieve_mode_override
            and allowed_modes
            and retrieve_mode_override not in allowed_modes
        ):
            retrieve_mode_override = ""

        allowed_sources = self.cfg_get_allowed_query_sources()
        primary_source = FlowConfig.normalize_query_source(
            profile_raw.get("main_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_MAIN_QUERY_SOURCE,
        )
        guardrail_source = FlowConfig.normalize_query_source(
            profile_raw.get("secondary_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_SECONDARY_QUERY_SOURCE,
        )
        rerank_source = FlowConfig.normalize_query_source(
            profile_raw.get("rerank_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_RERANK_QUERY_SOURCE,
        )
        language_mode = (
            FlowConfig.normalize_retrieval_language(
                profile_raw.get("retrieval_language")
            )
            or FlowConfig.DEFAULT_RETRIEVAL_LANGUAGE_MODE
        )

        # A missing on/off key counts as enabled (documented in Config_Orchestrator).
        switch_defaults = {
            field_name: bool(profile_raw.get(config_key, True))
            for field_name, config_key in self._FLOW_PROFILE_CONFIG_KEYS.items()
        }
        switch_defaults["enable_keyword_action_splitter"] = bool(
            profile_raw.get("use_keyword_action_splitter", True)
        )

        resolved_profile = OrchestrationFlowProfile(
            name=selected,
            retrieve_mode_override=retrieve_mode_override,
            primary_query_source=primary_source,
            guardrail_query_source=guardrail_source,
            confidence_evaluation=QualityDimensionEvaluator.validate_mode(
                profile_raw.get("confidence_evaluation", "both")
            ),
            rerank_query_source=rerank_source,
            retrieval_language_mode=language_mode,
            **switch_defaults,
        )

        mySession.orchestrator_flow = resolved_profile.name  # type: ignore[attr-defined]
        mySession.flow_name = resolved_profile.name  # type: ignore[attr-defined]
        return resolved_profile

    def resolve_effective_flow_profile_for_session(
        self,
        mySession: Session,
        flow_profile: OrchestrationFlowProfile,
    ) -> OrchestrationFlowProfile:
        """Apply session flow-related overrides over flow baseline values."""
        overrides: dict[str, Any] = {}
        confidence_mode = getattr(mySession, "confidence_evaluation", None)
        if confidence_mode is not None:
            overrides["confidence_evaluation"] = (
                QualityDimensionEvaluator.validate_mode(confidence_mode)
            )
        for field_name in self._FLOW_SWITCH_FIELDS:
            session_value = getattr(mySession, field_name, None)
            if session_value is None:
                continue
            overrides[field_name] = bool(session_value)

        session_mode = (
            str(getattr(mySession, "force_retrieve_mode", "") or "").strip().upper()
        )
        if session_mode:
            allowed_modes = self.cfg_get_allowed_retrieve_modes()
            if not allowed_modes or session_mode in allowed_modes:
                overrides["retrieve_mode_override"] = session_mode

        allowed_sources = self.cfg_get_allowed_query_sources()
        session_primary_source = (
            str(getattr(mySession, "main_query_source", "") or "").strip().upper()
        )
        if session_primary_source and session_primary_source in allowed_sources:
            overrides["primary_query_source"] = session_primary_source

        session_secondary_source = (
            str(getattr(mySession, "secondary_query_source", "") or "").strip().upper()
        )
        if session_secondary_source and session_secondary_source in allowed_sources:
            overrides["guardrail_query_source"] = session_secondary_source

        session_rerank_source = FlowConfig.normalize_name(
            getattr(mySession, "rerank_query_source", "")
        )
        if session_rerank_source and session_rerank_source in allowed_sources:
            overrides["rerank_query_source"] = session_rerank_source

        session_language = FlowConfig.normalize_retrieval_language(
            getattr(mySession, "retrieval_language_override", None)
        )
        if session_language:
            overrides["retrieval_language_mode"] = session_language

        if not overrides:
            return flow_profile
        return replace(flow_profile, **overrides)

    @staticmethod
    def narrow_retrievers_for_mode(
        flow_profile: OrchestrationFlowProfile,
        retrieve_mode: str | None,
    ) -> OrchestrationFlowProfile:
        """Limit local retrievers to an explicitly forced retrieve mode.

        Only an explicit force_retrieve_mode (flow or session) narrows; the
        plain session retrieve_mode never does.  ALL/WEB leave the run_*
        switches unchanged (WEB skips local retrieval elsewhere).
        """
        if not flow_profile.retrieve_mode_override:
            return flow_profile
        if not FlowConfig.narrows_local_retrievers(retrieve_mode):
            return flow_profile
        vector, bm25, graph, regex = FlowConfig.mode_flags_for_local_retrievers(
            retrieve_mode
        )
        return replace(
            flow_profile,
            enable_vector_retriever=flow_profile.enable_vector_retriever and vector,
            enable_bm25_retriever=flow_profile.enable_bm25_retriever and bm25,
            enable_graph_retriever=flow_profile.enable_graph_retriever and graph,
            enable_regex_retriever=flow_profile.enable_regex_retriever and regex,
        )

    def query_from_source(
        self,
        mySession: Session,
        *,
        plan: RetrievalPlan,
        source_name: str,
    ) -> str:
        """Resolve query text from a configured source selector."""
        source = str(source_name or self._DEFAULT_QUERY_SOURCE).strip().upper()
        if source == "TRANSLATED_QUERY":
            candidate = plan.orig_translated_query_en
        elif source == "ORIGINAL_QUERY":
            candidate = (
                getattr(mySession, "user_query_original", None)
                or plan.user_query_original
            )
        else:
            candidate = plan.bm25_query

        query_text = str(candidate or "").strip()
        if query_text:
            return query_text
        return str(plan.bm25_query or "")

    def resolve_retrieve_mode_for_flow(
        self,
        *,
        default_retrieve_mode: str,
        flow_profile: OrchestrationFlowProfile,
    ) -> str:
        """Resolve effective retrieve_mode for this turn."""
        override = str(flow_profile.retrieve_mode_override or "").strip().upper()
        if not override:
            return str(default_retrieve_mode or "VECTOR").upper()

        allowed_modes = self.cfg_get_allowed_retrieve_modes()
        if allowed_modes and override not in allowed_modes:
            return str(default_retrieve_mode or "VECTOR").upper()
        return override

    def resolve_grounding_cli_contradiction(
        self,
        mySession: Session,
        *,
        flow_enabled: bool,
    ) -> tuple[bool, str | None, str | None]:
        """Return contradiction details when flow grounding is disabled by CLI mark_text."""
        default_mark_text = self.cfg_get_str("_FRIENDLY_NAME", "") == "RAGChat"
        mark_text_value = bool(getattr(mySession, "mark_text", default_mark_text))
        mark_text_overridden = mark_text_value != default_mark_text
        if flow_enabled and (not mark_text_value) and mark_text_overridden:
            session_scope = "session CLI" if default_mark_text else "session"
            return (
                True,
                f"Deactivated in {session_scope}",
                f"mark_text={str(mark_text_value).lower()}",
            )
        return False, None, None

    def resolve_rerank_cli_contradiction(
        self,
        mySession: Session,
        *,
        flow_enabled: bool,
    ) -> tuple[bool, bool, str | None, str | None]:
        """Return rerank contradiction details plus effective rerank enablement."""
        rerank_raw = getattr(mySession, "rerank", None)
        # When rerank is unset on the session, inherit flow baseline.
        rerank_enabled = bool(flow_enabled) if rerank_raw is None else bool(rerank_raw)
        effective_enabled = bool(flow_enabled and rerank_enabled)

        if flow_enabled == rerank_enabled:
            return False, effective_enabled, None, None

        session_scope = "session override"
        action = (
            "Deactivated in " + session_scope
            if flow_enabled and (not rerank_enabled)
            else "Activated in " + session_scope
        )
        return True, effective_enabled, action, f"rerank={1 if rerank_enabled else 0}"

    @classmethod
    def apply_query_rewrite_flow_controls(
        cls,
        mySession: Session,
        *,
        flow_profile: OrchestrationFlowProfile,
    ) -> None:
        """Persist per-flow toggles on the session object."""
        for field_name in cls._FLOW_SWITCH_FIELDS:
            setattr(mySession, field_name, bool(getattr(flow_profile, field_name)))

        rerank_raw = getattr(mySession, "rerank", None)
        rerank_enabled = True if rerank_raw is None else bool(rerank_raw)
        mySession.enable_low_score_fallback = bool(  # type: ignore[attr-defined]
            flow_profile.enable_low_score_fallback
            and flow_profile.enable_rerank
            and rerank_enabled
        )
        mySession.enable_low_recall_rescue = bool(  # type: ignore[attr-defined]
            flow_profile.enable_low_recall_rescue
            and flow_profile.enable_rerank
            and rerank_enabled
        )
        mySession.enable_grounding = bool(flow_profile.enable_grounding)  # type: ignore[attr-defined]
        mySession.retrieval_language_mode = str(  # type: ignore[attr-defined]
            getattr(flow_profile, "retrieval_language_mode", "ENGLISH") or "ENGLISH"
        ).upper()

    @staticmethod
    def is_web_only_mode(retrieve_mode: str) -> bool:
        """Return True when retrieval mode requests only the web leg."""
        return str(retrieve_mode or "").strip().upper() == "WEB"

    def resolve_web_stage_mode(self, mySession: Session, retrieve_mode: str) -> str:
        """Resolve effective web mode from session and retrieve mode."""
        if self.is_web_only_mode(retrieve_mode):
            return "web_only"
        if bool(getattr(mySession, "web_search", False)):
            return "local_and_web"
        return "local_only"

    def apply_rerank_query_source(
        self,
        mySession: Session,
        *,
        plan: RetrievalPlan,
        flow_profile: OrchestrationFlowProfile,
    ) -> None:
        # None preserves the reranker's legacy fallback to the normalized query.
        mySession.rerank_query = (
            None
            if flow_profile.rerank_query_source == "REWRITTEN_QUERY"
            else self.query_from_source(
                mySession,
                plan=plan,
                source_name=flow_profile.rerank_query_source,
            )
        )
