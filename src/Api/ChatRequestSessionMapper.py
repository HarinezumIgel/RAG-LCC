from __future__ import annotations

import ntpath
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from fastapi import HTTPException

from Chat.ContextSummarizer import ContextSummarizer
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Config.Config import Config
from Globals.Session import Session
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Orchestrator import FlowConfig

if TYPE_CHECKING:
    from Api.ChatCompletionHandler import ChatCompletionRequest, ChatMessage

_OLLAMA_OPTIONS_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "repeat_last_n",
        "repeat_penalty",
        "tfs_z",
        "min_p",
        "use_mmap",
        "use_mlock",
        "num_keep",
        "num_batch",
        "num_thread",
        "num_gpu",
    }
)

# Top-level Ollama payload keys (not inside "options")
_OLLAMA_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"think", "keep_alive", "format"})

_ALLOWED_STRATEGIES: frozenset[str] = frozenset(
    {"ULTRA_WIDE", "WIDE", "BALANCED_FILE_CAP", "NARROW", "DEFAULT"}
)


def _buildQuery(messages: List[ChatMessage]) -> str:
    """
    Extract the user query from the messages array.
    If a system message is present, prepend it to the user query so the RAG
    prompt template still receives a simple {input} string.

    Note: OpenWebUI (and similar frontends) send the full conversation history
    in ``messages``.  We intentionally ignore all but the last user turn here —
    RAG-LCC maintains its own chat history via the persisted ``history/`` files,
    bounded by ``max_history_turns``.  The two histories are independent and do
    not interfere with each other.
    """
    system_text: str | None = None
    user_text: str = ""
    for msg in messages:
        if msg.role == "system" and system_text is None:
            system_text = msg.text().strip()
        elif msg.role == "user":
            user_text = msg.text()  # last user message wins; prior turns ignored
    command, argument = RAGChatImpl.parse_chat_command(user_text)
    if command == "new":
        user_text = argument
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


def _resolve_allowed_orchestration_flows(cfg: Config) -> list[str]:
    """Return configured orchestration flow names in uppercase."""
    return FlowConfig.allowed_flow_names(cfg)


def _resolve_default_orchestration_flow(cfg: Config) -> str:
    """Return default orchestration flow used when session value is unset."""
    return FlowConfig.default_flow_name(cfg)


def _getLastUserText(req: ChatCompletionRequest) -> str | None:
    """Return the text of the last user message, or None."""
    if not req.messages:
        return None
    last_user = None
    for msg in req.messages:
        if msg.role == "user":
            last_user = msg
    return last_user.text() if last_user is not None else None


class RequestSessionMapper:
    """Map one validated API request onto its request-scoped session."""

    def __init__(self, query_parts: QueryParts, cfg: Config) -> None:
        self.query_parts = query_parts
        self.cfg = cfg

    def apply(self, req: ChatCompletionRequest, session: Session) -> str:
        """Apply validated request overrides after strategy/flow defaults."""
        queryParts = self.query_parts
        cfg = self.cfg

        # Switch collection if needed.
        # Option 1 semantics: preserve user overrides on collection change,
        # but clear file/path/metadata filters that are collection-scoped.
        incoming_collection = req.model
        previous_collection = session.collection_name
        is_session_bootstrap = previous_collection is None
        collection_changed = (
            previous_collection is not None
            and previous_collection != incoming_collection
        )
        if collection_changed:
            session.file_name = None
            session.file_path = None
            if hasattr(session, "file_path_select"):
                setattr(session, "file_path_select", None)
            if hasattr(session, "metadata_filters"):
                setattr(session, "metadata_filters", {})
        session.collection_name = incoming_collection
        session.is_service = True

        selections: dict[str, str] = {}
        for name in ("file", "path"):
            raw = getattr(req, name)
            if raw is not None:
                value = raw.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1].strip()
                if value and (
                    "\x00" in value or ntpath.basename(value) in ("", ".", "..")
                ):
                    raise HTTPException(
                        status_code=400, detail=f"Invalid {name} selection."
                    )
                if name == "file" and ntpath.dirname(value):
                    raise HTTPException(
                        status_code=400,
                        detail="Use path for a document path, or file for a filename.",
                    )
                selections[name] = value
        if selections.get("file") and selections.get("path"):
            raise HTTPException(
                status_code=400, detail="Specify either file or path, not both."
            )
        if selections:
            session.file_name = selections.get("file") or None
            session.file_path = selections.get("path") or None
            session.file_path_select = (
                "path" if session.file_path else "file" if session.file_name else None
            )

        # Determine strategy: request > session > config default
        # Treat empty/whitespace-only values the same as absent (OpenWebUI may send strategy="" when unset)
        strategy_raw = req.strategy
        strategy_requested = bool(strategy_raw is not None and strategy_raw.strip())
        if strategy_requested:
            strategy_upper = str(strategy_raw).strip().upper()
            if strategy_upper not in _ALLOWED_STRATEGIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid strategy {req.strategy!r}. Allowed: {sorted(_ALLOWED_STRATEGIES)}",
                )
            session.strategy = strategy_upper

        should_apply_strategy_defaults = is_session_bootstrap or strategy_requested
        if should_apply_strategy_defaults:
            strategy = (
                session.strategy
                or cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY")
                or "DEFAULT"
            )
            queryParts.applyStrategyDefaults(strategy, session=session)

        allowed_orchestrator_flows = _resolve_allowed_orchestration_flows(cfg)
        flow_raw = req.orchestrator_flow
        flow_requested = bool(flow_raw is not None and flow_raw.strip())
        if flow_requested:
            flow_upper = FlowConfig.resolve_flow_alias(flow_raw, cfg)
            if flow_upper not in allowed_orchestrator_flows:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid orchestrator_flow {req.orchestrator_flow!r}. "
                        f"Allowed: {sorted(allowed_orchestrator_flows)}"
                    ),
                )
            session.orchestrator_flow = flow_upper

        has_flow_in_session = bool(str(session.orchestrator_flow or "").strip())
        should_apply_flow_defaults = (
            is_session_bootstrap or flow_requested or not has_flow_in_session
        )
        if should_apply_flow_defaults:
            selected_flow = (
                session.orchestrator_flow or _resolve_default_orchestration_flow(cfg)
            )
            queryParts.applyOrchestrationFlowDefaults(
                selected_flow,
                session=session,
            )

        if req.confidence_evaluation is not None:
            try:
                session.confidence_evaluation = QualityDimensionEvaluator.validate_mode(
                    req.confidence_evaluation
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        summary_settings: dict[str, str | float | int] = {}
        for name in ("summary_length", "summary_temperature", "chunk_overlap"):
            value = getattr(req, name)
            if value is not None:
                try:
                    summary_settings[name] = ContextSummarizer.validate_setting(
                        name, value
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        for name, value in summary_settings.items():
            setattr(session, name, value)

        request_notices: list[str] = []

        def _as_bool_or_none(value: Any) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off"):
                return False
            return None

        def _clamp_01(value: float) -> float:
            return max(0.0, min(1.0, value))

        def _normalize_web_search_mode(raw_value: bool | str) -> str:
            if isinstance(raw_value, bool):
                return "local_and_web" if raw_value else "local_only"
            text = str(raw_value).strip().strip("\"'").lower()
            if text in ("true", "1"):
                return "local_and_web"
            if text in ("false", "0"):
                return "local_only"
            return text

        def _apply_query_source(value: str | None, session_attr: str) -> None:
            if value is None or not value.strip():
                return
            requested = value.strip().upper()
            if requested not in allowed_query_sources:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid {session_attr} {value!r}. "
                        f"Allowed: {allowed_query_sources}"
                    ),
                )
            setattr(session, session_attr, requested)

        allowed_query_sources: list[str] = FlowConfig.allowed_query_sources(cfg)

        # RAG-LCC param overrides (applied after strategy defaults so they take precedence)
        retriever_k = req.fetch_k if req.fetch_k is not None else req.retriever_k
        if retriever_k is not None:
            session.retriever_k = retriever_k
        if req.rerank is not None:
            session.rerank = req.rerank
        if req.enable_keyword_action_splitter is not None:
            session.enable_keyword_action_splitter = req.enable_keyword_action_splitter

        model_extra = req.model_extra or {}
        for key, raw_value in model_extra.items():
            field_name = str(key)
            if not field_name.startswith("enable_"):
                continue
            if not hasattr(session, field_name):
                continue
            parsed_bool = _as_bool_or_none(raw_value)
            if parsed_bool is None:
                continue
            setattr(session, field_name, parsed_bool)
            if field_name == "enable_rerank" and req.rerank is None:
                session.rerank = parsed_bool

        if req.force_retrieve_mode is not None and req.force_retrieve_mode.strip():
            requested_mode = req.force_retrieve_mode.strip().upper()
            allowed_modes = {
                str(item).strip().upper()
                for item in cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                if str(item).strip()
            }
            if allowed_modes and requested_mode not in allowed_modes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid force_retrieve_mode {req.force_retrieve_mode!r}. "
                        f"Allowed: {sorted(allowed_modes)}"
                    ),
                )
            session.force_retrieve_mode = requested_mode

        _apply_query_source(req.main_query_source, "main_query_source")
        _apply_query_source(req.secondary_query_source, "secondary_query_source")
        _apply_query_source(req.rerank_query_source, "rerank_query_source")

        if req.retrieval_language is not None and req.retrieval_language.strip():
            language_mode = FlowConfig.normalize_retrieval_language(
                req.retrieval_language
            )
            if language_mode is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid retrieval_language {req.retrieval_language!r}. "
                        "Allowed: ['english', 'native']"
                    ),
                )
            session.retrieval_language_override = language_mode

        threshold = req.threshold if req.threshold is not None else req.chroma_threshold
        if threshold is not None:
            session.chroma_threshold = _clamp_01(threshold)

        chunks = (
            req.context_chunks
            if req.context_chunks is not None
            else req.final_chunks_to_llm
        )
        if chunks is not None:
            session.final_chunks_to_llm = chunks

        weight_overrides: dict[str, float | None] = {
            "vector_weight": req.vector_weight,
            "bm25_weight": req.bm25_weight,
            "graph_weight": req.graph_weight,
            "regex_weight": req.regex_weight,
        }
        for field_name, value in weight_overrides.items():
            if value is not None:
                setattr(session, field_name, _clamp_01(value))

        if req.use_chat_context is not None:
            session.use_chat_context = req.use_chat_context

        # chat_name priority: explicit param > OpenWebUI chat_id > keep existing
        if req.chat_name is not None:
            session.chat_name = req.chat_name
        elif req.chat_id is not None:
            session.chat_name = req.chat_id

        file_cap = req.file_cap if req.file_cap is not None else req.per_file_limit
        if file_cap is not None:
            session.per_file_limit = file_cap

        if req.web_search is not None:
            web_search_mode = _normalize_web_search_mode(req.web_search)
            if web_search_mode in ("local_and_web", "web_only"):
                web_mode = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
                if web_mode != "1":
                    request_notices.append(
                        "\U0001f6ab **Web search unavailable** \u2014 "
                        "disabled by the administrator "
                        '(`WEB_SEARCH_MODE != "1"` in `Config_Internet_Env.py`). '
                        "Answering from the local knowledge base only.\n\n---\n\n"
                    )
                    session.web_search = False
                else:
                    session.web_search = True
                    if web_search_mode == "web_only":
                        session.retrieve_mode = "WEB"
            else:  # "local_only" or unrecognised value
                session.web_search = False
                if getattr(session, "retrieve_mode", None) == "WEB":
                    # Keep session state coherent: if web search is off, WEB-only
                    # mode must be replaced with a local retrieval mode.
                    allowed_rm: list[str] = cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                    session.retrieve_mode = (
                        "ALL" if "ALL" in allowed_rm else allowed_rm[0]
                    )
        elif (
            cfg.get_bool("_OPENWEB_UI_WEBSEARCH", False)
            and str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower() == "1"
        ):
            # No explicit per-request value — apply the service-level default.
            session.web_search = True

        if req.web_weight is not None:
            session.web_weight = max(0.0, req.web_weight)
        if req.fetch_page_content is not None:
            session.fetch_page_content = req.fetch_page_content
        if req.web_fetch_k is not None:
            if req.web_fetch_k < 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid web_fetch_k {req.web_fetch_k!r}. "
                        "Value must be an integer >= 1."
                    ),
                )
            session.web_fetch_k = req.web_fetch_k

        # Enable visual markers (yellow highlighting) and HTTP serving only when enabled.
        # Answer grounding (orange) works independently via chunk_texts_for_grounding.
        session.mark_text = (
            os.environ.get("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", "0") == "1"
        )

        if req.debug_level is not None or req.debug_mode is not None:
            level = (
                req.debug_level
                if req.debug_level is not None
                else session.debug_level or 0
            )
            mode = (
                req.debug_mode.strip().lower() if req.debug_mode is not None else None
            ) or "ge"
            if mode not in ("ge", "is", "le"):
                mode = "ge"
            session.debug_level = level
            session.debug_mode = mode
            combined = "none" if level == 0 else f"{mode} {level}"
            cfg.set("DEBUG_LEVEL", combined)

        # Per-request LLM param overrides
        if req.temperature is not None:
            session.temperature = req.temperature
        if req.top_p is not None:
            session.top_p = req.top_p
        if req.top_k is not None:
            session.top_k = float(req.top_k)

        max_out = (
            req.max_output_tokens
            if req.max_output_tokens is not None
            else req.max_tokens
        )
        if max_out is not None:
            session.max_output_tokens_override = max_out

        ctx = req.context_size if req.context_size is not None else req.num_ctx
        if ctx is not None:
            session.context_size_override = ctx

        # Build extra Ollama options from the request
        extra_options: dict[str, Any] = {}
        for key in _OLLAMA_OPTIONS_KEYS:
            value = getattr(req, key, None)
            if value is not None:
                extra_options[key] = value
        session.extraOllamaOptions = extra_options if extra_options else None

        # Build top-level Ollama payload params from the request
        top_level: dict[str, Any] = {}
        for key in _OLLAMA_TOP_LEVEL_KEYS:
            value = getattr(req, key, None)
            if value is not None:
                top_level[key] = value
        session.ollamaTopLevelParams = top_level if top_level else None

        session.user_query_input = _getLastUserText(req)
        session.query = _buildQuery(req.messages)
        session.base_kwargs = {"k": session.retriever_k}
        if session.file_path:
            session.base_kwargs["filter"] = {"FilePath": {"$eq": session.file_path}}
        elif session.file_name:
            session.base_kwargs["filter"] = {"FileName": {"$eq": session.file_name}}
        pinned = session.file_path or session.file_name
        if pinned:
            command, target = RAGChatImpl.parse_chat_command(
                _getLastUserText(req) or ""
            )
            notice = f"Pinned document (chat and summarize): {pinned}"
            if command == "summarize" and target:
                notice += f"\nSummary target for this turn: {target} (pin unchanged)"
            request_notices.append(notice + "\n\n---\n\n")
        return "".join(request_notices)
