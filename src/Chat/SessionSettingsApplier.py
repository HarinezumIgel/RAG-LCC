"""Session-setting validation, mutation, and default application."""

import os
from typing import Any, Optional

from InquirerPy.validator import ValidationError  # type: ignore[attr-defined]

from AI.TokenBudget import TokenBudget
from Chat.ContextSummarizer import ContextSummarizer
from Chat.QueryPartsProtocol import QueryPartsContext
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import ORANGE, RESET, YELLOW
from Gui.HistoryManager import HistoryManager
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Orchestrator import FlowConfig


class SessionSettingsApplier:
    cfg: Config
    helpers: Helpers
    session: Session
    tokenBudget: TokenBudget
    allowed_orchestration_flows: list[str]

    def _cast_value(self: QueryPartsContext, tok: str, raw: str) -> Any:
        spec: dict[str, Any] | None = self.COMMAND_SPECS.get(tok)
        if not spec or "type" not in spec:
            return raw
        t: str = spec["type"]
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            return self.to_bool(raw)
        # default: string
        return str(raw)

    def _get_attr_name(self: QueryPartsContext, tok: str) -> Optional[str]:
        spec: dict[str, Any] | None = self.COMMAND_SPECS.get(tok)
        if not spec:
            return None
        return spec.get("attr")

    def _set_debug(self: QueryPartsContext, level: int, mode: str) -> None:
        self.session.debug_level = level
        self.session.debug_mode = mode
        combined = "none" if level == 0 else f"{mode} {level}"
        self.cfg.set("DEBUG_LEVEL", combined, force=True)
        FileUtils.configure_network_debug(
            self.cfg.get_bool("URL_DEBUG")
            or DebugHelper.check_session(self.session, 55)
        )

    def _set(self: QueryPartsContext, tok: str, p: str):
        if p is None:  # type: ignore[reportUnnecessaryComparison]
            print("No value specified\n")
            return
        if tok in ("file", "path"):
            fn, fp = os.path.basename(p), os.path.abspath(p)
            self.fileList.set(
                f"{self.session.collection_name}_{self.session.chat_name}",
                "File",
                fn,
                fp,
            )
            self.session.file_name = fn if tok == "file" else None
            self.session.file_path = fp if tok == "path" else None
            self.session.file_path_select = tok.lower()
            return

        if tok == "metadata":
            sep = ":" if ":" in p else ("=" if "=" in p else "")
            if not sep:
                print(
                    "⚠ metadata expects Field:Value (e.g. metadata=Author:python-docx)"
                )
                return
            field, _, value = p.partition(sep)
            field, value = field.strip(), value.strip().strip("'\"")
            if not field or not value:
                print("⚠ metadata expects a non-empty Field and Value")
                return
            self.session.metadata_filters[field] = value
            self._print_metadata_filters()
            return

        if tok.startswith("enable_"):
            if not hasattr(self.session, tok):
                print(f"⚠ Invalid command: {tok}")
                self.help()
                return
            try:
                self._validate_raw_for_type(p, "bool")
            except ValueError as e:
                print(f"⚠ {e}")
                return
            bool_value = self.to_bool(p)
            setattr(self.session, tok, bool_value)
            if tok == "enable_rerank":
                self.session.rerank = bool_value
            return

        if tok == "strategy":
            if p.upper() not in self.allowed_strategies:
                print(
                    f"⚠ Invalid strategy '{p.upper()}'. Allowed: {self.allowed_strategies}"
                )
            else:
                # delegate to helper so defaults and side effects are consistent
                self._read_and_apply_value(
                    "strategy", p.upper(), from_interactive=False
                )
            return

        if tok == "collection":
            if self.session.collection_name:
                self.hist.save(
                    HistoryManager.key(
                        self.session.collection_name, self.session.chat_name
                    ),
                    "Settings",
                )
            self.session.collection_name = p
            self.ragChatImpl.set_vector_store(self.session)
            self.reset_things()
            return

        if tok == "terminal_line_size":
            try:
                val = int(p)
            except ValueError:
                print(f"⚠ terminal_line_size must be an integer, got {p!r}")
                return
            if val < 40:
                print(f"\u26a0 terminal_line_size={val} is too small (minimum 40).")
                return
            self.cfg.set("TERMINAL_LINE_SIZE", val, force=True)
            return

        if tok == "debug_level":
            try:
                level, mode = DebugHelper.parse(p)
            except ValueError as e:
                print(f"⚠ {e}")
                return
            allowed_values: list[int] = list(self.allowed_debug_levels.values())
            if level != 0 and level not in allowed_values:
                allowed_names = "  ".join(
                    f"{v}={k}" for k, v in self.allowed_debug_levels.items() if v != 0
                )
                print(
                    f"⚠ {level} is not a recognised debug level. Allowed: {allowed_names}"
                )
                return
            self._set_debug(level, mode)
            return

        if tok == "debug_mode":
            v = p.strip().lower()
            if v not in ("ge", "is", "le"):
                print(
                    f"⚠ Invalid debug_mode '{p}'. Allowed: ge  (>=)   is  (==)   le  (<=)"
                )
                return
            self._set_debug(self.session.debug_level or 0, v)
            return

        if tok in self._SET_DIRECT_VALUE_TOKENS:
            try:
                self._read_and_apply_value(tok, p, from_interactive=False)
            except ValueError as exc:
                if tok in (
                    "confidence_evaluation",
                    "summary_length",
                    "summary_temperature",
                    "chunk_overlap",
                ):
                    print(f"\u26a0 {exc}")
                    return
                print(f"⚠ Invalid command: {tok}")
                self.help()
            return

    def _clear(self: QueryPartsContext, tok: str):
        s = self.session
        if tok in ("summary_length", "summary_temperature", "chunk_overlap"):
            setattr(s, tok, None)
            print(f"[{tok}] override cleared (configuration default will be used)")
        elif tok in ("file", "path"):
            s.file_name, s.file_path = None, None
            s.file_path_select = None
        elif tok == "metadata":
            s.metadata_filters = {}
            print("[metadata] all metadata filters cleared")
        elif tok == "max_output_tokens":
            s.max_output_tokens_override = None
            print("[max_output_tokens] override cleared — dynamic budget will be used")
        elif tok == "context_size":
            s.context_size_override = None
            print(
                f"[context_size] override cleared — auto={self.tokenBudget.get_context_limit()} will be used"
            )
        elif tok == "web_fetch_k":
            s.web_fetch_k = None
            default_k = self.cfg.get_int("_WEB_SEARCH.max_results") or 5
            print(
                f"[web_fetch_k] override cleared — default={default_k} from _WEB_SEARCH.max_results"
            )
        elif tok == "orchestrator_flow":
            resolved = self.applyOrchestrationFlowDefaults(
                self._default_orchestration_flow(), session=s
            )
            print(f"[orchestrator_flow] reset to {resolved}")
        elif tok in (
            "confidence_evaluation",
            "force_retrieve_mode",
            "main_query_source",
            "secondary_query_source",
            "rerank_query_source",
            "retrieval_language",
        ):
            setattr(s, self._get_attr_name(tok) or tok, None)
            print(f"[{tok}] cleared (flow baseline will be used)")
        elif tok.startswith("enable_") and hasattr(s, tok):
            setattr(s, tok, None)
            if tok == "enable_rerank":
                s.rerank = None
            print(f"[{tok}] cleared (flow baseline will be used)")

    def applyStrategyDefaults(
        self: QueryPartsContext, strategy: str, session: "Session | None" = None
    ) -> str:
        """Public entry point for loading strategy defaults into Session. Returns the collection name."""
        return self._base_defaults(strategy, session=session)

    def strategy_context_warning(
        self: QueryPartsContext, session: Session
    ) -> str | None:
        """Estimate selection-time overflow risk, not the fit of a future prompt."""
        chunks = session.final_chunks_to_llm
        if chunks is None or chunks <= 0:
            return "Cannot estimate strategy context size: context_chunks must be positive."
        chunk_words = self.helpers.get_chunker_max_size()
        if chunk_words <= 0:
            return "Cannot estimate strategy context size: the active chunker size must be positive."
        model_args = self.helpers.get_model_args("_ACTIVE_LLM")
        model: str = model_args["MODEL"]
        context_limit = self.tokenBudget.get_effective_context_limit(model, session)
        reserved = int(
            model_args.get(
                "TOKEN_BUDGET_RESERVED_SYSTEM", self.tokenBudget.reserved_system
            )
        ) + int(
            model_args.get(
                "TOKEN_BUDGET_RESERVED_OUTPUT", self.tokenBudget.reserved_output
            )
        )
        # Reuse the existing word-based estimator for one nominal-sized chunk.
        # Indexed chunks, headers and the actual query may differ from this estimate.
        chunk_tokens = self.tokenBudget.count_tokens_approx("word " * chunk_words)
        estimated_context = chunks * chunk_tokens
        if estimated_context <= context_limit - reserved:
            return None
        return (
            f"{session.strategy}: active context_chunks={chunks:,} could require approximately "
            f"{estimated_context:,} tokens of retrieved text, exceeding the "
            f"effective model window ({context_limit:,} tokens) for {model} after reserving "
            f"{reserved:,} tokens for system/output. Actual fit depends on the retrieved chunks "
            "and complete prompt. Consider a narrower strategy, fewer context chunks, or a larger "
            "LLM context window if supported by your model and available hardware."
        )

    def _warn_strategy_context(self: QueryPartsContext) -> None:
        warning = self.strategy_context_warning(self.session)
        if warning:
            self.pretty.write("W", "Strategy context", warning)

    def applyOrchestrationFlowDefaults(
        self: QueryPartsContext,
        orchestrator_flow: str,
        session: "Session | None" = None,
    ) -> str:
        """Apply orchestration-flow selection to session and return the resolved name."""
        s = session or self.session
        configured_flows = list(self.allowed_orchestration_flows)
        fallback = self._default_orchestration_flow()
        resolved = (
            FlowConfig.resolve_flow_alias(orchestrator_flow, self.cfg) or fallback
        )
        if configured_flows and resolved not in configured_flows:
            resolved = fallback if fallback in configured_flows else configured_flows[0]
        s.orchestrator_flow = resolved
        s.flow_name = resolved
        self._apply_flow_switch_defaults(
            s,
            self._resolve_flow_profile_defaults(resolved),
        )
        s.confidence_evaluation = None
        return resolved

    def _resolve_flow_profile_defaults(
        self: QueryPartsContext,
        flow_name: str,
    ) -> dict[str, Any]:
        """Return the raw config dictionary for one orchestration flow."""
        return FlowConfig.flow_profile_raw(self.cfg, flow_name)

    def _apply_flow_switch_defaults(
        self: QueryPartsContext,
        session: Session,
        profile_raw: dict[str, Any],
    ) -> None:
        """Persist orchestration-flow switch defaults on the session object."""
        for flow_switch, session_attr in self._FLOW_SWITCH_TO_SESSION_ATTR.items():
            setattr(session, session_attr, bool(profile_raw.get(flow_switch, True)))

        allowed_modes = {
            str(item).strip().upper()
            for item in self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
            if str(item).strip()
        }
        force_mode = (
            str(profile_raw.get("force_retrieve_mode", "") or "").strip().upper()
        )
        if force_mode and allowed_modes and force_mode not in allowed_modes:
            force_mode = ""
        session.force_retrieve_mode = force_mode or None

        allowed_sources = FlowConfig.allowed_query_sources(self.cfg)
        session.main_query_source = FlowConfig.normalize_query_source(
            profile_raw.get("main_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_MAIN_QUERY_SOURCE,
        )
        session.secondary_query_source = FlowConfig.normalize_query_source(
            profile_raw.get("secondary_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_SECONDARY_QUERY_SOURCE,
        )
        session.rerank_query_source = FlowConfig.normalize_query_source(
            profile_raw.get("rerank_query_source"),
            allowed_sources,
            FlowConfig.DEFAULT_RERANK_QUERY_SOURCE,
        )
        session.retrieval_language_override = (
            FlowConfig.normalize_retrieval_language(
                profile_raw.get("retrieval_language")
            )
            or FlowConfig.DEFAULT_RETRIEVAL_LANGUAGE_MODE
        )

        # Flow owns rerank baseline; explicit session/API overrides still apply later.
        session.rerank = bool(getattr(session, "enable_rerank", True))

    def _base_defaults(
        self: QueryPartsContext, strategy: str, *, session: "Session | None" = None
    ) -> str:
        """Read strategy config values and apply them to the session. Returns the collection name."""
        s = session or self.session
        key = strategy.upper()
        s.strategy = key  # record which strategy is active
        chunks_win = self.cfg.get_int(f"_STRATEGIES.{key}.final_chunks_to_llm")
        chroma_kval = self.cfg.get_int(f"_STRATEGIES.{key}.retriever_k")
        thr = self.cfg.get_float(f"_STRATEGIES.{key}.threshold")
        tok_val = self.cfg.get_int(f"_STRATEGIES.{key}.max_output_tokens")
        temp = self.cfg.get_float(f"_STRATEGIES.{key}.temperature")
        top_p = self.cfg.get_float(f"_STRATEGIES.{key}.top_p")
        top_k = self.cfg.get_int(f"_STRATEGIES.{key}.top_k")
        vw = self.cfg.get_float(f"_STRATEGIES.{key}.vector_weight")
        bw = self.cfg.get_float(f"_STRATEGIES.{key}.bm25_weight")
        gw = self.cfg.get_float(f"_STRATEGIES.{key}.graph_weight")
        rw = self.cfg.get_float(f"_STRATEGIES.{key}.regex_weight")
        ww = self.cfg.get_float(f"_STRATEGIES.{key}.web_weight")
        if self.cfg.get("COLLECTION") is not None:
            cn = self.helpers.embedder_collection_name()
        else:
            cn = self.cfg.get_str(f"_STRATEGIES.{key}.collection")
        fl = self.cfg.get_int(f"_STRATEGIES.{key}.filelim")
        use_ctx = self.cfg.get_bool(f"_STRATEGIES.{key}.use_chat_context")
        turn = self.cfg.get_int(f"_STRATEGIES.{key}.turns")
        bs = self.cfg.get_int(f"_STRATEGIES.{key}.prune_batch")
        mht = self.cfg.get_int(f"_STRATEGIES.{key}.max_history_turns")
        tsm = self.cfg.get_str(f"_STRATEGIES.{key}.TOPIC_SUMMARY_MODE") or "last"

        s.final_chunks_to_llm = chunks_win
        s.retriever_k = chroma_kval
        s.chroma_threshold = thr
        s.max_output_tokens = tok_val
        s.temperature = temp
        s.top_p = top_p
        s.top_k = top_k
        s.vector_weight = vw
        s.bm25_weight = bw
        s.graph_weight = gw
        s.regex_weight = rw
        # web_weight can be pre-set per-strategy; web_search and fetch_page_content
        # are session-persistent knobs set deliberately by the user or via OpenWebUI
        # Advanced Parameters — strategy loading does not reset them.
        s.web_weight = ww
        s.per_file_limit = fl
        s.use_chat_context = use_ctx
        s.turns = turn
        s.prune_batch = bs
        s.max_history_turns = mht
        s.topic_summary_mode = tsm

        # Always use debug_level from Config_Global.py (respects CLI override if present)
        _debug_str = self.cfg.get_str("DEBUG_LEVEL", "none")
        _lvl, _mode = DebugHelper.parse(_debug_str)
        s.debug_level = _lvl
        s.debug_mode = _mode

        return cn

    def _validate_raw_for_type(
        self: QueryPartsContext, raw: str, type_name: str
    ) -> None:
        t = type_name or "str"
        v = raw.strip()
        if t == "int":
            if not v.lstrip("-").isdigit():
                raise ValueError("Must be an integer")
            return
        if t == "debug_level":
            try:
                DebugHelper.parse(v)
            except ValueError as e:
                raise ValueError(str(e))
            return
        if t == "float":
            try:
                float(v)
            except ValueError:
                raise ValueError("Must be a number")
            return
        if t == "bool":
            if v.lower() not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
                raise ValueError("Must be a boolean (true/false, yes/no, 1/0)")
            return
        # string: non-empty
        if v == "":
            raise ValueError("Value cannot be empty")

    def _read_and_apply_value(
        self: QueryPartsContext, tok: str, raw: str, *, from_interactive: bool = False
    ):
        """
        Validate, cast and set the session attribute for tok.
        If tok is file/path this helper will not handle file list logic; callers handle pickers and file parsing.
        """
        spec = self.COMMAND_SPECS.get(tok, {})
        expected_type = spec.get("type", "str")
        if tok in ("summary_length", "summary_temperature", "chunk_overlap"):
            try:
                value = ContextSummarizer.validate_setting(tok, raw)
            except ValueError as exc:
                if from_interactive:
                    raise ValidationError(message=str(exc)) from exc
                raise
            setattr(self.session, tok, value)
            return
        # Validate
        try:
            self._validate_raw_for_type(raw, expected_type)
        except ValueError as e:
            # Interactive callers should raise ValidationError for inquirer
            if from_interactive:
                raise ValidationError(message=str(e))
            # Non-interactive callers get a ValueError
            raise

        if tok == "confidence_evaluation":
            try:
                mode = QualityDimensionEvaluator.validate_mode(raw)
            except ValueError as exc:
                if from_interactive:
                    raise ValidationError(message=str(exc)) from exc
                raise
            self.session.confidence_evaluation = mode
            return

        if tok == "retrieval_language":
            language_mode = FlowConfig.normalize_retrieval_language(raw)
            if language_mode is None:
                msg = f"Invalid retrieval_language '{raw.strip()}'. Allowed: english, native"
                if from_interactive:
                    raise ValidationError(message=msg)
                print(f"\u26a0 {msg}")
                return
            self.session.retrieval_language_override = language_mode
            return

        # Cast and set
        val_cast = self._cast_value(tok, raw.strip())
        if tok == "web_fetch_k" and int(val_cast) < 1:
            msg = "web_fetch_k must be an integer >= 1."
            if from_interactive:
                raise ValidationError(message=msg)
            print(f"⚠ {msg}")
            return

        attr = self._get_attr_name(tok)
        if attr:
            previous_value = getattr(self.session, attr, None)
            setattr(self.session, attr, val_cast)
            # special side effects
            if tok == "chat_name":
                self.reset_things()
            if tok == "strategy":
                # ensure defaults are applied when setting strategy via '='
                strategy: str = val_cast
                strategy.upper()
                self._base_defaults(strategy)
            if tok in ("strategy", "context_chunks", "context_size"):
                self._warn_strategy_context()
            if tok == "orchestrator_flow":
                requested_flow = FlowConfig.resolve_flow_alias(val_cast, self.cfg)
                allowed_flows = list(self.allowed_orchestration_flows)
                if allowed_flows and requested_flow not in allowed_flows:
                    msg = (
                        f"Invalid orchestrator_flow '{val_cast}'. "
                        f"Allowed: {', '.join(allowed_flows)}"
                    )
                    setattr(self.session, attr, previous_value)
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"⚠ {msg}")
                    return
                self.applyOrchestrationFlowDefaults(requested_flow)
            if tok == "force_retrieve_mode":
                mode_text = str(val_cast or "").strip().upper()
                allowed_modes = [
                    str(item).strip().upper()
                    for item in self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                    if str(item).strip()
                ]
                if mode_text not in allowed_modes:
                    msg = (
                        f"Invalid force_retrieve_mode '{val_cast}'. "
                        f"Allowed: {', '.join(allowed_modes)}"
                    )
                    setattr(self.session, attr, previous_value)
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"⚠ {msg}")
                    return
                setattr(self.session, attr, mode_text)
            if tok in (
                "main_query_source",
                "secondary_query_source",
                "rerank_query_source",
            ):
                source_text = str(val_cast or "").strip().upper()
                allowed_sources = FlowConfig.allowed_query_sources(self.cfg)
                if source_text not in allowed_sources:
                    msg = (
                        f"Invalid {tok} '{val_cast}'. "
                        f"Allowed: {', '.join(allowed_sources)}"
                    )
                    setattr(self.session, attr, previous_value)
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"⚠ {msg}")
                    return
                setattr(self.session, attr, source_text)
            if tok == "web_search":
                _ws_raw = val_cast.strip().lower()
                if _ws_raw not in ("local_only", "local_and_web", "web_only"):
                    msg = f"Invalid web_search value '{val_cast}'. Allowed: local_only, local_and_web, web_only."
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"\u26a0 {msg}")
                    return
                want_web = _ws_raw in ("local_and_web", "web_only")
                _web_mode = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
                if want_web and _web_mode != "1":
                    setattr(self.session, attr, False)  # undo premature general setattr
                    msg = 'Web search is disabled by administrator (WEB_SEARCH_MODE must be "1" in Config_Internet_Env.py).'
                    print(f"\u26a0 {msg}")
                    return
                # Apply: web_only sets retrieve_mode=WEB; local_and_web/local_only clear it if currently WEB.
                setattr(self.session, attr, want_web)
                if _ws_raw == "web_only":
                    self.session.retrieve_mode = "WEB"
                elif getattr(self.session, "retrieve_mode", None) == "WEB":
                    # Switching away from web_only — restore default retrieve_mode.
                    allowed_rm = self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                    self.session.retrieve_mode = (
                        "ALL" if "ALL" in allowed_rm else allowed_rm[0]
                    )
            if tok == "fetch_page_content":
                val = val_cast.strip().lower()
                _allowed_fpc = ("snippets only", "fetch pages")
                if val not in _allowed_fpc:
                    msg = f"Invalid fetch_page_content value '{val_cast}'. Allowed: {', '.join(_allowed_fpc)}"
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"\u26a0 {msg}")
                    return
                setattr(self.session, attr, val == "fetch pages")
            if tok == "context_size":
                cap = self.tokenBudget.get_context_limit()
                if val_cast > cap:
                    print(
                        f"{YELLOW}⚠ context_size={val_cast} exceeds hardware cap={cap}; "
                        f"Chatter will clamp to {cap}{RESET}"
                    )
                # Warn when num_ctx is so small that the full prompt
                # (template + retrieved context + query) cannot fit.
                # 256 is the absolute floor; below 2048 things get tight.
                if val_cast <= 256:
                    print(
                        f"{ORANGE}⚠ context_size={val_cast} is dangerously low. "
                        f"Ollama uses this as the total KV-cache for input + output. "
                        f"The full prompt (instructions, retrieved context, and query) "
                        f"will be physically truncated — the model will never see "
                        f"the grounding instructions or the retrieved documents and "
                        f"will hallucinate from its training data instead.{RESET}"
                    )
                elif val_cast < 2048:
                    print(
                        f"{YELLOW}⚠ context_size={val_cast} is tight. The full prompt "
                        f"(instructions + retrieved context + query) plus the model's "
                        f"reply must all fit within {val_cast} tokens. Key instructions "
                        f"or context may be truncated, causing the model to hallucinate "
                        f"rather than answer from the provided documents.{RESET}"
                    )
        else:
            raise ValueError(f"Invalid command: {tok}")

    def to_bool(self: QueryPartsContext, value: int | str | bool) -> bool:
        if isinstance(value, (bool, int)):
            return bool(value)
        return str(value).strip().lower() in ("true", "1", "yes", "y")
