# Local module imports
import os
# Standard library imports
import re
from typing import Any, List

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]

inquirer: Any = _inquirer
from InquirerPy.base.control import Choice  # type: ignore[attr-defined]
from InquirerPy.validator import ValidationError  # type: ignore[attr-defined]

from AI.TokenBudget import TokenBudget
from Chat.ContextSummarizer import ContextSummarizer
from Chat.QueryCommandMetadata import (_ASK_DIRECT_VALUE_TOKENS,
                                       _FLOW_SWITCH_TO_SESSION_ATTR,
                                       _FRIENDLY_TOKEN_MAP,
                                       _PARSER_ONLY_TOKENS,
                                       _SET_DIRECT_VALUE_TOKENS,
                                       _SETTINGS_LAYOUT)
from Chat.QueryCommandMetadata import COMMAND_SPECS as _COMMAND_SPECS
from Chat.QueryCommandMetadata import EXTRA_TOKENS
from Chat.QueryPartsProtocol import QueryPartsContext
from Chat.QuerySettingsPresenter import QuerySettingsPresenter
from Chat.RAGChatImpl import RAGChatImpl
from Chat.SessionSettingsApplier import SessionSettingsApplier
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Session import Session
from Gui.CollectionPicker import CollectionPicker
from Gui.FileList import FileList
from Gui.HistoryManager import HistoryManager
from Gui.MetadataPicker import MetadataPicker
from Gui.Picker import PickerCancelled, pick_value
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers
from Helpers.LanguageConfig import (get_active_language_codes,
                                    get_lang_code_to_name)
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Orchestrator import FlowConfig


# ——— Command & Cmd types ———
class Cmd:
    def __init__(self, token: str, op: str, payload: Any = None) -> None:
        self.token: str = token
        self.op: str = op
        self.payload: Any = payload


class QuerySettingsPicker:
    def _pick_file(self: QueryPartsContext, session: Session, mode: str) -> bool:
        store = f"{session.collection_name}_{session.chat_name}"
        entries = self.fileList.select(store, "File")
        if not entries:
            return False
        entry = entries[0]
        self.fileList.set(store, "File", entry.file, entry.path)
        session.file_name = entry.file if mode == "file" else None
        session.file_path = entry.path if mode == "path" else None
        session.file_path_select = mode
        return True

    def _pick_debug_mode(self: QueryPartsContext) -> str:
        return pick_value(
            inquirer,
            message="Choose comparison mode:",
            choices=[
                Choice(
                    name="ge — show messages at or below the selected level",
                    value="ge",
                ),
                Choice(
                    name="le — show messages at or above the selected level",
                    value="le",
                ),
                Choice(
                    name="is — show messages at the selected level only",
                    value="is",
                ),
            ],
            default=self.session.debug_mode or "ge",
        )

    def _ask_value(self: QueryPartsContext, tok: str):
        s = self.session
        spec = self.COMMAND_SPECS.get(tok, {})

        mode = spec.get("mode", "normal")

        if mode == "summary_length":
            choice = pick_value(
                inquirer,
                message="Choose summary length:",
                choices=list(ContextSummarizer.LENGTH_INSTRUCTIONS),
                default=s.summary_length or self.cfg.get_str("SUMMARY_LENGTH"),
            )
            self._read_and_apply_value(tok, str(choice), from_interactive=True)
            return

        if mode == "confidence_evaluation":
            choice = pick_value(
                inquirer,
                message="Choose confidence evaluation:",
                choices=list(QualityDimensionEvaluator.MODES),
                default=QualityDimensionEvaluator.resolve_mode(self.cfg, s),
            )
            self._read_and_apply_value(tok, str(choice), from_interactive=True)
            return

        if tok.startswith("enable_") and hasattr(s, tok):
            current_value = getattr(s, tok, None)
            default_enabled = True if current_value is None else bool(current_value)
            enabled = pick_value(
                inquirer,
                message=f"Set {tok}:",
                choices=[
                    Choice(name="true", value=True),
                    Choice(name="false", value=False),
                ],
                default=0 if default_enabled else 1,
            )
            setattr(s, tok, bool(enabled))
            if tok == "enable_rerank":
                s.rerank = bool(enabled)
            return

        if mode == "metadata":
            s.metadata_filters = self.metadataPicker.pick(
                s.collection_name, dict(s.metadata_filters or {})
            )
            self._print_metadata_filters()
            return

        if mode in ("file", "path"):
            self._pick_file(s, mode)
            return

        if mode == "strategy":
            choice = pick_value(
                inquirer,
                message="Choose strategy:",
                choices=self.allowed_strategies,
                default=s.strategy,
            )
            s.strategy = choice
            self._base_defaults(choice)
            self._warn_strategy_context()
            return

        if mode == "orchestrator_flow":
            if not self.allowed_orchestration_flows:
                print(
                    "⚠ No orchestration flows are configured. "
                    "Define _ORCHESTRATION_FLOWS in Config_Orchestrator.py first."
                )
                return
            choice = pick_value(
                inquirer,
                message="Choose orchestration flow:",
                choices=self.allowed_orchestration_flows,
                default=s.orchestrator_flow or self._default_orchestration_flow(),
            )
            self.applyOrchestrationFlowDefaults(choice)
            return

        if tok == "force_retrieve_mode":
            allowed_modes: list[str] = [
                str(item).strip().upper()
                for item in self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                if str(item).strip()
            ]
            if not allowed_modes:
                print("⚠ No retrieve modes configured in _ALLOWED_RETRIEVE_MODES")
                return
            current_mode = (
                str(getattr(s, "force_retrieve_mode", "") or "").strip().upper()
            )
            choice = pick_value(
                inquirer,
                message="Choose force_retrieve_mode:",
                choices=allowed_modes,
                default=(
                    current_mode if current_mode in allowed_modes else allowed_modes[0]
                ),
            )
            self._read_and_apply_value(tok, str(choice), from_interactive=True)
            return

        if tok in (
            "main_query_source",
            "secondary_query_source",
            "rerank_query_source",
        ):
            allowed_sources: list[str] = FlowConfig.allowed_query_sources(self.cfg)
            current_source = str(getattr(s, tok, "") or "").strip().upper()
            choice = pick_value(
                inquirer,
                message=f"Choose {tok}:",
                choices=allowed_sources,
                default=(
                    current_source
                    if current_source in allowed_sources
                    else allowed_sources[0]
                ),
            )
            self._read_and_apply_value(tok, str(choice), from_interactive=True)
            return

        if tok == "retrieval_language":
            choices = [mode.lower() for mode in FlowConfig.RETRIEVAL_LANGUAGE_MODES]
            current = str(getattr(s, "retrieval_language_override", "") or "").lower()
            choice = pick_value(
                inquirer,
                message="Choose retrieval_language:",
                choices=choices,
                default=current if current in choices else choices[0],
            )
            self._read_and_apply_value(tok, str(choice), from_interactive=True)
            return

        if mode == "debug":
            choices = [
                Choice(name=label, value=value)
                for label, value in self.allowed_debug_levels.items()
            ]
            choice_value = pick_value(
                inquirer,
                message="Choose debug level:",
                choices=choices,
                default=s.debug_level,
            )
            mode_choice = self._pick_debug_mode()
            self._set_debug(choice_value, mode_choice)
            return

        if mode == "debug_mode":
            mode_choice = self._pick_debug_mode()
            self._set_debug(s.debug_level or 0, mode_choice)
            return

        if mode == "preferred_response_language":
            # Build the language list from ACTIVE_LANGUAGES so all language-aware
            # modules share the same activation source of truth.
            lang_map: dict[str, str] = get_lang_code_to_name(self.cfg)
            iso_codes: set[str] = set(get_active_language_codes(self.cfg))
            names: list[str] = sorted(
                {lang_map.get(code, code).lower() for code in iso_codes}
            )
            if "english" not in names:
                names.insert(0, "english")
            choice = pick_value(
                inquirer,
                message="Choose your preferred response language:",
                choices=names,
                default=s.preferred_response_language or "english",
            )
            s.preferred_response_language = choice
            return

        if tok == "collection":
            collection = self.collectionPicker.pick_collection()
            if collection is None:
                return
            if s.collection_name:
                self.hist.save(
                    HistoryManager.key(s.collection_name, s.chat_name), "Settings"
                )
            s.collection_name = collection
            self.ragChatImpl.set_vector_store(s)
            self.reset_things()
            return

        if mode == "chat_name":
            existing = self._list_chat_names(s.collection_name or "")
            if existing:
                choices: list[Any] = [Choice(name=n, value=n) for n in existing]
                choices.append(Choice(name="<new chat name>", value="__new__"))
                picked = pick_value(
                    inquirer,
                    message="Pick a chat name (or choose <new chat name>):",
                    choices=choices,
                    default=s.chat_name if s.chat_name in existing else None,
                )
            else:
                picked = "__new__"
            if picked == "__new__":
                picked = (
                    inquirer.text(
                        message="Enter new chat name> ",
                    )
                    .execute()
                    .strip()
                )
                if not picked:
                    return
            self._read_and_apply_value(tok, picked, from_interactive=True)
            return

        if mode == "help":
            self.help()
            return

        if tok == "terminal_line_size":
            current = self._resolve_terminal_line_size()

            def _tls_validator(text: str) -> bool:
                if not text.strip():
                    return True
                try:
                    v = int(text)
                    if v < 40:
                        raise ValidationError(
                            message="terminal_line_size must be at least 40."
                        )
                    return True
                except ValueError:
                    raise ValidationError(
                        message="terminal_line_size must be an integer."
                    )

            val_str = (
                inquirer.text(
                    message=f"Enter terminal_line_size (current={current}, empty to keep)> ",
                    validate=_tls_validator,
                )
                .execute()
                .strip()
            )
            if not val_str:
                return
            self.cfg.set("TERMINAL_LINE_SIZE", int(val_str), force=True)
            return

        if mode == "web_search":
            _ws_current = (
                "web_only"
                if getattr(s, "retrieve_mode", None) == "WEB"
                else ("local_and_web" if s.web_search else "local_only")
            )
            _ws_choices = [
                Choice(name="local_only    — local indexes only", value="local_only"),
                Choice(
                    name="local_and_web — web + local retrieval", value="local_and_web"
                ),
                Choice(
                    name="web_only      — web only, skip local indexes",
                    value="web_only",
                ),
            ]
            ws_choice = pick_value(
                inquirer,
                message="Choose web search mode:",
                choices=_ws_choices,
                default=_ws_current,
            )
            self._read_and_apply_value(tok, ws_choice, from_interactive=True)
            return

        if mode == "fetch_page_content":
            _allowed_fpc = ["snippets only", "fetch pages"]
            choice = pick_value(
                inquirer,
                message="Choose web result mode:",
                choices=_allowed_fpc,
                default="fetch pages" if s.fetch_page_content else "snippets only",
            )
            self._read_and_apply_value(tok, choice, from_interactive=True)
            return

        if tok in self._ASK_DIRECT_VALUE_TOKENS:
            spec = self.COMMAND_SPECS.get(tok, {})
            expected_type = spec.get("type", "str")
            attr = spec.get("attr")
            current = getattr(s, attr, None) if attr else None

            # For override attributes, show override vs auto info
            if tok == "max_output_tokens":
                override = s.max_output_tokens_override
                if override is not None:
                    prompt = f"Enter {tok} (override={override}, last_used={s.max_output_tokens})> "
                else:
                    prompt = f"Enter {tok} (current={s.max_output_tokens}, dynamic)> "
            elif tok == "context_size":
                override = s.context_size_override
                auto = self.tokenBudget.get_context_limit()
                if override is not None:
                    prompt = f"Enter {tok} (override={override}, auto={auto})> "
                else:
                    prompt = f"Enter {tok} (current={auto}, auto)> "
            else:
                prompt = f"Enter {tok} (current={current})> "

            def validator(text: str) -> bool:
                if not text.strip():
                    return True
                try:
                    if tok in ("summary_temperature", "chunk_overlap"):
                        ContextSummarizer.validate_setting(tok, text)
                    self._validate_raw_for_type(text, expected_type)
                    return True
                except ValueError as e:
                    raise ValidationError(message=str(e))

            val = (
                inquirer.text(
                    message=prompt,
                    instruction="Enter an empty value to return without changing the setting",
                    validate=validator,
                )
                .execute()
                .strip()
            )
            if not val:
                return
            # apply via helper
            self._read_and_apply_value(tok, val, from_interactive=True)
            return


class QueryParts(
    QuerySettingsPresenter, QuerySettingsPicker, SessionSettingsApplier, SingletonMixin
):

    COMMAND_SPECS = _COMMAND_SPECS
    _FLOW_SWITCH_TO_SESSION_ATTR = _FLOW_SWITCH_TO_SESSION_ATTR
    _FRIENDLY_TOKEN_MAP = _FRIENDLY_TOKEN_MAP
    _PARSER_ONLY_TOKENS = _PARSER_ONLY_TOKENS
    EXTRA_TOKENS = EXTRA_TOKENS
    _ASK_DIRECT_VALUE_TOKENS = _ASK_DIRECT_VALUE_TOKENS
    _SET_DIRECT_VALUE_TOKENS = _SET_DIRECT_VALUE_TOKENS
    _SETTINGS_LAYOUT = _SETTINGS_LAYOUT

    def __new__(cls, *args: Any, **kwargs: Any) -> "QueryParts":
        return SingletonMixin.__new__(cls)

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return

        self._initialized = True

        # Build token list for regexes
        tokens: list[str] = (
            list(self.COMMAND_SPECS.keys())
            + self.EXTRA_TOKENS
            + list(self._FRIENDLY_TOKEN_MAP.keys())
            + self._PARSER_ONLY_TOKENS
        )
        token_alt: str = "|".join(tokens)

        # Keep original behavior, including the old "treshold" typo in CMD_RX
        self.split_regex: re.Pattern[str] = re.compile(
            rf"(?=\b({token_alt}|enable_[a-z_]+)\b\s*[\?\!\=\-\*])",
            re.IGNORECASE,
        )
        self.compiled_regex: re.Pattern[str] = re.compile(
            rf"^\s*\b({token_alt}|enable_[a-z_]+)\b\s*([\?\!\=\-\*])\s*(.*)$",
            re.IGNORECASE,
        )

        self.hist: HistoryManager = HistoryManager()
        self.collectionPicker: CollectionPicker = CollectionPicker()
        self.session: Session = Session()
        self.ragChatImpl: RAGChatImpl = RAGChatImpl()
        self.fileList: FileList = FileList()
        self.metadataPicker: MetadataPicker = MetadataPicker()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.tokenBudget: TokenBudget = TokenBudget()

        # load allowed strategies
        self.allowed_strategies: list[Any] = self.cfg.get_list("_ALLOWED_STRATEGIES")
        # Fail fast on misspelled flow keys, invalid values or unknown flow names.
        FlowConfig.validate_flow_config(self.cfg)
        self.allowed_orchestration_flows: list[str] = FlowConfig.allowed_flow_names(
            self.cfg
        )
        self.allowed_debug_levels: dict[str, Any] = self.cfg.get_dict(
            "_ALLOWED_DEBUG_LEVELS"
        )

        # derive prompt_map from COMMAND_SPECS for help/printing
        self.prompt_map: dict[str, dict[str, str]] = {
            name: {"prompt": spec["prompt"], "section": spec["section"]}
            for name, spec in self.COMMAND_SPECS.items()
            if "prompt" in spec and "section" in spec
        }

        # initial defaults
        self._defaults(
            "strategy", self.cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY"), True
        )
        self.applyOrchestrationFlowDefaults(
            self._default_orchestration_flow(),
            session=self.session,
        )

    def _default_orchestration_flow(self) -> str:
        return FlowConfig.default_flow_name(
            self.cfg, list(self.allowed_orchestration_flows) or None
        )

    # ——— Utility: type casting ———

    # ——— Printing current values ———

    # Shared display hierarchy for startup values, show? and help?.
    _SETTINGS_LAYOUT: dict[str, dict[str, tuple[str, ...]]] = {
        "Chat settings": {
            "Debug": ("debug_level", "debug_mode"),
            "Chat Context": (
                "use_chat_context",
                "history_keep",
                "history_prune",
                "rewrite_context",
                "topic_summary",
            ),
            "Talk with": ("collection", "chat_name"),
            "File Input": ("file", "path", "metadata", "file_cap"),
            "Visual": ("mark_text",),
            "Response": ("preferred_response_language",),
            "Commands": ("help",),
        },
        "Summarize settings": {
            "Summary": ("summary_length", "summary_temperature", "chunk_overlap"),
        },
        "Retrieval settings": {
            "Retrieval strategy": ("strategy",),
            "Strategy overrides": (),
            "LLM": ("temperature", "top_p", "top_k"),
            "Retriever tuning": ("fetch_k", "context_chunks", "threshold"),
            "Retriever weights": (
                "vector_weight",
                "bm25_weight",
                "graph_weight",
                "regex_weight",
                "web_weight",
            ),
            "Context/session": (
                "file_cap",
                "use_chat_context",
                "history_keep",
                "history_prune",
                "rewrite_context",
                "topic_summary",
            ),
            "Output": ("max_output_tokens", "context_size", "terminal_line_size"),
            "API passthrough": ("extra_ollama_options", "ollama_top_level_params"),
            "Web": (
                "web_search",
                "fetch_page_content",
                "web_fetch_k",
                "bm25_pre_filter",
                "cosine_pre_filter",
                "web_rerank_threshold",
            ),
        },
        "Orchestration settings": {
            "Orchestration strategy": ("orchestrator_flow",),
            "Orchestrator overrides": (),
            "Confidence": ("confidence_evaluation",),
            "Flow selectors": (
                "force_retrieve_mode",
                "main_query_source",
                "secondary_query_source",
                "rerank_query_source",
                "retrieval_language",
                "rerank",
            ),
            "Flow stages": (
                "secondary_query",
                "per_language_indexed_query",
                "indexed_translate",
                "indexed_wordnet",
                "orig_lang_vector",
                "vector_alternates",
            ),
            "Flow retrievers": ("vector", "bm25", "graph", "regex"),
            "Flow runtime": (
                "query_rewrite",
                "keyword_action_splitter",
                "pronoun_subst",
                "rerank_stage",
                "low_score_fallback",
                "low_recall_rescue",
                "grounding",
            ),
        },
    }

    def _print_settings(
        self, content: dict[str, Any], *, descriptions: bool = False
    ) -> None:
        print(self._format_settings(content, descriptions=descriptions))

    def print_values(self, session: Session | None = None) -> None:
        self._print_settings(self.session_values(session))
        print()

    def format_session_snapshot(self, session: Session | None = None) -> str:
        """Format user-facing settings only, using the same layout as the CLI."""
        text = self._format_settings(self.session_values(session))
        plain = re.sub(r"\x1b\[[0-9;]*m", "", text).strip()
        return (
            "\n\n---\n**Session settings (invalid request was not applied):**\n"
            f"```text\n{plain}\n```"
        )

    # ——— Asking for values interactively ———

    def _ask(self, tok: str):
        try:
            self._ask_value(tok)
        except PickerCancelled:
            return

    def pick_path(self, session: Session) -> bool:
        """Open the same path picker used by path!, applying only a selection."""
        return self._pick_file(session, "path")

    # ——— Chat name discovery ———

    def _list_chat_names(self, collection_name: str) -> list[str]:
        """Return sorted unique chat names found in the history directory for the given collection.

        History is shared across embedders, so files are matched by the base
        collection name (Test_snowflake -> Test_<chat>_Query.txt).
        """
        history_dir = self.hist.path
        if not collection_name or not os.path.isdir(history_dir):
            return []
        prefix = HistoryManager.key(collection_name, "")
        suffixes = ("_Query.txt", "_Settings.txt")
        names: set[str] = set()
        for fname in os.listdir(history_dir):
            if not fname.startswith(prefix):
                continue
            for sfx in suffixes:
                if fname.endswith(sfx):
                    chat = fname[len(prefix) : -len(sfx)]
                    if chat:
                        names.add(chat)
                    break
        return sorted(names)

    def _normalize_token(self, token: str) -> str:
        """Map friendly aliases to canonical command tokens."""
        key = str(token or "").strip().lower()
        return self._FRIENDLY_TOKEN_MAP.get(key, key)

    # ——— Parsing and applying commands ———

    def handle(self, raw: str) -> List[Cmd]:
        parts = [s.strip() for s in self.split_regex.split(raw) if s.strip()]
        out: List[Cmd] = []
        for seg in parts:
            m = self.compiled_regex.match(seg)
            if not m:
                continue
            tok, op, pay = m.group(1).lower(), m.group(2), m.group(3).strip()
            tok = self._normalize_token(tok)
            pay = pay.strip("'\"")
            out.append(Cmd(tok, op, pay))
        return out

    def _apply(self, cmd: Cmd):
        if cmd.op == "?":
            if cmd.token == "show":
                self.print_values()
                return
            return self._show(cmd.token)
        if cmd.op == "=":
            return self._set(cmd.token, cmd.payload)
        if cmd.op == "!":
            return self._ask(cmd.token)
        if cmd.op == "-":
            return self._clear(cmd.token)
        if cmd.op == "*":
            return self._defaults(cmd.token, cmd.payload)

    # ——— Showing current values ———

    def _show(self, tok: str):
        if tok == "help":
            self.help()
            return

        s = self.session
        if tok in ("summary_length", "summary_temperature", "chunk_overlap"):
            value = getattr(s, tok)
            if value is None:
                if tok == "summary_length":
                    value = self.cfg.get_str("SUMMARY_LENGTH")
                elif tok == "summary_temperature":
                    value = self.cfg.get_float("SUMMARY_TEMPERATURE")
                else:
                    value = self.cfg.get_int("CHUNK_OVERLAP")
            print(f"[{tok}] {value}")
            return
        if tok == "file":
            fn, fp = self.fileList.get_current()
            print(f"[file] File: {fn} Path: {fp}")
            return
        if tok == "path":
            _, fp = self.fileList.get_current()
            print(f"[path] Path: {fp}")
            return
        if tok == "metadata":
            self._print_metadata_filters()
            return

        if tok.startswith("enable_") and hasattr(s, tok):
            print(f"[{tok}] {getattr(s, tok)}")
            return

        handlers = self._build_show_handlers()
        handler = handlers.get(tok)
        if handler is not None:
            handler()
            return

        print(f"⚠ Invalid command: {tok}")
        self.help()

    # ——— Resetting on collection change ———

    def _print_metadata_filters(self) -> None:
        mf = self.session.metadata_filters or {}
        if not mf:
            print("[metadata] (none)")
        else:
            pairs = "  ".join(f"{k}={v}" for k, v in mf.items())
            print(f"[metadata] {pairs}")

    def reset_things(self):
        if self.session.file_name or self.session.file_path:
            print("⚠ File / path filter cleared")
            self.session.file_name, self.session.file_path = None, None
            self.session.file_path_select = None
        # Drop the FileList pointer too, otherwise `file`/`path` would still
        # report the previous collection's selection after a switch.
        self.fileList.clear_current()
        if self.session.metadata_filters:
            print("⚠ Metadata filters cleared")
            self.session.metadata_filters = {}
        self.hist.load(
            HistoryManager.key(self.session.collection_name, self.session.chat_name),
            "Query",
        )
        self.hist.load(
            HistoryManager.key(self.session.collection_name, self.session.chat_name),
            "Settings",
        )

    # ——— Setting values via "=" ———

    # ——— Clearing values ———

    # ——— Defaults loader for strategies ———

    def _defaults(self, tok: str, p: str, init_once: bool = True):
        try:
            self._pick_defaults(tok, p, init_once)
        except PickerCancelled:
            return

    def _pick_defaults(self, tok: str, p: str, init_once: bool = True):
        if tok == "strategy":
            strategy_input = str(p or "").strip().upper()
            if not strategy_input:
                if not self.allowed_strategies:
                    print("⚠ No strategies are configured in _ALLOWED_STRATEGIES")
                    return
                current_strategy = (
                    str(
                        self.session.strategy
                        or self.cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY")
                        or self.allowed_strategies[0]
                    )
                    .strip()
                    .upper()
                )
                strategy_input = (
                    str(
                        pick_value(
                            inquirer,
                            message="Choose strategy defaults:",
                            choices=self.allowed_strategies,
                            default=(
                                current_strategy
                                if current_strategy in self.allowed_strategies
                                else self.allowed_strategies[0]
                            ),
                        )
                    )
                    .strip()
                    .upper()
                )

            if strategy_input not in self.allowed_strategies:
                print(
                    f"⚠ Invalid value {strategy_input}. Defaults can be set only for strategies. Allowed: {self.allowed_strategies}"
                )
                return

            self.session.strategy = strategy_input
            cn = self._base_defaults(strategy_input)
            self._warn_strategy_context()

            if self.session.collection_name and self.session.chat_name:
                self.hist.save(
                    HistoryManager.key(
                        self.session.collection_name, self.session.chat_name
                    ),
                    "Settings",
                )
            self.session.collection_name = cn
            self.reset_things()

            if init_once:
                print(f"→ Loaded defaults for '{strategy_input}':")
                self.print_values()
        elif tok == "orchestrator_flow":
            flow_input = FlowConfig.resolve_flow_alias(p, self.cfg)
            if not flow_input:
                if not self.allowed_orchestration_flows:
                    print(
                        "⚠ No orchestration flows are configured. "
                        "Define _ORCHESTRATION_FLOWS in Config_Orchestrator.py first."
                    )
                    return
                current_flow = (
                    str(
                        self.session.orchestrator_flow
                        or self._default_orchestration_flow()
                    )
                    .strip()
                    .upper()
                )
                flow_input = (
                    str(
                        pick_value(
                            inquirer,
                            message="Choose orchestration flow defaults:",
                            choices=self.allowed_orchestration_flows,
                            default=(
                                current_flow
                                if current_flow in self.allowed_orchestration_flows
                                else self.allowed_orchestration_flows[0]
                            ),
                        )
                    )
                    .strip()
                    .upper()
                )

            if (
                self.allowed_orchestration_flows
                and flow_input not in self.allowed_orchestration_flows
            ):
                print(
                    f"⚠ Invalid value {flow_input}. "
                    "Defaults can be set only for configured orchestration flows. "
                    f"Allowed: {self.allowed_orchestration_flows}"
                )
                return

            resolved_flow = self.applyOrchestrationFlowDefaults(
                flow_input,
                session=self.session,
            )
            print(f"→ Loaded flow defaults for '{resolved_flow}':")
            self.print_values()
        else:
            print(
                "⚠ Default (*) only allowed for strategy or orchestrator_flow. "
                "Use e.g. strategy*default or orchestrator_flow*ENGLISH_ALL"
            )
        return

    # ——— Help ———

    def command_help(self):
        print(
            "\n  ▶ Usage\n"
            "    ▶ help? explains commands; show? displays grouped current values.\n"
            "    ▶ key? shows a value; key=value sets it; key! opens a picker; key- clears it.\n"
            "    ▶ strategy! / orchestrator_flow! open inline pickers.\n"
            "    ▶ strategy* / orchestrator_flow* list defaults; "
            "strategy*name / orchestrator_flow*FLOW apply defaults.\n"
            "    ▶ file! / path! open history pickers.\n"
            "    ▶ Quick defaults: narrow*, balanced_file_cap*, default*, wide*.\n"
            "    ▶ Flow overrides accept ?, =, ! and - (restore baseline where applicable).\n"
            "    ▶ Press Enter on an empty settings line to proceed to the query prompt."
        )

    def help(self):
        # Keep explanations in the command registry, but present them in the
        # same hierarchy as print_values() rather than alphabetic sections.
        flow_prompts = {
            "secondary_query": "Run a second local retrieval leg with the secondary query.",
            "per_language_indexed_query": "Shape BM25/Graph/Regex queries per indexed language.",
            "indexed_translate": "Allow translation during indexed query shaping.",
            "indexed_wordnet": "Append WordNet synonyms to indexed queries.",
            "orig_lang_vector": "Run the original-language vector retrieval leg.",
            "vector_alternates": "Enable MultiQuery alternate fanout for vector retrieval.",
            "vector": "Enable the vector local retriever leg.",
            "bm25": "Enable the BM25 local retriever leg.",
            "graph": "Enable the graph local retriever leg.",
            "regex": "Enable the regex local retriever leg.",
            "query_rewrite": "Enable query rewriting for this flow.",
            "keyword_action_splitter": "Split local retrieval needs/keywords from answer operations; web and history stay unchanged.",
            "pronoun_subst": "Enable pronoun referent substitution during rewriting.",
            "rerank_stage": "Set baseline reranking; the session rerank switch can override it per turn.",
            "low_score_fallback": "Allow low-score reranking fallback to retrieval order.",
            "low_recall_rescue": "Rescue query-overlap local chunks when hits are sparse; reserve the best retrieval-ranked table if no table survived.",
            "grounding": "Enable answer grounding and visual marking.",
        }
        for key, meta in self.prompt_map.items():
            if "section" not in meta:
                raise ValueError(f"Missing 'section' for key: {key}")

        prompts = dict(flow_prompts)
        prompts.update({key: meta["prompt"] for key, meta in self.prompt_map.items()})
        self._print_settings(prompts, descriptions=True)
        self.command_help()

    # ——— Bool helpers ———
