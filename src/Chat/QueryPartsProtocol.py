"""Structural contract shared by QueryParts responsibility mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from AI.TokenBudget import TokenBudget
from Config.Config import Config
from Globals.Session import Session
from Gui.CollectionPicker import CollectionPicker
from Gui.FileList import FileList
from Gui.HistoryManager import HistoryManager
from Gui.MetadataPicker import MetadataPicker
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers

if TYPE_CHECKING:
    from Chat.RAGChatImpl import RAGChatImpl


class QueryPartsContext(Protocol):
    cfg: Config
    helpers: Helpers
    hist: HistoryManager
    collectionPicker: CollectionPicker
    fileList: FileList
    metadataPicker: MetadataPicker
    pretty: PrettyWriter
    ragChatImpl: RAGChatImpl
    session: Session
    tokenBudget: TokenBudget
    allowed_strategies: list[Any]
    allowed_orchestration_flows: list[str]
    allowed_debug_levels: dict[str, Any]
    prompt_map: dict[str, dict[str, str]]
    COMMAND_SPECS: dict[str, dict[str, Any]]
    _SETTINGS_LAYOUT: dict[str, dict[str, tuple[str, ...]]]
    _FRIENDLY_TOKEN_MAP: dict[str, str]
    _ASK_DIRECT_VALUE_TOKENS: set[str]
    _SET_DIRECT_VALUE_TOKENS: set[str]
    _FLOW_SWITCH_TO_SESSION_ATTR: dict[str, str]
    EXTRA_TOKENS: list[str]
    _PARSER_ONLY_TOKENS: list[str]

    def _cast_value(self, tok: str, raw: str) -> Any: ...

    def _get_attr_name(self, tok: str) -> str | None: ...

    def _read_and_apply_value(
        self, tok: str, raw: str, *, from_interactive: bool = False
    ) -> None: ...

    def _validate_raw_for_type(self, raw: str, type_name: str) -> None: ...

    def to_bool(self, value: int | str | bool) -> bool: ...

    def _set_debug(self, level: int, mode: str) -> None: ...

    def _base_defaults(
        self, strategy: str, *, session: Session | None = None
    ) -> str: ...

    def strategy_context_warning(self, session: Session) -> str | None: ...

    def _warn_strategy_context(self) -> None: ...

    def _default_orchestration_flow(self) -> str: ...

    def _resolve_flow_profile_defaults(self, flow_name: str) -> dict[str, Any]: ...

    def _apply_flow_switch_defaults(
        self, session: Session, profile_raw: dict[str, Any]
    ) -> None: ...

    def applyOrchestrationFlowDefaults(
        self, orchestrator_flow: str, session: Session | None = None
    ) -> str: ...

    def reset_things(self) -> None: ...

    def _list_chat_names(self, collection_name: str) -> list[str]: ...

    def _print_metadata_filters(self) -> None: ...

    def help(self) -> None: ...

    def _pick_file(self, session: Session, mode: str) -> bool: ...

    def _pick_debug_mode(self) -> str: ...

    def _resolve_terminal_line_size(self, session: Session | None = None) -> int: ...
