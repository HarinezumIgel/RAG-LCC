# Local module imports
import os
from typing import Any

from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import BOLD, MAGENTA, ORANGE, RED, RESET, TURQUOISE, YELLOW
from Gui.HistoryManager import HistoryManager
from Helpers.ChromaDBHelper import ChromaDBHelper


# ——— your processor, now no longer requires constructor args ———
class CommandProcessor:
    def __init__(self) -> None:
        # now initialize your own helpers
        self.ragChatImpl: RAGChatImpl = RAGChatImpl()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.queryParts: QueryParts = QueryParts()
        self.cfg: Config = Config()
        self.session: Session = self.queryParts.session

        # load allowed strategies
        self.allowed_strategies: list[Any] = self.cfg.get_list("_ALLOWED_STRATEGIES")
        self.hist: HistoryManager = HistoryManager()
        _raw_size: Any = self.cfg.get("TERMINAL_LINE_SIZE")
        if isinstance(_raw_size, dict):
            from typing import cast as _cast

            from Helpers.DebugHelper import DebugHelper as _DH

            _size_dict: dict[str, Any] = _cast(dict[str, Any], _raw_size)
            _key = "debug" if _DH.level(self.cfg) > 0 else "no_debug"
            self.terminal_line_size: int = int(
                _size_dict.get(_key, _size_dict.get("debug", 160))
            )
        else:
            self.terminal_line_size: int = (
                int(_raw_size) if _raw_size is not None else 160
            )
        self._first_configure: bool = True

    def configure_and_query(self) -> Session:
        if self._first_configure:
            self.queryParts.print_values()
            self._first_configure = False
        print(
            f"{MAGENTA}help? for help   show? for current values\n"
            f"key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   "
            f"key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)\n"
            f"Press ↵ on an empty line to proceed to your query prompt{RESET}"
        )

        # ——————————————
        # 1) SETTINGS loop
        # ——————————————
        self.hist.load(
            HistoryManager.key(self.session.collection_name, self.session.chat_name),
            "Settings",
        )
        while True:
            raw = self.hist.prompt(" 🛠️  >").strip()
            if not raw:
                break

            cmds = self.queryParts.handle(raw)
            if not cmds:
                print(
                    f"{ORANGE}    That was not a valid command. help? or show? or ↵ to enter query.\n{RESET}"
                )
            else:
                for c in cmds:
                    self.queryParts.apply(c)
            self.hist.save(
                HistoryManager.key(
                    self.session.collection_name, self.session.chat_name
                ),
                "Settings",
            )
        # ——————————————
        # 2) FINAL QUERY
        # ——————————————
        self.hist.load(
            HistoryManager.key(self.session.collection_name, self.session.chat_name),
            "Query",
        )
        rewrite_hint = (
            f"type {BOLD}new: your question{RESET}{ORANGE} to start a new topic  · "
            if self.session.use_chat_context
            else ""
        )
        print(
            f"{ORANGE}{rewrite_hint}type {BOLD}summarize: [file name or path]{RESET}{ORANGE} "
            f"(alias: summary:) to summarize a file (no argument: pinned document; "
            f"no pin: open the path picker)"
            f"  · b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat{RESET}"
        )
        web_mode = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        web_active = self.session.web_search and web_mode == "1"
        web_only = web_active and getattr(self.session, "retrieve_mode", None) == "WEB"
        web_prefix = (
            f"{RED}{BOLD}🌐 Internet search is ON — web only, local indexes skipped.{RESET}  "
            if web_only
            else (
                f"{RED}{BOLD}🌐 Internet search is ON — web + local indexes.{RESET}  "
                if web_active
                else ""
            )
        )
        # File/path filter and metadata filter are independent notices.
        file_prefix = ""
        if self.session.file_name or self.session.file_path:
            pinned = self.session.file_path or self.session.file_name
            file_prefix = (
                f"{TURQUOISE}{BOLD}🔍 Pinned document (chat and summarize): "
                f"{pinned}{RESET}  "
            )
        meta_prefix = ""
        if self.session.metadata_filters:
            pairs = "  ".join(
                f"{k}={v}" for k, v in self.session.metadata_filters.items()
            )
            meta_prefix = f"{TURQUOISE}{BOLD}🔍 Metadata filter ON: {pairs}{RESET}  "
        # Notices (web / file / metadata) share one line; query prompt goes below.
        notices = f"{web_prefix}{file_prefix}{meta_prefix}".rstrip()
        notice_line = f"{notices}\n" if notices else ""
        query = self.hist.prompt(
            f"{notice_line}{YELLOW}{BOLD}💬 Your actual query>{RESET}  "
        ).strip()
        print()
        self.hist.save(
            HistoryManager.key(self.session.collection_name, self.session.chat_name),
            "Query",
        )

        if query.lower() in ("b", "b:"):
            return self.configure_and_query()

        self.session.query = query
        return self.session

    def help(self) -> None:
        self.queryParts.help()
