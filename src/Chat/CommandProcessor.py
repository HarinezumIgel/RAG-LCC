# Local module imports
import os
from typing import Any

from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import BOLD, MAGENTA, ORANGE, RED, RESET, YELLOW
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
        self.session: Session = Session()

        # load allowed strategies
        self.allowed_strategies: list[Any] = self.cfg.get_list("_ALLOWED_STRATEGIES")
        self.hist: HistoryManager = HistoryManager()
        _raw_size: Any = self.cfg.get("TERMINAL_LINE_SIZE")
        if isinstance(_raw_size, dict):
            from Helpers.DebugHelper import DebugHelper as _DH

            _key = "debug" if _DH.level(self.cfg) > 0 else "no_debug"
            self.terminal_line_size: int = int(
                _raw_size.get(_key, _raw_size.get("debug", 160))
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
            f"{MAGENTA}help? for help   show? for current values\nPress ↵ on an empty line to proceed to your query prompt{RESET}"
        )

        # ——————————————
        # 1) SETTINGS loop
        # ——————————————
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Settings"
        )
        while True:
            raw = self.hist.prompt(" 🛠️  >").strip()
            if not raw:
                break

            cmds = self.queryParts.handle(raw)
            if not cmds:
                print(
                    f"{ORANGE}    That was not a valid command. help? or ↵ to enter query.\n{RESET}"
                )
            else:
                for c in cmds:
                    self.queryParts._apply(c)  # type: ignore[reportPrivateUsage]
            self.hist.save(
                f"{self.session.collection_name}_{self.session.chat_name}", "Settings"
            )
        # ——————————————
        # 2) FINAL QUERY
        # ——————————————
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Query"
        )
        rewrite_hint = (
            f"  · type {BOLD}new: your question{RESET}{ORANGE} to start a new topic"
            if self.session.use_chat_context
            else ""
        )
        print(
            f"{ORANGE}b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat{rewrite_hint}{RESET}"
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
        query = self.hist.prompt(
            f"{web_prefix}{YELLOW}{BOLD}💬 Your actual query>{RESET}  "
        ).strip()
        print()
        self.hist.save(
            f"{self.session.collection_name}_{self.session.chat_name}", "Query"
        )

        if query.lower() == "b":
            return self.configure_and_query()

        self.session.query = query
        return self.session

    def help(self) -> None:
        self.queryParts.help()
