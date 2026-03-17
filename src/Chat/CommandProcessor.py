# Local module imports
from typing import Any

from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import BOLD, CYAN, MAGENTA, ORANGE, RESET, YELLOW
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
        self.allowed_strategies: list[Any] = self.cfg.get_list(
            "_ALLOWED_STRATEGIES"
        ) or [
            "wide",
            "medium",
            "narrow",
        ]
        self.hist: HistoryManager = HistoryManager()
        self.terminal_line_size: int = self.cfg.get_int("TERMINAL_LINE_SIZE")

    def configure_and_query(self) -> Session:
        self.queryParts.print_values()
        print(
            f"{MAGENTA}help? for help\nPress ↵ on an empty line to proceed to your query prompt{RESET}"
        )

        # ——————————————
        # 1) SETTINGS loop
        # ——————————————
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Settings"
        )
        while True:
            print(" 🛠️  >", end="")
            raw = input().strip()
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
                self.queryParts.print_values()
            self.hist.save(
                f"{self.session.collection_name}_{self.session.chat_name}", "Settings"
            )
        # ——————————————
        # 2) FINAL QUERY
        # ——————————————
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Query"
        )
        print(
            f"{ORANGE}b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat{RESET}\n"
            f"{CYAN}{BOLD}{"+" * self.terminal_line_size}{RESET}\n",
            f"{YELLOW}{BOLD}💬 Your actual query>{RESET}  ",
            end="",
        )
        query = input().strip()
        print(f"{CYAN}{BOLD}{"+" * self.terminal_line_size}{RESET}\n")
        self.hist.save(
            f"{self.session.collection_name}_{self.session.chat_name}", "Query"
        )

        if query.lower() == "b":
            return self.configure_and_query()

        self.session.query = query
        return self.session

    def help(self) -> None:
        self.queryParts.help()
