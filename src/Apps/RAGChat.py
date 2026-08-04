#!/usr/bin/env python3
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # side-effect import

if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1":
    from Commons.NetworkTracer import NetworkTracer

    NetworkTracer.enable_tracer()  # Patch socket to trace network calls
else:
    NetworkTracer = None  # type: ignore[assignment,misc]

from Commons.StartupCommons import StartupCommons

ctx = StartupCommons.common_start("RAGChat", "RAGChat – all Constants become CLI flags")
cfg = ctx.cfg  # typed as Config args = ctx.args

import re
from pprint import pprint
from typing import Any, Dict

import colorama

from AI.AIHelpers import AIHelpers
from AI.LLMCaller import LLMCaller
from AI.ModelOutputAdapter import ModelOutputAdapter
from AI.TokenBudget import TokenBudget
from Chat.Chatter import Chatter
from Chat.CommandProcessor import CommandProcessor
from Chat.QueryParts import QueryParts
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Compliance.HFDownloader import HFDownloader
from Config.Config import Config
from Globals.Globals import Globals
from Globals.Session import Session
from Gui.Colors import CYAN, GREEN, RESET, TURQUOISE
from Gui.Informer import Informer
from Gui.PrettyWriter import PrettyWriter
from Helpers.CSVWriter import CSVWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers

colorama.init()


class RAGChat:
    def __init__(self, cfg: Config) -> None:
        # your core components
        self.globalsInstance: Globals = Globals()
        self.cfg: Config = cfg
        self.hf_downloader: HFDownloader = HFDownloader()
        self.hf_downloader.download("_MODELS._EMBED")
        self.hf_downloader.download("_MODELS._CROSS")
        self.inf: Informer = Informer()
        self.helpers: Helpers = Helpers()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.fileUtils: FileUtils = FileUtils()

        self.logger: Any = self.helpers.setup_logger()
        self.globalsInstance.set_logger(self.logger)
        self.pretty: PrettyWriter = PrettyWriter()
        self.queryParts: QueryParts = QueryParts()
        self.csvWriter: CSVWriter = CSVWriter()
        self.bannedPhraseCollector: BannedPhraseCollector = BannedPhraseCollector()
        self.llmCaller: LLMCaller = LLMCaller()
        self.modelOutputAdapter: ModelOutputAdapter = ModelOutputAdapter()

        self.helpers.check_cpu_and_bits()
        self.fileUtils.setDebug()

        self.session: Session = Session()
        self.chatter: Chatter = Chatter()
        self.tokenBudget: TokenBudget = TokenBudget()

        # load parameters once
        self.cp: CommandProcessor = CommandProcessor()

        # Share the same per-session state across CLI components
        self.queryParts.session = self.session
        self.cp.session = self.session

        # Apply strategy defaults so session fields are populated even if the
        # user skips straight to the query prompt without setting a strategy.
        default_strategy: str = (
            self.session.strategy
            or self.cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY")
            or "DEFAULT"
        )
        self.session.collection_name = self.queryParts.applyStrategyDefaults(
            default_strategy, session=self.session
        )

        self.chunk_size: int = self.helpers.get_chunker_max_size()
        self.use_ollama_gpu: bool = self.helpers.get_active_endpoint_args().get(
            "USE_GPU", True
        )

        self.prompt: str
        self.prompt_name: str | None
        prompt_var: str = self.helpers.get_model_args("_ACTIVE_LLM")["PROMPT_CHAT"]
        self.prompt, self.prompt_name = self.cfg.indirect_get(prompt_var)

    def _showIntro(self):
        # print config summary
        self.inf.inform()
        # clean and show prompt template
        self._show_clean_prompt()

    def _show_clean_prompt(self):
        if DebugHelper.check(self.cfg, 60):
            self.pretty.write("D", "Prompt template:", f"{self.prompt_name}")
            raw = self.prompt
            clean = " ".join(raw.replace("\\n", " ").split())
            pprint(clean)
            print("\n")

    def _process_query(self) -> bool:
        # get session and user query
        # self.queryParts.help()

        self.session = self.cp.configure_and_query()
        if not self.session.query:
            return False
        elif re.fullmatch(r"^b:$", self.session.query.strip()):
            return True

        if not self.session.query.strip():
            return False

        # build base kwargs
        base_kwargs: Dict[str, Any] = {"k": self.session.retriever_k}
        fp, fn = self.session.file_path, self.session.file_name

        if fp is not None or fn is not None:
            if self.session.file_path_select == "file":
                base_kwargs["filter"] = {"FileName": {"$eq": fn}}
                self.pretty.write(
                    "I",
                    "Modified query:",
                    f"Filtering on FileName: {fn}",
                    color=CYAN,
                )
            elif self.session.file_path_select == "path":
                base_kwargs["filter"] = {"FilePath": {"$eq": fp}}
                self.pretty.write(
                    "I",
                    "Modified query:",
                    f"Filtering on FilePath: {fp}",
                    color=CYAN,
                )
            elif "Chat Context" in (fp, fn):
                base_kwargs["filter"] = {"FilePath": {"$eq": "Chat Context"}}
                fp = fn = "Chat_Context"
                self.pretty.write(
                    "W",
                    "",
                    f"Using Chat Context as filter. Use file- or path- to reset it.",
                )
                self.pretty.write(
                    "W",
                    "",
                    f"Set threshold to 0 or below score returned for Chat Context. Otherwise you will not get results based on Chat Context only.",
                )
            else:
                self.pretty.write("E", "", f"Invalid path or filename: {fn}{fp}")
                return True

        # Merge interactive metadata filters as additional flat conditions.
        # Kept flat (multi-key) so BM25/graph AND them directly; the vector
        # path normalises to ChromaDB's $and form in RAGChatImpl._vector_kwargs.
        meta_filters = self.session.metadata_filters or {}
        if meta_filters:
            combined: Dict[str, Any] = dict(base_kwargs.get("filter", {}) or {})
            for key, value in meta_filters.items():
                combined[key] = {"$eq": value}
            base_kwargs["filter"] = combined
            self.pretty.write(
                "I",
                "Modified query:",
                "Filtering on metadata: "
                + "  ".join(f"{k}={v}" for k, v in meta_filters.items()),
                color=CYAN,
            )

        self.session.base_kwargs = base_kwargs
        # Show the effective retrieval filter (file and/or metadata) before the
        # query runs so the user sees exactly what retrieval is restricted to.
        active_filter = base_kwargs.get("filter")
        if active_filter:
            filter_desc = "  ".join(
                f"{k}={v.get('$eq') if isinstance(v, dict) else v}"
                for k, v in active_filter.items()
            )
            self.pretty.write("I", "Active filter", filter_desc, color=TURQUOISE)
        stage = "PROMPT_CHECK"
        # Check user provided prompt
        assert self.session.query is not None
        human_review, phrase_table = self.aiHelpers.check_user_prompt_with_filter_chain(
            self.session.query, stage
        )
        status = "NOT_OK" if human_review is True else "OK"

        doc: Dict[str, Dict[str, Any]] = {}
        doc["meta"] = {
            "Stage": stage,
            "Time": datetime.now(),
            "Status": status,
        }
        self.csvWriter.write_json2csv(
            self.bannedPhraseCollector.prepare_for_csv_print(phrase_table, doc["meta"]),
            "HUMAN_REVIEW",
        )

        if human_review is True:
            return True

        # Now check user query against LLM

        human_review, _ = self.aiHelpers.check_prompt_with_llm_guard(self.session.query)

        status = "NOT_OK" if human_review is True else "OK"

        doc: Dict[str, Dict[str, Any]] = {}
        doc["meta"] = {
            "Stage": stage,
            "Time": datetime.now(),
            "Status": status,
            "Session": self.session.export_session_state_as_cell(),
        }
        self.csvWriter.write_json2csv(
            self.bannedPhraseCollector.prepare_print_for_chat(doc["meta"]),
            "HUMAN_REVIEW",
        )
        if human_review is False:
            # Will issue user query and check
            self.chatter.run(self.session)
        return True

    def _run_query(self):
        print(f"{GREEN}Chat with your documents\n{RESET}")
        while True:
            if self._process_query() == False:
                print(f"{GREEN}Exiting HarinezumIgel RAG chat.{RESET}")
                break

    def _show_results(self):
        # Failed (level "O" if zero else "W")
        self.inf.write_counter_and_csv(
            label="RAG_CHAT_QUERIES:  ",
            count=0,
            csv_key="HUMAN_REVIEW",
            log_message="Have a look at RAG_CHAT_QUERIES .xlsx / .csv file",
            failure_indication=True,
        )

    def _run(self):
        self._showIntro()
        self._run_query()
        self._show_results()
        if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1" and NetworkTracer is not None:
            NetworkTracer.disable_tracer()  # Patch socket to trace network calls


def main():
    rc = RAGChat(cfg)
    rc._run()  # type: ignore[reportPrivateUsage]


if __name__ == "__main__":
    StartupCommons.run_with_top_level_handlers(main)
