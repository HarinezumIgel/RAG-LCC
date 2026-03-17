import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # side-effect import

if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1":
    from Commons.NetworkTracer import NetworkTracer

    NetworkTracer.enable_tracer()  # Patch socket to trace network calls
else:
    NetworkTracer = None  # type: ignore[assignment,misc]

from Commons.StartupCommons import StartupCommons

ctx = StartupCommons.common_start(
    "DocClassify", "DocClassify – all Constants become CLI flags"
)
cfg = ctx.cfg  # typed as Config args = ctx.args

# Standard library imports
from datetime import datetime

from AI.AIHelpers import AIHelpers
from Compliance.HFDownloader import HFDownloader
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import ORANGE as ORANGE  # re-exported
from Gui.Colors import RED as RED
from Gui.Informer import Informer
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Pipeline.LoadAndClassifyProcessor import LoadAndClassifyProcessor
from Strategies.ClassifyStrategy import ClassifyStrategy


class DocClassify:
    def __init__(self, cfg: Config) -> None:
        self.globalsInstance: Globals = Globals()
        self.cfg: Config = cfg
        self.hf_downloader: HFDownloader = HFDownloader()
        self.hf_downloader.download("_MODELS._EMBED")
        self.informer: Informer = Informer()
        self.helpers: Helpers = Helpers()

        self.aiHelpers: AIHelpers = AIHelpers()
        self.fileUtils: FileUtils = FileUtils()
        self.logger: Any = self.helpers.setup_logger()
        self.globalsInstance.set_logger(self.logger)
        self.pretty: PrettyWriter = PrettyWriter()

        self.helpers.check_cpu_and_bits()
        self.fileUtils.setDebug()
        self.informer.inform()

    def _run(self):
        prompt_var: str = self.helpers.get_model_args("_LLM")["PROMPT_CLASSIFY"]
        self.pretty.write("N", f"Prompt used: {prompt_var}", "")
        for i, line in enumerate(cfg.get_str(prompt_var).splitlines(), start=1):
            self.pretty.write("N", "", f"Line {i}: {line}")
        self.pretty.write("N", "-", "----------------------")

        # Record the start time
        start_time = datetime.now()

        self.pretty.write("N", "", f"\nI am starting at {start_time}")
        self.pretty.write("I", "", "Starting Meta Data Extraction ...")

        self.strat: ClassifyStrategy = ClassifyStrategy()
        self.proc: LoadAndClassifyProcessor = LoadAndClassifyProcessor(self.strat)
        # Here we go
        self.proc.process_files()
        self.pretty.write("N", "-", "----------------------")

        # Show the results
        self.informer.show_results()

        # Record the end time
        end_time = datetime.now()
        self.pretty.write("N", "", f"I ended {end_time}")

        # Calculate the time difference
        elapsed_time = end_time - start_time
        self.pretty.write(
            "N",
            "",
            f"Workload took {elapsed_time} / {elapsed_time.total_seconds()} seconds",
        )
        if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1" and NetworkTracer is not None:
            NetworkTracer.disable_tracer()  # Patch socket to trace network calls


def main():
    dc = DocClassify(cfg)
    dc._run()  # type: ignore[reportPrivateUsage]


if __name__ == "__main__":
    StartupCommons.run_with_top_level_handlers(main)
