# Local module imports
from pathlib import Path
from typing import Any, Optional

import requests
# Standard library includes
import torch

from Commons.Exceptions import NoVirtualEnvError, OllamaNotRunning


# Custom exception for OpenWebUI
class OpenWebUINotRunning(Exception):
    pass


from Compliance.Exclusions import Exclusions
from Config.Config import Config
from Globals.CounterInstance import (FailedCount, HumanReviewCount,
                                     IgnoredCount, ProcessedCount)
from Gui.Colors import GREEN, ORANGE, RED, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.CSVWriter import CSVWriter
from Helpers.Helpers import Helpers
from Helpers.PipelineSettingsSummarizer import PipelineSettingsSummarizer


class Informer:
    def __init__(self) -> None:
        # Optionally, initialize instance variables (like a custom logger) here.
        # For example: self.logger = your_logger_instance
        self.cfg: Config = Config()
        self.friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        self.pretty: PrettyWriter = PrettyWriter()
        self.helpers: Helpers = Helpers()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.csvWriter: CSVWriter = CSVWriter()
        self.piplineSummarizer: PipelineSettingsSummarizer = (
            PipelineSettingsSummarizer()
        )
        self.processedCounter: ProcessedCount = ProcessedCount()
        self.ignoredCounter: IgnoredCount = IgnoredCount()
        self.failedCounter: FailedCount = FailedCount()
        self.humanReviewCounter: HumanReviewCount = HumanReviewCount()
        self.exclusions: Exclusions = Exclusions()
        self.exclusions_file_name: str = self.exclusions.get_exclusions_filename()

        self.default_algos: list[str] = self.cfg.get_list("_DEFAULT_ALGOS")
        self.is_streaming: bool = self.helpers.get_model_args("_OLLAMA").get(
            "STREAMING_REQ", False
        )

        if self.helpers.in_venv() is False:
            msg = f"Virtual environment expected to run this script."
            self.pretty.write("E", "venv", msg)
            raise NoVirtualEnvError(msg)

        self.use_exclusions: bool = self.cfg.get_bool("USE_EXCLUSIONS")

    def _check_openwebui_is_running(self) -> None:
        """
        Checks if OpenWebUI is running by sending a request to its /v1/models endpoint.
        Only runs if self.friendly_name == "RAGChatService".
        Throws OpenWebUINotRunning if not reachable.
        """
        # Get OpenWebUI host/port from config or use default
        openwebui_args: dict[str, Any] = self.helpers.get_model_args("_OPENWEBUI")
        base_url = openwebui_args["BASE_URL"]
        try:
            resp: requests.Response = requests.get(base_url, timeout=2)
            resp.raise_for_status()
        except requests.RequestException:
            msg: str = f"Can't reach OpenWebUI on: {base_url}"
            self.pretty.write(
                "E",
                "OpenWebUI",
                msg,
                color=RED,
            )
            raise OpenWebUINotRunning(msg)
        self.pretty.write(
            "O",
            "OpenWebUI",
            f"OpenWebUI is reachable on: {base_url}",
        )

    """
    A collection of helper methods for handling RAG (Retrieval-Augmented Generation)
    document queries. This implementation uses instance methods rather than static methods.
    """

    def _check_ollama_is_running(self) -> None:
        # Read the base URL (defaults to localhost if not set)
        base_url: str = self.helpers.get_model_args("_OLLAMA").get("BASE_URL", "")
        base_url = self.helpers.normalize_base_url(base_url)
        try:
            # Try to reach the /health endpoint with a short timeout
            resp: requests.Response = requests.get(base_url, timeout=2)
            resp.raise_for_status()
        except requests.RequestException:
            # Print error to stderr and exit with non-zero code
            msg: str = f"Can't reach OLLAMA on: {base_url}"
            self.pretty.write(
                "E",
                "OLLAMA",
                f"Can't reach OLLAMA on: {base_url}",
                color=RED,
            )
            raise OllamaNotRunning(msg)

        # If we get here, Ollama is healthy
        self.pretty.write(
            "O",
            "OLLAMA",
            f"OLLAMA is reachable on: {base_url} streaming: {self.is_streaming}",
        )

    def inform(self) -> None:
        self.piplineSummarizer.display()
        self._check_ollama_is_running()
        if self.friendly_name == "RAGChatService":
            self._check_openwebui_is_running()
        self._inform_cuda()
        if self.friendly_name == "RAGLoad":
            self._delete_collection()
        if self.cfg.get_int("DEBUG_LEVEL") >= 4:
            self.pretty.write(
                "I",
                "Actual config values:",
                f"CLI Arguments -> Program specific -> Global",
                30,
            )
            self.cfg.print_config_values()
            self.pretty.write("N", "-", "----------------------")
        self._llm_info()

    def _inform_cuda(self) -> None:
        if torch.cuda.is_available():
            # Probe the GPU with a tiny allocation to verify the CUDA
            # runtime/drivers are actually functional in this environment.
            try:
                torch.zeros(1, device="cuda")
            except RuntimeError:
                self._fallback_to_cpu(
                    "GPU reported as available but CUDA failed "
                    "(probably drivers/runtime not installed in this virtual environment)."
                )
                return

            self.pretty.write("O", "GPU", "GPU is available")
            self.pretty.write(
                "I", "GPU", f"Number of GPUs: {torch.cuda.device_count()}"
            )
            self.pretty.write("I", "GPU", f"GPU Name: {torch.cuda.get_device_name()}")
            if self.cfg.get_bool("USE_CPU"):
                self.pretty.write(
                    "W",
                    "GPU",
                    "Configuration USE_CPU is True. Using CPU although GPU is available.",
                )
        else:
            if not self.cfg.get_bool("USE_CPU"):
                self._fallback_to_cpu("No GPU detected.")

        self.pretty.write("N", "-", "----------------------")

    def _fallback_to_cpu(self, reason: str) -> None:
        """Switch configuration to CPU/32-bit and warn the user.

        Sets config flags directly without instantiating AIHelpers,
        which would trigger eager model loading via get_hf_embeddings().
        """
        self.cfg.set("USE_CPU", True, force=True)
        self.cfg.set("EMBEDDER_BITS", 32, force=True)
        self.pretty.write("W", "HF", reason, color=ORANGE)

        self.pretty.write(
            "W",
            "GPU → CPU",
            "(possible drivers/runtime not installed in this virtual environment). "
            "Automatically switched to CPU with 32-bit precision. "
            "To enable GPU acceleration, install the matching CUDA-enabled "
            "PyTorch wheels into your venv "
            "(see README.md → GPU Setup).",
            color=ORANGE,
        )

    def _delete_collection(self) -> None:
        keepCollection: bool = self.cfg.get_bool("CHROMA_COLLECTION_KEEP")

        if keepCollection is False:
            collection: str = self.cfg.get_str("COLLECTION")
            self.pretty.write(
                "W",
                "Chroma DB collection",
                f"Chroma DB collection: {collection} is deleted. Key Configuration/Config_Global.py CHROMA_COLLECTION_KEEP: {keepCollection}",
                color=ORANGE,
            )
            self.chromaDBHelper.chroma_coll_name_and_mkdir_or_del("delete", collection)
            self.pretty.write("N", "-", "----------------------")

    def show_results(self) -> None:
        """
        Print counters and convert provided CSVs to XLSX (strip .csv and replace with .xlsx).
        Returns a dict with created xlsx paths (or None).
        """
        # Good (always level "O" in original)
        self.write_counter_and_csv(
            label="Good:    ",
            count=self.processedCounter.get()
            - self.humanReviewCounter.get()
            - self.failedCounter.get(),
            csv_key="OK",
            log_message="Have a look at GOOD .xlsx / .csv file",
            failure_indication=False,
        )

        # Failed (level "O" if zero else "W")
        self.write_counter_and_csv(
            label="Failed:  ",
            count=self.failedCounter.get(),
            csv_key="NOT_OK",
            log_message="Have a look at Failed .xlsx / .csv file",
            failure_indication=True,
        )

        # Human Review (level "O" if zero else "W")

        self.write_counter_and_csv(
            label="Human Review:  ",
            count=self.humanReviewCounter.get(),
            csv_key="HUMAN_REVIEW",
            log_message="Have a look at Failed .xlsx / .csv file",
            failure_indication=True,
        )

        if self.use_exclusions:
            self.pretty.write(
                "I",
                "Not Processed:",
                f"Excluded: {self.exclusions.get_applied_exclusions()} Reason: Contained in exclusion file {self.exclusions_file_name}:",
            )
        # Ignored (no CSV/logs in original)
        self.pretty.write(
            "I",
            "Not processed: ",
            f"Files: {self.ignoredCounter.get()} Reason: Unchanged file hash",
        )

    def write_counter_and_csv(
        self,
        label: str,
        count: int,
        csv_key: str,
        log_message: str | None = None,
        failure_indication: bool = True,
    ) -> None:
        """
        Helper to write a counter line, convert the corresponding CSV to XLSX,
        and optionally write a Logs line if processedCounter is non-zero.
        - label: text label for the counter (e.g., "Failed:  ")
        - counter: counter object with .get()
        - csv_key: key passed to convert_csv2xlsx (e.g., "OK")
        - log_message: prefix for the Logs line (if None, no Logs line is written)
        - compute_level: if True compute level as "O" if counter==0 else "W";
                        if False use "O" (keeps original Good behavior)
        """

        level: str = "W"
        color: str = "ORANGE"
        if failure_indication is False and count == 0:
            level = "W"
            color = ORANGE
        elif failure_indication is False and count > 0:
            level = "O"
            color = GREEN
        elif failure_indication is True and count == 0:
            level = "O"
            color = GREEN
        elif failure_indication is True and count > 0:
            level = "W"
            color = ORANGE

        self.pretty.write(level, label, str(count), color=color)

        # Convert CSV to XLSX and prepare csv path
        xlsx_path: Optional[Path] = self.csvWriter.convert_csv2xlsx(csv_key)
        if xlsx_path is None:
            return
        csv_path: Path = xlsx_path.with_suffix(".csv")
        self.pretty.write("I", "Logs", f"{log_message} {xlsx_path} / {csv_path}")

    def _llm_info(self) -> None:
        llm_args: dict[str, Any] = self.helpers.get_model_args("_LLM")
        llm_chk_args: dict[str, Any] = self.helpers.get_model_args("_LLM_CHK")
        self.llm_model: str = llm_args["MODEL"]
        self.llm_compliance_msg: str = llm_args["COMPLIANCE_MSG"]
        self.llm_tag: str = llm_args["TAG"]
        self.llm_chk_tag: str = llm_chk_args["TAG"]
        self.llm_chk_model: str = llm_chk_args["MODEL"]
        self.llm_chk_compliance_msg: str = llm_chk_args["COMPLIANCE_MSG"]

        details = None
        if self.friendly_name == "DocClassify":
            details = "LLM for doc classification"
        if self.friendly_name == "RAGChat":
            details = "LLM for user chat"
        if details is not None:
            self.pretty.write(
                "I",
                f"LLM for prompt compliance",
                f"{self.llm_chk_model} ({self.llm_chk_tag})",
                color=YELLOW,
            )

            self.pretty.write(
                "I",
                f"LLM for prompt compliance",
                f"{self.llm_chk_compliance_msg}",
                color=YELLOW,
            )

            self.pretty.write(
                "I",
                f"{details}",
                f"{self.llm_compliance_msg}",
                color=YELLOW,
            )

            self.pretty.write(
                "I",
                f"{details}",
                f"{self.llm_model} ({self.llm_tag})",
                color=YELLOW,
            )
