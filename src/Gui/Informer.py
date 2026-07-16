# Local module imports
import os
from pathlib import Path
from typing import Any, Optional

import requests  # type: ignore[reportMissingModuleSource]
# Standard library includes
import torch  # type: ignore[reportMissingImports]

from Commons.Exceptions import (CollectionNotFoundError, NoVirtualEnvError,
                                OllamaNotRunning, VllmNotRunning)
from Strategies.BM25Retriever import BM25Retriever
from Strategies.GraphRetriever import GraphRetriever


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
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PipelineSettingsSummarizer import PipelineSettingsSummarizer


class Informer:
    def __init__(self) -> None:
        # Optionally, initialize instance variables (like a custom logger) here.
        # For example: self.logger = your_logger_instance
        self.cfg: Config = Config()
        self.friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        self.pretty: PrettyWriter = PrettyWriter(always_on=True)
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
        self.is_streaming: bool = self.helpers.get_active_endpoint_args().get(
            "STREAMING_REQ", False
        )

        if self.helpers.in_venv() is False:
            msg = f"Virtual environment expected to run this script."
            self.pretty.write("E", "venv", msg)
            raise NoVirtualEnvError(msg)

        self.use_exclusions: bool = self.cfg.get_bool("USE_EXCLUSIONS")

    def _check_openwebui_is_running(self) -> None:
        """
        Checks if OpenWebUI is running by probing with fallback logic.
        Only runs if self.friendly_name == "RAGChatService".
        Throws OpenWebUINotRunning if not reachable.
        """
        openwebui_args: dict[str, Any] = self.helpers.get_model_args(
            "_ACTIVE_OPENWEBUI"
        )
        base_url: str = str(openwebui_args.get("BASE_URL", ""))

        effective_url = self.helpers.find_provider_url(
            base_url=base_url,
            probe_path="/",
            generate_path="",  # OpenWebUI doesn't have a generate path, just use base
            default_port=8080,
            headers={},
            label="OpenWebUI",
        )
        if effective_url is None:
            configured_base = self.helpers.normalize_base_url(base_url)
            msg = (
                f"Can't reach OpenWebUI on: {configured_base}. "
                "Start OpenWebUI (`open-webui serve`) or update _MODELS.openwebui._OPENWEBUI.BASE_URL."
            )
            self.pretty.write("E", "OpenWebUI", msg, color=RED)
            raise OpenWebUINotRunning(msg)

        if effective_url != base_url:
            self.cfg.set(
                "_MODELS.openwebui._OPENWEBUI.BASE_URL", effective_url, force=True
            )
            self.pretty.write(
                "W",
                "OpenWebUI",
                f"Configured endpoint {base_url} unavailable; using fallback endpoint: {effective_url}",
                color=YELLOW,
            )
        self.pretty.write(
            "O", "OpenWebUI", f"OpenWebUI is reachable on: {effective_url}"
        )

    def _probe_headers(self, endpoint_args: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = str(endpoint_args.get("API_KEY", "")).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _check_ollama_is_running(self) -> None:
        endpoint_args: dict[str, Any] = self.helpers.get_active_endpoint_args()
        base_url: str = str(endpoint_args.get("BASE_URL", ""))
        headers = self._probe_headers(endpoint_args)

        effective_url = self.helpers.find_provider_url(
            base_url=base_url,
            probe_path="/api/tags",
            generate_path="/api/generate",
            default_port=11434,
            headers=headers,
            label="OLLAMA",
        )
        if effective_url is None:
            configured_base = self.helpers.normalize_base_url(base_url)
            model_name = str(
                self.helpers.get_model_args("_ACTIVE_LLM").get("MODEL", "")
            )
            model_info = f" (model: {model_name})" if model_name else ""
            msg = (
                f"Can't reach OLLAMA on: {configured_base}{model_info}. "
                "Start Ollama (`ollama serve`) or update _MODELS.ollama._OLLAMA.BASE_URL."
            )
            self.pretty.write("E", "OLLAMA", msg, color=RED)
            raise OllamaNotRunning(msg)

        if effective_url != base_url:
            self.cfg.set("_MODELS.ollama._OLLAMA.BASE_URL", effective_url, force=True)
            self.pretty.write(
                "W",
                "OLLAMA",
                f"Configured endpoint {base_url} unavailable; using fallback endpoint: {effective_url}",
                color=YELLOW,
            )
        probe_url = effective_url.replace("/api/generate", "/api/tags")
        self.pretty.write(
            "O",
            "OLLAMA",
            f"OLLAMA is reachable on: {probe_url} streaming: {self.is_streaming}",
        )

    def _check_vllm_is_running(self) -> None:
        endpoint_args: dict[str, Any] = self.helpers.get_active_endpoint_args()
        base_url: str = str(endpoint_args.get("BASE_URL", ""))
        configured_base = self.helpers.normalize_base_url(base_url)
        path_suffix = (
            base_url[len(configured_base) :]
            if base_url.startswith(configured_base)
            else "/v1/chat/completions"
        )
        headers = self._probe_headers(endpoint_args)

        effective_url = self.helpers.find_provider_url(
            base_url=base_url,
            probe_path="/v1/models",
            generate_path=path_suffix,
            default_port=4000,
            headers=headers,
            label="VLLM",
        )
        if effective_url is None:
            model_name = str(
                self.helpers.get_model_args("_ACTIVE_LLM").get("MODEL", "")
            )
            model_info = f" (model: {model_name})" if model_name else ""
            msg = (
                f"Can't reach VLLM on: {configured_base}{model_info}. "
                "Start vLLM or update _MODELS.vllm._VLLM.BASE_URL."
            )
            self.pretty.write("E", "VLLM", msg, color=RED)
            raise VllmNotRunning(msg)

        if effective_url != base_url:
            self.cfg.set("_MODELS.vllm._VLLM.BASE_URL", effective_url, force=True)
            self.pretty.write(
                "W",
                "VLLM",
                f"Configured endpoint {base_url} unavailable; using fallback endpoint: {effective_url}",
                color=YELLOW,
            )
        probe_url = f"{self.helpers.normalize_base_url(effective_url)}/v1/models"
        self.pretty.write(
            "O",
            "VLLM",
            f"VLLM is reachable on: {probe_url} streaming: {self.is_streaming}",
        )

    def inform(self) -> None:
        self.piplineSummarizer.display()
        if self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama").lower() == "vllm":
            self._check_vllm_is_running()
        else:
            self._check_ollama_is_running()
        if self.friendly_name == "RAGChatService":
            self._check_openwebui_is_running()
        self._inform_cuda()
        if self.friendly_name == "RAGLoad":
            self._delete_collection()
        if DebugHelper.check(self.cfg, 40):
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
        _torch: Any = torch
        if _torch.cuda.is_available():
            # Probe the GPU with a tiny allocation to verify the CUDA
            # runtime/drivers are actually functional in this environment.
            try:
                _torch.zeros(1, device="cuda")
            except RuntimeError:
                self._fallback_to_cpu(
                    "GPU reported as available but CUDA failed "
                    "(probably drivers/runtime not installed in this virtual environment)."
                )
                return

            self.pretty.write("O", "GPU", "GPU is available")
            self.pretty.write(
                "I", "GPU", f"Number of GPUs: {_torch.cuda.device_count()}"
            )
            self.pretty.write("I", "GPU", f"GPU Name: {_torch.cuda.get_device_name()}")
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
        keepIndexes: bool = self.cfg.get_bool("RETRIEVAL_STORES_KEEP")
        collection: str = self.cfg.get_str("COLLECTION")

        if keepIndexes is False:
            self.pretty.write(
                "W",
                "Chroma DB collection",
                f"Chroma DB collection: {collection} is deleted.\n"
                f"Key: Configuration/Config_Global.py RETRIEVAL_STORES_KEEP: {keepIndexes}",
            )
            self.chromaDBHelper.chroma_coll_name_and_mkdir_or_del("delete", collection)

            # Delete the BM25 index directory (sibling of the ChromaDB dir)
            bm25_dir = os.path.join(
                self.cfg.get_str("_BM25_INDEX.BM25_INDEX_DIR"), collection
            )
            if os.path.exists(bm25_dir):
                if self.chromaDBHelper.fileUtils.delete_file_or_dir(bm25_dir):
                    self.pretty.write(
                        "I", "BM25", f"Deleted BM25 index directory {bm25_dir}."
                    )

            # Delete the graph index directory (sibling of the ChromaDB dir)
            graph_dir = os.path.join(
                self.cfg.get_str("_GRAPH_INDEX.GRAPH_INDEX_DIR"), collection
            )
            if os.path.exists(graph_dir):
                if self.chromaDBHelper.fileUtils.delete_file_or_dir(graph_dir):
                    self.pretty.write(
                        "I", "Graph", f"Deleted graph index directory {graph_dir}."
                    )

            self.pretty.write("N", "-", "----------------------")
        else:
            # RETRIEVAL_STORES_KEEP is True — verify that the BM25 index exists on disk.
            # A missing BM25 index while the collection is supposed to be preserved
            # means the index is out of sync; abort early so the user can rebuild.
            bm25_dir = os.path.join(
                self.cfg.get_str("_BM25_INDEX.BM25_INDEX_DIR"), collection
            )
            bm25_index_path = os.path.join(bm25_dir, BM25Retriever.INDEX_FILENAME)
            if not os.path.isfile(bm25_index_path):
                msg = (
                    f"BM25 index for collection '{collection}' not found at "
                    f"{bm25_index_path}. "
                    f"Re-run RAGLoad with RETRIEVAL_STORES_KEEP = False to rebuild "
                    f"the collection and its BM25 index."
                )
                self.pretty.write("E", "BM25", msg, color=RED)
                raise CollectionNotFoundError(msg)

            # Verify that the graph index also exists on disk.
            graph_dir = os.path.join(
                self.cfg.get_str("_GRAPH_INDEX.GRAPH_INDEX_DIR"), collection
            )
            graph_index_path = os.path.join(graph_dir, GraphRetriever.INDEX_FILENAME)
            if not os.path.isfile(graph_index_path):
                msg = (
                    f"Graph index for collection '{collection}' not found at "
                    f"{graph_index_path}. "
                    f"Re-run RAGLoad with RETRIEVAL_STORES_KEEP = False to rebuild "
                    f"the collection and all its retrieval indexes."
                )
                self.pretty.write("E", "Graph", msg, color=RED)
                raise CollectionNotFoundError(msg)

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
        llm_args: dict[str, Any] = self.helpers.get_model_args("_ACTIVE_LLM")
        llm_chk_args: dict[str, Any] = self.helpers.get_model_args("_ACTIVE_LLM_CHK")
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
