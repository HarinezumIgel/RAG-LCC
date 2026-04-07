import argparse
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn

from Commons.Exceptions import (ArgosConsentMissingError,
                                ChromaInstallCurrentEmbeddingsMismatch,
                                CollectionNotFoundError,
                                ComplianceViolationError, ConfigPathError,
                                DataProcessingError, DocumentsDirError,
                                EmbedModelMismatch, ExclusionsError,
                                HFDownloaderError,
                                InternetConnectionDisabledError,
                                InvalidCollectionName, LLMComplianceCheckError,
                                LLMResultError, ModelLoadError,
                                OllamaNotRunning, PersistDirError,
                                PromptComplianceError, RerankError,
                                UserNoDownLoadAccept)
from Config.AddConstantsFromConfigFile import AddConstantsFromConfigFile
from Config.Config import Config
from Gui.Banner import Banner
from Gui.Colors import BRIGHT_BLUE, MAGENTA, ORANGE, RED
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.Helpers import Helpers


def suppress_argos_logging(debug_level: int = 0) -> None:
    """Suppress noisy argostranslate/stanza log messages.

    With ARGOS_CHUNK_TYPE=SPACY (the default) stanza is not used for
    sentence boundary detection, but argostranslate still emits warnings
    from its utils logger.  Silence them unless DEBUG_LEVEL is very high.
    """
    if debug_level < 50:
        logging.getLogger("argostranslate.utils").setLevel(logging.ERROR)


class StartupCommons:
    @staticmethod
    def _ensure_started_from_project_root(cfg: Config) -> None:
        configured_root = cfg.get("_ABSOLUTE_PATH", None)
        if configured_root:
            project_root = Path(str(configured_root)).expanduser().resolve()
        else:
            project_root = Path(__file__).resolve().parents[2]
        cwd = Path.cwd().resolve()

        project_root_norm = os.path.normcase(str(project_root))
        cwd_norm = os.path.normcase(str(cwd))

        if cwd_norm != project_root_norm:
            pretty = PrettyWriter()
            pretty.write(
                "E",
                "Startup",
                f"Start this app from project root: {project_root} (current working directory: {cwd})",
                color=RED,
            )
            sys.exit(1)

    @staticmethod
    def _validate_collection_config(cfg: Config) -> None:
        """
        Raise InvalidCollectionName early — right after CLI args are merged
        into config — so no subsystem ever receives a path as a collection name.
        """
        raw: str | None = cfg.get("COLLECTION")  # type: ignore[assignment]
        if raw is not None and any(ch in str(raw) for ch in ("/", "\\", ":")):
            raise InvalidCollectionName(str(raw))

    @staticmethod
    def _suppress_argos_logging(debug_level: int = 0) -> None:
        """Delegate to the module-level function."""
        suppress_argos_logging(debug_level)

    @staticmethod
    def _get_stacktrace(exc: BaseException) -> str:
        """
        Return a clean, deterministic stacktrace string for the given exception.
        """
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    @staticmethod
    def run_with_top_level_handlers(main_callable: Callable[[], None]) -> None:
        try:
            main_callable()
        except Exception as exc:
            StartupCommons._handle_top_level_exception(exc)

    @dataclass
    class StartupContext:
        args: argparse.Namespace
        cfg: Config
        banner: Banner

    @staticmethod
    def common_start(app_name: str, description: str) -> StartupContext:
        try:
            parser = AddConstantsFromConfigFile(description=description)
            args = parser.parse_args()
            cfg = Config(args)
            emoji_ok: bool = Symbols.store_emoji_preference(cfg)
            StartupCommons._ensure_started_from_project_root(cfg)
            StartupCommons._validate_collection_config(cfg)

            banner = Banner(cfg)
            banner.startup_banner()

            if not emoji_ok:
                PrettyWriter().write(
                    "W",
                    "Terminal",
                    "Emoji are not supported in this terminal. "
                    "Run inside VS Code's integrated terminal "
                    "for a richer display.",
                    color=ORANGE,
                )

            # Ensure we are running inside a virtual environment
            Helpers().is_in_venv(required=True)

            # Load configuration and writer
            pretty = PrettyWriter()

            # HF cache locations
            hf_home = cfg.get_str("_HF_HOME", "")
            hf_hub = cfg.get_str("_HF_HUB_CACHE", "")

            # Toggle boolean-style HF/transformers related env vars based on connection mode
            level = "O"
            color = BRIGHT_BLUE
            warn_print = False
            #                                   (expected, triggers_warn)
            _env_checks: dict[str, tuple[str | None, bool]] = {
                "HF_HUB_OFFLINE": ("1", True),
                "HF_DATASETS_OFFLINE": ("1", True),
                "TRANSFORMERS_OFFLINE": ("1", True),
                "LICENSE_DOWNLOAD": ("0", True),
                "RAG_LCC_NW_TRACE": ("0", False),
                "RAG_LCC_STACK_TRACE": ("0", False),
                "NLTK_STOPWORDS_DOWNLOAD": ("0", True),
                "ARGOS_MODEL_PROVIDER": ("OPENNMT", True),
                "ARGOS_CHUNK_TYPE": ("SPACY", True),
                "ARGOS_STANZA_DOWNLOAD": ("0", True),
                "HF_HUB_DISABLE_PROGRESS_BARS": (None, False),
            }
            friendly_name: str = cfg.get_str("_FRIENDLY_NAME", "")
            if friendly_name == "RAGChatService":
                _env_checks["SERVE_OPENWEBUI_CHAT"] = ("0", True)
            for key, (target_value, triggers_warn_flag) in _env_checks.items():
                if target_value is not None and os.environ[key] != target_value:
                    color = ORANGE
                    if triggers_warn_flag:
                        warn_print = True
                else:
                    color = BRIGHT_BLUE
                pretty.write(
                    "I", "Environment variable", f"{key}={os.environ[key]}", color=color
                )

            if warn_print:
                pretty.write("N", "", "")
                pretty.write(
                    f"{level}",
                    "Internet connection",
                    f"Internet settings allow full or partial download",
                    color=ORANGE,
                )

            if (
                cfg.get("_FRIENDLY_NAME") == "RAGChatService"
                and os.environ.get("SERVE_OPENWEBUI_CHAT") == "1"
            ):
                pretty.write("N", "", "")
                pretty.write(
                    "W",
                    "RAGChatService",
                    "RAGChatService is serving requests via OpenWebUI (SERVE_OPENWEBUI_CHAT=1)",
                    color=ORANGE,
                )

            StartupCommons._suppress_argos_logging(cfg.get_int("DEBUG_LEVEL"))

            pretty.write("N", "", "")
            pretty.write(
                "I",
                "OLLAMA",
                f"Access to Ollama is *always* enabled. {cfg.get('_MODELS.ollama._OLLAMA.BASE_URL')} streaming: {cfg.get_bool('_MODELS.ollama._OLLAMA.STREAMING_REQ')}",
            )
            if hf_home:
                os.environ["_HF_HOME"] = hf_home
            if hf_hub:
                os.environ["_HF_HUB_CACHE"] = hf_hub
            pretty.write(
                f"{level}", "HF cache", f"HF home: {hf_home} HF hub cache: {hf_hub}"
            )
            pretty.write("N", "", "")
            pretty.write(
                "N",
                "",
                f"Running {cfg.get('_FRIENDLY_NAME')}. Good luck!\n",
                color=MAGENTA,
            )
            from Compliance.Compliance import Compliance

            Compliance().verify()

            return StartupCommons.StartupContext(args=args, cfg=cfg, banner=banner)
        except Exception as exc:
            StartupCommons._handle_top_level_exception(exc)

    @staticmethod
    def _handle_top_level_exception(exc: BaseException) -> NoReturn:
        pretty = PrettyWriter()

        if isinstance(exc, FileNotFoundError):
            pretty.write("W", "FILE NOT FOUND", f"{exc.args[0]}", color=ORANGE)
            sys.exit(1)

        if isinstance(
            exc,
            (
                UserNoDownLoadAccept,
                ConfigPathError,
                CollectionNotFoundError,
                RerankError,
                PersistDirError,
                DocumentsDirError,
                EmbedModelMismatch,
                ChromaInstallCurrentEmbeddingsMismatch,
            ),
        ):
            sys.exit(1)

        if isinstance(
            exc,
            (ComplianceViolationError, PromptComplianceError, LLMComplianceCheckError),
        ):
            if isinstance(exc, ArgosConsentMissingError):
                pretty.write(
                    "E",
                    "ARGOS LICENSE",
                    f"{exc.args[0]}",
                    color=RED,
                )
                pretty.write(
                    "I",
                    "ARGOS LICENSE",
                    "Run:  python src/Scripts/ArgosTranslatePackages.py install  "
                    "to accept the Argos Translate license and then restart.",
                    color=ORANGE,
                )
                sys.exit(1)
            pretty.write(
                "E",
                "COMPLIANCE VIOLATION",
                f"Execution stopped due to compliance check: {exc.args[0]}",
                color=RED,
            )
            sys.exit(1)

        if isinstance(exc, ExclusionsError):
            pretty.write(
                "E",
                "Exclusions",
                f"Execution stopped due to: {exc.args[0]}",
                color=RED,
            )
            sys.exit(1)

        if isinstance(exc, InternetConnectionDisabledError):
            reason: str = str(exc) if str(exc) else ""
            pretty.write(
                "E",
                "INTERNET ACCESS DISABLED",
                f"{reason} "
                f"(Probably change internet access flags in Configuration/Config_Internet_Env.py)",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, HFDownloaderError):
            pretty.write(
                "E",
                "HF_Downloader",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, OllamaNotRunning):
            pretty.write(
                "E",
                "Ollama",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, ModelLoadError):
            pretty.write(
                "E",
                "Model Load Error",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, DataProcessingError):
            pretty.write(
                "E",
                "Data Processing",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, LLMResultError):
            pretty.write(
                "E",
                "LLM Result Error",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            sys.exit(1)

        if isinstance(exc, InvalidCollectionName):
            pretty.write(
                "E",
                "Collection name",
                f"'{exc.args[0]}' looks like a file-system path. "
                f"Please provide a plain collection name without '/', '\\\\', or ':' "
                f"(e.g. 'Test', not 'C:\\\\path\\\\to\\\\Test' or './Test').",
                color=RED,
            )
            sys.exit(1)

        # --- Unexpected errors ---
        pretty.write(
            "E",
            "UNEXPECTED TOP-LEVEL FAILURE",
            f"error_type={type(exc).__name__}, error_message={str(exc)}",
            color=RED,
        )
        if os.environ.get("RAG_LCC_STACK_TRACE", "0") == "1":
            print(StartupCommons._get_stacktrace(exc))
        else:
            pretty.write(
                "I",
                "Stack trace",
                'Stack trace was omitted because RAG_LCC_STACK_TRACE = "0"',
            )
        sys.exit(1)
