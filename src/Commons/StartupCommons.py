# --- venv bootstrap: re-exec with .venv Python if not already inside it ---
# Must be the very first code — project imports below require venv packages.
import os
import sys

if sys.prefix == sys.base_prefix:
    from pathlib import Path as _Path

    _venv_py = (
        _Path(__file__).resolve().parents[2]
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    if _venv_py.exists():
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
# -------------------------------------------------------------------------

import argparse
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn

import requests

from Commons.Exceptions import (ArgosConsentMissingError, ArgosPermissionError,
                                BackendUnavailableError,
                                ChromaInstallCurrentEmbeddingsMismatch,
                                CollectionNotFoundError,
                                ComplianceViolationError, ConfigPathError,
                                ConfigurationError, DataProcessingError,
                                DocumentsDirError, EmbedModelMismatch,
                                ExclusionsError, HFDownloaderError,
                                InternetConnectionDisabledError,
                                InvalidCollectionName, LLMComplianceCheckError,
                                LLMResultError, LocalLLMEndpointNotAvailable,
                                ModelLoadError, NoVirtualEnvError,
                                PersistDirError, PromptComplianceError,
                                RerankError, TesseractPathError,
                                UserNoDownLoadAccept)
from Config.AddConstantsFromConfigFile import AddConstantsFromConfigFile
from Config.Config import Config
from Gui.Banner import Banner
from Gui.Colors import BRIGHT_BLUE, BRIGHT_ORANGE, MAGENTA, ORANGE, RED, VIOLET
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers


def suppress_argos_logging(debug_level: int = 0) -> None:
    """Suppress noisy argostranslate/stanza log messages.

    With ARGOS_CHUNK_TYPE=SPACY (the default) stanza is not used for
    sentence boundary detection, but argostranslate still emits warnings
    from its utils logger.  Silence them unless DEBUG_LEVEL is very high.
    """
    if debug_level < 55:
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
            pretty = PrettyWriter(always_on=True)
            pretty.write(
                "E",
                "Startup",
                f"Start this app from project root: {project_root} (current working directory: {cwd})",
                color=RED,
            )
            StartupCommons._die()

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
    def _die(code: int = 1) -> NoReturn:
        """Hard-exit the process with a trailing newline.

        Uses ``os._exit()`` instead of ``sys.exit()`` so that debugpy cannot
        intercept ``SystemExit`` and hold the session open.  stdout/stderr are
        flushed explicitly first because ``os._exit()`` bypasses normal cleanup.
        """
        print(flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    @staticmethod
    def run_with_top_level_handlers(main_callable: Callable[[], None]) -> None:
        try:
            main_callable()
        except Exception as exc:
            StartupCommons._handle_top_level_exception(exc)
        finally:
            # Flush on normal completion so the shell prompt starts on a fresh line.
            # os._exit() in _die() bypasses finally, so error paths are unaffected.
            print(flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

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
                PrettyWriter(always_on=True).write(
                    "W",
                    "Terminal",
                    "Emoji are not supported in this terminal. "
                    "Run inside VS Code's integrated terminal "
                    "for a richer display.",
                    color=ORANGE,
                )

            # Ensure we are running inside a virtual environment
            Helpers().is_in_venv(required=True)

            # Refuse to run from a drive/filesystem root ('C:\' or '/')
            Helpers().is_in_drive_root(required=True)

            # Load configuration and writer
            pretty = PrettyWriter(always_on=True)

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
            if friendly_name in ("RAGChat", "RAGChatService"):
                _web_mode_env = (
                    str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
                )
                pretty.write(
                    "I",
                    "Environment variable",
                    f"WEB_SEARCH_MODE={_web_mode_env}",
                    color=(ORANGE if _web_mode_env == "1" else BRIGHT_BLUE),
                )
            if friendly_name == "RAGChatService":
                # SERVE_OPENWEBUI_CHAT is a service mode flag (inbound connections),
                # not an outbound download risk — reported as plain status, no warning.
                _owui = os.environ.get("SERVE_OPENWEBUI_CHAT", "0")
                pretty.write(
                    "I",
                    "Environment variable",
                    f"SERVE_OPENWEBUI_CHAT={_owui}",
                    color=(ORANGE if _owui == "1" else BRIGHT_BLUE),
                )
                _docs_http = os.environ.get("SERVE_IN_MEMORY_DOCS_HTTP", "0")
                pretty.write(
                    "I",
                    "Environment variable",
                    f"SERVE_IN_MEMORY_DOCS_HTTP={_docs_http}",
                    color=(ORANGE if _docs_http == "1" else BRIGHT_BLUE),
                )

            if warn_print:
                pretty.write("N", "", "")
                pretty.write(
                    f"{level}",
                    "Outbound downloads",
                    "One or more settings allow outbound model/data downloads "
                    "(HuggingFace hub, NLTK, Argos, etc.). "
                    "Set the relevant offline flags to prevent unexpected network access.",
                    color=ORANGE,
                )

            if (
                friendly_name == "RAGChatService"
                and os.environ.get("SERVE_OPENWEBUI_CHAT") == "1"
            ):
                pretty.write("N", "", "")
                pretty.write(
                    "I",
                    "RAGChatService",
                    "Accepting requests from OpenWebUI (SERVE_OPENWEBUI_CHAT=1) — inbound only, no outbound data risk.",
                    color=BRIGHT_BLUE,
                )

            _web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
            if _web_mode not in ("0", "1"):
                raise ConfigurationError(
                    f"WEB_SEARCH_MODE = {_web_mode!r} is not a valid value. "
                    'Allowed values: "0" | "1". '
                    "Fix Config_Internet_Env.py and restart."
                )
            if friendly_name in ("RAGChat", "RAGChatService"):
                if _web_mode == "1":
                    pretty.write(
                        "W",
                        "Web search",
                        'Web search is ENABLED (WEB_SEARCH_MODE="1"). User queries may be sent to the internet. '
                        "Review LEGAL.md \u00a7 Web Search\nand SECURITY.md before deploying.",
                        color=ORANGE,
                    )

                if (
                    friendly_name == "RAGChatService"
                    and _web_mode != "1"
                    and cfg.get_bool("_OPENWEB_UI_WEBSEARCH", False)
                ):
                    pretty.write("N", "", "")
                    pretty.write(
                        "W",
                        "Web search",
                        "_OPENWEB_UI_WEBSEARCH=True has no effect because WEB_SEARCH_MODE "
                        f'is {_web_mode!r}, not "1". '
                        'Set WEB_SEARCH_MODE="1" in Config_Internet_Env.py to activate the default.',
                        color=VIOLET,
                    )

            if friendly_name == "RAGChatService":
                docs_block: dict[str, Any] = cfg.get_dict(
                    "_SERVE_DOCS", {}, silent=True
                )
                docs_http_enabled = (
                    os.environ.get("SERVE_IN_MEMORY_DOCS_HTTP", "0") == "1"
                )
                pretty.write("N", "", "")
                if docs_http_enabled:
                    md_host = cfg.get_str("_MODELS.ragchatservice._RAGCHATSERVICE.HOST")
                    md_port = cfg.get_int("_MODELS.ragchatservice._RAGCHATSERVICE.PORT")
                    md_base = (
                        str(docs_block.get("public_base_url") or "").strip().rstrip("/")
                        or f"http://{md_host}:{md_port}"
                    )
                    pretty.write(
                        "W",
                        "Serve docs",
                        f"In-memory document-serving service active — "
                        f"endpoint: {md_base}/marked/<token>  "
                        f"(TTL {docs_block.get('ttl_seconds', 1800)} s, "
                        f"max {docs_block.get('max_total_mb', 200)} MB)",
                        color=BRIGHT_ORANGE,
                    )
                else:
                    pretty.write(
                        "I",
                        "Serve docs",
                        "In-memory document-serving service is DISABLED "
                        '(SERVE_IN_MEMORY_DOCS_HTTP="0" in Config_Internet_Env.py).',
                        color=BRIGHT_BLUE,
                    )

            StartupCommons._suppress_argos_logging(DebugHelper.level(cfg))

            endpoint_args = Helpers().get_active_endpoint_args()
            active_endpoint = str(cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()

            pretty.write("N", "", "")
            pretty.write("N", "", "")
            pretty.write(
                "I",
                active_endpoint.upper(),
                "Access to "
                f"{endpoint_args.get('FRIENDLY_NAME', active_endpoint.upper())} "
                f"is enabled. {endpoint_args.get('BASE_URL', '')} "
                f"streaming: {endpoint_args.get('STREAMING_REQ', False)}",
            )
            helpers = Helpers()
            base_url = str(endpoint_args.get("BASE_URL", ""))
            api_key = str(endpoint_args.get("API_KEY", "")).strip()
            probe_headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                probe_headers["Authorization"] = f"Bearer {api_key}"
            configured_base = helpers.normalize_base_url(base_url)
            if active_endpoint == "ollama":
                probe_path = "/api/tags"
                generate_path = "/api/generate"
                default_port = 11434
            elif active_endpoint == "vllm":
                path_suffix = (
                    base_url[len(configured_base) :]
                    if base_url.startswith(configured_base)
                    else "/v1/chat/completions"
                )
                probe_path = "/v1/models"
                generate_path = path_suffix
                default_port = 8000
            else:
                msg = (
                    f"Invalid _ACTIVE_ENDPOINT: {active_endpoint!r}. "
                    "Allowed values are 'ollama' and 'vllm'."
                )
                pretty.write("E", "Endpoint", msg, color=RED)
                raise LocalLLMEndpointNotAvailable(msg, provider=active_endpoint)
            try_fallback = bool(endpoint_args.get("TRY_FALLBACK_URLS", True))
            if try_fallback:
                effective_url = helpers.find_provider_url(
                    base_url=base_url,
                    probe_path=probe_path,
                    generate_path=generate_path,
                    default_port=default_port,
                    headers=probe_headers,
                    label=active_endpoint.upper(),
                )
                reachable = effective_url is not None
            else:
                probe_url = f"{configured_base}{probe_path}"
                pretty.write("I", active_endpoint.upper(), f"Probing {probe_url}")
                try:
                    requests.get(
                        probe_url, headers=probe_headers, timeout=2
                    ).raise_for_status()
                    pretty.write(
                        "I", active_endpoint.upper(), f"Probing {probe_url} → OK"
                    )
                    reachable = True
                except requests.RequestException:
                    pretty.write(
                        "I", active_endpoint.upper(), f"Probing {probe_url} → failed"
                    )
                    reachable = False
            if not reachable:
                model_name = str(helpers.get_model_args("_ACTIVE_LLM").get("MODEL", ""))
                model_info = f" (model: {model_name})" if model_name else ""
                msg = (
                    f"Can't reach {active_endpoint.upper()} on: {configured_base}{model_info}. "
                    f"Start {active_endpoint} or update BASE_URL in Config_Models.py."
                )
                pretty.write("E", active_endpoint.upper(), msg, color=RED)
                raise LocalLLMEndpointNotAvailable(msg, provider=active_endpoint)
            if hf_home:
                os.environ["HF_HOME"] = hf_home
            if hf_hub:
                os.environ["HF_HUB_CACHE"] = hf_hub
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
        pretty = PrettyWriter(always_on=True)

        if isinstance(exc, FileNotFoundError):
            pretty.write("W", "FILE NOT FOUND", str(exc), color=ORANGE)
            StartupCommons._die()

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
            StartupCommons._die()

        if isinstance(exc, NoVirtualEnvError):
            pretty.write(
                "E",
                "Virtual environment",
                f"{exc.args[0]}",
                color=RED,
            )
            StartupCommons._die()

        if isinstance(exc, TesseractPathError):
            pretty.write(
                "E",
                "Tesseract OCR",
                f"{exc.args[0]}",
                color=RED,
            )
            StartupCommons._die()

        if isinstance(exc, ConfigurationError):
            pretty.write(
                "E",
                "Config error",
                f"{exc.args[0]}",
                color=RED,
            )
            StartupCommons._die()

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
                StartupCommons._die()
            pretty.write(
                "E",
                "COMPLIANCE VIOLATION",
                f"Execution stopped due to compliance check: {exc.args[0]}",
                color=RED,
            )
            StartupCommons._die()

        if isinstance(exc, ExclusionsError):
            pretty.write(
                "E",
                "Exclusions",
                f"Execution stopped due to: {exc.args[0]}",
                color=RED,
            )
            StartupCommons._die()

        if isinstance(exc, ArgosPermissionError):
            pretty.write(
                "E",
                "ARGOS PERMISSION",
                f"Argos Translate runtime permission error: {exc}",
                color=RED,
            )
            StartupCommons._die()

        if isinstance(exc, InternetConnectionDisabledError):
            reason: str = str(exc) if str(exc) else ""
            pretty.write(
                "E",
                "INTERNET ACCESS DISABLED",
                f"{reason} "
                f"(Probably change internet access flags in Configuration/Config_Internet_Env.py)",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, HFDownloaderError):
            pretty.write(
                "E",
                "HF_Downloader",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, BackendUnavailableError):
            label = getattr(exc, "backend_name", "BACKEND")
            pretty.write(
                "E",
                label,
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, ModelLoadError):
            pretty.write(
                "E",
                "Model Load Error",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, DataProcessingError):
            pretty.write(
                "E",
                "Data Processing",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, LLMResultError):
            pretty.write(
                "E",
                "LLM Result Error",
                f"Execution stopped due to: {exc.args[0]}",
                color=ORANGE,
            )
            StartupCommons._die()

        if isinstance(exc, InvalidCollectionName):
            pretty.write(
                "E",
                "Collection name",
                f"'{exc.args[0]}' looks like a file-system path. "
                f"Please provide a plain collection name without '/', '\\\\', or ':' "
                f"(e.g. 'Test', not 'C:\\\\path\\\\to\\\\Test' or './Test').",
                color=RED,
            )
            StartupCommons._die()

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
        StartupCommons._die()
