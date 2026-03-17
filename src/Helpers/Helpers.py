# Standard library imports
import inspect
import logging
import os
import platform
import re
import sys
# Third-party imports
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar, Optional, OrderedDict, Tuple, cast
from urllib.parse import ParseResult, urlparse

import pytesseract  # type: ignore[reportMissingTypeStubs]
import torch
from langdetect import \
    detect  # type: ignore[reportMissingTypeStubs]  # noqa: F401

# optional dependency: charset-normalizer is more modern; chardet works too
_USE_CHARSET_NORMALIZER: bool = False  # noqa
try:
    from charset_normalizer import from_bytes as detect_charset

    _USE_CHARSET_NORMALIZER = True  # type: ignore[reportConstantRedefinition]
except Exception:
    import chardet

import hashlib

import nltk  # type: ignore[reportMissingTypeStubs]
from docx import Document  # type: ignore[reportUnusedImport]  # for Word files
from langdetect import \
    detect  # type: ignore[reportMissingTypeStubs]  # noqa: F811,F401
from nltk.corpus import \
    stopwords  # type: ignore[reportMissingTypeStubs, reportUnusedImport]
from openpyxl import load_workbook  # type: ignore[reportUnusedImport]
from pdf2image import \
    convert_from_path  # type: ignore[reportUnknownVariableType, reportUnusedImport]
from pptx import Presentation  # type: ignore[reportUnusedImport]

from Commons.Exceptions import ConfigurationError, DeviceConfigurationError
from Config.Config import Config
from Globals.CounterInstance import FailedCount, ProcessedCount
from Globals.Globals import Globals
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter


class Helpers:
    # Utility class for common helper functions used throughout the application.
    _stopwords_cache: ClassVar[dict[str, list[str]]] = {}

    def __init__(
        self, *, cfg: "Config | None" = None, pretty: "PrettyWriter | None" = None
    ):
        # Instantiate helper objects/singletons as instance attributes.
        # Cache for stopwords per language.

        self.failedCounter: FailedCount = FailedCount()
        self.processedCounter: ProcessedCount = ProcessedCount()
        self.globalsInstance: Globals = Globals()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.custom_ntlk_data_dir: str = self.cfg.get_str("_CUSTOM_NLTK_DATA_DIRECTORY")
        # Ensure that our custom directory is in nltk.data.path.
        nltk_path: list[str] = nltk.data.path  # type: ignore[reportUnknownMemberType]
        if self.custom_ntlk_data_dir not in nltk_path:
            nltk_path.append(self.custom_ntlk_data_dir)  # type: ignore[reportUnknownMemberType]

        self.label_alias: dict[str, Any] = self.cfg.get_dict("_LABEL_ALIAS")

    def check_cpu_and_bits(self) -> None:
        """
        Check for invalid CPU/bit configuration and raise error if detected.
        """
        cpu: bool = self.cfg.get_bool("USE_CPU")
        bits: int = self.cfg.get_int("EMBEDDER_BITS")
        if cpu and bits == 16:
            self.pretty.write(
                "E",
                "Device settings",
                f"USE_CPU {cpu} cannot be used with EMBEDDER_BITS: {bits}",
                color=RED,
            )
            raise DeviceConfigurationError(
                f"Invalid device configuration: CPU mode cannot be used with 16-bit embeddings"
            )

    def short_hash(self, value: str, length: int = 12) -> str:
        """
        Return a short hash (hex) of a string value.
        """
        h = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return h[:length]

    # def isOfficeComponentInstalled(self, component: str) -> Tuple[bool, Optional[str]]:
    #     """
    #     Checks the Windows Registry for installation paths of Microsoft Office components.
    #     Supported components: Word, Excel, PowerPoint.
    #     Returns a tuple (installed, path) where installed is True if found, and path is the installation path.
    #     """
    #     office_versions = ["16.0", "15.0", "14.0"]
    #     registry_base = r"SOFTWARE\Microsoft\Office"

    #     for version in office_versions:
    #         reg_path = rf"{registry_base}\{version}\{component}\InstallRoot"

    #         try:
    #             key = winreg.OpenKey(
    #                 winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_READ
    #             )
    #             install_path, _ = winreg.QueryValueEx(key, "Path")
    #             winreg.CloseKey(key)
    #             return True, install_path
    #         except FileNotFoundError:
    #             continue
    #         except Exception as e:
    #            self.pretty.write("E", "Office Installed", f"Error: {e}")
    #            return False, None

    #     return False, None

    def format_document(self, doc: Any) -> str:
        """
        Formats a document’s content and metadata as a string.


        Args:
            doc (Document): The document to format.

        Returns:
            str: A formatted string containing the document’s content and metadata.
        """
        formatted_doc: str = f"Content:\n{doc.page_content}\nMetadata:\n"
        for key, value in doc.metadata.items():
            formatted_doc += f"  {key}: {value}\n"
        return formatted_doc

    def normalize_base_url(self, url: str, default_scheme: str = "http") -> str:
        """
        Normalize a base URL, ensuring scheme and extracting host/port.

        Turn any input like
        - "localhost:11434/api/models"
        into
        - "http://localhost:11434"
        """
        # 1) Ensure there's a scheme so urlparse.netloc works
        if "://" not in url:
            url = f"{default_scheme}://{url}"

        parsed: ParseResult = urlparse(url)
        scheme: str = parsed.scheme
        host: str = parsed.hostname or ""
        port: int | None = parsed.port

        # 2) If no explicit port, check if the first path segment is pure digits
        if port is None and parsed.path:
            first_seg: str = parsed.path.lstrip("/").split("/", 1)[0]
            if first_seg.isdigit():
                port = int(first_seg)

        # 3) Reassemble
        port_part: str = f":{port}" if port else ""
        return f"{scheme}://{host}{port_part}"

    def setup_logger(
        self, logname: str | None = None, level: str = "DEBUG", show_log: bool = False
    ) -> logging.Logger:
        """
        Set up a logger for the application, optionally with console output.
        """
        if logname is None:
            logname = self.cfg.get_str("_FRIENDLY_NAME")
        log_path: str = os.path.join(
            self.cfg.get_str("_LOG_DIRECTORY"), logname + ".log"
        )

        logger: logging.Logger = logging.getLogger(logname)
        logger.propagate = False  # Prevent messages from bubbling up to root

        # Convert string level to numeric level, defaulting to DEBUG if invalid
        numeric_level: int = getattr(logging, level.upper(), logging.DEBUG)
        logger.setLevel(numeric_level)

        # Only add handlers once per process
        if not logger.handlers:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            file_handler: logging.FileHandler = logging.FileHandler(
                log_path, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(file_handler)

            # Optional console output
            if show_log:
                console_handler: logging.StreamHandler[Any] = logging.StreamHandler()
                console_handler.setFormatter(
                    logging.Formatter("%(levelname)s: %(message)s")
                )
                logger.addHandler(console_handler)
        return logger

    def sanitize_path_component(self, value: str) -> str:
        """
        Sanitize a string for use as a filesystem path component.
        """
        if not isinstance(value, str):  # type: ignore[reportUnnecessaryIsInstance]
            value = str(value)

        # Convert to lowercase
        value = value.lower()

        # Remove blanks (spaces, tabs, etc.)
        value = re.sub(r"\s+", "", value)

        # Remove invalid path characters (Windows + POSIX common set)
        # Characters: \ / : * ? " < > |
        invalid_chars: str = r'[\\/:*?"<>|]'
        value = re.sub(invalid_chars, "", value)
        return value

    def in_venv(self) -> bool:
        """
        Return True if the current Python process appears to be running inside a virtual environment.

        Heuristics used (in roughly decreasing reliability):
        - sys.base_prefix != sys.prefix or sys.base_exec_prefix != sys.exec_prefix (standard venv detection)
        - presence of sys.real_prefix (legacy virtualenv)
        - existence of VIRTUAL_ENV environment variable
        - presence of CONDA_PREFIX environment variable (treats conda envs as virtual environments)
        - presence of pyvenv.cfg in sys.prefix (common for venv-created envs)

        This function is intentionally prefers conservative signals that match how upstream libraries detect venvs.
        """
        try:
            # Primary check (works for venv and many virtualenv setups)
            if getattr(sys, "base_prefix", None) != getattr(sys, "prefix", None):
                return True
            if getattr(sys, "base_exec_prefix", None) != getattr(
                sys, "exec_prefix", None
            ):
                return True

            # Legacy virtualenv attribute added by virtualenv
            if hasattr(sys, "real_prefix"):
                return True

            # Environment variables commonly set by activation scripts
            if "VIRTUAL_ENV" in os.environ:
                return True

            # Treat conda environments as virtual environments
            if "CONDA_PREFIX" in os.environ or "CONDA_DEFAULT_ENV" in os.environ:
                return True

            # Look for pyvenv.cfg in the interpreter prefix directory
            prefix_path: Optional[Path] = None
            try:
                prefix_path = Path(sys.prefix)
            except Exception:
                prefix_path = None

            if prefix_path and (prefix_path / "pyvenv.cfg").is_file():
                return True

            return False

        except Exception:
            # Fail closed: if detection logic errors for some reason, assume not in venv
            return False

    def safe_decode_to_unicode(
        self, data: bytes | str, assumed_utf8: bool = True
    ) -> Tuple[str, str]:
        """
        Decode bytes to a Unicode string and return (decoded_text, used_encoding).

        Strategy:
        1. If assumed_utf8, try utf-8 decode first (fast common path).
        2. If that fails or data appears not to be bytes, run charset detector (charset-normalizer or chardet).
        3. Fall back to latin-1 as a last resort to avoid raising.
        """
        if not isinstance(data, (bytes, bytearray)):
            # already text-like
            text = str(data)
            return text, "str-converted"

        # fast path: try utf-8
        if assumed_utf8:
            try:
                return data.decode("utf-8"), "utf-8"
            except Exception:
                pass

        # try charset-normalizer if available (preferred)
        if _USE_CHARSET_NORMALIZER:
            try:
                results = detect_charset(data)  # type: ignore[reportPossiblyUnboundVariable]
                if results:
                    best = results.best()
                    if best:
                        return str(best.result), str(best.encoding)  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            except Exception:
                pass
        else:
            # fallback to chardet
            try:
                info: dict[str, Any] = chardet.detect(data)  # type: ignore[reportPossiblyUnboundVariable]
                enc: str | None = str(info.get("encoding", "")) if info else None
                if enc:
                    try:
                        return data.decode(enc), enc
                    except Exception:
                        pass
            except Exception:
                pass

        # last resort: latin-1 (never fails)
        try:
            return data.decode("latin-1"), "latin-1"
        except Exception:
            # extremely defensive fallback
            return data.decode("utf-8", errors="replace"), "utf-8-replace"

    def show_progress(
        self,
        processed: int,
        total: int,
        bar_length: int = 40,
        print_newline: bool = False,
    ) -> None:
        """
        Display a progress bar in place.

        Args:
            processed (int): Number of chunks processed so far.
            total (int): Total number of chunks.
            bar_length (int): Length of the progress bar in characters.
        """
        if print_newline:
            sys.stdout.write(f"\n")
        else:
            percent: float = processed / total
            filled: int = int(bar_length * percent)
            bar: str = "#" * filled + "-" * (bar_length - filled)
            sys.stdout.write(f"\r[{bar}] {processed}/{total} ({percent:.0%})")
        sys.stdout.flush()

    def make_ordered_dict(self, raw: Any) -> OrderedDict[str, bool]:
        """
        Normalize input into an OrderedDict[str, bool] for algorithm processing.

        Accepts:
        - dict: keys -> truthiness of values
        - list/tuple of str: each name -> True
        - list/tuple of pairs: (name, flag) -> bool(flag)
        - anything else -> empty OrderedDict

        Returns the OrderedDict and also assigns it to self.algos_to_process.
        """
        if isinstance(raw, dict):
            raw_dict: dict[str, Any] = cast(dict[str, Any], raw)
            algos: OrderedDict[str, bool] = OrderedDict(
                (str(k), bool(v)) for k, v in raw_dict.items()
            )
        elif isinstance(raw, (list, tuple)):
            raw_seq: list[Any] = [x for x in raw]  # type: ignore[reportUnknownArgumentType]
            # list of names
            if all(isinstance(x, str) for x in raw_seq):
                algos = OrderedDict((str(k), True) for k in raw_seq)
            else:
                pairs: list[Tuple[str, bool]] = []
                for entry in raw_seq:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:  # type: ignore[reportUnknownArgumentType]
                        entry_seq: list[Any] = list(entry)  # type: ignore[reportUnknownArgumentType]
                        pairs.append((str(entry_seq[0]), bool(entry_seq[1])))
                algos = OrderedDict(pairs)
        else:
            algos = OrderedDict()

        self.algos_to_process = algos
        return algos

    def get_model_args(self, what: str) -> dict[str, Any]:
        """
        Retrieve all model attributes for a given role from config.

        Uses the impl constant (e.g. _EMBED="snowflake") to navigate
        the nested _MODELS hierarchy: _MODELS[impl][role].

        Returns a shallow copy of the config entry with every key
        (MODEL, FRIENDLY_NAME, LICENSE, TAG, BASE_URL, etc.)
        plus the synthetic ``local_files_only`` flag derived from
        the HF_HUB_OFFLINE environment variable and normalised
        convenience aliases (model_name, friendly_name, source).
        """
        impl: str = self.cfg.get_str(what)
        models: dict[str, Any] = self.cfg.get_dict("_MODELS")
        if impl not in models:
            raise ValueError(f"Unknown impl: {impl!r} (from {what!r})")
        impl_set = models[impl]
        if what not in impl_set:
            raise ValueError(f"Role {what!r} not found in impl {impl!r}")
        entry: dict[str, Any] = dict(impl_set[what])  # shallow copy

        # Normalise common keys for backward compat with callers
        entry.setdefault("model_name", entry.get("MODEL", ""))
        entry.setdefault("friendly_name", entry.get("FRIENDLY_NAME", ""))
        entry.setdefault("source", entry.get("SOURCE", ""))
        entry["local_files_only"] = os.environ.get("HF_HUB_OFFLINE", "1") == "1"

        # Only expose revision when it is a valid 40-char hex commit hash
        raw_rev: str = entry.get("REVISION") or ""
        if raw_rev and re.fullmatch(r"[0-9a-f]{40}", raw_rev):
            entry["revision"] = raw_rev
        elif "revision" not in entry:
            entry["revision"] = None
        return entry

    def _get_keybert_config(self) -> dict[str, Any]:
        """Return the active _KEY_BERT config dict.

        Handles both layouts transparently:
        - **Nested** (DocClassify): ``_ACTIVE_KEYBERT_CONFIG = "STRICT"``
          \u2192 ``_KEY_BERT.STRICT.TOP_N_FIRST``, etc.
        - **Flat** (RAGChat / RAGLoad): ``_KEY_BERT.TOP_N_FIRST`` directly.
        """
        active: str = self.cfg.get_str("_ACTIVE_KEYBERT_CONFIG", "", silent=True)
        if active:
            return self.cfg.get_dict(f"_KEY_BERT.{active}")
        return self.cfg.get_dict("_KEY_BERT")

    def get_compliance_config_slot(self, stage: str) -> str:
        """
        Build config slot for compliance checks based on detection type and friendly name.
        Pure config lookup — no model loading.
        """
        check_type: str = self.cfg.get_str("_DETECTION_CONFIG")
        friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        self.require_set(check_type=check_type, friendly_name=friendly_name)
        return f"_BANNED_DETECT.{check_type}.{friendly_name}.{stage}"

    def get_banned_phrases_config_slot(self) -> list[str]:
        """
        Retrieve banned phrases from config slot.
        Pure config lookup — no model loading.
        """
        slot: str = self.cfg.get_str("_BANNED_CONFIG")
        slot = f"{slot}.BANNED"
        return self.cfg.get_list(slot)

    def get_masking_regexes_config_slot(self) -> dict[str, Any]:
        """
        Retrieve masking regexes from config slot.
        Pure config lookup — no model loading.
        """
        slot: str = self.cfg.get_str("_MASKING_CONFIG")
        slot = f"{slot}.MASKING_REGEXES"
        return self.cfg.get_dict(slot)

    def require_set(self, **kwargs: Any) -> None:
        """
        Require that all provided kwargs are set (not UNSET), else raise error.
        """
        UNSET = object()
        calling_function_name = inspect.stack()[1].function
        missing = [name for name, value in kwargs.items() if value is UNSET]
        if missing:
            self.pretty.write(
                "E",
                calling_function_name,
                f"Missing required configuration values: {', '.join(missing)}",
                color=RED,
            )
            raise ConfigurationError(
                f"Missing required configuration values: {', '.join(missing)}"
            )

    def pretty_sleep(self, seconds: int, message: str = "", symbol: str = ".") -> None:
        """
        Sleep with a visible per‑second progress indicator.
        """
        print(f"Sleeping for {seconds} seconds ", end="", flush=True)

        for _i in range(seconds):
            time.sleep(1)
            print(symbol, end="", flush=True)

        self.pretty.write("N", "", "")  # final newline

    def get_label_alias(self, algo: str) -> str:
        """
        Get alias for algorithm label from config.
        """
        return self.label_alias.get(algo) or algo

    def replace_keys_with_aliases(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        Return a new dict with keys replaced by their alias names.
        """
        transformed: dict[str, Any] = {}

        for k, v in input.items():
            alias: str = self.get_label_alias(k)
            transformed[alias] = v
        return transformed

    def is_in_venv(self, required: bool = True) -> bool:
        """
        Detects whether the current Python interpreter is running inside a virtual
        environment (venv, virtualenv, or conda). If `required=True`, raises a clear,
        compliance-friendly error when not inside a venv.

        Returns:
            bool: True if running inside a virtual environment, False otherwise.
        """
        # --- Detection logic -----------------------------------------------------
        in_venv: bool = False

        # Standard venv / virtualenv
        if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
            return True

        # Legacy virtualenv
        elif hasattr(sys, "real_prefix"):
            return True

        # Conda environments
        elif os.environ.get("CONDA_PREFIX"):
            return True

        # --- Enforcement ---------------------------------------------------------
        if required and not in_venv:
            self.pretty.write(
                "E",
                "Virtual environment",
                "Execution blocked: this framework must run inside a Python virtual environment. "
                "Reason: deterministic dependencies, reproducible masking behavior, and "
                "audit-friendly isolation.",
            )
            self.pretty.write(
                "E",
                "Virtual environment",
                "To fix: .venv\\Scripts\\activate # on Windows",
            )

        return False

    def bit_to_dtype(self, bits: int = 32) -> torch.dtype:
        """
        Convert bit-width to torch dtype.
        """
        dtype_map: dict[int, torch.dtype] = {
            32: torch.float32,
            16: torch.float16,
            8: torch.qint8,
        }
        if bits not in dtype_map:
            if bits == 4:
                info = "PyTorch core doesn’t ship 4-bit quant. Use bitsandbytes or GPTQ instead."
                self.pretty.write("E", "Device", info)
                raise RuntimeError(info)
            raise ValueError(f"unsupported bit-width: {bits}")

        target_dtype: torch.dtype = dtype_map[bits]
        return target_dtype

    def configure_tesseract(self) -> None:
        """
        Configure the Tesseract OCR executable path depending on the OS.
        """

        system: str = platform.system()

        if system == "Windows":
            default_path: Path = Path(self.cfg.get_str("_TESSERACT_PATH"))

            if not default_path.exists():
                if not default_path.exists():
                    raise FileNotFoundError(
                        f"Tesseract executable not found at the configured path:\n"
                        f"  {default_path}\n\n"
                        "Tesseract OCR must be installed separately and is not included with this project. "
                        "If Tesseract is installed in a different location, update the path in "
                        "Config_Global.py accordingly."
                    )

            pytesseract.pytesseract.tesseract_cmd = str(default_path)
            self.pretty.write(
                "I",
                "Tesseract OCR",
                f"Tesseract is ready. Configured Tesseract path: {default_path}",
            )
        else:
            # On macOS/Linux, assume tesseract is on PATH.
            # Users can override here if needed.
            pytesseract.pytesseract.tesseract_cmd = "tesseract"
