# Standard library imports
import inspect
import logging
import os
import re
import shutil
import sys
# Third-party imports
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar, Optional, OrderedDict, Tuple, cast
from urllib.parse import ParseResult, urlparse

import pytesseract  # type: ignore[reportMissingTypeStubs]
import torch  # type: ignore[reportMissingImports]
from langdetect import \
    detect  # type: ignore[reportMissingTypeStubs]  # noqa: F401

# optional dependency: charset-normalizer is more modern; chardet works too
_USE_CHARSET_NORMALIZER: bool = False  # noqa
try:
    from charset_normalizer import \
        from_bytes as detect_charset  # type: ignore[reportMissingImports]

    _USE_CHARSET_NORMALIZER = True  # type: ignore[reportConstantRedefinition]
except Exception:
    import chardet  # type: ignore[reportMissingImports]

import hashlib

import nltk  # type: ignore[reportMissingTypeStubs]
import requests  # type: ignore[reportMissingModuleSource]
from docx import Document  # type: ignore[reportUnusedImport]  # for Word files
from langdetect import \
    detect  # type: ignore[reportMissingTypeStubs]  # noqa: F811,F401
from nltk.corpus import \
    stopwords  # type: ignore[reportMissingTypeStubs, reportUnusedImport]
from openpyxl import load_workbook  # type: ignore[reportUnusedImport]
from pdf2image import \
    convert_from_path  # type: ignore[reportMissingImports]; type: ignore[reportUnknownVariableType, reportUnusedImport]
from pptx import Presentation  # type: ignore[reportUnusedImport]

from Commons.Exceptions import (ConfigurationError, DeviceConfigurationError,
                                NoVirtualEnvError, TesseractPathError)
from Config.Config import Config
from Globals.CounterInstance import FailedCount, ProcessedCount
from Globals.Globals import Globals
from Gui.Colors import ORANGE, RED, RESET, VIOLET
from Gui.PrettyWriter import PrettyWriter


def truncate_for_print(content: str, width: int) -> str:
    """Truncate *content* to *width* chars; if longer, cut from the middle and insert '...'."""
    if len(content) <= width:
        return content
    chars = width - 3
    left = chars // 2
    return content[:left] + "..." + content[-(chars - left) :]


def _current_uid() -> str:
    """Return current uid if available; otherwise a stable marker for non-POSIX systems."""
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        try:
            return str(getuid())
        except Exception:
            pass
    return "n/a"


class Helpers:
    def prepare_argos_runtime_dirs(self) -> None:
        """Configure Argos to use writable user-owned XDG directories.

        Argos imports its settings module at import time and creates
        data/config/cache folders under XDG_* paths. On this container the
        default parent ~/.local/share is root-owned and not writable by the
        current user, so we redirect Argos to writable subfolders under the
        existing ~/.local/share/hf-cache, pip-cache and stanza_resources tree.
        """
        user_home = Path.home()
        base_dir = user_home / ".cache" / "argos"
        data_dir = base_dir / "data"
        config_dir = base_dir / "config"
        cache_dir = base_dir / "cache"

        for candidate in (data_dir, config_dir, cache_dir):
            try:
                # Create directory if it does not exist
                if not candidate.exists():
                    candidate.mkdir(parents=True, exist_ok=True)

                # Check if directory is writable
                if not os.access(candidate, os.W_OK):
                    parent = candidate.parent
                    stat_info = os.stat(parent) if parent.exists() else None
                    owner = (
                        f"{stat_info.st_uid}:{stat_info.st_gid}"
                        if stat_info
                        else "unknown"
                    )
                    mode = oct(stat_info.st_mode & 0o777) if stat_info else "unknown"

                    msg = (
                        "Argos Translate cannot write its runtime directories. "
                        f"Tried: {candidate}. Parent: {parent} "
                        f"(owner={owner}, mode={mode}, user={_current_uid()}, home={Path.home()}). "
                        "Fix filesystem permissions or allow Argos to use writable user-owned cache folders."
                    )

                    from Commons.Exceptions import ArgosPermissionError

                    raise ArgosPermissionError(msg)

            except PermissionError as exc:
                parent = candidate.parent
                stat_info = os.stat(parent) if parent.exists() else None
                owner = (
                    f"{stat_info.st_uid}:{stat_info.st_gid}" if stat_info else "unknown"
                )
                mode = oct(stat_info.st_mode & 0o777) if stat_info else "unknown"

                msg = (
                    "Argos Translate cannot write its runtime directories. "
                    f"Tried: {candidate}. Parent: {parent} "
                    f"(owner={owner}, mode={mode}, user={_current_uid()}, home={Path.home()}). "
                    f"Original error: {exc}. "
                    "Fix filesystem permissions or allow Argos to use writable user-owned cache folders."
                )

                from Commons.Exceptions import ArgosPermissionError

                raise ArgosPermissionError(msg) from exc

            except OSError as exc:
                raise RuntimeError(
                    f"Could not prepare Argos runtime directories for {candidate}: {exc}"
                ) from exc

        os.environ["XDG_DATA_HOME"] = str(data_dir)
        os.environ["XDG_CONFIG_HOME"] = str(config_dir)
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
        os.environ.setdefault("ARGOS_PACKAGES_DIR", str(data_dir / "packages"))

    # Utility class for common helper functions used throughout the application.
    # Utility class for common helper functions used throughout the application.
    _stopwords_cache: ClassVar[dict[str, list[str]]] = {}

    def get_tesseract_path(self) -> str:
        """Return the configured Tesseract binary path.

        Resolution order:
        1. ``TESSERACT_PATH`` environment variable
        2. ``TESSERACT_PATH`` config setting
        """
        return os.getenv("TESSERACT_PATH") or self.cfg.get_str(
            "TESSERACT_PATH", "", silent=True
        )

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

    # Internal pipeline fields that add noise for the LLM — never sent in context.
    _INTERNAL_METADATA_KEYS: frozenset[str] = frozenset(
        {
            "chroma_score",
            "chroma_sim",
            "dist",
            "rerank_score",
            "raw_rerank_score",
            "rrf_score",
            "bm25_score",
            "graph_score",
            "position",
            "retriever_sources",
            "FileHash",
            "chunk_id",
            "id",
        }
    )

    def format_document(self, doc: Any) -> str:
        """
        Formats a document's content and metadata as a string.

        A source header is emitted BEFORE the content so the LLM knows which
        file/page it is reading before processing the text.  Internal pipeline
        fields (scores, hashes, IDs) are stripped to reduce token noise.

        Args:
            doc (Document): The document to format.

        Returns:
            str: A formatted string containing the document's content and metadata.
        """
        md: dict[str, Any] = doc.metadata or {}
        file_name: str = str(md.get("FileName", "")).strip()
        page_num: Any = md.get("PageNumber")
        url: str = str(md.get("FilePath", "")).strip()

        # Build a concise source label placed BEFORE the content.
        if file_name:
            label = (
                f"(Source: {file_name}"
                + (f" | Page {page_num}" if page_num is not None else "")
                + ")"
            )
        elif url:
            label = f"(Source: {url})"
        else:
            label = "(Source: unknown)"

        formatted_doc: str = f"{label}\nContent:\n{doc.page_content}\nMetadata:\n"
        for key, value in md.items():
            if key not in self._INTERNAL_METADATA_KEYS:
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

    def find_provider_url(
        self,
        base_url: str,
        probe_path: str,
        generate_path: str,
        default_port: int,
        headers: dict[str, str],
        label: str,
    ) -> str | None:
        """Probe a provider endpoint, trying configured root first then fallback hosts/ports.

        Builds up to 6 candidate roots in this order:
          1–4. configured / localhost / 127.0.0.1 / host.docker.internal  — all with configured port
          5–6. localhost / host.docker.internal                            — with *default_port* (when it differs)

        Emits an "I" log line for every probe attempt. Returns the effective
        generate URL on first success, or None when all candidates are exhausted.
        """
        configured_base = self.normalize_base_url(base_url)
        parsed: ParseResult = urlparse(
            configured_base if "://" in configured_base else f"http://{configured_base}"
        )
        scheme: str = parsed.scheme or "http"
        configured_host: str = parsed.hostname or "localhost"
        configured_port: int | None = parsed.port

        seen: set[str] = set()
        candidate_roots: list[str] = []

        def _add(host: str, port: int | None) -> None:
            port_str = f":{port}" if port else ""
            root = f"{scheme}://{host}{port_str}"
            if root not in seen:
                seen.add(root)
                candidate_roots.append(root)

        _add(configured_host, configured_port)
        _add("localhost", configured_port)
        _add("127.0.0.1", configured_port)
        _add("host.docker.internal", configured_port)
        if configured_port != default_port:
            _add("localhost", default_port)
            _add("host.docker.internal", default_port)

        for i, root in enumerate(candidate_roots[:6]):
            probe_url = f"{root}{probe_path}"
            try:
                requests.get(probe_url, headers=headers, timeout=2).raise_for_status()
                self.pretty.write("I", label, f"Probing {probe_url} → OK")
                return f"{root}{generate_path}"
            except requests.RequestException as exc:
                short = str(exc).split("\n")[0][:120]
                self.pretty.write("I", label, f"Probing {probe_url} → failed: {short}")
                remaining = candidate_roots[:6]
                if i + 1 < len(remaining):
                    next_probe = f"{remaining[i + 1]}{probe_path}"
                    self.pretty.write(
                        "W", label, f"Trying next: {next_probe}", color=ORANGE
                    )

        return None

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
                    best = results.best()  # type: ignore[reportUnknownMemberType]
                    if best:
                        return str(best.result), str(best.encoding)  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            except Exception:
                pass
        else:
            # fallback to chardet
            try:
                info: dict[str, Any] = chardet.detect(data)  # type: ignore[reportPossiblyUnboundVariable]
                enc: str | None = str(info.get("encoding", "")) if info else None  # type: ignore[reportUnknownArgumentType]
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
        label: str = "",
    ) -> None:
        """
        Display a progress bar in place.

        Args:
            processed (int): Number of chunks processed so far.
            total (int): Total number of chunks.
            bar_length (int): Length of the progress bar in characters.
            print_newline (bool): If True, terminate the in-place bar with a newline.
            label (str): Step label displayed to the left of the bar.
        """
        if print_newline:
            sys.stdout.write("\n")
        else:
            percent: float = processed / total
            filled: int = int(bar_length * percent)
            bar: str = "#" * filled + "-" * (bar_length - filled)
            sys.stdout.write(
                f"\r   {VIOLET}{label:<30} [{bar}] {processed}/{total} ({percent:.0%}){RESET}"
            )
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

    def get_model_args(
        self, selector: str, *, role: str | None = None
    ) -> dict[str, Any]:
        """
        Retrieve all model attributes for a given role from config.

        ``selector`` may be either an active-impl slot name (e.g. ``"_ACTIVE_EMBED"``)
        or a direct impl key such as ``"ollama"`` / ``"vllm"``.
        When ``selector`` is an ``_ACTIVE_*`` slot, the role is derived from the
        selector name. For direct impl lookups, pass ``role`` explicitly (for
        example ``"_OLLAMA"`` or ``"_VLLM"``).

        Returns a shallow copy of the config entry with every key
        (MODEL, FRIENDLY_NAME, LICENSE, TAG, BASE_URL, etc.)
        plus the synthetic ``local_files_only`` flag derived from
        the HF_HUB_OFFLINE environment variable and normalised
        convenience aliases (model_name, friendly_name, source).
        """
        if selector.startswith("_ACTIVE"):
            impl = self.cfg.get_str(selector)
            role = role or selector[len("_ACTIVE") :]
        else:
            impl = selector
            if role is None:
                raise ValueError(
                    "get_model_args expects an _ACTIVE_* selector or an explicit lookup key"
                )
        models: dict[str, Any] = self.cfg.get_dict("_MODELS")
        if impl not in models:
            raise ValueError(f"Unknown impl: {impl!r} (from {selector!r})")
        impl_set = models[impl]
        if role not in impl_set:
            raise ValueError(f"Role {role!r} not found in impl {impl!r}")
        entry: dict[str, Any] = dict(impl_set[role])  # shallow copy

        # Normalise common keys for backward compat with callers.
        # Some LLM roles keep endpoint-specific model names under MODEL_OLLAMA
        # / MODEL_VLLM, so expose the active endpoint's value as MODEL/model_name.
        endpoint = str(self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()
        if "MODEL_OLLAMA" in entry and "MODEL_VLLM" in entry:
            selected_model = (
                entry.get("MODEL_OLLAMA")
                if endpoint == "ollama"
                else entry.get("MODEL_VLLM")
            )
            entry.setdefault("MODEL", selected_model or entry.get("MODEL", ""))
            entry.setdefault("model_name", selected_model or entry.get("MODEL", ""))
        else:
            entry.setdefault("model_name", entry.get("MODEL", ""))

        entry.setdefault("friendly_name", entry.get("FRIENDLY_NAME", ""))
        entry.setdefault("source", entry.get("SOURCE", ""))
        scoped_hf_api_key = str(entry.get("HF_API_KEY") or "").strip()
        global_hf_api_key = str(
            self.cfg.get_str("_HF_API_KEY", "", silent=True) or ""
        ).strip()
        entry["hf_api_key"] = scoped_hf_api_key or global_hf_api_key
        entry["local_files_only"] = os.environ.get("HF_HUB_OFFLINE", "1") == "1"

        # Only expose revision when it is a valid 40-char hex commit hash
        raw_rev: str = entry.get("REVISION") or ""
        if raw_rev and re.fullmatch(r"[0-9a-f]{40}", raw_rev):
            entry["revision"] = raw_rev
        elif "revision" not in entry:
            entry["revision"] = None
        return entry

    def get_active_endpoint_args(self) -> dict[str, Any]:
        """Return the active LLM endpoint config for the current runtime selector."""
        endpoint = str(self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()
        return self.get_model_args(endpoint, role=f"_{endpoint.upper()}")

    def get_keybert_config(self) -> dict[str, Any]:
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

    def get_chroma_config_slot(self) -> str:
        """
        Return the dot-notation prefix for the active chroma retrieve variant.

        Reads ``_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG`` (e.g. ``"THOROUGH"``) and
        returns ``"_CHROMA_EMBED_AND_RETRIEVE_PARAMS.THOROUGH"``.
        """
        active: str = self.cfg.get_str(
            "_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG"
        )
        return f"_CHROMA_EMBED_AND_RETRIEVE_PARAMS.{active}"

    def get_chunker_config_slot(self) -> str:
        """
        Return the dot-notation prefix for the active chunker variant.

        Reads the DEFAULT chunker from the active ``_CHUNK_STRATEGY`` profile
        and returns e.g. ``"_CHUNKERS.SEMANTIC"``.
        """
        profile: str = self.cfg.get_str("_ACTIVE_CHUNKER_CONFIG")
        active: str = self.cfg.get_str(f"_CHUNK_STRATEGY.{profile}.DEFAULT", "SEMANTIC")
        return f"_CHUNKERS.{active}"

    def get_chunker_max_size(self) -> int:
        """
        Return the effective maximum chunk size for the active chunker.

        ``RECURSIVE`` → ``CHUNK_SIZE``,
        ``SEMANTIC`` / ``SENTENCE_WINDOW`` → ``MAX_CHUNK_SIZE``.
        Used for embedding truncation and model ``max_length`` parameters.
        """
        slot: str = self.get_chunker_config_slot()
        profile: str = self.cfg.get_str("_ACTIVE_CHUNKER_CONFIG")
        active: str = self.cfg.get_str(f"_CHUNK_STRATEGY.{profile}.DEFAULT", "SEMANTIC")
        if active in ("SEMANTIC", "SENTENCE_WINDOW"):
            return self.cfg.get_int(f"{slot}.MAX_CHUNK_SIZE")
        return self.cfg.get_int(f"{slot}.CHUNK_SIZE")

    def get_compliance_config_slot(self, stage: str) -> str:
        """
        Build config slot for compliance checks based on detection type and friendly name.
        Pure config lookup — no model loading.
        """
        check_type: str = self.cfg.get_str("_ACTIVE_DETECTION_CONFIG")
        friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        self.require_set(check_type=check_type, friendly_name=friendly_name)
        return f"_BANNED_DETECT.{check_type}.{friendly_name}.{stage}"

    def get_banned_phrases_config_slot(self) -> list[str]:
        """
        Retrieve banned phrases from config slot.
        Pure config lookup — no model loading.
        """
        slot: str = self.cfg.get_str("_ACTIVE_BANNED_CONFIG")
        slot = f"{slot}.BANNED"
        return self.cfg.get_list(slot)

    def get_masking_regexes_config_slot(self) -> dict[str, Any]:
        """
        Retrieve masking regexes from config slot.
        Pure config lookup — no model loading.
        """
        slot: str = self.cfg.get_str("_ACTIVE_MASKING_CONFIG")
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

        print()  # newline after the last dot — always printed regardless of debug level

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

        # Activation-script env var (set by source .venv/bin/activate on Linux/macOS
        # and .venv\Scripts\activate on Windows)
        elif os.environ.get("VIRTUAL_ENV"):
            return True

        # --- Enforcement ---------------------------------------------------------
        if required and not in_venv:
            msg = (
                "Execution blocked: this framework must run inside a Python virtual environment. "
                "Reason: deterministic dependencies, reproducible masking behavior, and "
                "audit-friendly isolation."
            )
            self.pretty.write("E", "Virtual environment", msg)
            self.pretty.write(
                "E",
                "Virtual environment",
                "To fix: activate the project .venv (source .venv/bin/activate on Linux/macOS "
                "or .venv\\Scripts\\activate on Windows).",
            )
            raise NoVirtualEnvError(msg)

        return False

    def bit_to_dtype(self, bits: int = 32) -> Any:
        """
        Convert bit-width to torch dtype.
        """
        dtype_map: dict[int, Any] = {
            32: torch.float32,  # type: ignore[reportUnknownMemberType]
            16: torch.float16,  # type: ignore[reportUnknownMemberType]
            8: torch.qint8,  # type: ignore[reportUnknownMemberType]
        }
        if bits not in dtype_map:
            if bits == 4:
                info = "PyTorch core doesn’t ship 4-bit quant. Use bitsandbytes or GPTQ instead."
                self.pretty.write("E", "Device", info)
                raise RuntimeError(info)
            raise ValueError(f"unsupported bit-width: {bits}")

        target_dtype: Any = dtype_map[bits]
        return target_dtype

    def configure_tesseract(self) -> str:
        """
        Resolve a usable Tesseract OCR executable path for the current environment.

        Supports OS-aware path configuration in format "windows_path|linux_path".
        Automatically selects the appropriate path based on sys.platform.
        """

        configured_path = self.get_tesseract_path().strip()

        if not configured_path:
            raise TesseractPathError(
                "TESSERACT_PATH is not configured. "
                "Set TESSERACT_PATH in Configuration/Config_Internet_Env.py "
                "using the format: 'windows_path|linux_path'"
            )

        # Parse OS-aware path format: "windows_path|linux_path"
        if "|" in configured_path:
            parts = configured_path.split("|", 1)
            if sys.platform == "win32":
                platform_path = parts[0].strip()
            else:  # Linux, macOS
                platform_path = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Single path (backward compatible)
            platform_path = configured_path

        if not platform_path:
            raise TesseractPathError(
                f"TESSERACT_PATH does not contain a path for the current platform ({sys.platform}). "
                "Update TESSERACT_PATH in Configuration/Config_Internet_Env.py"
            )

        # Resolve the path
        resolved_path: str | None = None
        path_candidate = Path(platform_path).expanduser()
        if path_candidate.exists():
            resolved_path = str(path_candidate)
        else:
            # Try PATH lookup if it's just a command name
            resolved_from_path = shutil.which(platform_path)
            if resolved_from_path:
                resolved_path = resolved_from_path

        if not resolved_path:
            if sys.platform == "win32":
                raise TesseractPathError(
                    f"Tesseract executable not found at: {platform_path}\n"
                    "  Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  Then update TESSERACT_PATH in Configuration/Config_Internet_Env.py\n"
                    '  Example: r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe|/usr/bin/tesseract"'
                )
            else:
                raise TesseractPathError(
                    f"Tesseract executable not found at: {platform_path}\n"
                    "  Install Tesseract using your package manager (e.g., apt-get install tesseract-ocr)\n"
                    "  Then update TESSERACT_PATH in Configuration/Config_Internet_Env.py\n"
                    '  Example: r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe|/usr/bin/tesseract"'
                )

        self.tesseract_path = resolved_path
        pytesseract.pytesseract.tesseract_cmd = resolved_path  # type: ignore[reportUnknownMemberType]
        self.pretty.write(
            "I",
            "Tesseract OCR",
            f"Tesseract is ready. Using: {resolved_path}",
            max_line_length=999,
        )
        return resolved_path
