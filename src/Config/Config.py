import inspect
import json
import os
import pprint
import sys
import threading
from typing import Any, Optional, Tuple, Union, cast

import Configuration.Config_Banned_Content as Config_Banned_Content
import Configuration.Config_Banned_Detection as Config_Banned_Detection
import Configuration.Config_Banned_Prompts as Config_Banned_Prompts
import Configuration.Config_DocClassify as Config_DocClassify
import Configuration.Config_Global as Config_Global
import Configuration.Config_Languages as Config_Languages
import Configuration.Config_Load_Chunkers as Config_Load_Chunkers
import Configuration.Config_Load_Retrievers as Config_Load_Retrievers
import Configuration.Config_Models as Config_Models
import Configuration.Config_RAGChat as Config_RAGChat
import Configuration.Config_RAGChat_Display as Config_RAGChat_Display
import Configuration.Config_RAGChat_Prompts as Config_RAGChat_Prompts
import Configuration.Config_RAGChat_Rewrite as Config_RAGChat_Rewrite
import Configuration.Config_RAGChat_Strategies as Config_RAGChat_Strategies
import Configuration.Config_RAGChatService as Config_RAGChatService
import Configuration.Config_RAGLoad as Config_RAGLoad
import Configuration.Config_WebSearch as Config_WebSearch
from Commons.Exceptions import ConfigPathError
from Commons.SingletonMixin import SingletonMixin
from Config.CliOverridePolicy import (app_uses_load_retrievers_cli_scope,
                                      build_allowed_cli_overrides,
                                      normalize_cli_overrides)
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter

config_modules = {
    "Config_Banned_Detection": Config_Banned_Detection,
    "Config_Banned_Content": Config_Banned_Content,
    "Config_Banned_Prompts": Config_Banned_Prompts,
    "Config_Load_Retrievers": Config_Load_Retrievers,
    "Config_Load_Chunkers": Config_Load_Chunkers,
    "Config_Models": Config_Models,
    "Config_RAGChat": Config_RAGChat,
    "Config_RAGChatService": Config_RAGChatService,
    "Config_RAGLoad": Config_RAGLoad,
    "Config_DocClassify": Config_DocClassify,
    "Config_Global": Config_Global,
    "Config_Languages": Config_Languages,
    "Config_WebSearch": Config_WebSearch,
}
# Case-insensitive lookup table (Windows preserves typed casing in sys.argv[0])
_config_modules_lower = {k.lower(): v for k, v in config_modules.items()}


class Config(SingletonMixin):

    @staticmethod
    def _collect_indirect_aliases(
        value: Any,
        path: str,
    ) -> list[tuple[str, str]]:
        """Recursively collect '$'-prefixed alias strings from a config value."""
        aliases: list[tuple[str, str]] = []
        if isinstance(value, str):
            if value.startswith("$"):
                aliases.append((path, value))
            return aliases

        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_path = f"{path}.{child_key}" if path else str(child_key)
                aliases.extend(
                    Config._collect_indirect_aliases(child_value, child_path)
                )
            return aliases

        if isinstance(value, list):
            for index, child_value in enumerate(value):
                child_path = f"{path}[{index}]"
                aliases.extend(
                    Config._collect_indirect_aliases(child_value, child_path)
                )
            return aliases

        return aliases

    def _validate_indirect_lookup_scope(
        self,
        module_label: str,
        module_values: dict[str, Any],
    ) -> None:
        """Reject aliases that target top-level keys outside the same module."""
        top_level_keys = set(module_values.keys())
        for key, value in module_values.items():
            aliases = self._collect_indirect_aliases(value, key)
            for alias_path, alias_target in aliases:
                target = alias_target[1:]
                target_top_level = target.split(".", 1)[0]
                if target_top_level not in top_level_keys:
                    raise ConfigPathError(
                        "Invalid cross-module indirect alias in "
                        f"{module_label}: '{alias_path}' -> '{alias_target}'. "
                        "The target top-level slot must be defined in the same "
                        f"module; missing '{target_top_level}'."
                    )

    def __init__(self, args: Any = None, source: str | None = None) -> None:
        # avoid re-init
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._lock = threading.RLock()

        # detect which program-specific constants module to use
        config_name: str = (
            "Config_" + os.path.splitext(os.path.basename(sys.argv[0]))[0]
        )
        self.cfgPy: Any = _config_modules_lower.get(config_name.lower())
        self.globalConfigPy: Any = Config_Global
        self.cfgLoadRetrievers: Any = Config_Load_Retrievers
        self.cfgLoadChunkers: Any = Config_Load_Chunkers
        self.cfgLanguages: Any = Config_Languages
        self.cfgModels: Any = Config_Models
        self.cfgRagchatCore: Any = Config_RAGChat
        self.cfgRagchatDisplay: Any = Config_RAGChat_Display
        self.cfgRagchatPrompts: Any = Config_RAGChat_Prompts
        self.cfgRagchatRewrite: Any = Config_RAGChat_Rewrite
        self.cfgRagchatStrategies: Any = Config_RAGChat_Strategies
        self.cfgBannedDetection: Any = Config_Banned_Detection
        self.cfgBannedContent: Any = Config_Banned_Content
        self.cfgBannedPrompts: Any = Config_Banned_Prompts
        self.cfgWebSearch: Any = Config_WebSearch

        # CLI overrides are restricted to Config_Global + active app slots.
        # For RAGLoad/RAGChat/RAGChatService, include Config_Load_Retrievers.
        load_retrievers_module: Any | None = None
        if app_uses_load_retrievers_cli_scope(self.cfgPy):
            load_retrievers_module = self.cfgLoadRetrievers

        self._allowed_cli_override_slots: set[str] = set(
            build_allowed_cli_overrides(
                self.globalConfigPy,
                self.cfgPy,
                load_retrievers_module,
            ).keys()
        )

        # capture CLI args as a validated uppercase dict
        self.args: dict[str, Any] = {}
        if args:
            raw_args = (
                cast(dict[str, Any], args)
                if isinstance(args, dict)
                else cast(dict[str, Any], vars(args))
            )
            self.args = normalize_cli_overrides(
                raw_args,
                self._allowed_cli_override_slots,
            )

        # final merged config
        self.cfg: dict[str, Any] = {}

        # load either from JSON or from constants
        if source:
            self.load(source)
        else:
            self._load_from_constants()

        # Avoid recursion since PrettyWriter also uses Config
        self.pretty: PrettyWriter = PrettyWriter()

    def _load_from_constants(self):
        # Priority order (lowest → highest; last update wins in the merge below):
        # 1) Config_Global  — shared defaults
        glob_raw = {
            k: getattr(self.globalConfigPy, k)
            for k in dir(self.globalConfigPy)
            if k.isupper()
        }

        # 2) Config_Load_Retrievers — collection and retriever-store settings
        glob_load_retrievers = {
            k: getattr(self.cfgLoadRetrievers, k)
            for k in dir(self.cfgLoadRetrievers)
            if k.isupper()
        }

        # 3) Config_Load_Chunkers — chunker routing and metadata extraction settings
        glob_load_chunkers = {
            k: getattr(self.cfgLoadChunkers, k)
            for k in dir(self.cfgLoadChunkers)
            if k.isupper()
        }

        # 4) Config_Languages — active language sets and Argos pair catalog
        glob_languages = {
            k: getattr(self.cfgLanguages, k)
            for k in dir(self.cfgLanguages)
            if k.isupper()
        }

        # 5) Config_Models
        glob_models = {
            k: getattr(self.cfgModels, k) for k in dir(self.cfgModels) if k.isupper()
        }

        # 6) Config_Banned_Detection
        glob_banned_detection = {
            k: getattr(self.cfgBannedDetection, k)
            for k in dir(self.cfgBannedDetection)
            if k.isupper()
        }

        # 7) Config_Banned_Content
        glob_banned_content = {
            k: getattr(self.cfgBannedContent, k)
            for k in dir(self.cfgBannedContent)
            if k.isupper()
        }

        # 8) Config_Banned_Prompts
        glob_banned_prompts = {
            k: getattr(self.cfgBannedPrompts, k)
            for k in dir(self.cfgBannedPrompts)
            if k.isupper()
        }

        # 9) Config_WebSearch
        glob_websearch = {
            k: getattr(self.cfgWebSearch, k)
            for k in dir(self.cfgWebSearch)
            if k.isupper()
        }

        # 10) RAGChat split modules (loaded by lookup for RAGChat / RAGChatService)
        use_ragchat_split = self.cfgPy in (Config_RAGChat, Config_RAGChatService)
        glob_ragchat_core: dict[str, Any] = {}
        glob_ragchat_strategies: dict[str, Any] = {}
        glob_ragchat_prompts: dict[str, Any] = {}
        glob_ragchat_rewrite: dict[str, Any] = {}
        glob_ragchat_display: dict[str, Any] = {}

        if use_ragchat_split:
            glob_ragchat_core = {
                k: getattr(self.cfgRagchatCore, k)
                for k in dir(self.cfgRagchatCore)
                if k.isupper()
            }
            glob_ragchat_strategies = {
                k: getattr(self.cfgRagchatStrategies, k)
                for k in dir(self.cfgRagchatStrategies)
                if k.isupper()
            }
            glob_ragchat_prompts = {
                k: getattr(self.cfgRagchatPrompts, k)
                for k in dir(self.cfgRagchatPrompts)
                if k.isupper()
            }
            glob_ragchat_rewrite = {
                k: getattr(self.cfgRagchatRewrite, k)
                for k in dir(self.cfgRagchatRewrite)
                if k.isupper()
            }
            glob_ragchat_display = {
                k: getattr(self.cfgRagchatDisplay, k)
                for k in dir(self.cfgRagchatDisplay)
                if k.isupper()
            }

        # 11) App-specific (Config_RAGChat / Config_RAGLoad / Config_DocClassify) — highest file priority
        prog_raw = {k: getattr(self.cfgPy, k) for k in dir(self.cfgPy) if k.isupper()}

        # Strict alias policy: '$' indirections must stay within the module where
        # they are declared to avoid ambiguous cross-module coupling.
        self._validate_indirect_lookup_scope("Config_Global", glob_raw)
        self._validate_indirect_lookup_scope(
            "Config_Load_Retrievers",
            glob_load_retrievers,
        )
        self._validate_indirect_lookup_scope(
            "Config_Load_Chunkers",
            glob_load_chunkers,
        )
        self._validate_indirect_lookup_scope("Config_Languages", glob_languages)
        self._validate_indirect_lookup_scope("Config_Models", glob_models)
        self._validate_indirect_lookup_scope(
            "Config_Banned_Detection",
            glob_banned_detection,
        )
        self._validate_indirect_lookup_scope(
            "Config_Banned_Content",
            glob_banned_content,
        )
        self._validate_indirect_lookup_scope(
            "Config_Banned_Prompts",
            glob_banned_prompts,
        )
        self._validate_indirect_lookup_scope("Config_WebSearch", glob_websearch)
        if use_ragchat_split:
            self._validate_indirect_lookup_scope(
                "Config_RAGChat",
                glob_ragchat_core,
            )
            self._validate_indirect_lookup_scope(
                "Config_RAGChat_Strategies",
                glob_ragchat_strategies,
            )
            self._validate_indirect_lookup_scope(
                "Config_RAGChat_Prompts",
                glob_ragchat_prompts,
            )
            self._validate_indirect_lookup_scope(
                "Config_RAGChat_Rewrite",
                glob_ragchat_rewrite,
            )
            self._validate_indirect_lookup_scope(
                "Config_RAGChat_Display",
                glob_ragchat_display,
            )
        self._validate_indirect_lookup_scope(
            getattr(self.cfgPy, "__name__", "AppConfig"),
            prog_raw,
        )

        # Merge: each layer overrides the previous; eligible CLI args override
        # merged slots via _get.
        raw = glob_raw.copy()
        raw.update(glob_load_retrievers)
        raw.update(glob_load_chunkers)
        raw.update(glob_languages)
        raw.update(glob_models)
        raw.update(glob_banned_detection)
        raw.update(glob_banned_content)
        raw.update(glob_banned_prompts)
        raw.update(glob_websearch)
        raw.update(glob_ragchat_core)
        raw.update(glob_ragchat_strategies)
        raw.update(glob_ragchat_prompts)
        raw.update(glob_ragchat_rewrite)
        raw.update(glob_ragchat_display)
        raw.update(prog_raw)

        # build the resolved config
        self.cfg = {}
        for key, val in raw.items():
            self.cfg[key] = val

    def load(self, filepath: str):
        """
        Load configuration from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def indirect_get(
        self, key: str, default: Any = None, max_depth: int = 5
    ) -> Tuple[Any, Optional[str]]:
        """
        Follow string indirections in self.cfg up to max_depth.

        Indirection targets may reference either top-level slots (``_SLOT``)
        or nested dotted paths (``_SLOT.sub.key``).

        Returns:
            Tuple[value, last_slot]
            - value: the resolved final value (or `default` if not found)
            - last_slot: the key itself when no indirection was followed,
              or the last intermediate key followed during indirection
        """
        current = key
        last_slot: Optional[str] = key
        INDIRECT_PREFIX = "$"
        _MISSING = object()

        for _ in range(max_depth):
            # Avoid re-triggering indirect-prefix logic while following indirections
            value = self.get(current, None, allow_indirect=False, silent=True)

            # If value is a string and points to a key, follow indirection.
            # Accept both plain key names ("_SLOT") and prefixed aliases
            # ("$_SLOT") for consistent config notation.
            if isinstance(value, str):
                next_key = value
                if next_key.startswith(INDIRECT_PREFIX):
                    next_key = next_key[len(INDIRECT_PREFIX) :]

                # Follow both top-level and dotted-path aliases.
                target_value = self.get(
                    next_key,
                    _MISSING,
                    allow_indirect=False,
                    silent=True,
                )
                if next_key in self.cfg or target_value is not _MISSING:
                    last_slot = next_key
                    current = next_key
                    continue

            # Otherwise treat it as final value
            return (value if value is not None else default, last_slot)

        # Safety fallback
        return (default, last_slot)

    def get(
        self,
        key: str,
        default: Any = None,
        allow_indirect: bool = True,
        *,
        silent: bool = False,
    ) -> Union[Any, Tuple[Any, Optional[str]]]:
        """Thread-safe wrapper — delegates to ``_get`` under ``self._lock``."""
        with self._lock:
            return self._get(key, default, allow_indirect, silent=silent)

    def _get(
        self,
        key: str,
        default: Any = None,
        allow_indirect: bool = True,
        *,
        silent: bool = False,
    ) -> Union[Any, Tuple[Any, Optional[str]]]:
        """
        Lookup a configuration value.

        Behavior:
        - If `key` starts with the indirect prefix ('$') and allow_indirect is True,
          return a tuple (value, last_slot) resolved via indirect_get.
        - Otherwise return the scalar value (as before) to preserve backward compatibility.
        - If ``silent`` is True, warning messages for missing keys are suppressed
          (useful for intentionally optional config entries).

        Return type:
        - Scalar Any for normal lookups.
        - Tuple[Any, Optional[str]] for indirect lookups (prefix '$').
        """
        INDIRECT_PREFIX = "$"

        # 0) If key is marked for indirection, handle it and return a tuple
        if allow_indirect and key.startswith(INDIRECT_PREFIX):
            stripped = key[len(INDIRECT_PREFIX) :]

            # CLI override for the stripped key (preserve existing behavior)
            key_up = stripped.upper()
            if key_up in self.args and self.args[key_up] is not None:
                # Return tuple: (cli_value, None) — no indirection followed
                return (self.args[key_up], None)

            # Use indirect_get which returns (value, last_slot)
            return self.indirect_get(stripped, default=default)

        # 1) CLI overrides for normal keys (scalar return)
        key_up = key.upper()
        if key_up in self.args and self.args[key_up] is not None:
            return self.args[key_up]

        # helper to write W/E with caller info
        def _show_error(level_tag: str, message: str):
            cls, func = ("<unknown>", "<unknown>")
            for depth in range(2, 5):
                try:
                    _cls, _func = self._get_caller_info(depth=depth)
                    if _cls and _cls not in ("Config", "<unknown>"):
                        cls, func = _cls, _func
                        break
                    if _cls:
                        cls, func = _cls, _func  # keep as fallback
                except Exception:
                    break
            color = RED if level_tag == "E" else None
            self.pretty.write(
                level_tag,
                "Config.get",
                f"Called from: {cls}.{func} {message}",
                color=color,
            )

        # 2) nested lookup in loaded config (scalar return)
        parts = key.split(".")
        node = self.cfg
        _MISSING = object()
        for part in parts:
            if not isinstance(node, dict):
                if default is None:
                    _show_error(
                        "E",
                        f"Not a dictionary at path segment '{part}' for key '{key}'; no default provided.",
                    )
                    raise ConfigPathError(
                        f"Not a dictionary at path segment '{part}' for key '{key}'; no default provided."
                    )
                else:
                    if not silent:
                        _show_error(
                            "W",
                            f"Not a dictionary at path segment '{part}' for key '{key}'; returning default.",
                        )
                return default

            node_d = cast(dict[str, Any], node)
            value = node_d.get(part, _MISSING)
            if value is _MISSING:
                if default is None:
                    _show_error(
                        "E",
                        f"Lookup of non-existent path '{key}'; no default provided.",
                    )
                    raise ConfigPathError
                else:
                    if not silent:
                        _show_error(
                            "W",
                            f"Lookup of non-existent path '{key}'; returning default.",
                        )
                return default
            node = value
        return node

    # ── typed convenience accessors ──────────────────────────────────

    def get_bool(
        self, key: str, default: bool = False, *, silent: bool = False
    ) -> bool:
        """Return a config value coerced to *bool*.

        Handles CLI strings like ``"True"``/``"False"`` correctly
        (plain ``bool("False")`` would return ``True``).
        """
        val: Any = self.get(key, default, silent=silent)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)

    def get_int(self, key: str, default: int = 0, *, silent: bool = False) -> int:
        """Return a config value coerced to *int*."""
        val: Any = self.get(key, default, silent=silent)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        return int(val)

    def get_float(
        self, key: str, default: float = 0.0, *, silent: bool = False
    ) -> float:
        """Return a config value coerced to *float*."""
        val: Any = self.get(key, default, silent=silent)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return float(val)

    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        """Return a config value coerced to *str*."""
        val: Any = self.get(key, default, silent=silent)
        if isinstance(val, str):
            return val
        return str(val)

    def _indirect_value(
        self,
        key: str,
        default: Any = None,
        *,
        silent: bool = False,
    ) -> Any:
        """Resolve an indirect key and return only the resolved value."""
        indirect_key = key if key.startswith("$") else f"${key}"
        result: Any = self.get(indirect_key, default, silent=silent)
        if isinstance(result, tuple):
            resolved = cast(tuple[Any, Optional[str]], result)
            return resolved[0]
        return result

    def indirect_list(
        self, key: str, default: list[Any] | None = None, *, silent: bool = False
    ) -> list[Any]:
        """Return a *list* resolved through indirect lookup semantics."""
        val: Any = self._indirect_value(key, default, silent=silent)
        if val is None:
            return default if default is not None else []
        if isinstance(val, list):
            return cast(list[Any], val)
        raise TypeError(
            "Config key "
            f"'{key}' expected list via indirect lookup, got {type(val).__name__}"
        )

    def indirect_dict(
        self, key: str, default: dict[str, Any] | None = None, *, silent: bool = False
    ) -> dict[str, Any]:
        """Return a *dict* resolved through indirect lookup semantics."""
        val: Any = self._indirect_value(key, default, silent=silent)
        if val is None:
            return default if default is not None else {}
        if isinstance(val, dict):
            return cast(dict[str, Any], val)
        raise TypeError(
            "Config key "
            f"'{key}' expected dict via indirect lookup, got {type(val).__name__}"
        )

    def get_list(
        self, key: str, default: list[Any] | None = None, *, silent: bool = False
    ) -> list[Any]:
        """Return a config value that is expected to be a *list*."""
        if key.startswith("$"):
            raise TypeError(
                "Config key "
                f"'{key}' uses indirect prefix '$'; use indirect_list(...) instead"
            )
        val: Any = self.get(key, default, silent=silent)
        if val is None:
            return default if default is not None else []
        if isinstance(val, list):
            return cast(list[Any], val)
        raise TypeError(f"Config key '{key}' expected list, got {type(val).__name__}")

    def get_dict(
        self, key: str, default: dict[str, Any] | None = None, *, silent: bool = False
    ) -> dict[str, Any]:
        """Return a config value that is expected to be a *dict*."""
        if key.startswith("$"):
            raise TypeError(
                "Config key "
                f"'{key}' uses indirect prefix '$'; use indirect_dict(...) instead"
            )
        val: Any = self.get(key, default, silent=silent)
        if val is None:
            return default if default is not None else {}
        if isinstance(val, dict):
            return cast(dict[str, Any], val)
        raise TypeError(f"Config key '{key}' expected dict, got {type(val).__name__}")

    # ── end typed accessors ──────────────────────────────────────────

    def __getitem__(self, key: str):
        return self.get(key)

    def print_config_values(self):
        """
        Pretty-print all top-level config keys.
        If a value is a dict, render it as a native dict.
        Honors CLI overrides via self.get().
        """
        # Collect and sort top-level keys
        keys = sorted(self.cfg.keys())

        # Determine padding for alignment
        max_key_len = max((len(k) for k in keys), default=0) + 2

        for key in keys:
            # Always use get() to respect CLI overrides
            val = self.get(key)

            # If it's a dict, print the key then the dict
            if isinstance(val, dict):
                print(f"{key.ljust(max_key_len)}:")
                pprint.pprint(cast(dict[str, Any], val), indent=4, width=120)
                self.pretty.write("N", "", "")  # blank line after each big dict
            else:
                # Simple aligned print for scalars or lists
                print(f"{key.ljust(max_key_len)}: {val}")

    def _get_caller_info(self, depth: int = 1):
        """
        Returns (class_name, function_name) of the caller `depth` frames above.
        depth=1 → direct caller
        depth=2 → caller of the caller
        """
        frame = inspect.currentframe()
        if frame is None:
            return (None, None)
        for _ in range(depth):
            parent = frame.f_back
            if parent is None:
                return (None, None)
            frame = parent

        func_name = frame.f_code.co_name

        # Detect class context
        cls_name = None
        if "self" in frame.f_locals:
            cls_name = frame.f_locals["self"].__class__.__name__
        elif "cls" in frame.f_locals:
            cls_name = frame.f_locals["cls"].__name__

        return cls_name, func_name

    def set(
        self,
        key: str,
        value: Any,
        *,
        force: bool = False,
        create_missing: bool = True,
        allow_indirect: bool = True,
    ) -> Any:
        """Thread-safe wrapper — delegates to ``_set`` under ``self._lock``."""
        with self._lock:
            return self._set(
                key,
                value,
                force=force,
                create_missing=create_missing,
                allow_indirect=allow_indirect,
            )

    def _set(
        self,
        key: str,
        value: Any,
        *,
        force: bool = False,
        create_missing: bool = True,
        allow_indirect: bool = True,
    ) -> Any:
        """
        Set a configuration value.

        Parameters
        - key: dotted path (e.g. "SECTION.SUBKEY") or indirect ("$OTHER_KEY")
        - value: value to assign
        - force: if False and a CLI override exists for the final key, do not overwrite it
        - create_missing: create intermediate dicts when traversing dotted path
        - allow_indirect: if True and key starts with '$', follow indirection logic

        Returns
        - previous value (or None if not present)
        """
        INDIRECT_PREFIX = "$"

        # Handle indirect keys like get() does
        if allow_indirect and key.startswith(INDIRECT_PREFIX):
            stripped = key[len(INDIRECT_PREFIX) :]
            # If CLI override exists for stripped key and not forcing, respect it
            stripped_up = stripped.upper()
            if (
                stripped_up in self.args
                and self.args[stripped_up] is not None
                and not force
            ):
                # Do not overwrite CLI override
                prev = self.args[stripped_up]
                self.pretty.write(
                    "W",
                    "Config.set",
                    f"CLI override present for {stripped_up}; not overwriting (use force=True to override).",
                )
                return prev

            # If the stripped key itself is an indirection chain, try to resolve last slot
            _, last_slot = self.indirect_get(stripped, default=None)
            target_key = last_slot or stripped
        else:
            target_key = key

        # If the target is a top-level CLI arg and not forcing, refuse to overwrite
        top_up = target_key.upper()
        if top_up in self.args and self.args[top_up] is not None and not force:
            prev = self.args[top_up]
            self.pretty.write(
                "W",
                "Config.set",
                f"CLI override present for {top_up}; not overwriting (use force=True to override).",
            )
            return prev

        # When forcing, also update args so get() returns the new value
        if top_up in self.args and force:
            self.args[top_up] = value

        # Traverse dotted path and set value
        parts = target_key.split(".")
        node = self.cfg
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1

            if not isinstance(node, dict):
                # Can't descend further
                self.pretty.write(
                    "E",
                    "Config.set",
                    f"Path segment '{part}' is not a dict while setting '{key}'.",
                    color=RED,
                )
                raise ConfigPathError(
                    f"Not a dictionary at path segment '{part}' for key '{key}'"
                )

            if is_last:
                node_d = cast(dict[str, Any], node)
                prev = node_d.get(part, None)
                node_d[part] = value
                return prev
            else:
                node_d = cast(dict[str, Any], node)
                # intermediate node
                if part not in node_d:
                    if create_missing:
                        node_d[part] = {}
                    else:
                        self.pretty.write(
                            "E",
                            "Config.set",
                            f"Missing path segment '{part}' for key '{key}'; create_missing=False.",
                            color=RED,
                        )
                        raise ConfigPathError(
                            f"Missing path segment '{part}' for key '{key}'"
                        )
                node = node_d[part]
