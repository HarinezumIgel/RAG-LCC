import inspect
import json
import os
import pprint
import sys
import threading
from typing import Any, Optional, Tuple, Union, cast

import Configuration.Config_Banned as Config_Banned
import Configuration.Config_DocClassify as Config_DocClassify
import Configuration.Config_Global as Config_Global
import Configuration.Config_Models as Config_Models
import Configuration.Config_RAGChat as Config_RAGChat
import Configuration.Config_RAGChatService as Config_RAGChatService
import Configuration.Config_RAGLoad as Config_RAGLoad
from Commons.Exceptions import ConfigPathError
from Commons.SingletonMixin import SingletonMixin
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter

config_modules = {
    "Config_Banned": Config_Banned,
    "Config_Models": Config_Models,
    "Config_RAGChat": Config_RAGChat,
    "Config_RAGChatService": Config_RAGChatService,
    "Config_RAGLoad": Config_RAGLoad,
    "Config_DocClassify": Config_DocClassify,
    "Config_Global": Config_Global,
}
# Case-insensitive lookup table (Windows preserves typed casing in sys.argv[0])
_config_modules_lower = {k.lower(): v for k, v in config_modules.items()}


class Config(SingletonMixin):

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
        self.cfgModels: Any = Config_Models
        self.cfgBanned: Any = Config_Banned

        # capture CLI args as an uppercase dict
        self.args: dict[str, Any] = args or {}
        if args:
            raw_args = cast(dict[str, Any], vars(args))
            self.args = {k.upper(): v for k, v in raw_args.items()}

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
        # 1) global definitions
        glob_raw = {
            k: getattr(self.globalConfigPy, k)
            for k in dir(self.globalConfigPy)
            if k.isupper()
        }
        # 2) Banned
        glob_banned = {
            k: getattr(self.cfgBanned, k) for k in dir(self.cfgBanned) if k.isupper()
        }

        # 3) Models
        glob_models = {
            k: getattr(self.cfgModels, k) for k in dir(self.cfgModels) if k.isupper()
        }

        # 1) program-specific overrides
        prog_raw = {k: getattr(self.cfgPy, k) for k in dir(self.cfgPy) if k.isupper()}

        # 3) merge: program-specific < global
        raw = glob_raw.copy()
        raw.update(glob_models)
        raw.update(glob_banned)
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

        Returns:
            Tuple[value, last_slot]
            - value: the resolved final value (or `default` if not found)
            - last_slot: the key itself when no indirection was followed,
              or the last intermediate key followed during indirection
        """
        current = key
        last_slot: Optional[str] = key

        for _ in range(max_depth):
            # Avoid re-triggering indirect-prefix logic while following indirections
            value = self.get(current, None, allow_indirect=False)

            # If value is a string AND is a valid key → follow indirection
            if isinstance(value, str) and value in self.cfg:
                last_slot = value
                current = value
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

    def get_list(
        self, key: str, default: list[Any] | None = None, *, silent: bool = False
    ) -> list[Any]:
        """Return a config value that is expected to be a *list*."""
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
                # Log the change
                if self.cfg.get("DEBUG_LEVEL", 3) >= 3:
                    self.pretty.write(
                        "D",
                        "Config.set",
                        f"Set key:'{target_key}' (was before: {pprint.pformat(prev)}, now: {pprint.pformat(value)})",
                    )
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
