import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any, cast

import Configuration.Config_DocClassify as Config_DocClassify
import Configuration.Config_Global as Config_Global
import Configuration.Config_RAGChat as Config_RAGChat
import Configuration.Config_RAGLoad as Config_RAGLoad

config_modules = {
    "Config_RAGChat": Config_RAGChat,
    "Config_RAGLoad": Config_RAGLoad,
    "Config_DocClassify": Config_DocClassify,
    "Config_Global": Config_Global,
}
# Case-insensitive lookup table (Windows preserves typed casing in sys.argv[0])
_config_modules_lower = {k.lower(): v for k, v in config_modules.items()}


class AddConstantsFromConfigFile(argparse.ArgumentParser):
    class _HelpAndListAction(argparse.Action):
        def __init__(
            self,
            option_strings: Sequence[str],
            dest: str = argparse.SUPPRESS,
            default: str = argparse.SUPPRESS,
            help: str | None = None,
        ) -> None:
            # nargs=0 => flag takes no value
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                help=help,
            )

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Sequence[Any] | None,
            option_string: str | None = None,
        ) -> None:
            # 1) print the normal help text
            parser.print_help()

            # 2) figure out which Config_<script>.py applied
            script_base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            cfg_mod = _config_modules_lower.get(f"Config_{script_base}".lower())

            # 3) extract your constants
            global_consts: dict[str, Any] = cast(dict[str, Any], parser._extract_flat_constants(Config_Global))  # type: ignore[attr-defined]
            script_consts: dict[str, Any] = cast(dict[str, Any], parser._extract_flat_constants(cfg_mod)) if cfg_mod else {}  # type: ignore[attr-defined]

            # 4) print them
            print("\nAllowed overrideable constants and their defaults:\n")
            for dct in (global_consts, script_consts):
                for key, val in dct.items():
                    flag = f"--{key.lower().replace('_','-')}"
                    print(f"  {flag:<30} {val!r}")
            print("\n")
            parser.exit()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # turn off the default -h/--help
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)

        # add our custom help/listing flag
        self.add_argument(
            "-h",
            "--help",
            action=self._HelpAndListAction,
            help="show this help message and list all overrideable constants",
        )

        # now add all the auto‐flags for your UPPER_CASE constants
        self._add_constants_flags()

    def str2bool(self, v: Any) -> bool:
        # ... your existing implementation ...
        if isinstance(v, bool):
            return v
        val = v.lower()
        if val in ("yes", "y", "true", "t", "1"):
            return True
        if val in ("no", "n", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")

    def _extract_flat_constants(self, module: Any) -> dict[str, Any]:
        mod_vars = cast(dict[str, Any], vars(module))
        return {
            name: value
            for name, value in mod_vars.items()
            if (
                name.isupper()
                and not name.startswith("_")
                and not name.startswith("INTERNAL_")
                and not isinstance(value, dict)
            )
        }

    def _add_flag(self, key: str, default: Any) -> None:
        flag = f"--{key.lower().replace('_','-')}"
        arg_type: type[Any] = (
            cast(type[Any], type(default)) if default is not None else str
        )
        if arg_type is bool:
            self.add_argument(
                flag,
                type=self.str2bool,
                nargs="?",
                const=True,
                default=None,
                help=f"override {key!r} (default={default!r})",
            )
        else:
            self.add_argument(
                flag,
                type=arg_type,
                default=None,
                help=f"override {key!r} (default={default!r})",
            )

    def _add_constants_flags(self):
        script_base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        config_name = f"Config_{script_base}"
        config_py = _config_modules_lower.get(config_name.lower())
        if not config_py:
            return

        global_consts = self._extract_flat_constants(Config_Global)
        script_consts = self._extract_flat_constants(config_py)

        added: set[str] = set()
        # globals first (unless overridden by script)
        for key, default in global_consts.items():
            if key in script_consts:
                continue
            self._add_flag(key, default)
            added.add(key)

        # then script‐specific constants
        for key, default in script_consts.items():
            if key in added:
                continue
            self._add_flag(key, default)
            added.add(key)
