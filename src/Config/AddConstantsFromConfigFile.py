import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any, cast

import Configuration.Config_DocClassify as Config_DocClassify
import Configuration.Config_Global as Config_Global
import Configuration.Config_Load_Retrievers as Config_Load_Retrievers
import Configuration.Config_RAGChat as Config_RAGChat
import Configuration.Config_RAGChatService as Config_RAGChatService
import Configuration.Config_RAGLoad as Config_RAGLoad
from Config.CliOverridePolicy import (app_uses_load_retrievers_cli_scope,
                                      build_allowed_cli_overrides)

config_modules = {
    "Config_RAGChat": Config_RAGChat,
    "Config_RAGChatService": Config_RAGChatService,
    "Config_RAGLoad": Config_RAGLoad,
    "Config_DocClassify": Config_DocClassify,
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
            cfg_mod = parser._active_app_module()  # type: ignore[attr-defined]

            # 3) extract your constants
            allowed_consts: dict[str, Any] = cast(
                dict[str, Any],
                parser._allowed_constants_for_active_app(cfg_mod),  # type: ignore[attr-defined]
            )

            # 4) print them
            print("\nAllowed overrideable constants and their defaults:\n")
            for key, val in allowed_consts.items():
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

    def _active_app_module(self) -> Any | None:
        script_base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        config_name = f"Config_{script_base}"
        return _config_modules_lower.get(config_name.lower())

    def _allowed_constants_for_active_app(
        self,
        app_module: Any | None = None,
    ) -> dict[str, Any]:
        if app_module is None:
            app_module = self._active_app_module()
        load_retrievers_module: Any | None = None
        if app_uses_load_retrievers_cli_scope(app_module):
            load_retrievers_module = Config_Load_Retrievers
        return build_allowed_cli_overrides(
            Config_Global,
            app_module,
            load_retrievers_module,
        )

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
        allowed_consts = self._allowed_constants_for_active_app()
        for key, default in allowed_consts.items():
            self._add_flag(key, default)
