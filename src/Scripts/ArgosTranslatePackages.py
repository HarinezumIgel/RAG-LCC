#!/usr/bin/env python3
"""
Thin CLI wrapper around ArgosDownloader.

Installs / removes the Argos Translate language packages used by the
Compliance pipeline to translate the English banlist into the target
languages (EN→X). When query translation is configured as
"argos", source-language→English pairs are also
required for user-query normalization.
User queries are translated using Argos Translate.
The pair catalog is defined by ``_ARGOS_DEFINITIONS.ARGOS_LANGUAGES`` and
filtered by ``_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES`` in Config_Global.py.

Usage:
    python src/Scripts/ArgosTranslatePackages.py           # show installed (default)
    python src/Scripts/ArgosTranslatePackages.py install   # consent + download
    python src/Scripts/ArgosTranslatePackages.py remove    # uninstall + cleanup
    python src/Scripts/ArgosTranslatePackages.py status    # show installed
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is on sys.path and load environment configuration
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import Configuration.Config_Global as _ConfigGlobal  # noqa: E402
# Resolve and validate project root before Config-dependent imports.
from Commons.DriveRootGuard import assert_script_project_root  # noqa: E402

_PROJECT_ROOT = assert_script_project_root(
    __file__,
    configured_project_root=getattr(_ConfigGlobal, "_ABSOLUTE_PATH", None),
    require_cwd_match=True,
)

import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # noqa: E402,F401 — side-effect import
from Config.Config import Config  # noqa: E402
from Gui.Colors import CYAN, ORANGE, RESET  # noqa: E402
from Gui.Symbols import Symbols  # noqa: E402

Symbols.store_emoji_preference(Config())

from Commons.StartupCommons import suppress_argos_logging  # noqa: E402
from Compliance.ArgosDownloader import ArgosDownloader  # noqa: E402
from Helpers.LanguageConfig import get_active_argos_pairs  # noqa: E402
from Scripts.ArgosTranslatePackagesImpl import (  # noqa: E402
    print_install_scope_note, report_argos_package_license_hints,
    report_xx_sent_ud_sm_state)

suppress_argos_logging()


def _confirm(prompt_msg: str) -> bool:
    """Loop asking yes/no until the user gives a clear answer."""
    while True:
        answer = input(f"{prompt_msg} [yes/no]: ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("  Please answer 'yes' or 'no'.")


def _join_targets(labels: list[str]) -> str:
    """Join a list of labels into a natural-language prompt phrase."""
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or remove Argos Translate language packages."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["install", "remove", "status"],
        help="Action to perform (default: status)",
    )
    args = parser.parse_args()

    cfg = Config()
    languages = get_active_argos_pairs(cfg)
    downloader = ArgosDownloader(
        _PROJECT_ROOT,
        languages,
        emit_startup_inventory=False,
    )

    if args.action == "install":
        print_install_scope_note()
        print()
        if not downloader.ensure_packages():
            print("\nInstallation aborted.")
            print(
                f"{CYAN}Hint: run python ./src/Scripts/ArgosTranslatePackages.py install when you are ready.{RESET}"
            )
            sys.exit(1)
        downloader.show_status()
        report_xx_sent_ud_sm_state()

    elif args.action == "remove":
        installed_packages = downloader.installed_packages()
        has_packages = bool(installed_packages)
        tokenizer_resources_dir = downloader.tokenizer_resources_dir
        has_tokenizer_resources = os.path.isdir(tokenizer_resources_dir)
        has_consent_metadata = downloader.has_consent_metadata()

        if (
            not has_packages
            and not has_tokenizer_resources
            and not has_consent_metadata
        ):
            print(
                "No Argos packages, tokenizer resources, or consent metadata installed - nothing to remove."
            )
            sys.exit(0)

        if not has_tokenizer_resources:
            print("No tokenizer resources directory present - nothing to remove.")

        if has_consent_metadata:
            print()
            print(
                f"{ORANGE}The following directories contain governance files that may be removed:{RESET}"
            )
            for cleanup_dir in downloader.consent_cleanup_directories():
                print(f"  {cleanup_dir}")
        if has_tokenizer_resources:
            print(f"  {tokenizer_resources_dir}  (tokenizer resources)")
        if has_consent_metadata or has_tokenizer_resources:
            print()

        removal_targets: list[str] = []
        if has_packages:
            removal_targets.append("Argos packages")
        if has_tokenizer_resources:
            removal_targets.append("tokenizer resources")
        if has_consent_metadata:
            removal_targets.append("consent metadata")

        if not _confirm(f"Remove {_join_targets(removal_targets)}?"):
            print("Aborted.")
            sys.exit(0)

        if has_packages:
            downloader.remove_all()
        if has_tokenizer_resources:
            if _confirm(
                f"Recursively delete tokenizer resources directory '{tokenizer_resources_dir}'?"
            ):
                downloader.remove_tokenizer_resources()
            else:
                print("  Skipping tokenizer resources removal.")
        if has_consent_metadata:
            downloader.remove_consent()
        downloader.show_status()

    elif args.action == "status":
        downloader.show_status()
        report_xx_sent_ud_sm_state()
        report_argos_package_license_hints()


if __name__ == "__main__":
    main()
