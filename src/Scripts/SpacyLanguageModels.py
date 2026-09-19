#!/usr/bin/env python3
"""
Thin CLI wrapper around SpacyDownloader.

Installs / removes the spaCy model packages used by BM25, graph, and regex
retrieval components. The model package names are read from:
    - _BM25_INDEX.spacy_model
        - _BM25_INDEX.spacy_models_by_language
  - _GRAPH_INDEX.spacy_model
    - _GRAPH_INDEX.spacy_models_by_language
  - _REGEX_INDEX.spacy_model
    - _REGEX_INDEX.spacy_models_by_language
Per-language model overrides are filtered by
_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES.

Usage:
    python src/Scripts/SpacyLanguageModels.py            # show installed (default)
    python src/Scripts/SpacyLanguageModels.py install  # consent + install
    python src/Scripts/SpacyLanguageModels.py remove   # uninstall + cleanup
    python src/Scripts/SpacyLanguageModels.py status   # show installed
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

import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # noqa: E402,F401
from Compliance.SpacyDownloader import SpacyDownloader  # noqa: E402
from Config.Config import Config  # noqa: E402
from Gui.Colors import CYAN, ORANGE, RESET  # noqa: E402
from Gui.Symbols import Symbols  # noqa: E402
from Scripts.SpacyLanguageModelsImpl import (  # noqa: E402
    configured_spacy_models, print_install_scope_note,
    report_spacy_model_license_hints)

Symbols.store_emoji_preference(Config())


def _confirm(prompt_msg: str) -> bool:
    """Loop asking yes/no until the user gives a clear answer."""
    while True:
        answer = input(f"{prompt_msg} [yes/no]: ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("  Please answer 'yes' or 'no'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or remove spaCy model packages used by RAG-LCC."
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
    models = configured_spacy_models(cfg)
    downloader = SpacyDownloader(_PROJECT_ROOT, models)

    if args.action == "install":
        print_install_scope_note()
        print()
        if not downloader.ensure_models():
            print("\nInstallation aborted.")
            print(
                f"{CYAN}Hint: run python ./src/Scripts/SpacyLanguageModels.py install when you are ready.{RESET}"
            )
            sys.exit(1)
        downloader.show_status()

    elif args.action == "remove":
        installed_models = downloader.installed_models()
        has_models = bool(installed_models)
        has_consent_metadata = downloader.has_consent_metadata()
        cleanup_dirs = downloader.consent_cleanup_directories()
        existing_cleanup_dirs = [
            cleanup_dir for cleanup_dir in cleanup_dirs if os.path.isdir(cleanup_dir)
        ]

        if not has_models and not has_consent_metadata and not existing_cleanup_dirs:
            print(
                "No configured spaCy model packages, consent metadata, or consent directories found - nothing to remove."
            )
            sys.exit(0)

        if not existing_cleanup_dirs:
            print("No configured spaCy consent directories found - nothing to remove.")
        else:
            print(
                f"{ORANGE}The following directories contain governance files that may be removed:{RESET}"
            )
            for cleanup_dir in existing_cleanup_dirs:
                print(f"  {cleanup_dir}")
            print()

        if not _confirm("Remove configured spaCy model packages and consent metadata?"):
            print("Aborted.")
            sys.exit(0)
        downloader.remove_all()
        downloader.remove_consent()

    elif args.action == "status":
        downloader.show_status()
        report_spacy_model_license_hints(models)


if __name__ == "__main__":
    main()
