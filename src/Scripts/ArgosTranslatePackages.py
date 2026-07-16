#!/usr/bin/env python3
"""
Thin CLI wrapper around ArgosDownloader.

Installs / removes the Argos Translate language packages used by the
Compliance pipeline to translate the English banlist into the target
languages (EN→X). User-query translation no longer uses Argos — that
path is handled by the m2m100 backend (see Compliance/HfTranslator.py).
The set of pairs to install is defined by
``_ARGOS_DEFINITIONS.ARGOS_LANGUAGES`` in Config_Global.py.

Usage:
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
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))

import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # noqa: E402,F401 — side-effect import
from Config.Config import Config  # noqa: E402
from Gui.Symbols import Symbols  # noqa: E402

Symbols.store_emoji_preference(Config())

from Commons.StartupCommons import suppress_argos_logging  # noqa: E402
from Compliance.ArgosDownloader import ArgosDownloader  # noqa: E402

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or remove Argos Translate language packages."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="install",
        choices=["install", "remove", "status"],
        help="Action to perform (default: install)",
    )
    args = parser.parse_args()

    cfg = Config()
    languages = cfg.get_list("_ARGOS_DEFINITIONS.ARGOS_LANGUAGES")
    downloader = ArgosDownloader(_PROJECT_ROOT, languages)

    if args.action == "install":
        if not downloader.ensure_packages():
            print("\nInstallation aborted.")
            sys.exit(1)
        downloader.show_status()

    elif args.action == "remove":
        downloader.show_status()
        print()
        if not _confirm(
            "Remove all Argos packages, stanza models, and consent metadata?"
        ):
            print("Aborted.")
            sys.exit(0)
        downloader.remove_all()
        stanza_dir = downloader.stanza_models_dir
        if os.path.isdir(stanza_dir):
            if _confirm(f"Recursively delete stanza models directory '{stanza_dir}'?"):
                downloader.remove_stanza_models()
            else:
                print("  Skipping stanza models removal.")
        else:
            print(f"  Stanza models directory not found — skipping ({stanza_dir})")
        downloader.remove_consent()
        downloader.show_status()

    elif args.action == "status":
        downloader.show_status()


if __name__ == "__main__":
    main()
