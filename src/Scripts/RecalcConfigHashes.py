#!/usr/bin/env python3
"""Update / clear the config hash entries in Config_Global.py.

Recomputes the SHA-256 file hashes of ``Config_Models.py``,
``Config_Banned.py``, ``Config_WebSearch.py``, and ``Config_Internet_Env.py``
and writes them into the
``_CRITICAL_CONFIG_HASHES`` dict in ``Config_Global.py`` (matching the format
``Compliance._check_models_config_hash`` validates against at startup).

Usage:
    python src/Scripts/config_hashes.py          # write current hashes
    python src/Scripts/config_hashes.py clean    # clear all slots ("")
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Default: script lives at <root>/src/Scripts/ → config is at <root>/src/Configuration/
_DEFAULT_CFG_DIR = os.path.normpath(os.path.join(_HERE, "..", "Configuration"))

# Refuse to run from a drive/filesystem root before doing anything
sys.path.insert(0, os.path.dirname(_HERE))
from Commons.DriveRootGuard import assert_not_drive_root  # noqa: E402

assert_not_drive_root(__file__)


def _sha256_file(path: str, chunk_size: int = 8_192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _replace_slot(text: str, slot: str, new_value: str) -> str:
    """Replace ``"<slot>": "..."`` inside ``_CRITICAL_CONFIG_HASHES`` in *text*.

    Matches the existing dict entry regardless of current quoting / spacing.
    Raises ``ValueError`` if the slot key is not found.
    """
    pattern = re.compile(
        rf'("{re.escape(slot)}"\s*:\s*)(["\'])(.*?)\2',
        re.MULTILINE,
    )
    if not pattern.search(text):
        raise ValueError(f"Slot {slot!r} not found in _CRITICAL_CONFIG_HASHES")
    return pattern.sub(lambda m: f'{m.group(1)}"{new_value}"', text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync _CRITICAL_CONFIG_HASHES entries in "
            "Config_Global.py with the current file hashes."
        )
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="update",
        choices=["update", "clean"],
        help=(
            "update (default): write current SHA-256 hashes; "
            "clean: blank all entries."
        ),
    )
    parser.add_argument(
        "--target-path",
        default=None,
        metavar="ROOT",
        help=(
            "Root directory of the project to operate on. "
            "Expects src/Configuration/ inside it. "
            "Defaults to the project root inferred from this script's location."
        ),
    )
    args = parser.parse_args()

    if args.target_path is not None:
        cfg_dir = os.path.normpath(
            os.path.join(args.target_path, "src", "Configuration")
        )
    else:
        cfg_dir = _DEFAULT_CFG_DIR

    CONFIG_GLOBAL = os.path.join(cfg_dir, "Config_Global.py")
    CONFIG_MODELS = os.path.join(cfg_dir, "Config_Models.py")
    CONFIG_BANNED = os.path.join(cfg_dir, "Config_Banned.py")
    CONFIG_WEB_SEARCH = os.path.join(cfg_dir, "Config_WebSearch.py")
    CONFIG_INTERNET_ENV = os.path.join(cfg_dir, "Config_Internet_Env.py")

    _SLOTS = [
        ("Config_Models", CONFIG_MODELS),
        ("Config_Banned", CONFIG_BANNED),
        ("Config_WebSearch", CONFIG_WEB_SEARCH),
        ("Config_Internet_Env", CONFIG_INTERNET_ENV),
    ]

    for path in (
        CONFIG_GLOBAL,
        CONFIG_MODELS,
        CONFIG_BANNED,
        CONFIG_WEB_SEARCH,
        CONFIG_INTERNET_ENV,
    ):
        if not os.path.isfile(path):
            print(f"ERROR: missing config file: {path}", file=sys.stderr)
            return 1

    if args.action == "update":
        print(
            "\nThis script rewrites the _CRITICAL_CONFIG_HASHES entries in Config_Global.py "
            "to match the current state of Config_Models.py, Config_Banned.py, "
            "Config_WebSearch.py, and Config_Internet_Env.py.\n"
            "Run this only after you have intentionally edited one of those files "
            "(src/Configuration/Config_Models.py, src/Configuration/Config_Banned.py, "
            "src/Configuration/Config_WebSearch.py, or "
            "src/Configuration/Config_Internet_Env.py).\n"
        )
        while True:
            answer = (
                input("Confirm you want to recalculate the config hashes [y/n]: ")
                .strip()
                .lower()
            )
            if answer in ("y", "yes"):
                break
            if answer in ("n", "no"):
                print("Aborted.")
                return 0
            print("  Please answer y or n.")

    with open(CONFIG_GLOBAL, "r", encoding="utf-8") as f:
        text = f.read()
    original = text

    for slot, src in _SLOTS:
        new_value = "" if args.action == "clean" else _sha256_file(src)
        text = _replace_slot(text, slot, new_value)
        label = "cleared" if args.action == "clean" else new_value
        print(f"  {slot:<26} <- {label}  ({os.path.basename(src)})")

    if text == original:
        print("Config_Global.py already up to date — no changes written.")
        return 0

    with open(CONFIG_GLOBAL, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"Updated {CONFIG_GLOBAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
