#!/usr/bin/env python3
"""
CopyExampleConfigs.py – copy Example_*.py files from the Examples folder
to the Configuration folder, stripping the ``Example_`` prefix from the
destination filename.

When ``--force`` is passed, existing files in Configuration are backed up
to ``.sav`` extensions and then overwritten with fresh examples. This ensures
Setup.py can apply runtime configuration updates to clean baseline files.

Without ``--force``, existing files are skipped so local edits are preserved.

Usage
-----
    python CopyExampleConfigs.py [--force]

Options
-------
    --force   Backup existing configuration files to .sav and overwrite with
              fresh examples from the Examples/ directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "Examples"
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "Configuration"
_PREFIX = "Example_"
_PREFERRED_EXAMPLES = [
    "Example_Config_Models.py",
    "Example_Config_Banned.py",
    "Example_Config_WebSearch.py",
    "Example_Config_Internet_Env.py",
]
_ORANGE = "\033[38;2;255;165;0m"
_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy example configs to Configuration/"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Backup existing files to .sav and overwrite with fresh examples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_sources = sorted(_EXAMPLES_DIR.glob(f"{_PREFIX}*.py"))
    source_map = {src.name: src for src in all_sources}
    missing_preferred = [name for name in _PREFERRED_EXAMPLES if name not in source_map]
    for name in missing_preferred:
        print(
            f"{_ORANGE}  WARN   {name} not copied because not present in {_EXAMPLES_DIR}{_RESET}"
        )

    preferred_sources = [
        source_map[name] for name in _PREFERRED_EXAMPLES if name in source_map
    ]
    preferred_names = {src.name for src in preferred_sources}
    other_sources = [src for src in all_sources if src.name not in preferred_names]
    sources = preferred_sources + other_sources

    if not sources:
        print(
            f"{_ORANGE}  WARN   no files matching '{_PREFIX}*.py' found in {_EXAMPLES_DIR}; nothing copied.{_RESET}"
        )
        return 0

    copied = skipped = overwritten_with_backup = 0
    for src in sources:
        dest_name = src.name[len(_PREFIX) :]
        dest = _CONFIG_DIR / dest_name

        if dest.exists() and not args.force:
            print(f"  SKIP   {dest_name}  (already exists; use --force to overwrite)")
            skipped += 1
            continue

        # If target exists, save it to .sav before overwriting
        if dest.exists():
            backup_path = dest.with_suffix(dest.suffix + ".sav")
            shutil.copy2(dest, backup_path)
            print(f"  BACKUP {dest_name}  ->  {backup_path.name}")
            overwritten_with_backup += 1
            action = "OVERWRITE"
        else:
            action = "COPY"

        shutil.copy2(src, dest)
        print(f"  {action:<9}{src.name}  ->  {dest_name}")
        copied += 1

    print(
        f"\nDone: {copied} copied, {skipped} skipped, {overwritten_with_backup} backed up to .sav"
    )
    if overwritten_with_backup > 0:
        print(
            f"Note: Original config files saved with .sav extension before overwriting."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
