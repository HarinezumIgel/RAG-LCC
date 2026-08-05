#!/usr/bin/env python3
"""Apply one or more config updates from a list of dictionaries."""

from __future__ import annotations

import argparse
import json
# Refuse to run from a drive/filesystem root before doing anything
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Commons.DriveRootGuard import assert_not_drive_root  # noqa: E402

assert_not_drive_root(__file__)


def _parse_args() -> argparse.Namespace:
    examples = (
        "Examples:\n"
        "  Single update:\n"
        '    python src/Scripts/UpdateConfigValues.py --config-root ./src/Configuration --conf Config_Internet_Env.py --replace_literal \'os.environ["WEB_SEARCH_MODE"] = "on"\' --replace_with \'os.environ["WEB_SEARCH_MODE"] = "off"\'\n\n'
        "  Bulk update via JSON string:\n"
        '    python src/Scripts/UpdateConfigValues.py --config-root ./src/Configuration --updates-json \'[{"conf":"Config_WebSearch.py","keyname":"_OPENWEB_UI_WEBSEARCH","value":"False"}]\'\n\n'
        "  Bulk update via JSON file:\n"
        "    python src/Scripts/UpdateConfigValues.py --config-root ./src/Configuration --updates-file ./updates.json\n\n"
        "Update dictionary keys:\n"
        "  Required: conf\n"
        "  Either: keyname + value, or replace_literal (+ optional replace_with)"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Update config values using either a JSON list of dictionaries "
            "or a single --conf/--keyname/--value call"
        ),
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--updates-json",
        help="JSON array with update dictionaries",
    )
    parser.add_argument(
        "--updates-file",
        help="Path to a JSON file containing the update list",
    )
    parser.add_argument(
        "--config-root",
        help="Base directory used to resolve relative values in 'conf'",
    )
    parser.add_argument(
        "--conf",
        help="Config name/path for single-update mode (or per-entry key in bulk mode)",
    )
    parser.add_argument(
        "--keyname",
        help="Key name (simple or dotted path) for single-update mode",
    )
    parser.add_argument(
        "--value",
        help="Python literal value for single-update mode",
    )
    return parser.parse_args()


def _replace_simple_assignment(content: str, slot_name: str, value: str) -> str:
    pattern = rf"(?m)^(\s*{re.escape(slot_name)}(?:\s*:\s*[\w\s,\[\]<>|]+?)?\s*=\s*).*$"
    replacement = rf"\g<1>{value}"
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Slot '{slot_name}' not found in config file.")
    return new_content


def _replace_dotted_path(content: str, slot_name: str, value: str) -> str:
    segments = slot_name.split(".")
    pos = 0

    for i, segment in enumerate(segments[:-1]):
        if i == 0:
            pattern = (
                rf"(?m)^\s*{re.escape(segment)}"
                rf"(?:\s*:\s*[\w\s,\[\]<>|]+?)?\s*=\s*\{{"
            )
        else:
            pattern = rf'"{re.escape(segment)}"\s*:\s*\{{'

        match = re.search(pattern, content[pos:], flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Segment '{segment}' not found in config file.")
        pos += match.end()

    final_segment = segments[-1]
    final_pattern = r'"' + re.escape(final_segment) + r'"\s*:\s*([^,\n}#]+)'
    match = re.search(final_pattern, content[pos:], flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Final key '{final_segment}' not found in config file.")

    start = pos + match.start(1)
    end = pos + match.end(1)
    return content[:start] + value + content[end:]


def _apply_single_update(content: str, update: dict[str, Any]) -> tuple[str, str]:
    if update.get("replace_literal") is not None:
        literal = str(update["replace_literal"])
        replace_with = str(update.get("replace_with", ""))
        if literal not in content:
            raise ValueError(f"Literal text not found: {literal!r}")
        return content.replace(literal, replace_with), "literal replacement"

    slot_name = update.get("slot_name")
    value = update.get("value")
    if not slot_name or value is None:
        raise ValueError("Each update needs either replace_literal or slot_name/value")

    slot_name_text = str(slot_name)
    value_text = str(value)
    if "." in slot_name_text and not slot_name_text.startswith("os.environ["):
        return _replace_dotted_path(content, slot_name_text, value_text), slot_name_text
    return (
        _replace_simple_assignment(content, slot_name_text, value_text),
        slot_name_text,
    )


def _load_updates(args: argparse.Namespace) -> list[dict[str, Any]]:
    has_bulk_mode = bool(args.updates_json or args.updates_file)
    has_single_mode = bool(args.conf or args.keyname or args.value is not None)

    if has_bulk_mode and has_single_mode:
        raise ValueError("Use either bulk mode or single mode, not both")

    if has_single_mode:
        if not args.conf or not args.keyname or args.value is None:
            raise ValueError("Single mode requires --conf --keyname --value")
        return [
            {
                "conf": args.conf,
                "keyname": args.keyname,
                "value": args.value,
            }
        ]

    if bool(args.updates_json) == bool(args.updates_file):
        raise ValueError(
            "Provide exactly one of --updates-json or --updates-file for bulk mode"
        )

    if args.updates_json:
        raw: Any = json.loads(args.updates_json)
    else:
        updates_file = Path(args.updates_file)
        raw = json.loads(updates_file.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError("Updates payload must be a JSON array")

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Every update entry must be a dictionary")

    return raw


def _resolve_config_file(conf: str, config_root: Path | None) -> Path:
    config_file = Path(conf)
    if config_file.is_absolute():
        return config_file
    if config_root:
        return config_root / config_file
    return config_file


def main() -> int:
    args = _parse_args()
    config_root = Path(args.config_root) if args.config_root else None

    try:
        updates = _load_updates(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"UpdateConfigValues error: {exc}", file=sys.stderr)
        return 1

    original_by_file: dict[Path, str] = {}
    current_by_file: dict[Path, str] = {}

    for index, update in enumerate(updates, start=1):
        conf = update.get("conf")
        if not conf:
            print(
                f"UpdateConfigValues error: update #{index} is missing 'conf'",
                file=sys.stderr,
            )
            return 1

        config_file = _resolve_config_file(str(conf), config_root)
        if not config_file.exists() or not config_file.is_file():
            print(f"Config file not found: {config_file}", file=sys.stderr)
            return 1

        if "slot_name" not in update and "keyname" in update:
            update["slot_name"] = update["keyname"]

        if config_file not in current_by_file:
            original_content = config_file.read_text(encoding="utf-8")
            original_by_file[config_file] = original_content
            current_by_file[config_file] = original_content

        try:
            updated, target = _apply_single_update(current_by_file[config_file], update)
        except ValueError as exc:
            print(
                f"UpdateConfigValues error in update #{index} for {config_file}: {exc}",
                file=sys.stderr,
            )
            return 1

        current_by_file[config_file] = updated
        print(f"Prepared update #{index}: {target} in {config_file}")

    for config_file, updated_content in current_by_file.items():
        if updated_content != original_by_file[config_file]:
            config_file.write_text(updated_content, encoding="utf-8")
            print(f"Updated {config_file}")
        else:
            print(f"Already up to date: {config_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
