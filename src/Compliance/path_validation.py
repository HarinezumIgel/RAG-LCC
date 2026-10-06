"""Utilities for validating configuration path values."""

import os
import sys
from typing import NoReturn

from Commons.DriveRootGuard import is_drive_root, resolve_guard_path

_BRIGHT_RED = "\033[91m"
_RESET = "\033[0m"


def _configured_root_message(shown_path: str) -> str:
    """Return the canonical startup refusal text for root-path config values."""
    return (
        "EXECUTION BLOCKED: CONFIGURED _ABSOLUTE_PATH RESOLVES TO A DRIVE OR "
        f"FILESYSTEM ROOT ('{shown_path}'). FIX _ABSOLUTE_PATH IN "
        "src/Configuration/Config_Global.py."
    )


def _fail_validation(message: str) -> NoReturn:
    """Terminate execution with bright-red output for invalid config paths."""
    print(f"{_BRIGHT_RED}{message}{_RESET}", file=sys.stderr)
    raise SystemExit(1)


def validate_absolute_path(
    path_value: str,
    setting_name: str,
    caller_name: str,
) -> str:
    """Validate that a config path value is present, direct, and absolute."""
    absolute_path = str(path_value or "").strip()
    if not absolute_path:
        _fail_validation(
            f"{setting_name} is required by {caller_name} and must be defined."
        )
    if absolute_path.startswith("$"):
        _fail_validation(
            f"{setting_name} must be a direct absolute path; "
            "indirection values ('$...') are not allowed.",
        )
    if is_drive_root(absolute_path, resolve_traversal=False):
        shown = absolute_path or "<UNRESOLVED>"
        _fail_validation(_configured_root_message(shown))
    if not os.path.isabs(absolute_path):
        _fail_validation(
            f"{setting_name} must be an absolute path, got: '{absolute_path}'",
        )

    resolved_path = resolve_guard_path(absolute_path)
    if is_drive_root(resolved_path):
        shown = resolved_path or "<UNRESOLVED>"
        _fail_validation(_configured_root_message(shown))

    return absolute_path
