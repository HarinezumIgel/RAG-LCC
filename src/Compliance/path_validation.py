"""Utilities for validating configuration path values."""

import os


def validate_absolute_path(
    path_value: str,
    setting_name: str,
    caller_name: str,
) -> str:
    """Validate that a config path value is present, direct, and absolute."""
    absolute_path = str(path_value or "").strip()
    if not absolute_path:
        raise RuntimeError(
            f"{setting_name} is required by {caller_name} and must be defined."
        )
    if absolute_path.startswith("$"):
        raise RuntimeError(
            f"{setting_name} must be a direct absolute path; "
            "indirection values ('$...') are not allowed."
        )
    if not os.path.isabs(absolute_path):
        raise RuntimeError(
            f"{setting_name} must be an absolute path, got: '{absolute_path}'"
        )
    return absolute_path
