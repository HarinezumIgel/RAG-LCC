#!/usr/bin/env python3
"""
PipInstall.py — install pinned Python dependencies with compliance notes.

This step installs Python packages from the pinned requirements file.

The signed release manifest covers:
  - requirements/requirements_final.txt
  - the generated pip-licenses evidence file
  - the hash of both files
  - the command/options used to generate the license evidence
  - the signature files

Run from the project root with the project virtual environment activated:
    python src/Scripts/PipInstall.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Refuse to run from a drive/filesystem root before any heavy imports
from Commons.DriveRootGuard import assert_not_drive_root  # noqa: E402

assert_not_drive_root(__file__)

_PROJECT_ROOT = _SRC_DIR.parent
_REQUIREMENTS = _PROJECT_ROOT / "requirements" / "requirements_final.txt"
_SETUP_REPORT_DIR = _PROJECT_ROOT / "ModelGovernance" / "consents"

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_WHITE = "\033[97m"

# ---------------------------------------------------------------------------
# Compliance note
# ---------------------------------------------------------------------------

_COMPLIANCE_NOTE = (
    "  This step installs Python packages from the pinned requirements file.\n"
    "\n"
    "  The signed release manifest covers:\n"
    "    · requirements/requirements_final.txt\n"
    "    · the generated pip-licenses evidence file\n"
    "    · the hash of both files\n"
    "    · the command/options used to generate the license evidence\n"
    "    · the signature files"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_required(command: list[str], label: str) -> None:
    print(f"{_DIM}  Running: {' '.join(command)}{_RESET}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print()
        print(f"{_RED}  ✖  {label} failed with exit code {result.returncode}.{_RESET}")
        sys.exit(result.returncode)


def _write_report(exit_code: int) -> None:
    """Write ModelGovernance/setup/pip_install_meta.json."""
    _SETUP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _SETUP_REPORT_DIR / "pip_install_meta.json"

    try:
        pip_version = (
            subprocess.check_output(
                [sys.executable, "-m", "pip", "--version"],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
    except Exception:
        pip_version = "unknown"

    report: dict = {
        "step": "pip_install",
        "requirements_file": (
            str(_REQUIREMENTS.relative_to(_PROJECT_ROOT))
            if _REQUIREMENTS.exists()
            else str(_REQUIREMENTS)
        ),
        "requirements_file_sha256": (
            _file_sha256(_REQUIREMENTS) if _REQUIREMENTS.exists() else None
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pip_version": pip_version,
        "installed_at": _utc_now(),
        "exit_code": exit_code,
        "success": exit_code == 0,
    }

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(
        f"{_GREEN}  ✔  pip install report written: {report_path.relative_to(_PROJECT_ROOT)}{_RESET}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -- compliance note ---------------------------------------------------
    print()
    print(f"{_DIM}{_COMPLIANCE_NOTE}{_RESET}")
    print()

    # -- preflight ---------------------------------------------------------
    if not _REQUIREMENTS.exists():
        print(f"{_RED}  ✖  Requirements file not found: {_REQUIREMENTS}{_RESET}")
        print(
            f"{_RED}     Aborting setup because Python dependencies cannot be installed.{_RESET}"
        )
        sys.exit(1)

    _run_required(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "pip upgrade",
    )

    print(f"{_DIM}  Installing Python packages into:{_RESET}")
    print(f"{_WHITE}  {sys.prefix}{_RESET}")
    print()

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(_REQUIREMENTS)]
    )
    _write_report(result.returncode)

    if result.returncode != 0:
        print()
        print(
            f"{_RED}  ✖  pip install failed with exit code {result.returncode}.{_RESET}"
        )
        print(
            f"{_RED}     Aborting setup to avoid a partially configured environment.{_RESET}"
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
