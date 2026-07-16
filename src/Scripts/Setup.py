#!/usr/bin/env python3
"""
Packages are installed from official package managers; licensing obligations remain with the original distributors and user environment.

PostContainerSetup.py – interactive one-shot setup wrapper run after container creation.

This script prepares the RAG-LCC environment inside the project virtual
environment.

Execution order:

  Preamble – Verify cache directory ownership
             Check that /home/vscode/.cache/ is owned by vscode user.
             Ask for permission and fix recursively if needed.

  Preamble – Display bundled third-party license information from
             3rdPartyLicenses/ before dependency installation.

  1. Install system packages
             Present bundled license texts for required system packages
             such as tesseract-ocr, collect user acceptance,
             record consent metadata, and install the packages with apt-get.

  2. Install Python dependencies
             Install Python runtime packages from
             requirements/requirements_final.txt into the active project
             virtual environment.

             Python dependency license evidence is generated separately with
             pip-licenses from the exact pinned package versions. The release
             manifest maps each pinned dependency version to its recorded
             license metadata and signed evidence files.

  3. Copy example configuration files
             Copy Example_*.py files from Examples/ into Configuration/
             where missing, so the application has default configuration
             files to start from.

  4. Recalculate configuration hashes
             Compute fresh SHA-256 hashes for Config_Models.py,
             Config_Banned.py, Config_WebSearch.py, and
             Config_Internet_Env.py, then write them into
             the corresponding *_CONFIG_HASH values in Config_Global.py.

  5. Install Argos Translate language packages
             Present the Argos Translate license/consent flow and download
             the language packages listed in ARGOS_LANGUAGES.

Run from the project root with the project virtual environment activated:

    source .venv/bin/activate
    python src/Scripts/PostContainerSetup.py

Integrity note:
    Signed release files allow users to verify that delivered requirements,
    license reports, manifests, setup scripts, and related deployment files
    have not changed since signing. Signature verification itself is expected
    to be performed by the release verification tooling.

Compliance note:
    This script records technical consent and setup metadata for auditability.
    It does not provide legal advice and does not replace legal review of
    third-party licenses where required.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re

try:
    import readline  # type: ignore  # noqa: F401  # Side-effect import: enables line editing in input()
except ImportError:
    pass  # readline is not available on Windows
import runpy
import shutil
import socket
import stat
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so child scripts can import project modules.
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_LICENSE_DIR = _PROJECT_ROOT / "3rdPartyLicenses"
_REQUIREMENTS = _PROJECT_ROOT / "requirements" / "requirements_final.txt"
_SETUP_LOG_DIR = _PROJECT_ROOT / "logs" / "setup"
_setup_log_path: Path | None = None
_CFG_DIR = _SRC_DIR / "Configuration"
_CONSENTS_DIR = _PROJECT_ROOT / "ModelGovernance" / "consents"
_PIP_INSTALL_META_PATH = _CONSENTS_DIR / "pip_install_meta.json"
_PY_REQ_CONSENT_PATH = _CONSENTS_DIR / "python_requirements_consent.json"

_devcontainer_modified: bool = False  # Track if devcontainer.json was changed


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
_WHITE = "\033[97m"
_ORANGE = "\033[38;2;255;165;0m"
_TURQUOISE = "\033[38;2;0;255;220m"

_TOTAL_STEPS = 6
_SECRET_FIELD_RE = re.compile(
    r"(secret|token|password|passwd|api[_-]?key|bearer|auth)",
    re.IGNORECASE,
)
# Sentinel returned by _prompt_secret_masked when the user explicitly types
# the word 'clear' to erase a previously stored credential.
_CLEAR_SENTINEL = "\x01"


class _AptPackageInfo(TypedDict):
    apt_name: str
    winget_id: str
    license_url_template: str
    license_dir: Path
    consent_dir: Path


# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------

_APT_PACKAGES: dict[str, _AptPackageInfo] = {
    "tesseract_ocr": {
        "apt_name": "tesseract-ocr",
        "winget_id": "UB-Mannheim.TesseractOCR",
        # License fetched from GitHub at the candidate version tag.
        "license_url_template": "https://raw.githubusercontent.com/tesseract-ocr/tesseract/{version}/LICENSE",
        "license_dir": _PROJECT_ROOT / "ModelGovernance" / "licenses" / "tesseract_ocr",
        "consent_dir": _PROJECT_ROOT / "ModelGovernance" / "consents" / "tesseract_ocr",
    },
}

_identity_cache: dict[str, Any] | None = None
_python_requirements_consent: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _make_scripts_executable(scripts_dir: Path) -> None:
    """Set executable permissions on Python scripts in the given directory.

    On Unix-like systems, sets the executable bit with chmod.
    On Windows, scripts can still be run with 'python script.py'.
    """
    if not scripts_dir.exists():
        return

    scripts = list(scripts_dir.glob("*.py"))
    if not scripts:
        return

    # On Unix-like systems, set executable permissions
    if os.name != "nt":
        for script in scripts:
            try:
                current_permissions = script.stat().st_mode
                script.chmod(
                    current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                )
            except Exception:
                # If chmod fails, continue anyway - scripts can still run with 'python script.py'
                pass


def _utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _banner() -> None:
    """Print the introductory banner shown once at startup."""
    w = 70
    print()
    print(f"{_CYAN}{_BOLD}{'=' * w}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'  RAG-LCC  —  Initial Setup':^{w}}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'=' * w}{_RESET}")
    print(f"{_DIM}  This wrapper runs {_TOTAL_STEPS} setup steps in sequence.{_RESET}")
    print(f"{_DIM}  Each step may ask questions — answer them as they appear.{_RESET}")
    print(f"{_CYAN}{_BOLD}{'=' * w}{_RESET}")


def _print_execution_plan() -> None:
    """Show exactly what the setup will do before execution starts."""
    w = 70
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  What Will Happen In Each Step{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(
        f"{_DIM}  Preamble:{_RESET} Verify file signatures to confirm shipped files have not been tampered with."
    )
    print(
        f"{_DIM}  Step 1:{_RESET} Install system OCR package (tesseract-ocr), show license text, require explicit consent, then install."
    )
    print(
        f"{_DIM}  Preamble:{_RESET} Show bundled Python dependency licenses from 3rdPartyLicenses/Licenses.txt and require explicit consent before pip install."
    )
    print(
        f"{_DIM}  Step 2:{_RESET} Install Python requirements from requirements/requirements_final.txt into the active virtual environment."
    )
    print(
        f"{_DIM}  Step 3:{_RESET} Copy Example_*.py files from Examples/ to src/Configuration/ (skips existing unless forced by child script options)."
    )
    print(
        f"{_DIM}  Step 5:{_RESET} Download NLTK stopwords + WordNet resources with consent flow for installed versions."
    )
    print(
        f"{_DIM}  Step 6:{_RESET} Optionally install Argos Translate language packages after explicit user confirmation."
    )
    print(
        f"{_DIM}  Runtime questions:{_RESET} After all downloads, ask for endpoint/internet/service settings (API keys masked), then write values to config files."
    )
    print(
        f"{_DIM}  Finalize:{_RESET} Recalculate and write critical config hashes in Config_Global.py."
    )
    print(f"{_CYAN}{'─' * w}{_RESET}")


def _ensure_project_root() -> None:
    """Abort unless the script appears to be running from the expected project."""
    if not _PROJECT_ROOT.exists():
        print(f"{_RED}  ✖  Project root not found: {_PROJECT_ROOT}{_RESET}")
        sys.exit(1)

    if not _SRC_DIR.exists():
        print(f"{_RED}  ✖  src directory not found: {_SRC_DIR}{_RESET}")
        sys.exit(1)


def _ensure_venv() -> None:
    """Abort unless a .venv directory exists and we're running inside a virtual environment."""
    venv_dir = _PROJECT_ROOT / ".venv"

    if not venv_dir.exists():
        print()
        print(f"{_YELLOW}  ⚠  No .venv found at {venv_dir}{_RESET}")
        print(
            f"{_DIM}     This is expected on a fresh container where the workspace{_RESET}"
        )
        print(
            f"{_DIM}     comes from the Docker image and .venv is not part of the repo.{_RESET}"
        )
        print()
        print(f"{_DIM}  Creating virtual environment at {venv_dir} ...{_RESET}")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "Permission denied" in result.stderr:
            print(
                f"{_YELLOW}  ⚠  Permission denied — workspace directory not writable by vscode.{_RESET}"
            )
            print(f"{_DIM}  Fixing ownership of {venv_dir.parent} with sudo...{_RESET}")
            fix = subprocess.run(
                ["sudo", "chown", "-R", "vscode:vscode", str(venv_dir.parent)],
                capture_output=True,
                text=True,
            )
            if fix.returncode != 0:
                print(f"{_RED}  ✖  sudo chown failed: {fix.stderr.strip()}{_RESET}")
                print(
                    f"{_YELLOW}  Run manually: sudo chown -R vscode:vscode {venv_dir.parent}{_RESET}"
                )
                sys.exit(1)
            print(f"{_GREEN}  ✔  Ownership fixed. Retrying venv creation...{_RESET}")
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            print(f"{_RED}  ✖  Failed to create virtual environment:{_RESET}")
            print(f"{_RED}     {result.stderr.strip()}{_RESET}")
            sys.exit(1)
        print(f"{_GREEN}  ✔  Virtual environment created: {venv_dir}{_RESET}")
        print()
        print(f"{_BOLD}{_YELLOW}  ➜  Activate it, then re-run Setup.py:{_RESET}")
        if _is_windows():
            print(f"{_WHITE}     .venv\\Scripts\\activate{_RESET}")
        else:
            print(f"{_WHITE}     source {venv_dir}/bin/activate{_RESET}")
        print(f"{_WHITE}     python src/Scripts/Setup.py{_RESET}")
        print()
        sys.exit(0)

    if sys.prefix == sys.base_prefix:
        # Not inside a venv — re-exec using the venv Python so the user
        # doesn't have to activate manually.
        if _is_windows():
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        if venv_python.exists():
            print(f"{_DIM}  Not inside venv — re-launching with {venv_python}{_RESET}")
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            # execv replaces the current process; code below is unreachable

        print()
        print(
            f"{_RED}  ✖  Setup must run inside the project virtual environment.{_RESET}"
        )
        print(f"{_YELLOW}     Activate it first, for example:{_RESET}")
        if _is_windows():
            print(f"{_WHITE}     .venv\\Scripts\\activate{_RESET}")
        else:
            print(f"{_WHITE}     source .venv/bin/activate{_RESET}")
        print()
        sys.exit(1)

    venv_path = Path(sys.prefix)
    print(f"{_DIM}  Active Python environment: {venv_path}{_RESET}")


def _ensure_cache_ownership() -> None:
    """Ensure /home/vscode/.cache/ is owned by vscode user."""
    cache_dir = Path("/home/vscode/.cache")

    # Skip if directory doesn't exist
    if not cache_dir.exists():
        return

    # Skip on Windows (no pwd module)
    if _is_windows():
        return

    # Check ownership
    try:
        import pwd

        stat_info = cache_dir.stat()
        owner = cast(str, pwd.getpwuid(stat_info.st_uid).pw_name)  # type: ignore[attr-defined]

        if owner == "vscode":
            return

        w = 70
        print()
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print(f"{_BOLD}{_YELLOW}  Cache Directory Ownership Issue Detected{_RESET}")
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print()
        print(f"{_WHITE}  Directory: {cache_dir}{_RESET}")
        print(f"{_WHITE}  Current owner: {owner}{_RESET}")
        print(f"{_WHITE}  Expected owner: vscode{_RESET}")
        print()
        print(
            f"{_DIM}  This directory is used for pip, Hugging Face, and other caches.{_RESET}"
        )
        print(f"{_DIM}  Running as the wrong user can cause permission errors.{_RESET}")
        print()

        if not _confirm(
            "Fix ownership by running 'sudo chown -R vscode:vscode /home/vscode/.cache'"
        ):
            print(
                f"{_YELLOW}  Skipping ownership fix. You may encounter permission errors.{_RESET}"
            )
            _write_setup_log(
                "cache_ownership_fix_declined",
                directory=str(cache_dir),
                current_owner=owner,
            )
            return

        print(f"{_DIM}  Running: sudo chown -R vscode:vscode {cache_dir}{_RESET}")
        result = subprocess.run(
            ["sudo", "chown", "-R", "vscode:vscode", str(cache_dir)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"{_RED}  ✖  Failed to change ownership: {result.stderr}{_RESET}")
            print(
                f"{_YELLOW}  Continuing anyway. You may encounter permission errors.{_RESET}"
            )
            _write_setup_log(
                "cache_ownership_fix_failed",
                directory=str(cache_dir),
                exit_code=result.returncode,
                stderr=result.stderr,
            )
        else:
            print(f"{_GREEN}  ✔  Cache directory ownership fixed.{_RESET}")
            _write_setup_log(
                "cache_ownership_fixed",
                directory=str(cache_dir),
                previous_owner=owner,
            )
    except ImportError:
        # pwd module not available (shouldn't happen on Linux/Unix)
        return
    except Exception as exc:
        print(f"{_YELLOW}  ⚠  Could not check cache directory ownership: {exc}{_RESET}")
        _write_setup_log(
            "cache_ownership_check_error",
            directory=str(cache_dir),
            error=str(exc),
        )
        return


def _compute_hash(text: str) -> str:
    """SHA-256 of canonicalised text."""
    return hashlib.sha256(
        text.replace("\r\n", "\n").strip().encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return SHA-256 for a file's exact bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _confirm(prompt: str) -> bool:
    """Ask the user a y/n question. Returns True for yes."""
    while True:
        answer = input(f"{_ORANGE}  {prompt} [y/n]: {_RESET}").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _test_endpoint_connectivity(endpoint: str) -> None:
    """Print curl test commands for the specified endpoint."""
    if endpoint == "ollama":
        port = "11434"
        test_path = "/api/tags"
    elif endpoint == "vllm":
        port = "4000"
        test_path = "/v1/models"
    else:  # openwebui
        port = "8080"
        test_path = "/"

    print(f"{_CYAN}  Test connectivity before continuing:{_RESET}")
    print(f"{_DIM}    # Docker Desktop (Windows/Mac):{_RESET}")
    print(
        f'{_DIM}    curl -v http://host.docker.internal:{port}{test_path} -H "Authorization: Bearer <API_KEY>"{_RESET}'
    )
    print()
    print(f"{_DIM}    # Linux container → host:{_RESET}")
    print(
        f'{_DIM}    curl -v http://172.17.0.1:{port}{test_path} -H "Authorization: Bearer <API_KEY>"{_RESET}'
    )
    print()
    print(f"{_DIM}    # Same machine:{_RESET}")
    print(
        f'{_DIM}    curl -v http://localhost:{port}{test_path} -H "Authorization: Bearer <API_KEY>"{_RESET}'
    )
    print()
    print(f"{_DIM}    # Another host on the network:{_RESET}")
    print(
        f'{_DIM}    curl -v http://<other host>:{port}{test_path} -H "Authorization: Bearer <API_KEY>"{_RESET}'
    )
    print()


def _print_endpoint_access_info(endpoint: str) -> None:
    """Print where the given endpoint can run, using a turquoise bullet list."""
    if endpoint == "ollama":
        title = "OLLAMA can run on"
        bullets = [
            ("This machine (localhost)", "http://localhost:11434/api/generate"),
            (
                "Docker Desktop host (Windows/Mac)",
                "http://host.docker.internal:11434/api/generate",
            ),
            (
                "Another server on your network",
                "http://<ollama host>:11434/api/generate",
            ),
        ]
    elif endpoint == "vllm":
        title = "vLLM can run on"
        bullets = [
            ("This machine (localhost)", "http://localhost:4000/v1/chat/completions"),
            (
                "Docker Desktop host (Windows/Mac)",
                "http://host.docker.internal:4000/v1/chat/completions",
            ),
            (
                "Another server on your network",
                "http://<vllm host>:4000/v1/chat/completions",
            ),
        ]
    else:  # openwebui
        title = "OpenWebUI GUI can run on"
        bullets = [
            ("This machine (localhost)", "http://localhost:8080"),
            ("Docker Desktop host (Windows/Mac)", "http://host.docker.internal:8080"),
            ("Another machine on the network", "http://<openwebui host>:8080"),
        ]

    print()
    print(f"{_TURQUOISE}  {title}:{_RESET}")
    for label, url in bullets:
        print(f"{_TURQUOISE}    \u2022 {label:<40} {url}{_RESET}")
    print()


def _run_required_command(command: list[str], label: str) -> None:
    """Run a command and abort setup if it fails."""
    print(f"{_DIM}  Running: {' '.join(command)}{_RESET}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print()
        print(f"{_RED}  ✖  {label} failed with exit code {result.returncode}.{_RESET}")
        print(
            f"{_RED}     Aborting setup to avoid a partially configured environment.{_RESET}"
        )
        sys.exit(result.returncode)


def _write_setup_log(event: str, **fields: object) -> None:
    """No-op: logging disabled."""
    pass


def _init_setup_log() -> None:
    """No-op: logging disabled."""
    pass


def _print_correction_hint(label: str, display_value: str) -> None:
    """Print current value and keep/replace instructions in correction mode."""
    print()
    print(f"  {_CYAN}◈  {label}{_RESET}")
    print(f"  {_DIM}   Current value:  {_CYAN}{_BOLD}{display_value}{_RESET}")
    print(
        f"  {_DIM}   Press Enter to keep it, or type a new value to replace it.{_RESET}"
    )
    print()


def _prompt_choice(
    prompt: str,
    choices: list[str],
    default: str,
    *,
    confirm_default: bool = True,
    correction_value: str | None = None,
) -> str:
    if correction_value is not None:
        _print_correction_hint(prompt, correction_value)
        default = correction_value
        confirm_default = False
    allowed = {c.lower(): c for c in choices}
    default_norm = default.lower()
    if default_norm not in allowed:
        raise ValueError(f"Default {default!r} is not in choices")

    while True:
        _label = "current" if correction_value is not None else "default"
        answer = input(
            f"{_ORANGE}  {prompt} [{'/'.join(choices)}] ({_label}: {default}): {_RESET}"
        ).strip()
        if not answer:
            if confirm_default:
                confirm = (
                    input(f"{_ORANGE}  Accept the default ({default})? [y/n]: {_RESET}")
                    .strip()
                    .lower()
                )
                if confirm in ("y", "yes"):
                    return allowed[default_norm]
                continue
            return allowed[default_norm]
        key = answer.lower()
        if key in allowed:
            return allowed[key]
        print(f"  Please choose one of: {', '.join(choices)}")


def _print_setting_context(
    config_name: str, config_rel_path: str, description: str
) -> None:
    """Print a short description and exact config target for a prompt."""
    print()
    print(f"{_DIM}    >> What: {description}{_RESET}")
    print(f"{_DIM}    Config: {config_name} ({config_rel_path}){_RESET}")


def _print_next_action_block(title: str, lines: list[tuple[str, str]]) -> None:
    """Print a framed next-action block in a consistent layout."""
    rule = "─" * 70
    print(rule)
    print(f"  {title}")
    for label, value in lines:
        print(f"  {label:<11}: {value}")
    print(rule)


def _prompt_bool(
    prompt: str,
    default: bool,
    *,
    confirm_default: bool = True,
    correction_value: bool | None = None,
) -> bool:
    if correction_value is not None:
        _print_correction_hint(prompt, "yes" if correction_value else "no")
        default = correction_value
        confirm_default = False
    opts = "Y/n" if default else "y/N"
    default_label = "Y=1" if default else "N=0"
    while True:
        answer = input(f"{_ORANGE}  {prompt} [{opts}]: {_RESET}").strip().lower()
        if not answer:
            if confirm_default:
                confirm = (
                    input(
                        f"{_ORANGE}  Accept the default ({default_label})? [y/n]: {_RESET}"
                    )
                    .strip()
                    .lower()
                )
                if confirm in ("y", "yes"):
                    return default
                continue
            return default
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("  Please answer y or n.")


def _prompt_text(
    prompt: str,
    default: str = "",
    *,
    secret: bool = False,
    confirm_default: bool = True,
    correction_value: str | None = None,
    no_yn: bool = False,
) -> str:
    _in_correction = correction_value is not None
    if _in_correction:
        display = (
            ("<set>" if correction_value else "<empty>") if secret else correction_value
        )
        _print_correction_hint(prompt, display)
        default = correction_value
        confirm_default = False
    if secret:
        while True:
            answer = _prompt_secret_masked(
                prompt, default, in_correction_mode=_in_correction
            )
            result = answer if answer else default
            if no_yn and result.lower() in ("y", "yes", "n", "no"):
                print(
                    f"  {_RED}✖  This field requires a specific value — "
                    f"'y' and 'n' are not valid here. Please enter again.{_RESET}"
                )
                continue
            return result

    while True:
        label = "current" if _in_correction else "default"
        answer = input(f"{_ORANGE}  {prompt} ({label}: {default}): {_RESET}").strip()
        if answer:
            if no_yn and answer.lower() in ("y", "yes", "n", "no"):
                print(
                    f"  {_RED}✖  This field requires a specific value — "
                    f"'y' and 'n' are not valid here. Please enter a URL, address, or key.{_RESET}"
                )
                continue
            return answer
        if confirm_default and default:
            confirm = (
                input(f"{_ORANGE}  Accept the default ({default})? [y/n]: {_RESET}")
                .strip()
                .lower()
            )
            if confirm in ("y", "yes"):
                return default
            continue
        return default


def _prompt_secret_masked(
    prompt: str, default: str, *, in_correction_mode: bool = False
) -> str:
    """Read secret input and print '*' for each entered character."""
    if os.name == "nt":
        import msvcrt

        sys.stdout.write(
            f"  {prompt} (leave empty to keep, type 'clear' to erase): "
            if (default or in_correction_mode)
            else f"  {prompt} (leave empty if you don't use a key): "
        )
        sys.stdout.flush()
        chars: list[str] = []
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                break
            if ch == "\003":
                raise KeyboardInterrupt
            if ch == "\b":
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch in ("\x00", "\xe0"):
                # Consume second char for special keys.
                _ = msvcrt.getwch()
                continue
            chars.append(ch)
            sys.stdout.write("*")
            sys.stdout.flush()
        answer = "".join(chars).strip()
        if answer.lower() == "clear":
            print(f"  {_YELLOW}  (credential cleared){_RESET}")
            return _CLEAR_SENTINEL
        return answer if answer else default

    # POSIX fallback
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write(
        f"  {prompt} (leave empty to keep, type 'clear' to erase): "
        if (default or in_correction_mode)
        else f"  {prompt} (leave empty for default): "
    )
    sys.stdout.flush()
    chars: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x7f", "\b"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            # Ignore ANSI escape sequences (arrow keys, etc.)
            if ch == "\x1b":  # ESC character
                # Read and discard the rest of the escape sequence
                next_ch = sys.stdin.read(1)
                if next_ch == "[":
                    # CSI sequence - read until we get a letter
                    while True:
                        seq_ch = sys.stdin.read(1)
                        if seq_ch.isalpha():
                            break
                continue
            # Ignore other control characters except printable ones
            if ord(ch) < 32:
                continue
            chars.append(ch)
            sys.stdout.write("*")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    answer = "".join(chars).strip()
    if answer.lower() == "clear":
        print(f"  {_YELLOW}  (credential cleared){_RESET}")
        return _CLEAR_SENTINEL
    return answer if answer else default


def _as_py_string(value: str) -> str:
    return json.dumps(value)


def _apply_config_updates(updates: list[dict[str, str]]) -> None:
    applicable_updates: list[dict[str, str]] = []
    for update in updates:
        conf = (update.get("conf") or "").strip()
        slot_name = (update.get("slot_name") or "").strip()
        if not conf:
            print(
                f"{_ORANGE}  WARN   Skipping config update without 'conf' field.{_RESET}"
            )
            continue

        config_path = _CFG_DIR / conf
        if not config_path.exists() or not config_path.is_file():
            print(
                f"{_ORANGE}  WARN   {conf} not updated because file is not present in {_CFG_DIR}.{_RESET}"
            )
            continue

        # No self-heal in setup: required config slots must exist already.
        if conf == "Config_Models.py" and slot_name == "_HF_API_KEY":
            try:
                content = config_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read {config_path} while validating _HF_API_KEY slot: {exc}"
                ) from exc
            if not re.search(r"(?m)^\s*_HF_API_KEY\s*=", content):
                raise RuntimeError(
                    "Config_Models.py is missing _HF_API_KEY. "
                    "Add '_HF_API_KEY = \"\"' manually in src/Configuration/Config_Models.py "
                    "and run setup again."
                )

        applicable_updates.append(update)

    if not applicable_updates:
        print(
            f"{_ORANGE}  WARN   No runtime config updates were applied because no target config files were present.{_RESET}"
        )
        return

    updater = _SRC_DIR / "Scripts" / "UpdateConfigValues.py"
    cmd = [
        sys.executable,
        str(updater),
        "--config-root",
        str(_CFG_DIR),
        "--updates-json",
        json.dumps(applicable_updates),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("UpdateConfigValues.py failed during setup questions")


def _run_setup_questions() -> None:
    """Ask interactive runtime questions and update config files."""
    w = 70
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Runtime configuration questions{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()
    print(f"{_BOLD}{_YELLOW}  ⚠️  Default settings implications:{_RESET}")
    print(
        f"{_DIM}  The suggested defaults configure RAG-LCC for offline/air-gapped operation:{_RESET}"
    )
    print()
    print(
        f"{_DIM}    • Internet access DISABLED (HF_HUB_OFFLINE=1, LICENSE_DOWNLOAD=0){_RESET}"
    )
    print(f"{_DIM}    • OpenWebUI connection DISABLED (SERVE_OPENWEBUI_CHAT=0){_RESET}")
    print(f"{_DIM}    • Marked document serving DISABLED{_RESET}")
    print(f"{_DIM}    • Hugging Face models must be pre-installed locally{_RESET}")
    print(f"{_DIM}    • Model license consent asked on-demand at first use{_RESET}")
    print(
        f"{_DIM}    • Default Argos languages installed by this script; new languages{_RESET}"
    )
    print(f"{_DIM}      will NOT be auto-installed on-the-fly{_RESET}")
    print(f"{_DIM}    • NLTK/WordNet corpora must be pre-downloaded{_RESET}")
    print(
        f"{_DIM}    • Web search features available only if enabled via WEB_SEARCH_MODE{_RESET}"
    )
    print()
    print(
        f"{_DIM}  To enable internet-connected features, answer accordingly below.{_RESET}"
    )
    print()
    print(
        f"{_DIM}  Hint: If HF_HUB_OFFLINE=1, required Hugging Face models must be installed locally.{_RESET}"
    )
    print(
        f"{_DIM}  Otherwise model downloads from the Hugging Face Hub are blocked at runtime.{_RESET}"
    )
    print()
    print(
        f"{_DIM}  Tip: To avoid re-downloading Hugging Face models on every container restart,{_RESET}"
    )
    print(
        f"{_DIM}  mount the host cache directory into the container as a bind volume:{_RESET}"
    )
    print(
        f'{_DIM}    Linux/macOS: -v "$HOME/.cache/huggingface:/home/vscode/.cache/huggingface"{_RESET}'
    )
    print(
        f'{_DIM}    Windows:     -v "%USERPROFILE%\\.cache\\huggingface:/home/vscode/.cache/huggingface"{_RESET}'
    )
    print(
        f"{_DIM}  Or set HF_HOME to a custom path if your cache is stored elsewhere.{_RESET}"
    )
    print(
        f"{_DIM}  See INSTALL.md (Docker section) for the devcontainer.json mount snippet.{_RESET}"
    )

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    input(
        f"{_BOLD}{_ORANGE}  Press Enter to start the interactive configuration...{_RESET}"
    )
    print()

    correction_mode = False

    # Initialize all variables for correction mode
    endpoint = "ollama"
    endpoint_url = ""
    endpoint_api_key = ""
    rag_chat_service_listener = ""
    rag_chat_service_listener_port = ""
    openwebui_base_url = ""
    openwebui_api_key = ""
    hf_hub_offline = True
    hf_api_key = ""
    argos_stanza_download = False
    nltk_stopwords_download = False
    license_download = False
    web_search_mode = "0"
    openweb_ui_websearch = False
    serve_openwebui_chat = False
    serve_in_memory_docs = False
    network_tracer = False

    while True:
        # ──────────────────────────────────────────────────────────────
        # Group 1: Model endpoint configuration
        # ──────────────────────────────────────────────────────────────
        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Select which model endpoint backend is active for runtime requests.",
        )
        endpoint = _prompt_choice(
            "Select _ACTIVE_ENDPOINT",
            ["ollama", "vllm"],
            "ollama",
            correction_value=endpoint if correction_mode else None,
        )

        _test_endpoint_connectivity(endpoint)
        print()
        print(f"{_RED}{'='*70}{_RESET}")
        print(
            f"{_RED}  ⚠️  IMPORTANT: Specify where {endpoint.upper()} is running{_RESET}"
        )
        print(f"{_RED}{'='*70}{_RESET}")
        _print_endpoint_access_info(endpoint)
        print(
            f"{_ORANGE}  You MUST enter the actual IP address or hostname where{_RESET}"
        )
        print(
            f"{_ORANGE}  {endpoint.upper()} is listening. Do NOT just accept the placeholder.{_RESET}"
        )
        print(f"{_RED}{'='*70}{_RESET}")
        print()
        default_endpoint_url = (
            f"http://<ollama host>:11434/api/generate"
            if endpoint == "ollama"
            else f"http://<vllm host>:4000/v1/chat/completions"
        )
        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Base URL used to call the selected endpoint provider.",
        )
        endpoint_url = _prompt_text(
            f"Set {endpoint.upper()} BASE_URL",
            default_endpoint_url,
            no_yn=True,
            correction_value=endpoint_url if correction_mode else None,
        )
        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "API key for the selected endpoint (if required by that endpoint).",
        )
        endpoint_api_key = _prompt_text(
            f"Set {endpoint.upper()} API key",
            "",
            secret=True,
            no_yn=True,
            correction_value=endpoint_api_key if correction_mode else None,
        )

        # RAGChatService host configuration
        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Specify where RAGChatService HTTP listener should bind (_MODELS.ragchatservice._RAGCHATSERVICE.HOST).",
        )

        print()
        print(
            f"{_CYAN}  RAGChatService listens for OpenWebUI connections on port 11435.{_RESET}"
        )
        print(
            f"{_TURQUOISE}    Bind address depends on where RAGChatService runs:{_RESET}"
        )
        print(
            f"{_TURQUOISE}    • Host machine:                       localhost or 127.0.0.1{_RESET}"
        )
        print(f"{_TURQUOISE}    • Docker container (host can reach):  0.0.0.0{_RESET}")
        print(
            f"{_TURQUOISE}    • OpenWebUI on another machine:       0.0.0.0 or specific IP{_RESET}"
        )
        print()
        print(f"{_CYAN}  Port forwarding — port 11435:{_RESET}")
        print(
            f"{_TURQUOISE}    Port 11435 is declared in .devcontainer/devcontainer.json{_RESET}"
        )
        print(
            f"{_TURQUOISE}    but VS Code does not always forward it automatically.{_RESET}"
        )
        print()
        print(f"{_TURQUOISE}    To forward it manually in VS Code:{_RESET}")
        print(
            f"{_TURQUOISE}      1. Open the Ports panel  (View → Ports  or  Ctrl+Shift+P → 'Focus on Ports'){_RESET}"
        )
        print(f"{_TURQUOISE}      2. Click '+ Forward a Port'{_RESET}")
        print(f"{_TURQUOISE}      3. Enter port 11435  and press Enter{_RESET}")
        print(
            f"{_TURQUOISE}    The port then appears in the Ports panel and OpenWebUI can connect.{_RESET}"
        )
        print()

        default_openwebui_api_host = "<RAGChatService host>"
        rag_chat_service_listener = _prompt_text(
            "Set _MODELS.ragchatservice._RAGCHATSERVICE.HOST",
            default_openwebui_api_host,
            no_yn=True,
            correction_value=rag_chat_service_listener if correction_mode else None,
        )

        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Port where RAGChatService HTTP listener should bind (_MODELS.ragchatservice._RAGCHATSERVICE.PORT).",
        )
        rag_chat_service_listener_port = _prompt_text(
            "Set _MODELS.ragchatservice._RAGCHATSERVICE.PORT",
            "11435",
            no_yn=True,
            correction_value=(
                rag_chat_service_listener_port if correction_mode else None
            ),
        )

        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Base URL where OpenWebUI is running (used by Informer to verify OpenWebUI is reachable).",
        )

        _print_endpoint_access_info("openwebui")

        default_openwebui_url = "http://<openwebui host>:8080"
        openwebui_base_url = _prompt_text(
            "Set OpenWebUI BASE_URL",
            default_openwebui_url,
            no_yn=True,
            correction_value=openwebui_base_url if correction_mode else None,
        )

        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Bearer API key expected by RAGChatService for incoming OpenWebUI requests (_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY).",
        )
        openwebui_api_key = _prompt_text(
            "Set _MODELS.ragchatservice._RAGCHATSERVICE.API_KEY",
            "",
            secret=True,
            no_yn=True,
            correction_value=openwebui_api_key if correction_mode else None,
        )

        # ──────────────────────────────────────────────────────────────
        # Group 2: Hugging Face model hub configuration
        # ──────────────────────────────────────────────────────────────
        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Hugging Face hub access gate: offline=1, online=0.",
        )
        hf_hub_offline = _prompt_bool(
            "Enable HF_HUB_OFFLINE",
            default=True,
            correction_value=hf_hub_offline if correction_mode else None,
        )

        _print_setting_context(
            "Config_Models.py",
            "src/Configuration/Config_Models.py",
            "Optional global Hugging Face API key used for model downloads when set.",
        )
        print(
            f"{_DIM}    Note: model-specific HF_API_KEY values can be set directly in Config_Models.py.{_RESET}"
        )
        hf_api_key = _prompt_text(
            "Set global _HF_API_KEY",
            "",
            secret=True,
            no_yn=True,
            correction_value=hf_api_key if correction_mode else None,
        )

        # ──────────────────────────────────────────────────────────────
        # Group 3: Language resources and corpus downloads
        # ──────────────────────────────────────────────────────────────
        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Enable Argos package/license download prompt at startup.",
        )
        argos_stanza_download = _prompt_bool(
            "Enable ARGOS_STANZA_DOWNLOAD",
            default=False,
            correction_value=argos_stanza_download if correction_mode else None,
        )

        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Enable NLTK stopwords/WordNet download prompt at startup.",
        )
        nltk_stopwords_download = _prompt_bool(
            "Enable NLTK_STOPWORDS_DOWNLOAD",
            default=False,
            correction_value=nltk_stopwords_download if correction_mode else None,
        )

        # ──────────────────────────────────────────────────────────────
        # Group 4: Network and compliance settings
        # ──────────────────────────────────────────────────────────────
        print()
        print(
            f"{_ORANGE}⚠️  WARNING: Enabling LICENSE_DOWNLOAD will fetch license files at every app run.{_RESET}"
        )
        print(
            f"{_ORANGE}   Recommendation: Keep the default 'N' unless you need fresh license checks each time.{_RESET}"
        )
        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Check licenses online on every run.",
        )
        license_download = _prompt_bool(
            "Enable LICENSE_DOWNLOAD",
            default=False,
            correction_value=license_download if correction_mode else None,
        )

        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Master switch for web search: 0 (off) or 1 (on).",
        )
        _web_search_enabled = _prompt_bool(
            "Enable WEB_SEARCH_MODE",
            default=False,
            correction_value=(web_search_mode == "1") if correction_mode else None,
        )
        web_search_mode = "1" if _web_search_enabled else "0"

        if web_search_mode == "1":
            _print_setting_context(
                "Config_WebSearch.py",
                "src/Configuration/Config_WebSearch.py",
                "Default web-search state for new OpenWebUI sessions.",
            )
            openweb_ui_websearch = _prompt_bool(
                "Enable _OPENWEB_UI_WEBSEARCH",
                default=False,
                correction_value=openweb_ui_websearch if correction_mode else None,
            )
        else:
            openweb_ui_websearch = False
            print(
                f"{_YELLOW}  WEB_SEARCH_MODE is 0, so _OPENWEB_UI_WEBSEARCH is forced to False.{_RESET}"
            )

        # ──────────────────────────────────────────────────────────────
        # Group 5: Service endpoints
        # ──────────────────────────────────────────────────────────────
        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Enable OpenWebUI-compatible chat service (RAGChatService HTTP API).",
        )
        serve_openwebui_chat = _prompt_bool(
            "Enable SERVE_OPENWEBUI_CHAT",
            default=False,
            correction_value=serve_openwebui_chat if correction_mode else None,
        )
        if not serve_openwebui_chat:
            serve_in_memory_docs = False
            print(
                f"{_YELLOW}  SERVE_OPENWEBUI_CHAT is disabled, so SERVE_IN_MEMORY_DOCS_HTTP is forced to 0.{_RESET}"
            )
        else:
            _print_setting_context(
                "Config_Internet_Env.py",
                "src/Configuration/Config_Internet_Env.py",
                "Enable in-memory docs HTTP token server for /marked (grounded documents) links.",
            )
            serve_in_memory_docs = _prompt_bool(
                "Enable SERVE_IN_MEMORY_DOCS_HTTP",
                default=False,
                correction_value=serve_in_memory_docs if correction_mode else None,
            )

        # ──────────────────────────────────────────────────────────────
        # Group 6: Debug and tracing
        # ──────────────────────────────────────────────────────────────
        _print_setting_context(
            "Config_Internet_Env.py",
            "src/Configuration/Config_Internet_Env.py",
            "Enable network-level socket tracing for debugging.",
        )
        network_tracer = _prompt_bool(
            "Enable RAG_LCC_NW_TRACE",
            default=False,
            correction_value=network_tracer if correction_mode else None,
        )

        print()
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print(f"{_BOLD}{_TURQUOISE}  Review your selected runtime settings{_RESET}")
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print(f"{_WHITE}  _ACTIVE_ENDPOINT:{_RESET} {endpoint}")
        print(f"{_WHITE}  {endpoint.upper()} BASE_URL:{_RESET} {endpoint_url}")
        print(
            f"{_WHITE}  {endpoint.upper()} API key:{_RESET} {'<set>' if endpoint_api_key else '<empty>'}"
        )
        print(f"{_WHITE}  HF_HUB_OFFLINE:{_RESET} {'1' if hf_hub_offline else '0'}")
        print(f"{_WHITE}  _HF_API_KEY:{_RESET} {'<set>' if hf_api_key else '<empty>'}")
        print(
            f"{_WHITE}  ARGOS_STANZA_DOWNLOAD:{_RESET} {'1' if argos_stanza_download else '0'}"
        )
        print(
            f"{_WHITE}  NLTK_STOPWORDS_DOWNLOAD:{_RESET} {'1' if nltk_stopwords_download else '0'}"
        )
        print(f"{_WHITE}  LICENSE_DOWNLOAD:{_RESET} {'1' if license_download else '0'}")
        print(f"{_WHITE}  WEB_SEARCH_MODE:{_RESET} {web_search_mode}")
        print(
            f"{_WHITE}  _OPENWEB_UI_WEBSEARCH:{_RESET} {'1' if openweb_ui_websearch else '0'}"
        )
        print(
            f"{_WHITE}  SERVE_OPENWEBUI_CHAT:{_RESET} {'1' if serve_openwebui_chat else '0'}"
        )
        print(
            f"{_WHITE}  SERVE_IN_MEMORY_DOCS_HTTP:{_RESET} {'1' if serve_in_memory_docs else '0'}"
        )
        print(
            f"{_WHITE}  _MODELS.ragchatservice._RAGCHATSERVICE.HOST:{_RESET} {rag_chat_service_listener}"
        )
        print(
            f"{_WHITE}  _MODELS.ragchatservice._RAGCHATSERVICE.PORT:{_RESET} {rag_chat_service_listener_port}"
        )
        print(f"{_WHITE}  OpenWebUI BASE_URL:{_RESET} {openwebui_base_url}")
        print(
            f"{_WHITE}  _MODELS.ragchatservice._RAGCHATSERVICE.API_KEY:{_RESET} {'<set>' if openwebui_api_key else '<empty>'}"
        )
        print(f"{_WHITE}  RAG_LCC_NW_TRACE:{_RESET} {'1' if network_tracer else '0'}")
        print(f"{_CYAN}{'─' * w}{_RESET}")

        next_action = _prompt_choice(
            "Choose: 1=Continue, 2=Correct a value, 3=Start again",
            ["1", "2", "3"],
            "1",
            confirm_default=False,
        )
        if next_action == "1":
            break

        correction_mode = next_action == "2"
        if correction_mode:
            print(
                f"{_YELLOW}  Re-running — your previously entered values are shown for each setting.{_RESET}"
            )
        else:
            print(
                f"{_YELLOW}  Re-running runtime configuration questions from scratch...{_RESET}"
            )

    updates = [
        {
            "conf": "Config_Models.py",
            "slot_name": "_ACTIVE_ENDPOINT",
            "value": _as_py_string(endpoint),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": f"_MODELS.{endpoint}.{('_OLLAMA' if endpoint == 'ollama' else '_VLLM')}.BASE_URL",
            "value": _as_py_string(endpoint_url),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": f"_MODELS.{endpoint}.{('_OLLAMA' if endpoint == 'ollama' else '_VLLM')}.API_KEY",
            "value": _as_py_string(endpoint_api_key),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["HF_HUB_OFFLINE"]',
            "value": _as_py_string("1" if hf_hub_offline else "0"),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": "_HF_API_KEY",
            "value": _as_py_string(hf_api_key),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["LICENSE_DOWNLOAD"]',
            "value": _as_py_string("1" if license_download else "0"),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["WEB_SEARCH_MODE"]',
            "value": _as_py_string(web_search_mode),
        },
        {
            "conf": "Config_WebSearch.py",
            "slot_name": "_OPENWEB_UI_WEBSEARCH",
            "value": str(openweb_ui_websearch),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["ARGOS_STANZA_DOWNLOAD"]',
            "value": _as_py_string("1" if argos_stanza_download else "0"),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["NLTK_STOPWORDS_DOWNLOAD"]',
            "value": _as_py_string("1" if nltk_stopwords_download else "0"),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["SERVE_OPENWEBUI_CHAT"]',
            "value": _as_py_string("1" if serve_openwebui_chat else "0"),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["SERVE_IN_MEMORY_DOCS_HTTP"]',
            "value": _as_py_string("1" if serve_in_memory_docs else "0"),
        },
        {
            "conf": "Config_Internet_Env.py",
            "slot_name": 'os.environ["RAG_LCC_NW_TRACE"]',
            "value": _as_py_string("1" if network_tracer else "0"),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": "_MODELS.ragchatservice._RAGCHATSERVICE.HOST",
            "value": _as_py_string(rag_chat_service_listener),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": "_MODELS.ragchatservice._RAGCHATSERVICE.PORT",
            "value": rag_chat_service_listener_port,
        },
        {
            "conf": "Config_Models.py",
            "slot_name": "_MODELS.openwebui._OPENWEBUI.BASE_URL",
            "value": _as_py_string(openwebui_base_url),
        },
        {
            "conf": "Config_Models.py",
            "slot_name": "_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY",
            "value": _as_py_string(openwebui_api_key),
        },
    ]

    _apply_config_updates(updates)

    if rag_chat_service_listener_port != "11435":
        print()
        print(
            f"{_BOLD}{_ORANGE}  ⚠️  Non-default listener port configured: {rag_chat_service_listener_port}{_RESET}"
        )
        print(
            f"{_ORANGE}  You must forward this port manually — Setup no longer writes devcontainer.json.{_RESET}"
        )
        print(f"{_DIM}  Options:{_RESET}")
        print(
            f"{_DIM}    • VS Code GUI: Ports panel → Forward a Port → enter {rag_chat_service_listener_port}{_RESET}"
        )
        print(
            f'{_DIM}    • devcontainer.json: add {rag_chat_service_listener_port} to "forwardPorts" and rebuild the container{_RESET}'
        )
        print()

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Values written to config{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    for _u in updates:
        _conf = _u.get("conf", "")
        _key = _u.get("slot_name", "")
        _val = _u.get("value", "")
        _display = "<redacted>" if _SECRET_FIELD_RE.search(_key) else _val
        print(f"{_DIM}    {_conf:<30}  {_key:<45}  = {_display}{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    _write_setup_log(
        "runtime_questions_applied",
        active_endpoint=endpoint,
        endpoint_base_url=endpoint_url,
        endpoint_api_key_set=bool(endpoint_api_key),
        hf_api_key_set=bool(hf_api_key),
        hf_hub_offline=hf_hub_offline,
        license_download=license_download,
        web_search_mode=web_search_mode,
        openweb_ui_websearch=openweb_ui_websearch,
        argos_stanza_download=argos_stanza_download,
        nltk_stopwords_download=nltk_stopwords_download,
        serve_openwebui_chat=serve_openwebui_chat,
        serve_in_memory_docs_http=serve_in_memory_docs,
        rag_lcc_nw_trace=network_tracer,
        openwebui_api_host=rag_chat_service_listener,
        openwebui_api_port=rag_chat_service_listener_port,
        openwebui_base_url=openwebui_base_url,
        openwebui_api_key_set=bool(openwebui_api_key),
    )
    print(f"{_GREEN}  ✔  Runtime settings updated from setup answers.{_RESET}")


# ---------------------------------------------------------------------------
# Identity and consent recording
# ---------------------------------------------------------------------------


def _capture_identity(require_confirmation: bool = False) -> dict[str, Any]:
    """Capture accepting identity, optionally requiring explicit re-confirmation."""
    global _identity_cache

    if _identity_cache is not None and not require_confirmation:
        return _identity_cache

    if _identity_cache is not None and require_confirmation:
        cached = _identity_cache
        accepted_by = (
            str(cached.get("accepted_by", "unknown-user")).strip() or "unknown-user"
        )
        source = str(cached.get("accepted_by_source", "os")).strip() or "os"
        print(f"  Detected identity: {accepted_by}  (source: {source})")
        override = input(
            f"{_ORANGE}  Press Enter to approve this identity, or type your email/ID to override: {_RESET}"
        ).strip()

        if override:
            accepted_by = override
            source = "interactive"

        host = (socket.gethostname() or "").strip() or "unknown-host"
        _identity_cache = {
            "accepted_by": accepted_by,
            "accepted_by_source": source,
            "accepted_by_verified": False,
            "host": host,
            "pid": os.getpid(),
        }
        return _identity_cache

    git_user: str | None = None

    try:
        git_user = (
            subprocess.check_output(
                ["git", "config", "user.email"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        ) or None
    except Exception:
        pass

    try:
        os_user = getpass.getuser()
    except Exception:
        os_user = os.getenv("USER") or os.getenv("USERNAME") or "unknown-user"

    accepted_by = (git_user or os_user or "").strip()
    source = "git" if git_user else "os"

    if not accepted_by:
        accepted_by = "unknown-user"
        source = "os"

    print(f"  Detected identity: {accepted_by}  (source: {source})")
    override = input(
        f"{_ORANGE}  Press Enter to accept as-is, or type your email/ID to override: {_RESET}"
    ).strip()

    if override:
        accepted_by = override
        source = "interactive"

    accepted_by = (accepted_by or "").strip() or "unknown-user"
    source = (source or "").strip() or "os"
    host = (socket.gethostname() or "").strip() or "unknown-host"

    _identity_cache = {
        "accepted_by": accepted_by,
        "accepted_by_source": source,
        "accepted_by_verified": False,
        "host": host,
        "pid": os.getpid(),
    }

    return _identity_cache


def _record_apt_consent(
    pkg_key: str,
    info: _AptPackageInfo,
    text: str,
    license_url: str,
    version: str,
    identity: dict[str, Any],
    now: str,
    install_exit_code: int,
    installer: str = "apt-get",
) -> None:
    """Write license copy, license_meta.json and install_meta.json."""
    lic_dir: Path = info["license_dir"]
    consent_dir: Path = info["consent_dir"]
    apt_name: str = info["apt_name"]

    lic_path = lic_dir / "LICENSE.txt"
    lic_meta_path = lic_dir / "license_meta.json"
    consent_path = consent_dir / "install_meta.json"

    lic_hash = _compute_hash(text)

    lic_dir.mkdir(parents=True, exist_ok=True)
    lic_path.write_text(text, encoding="utf-8")

    lic_meta: dict[str, Any] = {
        "component": apt_name,
        "component_key": pkg_key,
        "version": version,
        "license_url": license_url,
        "license_hash_text_canonical_sha256": lic_hash,
        **identity,
        "accepted_at": now,
        "consent": True,
    }

    with lic_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(lic_meta, fh, indent=2, ensure_ascii=False)

    consent_dir.mkdir(parents=True, exist_ok=True)

    install_meta: dict[str, Any] = {
        "component": apt_name,
        "component_key": pkg_key,
        "version": version,
        "installer": installer,
        "installer_exit_code": install_exit_code,
        "installed": install_exit_code == 0,
        **identity,
        "accepted_at": now,
        "consent": True,
    }

    with consent_path.open("w", encoding="utf-8") as fh:
        json.dump(install_meta, fh, indent=2, ensure_ascii=False)

    print(
        f"{_GREEN}  ✔  Consent recorded: {consent_path.relative_to(_PROJECT_ROOT)}{_RESET}"
    )


def _record_python_requirements_consent(identity: dict[str, Any]) -> None:
    """Persist explicit consent for Python requirements/license review."""
    global _python_requirements_consent

    accepted_at = _utc_now()
    requirements_rel = str(_REQUIREMENTS.relative_to(_PROJECT_ROOT))
    requirements_sha256 = (
        _file_sha256(_REQUIREMENTS) if _REQUIREMENTS.exists() else "missing"
    )

    payload: dict[str, Any] = {
        "step": "python_requirements_license_consent",
        "requirements_file": requirements_rel,
        "requirements_file_sha256": requirements_sha256,
        "consent": True,
        "accepted_at": accepted_at,
        **identity,
    }

    _CONSENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _PY_REQ_CONSENT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    _python_requirements_consent = payload
    print(
        f"{_GREEN}  ✔  Consent recorded: {_PY_REQ_CONSENT_PATH.relative_to(_PROJECT_ROOT)}{_RESET}"
    )
    _write_setup_log(
        "python_requirements_consent_recorded",
        consent_path=str(_PY_REQ_CONSENT_PATH),
        accepted_by=identity.get("accepted_by", "unknown-user"),
    )


def _enrich_pip_install_meta_with_consent() -> None:
    """Merge recorded consent/identity into pip_install_meta.json when available."""
    if _python_requirements_consent is None:
        return
    if not _PIP_INSTALL_META_PATH.exists():
        return

    try:
        with _PIP_INSTALL_META_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(data, dict):
        return

    data_dict = cast(dict[str, Any], data)

    data_dict["consent"] = True
    data_dict["accepted_at"] = _python_requirements_consent.get(
        "accepted_at", _utc_now()
    )
    data_dict["accepted_by"] = _python_requirements_consent.get(
        "accepted_by", "unknown-user"
    )
    data_dict["accepted_by_source"] = _python_requirements_consent.get(
        "accepted_by_source", "os"
    )
    data_dict["accepted_by_verified"] = _python_requirements_consent.get(
        "accepted_by_verified", False
    )
    data_dict["host"] = _python_requirements_consent.get(
        "host", socket.gethostname() or "unknown-host"
    )
    data_dict["requirements_consent_source"] = "Setup.py"

    try:
        with _PIP_INSTALL_META_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data_dict, fh, indent=2, ensure_ascii=False)
        _write_setup_log(
            "pip_install_meta_enriched",
            meta_path=str(_PIP_INSTALL_META_PATH),
        )
    except OSError:
        return


# ---------------------------------------------------------------------------
# License viewer
# ---------------------------------------------------------------------------


def _collect_license_text() -> str:
    """Return the contents of 3rdPartyLicenses/Licenses.txt."""
    licenses_file = _LICENSE_DIR / "Licenses.txt"

    if not licenses_file.exists():
        return "(3rdPartyLicenses/Licenses.txt not found.)"

    try:
        return licenses_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(Could not read Licenses.txt: {exc})"


_END_OF_LICENSE = (
    "\n\n" + "─" * 70 + "\n" "  >>>>>  End of license  <<<<<\n" + "─" * 70 + "\n"
)


def _page_text(text: str, *, end_marker: bool = False) -> None:
    """Display text using less if available, otherwise a small fallback pager."""
    if end_marker:
        text = text + _END_OF_LICENSE
    if shutil.which("less"):
        env = {**os.environ, "LESS": "FRX"}
        try:
            subprocess.run(["less", "-"], input=text, text=True, env=env)
            return
        except OSError:
            pass

    try:
        rows = shutil.get_terminal_size().lines - 2
    except Exception:
        rows = 20

    lines = text.splitlines()
    idx = 0

    while idx < len(lines):
        print("\n".join(lines[idx : idx + rows]))
        idx += rows

        if idx < len(lines):
            key = (
                input(f"{_ORANGE}[Enter] next page  [q] quit view: {_RESET}")
                .strip()
                .lower()
            )
            if key == "q":
                break


def _get_editor_command() -> list[str]:
    """Return platform-appropriate editor command for opening files."""
    if _is_windows():
        # On Windows, try notepad.exe
        return ["notepad.exe"]
    else:
        # On Unix, prefer VISUAL, then EDITOR, then fallback to nano/vi
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor:
            return [editor]
        # Try common editors in order of preference
        for candidate in ("nano", "vi", "vim"):
            if shutil.which(candidate):
                return [candidate]
        return ["vi"]  # Last resort fallback


def _open_file_in_editor(file_path: Path) -> None:
    """Open a file in the platform-appropriate editor."""
    editor_cmd = _get_editor_command()
    try:
        subprocess.run(editor_cmd + [str(file_path)])
    except Exception as exc:
        print(f"{_RED}  ✖  Could not open editor: {exc}{_RESET}")
        print(f"{_YELLOW}  Showing file content instead:{_RESET}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            _page_text(content)
        except OSError as read_exc:
            print(f"{_RED}  ✖  Could not read file: {read_exc}{_RESET}")


def _collect_example_files() -> list[tuple[str, str, Path]]:
    """Return list of (filename, criticality, path) tuples for example files.

    Criticality is either 'critical' or empty string.
    """
    examples_dir = _PROJECT_ROOT / "Examples"
    prefix = "Example_"

    # Define which examples are critical
    critical_examples = {
        "Example_Config_Models.py",
        "Example_Config_Banned.py",
        "Example_Config_WebSearch.py",
        "Example_Config_Internet_Env.py",
    }

    all_sources = sorted(examples_dir.glob(f"{prefix}*.py"))

    result: list[tuple[str, str, Path]] = []
    for src in all_sources:
        criticality = "critical" if src.name in critical_examples else ""
        result.append((src.name, criticality, src))

    return result


def _display_example_files_table(files: list[tuple[str, str, Path]]) -> None:
    """Display example files in a styled table format."""
    w = 70

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Example Configuration Files Review{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()
    print(f"{_DIM}  The following configuration files will be installed:{_RESET}")
    print()

    # Display table header
    print(f"  {'#':<5}{'Filename':<35}{'Status':<15}")
    print(f"  {'-'*5}{'-'*35}{'-'*15}")

    # Display files
    for idx, (filename, criticality, _) in enumerate(files, start=1):
        status_display = f"({criticality})" if criticality else ""
        print(f"  [{idx}] {filename:<33}{status_display:<15}")

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")


def _review_example_files_interactive(files: list[tuple[str, str, Path]]) -> bool:
    """Interactive review of example files with pager/editor options.

    Returns True if user wants to continue with copy, False to skip.
    """
    w = 70

    _display_example_files_table(files)

    print()
    print(f"{_WHITE}  Review files now?{_RESET}")
    print()
    print(f"{_DIM}    y = open in pager (less){_RESET}")
    print(f"{_DIM}    e = open in editor{_RESET}")
    print(f"{_DIM}    a = show all sequentially{_RESET}")
    print(f"{_DIM}    s = skip review{_RESET}")
    print()

    while True:
        choice = input(f"{_ORANGE}  Your choice [y/e/a/s]: {_RESET}").strip().lower()

        if choice == "s":
            print(f"{_DIM}  Skipping file review.{_RESET}")
            break
        elif choice == "y":
            # Open in pager
            print(f"{_DIM}  Opening files in pager...{_RESET}")
            for filename, _, path in files:
                print(f"\n{_CYAN}{'─' * w}{_RESET}")
                print(f"{_BOLD}{_WHITE}  File: {filename}{_RESET}")
                print(f"{_CYAN}{'─' * w}{_RESET}")
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    _page_text(content)
                except OSError as exc:
                    print(f"{_RED}  ✖  Could not read {filename}: {exc}{_RESET}")
            break
        elif choice == "e":
            # Open in editor
            print(f"{_DIM}  Opening files in editor...{_RESET}")
            for filename, _, path in files:
                print(f"\n{_WHITE}  Opening: {filename}{_RESET}")
                _open_file_in_editor(path)
            break
        elif choice == "a":
            # Show all sequentially without pager
            for filename, _, path in files:
                print(f"\n{_CYAN}{'─' * w}{_RESET}")
                print(f"{_BOLD}{_WHITE}  File: {filename}{_RESET}")
                print(f"{_CYAN}{'─' * w}{_RESET}")
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    print(content)
                except OSError as exc:
                    print(f"{_RED}  ✖  Could not read {filename}: {exc}{_RESET}")

                print()
                cont = (
                    input(
                        f"{_ORANGE}  Press Enter to continue to next file, or 'q' to stop: {_RESET}"
                    )
                    .strip()
                    .lower()
                )
                if cont == "q":
                    break
            break
        else:
            print(f"{_YELLOW}  Please choose y, e, a, or s.{_RESET}")

    # Final confirmation
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Configuration Files Review Complete{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()

    return _confirm("I have reviewed the configuration files and wish to continue")


def _show_licenses_pager() -> bool:
    """Display bundled Python third-party license summary before pip install."""
    w = 70

    print()
    _print_next_action_block(
        "Next action",
        [
            ("Step", "Review bundled Python dependency license information"),
            ("Then", "Explicit consent is required before pip install"),
        ],
    )
    print()
    print(f"{_CYAN}{_BOLD}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Third-party Python dependency licenses{_RESET}")
    print(f"{_DIM}  File: 3rdPartyLicenses/Licenses.txt{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(
        f"{_DIM}  Please review the bundled license information before proceeding.{_RESET}"
    )
    print(f"{_DIM}  Press q to close the viewer and continue.{_RESET}")
    print()
    input(f"{_ORANGE}  Press Enter to open the license viewer...{_RESET}")

    text = _collect_license_text()
    _page_text(text, end_marker=True)

    req_path = _REQUIREMENTS.resolve()

    print()
    print(f"{_DIM}  The Python dependencies will be installed from:{_RESET}")
    print(f"{_WHITE}  {req_path}{_RESET}")
    print()
    print(
        f"{_DIM}  The included license evidence should correspond to the pinned versions{_RESET}"
    )
    print(f"{_DIM}  in that requirements file and the signed release manifest.{_RESET}")
    print()

    if _confirm(
        "I have read the license information and agree to continue with installation"
    ):
        identity = _capture_identity(require_confirmation=True)
        _record_python_requirements_consent(identity)
        return True

    print(
        f"{_YELLOW}  Installation declined — setup aborted by user consent decision.{_RESET}"
    )
    return False


# ---------------------------------------------------------------------------
# apt-get version and license-fetch helpers
# ---------------------------------------------------------------------------


def _apt_candidate_version(apt_name: str) -> str | None:
    """Return the apt candidate version via apt-cache policy, or None."""
    try:
        result = subprocess.run(
            ["apt-cache", "policy", apt_name],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Candidate:"):
                ver = stripped.split(":", 1)[1].strip()
                return ver if ver and ver != "(none)" else None
    except Exception:
        pass
    return None


def _apt_base_version(apt_version: str) -> str:
    """Extract MAJOR.MINOR.PATCH from any version string.

    Handles Debian-style suffixes ('5.5.0-1+b1' → '5.5.0') and
    Windows/winget date suffixes ('5.4.0.20240606' → '5.4.0').
    """
    m = re.match(r"^(\d+\.\d+(?:\.\d+)?)", apt_version)
    return m.group(1) if m else apt_version


def _fetch_url(url: str) -> str | None:
    """Fetch URL text via urllib.  Returns None and prints an error on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-LCC/setup"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"{_RED}  ✖  Could not fetch {url}: {exc}{_RESET}")
        return None


def _check_tesseract_accessible() -> None:
    """Warn (without aborting) when tesseract is not yet reachable after installation."""
    found: str | None = os.environ.get("TESSERACT_PATH") or shutil.which("tesseract")
    if not found and _is_windows():
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe",
        ):
            if candidate.is_file():
                found = str(candidate)
                break

    if found:
        print(f"{_GREEN}  ✔  Tesseract is accessible: {found}{_RESET}")
        return

    print()
    print(
        f"{_YELLOW}  ⚠  Tesseract is not yet reachable in this shell session.{_RESET}"
    )
    if _is_windows():
        print(
            f"{_YELLOW}     Tesseract OCR is not installed or not found on PATH.{_RESET}"
        )
        print(
            f"{_YELLOW}     Install it from: https://github.com/UB-Mannheim/tesseract/wiki{_RESET}"
        )
        print(f"{_YELLOW}     After installation, either:{_RESET}")
        print(
            f"{_YELLOW}       - Add the Tesseract install folder to your PATH, or{_RESET}"
        )
        print(
            f"{_YELLOW}       - Set TESSERACT_PATH in Configuration/Config_Internet_Env.py{_RESET}"
        )
        print(f"{_YELLOW}         to the full path of tesseract.exe{_RESET}")
    else:
        print(f"{_YELLOW}     Tesseract executable path is not configured.{_RESET}")
        print(
            f"{_YELLOW}     Set TESSERACT_PATH in Configuration/Config_Internet_Env.py{_RESET}"
        )
        print(
            f"{_YELLOW}     to the full path of the tesseract binary or make sure the PATH resolves it.{_RESET}"
        )
    print(
        f"{_YELLOW}     Setup will continue — OCR features will fail at runtime until this is resolved.{_RESET}"
    )
    print()


def _is_windows() -> bool:
    return sys.platform == "win32"


def _tesseract_installed_version() -> str | None:
    """Return the installed Tesseract version string via `tesseract --version`, or None."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            m = re.search(
                r"tesseract\s+(\d+\.\d+(?:\.\d+)*)",
                result.stdout + result.stderr,
                re.IGNORECASE,
            )
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _winget_candidate_version(winget_id: str) -> str | None:
    """Return the winget candidate version for a package ID, or None."""
    try:
        result = subprocess.run(
            ["winget", "show", "--id", winget_id, "--exact"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("version:"):
                ver = stripped.split(":", 1)[1].strip()
                return ver if ver else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Step 1: apt packages
# ---------------------------------------------------------------------------


def _install_apt_packages() -> None:
    """Step 1: fetch per-package licenses by version, get consent, run apt-get."""
    step = 1
    w = 70

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(
        f"{_BOLD}{_WHITE}  Step {step}/{_TOTAL_STEPS}  ·  Install system packages{_RESET}"
    )

    apt_names = [info["apt_name"] for info in _APT_PACKAGES.values()]

    print(f"{_DIM}  Packages: {', '.join(apt_names)}{_RESET}")
    _installer_label = "winget" if _is_windows() else "apt-get"
    print(
        f"{_DIM}  Purpose : Install OCR engine via {_installer_label}; license fetched for the candidate version.{_RESET}"
    )
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()
    print(f"{_DIM}  Compliance note:{_RESET}")
    print(
        f"{_DIM}  The Python package 'pytesseract' is a wrapper for the Tesseract OCR engine.{_RESET}"
    )
    print(
        f"{_DIM}  If installed from requirements_final.txt, it is part of the pinned Python{_RESET}"
    )
    print(
        f"{_DIM}  dependency set and covered by the signed pip-licenses evidence.{_RESET}"
    )
    print()
    print(
        f"{_DIM}  The native 'tesseract-ocr' engine is an external system package. RAG-LCC{_RESET}"
    )
    print(
        f"{_DIM}  does not bundle it in the signed release manifest when installed via a{_RESET}"
    )
    print(
        f"{_DIM}  system package manager. Setup displays the license, asks for consent, and{_RESET}"
    )
    print(
        f"{_DIM}  records the installed package version and metadata in the runtime install log.{_RESET}"
    )
    print()
    print(f"{_DIM}  Determining package versions...{_RESET}")

    pkg_versions: dict[str, str] = {}
    pkg_urls: dict[str, str] = {}
    pkg_already_installed: dict[str, bool] = {}

    for pkg_key, info in _APT_PACKAGES.items():
        apt_name = info["apt_name"]

        if _is_windows():
            apt_ver = _tesseract_installed_version()
            if apt_ver is not None:
                pkg_already_installed[pkg_key] = True
            else:
                winget_id = info.get("winget_id", "")
                apt_ver = _winget_candidate_version(winget_id) if winget_id else None
                pkg_already_installed[pkg_key] = False

            if apt_ver is None:
                print(
                    f"{_RED}  ✖  Could not determine version for '{apt_name}'.{_RESET}"
                )
                print(
                    f"{_YELLOW}     Tesseract is not installed and winget returned no version.{_RESET}"
                )
                print(
                    f"{_YELLOW}     Install Tesseract manually, then re-run setup:{_RESET}"
                )
                print(
                    f"{_WHITE}     https://github.com/UB-Mannheim/tesseract/wiki{_RESET}"
                )
                sys.exit(1)
        else:
            apt_ver = _apt_candidate_version(apt_name)
            pkg_already_installed[pkg_key] = False

            if apt_ver is None:
                print(
                    f"{_RED}  ✖  Could not determine candidate version for '{apt_name}'.{_RESET}"
                )
                print(
                    f"{_RED}     Run 'apt-get update' and retry, or check your apt sources.{_RESET}"
                )
                sys.exit(1)

        base_ver = _apt_base_version(apt_ver)
        pkg_versions[pkg_key] = apt_ver
        pkg_urls[pkg_key] = info["license_url_template"].format(version=base_ver)

        print(f"  {apt_name}: {apt_ver}  →  {pkg_urls[pkg_key]}")

    print()

    # -- fetch all licenses before showing any consent prompts -------------
    print(f"{_DIM}  Fetching license files...{_RESET}")

    fetched: dict[str, str] = {}

    for pkg_key, url in pkg_urls.items():
        apt_name = _APT_PACKAGES[pkg_key]["apt_name"]
        text = _fetch_url(url)

        if text is None:
            print(
                f"{_RED}  Cannot continue without the license for '{apt_name}'"
                f" — aborting setup.{_RESET}"
            )
            sys.exit(1)

        fetched[pkg_key] = text
        print(
            f"{_GREEN}  ✔  License fetched for {apt_name} ({len(text)} chars).{_RESET}"
        )

    print()

    # -- show each license and collect acceptance --------------------------
    print(f"{_DIM}  Please review each package license before installation.{_RESET}")
    print()

    accepted: dict[str, str] = {}

    for pkg_key, info in _APT_PACKAGES.items():
        apt_name = info["apt_name"]
        apt_ver = pkg_versions[pkg_key]
        url = pkg_urls[pkg_key]
        text = fetched[pkg_key]

        print(f"{_BOLD}{_WHITE}  License: {apt_name}  {apt_ver}{_RESET}")
        print(f"{_DIM}  URL    : {url}{_RESET}")
        _print_next_action_block(
            "Next action",
            [
                ("Package", f"{apt_name} {apt_ver}"),
                ("Step", "Open license viewer"),
                ("Then", "Explicit consent required before installation continues"),
            ],
        )
        print()
        input(
            f"{_ORANGE}  Press Enter to open the license viewer for '{apt_name}'...{_RESET}"
        )

        _page_text(text, end_marker=True)

        print()

        if not _confirm(
            f"I have read the {apt_name} {apt_ver} license and agree to continue with installation"
        ):
            print(
                f"{_YELLOW}  Installation for '{apt_name}' declined — setup aborted by user consent decision.{_RESET}"
            )
            sys.exit(1)

        accepted[pkg_key] = text
        print()

    identity = _capture_identity()
    print()

    now = _utc_now()

    if _is_windows():
        # -- Windows: use winget, or record pre-installed -----------------
        for pkg_key, info in _APT_PACKAGES.items():
            apt_name = info["apt_name"]
            winget_id = info.get("winget_id", "")

            if pkg_already_installed[pkg_key]:
                print(
                    f"{_DIM}  {apt_name} is already installed; skipping package install.{_RESET}"
                )
                install_exit_code = 0
                installer = "pre-installed"
            elif winget_id and shutil.which("winget"):
                print(f"{_DIM}  Installing {apt_name} via winget...{_RESET}")
                install_cmd = [
                    "winget",
                    "install",
                    "--id",
                    winget_id,
                    "--exact",
                    "--silent",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ]
                _run_result = subprocess.run(install_cmd)
                # Normalize to unsigned 32-bit so HRESULT codes match regardless of
                # how Python/Windows represents the sign.
                # 0x8A150011 = APPINSTALLER_CLI_ERROR_PACKAGE_ALREADY_INSTALLED
                # 0x8A15002B = APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE
                _rc_u32 = _run_result.returncode & 0xFFFFFFFF
                install_exit_code = (
                    0
                    if _rc_u32 in (0x00000000, 0x8A150011, 0x8A15002B)
                    else _run_result.returncode
                )
                installer = "winget"
                if install_exit_code != 0:
                    print()
                    print(
                        f"{_RED}  ✖  winget install failed with exit code {_run_result.returncode}.{_RESET}"
                    )
                    print(
                        f"{_RED}     Aborting setup to avoid a partially configured environment.{_RESET}"
                    )
                    sys.exit(_run_result.returncode)
            else:
                print(
                    f"{_RED}  ✖  winget is not available and '{apt_name}' is not installed.{_RESET}"
                )
                print(
                    f"{_YELLOW}     Install Tesseract manually, then re-run setup:{_RESET}"
                )
                print(
                    f"{_WHITE}     https://github.com/UB-Mannheim/tesseract/wiki{_RESET}"
                )
                sys.exit(1)

            _record_apt_consent(
                pkg_key=pkg_key,
                info=info,
                text=accepted[pkg_key],
                license_url=pkg_urls[pkg_key],
                version=pkg_versions[pkg_key],
                identity=identity,
                now=now,
                install_exit_code=install_exit_code,
                installer=installer,
            )
    else:
        # -- Linux: use apt-get -------------------------------------------
        print(f"{_DIM}  Updating apt package index...{_RESET}")
        _run_required_command(["sudo", "apt-get", "update"], "apt-get update")

        pinned = [
            f"{info['apt_name']}={pkg_versions[k]}" for k, info in _APT_PACKAGES.items()
        ]
        print(f"{_DIM}  Installing system packages (pinned versions)...{_RESET}")
        install_cmd = [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
        ] + pinned
        result = subprocess.run(install_cmd)

        if result.returncode != 0:
            print()
            print(
                f"{_RED}  ✖  apt-get install failed with exit code {result.returncode}.{_RESET}"
            )
            print(
                f"{_RED}     Aborting setup to avoid a partially configured environment.{_RESET}"
            )
            sys.exit(result.returncode)

        for pkg_key, info in _APT_PACKAGES.items():
            _record_apt_consent(
                pkg_key=pkg_key,
                info=info,
                text=accepted[pkg_key],
                license_url=pkg_urls[pkg_key],
                version=pkg_versions[pkg_key],
                identity=identity,
                now=now,
                install_exit_code=result.returncode,
            )

    print(f"\n{_GREEN}  ✔  Step {step} completed successfully.{_RESET}")
    _check_tesseract_accessible()


# ---------------------------------------------------------------------------
# Script runner for existing helper scripts
# ---------------------------------------------------------------------------


def _step_header(step: int, label: str, script: str, description: str) -> None:
    w = 70

    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Step {step}/{_TOTAL_STEPS}  ·  {label}{_RESET}")
    print(f"{_DIM}  Script : src/Scripts/{script}{_RESET}")
    print(f"{_DIM}  Purpose: {description}{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()


def _step_ok(step: int) -> None:
    print(f"\n{_GREEN}  ✔  Step {step} completed successfully.{_RESET}")


def _run_step(
    step: int,
    label: str,
    script: str,
    description: str,
    extra_argv: list[str] | None = None,
    required: bool = True,
) -> None:
    """Execute script relative to src/Scripts using runpy."""
    script_file = _SCRIPTS_DIR / script

    if not script_file.exists():
        print(f"{_RED}  ✖  Required script not found: {script_file}{_RESET}")
        if required:
            sys.exit(1)
        return

    script_path = str(script_file)
    argv_backup = sys.argv[:]
    sys.argv = [script_path] + (extra_argv or [])
    _write_setup_log(
        "step_started",
        step=step,
        label=label,
        script=script,
        required=required,
        extra_argv=extra_argv or [],
    )

    _step_header(step, label, script, description)

    try:
        runpy.run_path(script_path, run_name="__main__")
        _step_ok(step)
        _write_setup_log(
            "step_completed",
            step=step,
            label=label,
            script=script,
            exit_code=0,
        )

    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0

        if code == 0:
            _step_ok(step)
            _write_setup_log(
                "step_completed",
                step=step,
                label=label,
                script=script,
                exit_code=0,
            )
        else:
            if required:
                _write_setup_log(
                    "step_failed",
                    step=step,
                    label=label,
                    script=script,
                    exit_code=code,
                    required=True,
                )
                print()
                print(f"{_RED}  ✖  Step {step} failed with exit code {code}.{_RESET}")
                print(f"{_RED}     Aborting setup.{_RESET}")
                sys.exit(code)

            _write_setup_log(
                "step_failed",
                step=step,
                label=label,
                script=script,
                exit_code=code,
                required=False,
            )
            print()
            print(
                f"{_YELLOW}  ⚠  Step {step} finished with exit code {code} — moving on.{_RESET}"
            )

    finally:
        sys.argv = argv_backup


# ---------------------------------------------------------------------------
# Post-setup notices
# ---------------------------------------------------------------------------


def _print_gpu_notice(w: int = 70) -> None:
    """Print a hardware-specific GPU support reminder after all steps complete."""
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(
        f"{_BOLD}{_WHITE}  GPU / hardware acceleration — manual step required{_RESET}"
    )
    print(
        f"{_DIM}  The requirements file installs the CPU-only build of PyTorch by default.{_RESET}"
    )
    print(
        f"{_DIM}  To enable GPU acceleration, reinstall PyTorch for your hardware after setup:{_RESET}"
    )
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()


def _print_setup_notice(w: int = 70) -> None:
    """Print endpoint/model configuration reminders after setup."""
    print(f"{_TURQUOISE}{'─' * w}{_RESET}")
    print(
        f"{_WHITE}    3) Make sure required models are installed and available on that endpoint:{_RESET}"
    )
    print(f"{_WHITE}       - Inference (_ACTIVE_LLM): mistral or llama{_RESET}")
    print(
        f"{_WHITE}       - Prompt compliance (_ACTIVE_LLM_CHK): llama_guard or mistral{_RESET}"
    )
    print()
    print(f"{_BOLD}{_WHITE}  Important{_RESET}")
    print(
        f"{_WHITE}    Before normal use, complete the one-time model license acceptance flow.{_RESET}"
    )
    print(
        f"{_WHITE}    If you changed config files, rehash before running the app:{_RESET}"
    )
    print(f"{_BOLD}{_WHITE}    python ./src/Scripts/RecalcConfigHashes.py{_RESET}")
    print()
    print(f"{_BOLD}{_WHITE}  Start sequence{_RESET}")
    print(f"{_WHITE}    Then start apps in this order:{_RESET}")
    print(
        f"{_BOLD}{_WHITE}    python ./src/Apps/RAGLoad.py (will load the TestDocs/* documents){_RESET}"
    )
    print(f"{_WHITE}    followed by either:{_RESET}")
    print(f"{_BOLD}{_WHITE}    python ./src/Apps/RAGChat.py{_RESET}")
    print(
        f"{_BOLD}{_WHITE}    python ./src/Apps/RAGChatService.py (for Open WebUI){_RESET}"
    )
    print(f"{_WHITE}    or{_RESET}")
    print(f"{_BOLD}{_WHITE}    python ./src/Apps/DocClassify.py{_RESET}")
    print()
    print(f"{_TURQUOISE}{'─' * w}{_RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG-LCC one-shot setup wrapper.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--no-examples-copy",
        action="store_true",
        help="Skip Step 3 — do not copy Example_*.py files into Configuration/.",
    )
    parser.add_argument(
        "--no-config-rehash",
        action="store_true",
        help="Skip Step 4 — do not recalculate SHA-256 config hashes.",
    )
    parser.add_argument(
        "--set-config-values-only",
        action="store_true",
        help=(
            "Ask only runtime configuration questions, apply the values, "
            "optionally rehash configs, and exit."
        ),
    )
    parser.add_argument(
        "--set-config",
        action="store_true",
        help=(
            "Short alias for --set-config-values-only. "
            "Skip all installation steps and go directly to runtime configuration."
        ),
    )
    parser.add_argument(
        "--skip-signature-verification",
        action="store_true",
        help="Skip the signature verification step.",
    )
    args = parser.parse_args()

    # Handle --set-config as an alias for --set-config-values-only
    if args.set_config:
        args.set_config_values_only = True

    _banner()
    _print_execution_plan()
    _ensure_project_root()
    _init_setup_log()
    _ensure_venv()
    _ensure_cache_ownership()
    _write_setup_log(
        "setup_started",
        no_examples_copy=args.no_examples_copy,
        no_config_rehash=args.no_config_rehash,
        set_config_values_only=args.set_config_values_only,
        skip_signature_verification=args.skip_signature_verification,
    )

    if args.set_config_values_only:
        print(f"{_CYAN}  Config-values-only mode enabled.{_RESET}")
        _write_setup_log("config_values_only_mode", enabled=True)

        _run_setup_questions()

        if args.no_config_rehash:
            print(f"{_YELLOW}  ⏭  Step 4 skipped (--no-config-rehash).{_RESET}")
            _write_setup_log(
                "step_skipped",
                step=4,
                reason="--no-config-rehash",
            )
        else:
            _run_step(
                step=4,
                label="Recalculate SHA-256 config hashes",
                script="RecalcConfigHashes.py",
                description=(
                    "Recomputes hashes for Config_Models.py, Config_Banned.py, "
                    "Config_WebSearch.py, and Config_Internet_Env.py and writes "
                    "them into Config_Global.py."
                ),
                required=True,
            )

        _write_setup_log("setup_completed", mode="set-config-values-only")
        return

    # ------------------------------------------------------------------
    # Preamble – Install cryptography module for signature verification.
    # ------------------------------------------------------------------
    w = 70
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  Preamble  ·  Install cryptography module{_RESET}")
    print(f"{_DIM}  Purpose: Required for signature verification{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print()

    print(f"{_DIM}  Installing cryptography module...{_RESET}")
    pip_cmd = [sys.executable, "-m", "pip", "install", "cryptography"]
    result = subprocess.run(pip_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print()
        print(f"{_RED}  ✖  Failed to install cryptography module.{_RESET}")
        print(f"{_RED}     Error: {result.stderr}{_RESET}")
        print(
            f"{_RED}     Signature verification requires the cryptography module.{_RESET}"
        )
        print(
            f"{_YELLOW}     You can bypass verification with --skip-signature-verification{_RESET}"
        )
        sys.exit(result.returncode)

    print(f"{_GREEN}  ✔  cryptography module installed successfully.{_RESET}")
    _write_setup_log(
        "cryptography_installed",
        exit_code=result.returncode,
    )

    # ------------------------------------------------------------------
    # Preamble – Verify file signatures.
    # ------------------------------------------------------------------
    if args.skip_signature_verification:
        print(
            f"{_YELLOW}  ⏭  Signature verification skipped (--skip-signature-verification).{_RESET}"
        )
        _write_setup_log(
            "step_skipped",
            step="signature_verification",
            reason="--skip-signature-verification",
        )
    else:
        w = 70
        print()
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print(f"{_BOLD}{_WHITE}  Preamble  ·  Verify file signatures{_RESET}")
        print(f"{_DIM}  Script : src/Scripts/VerifySignatures.py{_RESET}")
        print(
            f"{_DIM}  Purpose: Confirm shipped files have not been tampered with.{_RESET}"
        )
        print(f"{_CYAN}{'─' * w}{_RESET}")
        print()

        verify_script = _SCRIPTS_DIR / "VerifySignatures.py"
        if not verify_script.exists():
            print(
                f"{_YELLOW}  ⚠  Signature verification script not found: {verify_script}{_RESET}"
            )
            print(f"{_YELLOW}     Skipping signature verification.{_RESET}")
            _write_setup_log(
                "step_skipped",
                step="signature_verification",
                reason="script not found",
            )
        else:
            argv_backup = sys.argv[:]
            # Exclude .venv directory to avoid scanning thousands of Python package files
            sys.argv = [
                str(verify_script),
                "--input-dir",
                str(_PROJECT_ROOT),
                "--exclude-dirs",
                ".venv",
            ]
            _write_setup_log(
                "step_started",
                step="signature_verification",
                label="Verify file signatures",
                script="VerifySignatures.py",
            )

            try:
                runpy.run_path(str(verify_script), run_name="__main__")
                print(
                    f"\n{_GREEN}  ✔  Signature verification completed successfully.{_RESET}"
                )
                _write_setup_log(
                    "step_completed",
                    step="signature_verification",
                    label="Verify file signatures",
                    script="VerifySignatures.py",
                    exit_code=0,
                )
            except SystemExit as exc:
                code = exc.code if exc.code is not None else 0
                if code == 0:
                    print(
                        f"\n{_GREEN}  ✔  Signature verification completed successfully.{_RESET}"
                    )
                    _write_setup_log(
                        "step_completed",
                        step="signature_verification",
                        label="Verify file signatures",
                        script="VerifySignatures.py",
                        exit_code=0,
                    )
                else:
                    _write_setup_log(
                        "step_failed",
                        step="signature_verification",
                        label="Verify file signatures",
                        script="VerifySignatures.py",
                        exit_code=code,
                    )
                    print()
                    print(
                        f"{_RED}  ✖  Signature verification failed with exit code {code}.{_RESET}"
                    )
                    print(f"{_RED}     This may indicate file tampering.{_RESET}")
                    print(
                        f"{_YELLOW}     You can bypass this check with --skip-signature-verification{_RESET}"
                    )
                    print(
                        f"{_YELLOW}     if you are certain the files are safe.{_RESET}"
                    )
                    sys.exit(code)
            except Exception as exc:
                _write_setup_log(
                    "step_failed",
                    step="signature_verification",
                    label="Verify file signatures",
                    script="VerifySignatures.py",
                    error=str(exc),
                )
                print()
                print(
                    f"{_RED}  ✖  Signature verification raised an exception: {exc}{_RESET}"
                )
                print(
                    f"{_YELLOW}     You can bypass this check with --skip-signature-verification{_RESET}"
                )
                sys.exit(1)
            finally:
                sys.argv = argv_backup

    # ------------------------------------------------------------------
    # Step 1 – Install system packages via apt-get.
    # ------------------------------------------------------------------
    _install_apt_packages()

    # ------------------------------------------------------------------
    # Preamble – show Python dependency licenses before pip install.
    # ------------------------------------------------------------------
    if not _show_licenses_pager():
        return

    # ------------------------------------------------------------------
    # Step 2 – Install Python dependencies.
    # ------------------------------------------------------------------
    _run_step(
        step=2,
        label="Install Python dependencies",
        script="PipInstall.py",
        description=(
            "Installs all runtime packages from requirements/requirements_final.txt "
            "into the active project venv and records the pip install report."
        ),
        required=True,
    )
    _enrich_pip_install_meta_with_consent()

    # ------------------------------------------------------------------
    # Step 3 – Copy example configs into Configuration/.
    # Existing files are left untouched unless the child script changes that.
    # ------------------------------------------------------------------
    if args.no_examples_copy:
        print(f"{_YELLOW}  ⏭  Step 3 skipped (--no-examples-copy).{_RESET}")
        _write_setup_log(
            "step_skipped",
            step=3,
            reason="--no-examples-copy",
        )
    else:
        examples_dir = _PROJECT_ROOT / "Examples"
        print()
        print(
            f"{_DIM}  Before copying, please review the example configuration files in:{_RESET}"
        )
        print(f"{_WHITE}  {examples_dir}{_RESET}")
        print(
            f"{_DIM}  These files will be copied into Configuration/ as your default settings.{_RESET}"
        )
        print(
            f"{_DIM}  In particular, review Config_Models.py before enabling internet access{_RESET}"
        )
        print(f"{_DIM}  or accepting model licenses.{_RESET}")
        _print_next_action_block(
            "Next action",
            [
                ("Step", "Review example config files"),
                ("If approved", "Copy Example_*.py into src/Configuration/"),
            ],
        )
        print()

        # Collect example files
        example_files = _collect_example_files()

        if not example_files:
            print(f"{_ORANGE}  WARN   No example files found in {examples_dir}{_RESET}")
            _write_setup_log(
                "step_skipped",
                step=3,
                reason="no example files found",
            )
        else:
            # Show interactive review
            if not _review_example_files_interactive(example_files):
                print(f"{_YELLOW}  Skipping example config copy.{_RESET}")
                _write_setup_log(
                    "step_skipped",
                    step=3,
                    reason="user declined example copy after review",
                )
            else:
                _run_step(
                    step=3,
                    label="Copy example configuration files",
                    script="CopyExampleConfigs.py",
                    description=(
                        "Copies Example_*.py from Examples/ into Configuration/, "
                        "backing up existing files to .sav and overwriting them "
                        "to ensure fresh configs before applying runtime values."
                    ),
                    extra_argv=["--force"],
                    required=True,
                )

    # ------------------------------------------------------------------
    # Step 5 – Download NLTK corpora (stopwords + WordNet).
    # Must run after Step 2 (pip install) so NLTK is available.
    # ------------------------------------------------------------------
    _run_step(
        step=5,
        label="Download NLTK stopwords and WordNet",
        script="NLTK_Stopwords_WordNet.py",
        description=(
            "Fetches NLTK and WordNet licenses for the installed versions, "
            "records consent, and downloads the stopwords and wordnet corpora."
        ),
        required=True,
    )

    # ------------------------------------------------------------------
    # Step 6 – Install Argos Translate language packages.
    # ------------------------------------------------------------------
    _print_next_action_block(
        "Next action",
        [
            ("If approved", "Run Argos installer"),
            ("Step", "Show license and collect consent"),
            ("Then", "Download language pairs from ARGOS_LANGUAGES"),
        ],
    )
    if not _confirm("Ok to install Argos Translate language packages?"):
        print(f"{_YELLOW}  Skipping Argos Translate install.{_RESET}")
        _write_setup_log(
            "step_skipped",
            step=6,
            reason="user declined Argos install",
        )
    else:
        _run_step(
            step=6,
            label="Install Argos Translate language packages",
            script="ArgosTranslatePackages.py",
            description=(
                "Presents the Argos Translate license, records consent, and "
                "downloads the language pairs defined in ARGOS_LANGUAGES."
            ),
            extra_argv=["install"],
            required=True,
        )

    # ------------------------------------------------------------------
    # Runtime configuration questions.
    # Runs after all download/install steps by design.
    # ------------------------------------------------------------------
    _run_setup_questions()

    # ------------------------------------------------------------------
    # Finalize – Recalculate config hashes.
    # Must run after runtime question updates.
    # ------------------------------------------------------------------
    if args.no_config_rehash:
        print(f"{_YELLOW}  ⏭  Finalize skipped (--no-config-rehash).{_RESET}")
        _write_setup_log(
            "step_skipped",
            step=4,
            reason="--no-config-rehash",
        )
    else:
        _run_step(
            step=4,
            label="Recalculate SHA-256 config hashes",
            script="RecalcConfigHashes.py",
            description=(
                "Recomputes hashes for Config_Models.py, Config_Banned.py, "
                "Config_WebSearch.py, and Config_Internet_Env.py and writes "
                "them into Config_Global.py."
            ),
            required=True,
        )

    # Make application scripts executable on Unix systems
    print(f"{_DIM}  Setting executable permissions on application scripts...{_RESET}")
    _make_scripts_executable(_SRC_DIR / "Apps")
    _make_scripts_executable(_SCRIPTS_DIR)

    w = 70
    print()
    print(f"{_GREEN}{_BOLD}{'=' * w}{_RESET}")
    print(f"{_GREEN}{_BOLD}{'  All setup steps finished.':^{w}}{_RESET}")
    print(f"{_GREEN}{_BOLD}{'=' * w}{_RESET}")
    print()

    # If devcontainer.json was modified, prompt user to reopen container
    if _devcontainer_modified:
        print(f"{_YELLOW}{_BOLD}{'═' * w}{_RESET}")
        print(
            f"{_BOLD}{_YELLOW}{'  ⚠️  ACTION REQUIRED: Container Restart Needed':^{w}}{_RESET}"
        )
        print(f"{_YELLOW}{_BOLD}{'═' * w}{_RESET}")
        print()
        print(
            f"{_BOLD}{_WHITE}  Port forwarding configuration was updated in .devcontainer/devcontainer.json{_RESET}"
        )
        print(
            f"{_BOLD}{_WHITE}  The changes will NOT take effect until you reopen the container.{_RESET}"
        )
        print()
        print(f"{_BOLD}{_CYAN}  To apply the changes NOW:{_RESET}")
        print()
        print(
            f"{_BOLD}  1.{_RESET} Press {_BOLD}{_GREEN}F1{_RESET} or {_BOLD}{_GREEN}Ctrl+Shift+P{_RESET} (Command Palette)"
        )
        print(
            f"{_BOLD}  2.{_RESET} Type: {_BOLD}{_GREEN}Dev Containers: Rebuild and Reopen in Container{_RESET}"
        )
        print(f"{_BOLD}  3.{_RESET} Press {_BOLD}{_GREEN}Enter{_RESET}")
        print()
        print(
            f"{_DIM}  Alternative: Use 'Reopen in Container' (faster, no rebuild){_RESET}"
        )
        print(
            f"{_DIM}  VS Code may show a popup - click 'Rebuild' or 'Reopen' there.{_RESET}"
        )
        print()
        print(f"{_YELLOW}{_BOLD}{'═' * w}{_RESET}")
        print()

    _print_gpu_notice(w)
    _print_setup_notice(w)
    _write_setup_log("setup_completed")


if __name__ == "__main__":
    main()
