#!/usr/bin/env python3
"""
NLTK_Stopwords_WordNet.py — download NLTK stopwords and WordNet corpora
with per-component license fetch, consent, and consent recording.

License strategy
  Stopwords : NLTK Apache-2.0 license fetched from GitHub at the installed
              NLTK version tag.
  WordNet   : Princeton WordNet License fetched from the SPDX license list
              for the version listed in the NLTK data index.

If either license cannot be fetched from the network, execution stops with an
informative error message.

Consent is recorded in:
  ModelGovernance/licenses/<component>/   — LICENSE.txt + license_meta.json
  ModelGovernance/consents/<component>/   — install_meta.json


NLTK handling

The Python package "nltk" and NLTK data packages are different compliance
objects.

The Python package nltk is covered by the Python dependency process if it is
listed in requirements_final.txt.

NLTK data packages such as stopwords or wordnet are downloaded separately via
the NLTK downloader and are not automatically covered by the Python package
license evidence.

The manifest maps pinned package versions to the generated and signed license
evidence. It does not claim that a license itself is pinned; the package version
is pinned, and the generated evidence artifact is signed.

The Python package "nltk" is installed as part of the pinned Python dependency
set if present in requirements_final.txt. Its license evidence is captured in
the generated pip-licenses report and covered by the signed release manifest.


Usage:
    python src/Scripts/NLTK_Stopwords_WordNet.py           # show installed (default)
    python src/Scripts/NLTK_Stopwords_WordNet.py install
    python src/Scripts/NLTK_Stopwords_WordNet.py remove
    python src/Scripts/NLTK_Stopwords_WordNet.py status
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import Configuration.Config_Global as _ConfigGlobal  # noqa: E402
# Resolve and validate project root before any setup actions.
from Commons.DriveRootGuard import (assert_script_project_root,  # noqa: E402
                                    is_drive_root, resolve_guard_path)
from Gui.PrettyWriter import PrettyWriter  # noqa: E402

_PROJECT_ROOT = Path(
    assert_script_project_root(
        __file__,
        configured_project_root=getattr(_ConfigGlobal, "_ABSOLUTE_PATH", None),
        require_cwd_match=True,
    )
)
_GOVERNANCE = _PROJECT_ROOT / "ModelGovernance"

# ---------------------------------------------------------------------------
# ANSI colours
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

# ---------------------------------------------------------------------------
# License source URLs
# ---------------------------------------------------------------------------

# {version} is replaced with the installed NLTK version (e.g. "3.9.1")
_NLTK_LICENSE_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/nltk/nltk/v{version}/LICENSE.txt"
)

# Public XML index of NLTK data packages — used to resolve the WordNet version
_NLTK_DATA_INDEX_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data" "/refs/heads/gh-pages/index.xml"
)

# Princeton WordNet license — plain text from the SPDX license list
_WORDNET_LICENSE_URL = (
    "https://raw.githubusercontent.com/spdx/license-list-data/main/text/WordNet.txt"
)

_pretty_writer_instance: PrettyWriter | None = None


def _pretty_writer() -> PrettyWriter:
    """Return a cached PrettyWriter configured for always-on installer output."""
    global _pretty_writer_instance
    if _pretty_writer_instance is None:
        _pretty_writer_instance = PrettyWriter(always_on=True)
    return _pretty_writer_instance


# ---------------------------------------------------------------------------
# Helpers: network
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _fetch_url(url: str, *, strip_html: bool = False) -> str | None:
    """Return URL content as str, or None (with error message) on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RAG-LCC/setup"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        return _strip_html(content) if strip_html else content
    except Exception as exc:
        print(f"{_RED}  ✖  Could not fetch {url}: {exc}{_RESET}")
        return None


# ---------------------------------------------------------------------------
# Helpers: version resolution
# ---------------------------------------------------------------------------


def _nltk_installed_version() -> str | None:
    """Return the installed NLTK version via pip show, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "nltk"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _wordnet_version_from_index() -> str | None:
    """Parse the NLTK data index XML to find the wordnet package version."""
    content = _fetch_url(_NLTK_DATA_INDEX_URL)
    if content is None:
        return None
    try:
        root = ET.fromstring(content)
        for pkg in root.iter("package"):
            if pkg.get("id") == "wordnet":
                return pkg.get("version")
    except Exception as exc:
        print(f"{_YELLOW}  ⚠  Could not parse NLTK data index: {exc}{_RESET}")
    return None


# ---------------------------------------------------------------------------
# Helpers: consent recording
# ---------------------------------------------------------------------------


def _compute_hash(text: str) -> str:
    return hashlib.sha256(
        text.replace("\r\n", "\n").strip().encode("utf-8")
    ).hexdigest()


class _Identity(TypedDict):
    accepted_by: str
    accepted_by_source: str
    accepted_by_verified: bool
    host: str
    pid: int


_identity_cache: _Identity | None = None


def _capture_identity() -> _Identity:
    """Prompt once for the accepting user's identity; cache the result."""
    global _identity_cache
    if _identity_cache is not None:
        return _identity_cache

    git_user: str | None = None
    try:
        git_user = (
            subprocess.check_output(
                ["git", "config", "user.email"], stderr=subprocess.DEVNULL
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


def _record_consent(
    *,
    key: str,
    display: str,
    license_url: str,
    license_version: str,
    license_text: str,
    identity: _Identity,
    now: str,
    lic_dir: Path,
    consent_dir: Path,
) -> None:
    """Write LICENSE.txt, license_meta.json, and install_meta.json."""
    lic_path = lic_dir / "LICENSE.txt"
    lic_meta_path = lic_dir / "license_meta.json"
    consent_path = consent_dir / "install_meta.json"
    had_license_meta = lic_meta_path.is_file()
    had_download_meta = consent_path.is_file()

    lic_hash = _compute_hash(license_text)

    lic_dir.mkdir(parents=True, exist_ok=True)
    lic_path.write_text(license_text, encoding="utf-8")

    lic_meta: dict[str, object] = {
        "component": key,
        "display": display,
        "license_url": license_url,
        "license_version": license_version,
        "license_hash_text_canonical_sha256": lic_hash,
        **identity,
        "accepted_at": now,
        "consent": True,
    }
    with lic_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(lic_meta, fh, indent=2, ensure_ascii=False)

    consent_dir.mkdir(parents=True, exist_ok=True)
    install_meta: dict[str, object] = {
        "component": key,
        "corpus": key,
        "installer": "nltk.download",
        **identity,
        "accepted_at": now,
        "consent": True,
    }
    with consent_path.open("w", encoding="utf-8") as fh:
        json.dump(install_meta, fh, indent=2, ensure_ascii=False)

    pretty = _pretty_writer()
    license_action = "Updated" if had_license_meta else "Created"
    download_action = "Updated" if had_download_meta else "Created"
    pretty.write(
        "O",
        "NLTK License",
        f"{display}: {license_action} license consent metadata: {lic_meta_path}",
        color=_GREEN,
    )
    pretty.write(
        "O",
        "NLTK Download",
        f"{display}: {download_action} download consent metadata: {consent_path}",
        color=_GREEN,
    )


# ---------------------------------------------------------------------------
# Helpers: pager
# ---------------------------------------------------------------------------


_END_OF_LICENSE = (
    "\n\n" + "-" * 70 + "\n" "  >>>>>  End of license  <<<<<\n" + "-" * 70 + "\n"
)


def _page_text(text: str, *, end_marker: bool = False) -> None:
    if end_marker:
        text = text + _END_OF_LICENSE

    if os.name == "nt":
        try:
            subprocess.run("more", input=text, text=True, shell=True, encoding="utf-8")
            try:
                import msvcrt  # type: ignore[import]

                while msvcrt.kbhit():
                    msvcrt.getch()
            except Exception:
                pass
            return
        except Exception:
            pass

    if shutil.which("less"):
        env = {**os.environ, "LESS": "FRX"}
        try:
            subprocess.run(["less", "-"], input=text, text=True, env=env)
            return
        except OSError:
            pass

    try:
        rows = max(5, min(shutil.get_terminal_size().lines - 2, 40))
    except Exception:
        rows = 20
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        print("\n".join(lines[idx : idx + rows]))
        idx += rows
        if idx < len(lines):
            try:
                q = (
                    input(f"{_ORANGE}[Enter] next page  [q] quit view: {_RESET}")
                    .strip()
                    .lower()
                )
            except EOFError:
                break
            if q == "q":
                break


# ---------------------------------------------------------------------------
# Per-corpus download flow
# ---------------------------------------------------------------------------


def _download_corpus(
    *,
    key: str,
    display: str,
    corpus_id: str,
    license_url: str,
    license_version: str,
    strip_html: bool,
    lic_dir: Path,
    consent_dir: Path,
    identity: _Identity,
    now: str,
    additionalk_info: tuple[str, ...] | None = None,
) -> None:
    w = 70
    print()
    print(f"{_CYAN}{'-' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  {display}{_RESET}")
    print(f"{_DIM}  Version     : {license_version}{_RESET}")
    print(f"{_DIM}  License URL : {license_url}{_RESET}")
    if additionalk_info:
        for line in additionalk_info:
            clean = line.strip()
            if not clean:
                continue
            print(f"{_ORANGE}  {clean}{_RESET}")
    print(f"{_CYAN}{'-' * w}{_RESET}")
    print()

    print(f"{_DIM}  Fetching license...{_RESET}")
    license_text = _fetch_url(license_url, strip_html=strip_html)
    if license_text is None:
        print(
            f"{_RED}  Cannot continue without the license for '{display}' — aborting.{_RESET}"
        )
        sys.exit(1)
    print(f"{_GREEN}  ✔  License fetched ({len(license_text)} chars).{_RESET}")
    print()

    input(
        f"{_ORANGE}  Press Enter to open the license viewer for '{display}'...{_RESET}"
    )
    _page_text(license_text, end_marker=True)
    print()

    while True:
        ans = (
            input(
                f"{_ORANGE}  Accept the {display} license and allow download? [y/n]: {_RESET}"
            )
            .strip()
            .lower()
        )
        if ans in ("y", "yes"):
            break
        if ans in ("n", "no"):
            print(f"{_YELLOW}  License for '{display}' declined — aborting.{_RESET}")
            sys.exit(1)
        print("  Please answer y or n.")

    print()
    try:
        import importlib  # noqa: PLC0415

        nltk = cast(Any, importlib.import_module("nltk"))
        nltk_data_paths = cast(list[str], getattr(nltk.data, "path", []))

        configured_raw = _custom_nltk_data_directory()
        download_root = _configured_nltk_download_root()
        if download_root is None:
            print(
                f"{_RED}  ✖  Could not resolve _CUSTOM_NLTK_DATA_DIRECTORY from Config_Global.py.{_RESET}"
            )
            sys.exit(1)

        if not _passes_minimum_root_guard(
            download_root,
            raw_target_path=configured_raw or None,
        ):
            print(
                f"{_RED}  ✖  Refusing to use unsafe NLTK download root '{download_root}'.{_RESET}"
            )
            sys.exit(1)

        try:
            download_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                f"{_RED}  ✖  Could not create configured NLTK download directory '{download_root}': {exc}{_RESET}"
            )
            sys.exit(1)

        if not os.access(str(download_root), os.W_OK):
            print(
                f"{_RED}  ✖  Configured NLTK download directory is not writable: {download_root}{_RESET}"
            )
            sys.exit(1)

        download_dir = str(download_root)
        if download_dir not in nltk_data_paths:
            nltk_data_paths.insert(0, download_dir)

        if configured_raw:
            print(f"{_DIM}  Config_Global path : {configured_raw}{_RESET}")
            normalized_config = _normalize_abs_path(configured_raw)
            if normalized_config != download_root:
                print(f"{_DIM}  Derived download root : {download_root}{_RESET}")
        print(f"{_DIM}  Download directory : {download_dir}{_RESET}")
        print(
            f"{_DIM}  (Set _CUSTOM_NLTK_DATA_DIRECTORY in Config_Global.py"
            f" to change where the application reads NLTK data.){_RESET}"
        )

        print(f"{_DIM}  Downloading NLTK corpus '{corpus_id}'...{_RESET}")
        nltk.download(corpus_id, quiet=False, download_dir=download_dir)
        print(f"{_GREEN}  ✔  '{corpus_id}' downloaded.{_RESET}")
    except ImportError:
        print(f"{_RED}  ✖  NLTK is not installed.{_RESET}")
        print(
            f"{_CYAN}  Hint: run Setup.py Step 2 (pip install) before this step.{_RESET}"
        )
        sys.exit(1)

    _record_consent(
        key=key,
        display=display,
        license_url=license_url,
        license_version=license_version,
        license_text=license_text,
        identity=identity,
        now=now,
        lic_dir=lic_dir,
        consent_dir=consent_dir,
    )


# ---------------------------------------------------------------------------
# Shared CLI helpers
# ---------------------------------------------------------------------------


def _confirm(prompt_msg: str) -> bool:
    """Loop asking yes/no until the user gives a clear answer."""
    while True:
        answer = input(f"{prompt_msg} [yes/no]: ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("  Please answer 'yes' or 'no'.")


def _normalize_abs_path(path_value: str | Path) -> Path:
    """Return a normalized absolute path for a raw path string/value."""
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(str(path_value)))))


def _passes_minimum_root_guard(
    target_path: Path,
    *,
    raw_target_path: str | None = None,
) -> bool:
    """Refuse delete operations that resolve to drive/filesystem roots."""
    candidates: list[str] = []
    if raw_target_path:
        candidates.append(raw_target_path)
    candidates.append(str(target_path))

    for candidate in candidates:
        normalized_candidate = resolve_guard_path(candidate)
        if is_drive_root(normalized_candidate):
            print(
                f"{_RED}  ✖  Refusing to delete root or drive path "
                f"'{normalized_candidate}'.{_RESET}"
            )
            return False

    return True


def _custom_nltk_data_directory() -> str:
    """Return configured custom NLTK data directory from Config when available."""
    try:
        from Config.Config import Config  # noqa: PLC0415

        return str(Config().get_str("_CUSTOM_NLTK_DATA_DIRECTORY") or "").strip()
    except Exception:
        return ""


def _configured_nltk_download_root() -> Path | None:
    """Resolve the concrete NLTK download root from configured path.

    _CUSTOM_NLTK_DATA_DIRECTORY may point either to a data root, to
    ``.../corpora``, or to a concrete corpus directory such as
    ``.../corpora/stopwords``. Downloads must target the NLTK data root.
    """
    custom_dir = _custom_nltk_data_directory()
    if not custom_dir:
        return None

    configured = _normalize_abs_path(custom_dir)
    parts_lower = [part.lower() for part in configured.parts]

    if len(parts_lower) >= 2 and parts_lower[-2] == "corpora":
        if parts_lower[-1] in ("stopwords", "wordnet"):
            return configured.parent.parent
        if parts_lower[-1] == "corpora":
            return configured.parent

    return configured


def _dedupe_paths(raw_paths: Sequence[str | Path]) -> list[Path]:
    """Normalize and de-duplicate path values while preserving order."""
    deduped_paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        normalized = _normalize_abs_path(raw_path)
        key = os.path.normcase(str(normalized))
        if key in seen:
            continue
        seen.add(key)
        deduped_paths.append(normalized)
    return deduped_paths


def _resolve_nltk_data_paths() -> list[Path]:
    """Return candidate NLTK data paths for status/remove.

    Prefer the configured custom path from Config and nearby parent roots.
    Fall back to nltk.data.path only when no custom path is configured.
    """
    custom_dir = _custom_nltk_data_directory()
    if custom_dir:
        custom_path = _normalize_abs_path(custom_dir)
        preferred: list[Path] = []
        for candidate in (custom_path, custom_path.parent, custom_path.parent.parent):
            if is_drive_root(str(candidate)):
                continue
            preferred.append(candidate)
        return _dedupe_paths(preferred)

    raw_paths: list[str] = []
    try:
        import importlib  # noqa: PLC0415

        nltk = cast(Any, importlib.import_module("nltk"))
        raw_paths.extend(cast(list[str], getattr(nltk.data, "path", [])))
    except Exception:
        return []

    return _dedupe_paths(raw_paths)


def _print_nltk_data_path_status(
    nltk_data_paths: list[Path], *, present_only: bool = False
) -> None:
    """Print NLTK data directories and optional presence filtering."""
    if present_only:
        print("\nNLTK data directories present:")
        present_paths = [path for path in nltk_data_paths if path.exists()]
        if not present_paths:
            print("  (none present)")
            return
        for path in present_paths:
            print(f"  {path}")
        return

    print("\nNLTK data directories checked:")
    if not nltk_data_paths:
        print("  (none discovered)")
        return

    for path in nltk_data_paths:
        state = "present" if path.exists() else "not present"
        print(f"  {path}  ({state})")


def _print_custom_nltk_directory_hint() -> None:
    """Print where to change NLTK path configuration and its current value."""
    config_value = _custom_nltk_data_directory()
    print(
        f"{_CYAN}  Hint: change this path in Config_Global.py via "
        f"_CUSTOM_NLTK_DATA_DIRECTORY.{_RESET}"
    )
    if config_value:
        print(
            f"{_DIM}  Current _CUSTOM_NLTK_DATA_DIRECTORY: " f"{config_value}{_RESET}"
        )
    else:
        print(
            f"{_YELLOW}  Current _CUSTOM_NLTK_DATA_DIRECTORY: "
            f"(not set or unavailable){_RESET}"
        )


def _collect_corpus_removal_targets(
    nltk_data_paths: list[Path],
) -> list[tuple[str, Path]]:
    """Build stopwords/wordnet candidate paths that may be removed."""
    targets: list[tuple[str, Path]] = []
    seen: set[str] = set()

    corpora: tuple[tuple[str, str], ...] = (
        ("stopwords", "NLTK stopwords corpus data"),
        ("wordnet", "NLTK WordNet corpus data"),
    )

    for corpus_id, label in corpora:
        for base in nltk_data_paths:
            candidates: list[Path] = [
                base / "corpora" / corpus_id,
                base / "corpora" / f"{corpus_id}.zip",
                base / corpus_id,
                base / f"{corpus_id}.zip",
            ]

            base_name = base.name.lower()
            if base_name == corpus_id:
                candidates.append(base)
            if base_name == "corpora":
                candidates.extend(
                    [
                        base / corpus_id,
                        base / f"{corpus_id}.zip",
                    ]
                )

            for candidate in candidates:
                normalized = _normalize_abs_path(candidate)
                key = os.path.normcase(str(normalized))
                if key in seen:
                    continue
                seen.add(key)
                targets.append((label, normalized))

    return targets


def _collect_governance_removal_targets() -> list[tuple[str, Path]]:
    """Return governance directories/files created by this script."""
    return [
        (
            "Stopwords governance license directory",
            _GOVERNANCE / "licenses" / "nltk_stopwords",
        ),
        (
            "Stopwords governance consent directory",
            _GOVERNANCE / "consents" / "nltk_stopwords",
        ),
        (
            "WordNet governance license directory",
            _GOVERNANCE / "licenses" / "wordnet",
        ),
        (
            "WordNet governance consent directory",
            _GOVERNANCE / "consents" / "wordnet",
        ),
    ]


def _remove_target(target_path: Path) -> bool:
    """Delete one file/directory target when it exists and passes guard."""
    resolved_target = _normalize_abs_path(target_path)
    if not _passes_minimum_root_guard(resolved_target):
        return False

    if not resolved_target.exists():
        return True

    try:
        if resolved_target.is_dir():
            shutil.rmtree(resolved_target)
        else:
            resolved_target.unlink()
        return True
    except Exception as exc:
        print(f"{_RED}  ✖  Failed to remove '{resolved_target}': {exc}{_RESET}")
        return False


def _corpus_lookup_status(
    corpus_id: str,
    *,
    search_paths: list[Path] | None = None,
) -> tuple[bool, str]:
    """Return corpus installation status and location/error details.

    Checks configured NLTK paths first. Falls back to nltk.data.find only when
    no configured paths are available.
    """
    resolved_paths = (
        search_paths if search_paths is not None else _resolve_nltk_data_paths()
    )
    label_fragment = "stopwords" if corpus_id == "stopwords" else "wordnet"
    for label, candidate in _collect_corpus_removal_targets(resolved_paths):
        if label_fragment in label.lower() and candidate.exists():
            return True, str(candidate)

    if resolved_paths:
        return False, "not found under configured NLTK paths"

    try:
        import importlib  # noqa: PLC0415

        nltk = cast(Any, importlib.import_module("nltk"))
    except Exception:
        return False, "NLTK package not installed"

    try:
        location = nltk.data.find(f"corpora/{corpus_id}")
        return True, str(location)
    except LookupError:
        return False, "not found on nltk.data.path"
    except Exception as exc:
        return False, f"lookup failed: {exc}"


def _install_action() -> None:
    w = 70
    print()
    print(f"{_CYAN}{_BOLD}{'=' * w}{_RESET}")
    print(
        f"{_CYAN}{_BOLD}{'  NLTK Stopwords + WordNet  —  License & Download':^{w}}{_RESET}"
    )
    print(f"{_CYAN}{_BOLD}{'=' * w}{_RESET}")
    print()

    # --- resolve component versions ----------------------------------------
    print(f"{_DIM}  Determining component versions...{_RESET}")

    nltk_version = _nltk_installed_version()
    if nltk_version is None:
        print(f"{_RED}  ✖  Could not determine installed NLTK version.{_RESET}")
        print(
            f"{_RED}     Complete Step 2 (pip install) before running this step.{_RESET}"
        )
        sys.exit(1)
    print(f"  NLTK version    : {nltk_version}")

    wordnet_version = _wordnet_version_from_index()
    if wordnet_version is None:
        print(
            f"{_YELLOW}  ⚠  Could not read WordNet version from NLTK data index."
            f"  Using 'current' as label.{_RESET}"
        )
        wordnet_version = "current"
    else:
        print(f"  WordNet version : {wordnet_version}")

    # --- capture identity once for both consent records --------------------
    print()
    identity = _capture_identity()
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"

    # --- stopwords (NLTK Apache-2.0) ---------------------------------------
    print()
    print(f"{_ORANGE}  Stopwords compliance note:{_RESET}")
    print(
        f"{_ORANGE}  NLTK classifies stopwords as an NLTK data package with unclarified, unknown,{_RESET}"
    )
    print(f"{_ORANGE}  ambiguous, or citation-only licensing status.{_RESET}")
    print()
    _download_corpus(
        key="nltk_stopwords",
        display="NLTK stopwords corpus",
        corpus_id="stopwords",
        license_url=_NLTK_LICENSE_URL_TEMPLATE.format(version=nltk_version),
        license_version=nltk_version,
        strip_html=False,
        lic_dir=_GOVERNANCE / "licenses" / "nltk_stopwords",
        consent_dir=_GOVERNANCE / "consents" / "nltk_stopwords",
        identity=identity,
        now=now,
        additionalk_info=(
            "NLTK software license (used by downloader): Apache 2.0",
            "Stopwords corpus license status: Unclarified / Unknown according to "
            "NLTK dataset metadata",
        ),
    )

    # --- wordnet (Princeton WordNet License — plain text via SPDX) ---------
    print()
    print(f"{_DIM}  license metadata in the local runtime install log.{_RESET}")
    print()
    _download_corpus(
        key="wordnet",
        display="WordNet corpus (Princeton University)",
        corpus_id="wordnet",
        license_url=_WORDNET_LICENSE_URL,
        license_version=wordnet_version,
        strip_html=False,
        lic_dir=_GOVERNANCE / "licenses" / "wordnet",
        consent_dir=_GOVERNANCE / "consents" / "wordnet",
        identity=identity,
        now=now,
    )

    print()
    print(f"{_GREEN}{_BOLD}{'=' * w}{_RESET}")
    print(
        f"{_GREEN}{_BOLD}{'  NLTK corpora downloaded and consent recorded.':^{w}}{_RESET}"
    )
    print(f"{_GREEN}{_BOLD}{'=' * w}{_RESET}")
    print()


def _status_action() -> None:
    """Show corpus and governance installation status without changing files."""
    print()
    print(f"{_CYAN}{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'  NLTK Stopwords + WordNet  —  Status':^70}{_RESET}")
    print(f"{_CYAN}{_BOLD}{'=' * 70}{_RESET}")

    nltk_version = _nltk_installed_version()
    if nltk_version:
        print(f"\nNLTK package version : {nltk_version}")
    else:
        print(f"\nNLTK package version : {_YELLOW}not installed{_RESET}")

    print()
    _print_custom_nltk_directory_hint()

    nltk_data_paths = _resolve_nltk_data_paths()
    _print_nltk_data_path_status(nltk_data_paths)

    stopwords_ok, stopwords_detail = _corpus_lookup_status(
        "stopwords", search_paths=nltk_data_paths
    )
    wordnet_ok, wordnet_detail = _corpus_lookup_status(
        "wordnet", search_paths=nltk_data_paths
    )

    print("\nCorpus status:")
    print(
        "  stopwords: "
        f"{'installed' if stopwords_ok else 'missing'}"
        f" ({stopwords_detail})"
    )
    print(
        "  wordnet  : "
        f"{'installed' if wordnet_ok else 'missing'}"
        f" ({wordnet_detail})"
    )

    print("\nGovernance paths:")
    for label, path in _collect_governance_removal_targets():
        state = "present" if path.exists() else "not present"
        print(f"  {path}  ({state})  [{label}]")


def _remove_action() -> None:
    """Remove installed corpora files and governance metadata with safeguards."""
    print()
    print(f"{_CYAN}{_BOLD}{'=' * 70}{_RESET}")
    print(
        f"{_CYAN}{_BOLD}{'  NLTK Stopwords + WordNet  —  Removal Preview':^70}{_RESET}"
    )
    print(f"{_CYAN}{_BOLD}{'=' * 70}{_RESET}")

    _print_custom_nltk_directory_hint()

    nltk_data_paths = _resolve_nltk_data_paths()
    _print_nltk_data_path_status(nltk_data_paths, present_only=True)

    corpus_targets = _collect_corpus_removal_targets(nltk_data_paths)
    governance_targets = _collect_governance_removal_targets()
    all_targets = corpus_targets + governance_targets

    present_targets = [(label, path) for label, path in all_targets if path.exists()]

    if not present_targets:
        print(
            "\nNo stopwords/wordnet data or governance metadata found - nothing to remove."
        )
        return

    print(f"\n{_ORANGE}Present paths queued for removal:{_RESET}")
    for label, path in present_targets:
        print(f"  {path}  [{label}]")

    for _, path in present_targets:
        if not _passes_minimum_root_guard(path):
            print(f"{_RED}Removal aborted due to unsafe target path.{_RESET}")
            sys.exit(1)

    if not _confirm("Proceed with removal?"):
        print("Aborted.")
        return

    removed = 0
    failed = 0
    for _, path in present_targets:
        if _remove_target(path):
            removed += 1
        else:
            failed += 1

    print()
    if failed:
        print(
            f"{_YELLOW}Removal finished with errors: removed {removed}, failed {failed}.{_RESET}"
        )
        sys.exit(1)

    print(f"{_GREEN}Removal complete: removed {removed} path(s).{_RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install/remove/show status for NLTK stopwords and WordNet corpora."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["install", "remove", "status"],
        help="Action to perform (default: status)",
    )
    args = parser.parse_args()

    if args.action == "install":
        _install_action()
    elif args.action == "remove":
        _remove_action()
    else:
        _status_action()


if __name__ == "__main__":
    main()
