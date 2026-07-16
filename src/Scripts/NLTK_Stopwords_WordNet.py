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
    python src/Scripts/NLTK_Stopwords_WordNet.py
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_PROJECT_ROOT = _SRC_DIR.parent
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


_identity_cache: dict | None = None


def _capture_identity() -> dict:
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
    identity: dict,
    now: str,
    lic_dir: Path,
    consent_dir: Path,
) -> None:
    """Write LICENSE.txt, license_meta.json, and install_meta.json."""
    lic_path = lic_dir / "LICENSE.txt"
    lic_meta_path = lic_dir / "license_meta.json"
    consent_path = consent_dir / "install_meta.json"

    lic_hash = _compute_hash(license_text)

    lic_dir.mkdir(parents=True, exist_ok=True)
    lic_path.write_text(license_text, encoding="utf-8")

    lic_meta: dict = {
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
    install_meta: dict = {
        "component": key,
        "corpus": key,
        "installer": "nltk.download",
        **identity,
        "accepted_at": now,
        "consent": True,
    }
    with consent_path.open("w", encoding="utf-8") as fh:
        json.dump(install_meta, fh, indent=2, ensure_ascii=False)

    print(
        f"{_GREEN}  ✔  Consent recorded: {consent_path.relative_to(_PROJECT_ROOT)}{_RESET}"
    )


# ---------------------------------------------------------------------------
# Helpers: pager
# ---------------------------------------------------------------------------


_END_OF_LICENSE = (
    "\n\n" + "─" * 70 + "\n" "  >>>>>  End of license  <<<<<\n" + "─" * 70 + "\n"
)


def _page_text(text: str, *, end_marker: bool = False) -> None:
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
            q = (
                input(f"{_ORANGE}[Enter] next page  [q] quit view: {_RESET}")
                .strip()
                .lower()
            )
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
    identity: dict,
    now: str,
) -> None:
    w = 70
    print()
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_WHITE}  {display}{_RESET}")
    print(f"{_DIM}  Version     : {license_version}{_RESET}")
    print(f"{_DIM}  License URL : {license_url}{_RESET}")
    print(f"{_CYAN}{'─' * w}{_RESET}")
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
        import nltk  # noqa: PLC0415 — intentional late import

        # Determine effective download directory (first writable path in nltk.data.path).
        download_dir: str | None = None
        for candidate in nltk.data.path:
            try:
                Path(candidate).mkdir(parents=True, exist_ok=True)
                if os.access(candidate, os.W_OK):
                    download_dir = candidate
                    break
            except Exception:
                continue

        if download_dir:
            print(f"{_DIM}  Download directory : {download_dir}{_RESET}")
        else:
            print(
                f"{_YELLOW}  ⚠  Could not determine a writable NLTK data directory.{_RESET}"
            )
            print(f"{_DIM}  NLTK search path   : {nltk.data.path}{_RESET}")

        # Also show the project's custom NLTK data path if configured.
        try:
            from Config.Config import Config  # noqa: PLC0415

            custom_dir: str = Config().get_str("_CUSTOM_NLTK_DATA_DIRECTORY")
            if custom_dir:
                print(f"{_DIM}  Config_Global path : {custom_dir}{_RESET}")
                print(
                    f"{_DIM}  (Set _CUSTOM_NLTK_DATA_DIRECTORY in Config_Global.py"
                    f" to change where the application reads NLTK data.){_RESET}"
                )
        except Exception:
            pass

        print(f"{_DIM}  Downloading NLTK corpus '{corpus_id}'...{_RESET}")
        nltk.download(corpus_id, quiet=False)
        print(f"{_GREEN}  ✔  '{corpus_id}' downloaded.{_RESET}")
    except ImportError:
        print(
            f"{_RED}  ✖  NLTK is not installed."
            f"  Run Setup.py Step 2 (pip install) before this step.{_RESET}"
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
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
    print(f"{_DIM}  Stopwords compliance note:{_RESET}")
    print(
        f"{_DIM}  The NLTK Stopwords Corpus is not bundled with RAG-LCC and is not included{_RESET}"
    )
    print(f"{_DIM}  in the signed release manifest.{_RESET}")
    print(
        f"{_DIM}  NLTK classifies stopwords as an NLTK data package with unclarified, unknown,{_RESET}"
    )
    print(
        f"{_DIM}  ambiguous, or citation-only licensing status. Because of that, RAG-LCC does{_RESET}"
    )
    print(
        f"{_DIM}  not install it automatically. The consent decision and installed package{_RESET}"
    )
    print(f"{_DIM}  metadata are recorded in the local runtime install log.{_RESET}")
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
    )

    # --- wordnet (Princeton WordNet License — plain text via SPDX) ---------
    print()
    print(f"{_DIM}  WordNet compliance note:{_RESET}")
    print(
        f"{_DIM}  The NLTK WordNet corpus is not bundled with RAG-LCC and is not included in{_RESET}"
    )
    print(
        f"{_DIM}  the signed release manifest unless explicitly shipped as a release artifact.{_RESET}"
    )
    print(
        f"{_DIM}  If the user enables functionality requiring WordNet, setup guides the user{_RESET}"
    )
    print(
        f"{_DIM}  through an explicit local download and consent step. RAG-LCC records the{_RESET}"
    )
    print(
        f"{_DIM}  download decision, installed data package id, timestamp, and available{_RESET}"
    )
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


if __name__ == "__main__":
    main()
