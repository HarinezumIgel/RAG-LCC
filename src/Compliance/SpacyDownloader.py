"""
SpacyDownloader - spaCy model license consent and model installer.

Flow mirrors ArgosDownloader:
  1. Show the spaCy models license and ask for acceptance.
  2. Ask confirmation before downloading missing model packages.
  3. Record license metadata + download consent metadata.

Metadata paths:
  - ModelGovernance/licenses/spacy_models/LICENSE.txt
  - ModelGovernance/licenses/spacy_models/license_meta.json
  - ModelGovernance/consents/spacy_models/download_meta.json
"""

import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, cast

import requests

from Commons.DriveRootGuard import is_drive_root, resolve_guard_path
from Commons.Exceptions import SpacyConsentMissingError
from Gui.Colors import BRIGHT_BLUE, GREEN, ORANGE, RED, RESET, YELLOW
from Gui.LicensePager import show_license
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Scripts.SpacyLanguageModelsImpl import install_models
from Scripts.SpacyLanguageModelsImpl import \
    installed_models as impl_installed_models
from Scripts.SpacyLanguageModelsImpl import \
    missing_models as impl_missing_models
from Scripts.SpacyLanguageModelsImpl import (remove_models,
                                             report_spacy_model_license_hints)
from Scripts.SpacyLanguageModelsImpl import show_status as show_spacy_status

_CONSENT_DISCLAIMER = (
    "This record indicates technical acknowledgement of license terms. "
    "It does not constitute a legal contract or replace formal legal review."
)


class SpacyDownloader:
    """Manage spaCy model installation with license consent."""

    _LICENSE_URL = "https://raw.githubusercontent.com/explosion/spaCy/main/LICENSE"
    _LICENSE_DIR_REL = os.path.join("ModelGovernance", "licenses", "spacy_models")
    _CONSENT_DIR_REL = os.path.join("ModelGovernance", "consents", "spacy_models")

    def __init__(self, project_root: str, models: Sequence[str]) -> None:
        self.root: str = project_root
        deduped: List[str] = sorted(
            {model.strip() for model in models if model.strip()}
        )
        self.models: List[str] = deduped or ["en_core_web_sm"]

        self.license_dir: str = os.path.join(project_root, self._LICENSE_DIR_REL)
        self.license_path: str = os.path.join(self.license_dir, "LICENSE.txt")
        self.license_meta_path: str = os.path.join(
            self.license_dir, "license_meta.json"
        )
        self.consent_dir: str = os.path.join(project_root, self._CONSENT_DIR_REL)
        self.download_meta_path: str = os.path.join(
            self.consent_dir, "download_meta.json"
        )

        self.helpers: Helpers = Helpers()
        self.file_utils: FileUtils = FileUtils()
        self.pretty: PrettyWriter = PrettyWriter(always_on=True)
        self.logger: logging.Logger = self.helpers.setup_logger("Compliance")
        self._license_url_used: str = self._LICENSE_URL
        self._identity_cache: Dict[str, object] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _check_existing_consent(self) -> bool:
        """Return True and emit a green message if consent JSON is valid."""
        shift_reason = self._consent_shift_reason()
        if shift_reason is not None:
            self.pretty.write(
                "W",
                "spaCy License",
                shift_reason,
                color=YELLOW,
            )
            return False

        n = len(self.models)
        self.logger.info("spaCy model consent valid (%d model(s))", n)
        self.pretty.write(
            "O",
            "spaCy License",
            f"Consent valid for {n} configured model(s)",
            color=GREEN,
        )
        return True

    def report_consent_status(self) -> None:
        """Emit the consent-valid message if metadata exists and is valid."""
        self._check_existing_consent()

    def assert_consent_current_or_raise(self) -> None:
        """Require current consent metadata; raise with script guidance if stale."""
        if self._check_existing_consent():
            return

        configured_models = ", ".join(self.models)
        raise SpacyConsentMissingError(
            "spaCy model-package consent metadata is missing or outdated for "
            "the current configuration"
            + (f" ({configured_models})" if configured_models else "")
            + ". Run python src/Scripts/SpacyLanguageModels.py install."
        )

    def ensure_models(self) -> bool:
        """Check consent and install missing models.

        Returns True when models are ready for use; False on declined consent,
        fetch failures, or installation failures.
        """
        if self._check_existing_consent():
            missing_model_names = self._missing_models(self.models)
            if missing_model_names:
                self.pretty.write(
                    "W",
                    "spaCy Download",
                    "Consent valid but models are missing ("
                    + ", ".join(missing_model_names)
                    + ") - reinstalling.",
                    color=YELLOW,
                )
                all_ok = self._install_models(missing_model_names)
                if all_ok:
                    self.pretty.write(
                        "O",
                        "spaCy Download",
                        f"Installed {len(missing_model_names)} spaCy model package(s).",
                        color=GREEN,
                    )
                    report_spacy_model_license_hints(self.models)
                return all_ok

            report_spacy_model_license_hints(self.models)
            return True

        license_text = self._fetch_license()
        if license_text is None:
            return False
        license_hash: str = self._compute_text_hash(license_text)

        print(f"{BRIGHT_BLUE}\n{'=' * 70}")
        print("  spaCy Models License Consent")
        print(f"  Models: {', '.join(self.models)}")
        print(f"{'=' * 70}{RESET}\n")

        if not self._wait_for_enter(
            "Press Enter to review the spaCy models license ..."
        ):
            self.pretty.write(
                "W",
                "spaCy License",
                "Input cancelled before license review - install aborted.",
                color=YELLOW,
            )
            return False
        self._show_license_pager(license_text)

        print(f"\n{ORANGE}>>>> Do you accept the spaCy models license?{RESET}")
        if not self._prompt_yes_no("Accept? [y/N] "):
            self._show_non_consent_msg()
            return False

        print(f"\n{BRIGHT_BLUE}The following model packages will be checked:{RESET}")
        for model in self.models:
            print(f"  spaCy model  {model}")

        print(f"\n{ORANGE}>>>> Proceed with model install/download?{RESET}")
        if not self._prompt_yes_no("Proceed? [y/N] "):
            self._show_non_consent_msg()
            return False

        self.logger.info(
            "User accepted download for %d spaCy model(s)", len(self.models)
        )

        identity = self._capture_acceptance_identity_once()
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"

        license_meta: Dict[str, Any] = {
            "component": "spacy_models",
            "license_url": getattr(self, "_license_url_used", self._LICENSE_URL),
            "license_hash": license_hash,
            **identity,
            "accepted_at": now,
            "consent": True,
            "disclaimer": _CONSENT_DISCLAIMER,
        }
        os.makedirs(self.license_dir, exist_ok=True)
        with open(self.license_path, "w", encoding="utf-8") as fh:
            fh.write(license_text)
        with open(self.license_meta_path, "w", encoding="utf-8") as fh:
            json.dump(license_meta, fh, indent=2, ensure_ascii=False)

        download_meta: Dict[str, Any] = {
            "component": "spacy_models",
            "models": list(self.models),
            **identity,
            "accepted_at": now,
            "source": "downloaded",
            "consent": True,
            "disclaimer": _CONSENT_DISCLAIMER,
        }
        os.makedirs(self.consent_dir, exist_ok=True)
        with open(self.download_meta_path, "w", encoding="utf-8") as fh:
            json.dump(download_meta, fh, indent=2, ensure_ascii=False)

        self.pretty.write(
            "O",
            "spaCy License",
            f"License consent recorded in {self.license_meta_path}",
            color=GREEN,
        )
        self.pretty.write(
            "O",
            "spaCy Download",
            f"Download consent recorded in {self.download_meta_path}",
            color=GREEN,
        )

        missing_model_names = self._missing_models(self.models)
        if not missing_model_names:
            self.pretty.write(
                "O",
                "spaCy Download",
                "All configured spaCy models are already installed.",
                color=GREEN,
            )
            report_spacy_model_license_hints(self.models)
            return True

        all_ok = self._install_models(missing_model_names)
        if all_ok:
            self.pretty.write(
                "O",
                "spaCy Download",
                f"Installed {len(missing_model_names)} spaCy model package(s).",
                color=GREEN,
            )
            report_spacy_model_license_hints(self.models)
        return all_ok

    def _show_non_consent_msg(self) -> None:
        self.pretty.write(
            "W",
            "spaCy License",
            "Download declined. Run python src/Scripts/SpacyLanguageModels.py install when you are ready.",
            color=YELLOW,
        )
        self.pretty.write(
            "W",
            "spaCy License",
            "If spaCy model packages are missing, graph/regex retrieval and query-rewrite gating may fail.",
            color=ORANGE,
        )

    def remove_all(self) -> None:
        """Uninstall configured spaCy model packages from the active environment."""
        installed = self.installed_models()
        if not installed:
            self.pretty.write(
                "I",
                "spaCy",
                "No configured model packages installed - nothing to remove.",
            )
            return

        self.pretty.write(
            "W",
            "spaCy",
            f"About to uninstall {len(installed)} configured model package(s):",
            color=YELLOW,
        )
        for model in installed:
            print(f"    {model}")

        if not self._prompt_yes_no("Uninstall all of these? [y/N] "):
            self.pretty.write("I", "spaCy", "Removal cancelled.")
            return

        for model in installed:
            print(f"  Removing {model} ...")

        def _on_uninstall_error(model: str, rc: int) -> None:
            self.pretty.write(
                "W",
                "spaCy",
                f"pip uninstall failed for {model} (exit={rc})",
                color=YELLOW,
            )

        removed, total = remove_models(
            installed,
            on_uninstall_error=_on_uninstall_error,
        )

        self.pretty.write(
            "O",
            "spaCy",
            f"Removed {removed}/{total} configured model package(s).",
            color=GREEN,
        )

    def installed_models(self) -> list[str]:
        """Return configured spaCy models that are currently installed."""
        return impl_installed_models(self.models)

    def has_consent_metadata(self) -> bool:
        """Return True when any spaCy consent/license metadata file exists."""
        return any(
            os.path.isfile(path)
            for path in (
                self.license_path,
                self.license_meta_path,
                self.download_meta_path,
            )
        )

    def consent_cleanup_directories(self) -> list[str]:
        """Return metadata directories affected by consent cleanup."""
        return [self.license_dir, self.consent_dir]

    def remove_consent(self) -> None:
        """Remove consent metadata via FileUtils path guard."""
        for cleanup_dir in self.consent_cleanup_directories():
            if not self._passes_minimum_root_guard(cleanup_dir):
                return

        for target_path in (
            self.license_meta_path,
            self.license_path,
            self.download_meta_path,
        ):
            if not self._passes_minimum_root_guard(target_path):
                continue
            if os.path.isfile(target_path):
                self.file_utils.delete_file_or_dir(target_path)

    def show_status(self) -> None:
        """Print configured model packages and whether they are installed."""
        show_spacy_status(self.models)

    def _missing_models(self, models: Sequence[str]) -> list[str]:
        """Return configured model packages that are currently missing."""
        return impl_missing_models(models)

    def _install_models(self, models: Sequence[str]) -> bool:
        """Install model packages and report per-model install failures."""

        def _on_install_error(model: str, rc: int) -> None:
            self.pretty.write(
                "E",
                "spaCy Download",
                f"Failed to install spaCy model '{model}' (exit={rc})",
                color=RED,
            )

        return install_models(models, on_install_error=_on_install_error)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _consent_shift_reason(self) -> str | None:
        """Return a message when consent metadata is missing/stale, else None."""
        if (
            not os.path.isfile(self.license_path)
            or not os.path.isfile(self.license_meta_path)
            or not os.path.isfile(self.download_meta_path)
        ):
            return (
                "Consent metadata files are missing "
                "(LICENSE.txt, license_meta.json, or download_meta.json)."
            )

        try:
            with open(self.license_path, "r", encoding="utf-8") as fh:
                license_hash: str = self._compute_text_hash(fh.read())
            with open(self.license_meta_path, "r", encoding="utf-8") as fh:
                meta: Dict[str, Any] = json.load(fh)
        except Exception as exc:
            return f"Unable to read spaCy consent metadata: {exc}"

        if meta.get("consent") is not True:
            return "License consent flag is not set to true in license_meta.json."
        if meta.get("license_hash") != license_hash:
            return "spaCy license text hash changed since acceptance - re-consent required."

        consented_models: set[str] = set()
        try:
            with open(self.download_meta_path, "r", encoding="utf-8") as fh:
                dl_meta: Dict[str, Any] = json.load(fh)
            raw_models_obj = dl_meta.get("models")
            if isinstance(raw_models_obj, list):
                raw_models = cast(List[Any], raw_models_obj)
                for model in raw_models:
                    if not isinstance(model, str):
                        continue
                    model_name = model.strip()
                    if model_name:
                        consented_models.add(model_name)
        except Exception as exc:
            return f"Unable to read spaCy download consent metadata: {exc}"

        configured_models = set(self.models)
        new_models = configured_models - consented_models
        if new_models:
            return (
                "Configuration adds "
                f"{len(new_models)} new model(s) ({', '.join(sorted(new_models))}) "
                "- re-consent required."
            )

        return None

    def _fetch_license(self) -> str | None:
        """Download the spaCy models license from GitHub."""
        url = self._LICENSE_URL
        self.logger.info("Fetching spaCy models license from %s", url)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            self._license_url_used = url
            return resp.text
        except Exception as exc:
            msg = f"Failed to fetch spaCy models license from {url}: {exc}"
            self.logger.error(msg)
            self.pretty.write("E", "spaCy License", msg, color=RED)
            return None

    @staticmethod
    def _show_license_pager(text: str) -> None:
        """Display license text page-by-page."""
        show_license(text)

    def _passes_minimum_root_guard(self, target_path: str) -> bool:
        """Apply minimum root guard before any delete operation."""
        normalized_path = resolve_guard_path(target_path)
        if is_drive_root(normalized_path):
            self.pretty.write(
                "E",
                "Path Guard",
                f"Refusing to delete root or drive path '{normalized_path}'.",
                color=RED,
            )
            return False
        return True

    def _wait_for_enter(self, prompt: str) -> bool:
        try:
            _ = input(prompt)
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    def _prompt_yes_no(self, prompt: str) -> bool:
        try:
            answer = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() == "y"

    def _capture_acceptance_identity_once(self) -> Dict[str, object]:
        """Capture identity metadata once per process for consent records."""
        identity_cache_obj = getattr(self, "_identity_cache", None)
        if isinstance(identity_cache_obj, dict):
            return cast(Dict[str, object], identity_cache_obj)

        git_user: str | None = None
        try:
            git_user = (
                subprocess.check_output(
                    ["git", "config", "user.email"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            if not git_user:
                git_user = None
        except Exception:
            git_user = None

        try:
            os_user = getpass.getuser()
        except Exception:
            os_user = os.getenv("_USER") or os.getenv("USERNAME") or "unknown-user"

        accepted_by = (git_user or os_user or "").strip()
        accepted_by_source = "git" if git_user else "os"
        if not accepted_by:
            accepted_by = "unknown-user"
            accepted_by_source = "os"

        accepted_by_verified = False
        print(
            f"Detected identity for acceptance: {accepted_by} "
            f"(source: {accepted_by_source})"
        )
        prompt_for_override = False
        stdin_obj = getattr(sys, "stdin", None)
        if stdin_obj is not None:
            try:
                prompt_for_override = bool(stdin_obj.isatty())
            except Exception:
                prompt_for_override = False

        if prompt_for_override:
            try:
                override = input(
                    f"{YELLOW}Press Enter to accept as-is or type your email/ID to override: {RESET}"
                ).strip()
                if override:
                    accepted_by = override
                    accepted_by_source = "interactive"
            except (EOFError, KeyboardInterrupt):
                pass

        host = (socket.gethostname() or "").strip() or "unknown-host"
        identity: Dict[str, object] = {
            "accepted_by": accepted_by,
            "accepted_by_source": accepted_by_source,
            "accepted_by_verified": accepted_by_verified,
            "host": host,
            "pid": os.getpid(),
        }
        self._identity_cache = identity
        return identity

    @staticmethod
    def _compute_text_hash(text: str) -> str:
        """SHA-256 hash of canonicalized text (CRLF->LF, stripped)."""
        return hashlib.sha256(
            text.replace("\r\n", "\n").strip().encode("utf-8")
        ).hexdigest()
