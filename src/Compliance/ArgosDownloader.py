"""
ArgosDownloader — Argos Translate license consent and package installer.

Follows the same interactive prompt pattern as HFDownloader:
    1. Show the Argos Translate project license (MIT) and ask the user to
         accept.
    2. Ask confirmation to download language packages. Package artifacts may
      include additional third-party model/tokenizer/vocabulary/data licenses.
        3. If the user declines at either step, return without install.

The license is downloaded from GitHub and stored under
ModelGovernance/licenses/argos_translate/LICENSE.txt with license
metadata in license_meta.json.  Download consent is recorded in
ModelGovernance/consents/argos_translate/download_meta.json.
The check is a no-op on subsequent runs while the license hash matches.
Package-specific notices can be found in each installed package README.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, cast

import requests

from Commons.DriveRootGuard import is_drive_root, resolve_guard_path
from Commons.Exceptions import ArgosConsentMissingError
from Compliance.ArgosSpacySentencizerPatch import \
    prepare_argos_runtime_for_patch
from Gui.LicensePager import show_license
from Helpers.Helpers import Helpers
from Scripts.ArgosTranslatePackagesImpl import install_language_pairs
from Scripts.ArgosTranslatePackagesImpl import \
    installed_packages as impl_installed_packages
from Scripts.ArgosTranslatePackagesImpl import (
    missing_language_pairs, remove_packages,
    report_argos_package_license_hints)
from Scripts.ArgosTranslatePackagesImpl import show_status as show_argos_status

prepare_argos_runtime_for_patch()

from Compliance.SharedHelpers import SharedHelpers
from Gui.Colors import BRIGHT_BLUE, GREEN, ORANGE, RED, RESET, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils

_CONSENT_DISCLAIMER = (
    "This record indicates technical acknowledgement of license terms. "
    "It does not constitute a legal contract or replace formal legal review."
)


class ArgosDownloader:
    """Manage Argos Translate package installation with license consent."""

    _LICENSE_URL = (
        "https://raw.githubusercontent.com/argosopentech/argos-translate/master/LICENSE"
    )
    _LICENSE_DIR_REL = os.path.join("ModelGovernance", "licenses", "argos_translate")
    _CONSENT_DIR_REL = os.path.join("ModelGovernance", "consents", "argos_translate")
    _TOKENIZER_RESOURCES_DEFAULT_DIR = os.path.join(
        os.path.expanduser("~"), ".local", "share", "resources"
    )
    _TOKENIZER_RESOURCES_DEFAULT_PATH = os.path.normpath(
        os.path.abspath(os.path.expanduser(_TOKENIZER_RESOURCES_DEFAULT_DIR))
    )
    if is_drive_root(_TOKENIZER_RESOURCES_DEFAULT_PATH):
        raise RuntimeError(
            "Invalid default tokenizer resources path: "
            f"{_TOKENIZER_RESOURCES_DEFAULT_PATH!r} resolves to a drive/filesystem root."
        )

    def __init__(
        self,
        project_root: str,
        languages: List[Tuple[str, str]],
        *,
        emit_startup_inventory: bool = True,
    ) -> None:
        self.root: str = project_root
        self.languages: List[Tuple[str, str]] = languages

        self.license_dir: str = os.path.join(project_root, self._LICENSE_DIR_REL)
        self.license_path: str = os.path.join(self.license_dir, "LICENSE.txt")
        self.license_meta_path: str = os.path.join(
            self.license_dir, "license_meta.json"
        )
        self.consent_dir: str = os.path.join(project_root, self._CONSENT_DIR_REL)
        self.download_meta_path: str = os.path.join(
            self.consent_dir, "download_meta.json"
        )

        self.shared: SharedHelpers = SharedHelpers(
            emit_startup_inventory=emit_startup_inventory
        )
        self.helpers: Helpers = Helpers()
        self.file_utils: FileUtils = FileUtils()
        self.pretty: PrettyWriter = PrettyWriter(always_on=True)
        self.logger: logging.Logger = self.helpers.setup_logger("Compliance")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _check_existing_consent(self) -> bool:
        """Return True and emit a green message if consent JSON is valid."""
        shift_reason = self._consent_shift_reason()
        if shift_reason is not None:
            self.pretty.write(
                "W",
                "Argos License",
                shift_reason,
                color=YELLOW,
            )
            return False

        n = len(self.languages)
        self.logger.info("Argos Translate consent valid (%d pair(s))", n)
        self.pretty.write(
            "O",
            "Argos License",
            f"Consent valid for {n} configured language pair(s)",
            color=GREEN,
        )
        self.pretty.write(
            "I",
            "Tokenizer Resources",
            self._tokenizer_resources_location_hint(),
        )
        return True

    def report_consent_status(self) -> None:
        """Emit the consent status message from current metadata state."""
        self._check_existing_consent()

    def assert_consent_current_or_raise(self) -> None:
        """Require current consent metadata; raise with script guidance if stale."""
        if self._check_existing_consent():
            return

        configured_pairs = ", ".join(f"{a}->{b}" for a, b in self.languages)
        raise ArgosConsentMissingError(
            "Argos Translate consent metadata is missing or outdated for the "
            "current configuration"
            + (f" ({configured_pairs})" if configured_pairs else "")
            + ". Run python src/Scripts/ArgosTranslatePackages.py install."
        )

    def ensure_packages(self) -> bool:
        """Check consent and install missing packages.

        Returns True if new packages were just installed (caller should refresh
        the installed-languages cache), False otherwise (consent declined, or
        packages were already present and no refresh is needed).
        """
        # --- existing consent still valid? ---
        if self._check_existing_consent():
            # Consent is valid; install any packages not yet present
            try:
                missing_pairs = missing_language_pairs(self.languages)
            except Exception as exc:
                self.logger.warning("Argos package inventory unavailable: %s", exc)
                missing_pairs: set[tuple[str, str]] = set()

            if missing_pairs:
                self.pretty.write(
                    "W",
                    "Argos Download",
                    "Consent valid but packages not installed ("
                    + ", ".join(f"{a}\u2192{b}" for a, b in sorted(missing_pairs))
                    + ") — reinstalling.",
                    color=YELLOW,
                )
                self._install_packages()
                return True  # newly installed — caller should refresh language cache
            return True  # already present with valid consent — packages are ready

        # --- download license from upstream ---
        license_text = self._fetch_license()
        if license_text is None:
            return False
        license_hash: str = self.shared.compute_text_hash(license_text)

        # --- prompt: accept the license ---
        print(f"{BRIGHT_BLUE}\n{'=' * 70}")
        print("  Argos Translate (MIT) License Consent")
        print(
            f"  Language pairs: {', '.join(f'{a}\u2192{b}' for a, b in self.languages)}"
        )
        print(f"{'=' * 70}{RESET}\n")

        input("Press Enter to review the Argos Translate license ...")
        self._show_license_pager(license_text)

        print(f"\n{ORANGE}>>>> Do you accept the Argos Translate license?{RESET}")
        ans = input("Accept? [y/N] ").strip().lower()
        if ans != "y":
            self._show_non_consent_msg()
            return False

        # --- prompt: proceed with download ---
        print(f"\n{BRIGHT_BLUE}The following will be downloaded:{RESET}")
        for from_code, to_code in self.languages:
            print(
                f"  Argos package  {from_code} \u2192 {to_code}  "
                "(may include model/tokenizer/vocabulary/data artifacts)"
            )
        print(
            f"{ORANGE}  Note: package artifacts may include additional "
            f"third-party model/tokenizer/vocabulary/data licenses; review each installed "
            f"package README. {RESET}"
        )

        print(f"\n{ORANGE}>>>> Proceed with download?{RESET}")
        ans = input("Download? [y/N] ").strip().lower()
        if ans != "y":
            self._show_non_consent_msg()
            return False
        self.logger.info(
            "User accepted download for %d language pair(s)", len(self.languages)
        )
        # --- persist license + license metadata ---
        identity = self.shared.capture_acceptance_identity_once()
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"

        license_meta: Dict[str, Any] = {
            "component": "argostranslate",
            "license_url": self._LICENSE_URL,
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

        # --- persist download consent ---
        download_meta: Dict[str, Any] = {
            "component": "argostranslate",
            "languages": [list(p) for p in self.languages],
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
            "Argos License",
            f"License consent recorded in {self.license_meta_path}",
            color=GREEN,
        )
        self.pretty.write(
            "O",
            "Argos Download",
            f"Download consent recorded in {self.download_meta_path}",
            color=GREEN,
        )

        # --- install ---
        self._install_packages()
        return True

    def _show_non_consent_msg(self) -> None:
        self.pretty.write(
            "W",
            "Argos License",
            "Download declined. Run python src/Scripts/ArgosTranslatePackages.py install when you are ready.",
            color=YELLOW,
        )
        self.pretty.write(
            "W",
            "Argos License",
            "If translation from banned words (Config_Banned.py) to the document's target language is not possible,"
            "filter chain results degrade considerably because the English words will be applied to the extracted documents in non-English languages.",
            color=ORANGE,
        )

    def remove_all(self) -> None:
        """Uninstall every installed Argos Translate language package."""
        installed = self.installed_packages()
        if not installed:
            self.pretty.write(
                "I", "Argos", "No packages installed — nothing to remove."
            )
            return
        self.pretty.write(
            "W",
            "Argos",
            f"About to uninstall {len(installed)} language package(s):",
            color=YELLOW,
        )
        for pkg in installed:
            print(f"    {pkg.from_code} \u2192 {pkg.to_code}")
        if input("Uninstall all of these? [y/N] ").strip().lower() != "y":
            self.pretty.write("I", "Argos", "Removal cancelled.")
            return
        for pkg in installed:
            print(f"  Removing {pkg.from_code} \u2192 {pkg.to_code} ...")

        def _on_uninstall_error(from_code: str, to_code: str, exc: Exception) -> None:
            self.pretty.write(
                "W",
                "Argos",
                f"Failed to remove {from_code} \u2192 {to_code}: {exc}",
                color=YELLOW,
            )

        removed = remove_packages(installed, on_uninstall_error=_on_uninstall_error)
        self.pretty.write(
            "O", "Argos", f"Removed {removed}/{len(installed)} package(s).", color=GREEN
        )

    def installed_packages(self) -> list[Any]:
        """Return currently installed Argos packages without printing."""
        return impl_installed_packages()

    def has_consent_metadata(self) -> bool:
        """Return True when any Argos consent/license metadata file exists."""
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

    @property
    def tokenizer_resources_dir(self) -> str:
        """Resolved absolute path of the tokenizer resources directory."""
        return self._TOKENIZER_RESOURCES_DEFAULT_PATH

    def remove_tokenizer_resources(self) -> None:
        """Remove the tokenizer resources directory (outside project root)."""
        abs_path = self.tokenizer_resources_dir

        # Minimal safety: block drive/filesystem roots only.
        if not self._passes_minimum_root_guard(abs_path):
            return

        if os.path.isdir(abs_path):
            self.pretty.write(
                "W",
                "Tokenizer Resources",
                f"About to permanently delete directory: {abs_path}",
                color=YELLOW,
            )
            if input("Delete this directory? [y/N] ").strip().lower() != "y":
                self.pretty.write("I", "Tokenizer Resources", "Removal cancelled.")
                return
            shutil.rmtree(abs_path)
            self.pretty.write(
                "O", "Tokenizer Resources", f"Removed {abs_path}", color=GREEN
            )
        else:
            self.pretty.write(
                "I",
                "Tokenizer Resources",
                "No tokenizer resources directory present - nothing to remove.",
            )

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
        """Print currently installed Argos languages and packages."""
        show_argos_status()
        self.pretty.write(
            "I",
            "Tokenizer Resources",
            self._tokenizer_resources_location_hint(),
        )

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
                license_hash = self.shared.compute_text_hash(fh.read())
            with open(self.license_meta_path, "r", encoding="utf-8") as fh:
                meta: Dict[str, Any] = json.load(fh)
        except Exception as exc:
            return f"Unable to read Argos consent metadata: {exc}"

        if meta.get("consent") is not True:
            return "License consent flag is not set to true in license_meta.json."
        if meta.get("license_hash") != license_hash:
            return "Argos license text hash changed since acceptance - re-consent required."

        consented_pairs: set[tuple[str, str]] = set()
        try:
            with open(self.download_meta_path, "r", encoding="utf-8") as fh:
                dl_meta: Dict[str, Any] = json.load(fh)
            raw_languages = cast(list[Any], dl_meta.get("languages") or [])
            for pair in raw_languages:
                if isinstance(pair, (list, tuple)):
                    pair_tuple = cast(tuple[Any, Any], pair)
                    if len(pair_tuple) >= 2:
                        consented_pairs.add((str(pair_tuple[0]), str(pair_tuple[1])))
        except Exception as exc:
            return f"Unable to read Argos download consent metadata: {exc}"

        configured_pairs: set[tuple[str, str]] = {
            (str(a), str(b)) for a, b in self.languages
        }
        new_pairs = configured_pairs - consented_pairs
        if new_pairs:
            return (
                f"Configuration adds {len(new_pairs)} new language pair(s) "
                f"({', '.join(f'{a}\u2192{b}' for a, b in sorted(new_pairs))}) "
                "- re-consent required."
            )

        return None

    def _tokenizer_resources_location_hint(self) -> str:
        """Return a status line for the default tokenizer resources path."""
        resources_dir = self.tokenizer_resources_dir
        existence = (
            "directory exists"
            if os.path.isdir(resources_dir)
            else "directory not present yet"
        )
        return f"Default tokenizer resources directory: {resources_dir} ({existence})"

    def _install_packages(self) -> None:
        """Install configured language pairs and show local package license hints."""

        def _warn_missing_pair(from_code: str, to_code: str) -> None:
            self.pretty.write(
                "W",
                "Argos",
                f"No package found for {from_code} \u2192 {to_code}",
                color=YELLOW,
            )

        install_language_pairs(self.languages, on_missing_pair=_warn_missing_pair)
        report_argos_package_license_hints()

    def _fetch_license(self) -> str | None:
        """Download the Argos Translate license from GitHub.

        Returns the license text on success, or None on failure.
        """
        url = self._LICENSE_URL
        self.logger.info("Fetching Argos license from %s", url)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            msg = f"Failed to fetch Argos license from {url}: {exc}"
            self.logger.error(msg)
            self.pretty.write("E", "Argos License", msg, color=RED)
            return None

    def _download_tokenizer_resources(self) -> None:
        """Download tokenizer resources for all configured language codes.

        NOTE: Argos Translate already bundles stanza tokenizer models inside
        each installed language package (``<pkg_path>/stanza/``) and passes
        ``dir=<pkg_path>/stanza`` when creating ``stanza.Pipeline``.  This
        method downloads a *separate* copy into ``~/.local/share/resources``,
        which is only useful for standalone stanza usage outside of Argos.
        """
        tokenizer_resources_dir = self.tokenizer_resources_dir
        self.pretty.write(
            "I",
            "Tokenizer Resources",
            f"Tokenizer resources will be downloaded to {tokenizer_resources_dir}.",
            color=YELLOW,
        )

        try:
            import stanza  # type: ignore[reportMissingImports]
        except ImportError:
            self.pretty.write(
                "W",
                "Tokenizer Resources",
                "stanza is not installed — skipping model downloads",
                color=YELLOW,
            )
            return

        all_codes = sorted({c for pair in self.languages for c in pair})
        for code in all_codes:
            try:
                print(f"  Downloading stanza model for '{code}' ...")
                stanza.download(code, processors="tokenize,mwt", logging_level="WARNING")  # type: ignore[reportUnknownMemberType]
            except Exception as exc:
                self.pretty.write(
                    "W",
                    "Tokenizer Resources",
                    f"Failed to download stanza model for '{code}': {exc}",
                    color=YELLOW,
                )

    @staticmethod
    def _show_license_pager(text: str) -> None:
        """Display license text page-by-page."""
        show_license(text)

    def _passes_minimum_root_guard(self, target_path: str) -> bool:
        """Return False when a path resolves to a drive or filesystem root."""
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
