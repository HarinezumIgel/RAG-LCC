"""
ArgosDownloader — Argos Translate license consent and package installer.

Follows the same interactive prompt pattern as HFDownloader:
  1. Show the Argos MIT license and ask the user to accept.
  2. Ask confirmation to download language packages + stanza models.
  3. If the user declines at either step, warn to set
     ARGOS_STANZA_DOWNLOAD="0" and return (no crash).

The license is downloaded from GitHub and stored under
ModelGovernance/licenses/argos_translate/LICENSE.txt with license
metadata in license_meta.json.  Download consent is recorded in
ModelGovernance/consents/argos_translate/download_meta.json.
The check is a no-op on subsequent runs while the license hash matches.
"""

import json
import logging
import ntpath
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, cast

import requests

from Gui.LicensePager import show_license
from Helpers.Helpers import Helpers

Helpers().prepare_argos_runtime_dirs()

import argostranslate.package  # type: ignore[reportMissingTypeStubs]
import argostranslate.translate  # type: ignore[reportMissingTypeStubs]

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

    def __init__(
        self,
        project_root: str,
        languages: List[Tuple[str, str]],
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

        self.shared: SharedHelpers = SharedHelpers()
        self.helpers: Helpers = Helpers()
        self.file_utils: FileUtils = FileUtils()
        self.pretty: PrettyWriter = PrettyWriter(always_on=True)
        self.logger: logging.Logger = self.helpers.setup_logger("Compliance")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _check_existing_consent(self) -> bool:
        """Return True and emit a green message if consent JSON is valid."""
        if (
            not os.path.isfile(self.license_path)
            or not os.path.isfile(self.license_meta_path)
            or not os.path.isfile(self.download_meta_path)
        ):
            return False
        with open(self.license_path, "r", encoding="utf-8") as fh:
            license_hash: str = self.shared.compute_text_hash(fh.read())
        with open(self.license_meta_path, "r", encoding="utf-8") as fh:
            meta: Dict[str, Any] = json.load(fh)
        if not (
            meta.get("consent") is True and meta.get("license_hash") == license_hash
        ):
            return False

        # --- detect config drift: new language pairs added since consent ---
        try:
            with open(self.download_meta_path, "r", encoding="utf-8") as fh:
                dl_meta: Dict[str, Any] = json.load(fh)
            raw_languages = cast(list[Any], dl_meta.get("languages") or [])
            consented_pairs: set[tuple[str, str]] = set()
            for pair in raw_languages:
                if isinstance(pair, (list, tuple)):
                    pair_tuple = cast(tuple[Any, Any], pair)
                    if len(pair_tuple) >= 2:
                        consented_pairs.add((str(pair_tuple[0]), str(pair_tuple[1])))
        except Exception:
            consented_pairs = set()
        configured_pairs: set[tuple[str, str]] = {
            (str(a), str(b)) for a, b in self.languages
        }
        new_pairs: set[tuple[str, str]] = configured_pairs - consented_pairs
        if new_pairs:
            self.pretty.write(
                "W",
                "Argos License",
                f"Configuration adds {len(new_pairs)} new language pair(s) "
                f"({', '.join(f'{a}\u2192{b}' for a, b in sorted(new_pairs))}) "
                f"\u2014 re-consent required.",
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
            "Stanza",
            f"Stanza models directory: {self.stanza_models_dir} "
            "(set STANZA_RESOURCES_DIR to override this location)",
        )
        return True

    def report_consent_status(self) -> None:
        """Emit the green consent-valid message if consent JSON exists and is valid.

        Used when ARGOS_STANZA_DOWNLOAD is "0" but packages were
        pre-installed via scripts/ArgosTranslatePackages.py.
        """
        self._check_existing_consent()

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
                installed = argostranslate.package.get_installed_packages()
                installed_pairs: set[tuple[str, str]] = {
                    (
                        str(getattr(pkg, "from_code", "")),
                        str(getattr(pkg, "to_code", "")),
                    )
                    for pkg in installed
                }
                configured_pairs: set[tuple[str, str]] = {
                    (str(a), str(b)) for a, b in self.languages
                }
                missing_pairs = configured_pairs - installed_pairs
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
        print("  Argos Translate License Consent")
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
                f"  Argos package  {from_code} \u2192 {to_code}  (includes bundled stanza tokenizer)"
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
            'Download declined. Set ARGOS_STANZA_DOWNLOAD="0" to suppress this prompt in the future.',
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
        installed = argostranslate.package.get_installed_packages()
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
            argostranslate.package.uninstall(pkg)
        self.pretty.write(
            "O", "Argos", f"Removed {len(installed)} package(s).", color=GREEN
        )

    @property
    def stanza_models_dir(self) -> str:
        """Resolved absolute path of the stanza_resources directory."""
        default_dir = os.path.join(
            os.path.expanduser("~"), ".local", "share", "stanza_resources"
        )
        resources_dir = os.environ.get("STANZA_RESOURCES_DIR", default_dir)
        raw_dir = os.path.expanduser(resources_dir)
        if ntpath.isabs(raw_dir) or os.path.isabs(raw_dir):
            return os.path.normpath(raw_dir)
        return os.path.normpath(os.path.abspath(raw_dir))

    def remove_stanza_models(self) -> None:
        """Remove the stanza_resources directory (outside project root)."""
        abs_path = self.stanza_models_dir

        # Minimal safety: block drive/filesystem roots and absurdly short paths
        # (e.g. 'C:\\' or 'C:' on Windows, '/' on POSIX). Uses len(tail) <= 2 for
        # parity with FileUtils.delete_file_or_dir. Project-root containment is
        # intentionally NOT applied — the stanza dir lives outside the project.
        _, win_tail = ntpath.splitdrive(abs_path)
        _, posix_tail = os.path.splitdrive(abs_path)
        if len(win_tail) <= 2 or len(posix_tail) <= 2:
            self.pretty.write(
                "E",
                "Path Guard",
                f"Refusing to delete root or drive path '{abs_path}'.",
                color=RED,
            )
            return

        if os.path.isdir(abs_path):
            self.pretty.write(
                "W",
                "Stanza",
                f"About to permanently delete directory: {abs_path}",
                color=YELLOW,
            )
            if input("Delete this directory? [y/N] ").strip().lower() != "y":
                self.pretty.write("I", "Stanza", "Removal cancelled.")
                return
            shutil.rmtree(abs_path)
            self.pretty.write("O", "Stanza", f"Removed {abs_path}", color=GREEN)
        else:
            self.pretty.write(
                "I", "Stanza", f"Directory not found — skipping ({abs_path})"
            )

    def remove_consent(self) -> None:
        """Remove consent metadata via FileUtils path guard."""
        if os.path.isfile(self.license_meta_path):
            self.file_utils.delete_file_or_dir(self.license_meta_path)
        if os.path.isfile(self.license_path):
            self.file_utils.delete_file_or_dir(self.license_path)
        if os.path.isfile(self.download_meta_path):
            self.file_utils.delete_file_or_dir(self.download_meta_path)

    def show_status(self) -> None:
        """Print currently installed Argos languages and packages."""
        langs = argostranslate.translate.get_installed_languages()
        print(f"\nInstalled languages ({len(langs)}):")
        for lang in langs:
            print(f"  {lang.name} ({lang.code})")

        installed = argostranslate.package.get_installed_packages()
        print(f"\nInstalled packages ({len(installed)}):")
        for pkg in installed:
            print(f"  {pkg.from_code} \u2192 {pkg.to_code}")
        self.pretty.write(
            "I",
            "Stanza",
            f"Stanza models directory: {self.stanza_models_dir} "
            "(set STANZA_RESOURCES_DIR to override this location)",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _install_packages(self) -> None:
        """Download and install configured language pairs from the Argos index."""
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()

        for from_code, to_code in self.languages:
            try:
                pkg = next(
                    p
                    for p in available
                    if p.from_code == from_code and p.to_code == to_code
                )
                print(f"  Installing {from_code} \u2192 {to_code} ...")
                argostranslate.package.install_from_path(pkg.download())
            except StopIteration:
                self.pretty.write(
                    "W",
                    "Argos",
                    f"No package found for {from_code} \u2192 {to_code}",
                    color=YELLOW,
                )

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

    def _download_stanza_models(self) -> None:
        """Download stanza tokenizer models for all configured language codes.

        NOTE: Argos Translate already bundles stanza tokenizer models inside
        each installed language package (``<pkg_path>/stanza/``) and passes
        ``dir=<pkg_path>/stanza`` when creating ``stanza.Pipeline``.  This
        method downloads a *separate* copy into ``~/.local/share/stanza_resources``
        (or ``STANZA_RESOURCES_DIR`` if set), which is only useful for
        standalone stanza usage outside of Argos.
        """
        stanza_dir = self.stanza_models_dir
        self.pretty.write(
            "I",
            "Stanza",
            f"Stanza models will be downloaded to {stanza_dir} "
            "(set STANZA_RESOURCES_DIR to override this location).",
            color=YELLOW,
        )

        try:
            import stanza  # type: ignore[reportMissingImports]
        except ImportError:
            self.pretty.write(
                "W",
                "Stanza",
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
                    "Stanza",
                    f"Failed to download stanza model for '{code}': {exc}",
                    color=YELLOW,
                )

    @staticmethod
    def _show_license_pager(text: str) -> None:
        """Display license text page-by-page."""
        show_license(text)
