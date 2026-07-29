"""
Compliance module

Manages per-model LICENSE.txt + license_meta.json storage under ModelGovernance/licenses/.
Key operations:
- _update_licenses(): fetch licenses from sources with consent tracking
- verify(): ensure licenses exist and hashes match

Key features:
- Stores acceptance metadata in the format:
  {
    "model_id": "..",
    "license_url": "..",
    "license_hash": "..",
    "tls_cert_fingerprint": "..",
    "accepted_by": "..",
    "accepted_by_source": "..",
    "accepted_by_verified": false,
    "accepted_at": "..",
    "host": "..",
    "pid": 1234,
    "config_hash": "..",
    "downloaded_at": "..",
    "source": "bundled" | "fetched",
    "consent": true,
    "config": {...}
  }
- Attempts to capture TLS certificate fingerprint for verification on live fetch
- Offline-first: supports bundled LICENSE.txt files
- Acceptance identity is determined once per run and reused for all acceptances
"""

import hashlib
import json
import logging
import os
import shutil
import ssl
from datetime import datetime, timezone
from typing import Any, Optional, cast
from urllib.parse import urlparse

import requests

import Configuration.Config_Banned as Config_Banned
import Configuration.Config_Internet_Env as Config_Internet_Env
import Configuration.Config_Models as Config_Models
import Configuration.Config_WebSearch as Config_WebSearch
from Commons.Exceptions import (ComplianceViolationError,
                                InternetConnectionDisabledError)
from Commons.SingletonMixin import SingletonMixin
from Compliance.ArgosDownloader import ArgosDownloader
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import BRIGHT_BLUE, CYAN, GREEN, ORANGE, RED, RESET, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils

_CONSENT_DISCLAIMER = (
    "This record indicates technical acknowledgement of license terms. "
    "It does not constitute a legal contract or replace formal legal review."
)
from Gui.LicensePager import show_license
from Helpers.Helpers import Helpers


class Compliance(SingletonMixin):

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.globalsInstance: Globals = Globals()
        self.helpers: Helpers = helpers or Helpers()
        self.sharedHelpers: SharedHelpers = SharedHelpers()
        self.fileUtils: FileUtils = FileUtils()
        self.logger: logging.Logger = self.helpers.setup_logger("Compliance")
        self.pretty: PrettyWriter = pretty or PrettyWriter(always_on=True)
        self.cfg: Config = cfg or Config()
        # Flatten _MODELS to {"impl.role": config}, filtered by USED_BY
        raw_models: dict[str, Any] = self.cfg.get_dict("_MODELS", {})
        self.models: dict[str, dict[str, Any]] = {}
        self.friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        for _impl, _roles in raw_models.items():
            if isinstance(_roles, dict):
                for _role, _config in cast(dict[str, Any], _roles).items():
                    if isinstance(_config, dict):
                        used_by: Any = cast(dict[str, Any], _config).get("USED_BY")
                        if (
                            isinstance(used_by, list)
                            and self.friendly_name not in used_by
                        ):
                            continue
                        self.models[f"{_impl}.{_role}"] = cast(dict[str, Any], _config)
        self.license_download: str | None = os.environ.get("LICENSE_DOWNLOAD", "0")

        self.base_dir: str = "ModelGovernance/licenses"
        os.makedirs(self.base_dir, exist_ok=True)

        # cached acceptance identity (created/queried once per run)
        self.acceptance_identity: dict[str, Any] | None = None

    def _get_acceptance_identity(self) -> dict[str, Any]:
        """Return cached acceptance identity, capturing it once per run."""
        if self.acceptance_identity is None:
            self.acceptance_identity = (
                self.sharedHelpers.capture_acceptance_identity_once()
            )
        return self.acceptance_identity

    # Helper methods for IO, license fetching, and pager display
    # ---------------------------
    def _load_meta(self, path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result

    def _save_meta(self, path: str, meta: dict[str, Any]) -> None:
        # Ensure every consent record carries the legal/operational disclaimer.
        if meta.get("consent") is True and "disclaimer" not in meta:
            meta["disclaimer"] = _CONSENT_DISCLAIMER
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _fetch_license(self, url: str, label: str, lic_path: str) -> str:
        """
        Live fetching license. On error program will abort with ComplianceViolationError.
        If LICENSE_DOWNLOAD is set to "0", user will be prompted to allow online fetch before proceeding.
        """
        fetch_url = url

        if self.license_download == "0":
            msg = f">>>> Internet connection is set to 'None'"
            self.pretty.write("I", "License", msg, color=YELLOW)
            self.logger.warning(msg)
            msg = f">>>> Do you allow to online fetch license for  [{label}] from URL: {url}"
            self.pretty.write("I", "License", msg, color=ORANGE)
            self.logger.warning(msg)

            prompt = f"{YELLOW}\n\n>>>>  [y/N] {RESET}"
            ans = input(prompt).strip().lower()
            msg = f"Aborting. Fetch license and install manually into ModelGovernancelicenses/{lic_path}"
            if ans != "y":
                self.pretty.write(
                    "E",
                    "License",
                    f"{msg}",
                    color=RED,
                )
                self.logger.error(msg)
                raise InternetConnectionDisabledError()

        msg = f"Fetching license online from: {fetch_url}"
        self.logger.warning(msg)
        self.pretty.write("I", "License", msg)
        try:
            resp = requests.get(fetch_url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            error_msg = f"Failed to fetch license for [{label}] from URL: {fetch_url}. Error: {str(e)}"
            self.logger.error(error_msg)
            self.pretty.write("E", "License", error_msg, color=RED)
            raise ComplianceViolationError(msg)

    def _show_pager(self, text: str) -> None:
        show_license(text)

    def _compute_height(self) -> int:
        try:
            h = shutil.get_terminal_size().lines - 2
            return h if h >= 5 else 15
        except Exception:
            return 15

    def _license_subdir_for(self, chk: dict[str, Any], section: str) -> str:
        source: Any = chk.get("SOURCE")
        model_or_provider: Any = chk.get("MODEL") or chk.get("PROVIDER") or {}

        if isinstance(model_or_provider, str):
            lastItem = model_or_provider.split("/")[-1]
            # Include the section key so that two roles sharing the same
            # model (e.g. _LLM and _LLM_CHK both using mistral:7b) each get
            # their own license_meta.json and don't overwrite each other.
            safe_section = self.helpers.sanitize_path_component(section)
            safe_model = self.helpers.sanitize_path_component(lastItem)
            return f"{safe_model}_{safe_section}"

        if isinstance(source, str):
            parsed = urlparse(source)
            tail = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
            if tail:
                safe_section = self.helpers.sanitize_path_component(section)
                return f"{tail}_{safe_section}"

        return section

    # ---------------------------
    # TLS fingerprint
    # ---------------------------
    def _get_tls_fingerprint(self, url: str) -> str:
        try:
            p = urlparse(url)
            host = p.hostname
            port = p.port or (443 if p.scheme in ("https", "") else None)
            if not host or not port:
                return ""
            pem = ssl.get_server_certificate((host, port))
            if not pem:
                return ""
            pem_bytes = pem.encode("utf-8")
            fp = hashlib.sha256(pem_bytes).hexdigest()
            return fp
        except Exception as e:
            self.logger.debug(f"TLS fingerprint fetch failed for {url}: {e}")
            return ""

    # ---------------------------
    # Canonicalize & hash helpers
    # ---------------------------
    def _canonicalize_text(self, text: str) -> str:
        return text.replace("\r\n", "\n").strip()

    def _compute_text_hash(self, text: str) -> str:
        b = self._canonicalize_text(text).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    # ---------------------------
    # Main flows
    # ---------------------------
    def _update_licenses(self) -> None:
        """Query the acceptance identity once and perform license acceptance for all models."""
        # capture identity once for the whole run
        identity: dict[str, object] = self._get_acceptance_identity()

        for section in self.models:
            self._process_one(self.models, section, identity)
        self.pretty.write(
            "I", "License", "All licenses updated under ModelGovernance/licenses/"
        )

    def _process_one(
        self,
        chk: dict[str, Any],
        section: str,
        identity: Optional[dict[str, object]] = None,
    ) -> None:
        """Process a single model section.

        When *identity* is provided, ``accepted_by`` is taken from it
        instead of prompting the user interactively again.
        """
        if identity is None:
            identity = self._get_acceptance_identity()

        chk = self.models[section]

        url: str = chk.get("LICENSE_URL", "")  # e.g. "https://…/LICENSE"
        if not url:
            msg = f"{section}: Key LICENSE_URL not present or no value"
            self.logger.error(msg)
            self.pretty.write("E", "License", msg, color=RED)
            raise ComplianceViolationError(msg)
        label: str = chk.get("FRIENDLY_NAME", section)
        subdir: str = self._license_subdir_for(chk, section)
        lic_dir: str = os.path.join(self.base_dir, subdir)
        lic_path: str = os.path.join(lic_dir, "LICENSE.txt")
        meta_path: str = os.path.join(lic_dir, "license_meta.json")
        os.makedirs(lic_dir, exist_ok=True)

        now: str = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"

        # 1) OFFLINE-FIRST: bundled license exists but no meta.json → prompt once
        if os.path.isfile(lic_path) and not os.path.isfile(meta_path):
            with open(lic_path, "r", encoding="utf-8") as f:
                bundled_text = f.read()
            bundled_hash = self._compute_text_hash(bundled_text)
            chk_str = json.dumps(chk, sort_keys=True)
            config_hash = self.fileUtils.compute_hash(chk_str)
            self.pretty.write(
                "W", "License", f"{section} Not accepted license detected"
            )
            print(f"{BRIGHT_BLUE}\n{'=' * 70}")
            print(f"  Model License Consent")
            print(f"  Section: {section}  /  {label}")
            print(f"{'=' * 70}{RESET}\n")
            input("Press Enter to review the shipped license ...")
            self._show_pager(bundled_text)
            print(f"\n{ORANGE}>>>> Do you accept this license?{RESET}")
            ans = input("Accept? [y/N] ").strip().lower()
            if ans != "y":
                msg = f"Aborting. License for {section} not accepted"
                self.logger.error(msg)
                self.pretty.write("E", "License", msg, color=RED)
                raise ComplianceViolationError(msg)

            meta: dict[str, Any] = {
                "model_id": chk.get("MODEL") or chk.get("PROVIDER") or section,
                "license_url": url,
                "license_hash": bundled_hash,
                "tls_cert_fingerprint": "",  # bundled license
                "accepted_by": identity.get("accepted_by"),
                "accepted_by_source": identity.get("accepted_by_source"),
                "accepted_by_verified": identity.get("accepted_by_verified", False),
                "accepted_at": now,
                "host": identity.get("host"),
                "pid": identity.get("pid"),
                "config_hash": config_hash,
                "downloaded_at": now,
                "source": "bundled",
                "consent": True,
                "config": chk,
            }
            self._save_meta(meta_path, meta)
            msg = f"{section}: Offline bundled license accepted"
            self.logger.info(msg)
            self.pretty.write("O", "License", msg)
            self.pretty.write(
                "O",
                "License",
                f"License consent recorded in {meta_path}",
                color=GREEN,
            )
            return

        # 2) if both exist, hashes match, and consent was given → nothing to do
        if os.path.isfile(lic_path) and os.path.isfile(meta_path):
            with open(lic_path, "r", encoding="utf-8") as f:
                disk_text = f.read()
            disk_hash = self._compute_text_hash(disk_text)
            chk_str = json.dumps(chk, sort_keys=True)
            config_hash = self.fileUtils.compute_hash(chk_str)
            meta = self._load_meta(meta_path)

            if (
                disk_hash == meta.get("license_hash")
                and config_hash == meta.get("config_hash")
                and meta.get("consent") is True
            ):
                # In online mode, also verify the remote license hasn't changed
                if self.license_download == "1":
                    try:
                        remote_text = self._fetch_license(url, label, lic_path)
                        remote_hash = self._compute_text_hash(remote_text)
                        if remote_hash != meta.get("license_hash"):
                            msg = f"{section}: Remote license changed; re-acceptance needed"
                            self.logger.warning(msg)
                            self.pretty.write("W", "License", msg, color=YELLOW)
                            pass  # fall through to step 3
                        else:
                            return
                    except Exception:
                        pass  # fall through to step 3
                else:
                    return

        # 3) otherwise → live-fetch

        msg = f"{label}: fetching LICENSE from {url}"
        self.logger.info(msg)
        self.pretty.write("I", "License", msg)
        new_text = self._fetch_license(url, label, lic_path)
        new_hash = self._compute_text_hash(new_text)
        chk_str = json.dumps(chk, sort_keys=True)
        config_hash = self.fileUtils.compute_hash(chk_str)

        old_text = None
        if os.path.isfile(lic_path):
            with open(lic_path, "r", encoding="utf-8") as f:
                old_text = f.read()

        if old_text is None:
            msg = f"First time online fetch: [{section}] [{label}]"
            self.logger.info(msg)
            self.pretty.write("I", "License", msg, color=YELLOW)
            print(f"{BRIGHT_BLUE}\n{'=' * 70}")
            print(f"  Model License Consent  (first online fetch)")
            print(f"  Section: {section}  /  {label}")
            print(f"{'=' * 70}{RESET}\n")
            input("Press Enter to review the license ...")
            self._show_pager(new_text)
        else:
            prev_meta = self._load_meta(meta_path)
            textHashIdentical = new_hash == prev_meta.get("license_hash")
            config_hashIdentical = config_hash == prev_meta.get("config_hash")
            print(f"{BRIGHT_BLUE}\n{'=' * 70}")
            print(f"  Model License Consent  (re-acceptance)")
            print(f"  Section: {section}  /  {label}")
            print(
                f"  License text identical: {textHashIdentical}  |  Config identical: {config_hashIdentical}"
            )
            print(f"{'=' * 70}{RESET}\n")
            input("Press Enter to review the license ...")
            self._show_pager(new_text)

        print(
            f"\n{ORANGE}>>>> Do you accept this license for [{section}] [{label}]?{RESET}"
        )
        ans = input("Accept? [y/N] ").strip().lower()

        if ans != "y":
            msg = "Aborting. License for {section} not accepted"
            self.logger.error(msg)
            self.pretty.write(
                "E",
                "License",
                f"Aborting. License for {section} not accepted",
                color=RED,
            )
            raise ComplianceViolationError(msg)

        # persist fetched license + new meta.json
        # Canonicalise before writing so the on-disk text always has \n-only line
        # endings.  Writing with newline="" prevents Python text mode from adding a
        # second \r on Windows (raw \r\n → \r\r\n → doubled blank lines on re-read).
        canonical_text = self._canonicalize_text(new_text)
        with open(lic_path, "w", encoding="utf-8", newline="") as f:
            f.write(canonical_text)

        # compute TLS fingerprint (best-effort)
        tls_fp = self._get_tls_fingerprint(url)

        meta: dict[str, Any] = {
            "model_id": chk.get("MODEL") or chk.get("PROVIDER") or section,
            "license_url": url,
            "license_hash": new_hash,
            "tls_cert_fingerprint": tls_fp,
            "accepted_by": identity.get("accepted_by"),
            "accepted_by_source": identity.get("accepted_by_source"),
            "accepted_by_verified": identity.get("accepted_by_verified", False),
            "accepted_at": now,
            "host": identity.get("host"),
            "pid": identity.get("pid"),
            "config_hash": config_hash,
            "downloaded_at": now,
            "source": "fetched",
            "consent": True,
            "config": chk,
        }
        self._save_meta(meta_path, meta)
        msg = f"{section}: Online license accepted and stored"
        self.logger.info(msg)
        self.pretty.write("O", "License", msg)
        self.pretty.write(
            "O",
            "License",
            f"License consent recorded in {meta_path}",
            color=GREEN,
        )

    # ---------------------------
    # Config hash checks
    # ---------------------------
    def _check_models_config_hash(self):
        # These slots are hardcoded — all must be present in _CRITICAL_CONFIG_HASHES.
        # Deleting any entry from the dict in Config_Global.py will be detected as a mismatch.
        modules = [
            (Config_Models, "Config_Models"),
            (Config_Banned, "Config_Banned"),
            (Config_WebSearch, "Config_WebSearch"),
            (Config_Internet_Env, "Config_Internet_Env"),
        ]
        hashes: dict[str, Any] = self.cfg.get_dict("_CRITICAL_CONFIG_HASHES", {})
        mismatches: list[tuple[str, str | None, str, str | None, str]] = []

        for module, slot_key in modules:
            module_hash = self.fileUtils.hash_module(module)
            ref_hash: str | None = str(hashes[slot_key]) if slot_key in hashes else None

            module_name = getattr(module, "__name__", str(module))
            module_path = getattr(module, "__file__", None)

            if module_hash != ref_hash:
                mismatches.append(
                    (module_name, module_path, slot_key, ref_hash, module_hash)
                )
            else:
                self.pretty.write(
                    "O",
                    module_name,
                    f"No change detected in config file {module_path}",
                    color=GREEN,
                )

        if mismatches:
            for module_name, module_path, slot_key, ref_hash, module_hash in mismatches:
                msg = f"Detected modification of {module_name} ({module_path})"
                self.logger.warning(msg)
                self.pretty.write("W", module_name, msg)

                msg = (
                    f'Reference hash _CRITICAL_CONFIG_HASHES["{slot_key}"]: {ref_hash} '
                )
                self.logger.error(msg)
                self.pretty.write("E", module_name, msg)
                msg = f"Expected {module_name} hash: {module_hash}"
                self.logger.error(msg)
                self.pretty.write("W", module_name, msg, color=ORANGE)

                msg = (
                    f"Update Configuration/Config_Global.py "
                    f'_CRITICAL_CONFIG_HASHES["{slot_key}"] to match expected hash'
                )
                self.logger.error(msg)
                self.pretty.write("E", module_name, msg, color=ORANGE)

            changed = ", ".join(m[0] for m in mismatches)
            msg = f"The goal of this is that you consent your changes in: {changed}"
            self.logger.error(msg)
            self.pretty.write("I", "Compliance", msg)
            self.pretty.write(
                "I",
                "Compliance",
                "Run:  python .\\src\\Scripts\\RecalcConfigHashes.py  to update the hash(es) automatically.",
                color=CYAN,
            )
            raise ComplianceViolationError(msg)

        self.pretty.write(
            "N",
            "",
            "",
        )

    # ---------------------------
    # Argos Translate license consent + download
    # ---------------------------
    def _verify_argos_consent(self) -> None:
        """Check Argos Translate consent and download packages if needed.

        When ARGOS_STANZA_DOWNLOAD is "1", prompt the user to accept the
        license and download packages (like HFDownloader).  If the user
        declines, warn and continue — no exception is raised.

        When ARGOS_STANZA_DOWNLOAD is "0" the check is skipped.
        """
        project_root: str = self.cfg.get_str("_ABSOLUTE_PATH")
        languages = self.cfg.get_list("_ARGOS_DEFINITIONS.ARGOS_LANGUAGES")
        downloader = ArgosDownloader(project_root, languages)

        stanza_download: str = os.environ.get("ARGOS_STANZA_DOWNLOAD", "0").strip()
        if stanza_download != "1":
            # Even when downloads are disabled, show consent status if
            # packages were pre-installed via scripts/ArgosTranslatePackages.py.
            downloader.report_consent_status()
            return

        installed = downloader.ensure_packages()

        # Refresh installed languages whenever new packages were just installed
        # so translation is available immediately without a restart.
        refresh_langs = getattr(self.sharedHelpers, "refresh_installed_languages", None)
        if installed and callable(refresh_langs):
            refresh_langs()

    # ---------------------------
    # Verify flow
    # ---------------------------
    def verify(self):
        """
        Ensures all LICENSE.txt + metadata exist, match their recorded hashes,
        and that user consent is present. If anything is missing,
        invoke _update_licenses() once (which will prompt the user) and then exit.
        """
        need_update: bool = False
        self._check_models_config_hash()

        for section in self.models:
            chk = self.models[section]
            label = chk.get("MODEL", section)
            subdir = self._license_subdir_for(chk, section)

            lic_path = os.path.join(self.base_dir, subdir, "LICENSE.txt")
            meta_path = os.path.join(self.base_dir, subdir, "license_meta.json")

            if not os.path.isfile(lic_path) or not os.path.isfile(meta_path):
                msg = f"{section}: missing license or metadata"
                self.logger.warning(msg)
                self.pretty.write("W", "License", msg, color=YELLOW)
                need_update = True
                break

            meta = self._load_meta(meta_path)

            if not meta.get("consent", False):
                msg = f"{section}: user consent not recorded"
                self.logger.warning(msg)
                self.pretty.write("W", "License", msg, color=YELLOW)
                need_update = True
                break

            with open(lic_path, "r", encoding="utf-8") as f:
                content = f.read()
            actual = self._compute_text_hash(content)
            recorded = meta.get("license_hash", "").lower()

            if actual != recorded:
                msg = f"{section}: License text hash mismatch (recorded {recorded})"
                self.logger.warning(msg)
                self.pretty.write("W", "License", msg, color=YELLOW)
                need_update = True
                break

            chk_str = json.dumps(chk, sort_keys=True)
            config_hash = self.fileUtils.compute_hash(chk_str)
            if config_hash != meta.get("config_hash"):
                msg = (
                    f"{section}: Hash over config (Config_models.py) mismatch recorded"
                )
                self.logger.warning(msg)
                self.pretty.write("W", "License", msg, color=YELLOW)
                need_update = True
                break

            # Optional: re-verify remote license and TLS fingerprint if configured
            if self.license_download == "1":
                license_url: str = meta.get("license_url", "")
                try:
                    current_text = self._fetch_license(license_url, label, lic_path)
                    current_hash = self._compute_text_hash(current_text)
                    if current_hash != meta.get("license_hash"):
                        msg = f"{section}: Remote license text changed since acceptance; re-accept required"
                        self.logger.warning(msg)
                        self.pretty.write("W", "License", msg, color=YELLOW)
                        need_update = True
                        break
                    current_tls_fp = self._get_tls_fingerprint(license_url)
                    if meta.get("tls_cert_fingerprint") and current_tls_fp != meta.get(
                        "tls_cert_fingerprint"
                    ):
                        msg = f"{section}: TLS certificate fingerprint changed since acceptance; re-accept required"
                        self.logger.warning(msg)
                        self.pretty.write("W", "License", msg, color=YELLOW)
                        need_update = True
                        break
                except Exception:
                    need_update = True
                    break

        if need_update:
            msg = "License consent required"
            self.logger.warning(msg)
            self.pretty.write("W", "License", msg, color=YELLOW)
            self._update_licenses()

        msg = "All LICENSE.txt files for active models in Config_Models.py (see _MODELS) are consented"
        self.logger.info(msg)
        self.pretty.write("O", "License", msg, color=GREEN)

        # Argos Translate package license consent
        self._verify_argos_consent()
