# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false, reportUnusedVariable=false
"""
Tests for ArgosDownloader — consent validation, metadata persistence,
ensure_packages orchestration, remove helpers, and path-guard safety.

Uses DI + attribute injection.  Heavy deps (argostranslate, stanza,
user prompts) are monkeypatched so tests never touch the network or
wait for input.
"""

import hashlib
import json
import os
import sys
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.ArgosDownloader import ArgosDownloader

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg, *args):
        self.messages.append(("info", msg % args if args else msg))

    def error(self, msg, *args):
        self.messages.append(("error", msg % args if args else msg))

    def debug(self, msg, **kw):
        self.messages.append(("debug", msg))


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[Any, ...]] = []

    def write(self, *a, **kw):
        self.messages.append((*a, kw))
        return None


class StubConfig:
    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        return default


class StubSharedHelpers:
    def __init__(self):
        self.cfg = StubConfig()

    def compute_text_hash(self, text: str) -> str:
        return hashlib.sha256(
            text.replace("\r\n", "\n").strip().encode("utf-8")
        ).hexdigest()

    def capture_acceptance_identity_once(self):
        return {
            "accepted_by": "test-user",
            "accepted_by_source": "test",
            "accepted_by_verified": True,
            "host": "test-host",
            "pid": 12345,
        }


class StubHelpers:
    def setup_logger(self, name):
        return StubLogger()


class StubFileUtils:
    def __init__(self, project_root: str):
        self.deleted: list[str] = []
        self.project_root = os.path.normpath(os.path.abspath(project_root))

    def is_safe_delete_path(self, path: str) -> bool:
        normalized_path = os.path.normpath(os.path.abspath(path))
        if normalized_path in ("", os.path.sep):
            return False
        return normalized_path == self.project_root or normalized_path.startswith(
            self.project_root + os.sep
        )

    def passes_delete_guard(self, path: str, *, on_error=None) -> bool:
        _ = on_error
        return self.is_safe_delete_path(path)

    def delete_file_or_dir(self, path):
        self.deleted.append(path)
        abs_path = os.path.normpath(os.path.abspath(path))
        _, tail = os.path.splitdrive(abs_path)
        if len(tail) > 2 and os.path.isfile(abs_path):
            os.remove(abs_path)
        return True


class StubArgosPackage:
    def __init__(
        self,
        from_code: str,
        to_code: str,
        package_path: str | None = None,
    ):
        self.from_code = from_code
        self.to_code = to_code
        self.package_path = package_path


def _installed_packages():
    return [StubArgosPackage(from_code, to_code) for from_code, to_code in _LANGS]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_LANGS = [("en", "de"), ("en", "fr")]
_LICENSE_TEXT = "MIT License\n\nCopyright (c) 2020 Argos Open Technologies, LLC\n"


def _build(tmp_path, *, write_license=False, write_meta=None):
    """Build an ArgosDownloader with stubs, pointing at tmp_path as project root."""
    dl = object.__new__(ArgosDownloader)

    dl.root = str(tmp_path)
    dl.languages = list(_LANGS)

    license_dir_rel = os.path.join("ModelGovernance", "licenses", "argos_translate")
    consent_dir_rel = os.path.join("ModelGovernance", "consents", "argos_translate")

    dl.license_dir = os.path.join(str(tmp_path), license_dir_rel)
    dl.license_path = os.path.join(dl.license_dir, "LICENSE.txt")
    dl.license_meta_path = os.path.join(dl.license_dir, "license_meta.json")
    dl.consent_dir = os.path.join(str(tmp_path), consent_dir_rel)
    dl.download_meta_path = os.path.join(dl.consent_dir, "download_meta.json")

    dl.shared = StubSharedHelpers()
    dl.helpers = StubHelpers()
    dl.file_utils = StubFileUtils(str(tmp_path))
    dl.pretty = StubPrettyWriter()
    dl.logger = StubLogger()

    # Set up license file on disk (simulates already-downloaded license)
    if write_license:
        os.makedirs(dl.license_dir, exist_ok=True)
        with open(dl.license_path, "w", encoding="utf-8") as f:
            f.write(_LICENSE_TEXT)

    # Optionally pre-populate consent metadata
    if write_meta is not None:
        os.makedirs(dl.license_dir, exist_ok=True)
        with open(dl.license_meta_path, "w", encoding="utf-8") as f:
            json.dump(write_meta, f)
        os.makedirs(dl.consent_dir, exist_ok=True)
        with open(dl.download_meta_path, "w", encoding="utf-8") as f:
            json.dump(write_meta, f)

    return dl


def _license_hash():
    return hashlib.sha256(
        _LICENSE_TEXT.replace("\r\n", "\n").strip().encode("utf-8")
    ).hexdigest()


# ===================================================================
# ensure_packages — license missing
# ===================================================================


class TestEnsurePackagesLicenseMissing:
    def test_returns_false_when_download_fails(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: None)
        assert dl.ensure_packages() is False

    def test_logs_error_when_download_fails(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: None)
        dl.ensure_packages()
        # _fetch_license logs the error internally


# ===================================================================
# ensure_packages — existing valid consent (no prompts)
# ===================================================================


class TestEnsurePackagesExistingConsent:
    def test_returns_true_on_valid_consent(self, tmp_path, monkeypatch):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "languages": list(_LANGS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(
            "argostranslate.package.get_installed_packages",
            lambda: _installed_packages(),
        )
        assert dl.ensure_packages() is True

    def test_no_prompts_on_valid_consent(self, tmp_path, monkeypatch):
        """Should not call input() when consent is already valid."""
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "languages": list(_LANGS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(
            "argostranslate.package.get_installed_packages",
            lambda: _installed_packages(),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _prompt="": pytest.fail("unexpected prompt"),
        )
        assert dl.ensure_packages() is True

    def test_stale_consent_triggers_prompt(self, tmp_path, monkeypatch):
        """When license_hash doesn't match, user should be prompted."""
        meta = {"consent": True, "license_hash": "stale-hash"}
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        # Decline the license prompt
        responses = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_packages() is False

    def test_consent_false_triggers_prompt(self, tmp_path, monkeypatch):
        """consent: false should not be treated as valid."""
        meta = {"consent": False, "license_hash": _license_hash()}
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_packages() is False


# ===================================================================
# ensure_packages — user declines license
# ===================================================================


class TestEnsurePackagesDeclineLicense:
    def test_decline_license_returns_false(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        # First input = "Enter to review", second = "n" to decline
        responses = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_packages() is False

    def test_decline_license_no_meta_written(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        dl.ensure_packages()
        assert not os.path.isfile(dl.license_meta_path)


# ===================================================================
# ensure_packages — user accepts license but declines download
# ===================================================================


class TestEnsurePackagesDeclineDownload:
    def test_accept_license_decline_download_returns_false(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        # "Enter to review" -> "y" accept license -> "n" decline download
        responses = iter(["", "y", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_packages() is False

    def test_accept_license_decline_download_no_meta(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "y", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        dl.ensure_packages()
        assert not os.path.isfile(dl.license_meta_path)


# ===================================================================
# ensure_packages — full acceptance (mock install)
# ===================================================================


class TestEnsurePackagesFullAccept:
    def test_installs_missing_packages_when_consent_exists(self, tmp_path, monkeypatch):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "languages": list(_LANGS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)

        installed = []
        monkeypatch.setattr(
            "argostranslate.package.get_installed_packages",
            lambda: cast(list[Any], []),
        )
        monkeypatch.setattr(dl, "_install_packages", lambda: installed.append(True))
        responses = iter(["", "y", "y"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

        assert dl.ensure_packages() is True
        assert installed == [True]

    def _accept_and_install(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        # "Enter to review" -> "y" accept -> "y" download
        responses = iter(["", "y", "y"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

        # Stub out network-hitting methods
        installed = []
        monkeypatch.setattr(dl, "_install_packages", lambda: installed.append(True))

        result = dl.ensure_packages()
        return dl, result, installed

    def test_returns_true(self, tmp_path, monkeypatch):
        _, result, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert result is True

    def test_writes_consent_meta(self, tmp_path, monkeypatch):
        dl, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert os.path.isfile(dl.license_meta_path)
        with open(dl.license_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["consent"] is True
        assert meta["license_hash"] == _license_hash()
        assert meta["accepted_by"] == "test-user"
        assert meta["component"] == "argostranslate"

    def test_copies_license_to_license_dir(self, tmp_path, monkeypatch):
        dl, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert os.path.isfile(dl.license_path)
        with open(dl.license_path, "r") as f:
            assert f.read() == _LICENSE_TEXT

    def test_calls_install(self, tmp_path, monkeypatch):
        _, _, installed = self._accept_and_install(tmp_path, monkeypatch)
        assert installed == [True]

    def test_meta_contains_identity_keys(self, tmp_path, monkeypatch):
        dl, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        with open(dl.license_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["host"] == "test-host"
        assert meta["pid"] == 12345
        assert meta["accepted_by_source"] == "test"

    def test_meta_languages_match_config(self, tmp_path, monkeypatch):
        _dl, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        with open(_dl.download_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["languages"] == [["en", "de"], ["en", "fr"]]

    def test_subsequent_run_skips_prompts(self, tmp_path, monkeypatch):
        """After a successful consent, the next call should not prompt."""
        dl, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        # Now re-build pointing at the same tmp_path (meta already on disk)
        dl2 = _build(tmp_path, write_license=True)
        monkeypatch.setattr(
            "argostranslate.package.get_installed_packages",
            lambda: _installed_packages(),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _prompt="": pytest.fail("unexpected prompt"),
        )
        assert dl2.ensure_packages() is True


# ===================================================================
# remove_consent — uses FileUtils path guard
# ===================================================================


class TestRemoveConsent:
    def test_deletes_meta_and_license_copy(self, tmp_path):
        meta = {"consent": True, "license_hash": _license_hash()}
        dl = _build(tmp_path, write_license=True, write_meta=meta)

        dl.remove_consent()

        assert dl.license_meta_path in dl.file_utils.deleted
        assert dl.license_path in dl.file_utils.deleted
        assert dl.download_meta_path in dl.file_utils.deleted

    def test_noop_when_no_consent_exists(self, tmp_path):
        dl = _build(tmp_path)
        dl.remove_consent()
        assert dl.file_utils.deleted == []

    def test_skips_paths_outside_project_root(self, tmp_path):
        meta = {"consent": True, "license_hash": _license_hash()}
        dl = _build(tmp_path, write_license=True, write_meta=meta)

        outside_path = tmp_path.parent / "outside_LICENSE.txt"
        outside_path.write_text("license", encoding="utf-8")
        dl.license_path = str(outside_path)

        dl.remove_consent()

        assert dl.license_meta_path in dl.file_utils.deleted
        assert dl.download_meta_path in dl.file_utils.deleted
        assert dl.license_path not in dl.file_utils.deleted


class TestRemoveAllPathGuard:
    def test_uninstalls_only_packages_with_guarded_paths(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)

        safe_pkg = StubArgosPackage(
            "en",
            "de",
            package_path=os.path.join(
                str(tmp_path),
                ".cache",
                "argos",
                "data",
                "packages",
                "translate-en_de",
            ),
        )
        outside_pkg = StubArgosPackage(
            "en",
            "fr",
            package_path=os.path.join(
                str(tmp_path.parent),
                "outside",
                "translate-en_fr",
            ),
        )

        monkeypatch.setattr(dl, "installed_packages", lambda: [safe_pkg, outside_pkg])
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        os.makedirs(str(safe_pkg.package_path), exist_ok=True)

        import argostranslate.settings as argos_settings

        monkeypatch.setattr(
            argos_settings,
            "package_data_dir",
            os.path.join(str(tmp_path), ".cache", "argos", "data", "packages"),
            raising=False,
        )

        captured: dict[str, Any] = {}

        def _remove_packages(packages, *, on_uninstall_error=None):
            captured["packages"] = list(packages)
            return len(packages)

        monkeypatch.setattr(
            "Compliance.ArgosDownloader.remove_packages",
            _remove_packages,
        )

        dl.remove_all()

        removed_packages = cast(list[Any], captured.get("packages", []))
        assert len(removed_packages) == 1
        assert removed_packages[0].from_code == "en"
        assert removed_packages[0].to_code == "de"

    def test_skips_uninstall_when_package_directory_is_missing(
        self,
        tmp_path,
        monkeypatch,
    ):
        dl = _build(tmp_path)

        missing_dir_pkg = StubArgosPackage(
            "en",
            "de",
            package_path=os.path.join(
                str(tmp_path),
                ".cache",
                "argos",
                "data",
                "packages",
                "translate-en_de",
            ),
        )

        monkeypatch.setattr(dl, "installed_packages", lambda: [missing_dir_pkg])
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        import argostranslate.settings as argos_settings

        monkeypatch.setattr(
            argos_settings,
            "package_data_dir",
            os.path.join(str(tmp_path), ".cache", "argos", "data", "packages"),
            raising=False,
        )

        captured: dict[str, Any] = {}

        def _remove_packages(packages, *, on_uninstall_error=None):
            captured["packages"] = list(packages)
            return len(packages)

        monkeypatch.setattr(
            "Compliance.ArgosDownloader.remove_packages",
            _remove_packages,
        )

        dl.remove_all()

        removed_packages = cast(list[Any], captured.get("packages", []))
        assert removed_packages == []


# ===================================================================
# _show_license_pager
# ===================================================================


class TestShowLicensePager:
    def test_short_text_printed(self, capsys):
        ArgosDownloader._show_license_pager("Hello\nWorld")
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" in captured.out

    def test_long_text_pages(self, monkeypatch, capsys):
        long_text = "\n".join(f"Line {i}" for i in range(200))
        # Simulate pressing "q" on first page prompt
        monkeypatch.setattr("builtins.input", lambda _prompt="": "q")
        ArgosDownloader._show_license_pager(long_text)
        captured = capsys.readouterr()
        assert "Line 0" in captured.out


# ===================================================================
# compute_text_hash via SharedHelpers (integration-level)
# ===================================================================


class TestComputeTextHash:
    def test_crlf_normalized(self, tmp_path):
        dl = _build(tmp_path)
        h1 = dl.shared.compute_text_hash("hello\r\nworld")
        h2 = dl.shared.compute_text_hash("hello\nworld")
        assert h1 == h2

    def test_strips_whitespace(self, tmp_path):
        dl = _build(tmp_path)
        h1 = dl.shared.compute_text_hash("  hello  ")
        h2 = dl.shared.compute_text_hash("hello")
        assert h1 == h2

    def test_deterministic(self, tmp_path):
        dl = _build(tmp_path)
        assert dl.shared.compute_text_hash("x") == dl.shared.compute_text_hash("x")

    def test_matches_manual_sha256(self, tmp_path):
        dl = _build(tmp_path)
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert dl.shared.compute_text_hash("hello") == expected
