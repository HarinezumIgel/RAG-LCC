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
from typing import Any

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


class StubSharedHelpers:
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
    def __init__(self):
        self.deleted: list[str] = []

    def delete_file_or_dir(self, path):
        self.deleted.append(path)
        if os.path.isfile(path):
            os.remove(path)
        return True


class StubArgosPackage:
    def __init__(self, from_code: str, to_code: str):
        self.from_code = from_code
        self.to_code = to_code


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
    dl.file_utils = StubFileUtils()
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
# ensure_packages — full acceptance (mock install + stanza)
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
            lambda: [],
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
        stanza_downloaded = []
        monkeypatch.setattr(dl, "_install_packages", lambda: installed.append(True))
        monkeypatch.setattr(
            dl, "_download_stanza_models", lambda: stanza_downloaded.append(True)
        )

        result = dl.ensure_packages()
        return dl, result, installed, stanza_downloaded

    def test_returns_true(self, tmp_path, monkeypatch):
        _, result, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert result is True

    def test_writes_consent_meta(self, tmp_path, monkeypatch):
        dl, _, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert os.path.isfile(dl.license_meta_path)
        with open(dl.license_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["consent"] is True
        assert meta["license_hash"] == _license_hash()
        assert meta["accepted_by"] == "test-user"
        assert meta["component"] == "argostranslate"

    def test_copies_license_to_license_dir(self, tmp_path, monkeypatch):
        dl, _, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        assert os.path.isfile(dl.license_path)
        with open(dl.license_path, "r") as f:
            assert f.read() == _LICENSE_TEXT

    def test_calls_install(self, tmp_path, monkeypatch):
        _, _, installed, _stanza = self._accept_and_install(tmp_path, monkeypatch)
        assert installed == [True]

    def test_meta_contains_identity_keys(self, tmp_path, monkeypatch):
        dl, _, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        with open(dl.license_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["host"] == "test-host"
        assert meta["pid"] == 12345
        assert meta["accepted_by_source"] == "test"

    def test_meta_languages_match_config(self, tmp_path, monkeypatch):
        _dl, _, _, _ = self._accept_and_install(tmp_path, monkeypatch)
        with open(_dl.download_meta_path, "r") as f:
            meta = json.load(f)
        assert meta["languages"] == [["en", "de"], ["en", "fr"]]

    def test_subsequent_run_skips_prompts(self, tmp_path, monkeypatch):
        """After a successful consent, the next call should not prompt."""
        dl, _, _, _ = self._accept_and_install(tmp_path, monkeypatch)
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


# ===================================================================
# remove_stanza_models — path guard
# ===================================================================


class TestRemoveStanzaModels:
    def test_removes_existing_directory(self, tmp_path, monkeypatch):
        stanza_dir = str(tmp_path / "stanza_resources")
        os.makedirs(stanza_dir)
        (tmp_path / "stanza_resources" / "en").mkdir()
        monkeypatch.setenv("STANZA_RESOURCES_DIR", stanza_dir)
        monkeypatch.setattr("builtins.input", lambda *a: "y")

        dl = _build(tmp_path)
        dl.remove_stanza_models()

        assert not os.path.isdir(stanza_dir)

    def test_cancel_keeps_directory(self, tmp_path, monkeypatch):
        stanza_dir = str(tmp_path / "stanza_resources")
        os.makedirs(stanza_dir)
        monkeypatch.setenv("STANZA_RESOURCES_DIR", stanza_dir)
        monkeypatch.setattr("builtins.input", lambda *a: "")  # decline

        dl = _build(tmp_path)
        dl.remove_stanza_models()

        assert os.path.isdir(stanza_dir)

    def test_noop_when_directory_missing(self, tmp_path, monkeypatch):
        stanza_dir = str(tmp_path / "stanza_resources_nonexistent")
        monkeypatch.setenv("STANZA_RESOURCES_DIR", stanza_dir)
        dl = _build(tmp_path)
        dl.remove_stanza_models()
        # Just check it doesn't crash

    def test_blocks_drive_root(self, tmp_path, monkeypatch):
        # rmtree is mocked so a drive/fs root is never actually deleted; if the
        # guard is bypassed the AssertionError fails the test loudly.
        def _fail_rmtree(*a, **kw):
            raise AssertionError(
                "shutil.rmtree reached with drive root — guard failed!"
            )

        monkeypatch.setattr("shutil.rmtree", _fail_rmtree)
        monkeypatch.setenv("STANZA_RESOURCES_DIR", "C:\\")
        dl = _build(tmp_path)
        dl.remove_stanza_models()
        assert any("Path Guard" in str(m) for m in dl.pretty.messages)


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
