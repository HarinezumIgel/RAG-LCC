# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false, reportUnusedVariable=false, reportUnknownLambdaType=false
"""
Tests for SpacyDownloader - consent validation, metadata persistence,
ensure_models orchestration, and consent cleanup.

Uses DI + attribute injection. Heavy deps (network, subprocess, user prompts)
are monkeypatched so tests never touch the network or install packages.
"""

import hashlib
import json
import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.SpacyDownloader import SpacyDownloader


class StubLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg, *args):
        self.messages.append(("info", msg % args if args else msg))

    def error(self, msg, *args):
        self.messages.append(("error", msg % args if args else msg))


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


_MODELS = ["en_core_web_sm", "de_core_news_sm"]
_LICENSE_TEXT = "MIT License\n\nCopyright (C) Explosion AI\n"


def _build(tmp_path, *, write_license=False, write_meta=None):
    """Build a SpacyDownloader with stubs, pointing at tmp_path as project root."""
    dl = object.__new__(SpacyDownloader)

    dl.root = str(tmp_path)
    dl.models = list(_MODELS)

    license_dir_rel = os.path.join("ModelGovernance", "licenses", "spacy_models")
    consent_dir_rel = os.path.join("ModelGovernance", "consents", "spacy_models")

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

    if write_license:
        os.makedirs(dl.license_dir, exist_ok=True)
        with open(dl.license_path, "w", encoding="utf-8") as f:
            f.write(_LICENSE_TEXT)

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


class TestEnsureModelsLicenseMissing:
    def test_returns_false_when_download_fails(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: None)
        assert dl.ensure_models() is False


class TestEnsureModelsExistingConsent:
    def test_returns_true_on_valid_consent(self, tmp_path, monkeypatch):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "models": list(_MODELS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(dl, "_missing_models", lambda _models: [])
        assert dl.ensure_models() is True

    def test_no_prompts_on_valid_consent(self, tmp_path, monkeypatch):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "models": list(_MODELS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)
        monkeypatch.setattr(dl, "_missing_models", lambda _models: [])
        monkeypatch.setattr(
            "builtins.input",
            lambda _prompt="": pytest.fail("unexpected prompt"),
        )
        assert dl.ensure_models() is True

    def test_drifted_model_list_requires_reconsent(self, tmp_path):
        license_meta = {
            "consent": True,
            "license_hash": _license_hash(),
        }
        dl = _build(tmp_path, write_license=True, write_meta=license_meta)
        with open(dl.download_meta_path, "w", encoding="utf-8") as f:
            json.dump({"models": ["en_core_web_sm"]}, f)

        assert dl._check_existing_consent() is False


class TestEnsureModelsPrompts:
    def test_decline_license_returns_false(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_models() is False

    def test_accept_then_decline_download_returns_false(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "y", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))
        assert dl.ensure_models() is False

    def test_full_accept_writes_meta_and_installs(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)
        monkeypatch.setattr(dl, "_fetch_license", lambda: _LICENSE_TEXT)
        responses = iter(["", "y", "y"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

        installs: list[list[str]] = []
        monkeypatch.setattr(dl, "_missing_models", lambda _models: list(_MODELS))
        monkeypatch.setattr(
            dl,
            "_install_models",
            lambda models: installs.append(list(models)) or True,
        )

        assert dl.ensure_models() is True
        assert installs == [list(_MODELS)]

        assert os.path.isfile(dl.license_meta_path)
        with open(dl.license_meta_path, "r", encoding="utf-8") as f:
            license_meta = json.load(f)
        assert license_meta["component"] == "spacy_models"
        assert license_meta["consent"] is True
        assert license_meta["license_hash"] == _license_hash()

        assert os.path.isfile(dl.download_meta_path)
        with open(dl.download_meta_path, "r", encoding="utf-8") as f:
            download_meta = json.load(f)
        assert download_meta["models"] == list(_MODELS)


class TestEnsureModelsMissingWithConsent:
    def test_installs_missing_models_when_consent_exists(self, tmp_path, monkeypatch):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "models": list(_MODELS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)

        installs: list[list[str]] = []
        monkeypatch.setattr(dl, "_missing_models", lambda _models: ["de_core_news_sm"])
        monkeypatch.setattr(
            dl,
            "_install_models",
            lambda models: installs.append(list(models)) or True,
        )

        assert dl.ensure_models() is True
        assert installs == [["de_core_news_sm"]]


class TestRemoveConsent:
    def test_deletes_meta_and_license_copy(self, tmp_path):
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "models": list(_MODELS),
        }
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
        meta = {
            "consent": True,
            "license_hash": _license_hash(),
            "models": list(_MODELS),
        }
        dl = _build(tmp_path, write_license=True, write_meta=meta)

        outside_path = tmp_path.parent / "outside_spacy_LICENSE.txt"
        outside_path.write_text("license", encoding="utf-8")
        dl.license_path = str(outside_path)

        dl.remove_consent()

        assert dl.license_meta_path in dl.file_utils.deleted
        assert dl.download_meta_path in dl.file_utils.deleted
        assert dl.license_path not in dl.file_utils.deleted


class TestRemoveAllPathGuard:
    def test_uninstalls_only_models_with_guarded_paths(self, tmp_path, monkeypatch):
        dl = _build(tmp_path)

        safe_model = "en_core_web_sm"
        outside_model = "de_core_news_sm"

        safe_model_path = os.path.join(
            str(tmp_path),
            ".venv",
            "Lib",
            "site-packages",
            safe_model,
        )
        os.makedirs(safe_model_path, exist_ok=True)

        monkeypatch.setattr(dl, "installed_models", lambda: [safe_model, outside_model])
        monkeypatch.setattr(dl, "_prompt_yes_no", lambda _prompt="": True)
        monkeypatch.setattr(
            "Compliance.SpacyDownloader.model_location",
            lambda model: (
                safe_model_path
                if model == safe_model
                else os.path.join(str(tmp_path.parent), "outside", model)
            ),
        )

        captured: dict[str, Any] = {}

        def _remove_models(models, *, on_uninstall_error=None):
            captured["models"] = list(models)
            return len(models), len(models)

        monkeypatch.setattr("Compliance.SpacyDownloader.remove_models", _remove_models)

        dl.remove_all()

        removed_models = captured.get("models", [])
        assert removed_models == [safe_model]

    def test_skips_uninstall_when_model_directory_is_missing(
        self,
        tmp_path,
        monkeypatch,
    ):
        dl = _build(tmp_path)

        model_name = "en_core_web_sm"
        missing_model_path = os.path.join(
            str(tmp_path),
            ".venv",
            "Lib",
            "site-packages",
            model_name,
        )

        monkeypatch.setattr(dl, "installed_models", lambda: [model_name])
        monkeypatch.setattr(dl, "_prompt_yes_no", lambda _prompt="": True)
        monkeypatch.setattr(
            "Compliance.SpacyDownloader.model_location",
            lambda _model: missing_model_path,
        )

        captured: dict[str, Any] = {}

        def _remove_models(models, *, on_uninstall_error=None):
            captured["models"] = list(models)
            return len(models), len(models)

        monkeypatch.setattr("Compliance.SpacyDownloader.remove_models", _remove_models)

        dl.remove_all()

        removed_models = captured.get("models", [])
        assert removed_models == []


class TestShowLicensePager:
    def test_short_text_printed(self, capsys):
        SpacyDownloader._show_license_pager("Hello\nWorld")
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" in captured.out
