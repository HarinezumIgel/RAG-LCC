# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false
"""
Tests for HFDownloader — hash helpers, metadata I/O, snapshot detection,
cache lookup, internet-check, and the download orchestration logic.

Uses DI + attribute injection.  Heavy deps (snapshot_download, user prompts)
are monkeypatched so tests never touch the network or wait for input.
"""

import json
import hashlib
import pytest
import sys
import os
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.HFDownloader import HFDownloader
from Commons.Exceptions import HFDownloaderError, InternetConnectionDisabledError

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def __init__(self, overrides=None):
        self._overrides = overrides or {}
        self._sets: list[tuple[str, object]] = []

    def get(self, key, default=None, **kwargs):
        if key in self._overrides:
            return self._overrides[key]
        mapping = {
            "_ACTIVE_EMBED": "test-impl",
            "_ACTIVE_CROSS": "test-impl",
            "_HF_HUB_CACHE": "",
            "_MODELS": {
                "test-impl": {
                    "_EMBED": {
                        "MODEL": "test-org/test-embed",
                        "REVISION": "",
                        "FRIENDLY_NAME": "Test Embed Model",
                        "SOURCE": "huggingface",
                    },
                    "_CROSS": {
                        "MODEL": "test-org/test-cross",
                        "REVISION": "abcdef0123456789abcdef0123456789abcdef01",
                        "FRIENDLY_NAME": "Test Cross Model",
                        "SOURCE": "huggingface",
                    },
                },
            },
            "_MODELS.test-impl._EMBED": {
                "MODEL": "test-org/test-embed",
                "REVISION": "",
                "FRIENDLY_NAME": "Test Embed Model",
                "SOURCE": "huggingface",
            },
            "_MODELS.test-impl._CROSS": {
                "MODEL": "test-org/test-cross",
                "REVISION": "abcdef0123456789abcdef0123456789abcdef01",
                "FRIENDLY_NAME": "Test Cross Model",
                "SOURCE": "huggingface",
            },
        }
        if key in mapping:
            return mapping[key]
        if default is not None:
            return default
        return None

    def get_int(self, key, default=0):
        val = self.get(key, default)
        return int(val) if val is not None else default

    def get_str(self, key, default=""):
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_bool(self, key, default=False):
        val = self.get(key, default)
        return bool(val) if val is not None else default

    def get_dict(self, key, default=None):
        return self.get(key, default if default is not None else {})

    def set(self, key, value, force=False):
        self._sets.append((key, value))
        self._overrides[key] = value


class StubGlobals:
    pass


class StubHelpers:
    def __init__(self, model_args=None):
        self._model_args = model_args

    def get_model_args(self, what):
        if self._model_args and what in self._model_args:
            return dict(self._model_args[what])
        # Strip _ACTIVE prefix (e.g. "_ACTIVE_EMBED" -> "_EMBED") so the
        # generated model name matches what the real Helpers would return.
        role = what[len("_ACTIVE") :] if what.startswith("_ACTIVE") else what
        return {
            "model_name": f"test-org/test-{role.lower()}",
            "revision": "",
            "local_files_only": True,
            "friendly_name": f"Test {role}",
            "source": "huggingface",
        }

    def setup_logger(self, name):
        return StubLogger()


class StubLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def error(self, msg):
        self.messages.append(("error", msg))

    def debug(self, msg, **kw):
        self.messages.append(("debug", msg))


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[str, ...]] = []

    def write(self, *a, **kw):
        self.messages.append(a)
        return None


class StubSharedHelpers:
    def capture_acceptance_identity_once(self):
        return {
            "accepted_by": "test-user",
            "accepted_by_source": "test",
            "accepted_by_verified": True,
            "host": "test-host",
            "pid": 12345,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_hfdownloader(cfg_overrides=None, helpers=None, base_dir=None):
    """Build an HFDownloader with all deps stubbed, bypassing __init__."""
    dl = object.__new__(HFDownloader)

    dl.cfg = StubConfig(cfg_overrides)
    dl.globalsInstance = StubGlobals()
    dl.helpers = helpers or StubHelpers()
    dl.sharedHelpers = StubSharedHelpers()
    dl.logger = StubLogger()
    dl.pretty = StubPrettyWriter()
    dl.base_dir = Path(base_dir) if base_dir else Path("ModelGovernance/consents")
    dl.identity_cache = None
    dl.hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "1")

    return dl


# ===================================================================
# _hash_text
# ===================================================================


class TestHashText:
    def test_deterministic(self):
        dl = _build_hfdownloader()
        h1 = dl._hash_text("hello")
        h2 = dl._hash_text("hello")
        assert h1 == h2

    def test_matches_sha256(self):
        dl = _build_hfdownloader()
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert dl._hash_text("hello") == expected

    def test_different_inputs_produce_different_hashes(self):
        dl = _build_hfdownloader()
        assert dl._hash_text("a") != dl._hash_text("b")


# ===================================================================
# _compute_cfg_hash
# ===================================================================


class TestComputeCfgHash:
    def test_deterministic(self):
        dl = _build_hfdownloader()
        cfg = {"model": "test", "revision": "abc"}
        h1 = dl._compute_cfg_hash(cfg)
        h2 = dl._compute_cfg_hash(cfg)
        assert h1 == h2

    def test_order_independent(self):
        dl = _build_hfdownloader()
        h1 = dl._compute_cfg_hash({"a": 1, "b": 2})
        h2 = dl._compute_cfg_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs_produce_different_hashes(self):
        dl = _build_hfdownloader()
        h1 = dl._compute_cfg_hash({"a": 1})
        h2 = dl._compute_cfg_hash({"a": 2})
        assert h1 != h2


# ===================================================================
# _write_meta / _load_meta
# ===================================================================


class TestMetadataIO:
    def test_roundtrip(self, tmp_path):
        dl = _build_hfdownloader()
        meta = {"model_id": "test-model", "revision": "abc123"}
        meta_path = tmp_path / "meta.json"

        dl._write_meta(meta_path, meta)
        loaded = dl._load_meta(meta_path)

        assert loaded == meta

    def test_load_nonexistent_returns_empty(self, tmp_path):
        dl = _build_hfdownloader()
        loaded = dl._load_meta(tmp_path / "nonexistent.json")
        assert loaded == {}

    def test_write_creates_file(self, tmp_path):
        dl = _build_hfdownloader()
        meta_path = tmp_path / "out.json"

        dl._write_meta(meta_path, {"key": "value"})

        assert meta_path.exists()
        with open(meta_path) as f:
            payload = json.load(f)
        assert payload["key"] == "value"
        assert isinstance(payload.get("disclaimer"), str)
        assert payload["disclaimer"]


# ===================================================================
# _snapshot_has_content
# ===================================================================


class TestSnapshotHasContent:
    def test_empty_dir_returns_false(self, tmp_path):
        assert HFDownloader._snapshot_has_content(tmp_path) is False

    def test_dir_with_marker_file_returns_true(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert HFDownloader._snapshot_has_content(tmp_path) is True

    def test_dir_with_tokenizer_returns_true(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        assert HFDownloader._snapshot_has_content(tmp_path) is True

    def test_dir_with_safetensors_returns_true(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        assert HFDownloader._snapshot_has_content(tmp_path) is True

    def test_dir_with_pytorch_bin_returns_true(self, tmp_path):
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00")
        assert HFDownloader._snapshot_has_content(tmp_path) is True

    def test_nonexistent_dir_returns_false(self, tmp_path):
        assert HFDownloader._snapshot_has_content(tmp_path / "nope") is False

    def test_unrelated_files_return_false(self, tmp_path):
        (tmp_path / "readme.md").write_text("hello")
        (tmp_path / "random.txt").write_text("x")
        assert HFDownloader._snapshot_has_content(tmp_path) is False

    def test_nested_marker_file_found(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "config.json").write_text("{}")
        assert HFDownloader._snapshot_has_content(tmp_path) is True


# ===================================================================
# _find_cached_snapshot
# ===================================================================


class TestFindCachedSnapshot:
    def _make_snapshot(self, cache_root, model_id, revision, marker="config.json"):
        """Create a fake HF cache snapshot directory structure."""
        sanitized = model_id.replace("/", "--")
        snap_dir = cache_root / f"models--{sanitized}" / "snapshots" / revision
        snap_dir.mkdir(parents=True)
        (snap_dir / marker).write_text("{}")
        return snap_dir

    def test_finds_exact_revision(self, tmp_path):
        dl = _build_hfdownloader()
        snap_dir = self._make_snapshot(tmp_path, "org/model", "rev123")

        path, rev = dl._find_cached_snapshot(str(tmp_path), "org/model", "rev123")

        assert path == snap_dir
        assert rev == "rev123"

    def test_returns_none_when_no_cache(self):
        dl = _build_hfdownloader()
        path, rev = dl._find_cached_snapshot(None, "org/model", "rev123")
        assert path is None
        assert rev == ""

    def test_returns_none_for_empty_string_cache(self):
        dl = _build_hfdownloader()
        path, rev = dl._find_cached_snapshot("", "org/model", "rev123")
        assert path is None
        assert rev == ""

    def test_finds_latest_when_no_revision(self, tmp_path):
        dl = _build_hfdownloader()
        self._make_snapshot(tmp_path, "org/model", "older")
        import time

        time.sleep(0.05)
        newer_dir = self._make_snapshot(tmp_path, "org/model", "newer")

        path, rev = dl._find_cached_snapshot(str(tmp_path), "org/model", None)

        assert path == newer_dir
        assert rev == "newer"

    def test_returns_none_for_nonexistent_model(self, tmp_path):
        dl = _build_hfdownloader()
        path, rev = dl._find_cached_snapshot(str(tmp_path), "org/nonexistent", None)
        assert path is None
        assert rev == ""

    def test_empty_snapshot_directory_is_skipped(self, tmp_path):
        """An empty snapshot dir (no marker files) should not be returned."""
        dl = _build_hfdownloader()
        sanitized = "org--model"
        snap_dir = tmp_path / f"models--{sanitized}" / "snapshots" / "rev1"
        snap_dir.mkdir(parents=True)
        # No marker files → empty

        path, _rev = dl._find_cached_snapshot(str(tmp_path), "org/model", "rev1")

        assert path is None

    def test_nonexistent_cache_root_returns_none(self, tmp_path):
        dl = _build_hfdownloader()
        path, rev = dl._find_cached_snapshot(
            str(tmp_path / "does_not_exist"), "org/model", None
        )
        assert path is None
        assert rev == ""


# ===================================================================
# _check_no_internet_allowed
# ===================================================================


class TestCheckNoInternetAllowed:
    def test_raises_when_offline(self):
        dl = _build_hfdownloader()
        dl.hf_hub_offline = "1"

        with pytest.raises(InternetConnectionDisabledError):
            dl._check_no_internet_allowed("model-x", "rev1", "/cache")

    def test_no_raise_when_online(self):
        dl = _build_hfdownloader()
        dl.hf_hub_offline = "0"

        # Should not raise
        dl._check_no_internet_allowed("model-x", "rev1", "/cache")

    def test_error_message_includes_model_name(self):
        dl = _build_hfdownloader()
        dl.hf_hub_offline = "1"

        with pytest.raises(InternetConnectionDisabledError, match="model-xyz"):
            dl._check_no_internet_allowed("model-xyz", "rev42", "/cache")


# ===================================================================
# download — orchestration
# ===================================================================


class TestDownload:
    def _make_downloadable(self, dl, tmp_path, model_key="_MODELS._EMBED"):
        """Configure a downloader for a test download scenario."""
        model_type = model_key.rsplit(".", 1)[-1]
        model_cfg = {
            "MODEL": f"test-org/test-{model_type.lower()}",
            "REVISION": "",
            "FRIENDLY_NAME": f"Test {model_type}",
            "SOURCE": "huggingface",
        }
        selector = f"_ACTIVE{model_type}" if model_type.startswith("_") else model_type
        impl = dl.cfg.get_str(selector)  # e.g. "test-impl"
        resolved_key = f"_MODELS.{impl}.{model_type}"
        dl.cfg._overrides[resolved_key] = model_cfg
        dl.cfg._overrides["_HF_HUB_CACHE"] = str(tmp_path / "hf_cache")
        dl.base_dir = tmp_path / "consents"
        dl.base_dir.mkdir(parents=True, exist_ok=True)
        dl.hf_hub_offline = "0"

        return model_cfg

    def test_download_skips_when_meta_matches_and_cached(self, tmp_path):
        """If existing metadata matches config and local path exists, skip download."""
        dl = _build_hfdownloader()
        model_cfg = self._make_downloadable(dl, tmp_path)

        cfg_hash = dl._compute_cfg_hash(model_cfg)
        local_path = tmp_path / "local_model"
        local_path.mkdir()
        (local_path / "config.json").write_text("{}")

        # Write matching metadata
        model_dir = dl.base_dir / "_MODELS.test-impl._EMBED"
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": "test-org/test-_embed",
            "revision": "",
            "config_hash": cfg_hash,
            "local_path": str(local_path),
        }
        dl._write_meta(model_dir / "download_meta.json", meta)

        download_result = dl.download("_MODELS._EMBED")

        assert download_result["model_id"] == "test-org/test-_embed"
        # No actual download should have occurred

    def test_download_uses_cached_snapshot(self, tmp_path, monkeypatch):
        """If model found in HF cache, metadata is written without download."""
        dl = _build_hfdownloader()
        self._make_downloadable(dl, tmp_path)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        # Create a fake HF cache snapshot
        hf_cache = tmp_path / "hf_cache"
        snap_dir = (
            hf_cache / "models--test-org--test-_embed" / "snapshots" / "snap_hash"
        )
        snap_dir.mkdir(parents=True)
        (snap_dir / "config.json").write_text("{}")

        result = dl.download("_MODELS._EMBED")

        assert result["local_path"] == str(snap_dir)
        assert result["revision"] == "snap_hash"

    def test_download_raises_when_no_hf_cache_configured(self, tmp_path):
        """If _HF_HUB_CACHE is empty and no cached snapshot, raise HFDownloaderError."""
        dl = _build_hfdownloader()
        dl.cfg._overrides["_MODELS.test-impl._EMBED"] = {
            "MODEL": "test-org/test-embed",
            "REVISION": "",
            "FRIENDLY_NAME": "Test Embed",
            "SOURCE": "huggingface",
        }
        dl.cfg._overrides["_HF_HUB_CACHE"] = ""  # No cache configured
        dl.base_dir = tmp_path / "consents"
        dl.base_dir.mkdir(parents=True, exist_ok=True)
        dl.hf_hub_offline = "0"

        with pytest.raises(HFDownloaderError, match="No HF Hub cache configured"):
            dl.download("_MODELS._EMBED")

    def test_download_raises_when_offline_and_not_cached(self, tmp_path):
        """If model not cached and HF_HUB_OFFLINE=1, should raise."""
        dl = _build_hfdownloader()
        self._make_downloadable(dl, tmp_path)
        dl.hf_hub_offline = "1"  # Offline

        # No cached snapshot exists → should hit internet check
        with pytest.raises(InternetConnectionDisabledError):
            dl.download("_MODELS._EMBED")

    def test_download_no_config_raises_valueerror(self, tmp_path):
        """If config key yields None, should raise ValueError."""
        dl = _build_hfdownloader()
        dl.base_dir = tmp_path / "consents"
        dl.base_dir.mkdir(parents=True, exist_ok=True)

        # Use a key that doesn't have any config
        with pytest.raises(ValueError, match="No configuration found"):
            dl.download("_MODELS.NONEXISTENT")

    def test_download_persists_revision_to_config(self, tmp_path, monkeypatch):
        """When a cached snapshot is found, resolved revision should be persisted."""
        dl = _build_hfdownloader()
        self._make_downloadable(dl, tmp_path)
        monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

        # Create snapshot with resolved hash
        hf_cache = tmp_path / "hf_cache"
        snap_dir = (
            hf_cache / "models--test-org--test-_embed" / "snapshots" / "abc123hash"
        )
        snap_dir.mkdir(parents=True)
        (snap_dir / "config.json").write_text("{}")

        dl.download("_MODELS._EMBED")

        # Check that revision was persisted to config
        assert any(
            k == "_MODELS._EMBED.REVISION" and v == "abc123hash"
            for k, v in dl.cfg._sets
        )


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_model_marker_files_set_is_correct(self):
        """Verify the expected marker files are in the set."""
        expected = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        }
        assert HFDownloader._MODEL_MARKER_FILES == expected

    def test_base_dir_is_path_object(self):
        dl = _build_hfdownloader()
        assert isinstance(dl.base_dir, Path)
