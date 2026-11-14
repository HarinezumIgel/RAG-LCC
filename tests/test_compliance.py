# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false
"""
Tests for Compliance — metadata I/O, text hashing, canonicalization,
license subdirectory resolution, model flattening, verify flow, and
_process_one orchestration.

Uses DI + attribute injection.  Heavy deps (network, user prompts, TLS)
are monkeypatched so tests never touch the network or wait for input.
"""

import hashlib
import json
import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.Compliance import Compliance
from Helpers.ModelSelection import model_selectors
from Commons.Exceptions import (
    ArgosConsentMissingError,
    ComplianceViolationError,
    InternetConnectionDisabledError,
    SpacyConsentMissingError,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

SAMPLE_MODELS: dict[str, Any] = {
    "snowflake": {
        "_EMBED": {
            "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
            "FRIENDLY_NAME": "Snowflake Arctic Embed L v2.0",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "Embedder: Snowflake arctic-embed-l-v2.0",
            "MODEL_CARD": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
        },
    },
    "mistral": {
        "_LLM": {
            "MODEL": "mistral:7b",
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/mistral-7b",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://example.com/license",
            "COMPLIANCE_MSG": "LLM: Mistral 7B",
            "TAG": "LLM: Mistral 7B",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_MISTRAL",
        },
        "_LLM_CHK": {
            "MODEL": "mistral:7b",
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/mistral-7b",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://example.com/license",
            "COMPLIANCE_MSG": "LLM: Mistral 7B",
            "TAG": "LLM: Mistral 7B",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
        },
    },
    "ollama": {
        "_OLLAMA": {
            "PROVIDER": "ollama",
            "FRIENDLY_NAME": "Ollama Local LLM Provider",
            "SOURCE": "https://github.com/ollama/ollama",
            "BASE_URL": "http://localhost:11434",
            "LICENSE": "MIT",
            "LICENSE_URL": "https://example.com/mit-license",
            "MODEL_LICENSES_NOTE": "Each model has its own license",
            "COMPLIANCE_MSG": "OLLAMA Local LLM Provider",
            "MODEL_CARD": "https://ollama.com",
        },
    },
}


class StubConfig:
    """Provides deterministic config values for Compliance tests."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        self._overrides = overrides or {}

    def get(self, key, default=None):
        if key in self._overrides:
            return self._overrides[key]
        mapping: dict[str, Any] = {
            "_MODELS": SAMPLE_MODELS,
            "_ACTIVE_EMBED": "snowflake",
            "_ACTIVE_LLM": "mistral",
            "_ACTIVE_LLM_CHK": "mistral",
            "_ACTIVE_ENDPOINT": "ollama",
            "_FRIENDLY_NAME": "TestApp",
            "_CRITICAL_CONFIG_HASHES": {
                "Config_Models": "fakehash",
                "Config_Banned_Detection": "fakehash",
                "Config_Banned_Content": "fakehash",
                "Config_Banned_Prompts": "fakehash",
                "Config_Load_Retrievers": "fakehash",
                "Config_Load_Chunkers": "fakehash",
                "Config_WebSearch": "fakehash",
                "Config_Internet_Env": "fakehash",
            },
            "DEBUG_LEVEL": 0,
            "_ARGOS_DEFINITIONS": {
                "ARGOS_LANGUAGES": [("en", "de")],
                "LANG_CODE_TO_NAME": {"de": "german", "en": "english"},
            },
        }
        # support dot-notation for nested lookups
        parts = key.split(".")
        node: Any = mapping
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_int(self, key, default=0):
        val = self.get(key, default)
        return int(val) if val is not None else default

    def get_str(self, key, default=""):
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_bool(self, key, default=False):
        val = self.get(key, default)
        return bool(val) if val is not None else default

    def get_dict(self, key, default=None, *, silent=False):
        return self.get(key, default if default is not None else {})

    def get_list(self, key, default=None, *, silent=False):
        val = self.get(key, default if default is not None else [])
        return (
            val if isinstance(val, list) else (default if default is not None else [])
        )

    indirect_list = get_list
    indirect_dict = get_dict

    def indirect_get(self, key, default=None):
        return self.get(key, default), key

    def set(self, key, value, force=False):
        self._overrides[key] = value


class StubLogger:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

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


class StubGlobals:
    pass


class StubHelpers:
    def setup_logger(self, name):
        return StubLogger()

    def sanitize_path_component(self, value: str) -> str:
        import re as _re

        value = value.lower()
        value = _re.sub(r"\s+", "", value)
        value = _re.sub(r'[\\/:*?"<>|]', "", value)
        return value


class StubSharedHelpers:
    def capture_acceptance_identity_once(self):
        return {
            "accepted_by": "test-user",
            "accepted_by_source": "test",
            "accepted_by_verified": True,
            "host": "test-host",
            "pid": 12345,
        }


class StubFileUtils:
    def compute_hash(self, text: str) -> str:
        if isinstance(text, (dict, list, tuple)):
            text = json.dumps(text, sort_keys=True, separators=(",", ":"))
        else:
            text = str(text)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def hash_module(self, mod, algo="sha256"):
        return "fakehash"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_compliance(
    *,
    cfg_overrides: dict[str, Any] | None = None,
    connection_mode: str = "0",
    models_override: dict[str, Any] | None = None,
    friendly_name: str = "TestApp",
    active_models: dict[str, str] | None = None,
) -> Compliance:
    """Build a Compliance instance with all deps stubbed, bypassing __init__.

    ``active_models`` replaces every _ACTIVE_* selector (missing ones = "").
    """
    c = object.__new__(Compliance)

    cfg = StubConfig(cfg_overrides)
    if models_override is not None:
        cfg.set("_MODELS", models_override)
    if active_models is not None:
        for selector in model_selectors():
            cfg.set(selector, active_models.get(selector, ""))

    c.cfg = cfg
    c.globalsInstance = StubGlobals()
    c.helpers = StubHelpers()
    c.sharedHelpers = StubSharedHelpers()
    c.fileUtils = StubFileUtils()
    c.logger = StubLogger()
    c.pretty = StubPrettyWriter()
    c.friendly_name = friendly_name
    c.base_dir = "ModelGovernance/licenses"
    c.license_download = connection_mode
    c.acceptance_identity = None

    # Select the active models exactly as __init__ does
    c.models = c._select_active_models()

    return c


@pytest.fixture(autouse=True)
def reset_compliance():
    """Reset the Compliance singleton before and after each test."""
    Compliance._reset()
    yield
    Compliance._reset()


# ===================================================================
# _canonicalize_text
# ===================================================================


class TestCanonicalizeText:
    def test_strips_and_normalizes_newlines(self):
        c = _build_compliance()
        assert c._canonicalize_text("  hello\r\nworld  ") == "hello\nworld"

    def test_empty_string(self):
        c = _build_compliance()
        assert c._canonicalize_text("") == ""

    def test_already_canonical(self):
        c = _build_compliance()
        assert c._canonicalize_text("hello\nworld") == "hello\nworld"

    def test_only_whitespace(self):
        c = _build_compliance()
        assert c._canonicalize_text("   \r\n   ") == ""


# ===================================================================
# _compute_text_hash
# ===================================================================


class TestComputeTextHash:
    def test_deterministic(self):
        c = _build_compliance()
        h1 = c._compute_text_hash("hello world")
        h2 = c._compute_text_hash("hello world")
        assert h1 == h2

    def test_matches_sha256_of_canonical(self):
        c = _build_compliance()
        text = "  hello\r\nworld  "
        canonical = text.replace("\r\n", "\n").strip()
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert c._compute_text_hash(text) == expected

    def test_different_inputs_differ(self):
        c = _build_compliance()
        assert c._compute_text_hash("a") != c._compute_text_hash("b")

    def test_crlf_and_lf_produce_same_hash(self):
        c = _build_compliance()
        assert c._compute_text_hash("a\r\nb") == c._compute_text_hash("a\nb")


# ===================================================================
# _load_meta / _save_meta
# ===================================================================


class TestMetadataIO:
    def test_roundtrip(self, tmp_path):
        c = _build_compliance()
        meta = {"model_id": "test-model", "consent": True}
        meta_path = str(tmp_path / "meta.json")

        c._save_meta(meta_path, meta)
        loaded = c._load_meta(meta_path)

        assert loaded == meta

    def test_load_nonexistent_returns_empty(self, tmp_path):
        c = _build_compliance()
        loaded = c._load_meta(str(tmp_path / "nonexistent.json"))
        assert loaded == {}

    def test_save_creates_file(self, tmp_path):
        c = _build_compliance()
        meta_path = str(tmp_path / "out.json")
        c._save_meta(meta_path, {"key": "value"})
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            assert json.load(f) == {"key": "value"}

    def test_save_overwrites_existing(self, tmp_path):
        c = _build_compliance()
        meta_path = str(tmp_path / "meta.json")
        c._save_meta(meta_path, {"v": 1})
        c._save_meta(meta_path, {"v": 2})
        loaded = c._load_meta(meta_path)
        assert loaded["v"] == 2


# ===================================================================
# _compute_height
# ===================================================================


class TestComputeHeight:
    def test_returns_positive_integer(self):
        c = _build_compliance()
        h = c._compute_height()
        assert isinstance(h, int)
        assert h >= 5

    def test_fallback_minimum(self):
        """When terminal size is very small, clamp to at least 5."""
        import shutil as _shutil

        orig = _shutil.get_terminal_size
        _shutil.get_terminal_size = lambda *a: os.terminal_size((80, 6))  # type: ignore[reportUnknownParameterType]
        try:
            c = _build_compliance()
            assert c._compute_height() >= 4  # 6 - 2 = 4, but >= 5 guard
        finally:
            _shutil.get_terminal_size = orig

    def test_exception_returns_15(self):
        import shutil as _shutil

        orig = _shutil.get_terminal_size

        def _raise(*a, **kw):
            raise OSError("no terminal")

        _shutil.get_terminal_size = _raise
        try:
            c = _build_compliance()
            assert c._compute_height() == 15
        finally:
            _shutil.get_terminal_size = orig


# ===================================================================
# _license_subdir_for
# ===================================================================


class TestLicenseSubdirFor:
    def test_model_with_slash(self):
        c = _build_compliance()
        chk = {
            "MODEL": "snowflake/arctic-embed-l-v2.0",
            "SOURCE": "https://example.com",
        }
        result = c._license_subdir_for(chk, "snowflake._EMBED")
        assert "arctic-embed-l-v2.0" in result
        assert "snowflake._embed" in result

    def test_provider_instead_of_model(self):
        c = _build_compliance()
        chk = {"PROVIDER": "ollama", "SOURCE": "https://github.com/ollama/ollama"}
        result = c._license_subdir_for(chk, "ollama.OLLAMA")
        assert "ollama" in result

    def test_no_model_no_provider_falls_back_to_source(self):
        c = _build_compliance()
        chk = {"SOURCE": "https://example.com/some-project"}
        result = c._license_subdir_for(chk, "test.ROLE")
        assert "some-project" in result

    def test_no_model_no_source_returns_section(self):
        c = _build_compliance()
        chk: dict[str, Any] = {}
        result = c._license_subdir_for(chk, "fallback_section")
        assert result == "fallback_section"

    def test_same_model_different_sections_produce_different_subdirs(self):
        c = _build_compliance()
        chk = {"MODEL": "mistral:7b", "SOURCE": "https://example.com"}
        sub1 = c._license_subdir_for(chk, "mistral.LLM")
        sub2 = c._license_subdir_for(chk, "mistral.LLM_CHK")
        assert sub1 != sub2


# ===================================================================
# Model flattening (_MODELS -> self.models)
# ===================================================================


class TestActiveModelSelection:
    def test_selects_active_models_as_impl_dot_role(self):
        c = _build_compliance()
        assert "snowflake._EMBED" in c.models
        assert "mistral._LLM" in c.models
        assert "mistral._LLM_CHK" in c.models
        assert "ollama._OLLAMA" in c.models

    def test_selected_keys_count(self):
        c = _build_compliance()
        # Stub selectors: EMBED=snowflake, LLM=mistral, LLM_CHK=mistral, ENDPOINT=ollama
        assert len(c.models) == 4

    def test_config_values_preserved(self):
        c = _build_compliance()
        embed = c.models["snowflake._EMBED"]
        assert embed["MODEL"] == "snowflake/snowflake-arctic-embed-l-v2.0"
        assert embed["LICENSE"] == "Apache-2.0"

    def test_no_selectors_selects_nothing(self):
        c = _build_compliance(active_models={})
        assert c.models == {}

    def test_inactive_roles_of_active_impl_are_ignored(self):
        c = _build_compliance(active_models={"_ACTIVE_LLM": "mistral"})
        assert set(c.models) == {"mistral._LLM"}

    def test_inactive_impls_are_ignored(self):
        models = {
            "a": {"_EMBED": {"MODEL": "a"}},
            "b": {"_EMBED": {"MODEL": "b"}},
        }
        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "b"}
        )
        assert set(c.models) == {"b._EMBED"}

    def test_switching_selector_switches_checked_model(self):
        models = {
            "a": {"_EMBED": {"MODEL": "a"}},
            "b": {"_EMBED": {"MODEL": "b"}},
        }
        c1 = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "a"}
        )
        c2 = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "b"}
        )
        assert set(c1.models) == {"a._EMBED"}
        assert set(c2.models) == {"b._EMBED"}

    def test_endpoint_selects_provider_role_of_endpoint_impl(self):
        models = {
            "vllm": {"_VLLM": {"PROVIDER": "vllm"}, "_OTHER": {"PROVIDER": "x"}},
            "ollama": {"_OLLAMA": {"PROVIDER": "ollama"}},
        }
        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_ENDPOINT": "vllm"}
        )
        assert set(c.models) == {"vllm._VLLM"}

    def test_used_by_excludes_other_apps(self):
        models = {"m": {"_LLM": {"MODEL": "m", "USED_BY": ["OtherApp"]}}}
        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_LLM": "m"}
        )
        assert c.models == {}

    def test_unknown_active_impl_raises(self):
        with pytest.raises(
            ComplianceViolationError, match="_ACTIVE_LLM = 'nope': Unknown impl"
        ):
            _build_compliance(active_models={"_ACTIVE_LLM": "nope"})

    def test_missing_role_in_active_impl_raises(self):
        with pytest.raises(
            ComplianceViolationError, match="Role '_CROSS' not found in impl 'mistral'"
        ):
            _build_compliance(active_models={"_ACTIVE_CROSS": "mistral"})

    def test_real_config_selects_only_active_models(self):
        import Configuration.Config_Models as cm

        active = {sel: getattr(cm, sel) for sel in model_selectors()}
        c = _build_compliance(
            models_override=cm._MODELS, active_models=active, friendly_name="RAGChat"
        )
        impls = {key.split(".")[0] for key in c.models}
        expected = {
            active[sel]
            for sel in (
                "_ACTIVE_LLM",
                "_ACTIVE_LLM_CHK",
                "_ACTIVE_LLM_REWRITE_PROMPT",
                "_ACTIVE_LLM_JUDGE",
                "_ACTIVE_EMBED",
                "_ACTIVE_CROSS",
                "_ACTIVE_ENDPOINT",
            )
        }
        assert impls == expected
        all_keys = {f"{i}.{r}" for i, roles in cm._MODELS.items() for r in roles}
        assert set(c.models) < all_keys

    @pytest.mark.parametrize(
        ("app", "expected"),
        [
            ("RAGChat", True),
            ("RAGChatService", True),
            ("RAGLoad", False),
            ("DocClassify", False),
        ],
    )
    def test_judge_license_consent_scoped_to_chat_apps(self, app, expected):
        from Configuration.Config_Models import _ACTIVE_LLM_JUDGE, _MODELS

        c = _build_compliance(
            models_override=_MODELS,
            friendly_name=app,
            active_models={"_ACTIVE_LLM_JUDGE": _ACTIVE_LLM_JUDGE},
        )
        key = f"{_ACTIVE_LLM_JUDGE}._LLM_JUDGE"
        assert (key in c.models) is expected
        if expected:
            judge = c.models[key]
            assert judge["LICENSE"] == "Apache-2.0"
            assert judge["LICENSE_URL"] == (
                "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/resolve/main/LICENSE"
            )
            assert judge["SOURCE"] == "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct"


# ===================================================================
# _get_acceptance_identity
# ===================================================================


class TestAcceptanceIdentity:
    def test_returns_dict_with_expected_keys(self):
        c = _build_compliance()
        identity = c._get_acceptance_identity()
        assert identity["accepted_by"] == "test-user"
        assert identity["host"] == "test-host"
        assert identity["pid"] == 12345

    def test_cached_after_first_call(self):
        c = _build_compliance()
        id1 = c._get_acceptance_identity()
        id2 = c._get_acceptance_identity()
        assert id1 is id2


# ===================================================================
# _get_tls_fingerprint
# ===================================================================


class TestTlsFingerprint:
    def test_non_https_returns_empty(self):
        c = _build_compliance()
        assert c._get_tls_fingerprint("http://example.com") == ""

    def test_invalid_url_returns_empty(self):
        c = _build_compliance()
        assert c._get_tls_fingerprint("not-a-url") == ""

    def test_connection_failure_returns_empty(self):
        c = _build_compliance()
        # unreachable host → should gracefully return ""
        result = c._get_tls_fingerprint("https://192.0.2.1:1")
        assert result == ""


# ===================================================================
# _fetch_license — offline gate
# ===================================================================


class TestFetchLicenseOfflineGate:
    def test_offline_mode_prompts_and_aborts_on_no(self, monkeypatch):
        c = _build_compliance(connection_mode="0")
        monkeypatch.setattr("builtins.input", lambda _: "n")  # type: ignore[reportUnknownParameterType]
        with pytest.raises(InternetConnectionDisabledError):
            c._fetch_license("https://example.com/license", "TestModel", "test/path")

    def test_offline_mode_proceeds_on_yes(self, monkeypatch):
        c = _build_compliance(connection_mode="0")
        inputs = iter(["y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))  # type: ignore[reportUnknownParameterType]

        import requests as _req

        class FakeResponse:
            text = "MIT License\n..."
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *args, **kwargs: FakeResponse())  # type: ignore[reportUnknownParameterType]
        result = c._fetch_license("https://example.com/license", "Test", "path")
        assert "MIT License" in result

    def test_online_mode_skips_prompt(self, monkeypatch):
        c = _build_compliance(connection_mode="1")

        import requests as _req

        class FakeResponse:
            text = "Apache License"
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *args, **kwargs: FakeResponse())  # type: ignore[reportUnknownParameterType]
        result = c._fetch_license("https://example.com/license", "Test", "path")
        assert "Apache License" in result

    def test_network_error_raises_compliance_violation(self, monkeypatch):
        c = _build_compliance(connection_mode="1")

        import requests as _req

        def fail_get(*args):
            raise ConnectionError("network down")

        monkeypatch.setattr(_req, "get", fail_get)
        with pytest.raises(ComplianceViolationError):
            c._fetch_license("https://example.com/license", "Test", "path")


# ===================================================================
# _process_one — bundled license (offline-first path)
# ===================================================================


class TestProcessOneBundled:
    def test_bundled_license_accepted(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        section = "snowflake._EMBED"
        chk = c.models[section]
        subdir = c._license_subdir_for(chk, section)
        lic_dir = os.path.join(str(tmp_path), subdir)
        os.makedirs(lic_dir, exist_ok=True)
        lic_path = os.path.join(lic_dir, "LICENSE.txt")
        with open(lic_path, "w") as f:
            f.write("Apache License Version 2.0")

        # Simulate user accepting
        monkeypatch.setattr("builtins.input", lambda _: "y")  # type: ignore[reportUnknownParameterType]

        c._process_one(c.models, section)

        meta_path = os.path.join(lic_dir, "license_meta.json")
        assert os.path.isfile(meta_path)
        meta = json.loads(open(meta_path).read())
        assert meta["consent"] is True
        assert meta["source"] == "bundled"
        assert meta["accepted_by"] == "test-user"

    def test_bundled_license_rejected_raises(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        section = "snowflake._EMBED"
        chk = c.models[section]
        subdir = c._license_subdir_for(chk, section)
        lic_dir = os.path.join(str(tmp_path), subdir)
        os.makedirs(lic_dir, exist_ok=True)
        lic_path = os.path.join(lic_dir, "LICENSE.txt")
        with open(lic_path, "w") as f:
            f.write("Some License Text")

        # Simulate user rejecting
        monkeypatch.setattr("builtins.input", lambda _: "n")  # type: ignore[reportUnknownParameterType]

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, section)


# ===================================================================
# _process_one — already consented (skip path)
# ===================================================================


class TestProcessOneAlreadyConsented:
    def test_skips_when_hashes_match_and_consent_granted(self, tmp_path):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        section = "snowflake._EMBED"
        chk = c.models[section]
        subdir = c._license_subdir_for(chk, section)
        lic_dir = os.path.join(str(tmp_path), subdir)
        os.makedirs(lic_dir, exist_ok=True)

        license_text = "Apache License Version 2.0"
        lic_path = os.path.join(lic_dir, "LICENSE.txt")
        with open(lic_path, "w") as f:
            f.write(license_text)

        license_hash = c._compute_text_hash(license_text)
        chk_str = json.dumps(chk, sort_keys=True)
        config_hash = c.fileUtils.compute_hash(chk_str)

        meta = {
            "license_hash": license_hash,
            "config_hash": config_hash,
            "consent": True,
        }
        meta_path = os.path.join(lic_dir, "license_meta.json")
        c._save_meta(meta_path, meta)

        # Should return without prompting (no input mock needed)
        c._process_one(c.models, section)

    def test_reruns_when_config_hash_changed(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        section = "snowflake._EMBED"
        chk = c.models[section]
        subdir = c._license_subdir_for(chk, section)
        lic_dir = os.path.join(str(tmp_path), subdir)
        os.makedirs(lic_dir, exist_ok=True)

        license_text = "Apache License Version 2.0"
        lic_path = os.path.join(lic_dir, "LICENSE.txt")
        with open(lic_path, "w") as f:
            f.write(license_text)

        license_hash = c._compute_text_hash(license_text)

        meta = {
            "license_hash": license_hash,
            "config_hash": "stale_hash_that_no_longer_matches",
            "consent": True,
            "source": "fetched",
            "config": {**chk, "REVISION": "old-rev", "API_KEY": "secret-old"},
        }
        meta_path = os.path.join(lic_dir, "license_meta.json")
        c._save_meta(meta_path, meta)

        # Offline mode: re-consent uses the stored license, no fetch
        import requests as _req

        def _no_fetch(*args, **kwargs):
            raise AssertionError("offline config re-consent must not fetch")

        monkeypatch.setattr(_req, "get", _no_fetch)  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr("builtins.input", lambda _: "y")  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr(c, "_show_pager", lambda text: None)

        c._process_one(c.models, section)

        saved = c._load_meta(meta_path)
        assert saved["config_hash"] == c.fileUtils.compute_hash(
            json.dumps(chk, sort_keys=True)
        )
        assert saved["config"] == chk
        assert saved["license_hash"] == license_hash
        assert saved["source"] == "fetched"
        assert saved["accepted_by"] == "test-user"

    def test_config_change_declined_raises(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        section = "snowflake._EMBED"
        chk = c.models[section]
        lic_dir = os.path.join(str(tmp_path), c._license_subdir_for(chk, section))
        os.makedirs(lic_dir, exist_ok=True)
        with open(os.path.join(lic_dir, "LICENSE.txt"), "w") as f:
            f.write("Apache License Version 2.0")
        meta = {
            "license_hash": c._compute_text_hash("Apache License Version 2.0"),
            "config_hash": "stale",
            "consent": True,
            "config": {},
        }
        meta_path = os.path.join(lic_dir, "license_meta.json")
        c._save_meta(meta_path, meta)

        answers = iter(["", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr(c, "_show_pager", lambda text: None)

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, section)
        assert c._load_meta(meta_path)["config_hash"] == "stale"

    def test_changed_config_keys_lists_names_only(self):
        changed = Compliance._changed_config_keys(
            {"A": 1, "API_KEY": "old", "GONE": True},
            {"A": 1, "API_KEY": "new", "NEW": 2},
        )
        assert changed == ["API_KEY", "GONE", "NEW"]
        assert Compliance._changed_config_keys(None, {"A": 1}) == []


# ===================================================================
# _process_one — live fetch path
# ===================================================================


class TestProcessOneLiveFetch:
    def test_first_time_fetch_stores_license_and_meta(self, tmp_path, monkeypatch):
        c = _build_compliance(connection_mode="1")
        c.base_dir = str(tmp_path)

        section = "mistral._LLM"
        chk = c.models[section]

        import requests as _req

        class FakeResponse:
            text = "Apache License, Version 2.0"
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *args, **kwargs: FakeResponse())  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr("builtins.input", lambda _: "y")  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr(
            "Compliance.Compliance.Compliance._get_tls_fingerprint",
            lambda self, url: "ab" * 32,  # type: ignore[reportUnknownParameterType]
        )

        c._process_one(c.models, section)

        subdir = c._license_subdir_for(chk, section)
        lic_dir = os.path.join(str(tmp_path), subdir)

        lic_path = os.path.join(lic_dir, "LICENSE.txt")
        meta_path = os.path.join(lic_dir, "license_meta.json")

        assert os.path.isfile(lic_path)
        assert os.path.isfile(meta_path)

        meta = json.loads(open(meta_path).read())
        assert meta["consent"] is True
        assert meta["source"] == "fetched"
        assert meta["model_id"] == "mistral:7b"
        assert meta["accepted_by"] == "test-user"

        with open(lic_path) as f:
            assert f.read() == "Apache License, Version 2.0"

    def test_rejection_raises_compliance_violation(self, tmp_path, monkeypatch):
        c = _build_compliance(connection_mode="1")
        c.base_dir = str(tmp_path)

        section = "mistral._LLM"

        import requests as _req

        class FakeResponse:
            text = "Some License"
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *args, **kwargs: FakeResponse())  # type: ignore[reportUnknownParameterType]
        monkeypatch.setattr("builtins.input", lambda _: "n")  # type: ignore[reportUnknownParameterType]

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, section)


# ===================================================================
# _process_one — missing LICENSE_URL
# ===================================================================


class TestProcessOneMissingUrl:
    def test_missing_license_url_raises(self, tmp_path):
        c = _build_compliance(
            models_override={"stub": {"_LLM": {"MODEL": "x", "FRIENDLY_NAME": "X"}}},
            active_models={"_ACTIVE_LLM": "stub"},
        )
        c.base_dir = str(tmp_path)

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, "stub._LLM")


# ===================================================================
# verify — full flow
# ===================================================================


class TestVerify:
    def _setup_consented(self, c: Compliance, tmp_path) -> None:
        """Create LICENSE.txt + license_meta.json for every model, fully consented."""
        for section, chk in c.models.items():
            subdir = c._license_subdir_for(chk, section)
            lic_dir = os.path.join(str(tmp_path), subdir)
            os.makedirs(lic_dir, exist_ok=True)

            license_text = f"License for {section}"
            lic_path = os.path.join(lic_dir, "LICENSE.txt")
            with open(lic_path, "w") as f:
                f.write(license_text)

            license_hash = c._compute_text_hash(license_text)
            chk_str = json.dumps(chk, sort_keys=True)
            config_hash = c.fileUtils.compute_hash(chk_str)

            meta = {
                "license_hash": license_hash,
                "config_hash": config_hash,
                "consent": True,
            }
            meta_path = os.path.join(lic_dir, "license_meta.json")
            c._save_meta(meta_path, meta)

    def test_all_consented_passes(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        # Stub _check_models_config_hash to avoid needing real module files
        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)

        self._setup_consented(c, tmp_path)
        # Should not raise
        c.verify()

    def test_switch_to_previously_consented_model_needs_no_prompt(
        self, tmp_path, monkeypatch
    ):
        models = {
            "a": {"_EMBED": {"MODEL": "org/a", "LICENSE_URL": "https://x/a"}},
            "b": {"_EMBED": {"MODEL": "org/b", "LICENSE_URL": "https://x/b"}},
        }
        both = _build_compliance(models_override=models, active_models={})
        both.models = {
            "a._EMBED": models["a"]["_EMBED"],
            "b._EMBED": models["b"]["_EMBED"],
        }
        self._setup_consented(both, tmp_path)

        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "b"}
        )
        c.base_dir = str(tmp_path)
        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)
        monkeypatch.setattr(c, "_update_licenses", lambda: pytest.fail("prompted"))
        c.verify()

    def test_switch_to_unconsented_model_requires_consent(self, tmp_path, monkeypatch):
        models = {
            "a": {"_EMBED": {"MODEL": "org/a", "LICENSE_URL": "https://x/a"}},
            "b": {"_EMBED": {"MODEL": "org/b", "LICENSE_URL": "https://x/b"}},
        }
        only_a = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "a"}
        )
        self._setup_consented(only_a, tmp_path)

        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "b"}
        )
        c.base_dir = str(tmp_path)
        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)
        update_called: list[bool] = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))
        c.verify()
        assert update_called == [True]

    def test_inactive_model_config_change_needs_no_prompt(self, tmp_path, monkeypatch):
        models = {
            "a": {"_EMBED": {"MODEL": "org/a", "LICENSE_URL": "https://x/a"}},
            "b": {"_EMBED": {"MODEL": "org/b", "LICENSE_URL": "https://x/b"}},
        }
        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "a"}
        )
        self._setup_consented(c, tmp_path)
        models["b"]["_EMBED"]["REVISION"] = "changed"
        c.base_dir = str(tmp_path)
        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)
        monkeypatch.setattr(c, "_update_licenses", lambda: pytest.fail("prompted"))
        c.verify()

    def test_active_model_config_change_requires_consent(self, tmp_path, monkeypatch):
        models = {"a": {"_EMBED": {"MODEL": "org/a", "LICENSE_URL": "https://x/a"}}}
        c = _build_compliance(
            models_override=models, active_models={"_ACTIVE_EMBED": "a"}
        )
        self._setup_consented(c, tmp_path)
        models["a"]["_EMBED"]["REVISION"] = "changed"
        c.base_dir = str(tmp_path)
        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)
        update_called: list[bool] = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))
        c.verify()
        assert update_called == [True]

    def test_missing_license_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)

        update_called = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))

        # No license files → should call _update_licenses
        c.verify()
        assert len(update_called) == 1

    def test_consent_false_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)

        self._setup_consented(c, tmp_path)

        # Tamper with first model's consent
        first_section = next(iter(c.models))
        chk = c.models[first_section]
        subdir = c._license_subdir_for(chk, first_section)
        meta_path = os.path.join(str(tmp_path), subdir, "license_meta.json")
        meta = c._load_meta(meta_path)
        meta["consent"] = False
        c._save_meta(meta_path, meta)

        update_called = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))
        c.verify()
        assert len(update_called) == 1

    def test_hash_mismatch_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)

        self._setup_consented(c, tmp_path)

        # Tamper with first model's license text
        first_section = next(iter(c.models))
        chk = c.models[first_section]
        subdir = c._license_subdir_for(chk, first_section)
        lic_path = os.path.join(str(tmp_path), subdir, "LICENSE.txt")
        with open(lic_path, "w") as f:
            f.write("TAMPERED LICENSE TEXT")

        update_called = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))
        c.verify()
        assert len(update_called) == 1

    def test_config_hash_mismatch_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)
        monkeypatch.setattr(c, "_verify_argos_consent", lambda: None)
        monkeypatch.setattr(c, "_verify_spacy_consent", lambda: None)

        self._setup_consented(c, tmp_path)

        # Tamper with first model's config_hash in meta
        first_section = next(iter(c.models))
        chk = c.models[first_section]
        subdir = c._license_subdir_for(chk, first_section)
        meta_path = os.path.join(str(tmp_path), subdir, "license_meta.json")
        meta = c._load_meta(meta_path)
        meta["config_hash"] = "stale_config_hash"
        c._save_meta(meta_path, meta)

        update_called = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))
        c.verify()
        assert len(update_called) == 1


# ===================================================================
# Argos / spaCy consent gates
# ===================================================================


class TestArgosConsentGate:
    def test_verify_argos_consent_skips_when_no_active_pairs(self, monkeypatch):
        c = _build_compliance(
            cfg_overrides={
                "_ABSOLUTE_PATH": "D:/RAG-LCC",
            }
        )
        calls: list[tuple[str, Any]] = []

        class StubArgosDownloader:
            def __init__(self, project_root, languages):
                calls.append(("init", project_root, list(languages)))

            def assert_consent_current_or_raise(self):
                calls.append(("assert",))

        monkeypatch.setattr(
            "Compliance.Compliance.ArgosDownloader", StubArgosDownloader
        )
        monkeypatch.setattr(
            "Compliance.Compliance.get_active_argos_pairs", lambda _cfg: []
        )

        c._verify_argos_consent()

        assert calls == []

    def test_verify_argos_consent_raises_when_missing(self, monkeypatch):
        c = _build_compliance(
            cfg_overrides={
                "_ABSOLUTE_PATH": "D:/RAG-LCC",
            }
        )

        class StubArgosDownloader:
            def __init__(self, project_root, languages):
                _ = (project_root, languages)

            def assert_consent_current_or_raise(self):
                raise ArgosConsentMissingError("missing argos consent")

        monkeypatch.setattr(
            "Compliance.Compliance.ArgosDownloader", StubArgosDownloader
        )
        monkeypatch.setattr(
            "Compliance.Compliance.get_active_argos_pairs",
            lambda _cfg: [("en", "de")],
        )

        with pytest.raises(ArgosConsentMissingError):
            c._verify_argos_consent()


class TestSpacyConsentGate:
    def test_configured_spacy_models_dedup(self):
        c = _build_compliance(
            cfg_overrides={
                "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
                "_REGEX_INDEX.spacy_model": "en_core_web_sm",
            }
        )
        assert c._configured_spacy_models() == ["en_core_web_sm"]

    def test_configured_spacy_models_include_language_maps(self):
        c = _build_compliance(
            cfg_overrides={
                "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES": [
                    "en",
                    "de",
                    "es",
                    "fr",
                    "it",
                ],
                "_BM25_INDEX.spacy_model": "en_core_web_sm",
                "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
                "_REGEX_INDEX.spacy_model": "en_core_web_sm",
                "_BM25_INDEX.spacy_models_by_language": {
                    "es": "es_core_news_sm",
                },
                "_GRAPH_INDEX.spacy_models_by_language": {
                    "de": "de_core_news_sm",
                    "fr": "fr_core_news_sm",
                },
                "_REGEX_INDEX.spacy_models_by_language": {
                    "de": "de_core_news_sm",
                    "it": "it_core_news_sm",
                },
            }
        )
        assert c._configured_spacy_models() == [
            "de_core_news_sm",
            "en_core_web_sm",
            "es_core_news_sm",
            "fr_core_news_sm",
            "it_core_news_sm",
        ]

    def test_configured_spacy_models_defaults_when_missing(self):
        c = _build_compliance(
            cfg_overrides={
                "_GRAPH_INDEX.spacy_model": "",
                "_REGEX_INDEX.spacy_model": "",
            }
        )
        assert c._configured_spacy_models() == ["en_core_web_sm"]

    def test_verify_spacy_consent_calls_assert(self, monkeypatch):
        c = _build_compliance(
            cfg_overrides={
                "_ABSOLUTE_PATH": "D:/RAG-LCC",
                "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
                "_REGEX_INDEX.spacy_model": "en_core_web_sm",
            }
        )
        calls: list[tuple[str, Any]] = []

        class StubSpacyDownloader:
            def __init__(self, project_root, models):
                calls.append(("init", project_root, list(models)))

            def assert_consent_current_or_raise(self):
                calls.append(("assert",))

        monkeypatch.setattr(
            "Compliance.Compliance.SpacyDownloader", StubSpacyDownloader
        )

        c._verify_spacy_consent()

        assert ("assert",) in calls

    def test_verify_spacy_consent_raises_when_missing(self, monkeypatch):
        c = _build_compliance(
            cfg_overrides={
                "_ABSOLUTE_PATH": "D:/RAG-LCC",
                "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
                "_REGEX_INDEX.spacy_model": "en_core_web_sm",
            }
        )

        class StubSpacyDownloader:
            def __init__(self, project_root, models):
                _ = (project_root, models)

            def assert_consent_current_or_raise(self):
                raise SpacyConsentMissingError("missing spacy consent")

        monkeypatch.setattr(
            "Compliance.Compliance.SpacyDownloader", StubSpacyDownloader
        )

        with pytest.raises(SpacyConsentMissingError):
            c._verify_spacy_consent()


# ===================================================================
# _check_models_config_hash
# ===================================================================


class TestCheckModelsConfigHash:
    def test_matching_hashes_pass(self, monkeypatch):
        c = _build_compliance()

        # Stub the module objects to have __name__ and __file__
        fake_models = types.ModuleType("Config_Models")
        fake_models.__file__ = "fake_models.py"
        fake_banned_detection = types.ModuleType("Config_Banned_Detection")
        fake_banned_detection.__file__ = "fake_banned_detection.py"
        fake_banned_content = types.ModuleType("Config_Banned_Content")
        fake_banned_content.__file__ = "fake_banned_content.py"
        fake_banned_prompts = types.ModuleType("Config_Banned_Prompts")
        fake_banned_prompts.__file__ = "fake_banned_prompts.py"

        monkeypatch.setattr("Compliance.Compliance.Config_Models", fake_models)
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Detection", fake_banned_detection
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Content", fake_banned_content
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Prompts", fake_banned_prompts
        )

        # fileUtils.hash_module returns "fakehash" which matches cfg
        c._check_models_config_hash()  # Should not raise

    def test_mismatched_hash_raises(self, monkeypatch):
        c = _build_compliance()

        fake_models = types.ModuleType("Config_Models")
        fake_models.__file__ = "fake_models.py"
        fake_banned_detection = types.ModuleType("Config_Banned_Detection")
        fake_banned_detection.__file__ = "fake_banned_detection.py"
        fake_banned_content = types.ModuleType("Config_Banned_Content")
        fake_banned_content.__file__ = "fake_banned_content.py"
        fake_banned_prompts = types.ModuleType("Config_Banned_Prompts")
        fake_banned_prompts.__file__ = "fake_banned_prompts.py"

        monkeypatch.setattr("Compliance.Compliance.Config_Models", fake_models)
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Detection", fake_banned_detection
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Content", fake_banned_content
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Prompts", fake_banned_prompts
        )

        # Make hash_module return a non-matching hash
        c.fileUtils.hash_module = lambda mod, algo="sha256": "different_hash"

        with pytest.raises(ComplianceViolationError):
            c._check_models_config_hash()

    @pytest.mark.parametrize(
        ("platform_name", "expected_fragment", "unexpected_fragment"),
        [
            (
                "posix",
                "python ./src/Scripts/RecalcConfigHashes.py",
                "python .\\src\\Scripts\\RecalcConfigHashes.py",
            ),
            (
                "nt",
                "python .\\src\\Scripts\\RecalcConfigHashes.py",
                "python ./src/Scripts/RecalcConfigHashes.py",
            ),
        ],
    )
    def test_mismatch_hint_uses_platform_specific_recalc_command(
        self,
        monkeypatch,
        platform_name,
        expected_fragment,
        unexpected_fragment,
    ):
        c = _build_compliance()

        fake_models = types.ModuleType("Config_Models")
        fake_models.__file__ = "fake_models.py"
        fake_banned_detection = types.ModuleType("Config_Banned_Detection")
        fake_banned_detection.__file__ = "fake_banned_detection.py"
        fake_banned_content = types.ModuleType("Config_Banned_Content")
        fake_banned_content.__file__ = "fake_banned_content.py"
        fake_banned_prompts = types.ModuleType("Config_Banned_Prompts")
        fake_banned_prompts.__file__ = "fake_banned_prompts.py"

        monkeypatch.setattr("Compliance.Compliance.Config_Models", fake_models)
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Detection", fake_banned_detection
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Content", fake_banned_content
        )
        monkeypatch.setattr(
            "Compliance.Compliance.Config_Banned_Prompts", fake_banned_prompts
        )
        monkeypatch.setattr("Compliance.Compliance.os.name", platform_name)

        c.fileUtils.hash_module = lambda mod, algo="sha256": "different_hash"

        with pytest.raises(ComplianceViolationError):
            c._check_models_config_hash()

        messages = [
            msg[2]
            for msg in c.pretty.messages
            if len(msg) >= 3 and msg[0] == "I" and msg[1] == "Compliance"
        ]
        assert any(expected_fragment in message for message in messages)
        assert all(unexpected_fragment not in message for message in messages)


# ===================================================================
# _update_licenses
# ===================================================================


class TestUpdateLicenses:
    def test_calls_process_one_for_each_model(self, monkeypatch):
        c = _build_compliance()
        processed: list[str] = []

        def fake_process(models, section, identity=None):
            processed.append(section)

        monkeypatch.setattr(c, "_process_one", fake_process)

        c._update_licenses()
        assert set(processed) == set(c.models.keys())

    def test_identity_passed_to_all(self, monkeypatch):
        c = _build_compliance()
        identities: list[dict[str, Any]] = []

        def fake_process(models, section, identity=None):
            identities.append(identity)

        monkeypatch.setattr(c, "_process_one", fake_process)

        c._update_licenses()
        # All should be the same cached identity dict
        assert all(i is identities[0] for i in identities)
        assert identities[0]["accepted_by"] == "test-user"
