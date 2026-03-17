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
from Commons.Exceptions import ComplianceViolationError, InternetConnectionDisabledError

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
            "_FRIENDLY_NAME": "TestApp",
            "_MODELS_CONFIG_HASH": "fakehash",
            "_BANNED_CONFIG_HASH": "fakehash",
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

    def get_dict(self, key, default=None):
        return self.get(key, default if default is not None else {})

    def get_list(self, key, default=None):
        val = self.get(key, default if default is not None else [])
        return (
            val if isinstance(val, list) else (default if default is not None else [])
        )

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
) -> Compliance:
    """Build a Compliance instance with all deps stubbed, bypassing __init__."""
    c = object.__new__(Compliance)

    cfg = StubConfig(cfg_overrides)
    if models_override is not None:
        cfg.set("_MODELS", models_override)

    c.cfg = cfg
    c.globalsInstance = StubGlobals()
    c.helpers = StubHelpers()
    c.sharedHelpers = StubSharedHelpers()
    c.fileUtils = StubFileUtils()
    c.logger = StubLogger()
    c.pretty = StubPrettyWriter()
    c.friendly_name = "TestApp"
    c.base_dir = "ModelGovernance/licenses"
    c.license_download = connection_mode
    c._acceptance_identity = None

    # Flatten _MODELS exactly as __init__ does
    raw = cfg.get_dict("_MODELS", {})
    c.models = {}
    for impl, roles in raw.items():
        if isinstance(roles, dict):
            for role, config in roles.items():
                if isinstance(config, dict):
                    c.models[f"{impl}.{role}"] = config

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
        _shutil.get_terminal_size = lambda *a, **kw: os.terminal_size((80, 6))
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


class TestModelFlattening:
    def test_flattens_to_impl_dot_role(self):
        c = _build_compliance()
        assert "snowflake._EMBED" in c.models
        assert "mistral._LLM" in c.models
        assert "mistral._LLM_CHK" in c.models
        assert "ollama._OLLAMA" in c.models

    def test_flat_keys_count(self):
        c = _build_compliance()
        # SAMPLE_MODELS has snowflake._EMBED, mistral._LLM, mistral._LLM_CHK, ollama._OLLAMA
        assert len(c.models) == 4

    def test_config_values_preserved(self):
        c = _build_compliance()
        embed = c.models["snowflake._EMBED"]
        assert embed["MODEL"] == "snowflake/snowflake-arctic-embed-l-v2.0"
        assert embed["LICENSE"] == "Apache-2.0"

    def test_empty_models(self):
        c = _build_compliance(models_override={})
        assert c.models == {}

    def test_non_dict_roles_skipped(self):
        c = _build_compliance(
            models_override={"impl1": "not_a_dict", "impl2": {"ROLE": {"MODEL": "x"}}}
        )
        assert "impl2.ROLE" in c.models
        assert len(c.models) == 1

    def test_non_dict_config_skipped(self):
        c = _build_compliance(
            models_override={"impl": {"ROLE1": "not_a_dict", "ROLE2": {"MODEL": "x"}}}
        )
        assert "impl.ROLE2" in c.models
        assert "impl.ROLE1" not in c.models


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
        monkeypatch.setattr("builtins.input", lambda _: "n")
        with pytest.raises(InternetConnectionDisabledError):
            c._fetch_license("https://example.com/license", "TestModel", "test/path")

    def test_offline_mode_proceeds_on_yes(self, monkeypatch):
        c = _build_compliance(connection_mode="0")
        inputs = iter(["y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        import requests as _req

        class FakeResponse:
            text = "MIT License\n..."
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *a, **kw: FakeResponse())
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

        monkeypatch.setattr(_req, "get", lambda *a, **kw: FakeResponse())
        result = c._fetch_license("https://example.com/license", "Test", "path")
        assert "Apache License" in result

    def test_network_error_raises_compliance_violation(self, monkeypatch):
        c = _build_compliance(connection_mode="1")

        import requests as _req

        def fail_get(*a, **kw):
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
        monkeypatch.setattr("builtins.input", lambda _: "y")

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
        monkeypatch.setattr("builtins.input", lambda _: "n")

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
        }
        meta_path = os.path.join(lic_dir, "license_meta.json")
        c._save_meta(meta_path, meta)

        # Will enter live-fetch path → mock fetch + accept
        import requests as _req

        class FakeResponse:
            text = "Apache License Version 2.0"
            status_code = 200

            def raise_for_status(self):
                pass

        monkeypatch.setattr(_req, "get", lambda *a, **kw: FakeResponse())
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "Compliance.Compliance.Compliance._get_tls_fingerprint",
            lambda self, url: "ab" * 32,
        )

        c._process_one(c.models, section)


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

        monkeypatch.setattr(_req, "get", lambda *a, **kw: FakeResponse())
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "Compliance.Compliance.Compliance._get_tls_fingerprint",
            lambda self, url: "ab" * 32,
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

        monkeypatch.setattr(_req, "get", lambda *a, **kw: FakeResponse())
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, section)


# ===================================================================
# _process_one — missing LICENSE_URL
# ===================================================================


class TestProcessOneMissingUrl:
    def test_missing_license_url_raises(self, tmp_path):
        c = _build_compliance(
            models_override={"stub": {"ROLE": {"MODEL": "x", "FRIENDLY_NAME": "X"}}}
        )
        c.base_dir = str(tmp_path)

        with pytest.raises(ComplianceViolationError):
            c._process_one(c.models, "stub.ROLE")


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

        self._setup_consented(c, tmp_path)
        # Should not raise
        c.verify()

    def test_missing_license_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)

        update_called = []
        monkeypatch.setattr(c, "_update_licenses", lambda: update_called.append(True))

        # No license files → should call _update_licenses
        c.verify()
        assert len(update_called) == 1

    def test_consent_false_triggers_update(self, tmp_path, monkeypatch):
        c = _build_compliance()
        c.base_dir = str(tmp_path)

        monkeypatch.setattr(c, "_check_models_config_hash", lambda: None)

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
# _check_models_config_hash
# ===================================================================


class TestCheckModelsConfigHash:
    def test_matching_hashes_pass(self, monkeypatch):
        c = _build_compliance()

        # Stub the module objects to have __name__ and __file__
        fake_models = types.ModuleType("Config_Models")
        fake_models.__file__ = "fake_models.py"
        fake_banned = types.ModuleType("Config_Banned")
        fake_banned.__file__ = "fake_banned.py"

        monkeypatch.setattr("Compliance.Compliance.Config_Models", fake_models)
        monkeypatch.setattr("Compliance.Compliance.Config_Banned", fake_banned)

        # fileUtils.hash_module returns "fakehash" which matches cfg
        c._check_models_config_hash()  # Should not raise

    def test_mismatched_hash_raises(self, monkeypatch):
        c = _build_compliance()

        fake_models = types.ModuleType("Config_Models")
        fake_models.__file__ = "fake_models.py"
        fake_banned = types.ModuleType("Config_Banned")
        fake_banned.__file__ = "fake_banned.py"

        monkeypatch.setattr("Compliance.Compliance.Config_Models", fake_models)
        monkeypatch.setattr("Compliance.Compliance.Config_Banned", fake_banned)

        # Make hash_module return a non-matching hash
        c.fileUtils.hash_module = lambda mod, algo="sha256": "different_hash"

        with pytest.raises(ComplianceViolationError):
            c._check_models_config_hash()


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
