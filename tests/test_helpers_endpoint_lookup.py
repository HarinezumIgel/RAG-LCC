import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.Helpers import Helpers
from Configuration import Config_Models

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def write(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class StubConfig:
    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        if key == "_ACTIVE_ENDPOINT":
            return "vllm"
        return default

    def get_dict(self, key: str, default: Any = None, *, silent: bool = False) -> Any:
        if key == "_MODELS":
            return {
                "vllm": {
                    "_VLLM": {
                        "PROVIDER": "vllm",
                        "BASE_URL": "http://localhost:8000/v1/chat/completions",
                        "STREAMING_REQ": True,
                        "USE_GPU": True,
                    }
                }
            }
        return default if default is not None else {}


def _make_helpers() -> Helpers:
    return Helpers(cfg=StubConfig(), pretty=StubPrettyWriter())


def _ok() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    return resp


def _fail(*_a: Any, **_kw: Any) -> None:
    raise requests_lib.ConnectionError("connection refused")


# ---------------------------------------------------------------------------
# get_active_endpoint_args
# ---------------------------------------------------------------------------


def test_get_active_endpoint_args_uses_active_endpoint_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    helpers = _make_helpers()

    args = helpers.get_active_endpoint_args()

    assert args["PROVIDER"] == "vllm"
    assert args["BASE_URL"] == "http://localhost:8000/v1/chat/completions"
    assert args["model_name"] == ""
    assert args["friendly_name"] == ""
    assert args["local_files_only"] is True


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("ollama", "qwen2.5:14b"),
        ("vllm", "openai/qwen2_5_14b_instruct"),
    ],
)
def test_judge_model_resolves_for_active_endpoint(endpoint: str, expected: str) -> None:
    helpers = _make_helpers()
    with (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "_ACTIVE_ENDPOINT": endpoint,
                "_ACTIVE_LLM_JUDGE": Config_Models._ACTIVE_LLM_JUDGE,
            }.get(key, default),
        ),
        patch.object(helpers.cfg, "get_dict", return_value=Config_Models._MODELS),
    ):
        args = helpers.get_model_args("_ACTIVE_LLM_JUDGE")

    assert args["MODEL"] == expected
    assert args["model_name"] == expected
    assert args["LICENSE"] == "Apache-2.0"
    assert args["LICENSE_URL"] == (
        "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/resolve/main/LICENSE"
    )
    assert args["USED_BY"] == ["RAGChat", "RAGChatService"]
    assert args["TOKEN_BUDGET_RESERVED_OUTPUT"] == 256


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("ollama", "qwen2.5:1.5b"),
        ("vllm", "openai/qwen2_5_1_5b_instruct"),
    ],
)
def test_small_judge_model_spec_resolves_through_judge_selector(
    endpoint: str, expected: str
) -> None:
    helpers = _make_helpers()
    with (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "_ACTIVE_ENDPOINT": endpoint,
                "_ACTIVE_LLM_JUDGE": "qwen2_5_1_5b_instruct",
            }.get(key, default),
        ),
        patch.object(helpers.cfg, "get_dict", return_value=Config_Models._MODELS),
    ):
        args = helpers.get_model_args("_ACTIVE_LLM_JUDGE")

    assert args["MODEL"] == expected
    assert args["FRIENDLY_NAME"] == "Qwen2.5 1.5B Instruct (Small Judge)"
    assert args["LICENSE"] == "Apache-2.0"
    assert args["USED_BY"] == ["RAGChat", "RAGChatService"]


@pytest.mark.parametrize(
    ("endpoint", "model"),
    [
        ("ollama", "snowflake/snowflake-arctic-embed-l-v2.0"),
        ("vllm", "snowflake-embedder"),
    ],
)
def test_embedder_model_follows_endpoint_and_keeps_hf_id(
    endpoint: str, model: str
) -> None:
    helpers = _make_helpers()
    with (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "_ACTIVE_ENDPOINT": endpoint,
                "_ACTIVE_EMBED": "snowflake",
            }.get(key, default),
        ),
        patch.object(helpers.cfg, "get_dict", return_value=Config_Models._MODELS),
    ):
        args = helpers.get_model_args("_ACTIVE_EMBED")

    assert args["MODEL"] == model
    assert args["model_name"] == model
    assert args["hf_model_name"] == "snowflake/snowflake-arctic-embed-l-v2.0"
    assert "VLLM" not in args and "OLLAMA" not in args


def _models_with_slot_overrides() -> dict[str, Any]:
    return {
        "vllm": {
            "_VLLM": {
                "BASE_URL": "http://generic/v1",
                "REQUEST_TIMEOUT": 120.0,
                "API_KEY": "g",
            },
        },
        "m": {
            "_LLM": {"VLLM": {"MODEL": "m_vllm", "REQUEST_TIMEOUT": 300.0}},
            "_EMBED": {
                "MODEL": "hf/m",
                "VLLM": {
                    "MODEL": "m-embed",
                    "BATCH_SIZE": 8,
                    "BASE_URL": "http://embed/v1",
                },
            },
        },
    }


def _patched(helpers: Helpers):
    return (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "_ACTIVE_ENDPOINT": "vllm",
                "_ACTIVE_LLM": "m",
                "_ACTIVE_EMBED": "m",
            }.get(key, default),
        ),
        patch.object(
            helpers.cfg, "get_dict", return_value=_models_with_slot_overrides()
        ),
    )


def test_model_slot_overrides_generic_endpoint_values() -> None:
    helpers = _make_helpers()
    p1, p2 = _patched(helpers)
    with p1, p2:
        generic = helpers.get_active_endpoint_args()
        llm = helpers.get_active_endpoint_args("_ACTIVE_LLM")
        embed = helpers.get_active_endpoint_args("_ACTIVE_EMBED")

    assert generic["REQUEST_TIMEOUT"] == 120.0
    assert llm["REQUEST_TIMEOUT"] == 300.0
    assert llm["BASE_URL"] == "http://generic/v1" and llm["API_KEY"] == "g"
    assert "MODEL" not in llm
    assert embed["BASE_URL"] == "http://embed/v1" and embed["BATCH_SIZE"] == 8


# ---------------------------------------------------------------------------
# find_provider_url
# ---------------------------------------------------------------------------

_PATCH = "Helpers.Helpers.requests.get"


class TestFindProviderUrl:
    """Unit tests for Helpers.find_provider_url."""

    def test_configured_url_works_first_try(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()

        def fake_get(url: str, **kw: Any) -> MagicMock:
            if url == "http://myhost:8000/v1/models":
                return _ok()
            raise requests_lib.ConnectionError("not this one")

        with patch(_PATCH, side_effect=fake_get):
            result = h.find_provider_url(
                base_url="http://myhost:8000/v1/chat/completions",
                probe_path="/v1/models",
                generate_path="/v1/chat/completions",
                default_port=8000,
                headers={},
                label="VLLM",
            )

        assert result == "http://myhost:8000/v1/chat/completions"

    def test_falls_back_to_localhost_same_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()

        def fake_get(url: str, **kw: Any) -> MagicMock:
            if url == "http://localhost:11434/api/tags":
                return _ok()
            raise requests_lib.ConnectionError("refused")

        with patch(_PATCH, side_effect=fake_get):
            result = h.find_provider_url(
                base_url="http://host.docker.internal:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        assert result == "http://localhost:11434/api/generate"

    def test_falls_back_to_default_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When configured port differs, default-port fallback candidates are tried."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()

        def fake_get(url: str, **kw: Any) -> MagicMock:
            if url == "http://localhost:11434/api/tags":
                return _ok()
            raise requests_lib.ConnectionError("refused")

        with patch(_PATCH, side_effect=fake_get):
            result = h.find_provider_url(
                base_url="http://host.docker.internal:9999/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        assert result == "http://localhost:11434/api/generate"

    def test_returns_none_when_all_candidates_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()

        with patch(_PATCH, side_effect=_fail):
            result = h.find_provider_url(
                base_url="http://host.docker.internal:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        assert result is None

    def test_no_duplicate_when_configured_host_is_localhost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()
        tried: list[str] = []

        def fake_get(url: str, **kw: Any) -> None:
            tried.append(url)
            raise requests_lib.ConnectionError("refused")

        with patch(_PATCH, side_effect=fake_get):
            h.find_provider_url(
                base_url="http://localhost:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        assert tried.count("http://localhost:11434/api/tags") == 1

    def test_max_six_candidates_tried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()
        tried: list[str] = []

        def fake_get(url: str, **kw: Any) -> None:
            tried.append(url)
            raise requests_lib.ConnectionError("refused")

        with patch(_PATCH, side_effect=fake_get):
            h.find_provider_url(
                base_url="http://myhost:9999/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        assert len(tried) <= 6

    def test_headers_forwarded_to_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()
        captured: list[dict] = []

        def fake_get(url: str, headers: dict | None = None, **kw: Any) -> MagicMock:
            captured.append(headers or {})
            return _ok()

        custom = {"Authorization": "Bearer tok", "Content-Type": "application/json"}
        with patch(_PATCH, side_effect=fake_get):
            h.find_provider_url(
                base_url="http://localhost:8000/v1/chat/completions",
                probe_path="/v1/models",
                generate_path="/v1/chat/completions",
                default_port=8000,
                headers=custom,
                label="VLLM",
            )

        assert captured[0] == custom

    def test_pretty_writer_emits_ok_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        pretty = StubPrettyWriter()
        h = Helpers(cfg=StubConfig(), pretty=pretty)

        with patch(_PATCH, return_value=_ok()):
            h.find_provider_url(
                base_url="http://localhost:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        messages = [args[0][2] for args in pretty.calls]
        assert any("OK" in m for m in messages)

    def test_pretty_writer_emits_failed_on_all_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        pretty = StubPrettyWriter()
        h = Helpers(cfg=StubConfig(), pretty=pretty)

        with patch(_PATCH, side_effect=_fail):
            h.find_provider_url(
                base_url="http://localhost:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        messages = [args[0][2] for args in pretty.calls]
        # Filter to only probe result messages (not "Trying next:" informational messages)
        probe_messages = [m for m in messages if m.startswith("Probing")]
        assert all("failed" in m for m in probe_messages)
        assert len(probe_messages) >= 1

    def test_configured_port_equals_default_no_extra_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When configured port == default_port no extra candidates are added."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        h = _make_helpers()
        tried: list[str] = []

        def fake_get(url: str, **kw: Any) -> None:
            tried.append(url)
            raise requests_lib.ConnectionError("refused")

        with patch(_PATCH, side_effect=fake_get):
            h.find_provider_url(
                base_url="http://otherhost:11434/api/generate",
                probe_path="/api/tags",
                generate_path="/api/generate",
                default_port=11434,
                headers={},
                label="OLLAMA",
            )

        # Candidates: otherhost:11434, localhost:11434, 127.0.0.1:11434, host.docker.internal:11434
        # No extra candidates for default_port (same as configured)
        assert len(tried) == 4


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("ollama", "hf_slot"),
        ("vllm", "hf_role"),
    ],
)
def test_model_hf_api_key_supersedes_global(endpoint: str, expected: str) -> None:
    helpers = _make_helpers()
    models = {
        "m": {
            "_LLM_REWRITE_PROMPT": {
                "HF_API_KEY": "hf_role",
                "OLLAMA": {"MODEL": "m", "HF_API_KEY": "hf_slot"},
            }
        }
    }
    with (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "_ACTIVE_ENDPOINT": endpoint,
                "_ACTIVE_LLM_REWRITE_PROMPT": "m",
                "_HF_API_KEY": "hf_global",
            }.get(key, default),
        ),
        patch.object(helpers.cfg, "get_dict", return_value=models),
    ):
        assert (
            helpers.get_model_args("_ACTIVE_LLM_REWRITE_PROMPT")["hf_api_key"]
            == expected
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "Test_embedder_snowflake"),  # config COLLECTION / --collection
        ("MyDocs", "MyDocs_embedder_snowflake"),
        ("Test_embedder_snowflake", "Test_embedder_snowflake"),
        ("Test_embedder_bge_m3", "Test_embedder_bge_m3"),  # other embedder: left as is
        (
            "Test_snowflake",
            "Test_snowflake_embedder_snowflake",
        ),  # plain impl suffix is no embedder suffix
        ("", ""),
    ],
)
def test_embedder_collection_name(name, expected) -> None:
    helpers = _make_helpers()
    models = {
        "snowflake": {"_EMBED": {}},
        "bge_m3": {"_EMBED": {}},
        "mistral": {"_LLM": {}},
    }
    with (
        patch.object(
            helpers.cfg,
            "get_str",
            side_effect=lambda key, default="", **kw: {
                "COLLECTION": "Test",
                "_ACTIVE_EMBED": "snowflake",
            }.get(key, default),
        ),
        patch.object(helpers.cfg, "get_dict", return_value=models),
    ):
        assert helpers.embedder_collection_name(name) == expected


def test_active_embedder_collections(tmp_path) -> None:
    for d in (
        "Test_embedder_snowflake",
        "Docs_embedder_snowflake",
        "Test_embedder_bge_m3",
        "Test_embedder_snowflake_ChatContext",
        ".hidden_embedder_snowflake",
        "Legacy_snowflake",
        "Legacy",
    ):
        (tmp_path / d).mkdir()
    (tmp_path / "file_embedder_snowflake").write_text("x")
    helpers = _make_helpers()
    with patch.object(
        helpers.cfg,
        "get_str",
        side_effect=lambda key, default="", **kw: {
            "_ACTIVE_EMBED": "snowflake",
            "_CHROMA_DB_DIR": str(tmp_path),
        }.get(key, default),
    ):
        assert helpers.active_embedder_collections() == [
            "Docs_embedder_snowflake",
            "Test_embedder_snowflake",
        ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Test_embedder_snowflake", "Test"),
        ("Test_embedder_bge_m3", "Test"),
        ("Test", "Test"),
        ("Test_Second_embedder_snowflake", "Test_Second"),
        ("Test_snowflake", "Test_snowflake"),
        ("Test_embedder_mistral", "Test_embedder_mistral"),
    ],
)
def test_base_collection_name(name, expected) -> None:
    helpers = _make_helpers()
    models = {
        "snowflake": {"_EMBED": {}},
        "bge_m3": {"_EMBED": {}},
        "mistral": {"_LLM": {}},
    }
    with patch.object(helpers.cfg, "get_dict", return_value=models):
        assert helpers.base_collection_name(name) == expected


def test_history_key_is_shared_across_embedders() -> None:
    from Gui.HistoryManager import HistoryManager

    assert HistoryManager.key("Test_embedder_snowflake", "MyChat") == "Test_MyChat"
    assert HistoryManager.key("Test_embedder_bge_m3", "MyChat") == "Test_MyChat"
    assert HistoryManager.key("Test", "MyChat") == "Test_MyChat"
