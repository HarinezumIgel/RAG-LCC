import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.Helpers import Helpers

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
