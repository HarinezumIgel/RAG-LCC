# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""
Tests for AI.LLMCaller:

  - redact()
  - make_on_chunk() — closure factory
  - call_llm() — ollama_options dict passthrough, token budget resolution,
    streaming, label detection, error handling, top_level_params merge
  - get_model_context_limit() — /api/show integration

All network calls are mocked via monkeypatch so no real Ollama is required.
"""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Stubs — replace heavy dependencies so LLMCaller can be instantiated
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self, *a, **k):
        self.calls: list[tuple] = []

    def write(self, *a, **k):
        self.calls.append((a, k))
        return ""


class StubConfig:
    def __init__(self, overrides: dict[str, Any] | None = None):
        self._data: dict[str, Any] = {
            "REQUEST_TIMEOUT": 30,
            "DEBUG_LEVEL": 0,
            "_MODELS.ollama._OLLAMA.BASE_URL": "http://localhost:11434/api/generate",
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def get_str(self, key, default="") -> str:
        return str(self._data.get(key, default))

    def get_int(self, key, default=0) -> int:
        return int(self._data.get(key, default))

    def get_bool(self, key, default=False) -> bool:
        return bool(self._data.get(key, default))

    def get_float(self, key, default=0.0) -> float:
        return float(self._data.get(key, default))

    def set(self, key, value):
        self._data[key] = value


class StubHelpers:
    def get_model_args(self, key):
        return {
            "BASE_URL": "http://localhost:11434/api/generate",
            "STREAMING_REQ": False,
            "USE_GPU": True,
        }


class StubTokenBudget:
    def __init__(self, ctx=8192, output=2048):
        self._ctx = ctx
        self._output = output

    def get_context_limit(self, model=None):
        return self._ctx

    def compute_dynamic_max_tokens(self, prompt, model=None, silent=False):
        return self._output

    def count_tokens_approx(self, text):
        return len(text.split())


class StubAIHelpers:
    pass


class StubGlobals:
    def get_logger(self):
        return None


def _make_caller(cfg_overrides=None, token_budget=None):
    """Build an LLMCaller with all deps replaced by stubs."""
    from AI.LLMCaller import LLMCaller

    caller = LLMCaller.__new__(LLMCaller)
    caller.cfg = StubConfig(cfg_overrides)
    caller.pretty = StubPrettyWriter()
    caller.helpers = StubHelpers()
    caller.aiHelpers = StubAIHelpers()
    # Patch _resolve_token_budget to use our stub
    tb = token_budget or StubTokenBudget()

    def _resolve(model, prompt, max_out, ctx_size):
        resolved_ctx = ctx_size if ctx_size is not None else tb.get_context_limit(model)
        resolved_out = tb.compute_dynamic_max_tokens(prompt, model)
        return resolved_out, resolved_ctx

    caller._resolve_token_budget = _resolve
    return caller


def _fake_response(lines: list[str], status_code=200):
    """Create a mock requests.Response that yields the given lines."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": "application/json"}
    resp.raise_for_status = MagicMock()
    resp.iter_lines = MagicMock(return_value=[line.encode("utf-8") for line in lines])
    return resp


# ===========================================================================
# redact()
# ===========================================================================


class TestRedact:
    def test_redacts_default_keys(self):
        caller = _make_caller()
        result = caller.redact({"authorization": "Bearer secret", "data": "ok"})
        assert "<REDACTED>" in result
        assert "Bearer secret" not in result

    def test_redacts_custom_keys(self):
        caller = _make_caller()
        result = caller.redact({"password": "hunter2"}, keys=("password",))
        assert "<REDACTED>" in result
        assert "hunter2" not in result

    def test_preserves_non_sensitive(self):
        caller = _make_caller()
        result = caller.redact({"model": "mistral:7b", "prompt": "hello"})
        assert "mistral:7b" in result
        assert "hello" in result


# ===========================================================================
# make_on_chunk()
# ===========================================================================


class TestMakeOnChunk:
    def test_returns_callable(self):
        caller = _make_caller()
        opts = {"temperature": 0.1, "top_k": 40}
        handler = caller.make_on_chunk(opts)
        assert callable(handler)

    def test_callback_accepts_chunk(self):
        caller = _make_caller()
        handler = caller.make_on_chunk({"temperature": 0.1})
        # Should not raise
        handler({"thinking": "", "response": "hello", "raw": "{}"})

    def test_closure_captures_options(self):
        caller = _make_caller()
        opts = {"temperature": 0.5, "top_k": 20}
        handler = caller.make_on_chunk(opts)
        # The closure should reference the options dict
        found = False
        for cell in handler.__closure__ or []:
            if cell.cell_contents is opts:
                found = True
        assert found, "make_on_chunk closure should capture ollama_options"


# ===========================================================================
# call_llm() — ollama_options dict handling
# ===========================================================================


class TestCallLlmOptions:
    """Verify the unified ollama_options dict is correctly assembled into the payload."""

    def _capture_post(self, caller, lines=None, **call_kwargs):
        """Call call_llm and capture the payload sent to requests.post."""
        if lines is None:
            lines = [json.dumps({"response": "hi", "done": True})]

        captured = {}

        def fake_post(url, json=None, headers=None, stream=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["stream"] = stream
            return _fake_response(lines)

        with patch("AI.LLMCaller.requests.post", side_effect=fake_post):
            result = caller.call_llm(**call_kwargs)

        return captured, result

    def test_sampling_params_in_options(self):
        caller = _make_caller()
        opts = {
            "temperature": 0.3,
            "top_k": 50,
            "top_p": 0.9,
            "num_predict": 512,
            "num_ctx": 4096,
        }
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        payload_opts = captured["json"]["options"]
        assert payload_opts["temperature"] == 0.3
        assert payload_opts["top_k"] == 50
        assert payload_opts["top_p"] == 0.9

    def test_num_predict_resolved_by_budget(self):
        """num_predict should be overwritten by _resolve_token_budget (stub returns 2048)."""
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 999, "num_ctx": 4096}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        assert (
            captured["json"]["options"]["num_predict"] == 2048
        )  # from StubTokenBudget

    def test_num_ctx_resolved_by_budget_when_none(self):
        """When num_ctx is not in options, budget system fills it in."""
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 512}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        assert (
            captured["json"]["options"]["num_ctx"] == 8192
        )  # from StubTokenBudget default

    def test_num_ctx_respected_when_provided(self):
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 512, "num_ctx": 2048}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        assert captured["json"]["options"]["num_ctx"] == 2048

    def test_extra_options_pass_through(self):
        """Custom Ollama options like seed, mirostat should pass through untouched."""
        caller = _make_caller()
        opts = {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 4096,
            "seed": 42,
            "mirostat": 2,
            "repeat_penalty": 1.1,
        }
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        payload_opts = captured["json"]["options"]
        assert payload_opts["seed"] == 42
        assert payload_opts["mirostat"] == 2
        assert payload_opts["repeat_penalty"] == 1.1

    def test_num_gpu_zero_passes_through(self):
        """num_gpu=0 (CPU mode) should appear in options."""
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096, "num_gpu": 0}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
        )
        assert captured["json"]["options"]["num_gpu"] == 0

    def test_top_level_params_merged_into_payload(self):
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello",
            ollama_options=opts,
            top_level_params={"think": True, "keep_alive": "5m"},
        )
        assert captured["json"]["think"] is True
        assert captured["json"]["keep_alive"] == "5m"
        # top-level params should NOT be inside options
        assert "think" not in captured["json"]["options"]

    def test_payload_structure(self):
        caller = _make_caller()
        opts = {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096}
        captured, _ = self._capture_post(
            caller,
            model="mistral:7b",
            prompt="hello world",
            ollama_options=opts,
        )
        payload = captured["json"]
        assert payload["model"] == "mistral:7b"
        assert payload["prompt"] == "hello world"
        assert "stream" in payload
        assert "options" in payload


# ===========================================================================
# call_llm() — streaming and content accumulation
# ===========================================================================


class TestCallLlmStreaming:
    def _call(self, caller, lines, **kwargs):
        defaults = {
            "model": "mistral:7b",
            "prompt": "test",
            "ollama_options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
        }
        defaults.update(kwargs)

        with patch("AI.LLMCaller.requests.post", return_value=_fake_response(lines)):
            return caller.call_llm(**defaults)

    def test_single_response_chunk(self):
        caller = _make_caller()
        lines = [json.dumps({"response": "Hello world", "done": True})]
        result = self._call(caller, lines)
        assert result["content"] == "Hello world"
        assert "error" not in result

    def test_multi_chunk_accumulation(self):
        caller = _make_caller()
        lines = [
            json.dumps({"response": "Hello "}),
            json.dumps({"response": "world", "done": True}),
        ]
        result = self._call(caller, lines)
        assert result["content"] == "Hello world"

    def test_done_signal_stops_reading(self):
        caller = _make_caller()
        lines = [
            json.dumps({"response": "first", "done": True}),
            json.dumps({"response": " should not appear"}),
        ]
        result = self._call(caller, lines)
        assert "should not appear" not in result["content"]

    def test_non_json_line_accumulated_as_raw(self):
        caller = _make_caller()
        lines = ["plain text response"]
        result = self._call(caller, lines)
        assert "plain text response" in result["content"]

    def test_on_chunk_callback_invoked(self):
        caller = _make_caller()
        chunks_received: list[dict] = []

        def on_chunk(chunk):
            chunks_received.append(chunk)

        lines = [json.dumps({"response": "data", "done": True})]
        self._call(caller, lines, on_chunk=on_chunk)
        assert len(chunks_received) >= 1
        assert any("data" in c.get("content", "") for c in chunks_received)

    def test_message_content_extraction(self):
        """Ollama chat format uses message.content instead of response."""
        caller = _make_caller()
        lines = [json.dumps({"message": {"content": "from chat"}, "done": True})]
        result = self._call(caller, lines)
        assert result["content"] == "from chat"

    def test_raw_accumulated(self):
        caller = _make_caller()
        lines = [json.dumps({"response": "hi", "done": True})]
        result = self._call(caller, lines)
        assert len(result["raw"]) > 0

    def test_result_keys(self):
        caller = _make_caller()
        lines = [json.dumps({"response": "hi", "done": True})]
        result = self._call(caller, lines)
        assert "content" in result
        assert "thinking" in result
        assert "raw" in result


# ===========================================================================
# call_llm() — classifier label detection
# ===========================================================================


class TestCallLlmLabelDetection:
    def _call(self, caller, lines):
        with patch("AI.LLMCaller.requests.post", return_value=_fake_response(lines)):
            return caller.call_llm(
                model="mistral:7b",
                prompt="classify",
                ollama_options={
                    "temperature": 0.0,
                    "num_predict": 100,
                    "num_ctx": 4096,
                },
            )

    def test_label_s6_detected(self):
        caller = _make_caller()
        lines = [json.dumps({"response": "S6", "done": False})]
        result = self._call(caller, lines)
        assert result["content"] == "S6"

    def test_label_s_space_3_normalized(self):
        caller = _make_caller()
        lines = [json.dumps({"response": "S 3", "done": False})]
        result = self._call(caller, lines)
        assert result["content"] == "S3"

    def test_non_label_not_short_circuited(self):
        caller = _make_caller()
        lines = [
            json.dumps(
                {"response": "This is a full answer about hedgehogs.", "done": True}
            ),
        ]
        result = self._call(caller, lines)
        assert "hedgehogs" in result["content"]


# ===========================================================================
# call_llm() — error handling
# ===========================================================================


class TestCallLlmErrors:
    def test_request_exception_returns_error(self):
        caller = _make_caller()
        import requests as req_mod

        with patch(
            "AI.LLMCaller.requests.post",
            side_effect=req_mod.exceptions.ConnectionError("refused"),
        ):
            result = caller.call_llm(
                model="mistral:7b",
                prompt="test",
                ollama_options={
                    "temperature": 0.1,
                    "num_predict": 512,
                    "num_ctx": 4096,
                },
            )
        assert "error" in result
        assert "refused" in result["error"]

    def test_timeout_returns_error(self):
        caller = _make_caller()
        import requests as req_mod

        with patch(
            "AI.LLMCaller.requests.post",
            side_effect=req_mod.exceptions.Timeout("timed out"),
        ):
            result = caller.call_llm(
                model="mistral:7b",
                prompt="test",
                ollama_options={
                    "temperature": 0.1,
                    "num_predict": 512,
                    "num_ctx": 4096,
                },
            )
        assert "error" in result
        assert "timed out" in result["error"]

    def test_partial_content_preserved_on_error(self):
        """If connection drops mid-stream, partial content should still be returned."""
        caller = _make_caller()
        import requests as req_mod

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        # iter_lines raises mid-stream
        resp.iter_lines = MagicMock(
            side_effect=req_mod.exceptions.ChunkedEncodingError("broken")
        )

        with patch("AI.LLMCaller.requests.post", return_value=resp):
            result = caller.call_llm(
                model="mistral:7b",
                prompt="test",
                ollama_options={
                    "temperature": 0.1,
                    "num_predict": 512,
                    "num_ctx": 4096,
                },
            )
        assert "error" in result


# ===========================================================================
# get_model_context_limit()
# ===========================================================================


class TestGetModelContextLimit:
    def _make_caller_with_real_resolve(self):
        from AI.LLMCaller import LLMCaller

        caller = LLMCaller.__new__(LLMCaller)
        caller.cfg = StubConfig()
        caller.pretty = StubPrettyWriter()
        caller.helpers = StubHelpers()
        caller.aiHelpers = StubAIHelpers()
        return caller

    def test_returns_context_length(self):
        caller = self._make_caller_with_real_resolve()
        api_response = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 131072,
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("AI.LLMCaller.requests.post", return_value=mock_resp):
            result = caller.get_model_context_limit("llama3.1:8b")
        assert result == 131072

    def test_returns_none_on_missing_arch(self):
        caller = self._make_caller_with_real_resolve()
        api_response = {"model_info": {}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("AI.LLMCaller.requests.post", return_value=mock_resp):
            result = caller.get_model_context_limit("unknown:model")
        assert result is None

    def test_returns_none_on_network_error(self):
        caller = self._make_caller_with_real_resolve()
        import requests as req_mod

        with patch(
            "AI.LLMCaller.requests.post",
            side_effect=req_mod.exceptions.ConnectionError("offline"),
        ):
            result = caller.get_model_context_limit("mistral:7b")
        assert result is None

    def test_returns_none_on_missing_context_key(self):
        caller = self._make_caller_with_real_resolve()
        api_response = {
            "model_info": {
                "general.architecture": "llama",
                # no llama.context_length key
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("AI.LLMCaller.requests.post", return_value=mock_resp):
            result = caller.get_model_context_limit("llama3.1:8b")
        assert result is None
