# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportUnusedVariable=false
"""
Tests for ModelOutputAdapter.interpret()

Coverage matrix
---------------
                    | streaming=True           | streaming=False
--------------------|--------------------------|-------------------------------
llama (answer key)  | assembled JSON           | Ollama wrapper (response field)
llama (allowed key) | assembled JSON           | Ollama wrapper
mistral (allowed)   | assembled JSON           | Ollama wrapper
mistral (answer)    | assembled JSON           | Ollama wrapper
guard (safe)        | NDJSON safe              | plain "safe"
guard (unsafe S2)   | NDJSON unsafe\nS2        | Ollama wrapper response field
generic compliance  | plain text               | plain text
generic LLM         | plain JSON               | plain JSON
edge cases          | split JSON objects       | empty / unknown
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from AI.ModelOutputAdapter import ModelOutputAdapter, ModelOutput

# ---------------------------------------------------------------------------
# Helpers to build mock Ollama payloads
# ---------------------------------------------------------------------------


def _ollama_wrap(response_text: str, done: bool = True) -> str:
    """Simulate a single non-streaming Ollama response dict."""
    return json.dumps(
        {
            "model": "test-model",
            "created_at": "2026-01-01T00:00:00Z",
            "response": response_text,
            "done": done,
            "done_reason": "stop",
        }
    )


def _ollama_stream(fragments: list[str]) -> str:
    """Simulate streaming NDJSON lines, each with a 'response' fragment."""
    lines = []
    for i, frag in enumerate(fragments):
        done = i == len(fragments) - 1
        lines.append(
            json.dumps(
                {
                    "model": "test-model",
                    "created_at": "2026-01-01T00:00:00Z",
                    "response": frag,
                    "done": done,
                }
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return ModelOutputAdapter()


# ===========================================================================
# Llama adapter – answer/reason key
# ===========================================================================


class TestLlamaAnswerKey:
    """{"answer": "not allowed", "reason": "..."}"""

    def test_streaming_not_allowed(self, adapter):
        # After stream assembly the text is the plain JSON string
        assembled = '{"answer": "not allowed", "reason": "violates Article 5"}'
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"
        assert result.reason == "violates Article 5"
        assert result.content == "not allowed"

    def test_streaming_allowed(self, adapter):
        assembled = '{"answer": "allowed", "reason": "content is safe"}'
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"
        assert result.reason == "content is safe"

    def test_non_streaming_ollama_wrapper(self, adapter):
        inner = '{"answer": "not allowed", "reason": "terrorism instruction"}'
        raw = _ollama_wrap(inner)
        result = adapter.interpret(
            {"raw": raw}, "llama3.1:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"
        assert result.reason == "terrorism instruction"

    def test_non_streaming_allowed_wrapper(self, adapter):
        inner = '{"answer": "allowed", "reason": "no policy violation"}'
        raw = _ollama_wrap(inner)
        result = adapter.interpret(
            {"raw": raw}, "llama3.1:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "allow"

    def test_answer_synonyms_allow(self, adapter):
        for val in ("allow", "allowed", "safe", "compliant", "yes", "true"):
            assembled = json.dumps({"answer": val, "reason": "ok"})
            result = adapter.interpret(
                assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
            )
            assert result.decision == "allow", f"Expected allow for answer={val!r}"

    def test_answer_non_allow_values(self, adapter):
        for val in ("not allowed", "block", "unsafe", "forbidden"):
            assembled = json.dumps({"answer": val, "reason": "bad"})
            result = adapter.interpret(
                assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
            )
            assert result.decision == "block", f"Expected block for answer={val!r}"

    def test_raw_preserved(self, adapter):
        assembled = '{"answer": "not allowed", "reason": "test"}'
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.raw == assembled


# ===========================================================================
# Llama adapter – allowed key (boolean and string forms)
# ===========================================================================


class TestLlamaAllowedKey:
    """{"allowed": ..., "explanation": "..."}"""

    def test_bool_true_streaming(self, adapter):
        assembled = json.dumps({"allowed": True, "explanation": "looks fine"})
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"
        assert result.reason == "looks fine"

    def test_bool_false_streaming(self, adapter):
        assembled = json.dumps({"allowed": False, "explanation": "prohibited"})
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"

    def test_string_allowed_streaming(self, adapter):
        assembled = json.dumps({"allowed": "allowed", "explanation": "safe"})
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"

    def test_string_not_allowed_streaming(self, adapter):
        assembled = json.dumps({"allowed": "not allowed", "explanation": "blocked"})
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"

    def test_non_streaming_wrapper(self, adapter):
        inner = json.dumps({"allowed": False, "explanation": "GDPR violation"})
        raw = _ollama_wrap(inner)
        result = adapter.interpret(
            {"raw": raw}, "llama3.1:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"
        assert result.reason == "GDPR violation"


# ===========================================================================
# Mistral adapter – same keys, fallback_to_generic=True
# ===========================================================================


class TestMistralAdapter:
    def test_allowed_key_streaming(self, adapter):
        assembled = json.dumps({"allowed": "allowed", "explanation": "compliant"})
        result = adapter.interpret(
            assembled, "mistral:7b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"
        assert result.reason == "compliant"

    def test_allowed_key_not_allowed_streaming(self, adapter):
        assembled = json.dumps(
            {"allowed": "not allowed", "explanation": "EU AI Act violation"}
        )
        result = adapter.interpret(
            assembled, "mistral:7b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"

    def test_answer_key_streaming(self, adapter):
        assembled = json.dumps({"answer": "allowed", "reason": "safe prompt"})
        result = adapter.interpret(
            assembled, "mistral:7b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"

    def test_non_streaming_wrapper(self, adapter):
        inner = json.dumps({"allowed": "not allowed", "explanation": "terrorism"})
        raw = _ollama_wrap(inner)
        result = adapter.interpret(
            {"raw": raw}, "mistral:7b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"

    def test_bool_false_non_streaming(self, adapter):
        inner = json.dumps({"allowed": False, "explanation": "blocked"})
        raw = _ollama_wrap(inner)
        result = adapter.interpret(
            {"raw": raw}, "mistral:7b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"
        assert result.reason == "blocked"

    def test_non_json_fallback_to_generic(self, adapter):
        # Mistral falls back to _adapter_generic_llm on total parse failure
        result = adapter.interpret(
            "this is not json", "mistral:7b", is_compliance=False, is_streaming=False
        )
        assert result.content == "this is not json"


# ===========================================================================
# Guard adapter (llama-guard3 / promptguard)
# ===========================================================================


class TestGuardAdapter:
    def test_non_streaming_safe(self, adapter):
        result = adapter.interpret(
            "safe", "llama-guard3:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "allow"

    def test_non_streaming_unsafe_s2(self, adapter):
        result = adapter.interpret(
            "unsafe\nS2", "llama-guard3:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"
        assert "harassment" in (result.reason or "").lower()

    def test_non_streaming_ollama_wrapper(self, adapter):
        raw = _ollama_wrap("unsafe\nS5")
        result = adapter.interpret(
            {"raw": raw}, "llama-guard3:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"
        assert "criminal" in (result.reason or "").lower()

    def test_streaming_ndjson_safe(self, adapter):
        ndjson = _ollama_stream(["saf", "e"])
        result = adapter.interpret(
            ndjson, "llama-guard3:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"

    def test_streaming_ndjson_unsafe_s1(self, adapter):
        ndjson = _ollama_stream(["unsa", "fe\n", "S1"])
        result = adapter.interpret(
            ndjson, "llama-guard3:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"
        assert "hate" in (result.reason or "").lower()

    def test_unknown_s_token(self, adapter):
        result = adapter.interpret(
            "unsafe\nS9", "llama-guard3:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"


# ===========================================================================
# Generic compliance adapter (fallback for unrecognised model names)
# ===========================================================================


class TestGenericCompliance:
    def test_safe_keyword(self, adapter):
        result = adapter.interpret(
            "safe", "unknown-model", is_compliance=True, is_streaming=False
        )
        assert result.decision == "allow"

    def test_unsafe_keyword(self, adapter):
        result = adapter.interpret(
            "unsafe", "unknown-model", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"

    def test_allowed_keyword(self, adapter):
        result = adapter.interpret(
            "allowed", "unknown-model", is_compliance=True, is_streaming=False
        )
        assert result.decision == "allow"

    def test_not_allowed_keyword(self, adapter):
        result = adapter.interpret(
            "not allowed", "unknown-model", is_compliance=True, is_streaming=False
        )
        assert result.decision == "block"

    def test_unknown_label_gives_error(self, adapter):
        result = adapter.interpret(
            "something completely random",
            "unknown-model",
            is_compliance=True,
            is_streaming=False,
        )
        assert result.decision == "error"


# ===========================================================================
# Generic LLM adapter (non-compliance)
# ===========================================================================


class TestGenericLLM:
    def test_json_content_extracted(self, adapter):
        payload = json.dumps({"summary": "Paris is the capital of France."})
        result = adapter.interpret(
            payload, "llama3.1:8b", is_compliance=False, is_streaming=False
        )
        assert result.is_json is True
        assert "Paris" in (result.content or "")

    def test_plain_text_passthrough(self, adapter):
        result = adapter.interpret(
            "Hello world", "llama3.1:8b", is_compliance=False, is_streaming=False
        )
        assert result.content == "Hello world"
        assert result.is_json is False

    def test_non_streaming_ollama_wrapper_content(self, adapter):
        raw = _ollama_wrap("The answer is 42.")
        result = adapter.interpret(
            {"raw": raw}, "llama3.1:8b", is_compliance=False, is_streaming=False
        )
        assert "42" in (result.content or "")


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_split_json_objects_merged(self, adapter):
        """Model emits {"answer":"not allowed"} and {"reason":"bad"} on separate lines."""
        split = '{"answer": "not allowed"}\n{"reason": "bad content"}'
        result = adapter.interpret(
            split, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "block"
        assert result.reason == "bad content"

    def test_raw_field_preserved_streaming(self, adapter):
        ndjson = _ollama_stream(['{"ans', 'wer":"not ', 'allowed","reason":"x"}'])
        result = adapter.interpret(
            ndjson, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        # raw should be the original NDJSON, not the assembled text
        assert result.raw == ndjson

    def test_empty_string(self, adapter):
        # llama adapter: empty string -> no JSON -> ModelOutput(content="", decision=None)
        result = adapter.interpret(
            "", "llama3.1:8b", is_compliance=True, is_streaming=False
        )
        assert result.content == ""
        assert result.decision is None
        # unknown-model (generic compliance path): empty string -> "error"
        result2 = adapter.interpret(
            "", "unknown-model", is_compliance=True, is_streaming=False
        )
        assert result2.decision == "error"

    def test_dict_input_raw_key(self, adapter):
        """When raw_output is a dict with a 'raw' key."""
        inner = json.dumps({"answer": "allowed", "reason": "clean"})
        result = adapter.interpret(
            {"raw": inner}, "llama3.1:8b", is_compliance=True, is_streaming=False
        )
        assert result.decision == "allow"

    def test_dict_input_content_key(self, adapter):
        """When raw_output is a dict with a 'content' key (non-compliance path)."""
        result = adapter.interpret(
            {"content": "hello"}, "llama3.1:8b", is_compliance=False, is_streaming=False
        )
        assert "hello" in (result.content or "")

    def test_answer_key_case_insensitive(self, adapter):
        assembled = json.dumps({"answer": "Allowed", "reason": "fine"})
        result = adapter.interpret(
            assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
        )
        assert result.decision == "allow"

    def test_explanation_fallback_to_reason(self, adapter):
        """Both 'reason' and 'explanation' aliases work."""
        for key in ("reason", "explanation"):
            assembled = json.dumps({"answer": "not allowed", key: "some reason"})
            result = adapter.interpret(
                assembled, "llama3.1:8b", is_compliance=True, is_streaming=True
            )
            assert result.reason == "some reason", f"Failed for key={key!r}"
