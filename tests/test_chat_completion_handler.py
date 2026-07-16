# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportOptionalOperand=false, reportUnusedVariable=false
"""
Tests for Api.ChatCompletionHandler:

  - ChatMessage.text()
  - ChatCompletionRequest  (Pydantic model + _coerce_stop validator)
  - buildQuery()
  - applyRequestToSession()
  - _complianceResponse()
  - handleRequest()  (happy paths + error branches, non-streaming and streaming)

All heavy runtime dependencies (Session, Chatter, RAGChatImpl, AIHelpers, …) are
replaced with lightweight stubs so the tests never load ML models or touch
ChromaDB.
"""

import asyncio
import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Import the module under test
# (ChatCompletionHandler only imports class *definitions* from its deps, so
#  the module-level import is safe without pre-stubbing sys.modules.)
# ---------------------------------------------------------------------------
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from Api.ChatCompletionHandler import (
    ChatMessage,
    ChatCompletionRequest,
    _buildQuey,
    _applyRequestToSession,
    _complianceResponse,
    handleRequest,
)

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class StubSession:
    """Minimal Session-like object with all attributes used by applyRequestToSession."""

    def __init__(self):
        self.file_name = None
        self.file_path = None
        self.query: str | None = None
        self.strategy: str | None = None
        self.retriever_k: int | None = None
        self.rerank: bool | None = None
        self.final_chunks_to_llm: int | None = None
        self.chroma_threshold: float | None = None
        self.per_file_limit: int | None = None
        self.use_chat_context: bool | None = None
        self.turns: int | None = None
        self.prune_batch: int | None = None
        self.vector_weight: float | None = None
        self.retrieve_mode: str | None = None
        self.chat_name: str = "default_chat"
        self.max_output_tokens: int | None = None
        self.max_output_tokens_override: int | None = None
        self.context_size_override: int | None = None
        self.temperature: float | None = None
        self.top_k: float | None = None
        self.top_p: float | None = None
        self.base_kwargs: dict | None = None
        self.collection_name: str | None = None
        self.debug_level: int | None = None
        self.extraOllamaOptions: dict | None = None
        self.ollamaTopLevelParams: dict | None = None


class StubQueryParts:
    def applyStrategyDefaults(self, strategy: str, session=None) -> str:
        return ""


class StubConfig:
    def get(self, key, default=None):
        return default

    def get_str(self, key, default="") -> str:
        return default

    def get_int(self, key, default=0) -> int:
        return default

    def get_bool(self, key, default=False) -> bool:
        return default

    def get_float(self, key, default=0.0) -> float:
        return default

    def get_dict(self, key, default=None, *, silent=False) -> dict:
        return default if default is not None else {}

    def set(self, key, value) -> None:
        pass


class _StubIntentFilter:
    def score_query(self, query: str, *, path: str = "web") -> tuple[int, str, list]:
        return (0, "ALLOW", [])


class StubRAG:
    intent_filter = _StubIntentFilter()

    def set_vector_store(self, session) -> None:
        pass


def _make_phrase(phrase: str):
    """Create a minimal ResultsForPrint-like object with only the phrase field."""
    obj = types.SimpleNamespace()
    obj.phrase = phrase
    obj.algo = "Regex+Levenshtein"
    obj.score_str = "1.0000/0.5000"
    obj.algos_matched = "1/2 1/3"
    return obj


class StubAIHelpers:
    """Default: prompt passes, answer passes."""

    def check_user_prompt_with_filter_chain(self, query: str, stage: str):
        return (False, [])

    def check_prompt_with_llm_guard(self, user_query: str):
        return (False, "")


class StubAIHelpersBlock:
    """Prompt compliance always fails with a known phrase."""

    def __init__(self, phrases=("buffer overflow",)):
        self._phrases = phrases

    def check_user_prompt_with_filter_chain(self, query: str, stage: str):
        return (True, [_make_phrase(p) for p in self._phrases])

    def check_prompt_with_llm_guard(self, user_query: str):
        return (False, "")


class StubChatter:
    def __init__(self, run_returns=True, answer="The hedgehog answer."):
        self._answer = answer
        self._run_returns = run_returns

    def run(
        self, session, *, apiChunkHandler=None, is_streaming=None
    ) -> tuple[bool, str | None]:
        return (self._run_returns, self._answer if self._run_returns else None)


# ---------------------------------------------------------------------------
# Helpers for async tests
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously (compatible with pytest without asyncio plugin)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _make_lock():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return asyncio.Lock()


def _base_req(**overrides) -> ChatCompletionRequest:
    """Build a minimal ChatCompletionRequest, optionally overriding fields."""
    defaults: dict[str, Any] = {
        "model": "AnimalDocs",
        "messages": [{"role": "user", "content": "What do hedgehogs eat?"}],
        "stream": False,
    }
    defaults.update(overrides)
    return ChatCompletionRequest(**defaults)


# ===========================================================================
# ChatMessage.text()
# ===========================================================================


class TestChatMessageText:
    def test_string_content_passthrough(self):
        msg = ChatMessage(role="user", content="hello world")
        assert msg.text() == "hello world"

    def test_empty_string_content(self):
        msg = ChatMessage(role="user", content="")
        assert msg.text() == ""

    def test_list_of_text_dicts_joined(self):
        msg = ChatMessage(
            role="user",
            content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
        )
        assert msg.text() == "foo bar"

    def test_list_of_mixed_elements(self):
        msg = ChatMessage(role="user", content=[{"text": "hello"}, "world", 42])
        assert msg.text() == "hello world 42"

    def test_empty_list_returns_empty_string(self):
        msg = ChatMessage(role="user", content=[])
        assert msg.text() == ""

    def test_list_dict_without_text_key_returns_empty(self):
        msg = ChatMessage(
            role="user", content=[{"type": "image_url", "url": "http://x"}]
        )
        assert msg.text() == ""


# ===========================================================================
# ChatCompletionRequest (Pydantic model)
# ===========================================================================


class TestChatCompletionRequest:
    def test_minimal_construction(self):
        req = _base_req()
        assert req.model == "AnimalDocs"
        assert req.stream is False

    def test_stop_string_coerced_to_list(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            stop="<|end|>",
        )
        assert req.stop == ["<|end|>"]

    def test_stop_list_unchanged(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            stop=["<|end|>", "</s>"],
        )
        assert req.stop == ["<|end|>", "</s>"]

    def test_stop_none_stays_none(self):
        req = _base_req()
        assert req.stop is None

    def test_extra_fields_accepted(self):
        """extra='allow' means OpenWebUI custom fields do not raise."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            unknown_openwebui_param=True,  # pyright: ignore[reportCallIssue]
        )
        assert req.model_dump().get("unknown_openwebui_param") is True

    def test_chat_id_field_captured(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            chat_id="abc-123",
        )
        assert req.chat_id == "abc-123"

    def test_chat_id_defaults_none(self):
        req = _base_req()
        assert req.chat_id is None

    def test_rag_specific_fields_accepted(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            strategy="WIDE",
            retriever_k=30,
            rerank=False,
            chroma_threshold=0.6,
            final_chunks_to_llm=20,
            vector_weight=0.5,
            use_chat_context=True,
            chat_name="my_chat",
            per_file_limit=5,
        )
        assert req.strategy == "WIDE"
        assert req.retriever_k == 30
        assert req.rerank is False
        assert req.chroma_threshold == 0.6
        assert req.final_chunks_to_llm == 20
        assert req.vector_weight == 0.5
        assert req.use_chat_context is True
        assert req.chat_name == "my_chat"
        assert req.per_file_limit == 5

    def test_string_int_coerced(self):
        """OpenWebUI sends custom params as strings — ints must be coerced."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            retriever_k="200",
            final_chunks_to_llm="50",
            per_file_limit="3",
            max_tokens="1024",
            debug_level="10",
        )
        assert req.retriever_k == 200
        assert req.final_chunks_to_llm == 50
        assert req.per_file_limit == 3
        assert req.max_tokens == 1024
        assert req.debug_level == 10

    def test_string_float_coerced(self):
        """OpenWebUI sends custom params as strings — floats must be coerced."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            chroma_threshold="0.75",
            vector_weight="0.5",
            temperature="0.8",
            top_p="0.9",
            top_k="40.5",
        )
        assert req.chroma_threshold == 0.75
        assert req.vector_weight == 0.5
        assert req.temperature == 0.8
        assert req.top_p == 0.9
        assert req.top_k == 40.5

    def test_string_bool_coerced(self):
        """OpenWebUI sends custom params as strings — bools must be coerced."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            rerank="true",
            use_chat_context="false",
            stream="true",
        )
        assert req.rerank is True
        assert req.use_chat_context is False
        assert req.stream is True

    def test_native_types_unchanged(self):
        """Values already of the correct type must pass through unmodified."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            retriever_k=60,
            chroma_threshold=0.55,
            rerank=True,
        )
        assert req.retriever_k == 60
        assert req.chroma_threshold == 0.55
        assert req.rerank is True

    def test_float_to_int_coerced(self):
        """OpenWebUI may send JSON floats (e.g. 100.0) for int fields."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            retriever_k=100.0,
            max_tokens=400.0,
            final_chunks_to_llm=50.0,
        )
        assert req.retriever_k == 100
        assert isinstance(req.retriever_k, int)
        assert req.max_tokens == 400
        assert isinstance(req.max_tokens, int)
        assert req.final_chunks_to_llm == 50
        assert isinstance(req.final_chunks_to_llm, int)

    def test_top_k_accepts_float(self):
        """top_k is a float field — fractional values must be accepted."""
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            top_k=313.5,
        )
        assert req.top_k == 313.5


# ===========================================================================
# buildQuery()
# ===========================================================================


class TestBuildQuery:
    def _msg(self, role, content="") -> ChatMessage:
        return ChatMessage(role=role, content=content)

    def test_single_user_message(self):
        msgs = [self._msg("user", "what is a hedgehog?")]
        assert _buildQuey(msgs) == "what is a hedgehog?"

    def test_last_user_message_wins(self):
        msgs = [
            self._msg("user", "first question"),
            self._msg("assistant", "first answer"),
            self._msg("user", "second question"),
        ]
        assert _buildQuey(msgs) == "second question"

    def test_system_message_prepended(self):
        msgs = [
            self._msg("system", "You are a helpful assistant."),
            self._msg("user", "what is a hedgehog?"),
        ]
        result = _buildQuey(msgs)
        assert result.startswith("You are a helpful assistant.")
        assert "what is a hedgehog?" in result

    def test_system_plus_last_user(self):
        msgs = [
            self._msg("system", "System context."),
            self._msg("user", "Q1"),
            self._msg("assistant", "A1"),
            self._msg("user", "Q2"),
        ]
        result = _buildQuey(msgs)
        assert result.startswith("System context.")
        assert result.endswith("Q2")

    def test_only_system_no_user(self):
        msgs = [self._msg("system", "only system")]
        # system_text set but user_text is "" → "only system\n\n"
        result = _buildQuey(msgs)
        assert "only system" in result

    def test_empty_messages(self):
        assert _buildQuey([]) == ""

    def test_no_user_messages(self):
        msgs = [self._msg("assistant", "some prior answer")]
        assert _buildQuey(msgs) == ""

    def test_only_second_system_ignored(self):
        """Only the first system message is picked up."""
        msgs = [
            self._msg("system", "first system"),
            self._msg("system", "second system"),
            self._msg("user", "query"),
        ]
        result = _buildQuey(msgs)
        assert "first system" in result
        assert "second system" not in result

    def test_multipart_user_content(self):
        msg = ChatMessage(
            role="user", content=[{"text": "describe"}, {"text": "a fox"}]
        )
        result = _buildQuey([msg])
        assert "describe" in result
        assert "fox" in result


# ===========================================================================
# applyRequestToSession()
# ===========================================================================


class TestApplyRequestToSession:
    def _apply(self, req: ChatCompletionRequest, session=None, qp=None, cfg=None):
        session = session or StubSession()
        qp = qp or StubQueryParts()
        cfg = cfg or StubConfig()
        _applyRequestToSession(req, session, qp, cfg)
        return session

    # --- collection / model -------------------------------------------------

    def test_collection_name_set_from_model(self):
        req = _base_req(model="DocsToLoadOrClassify")
        session = self._apply(req)
        assert session.collection_name == "DocsToLoadOrClassify"

    # --- strategy -----------------------------------------------------------

    def test_valid_strategy_applied(self):
        for strat in ("ULTRA_WIDE", "WIDE", "BALANCED_FILE_CAP", "NARROW"):
            session = StubSession()
            req = _base_req(strategy=strat)
            self._apply(req, session=session)
            assert session.strategy == strat

    def test_strategy_case_insensitive(self):
        req = _base_req(strategy="wide")
        session = self._apply(req)
        assert session.strategy == "WIDE"

    def test_invalid_strategy_raises_400(self):
        req = _base_req(strategy="SUPER_WIDE")
        with pytest.raises(HTTPException) as exc_info:
            self._apply(req)
        assert exc_info.value.status_code == 400
        assert "SUPER_WIDE" in str(exc_info.value.detail)

    def test_no_strategy_in_request_leaves_session_strategy(self):
        session = StubSession()
        session.strategy = "NARROW"
        req = _base_req()  # no strategy field
        self._apply(req, session=session)
        assert session.strategy == "NARROW"

    def test_applyStrategyDefaults_called(self):
        called_with = []

        class RecordingQP:
            def applyStrategyDefaults(self, strategy, session=None):
                called_with.append(strategy)
                return ""

        req = _base_req(strategy="WIDE")
        _applyRequestToSession(req, StubSession(), RecordingQP(), StubConfig())
        assert called_with == ["WIDE"]

    # --- RAG overrides ------------------------------------------------------

    def test_retriever_k_applied(self):
        req = _base_req(retriever_k=42)
        session = self._apply(req)
        assert session.retriever_k == 42

    def test_retriever_k_not_applied_when_none(self):
        session = StubSession()
        session.retriever_k = 99
        self._apply(_base_req(), session=session)
        assert session.retriever_k == 99  # unchanged

    def test_rerank_applied(self):
        session = self._apply(_base_req(rerank=False))
        assert session.rerank is False

    def test_chroma_threshold_clamped_low(self):
        session = self._apply(_base_req(chroma_threshold=-0.5))
        assert session.chroma_threshold == 0.0

    def test_chroma_threshold_clamped_high(self):
        session = self._apply(_base_req(chroma_threshold=1.5))
        assert session.chroma_threshold == 1.0

    def test_chroma_threshold_in_range_unchanged(self):
        session = self._apply(_base_req(chroma_threshold=0.7))
        assert abs(session.chroma_threshold - 0.7) < 1e-9

    def test_vector_weight_clamped(self):
        assert self._apply(_base_req(vector_weight=-1.0)).vector_weight == 0.0
        assert self._apply(_base_req(vector_weight=2.0)).vector_weight == 1.0
        assert abs(self._apply(_base_req(vector_weight=0.5)).vector_weight - 0.5) < 1e-9

    def test_final_chunks_to_llm_applied(self):
        session = self._apply(_base_req(final_chunks_to_llm=15))
        assert session.final_chunks_to_llm == 15

    def test_use_chat_context_applied(self):
        session = self._apply(_base_req(use_chat_context=True))
        assert session.use_chat_context is True

    def test_per_file_limit_applied(self):
        session = self._apply(_base_req(per_file_limit=3))
        assert session.per_file_limit == 3

    # --- chat_name priority -------------------------------------------------

    def test_explicit_chat_name_wins(self):
        req = _base_req(chat_name="my_chat", chat_id="irrelevant-id")
        session = self._apply(req)
        assert session.chat_name == "my_chat"

    def test_chat_id_used_when_no_chat_name(self):
        req = _base_req(chat_id="conv-uuid-42")
        session = self._apply(req)
        assert session.chat_name == "conv-uuid-42"

    def test_session_chat_name_preserved_when_neither_set(self):
        session = StubSession()
        session.chat_name = "existing"
        self._apply(_base_req(), session=session)
        assert session.chat_name == "existing"

    # --- LLM overrides ------------------------------------------------------

    def test_temperature_applied(self):
        session = self._apply(_base_req(temperature=0.3))
        assert session.temperature == pytest.approx(0.3)

    def test_top_p_applied(self):
        session = self._apply(_base_req(top_p=0.9))
        assert session.top_p == pytest.approx(0.9)

    def test_top_k_applied_as_float(self):
        session = self._apply(_base_req(top_k=40))
        assert session.top_k == 40.0
        assert isinstance(session.top_k, float)

    def test_max_tokens_mapped_to_override(self):
        session = self._apply(_base_req(max_tokens=512))
        assert session.max_output_tokens_override == 512

    def test_num_ctx_mapped_to_context_size_override(self):
        session = self._apply(_base_req(num_ctx=4096))
        assert session.context_size_override == 4096

    # --- Ollama options -----------------------------------------------------

    def test_ollama_option_collected(self):
        session = self._apply(_base_req(seed=42))
        assert session.extraOllamaOptions is not None
        assert session.extraOllamaOptions.get("seed") == 42

    def test_no_ollama_options_none(self):
        session = self._apply(_base_req())
        assert session.extraOllamaOptions is None

    def test_ollama_top_level_collected(self):
        session = self._apply(_base_req(think=True))
        assert session.ollamaTopLevelParams is not None
        assert session.ollamaTopLevelParams.get("think") is True

    # --- query / base_kwargs ------------------------------------------------

    def test_query_built_from_messages(self):
        req = _base_req(messages=[{"role": "user", "content": "tell me about foxes"}])
        session = self._apply(req)
        assert session.query == "tell me about foxes"

    def test_base_kwargs_set(self):
        req = _base_req(retriever_k=20)
        session = self._apply(req)
        assert session.base_kwargs == {"k": 20}


# ===========================================================================
# _complianceResponse()
# ===========================================================================


class TestComplianceResponse:
    def test_non_streaming_returns_json_response(self):
        resp = _complianceResponse("req-1", "AnimalDocs", "I'm sorry.", stream=False)
        assert isinstance(resp, JSONResponse)

    def test_non_streaming_shape(self):
        resp = _complianceResponse("req-1", "AnimalDocs", "refusal msg", stream=False)
        body = json.loads(resp.body)
        assert body["id"] == "req-1"
        assert body["model"] == "AnimalDocs"
        assert body["object"] == "chat.completion"
        choices = body["choices"]
        assert len(choices) == 1
        assert choices[0]["message"]["role"] == "assistant"
        assert choices[0]["message"]["content"] == "refusal msg"
        assert choices[0]["finish_reason"] == "stop"

    def test_non_streaming_200_status(self):
        resp = _complianceResponse("r", "m", "sorry", stream=False)
        assert resp.status_code == 200

    def test_streaming_returns_streaming_response(self):
        resp = _complianceResponse("req-2", "AnimalDocs", "sorry", stream=True)
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"

    def test_streaming_sse_chunks(self):
        resp = _complianceResponse("req-2", "AnimalDocs", "blocked!", stream=True)

        async def _collect():
            chunks = []
            async for chunk in resp.body_iterator:  # type: ignore[attr-defined]
                if isinstance(chunk, bytes):
                    chunk = chunk.decode()
                chunks.append(chunk)
            return "".join(chunks)

        raw = _run(_collect())
        # must contain role opening, the message, stop chunk, and [DONE]
        # json.dumps adds spaces after colons, so match with spaces
        assert '"role": "assistant"' in raw
        assert "blocked!" in raw
        assert '"finish_reason": "stop"' in raw
        assert "data: [DONE]" in raw


# ===========================================================================
# handleRequest()  — integration tests with stub objects
# ===========================================================================


class TestHandleRequest:
    """
    All heavy deps are passed as stub instances so nothing real is ever
    instantiated.  Requests are serialised through an asyncio.Lock just as
    in production.
    """

    def _run_handle(
        self,
        req,
        chatter=None,
        ai_helpers=None,
        rag=None,
        run_returns=True,
        answer="The answer.",
    ):
        if chatter is None:
            chatter = StubChatter(run_returns=run_returns, answer=answer)
        if ai_helpers is None:
            ai_helpers = StubAIHelpers()
        if rag is None:
            rag = StubRAG()

        qp = StubQueryParts()
        cfg = StubConfig()
        lock = _make_lock()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            return _run(
                handleRequest(
                    req=req,
                    chatter=chatter,
                    rag=rag,
                    queryParts=qp,
                    aiHelpers=ai_helpers,
                    cfg=cfg,
                    lock=lock,
                    executor=executor,
                    serviceCfgDebugLevel=0,
                )
            )
        finally:
            executor.shutdown(wait=False)

    # --- happy path: non-streaming ------------------------------------------

    def test_non_streaming_returns_json_response(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req)
        assert isinstance(resp, JSONResponse)

    def test_non_streaming_200_status(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req)
        assert resp.status_code == 200

    def test_non_streaming_answer_in_body(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req, answer="Hedgehogs eat insects.")
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "Hedgehogs eat insects."
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["model"] == "AnimalDocs"

    def test_non_streaming_response_shape(self):
        resp = self._run_handle(_base_req(stream=False))
        body = json.loads(resp.body)
        assert "id" in body
        assert body["object"] == "chat.completion"
        assert "created" in body
        assert "usage" in body

    # --- happy path: streaming ----------------------------------------------

    def test_streaming_returns_streaming_response(self):
        # Ensure grounding doesn't force buffered reply (which disables streaming)
        old_val = os.environ.get("SERVE_IN_MEMORY_DOCS_HTTP")
        os.environ["SERVE_IN_MEMORY_DOCS_HTTP"] = "0"
        try:
            req = _base_req(stream=True)
            resp = self._run_handle(req)
            assert isinstance(resp, StreamingResponse)
            assert resp.media_type == "text/event-stream"
        finally:
            if old_val is None:
                os.environ.pop("SERVE_IN_MEMORY_DOCS_HTTP", None)
            else:
                os.environ["SERVE_IN_MEMORY_DOCS_HTTP"] = old_val

    # --- prompt compliance blocked ------------------------------------------

    def test_compliance_block_non_streaming_returns_json(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req, ai_helpers=StubAIHelpersBlock(["buffer overflow"]))
        assert isinstance(resp, JSONResponse)
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "buffer overflow" in content
        assert "I'm sorry" in content

    def test_compliance_block_streaming_returns_stream(self):
        req = _base_req(stream=True)
        resp = self._run_handle(req, ai_helpers=StubAIHelpersBlock(["buffer overflow"]))
        assert isinstance(resp, StreamingResponse)

    def test_compliance_block_multiple_phrases(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req, ai_helpers=StubAIHelpersBlock(["foo", "bar"]))
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "foo" in content
        assert "bar" in content

    def test_compliance_block_deduplicates_phrases(self):
        req = _base_req(stream=False)
        resp = self._run_handle(
            req, ai_helpers=StubAIHelpersBlock(["foo", "foo", "foo"])
        )
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        # The user-facing hint (before the --- details block) should mention "foo" only once
        hint = content.split("---")[0]
        assert hint.count("foo") == 1

    def test_compliance_block_no_phrases_generic_message(self):
        """Phrase list empty → generic refusal without hint."""
        req = _base_req(stream=False)
        resp = self._run_handle(req, ai_helpers=StubAIHelpersBlock(phrases=()))
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "I'm sorry" in content

    # --- answer compliance fail (chatter.run returns False) -----------------

    def test_answer_compliance_fail_returns_json(self):
        req = _base_req(stream=False)
        resp = self._run_handle(req, run_returns=False)
        assert isinstance(resp, JSONResponse)
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "compliance" in content.lower() or "can't provide" in content.lower()

    # --- collection not found -----------------------------------------------

    def test_collection_not_found_returns_friendly_message(self):
        from Commons.Exceptions import CollectionNotFoundError

        class MissingRAG:
            def set_vector_store(self, session):
                raise CollectionNotFoundError("unknown_collection")

        req = _base_req(stream=False)
        resp = self._run_handle(req, rag=MissingRAG())
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "ragchatservice" in content.lower()
        assert "unknown_collection" in content

    # --- LLMResultError → 502 -----------------------------------------------

    def test_llm_result_error_returns_friendly_message(self):
        from Commons.Exceptions import LLMResultError

        class FailingChatter(StubChatter):
            def run(self, session, *, apiChunkHandler=None, is_streaming=None):
                raise LLMResultError("LLM timed out")

        req = _base_req(stream=False)
        resp = self._run_handle(req, chatter=FailingChatter())
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "llm" in content.lower() or "ragchatservice" in content.lower()

    def test_vllm_backend_error_returns_backend_specific_message(self):
        from Commons.Exceptions import VllmNotRunning

        class FailingChatter(StubChatter):
            def run(self, session, *, apiChunkHandler=None, is_streaming=None):
                raise VllmNotRunning("Can't reach VLLM on: http://example/v1/models")

        req = _base_req(stream=False)
        resp = self._run_handle(req, chatter=FailingChatter())

        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert "vllm" in body["detail"].lower()

    # --- generic exception → 500 --------------------------------------------

    def test_unexpected_exception_returns_friendly_message(self):
        class BrokenRAG:
            def set_vector_store(self, session):
                raise RuntimeError("disk full")

        req = _base_req(stream=False)
        resp = self._run_handle(req, rag=BrokenRAG())
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "ragchatservice" in content.lower() or "disk full" in content.lower()
