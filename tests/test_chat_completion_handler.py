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
from unittest.mock import MagicMock

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

from Chat.FileSelector import ServiceFileSelector
from Api.ChatCompletionHandler import (
    ChatMessage,
    ChatCompletionRequest,
    _buildQuery,
    _applyRequestToSession,
    _complianceResponse,
    _isTitleRequest,
    handleRequest,
)

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class StubSession:
    """Minimal Session-like object with all attributes used by applyRequestToSession."""

    def __init__(self):
        self.is_service: bool = False
        self.file_name: str | None = None
        self.summary_length: str | None = None
        self.summary_temperature: float | None = None
        self.chunk_overlap: int | None = None
        self.file_path: str | None = None
        self.metadata_filters: dict[str, str] = {}
        self.query: str | None = None
        self.user_query_input: str | None = None
        self.indexed_query_error: str | None = None
        self.strategy: str | None = None
        self.orchestrator_flow: str | None = None
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

    def applyOrchestrationFlowDefaults(
        self, orchestrator_flow: str, session=None
    ) -> str:
        _ = session
        return orchestrator_flow


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
        if key == "_ORCHESTRATION_FLOW_ALIASES":
            return {"THOROUGH_QUERY_REWRITE": "ENGLISH_ALL_MULTILEG"}
        return default if default is not None else {}

    def get_list(self, key, default=None, *, silent=False) -> list:
        _ = silent
        if key == "_ALLOWED_ORCHESTRATION_FLOWS":
            return [
                "ENGLISH_ALL_MULTILEG",
                "ENGLISH_ALL",
                "ORIGINAL_VECTOR",
            ]
        if key == "_ALLOWED_RETRIEVE_MODES":
            return [
                "VECTOR",
                "BM25",
                "GRAPH",
                "REGEX",
                "VECTOR_BM25",
                "VECTOR_GRAPH",
                "BM25_GRAPH",
                "VECTOR_REGEX",
                "BM25_REGEX",
                "GRAPH_REGEX",
                "ALL",
                "WEB",
            ]
        return default if default is not None else []

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
        assert _buildQuery(msgs) == "what is a hedgehog?"

    @pytest.mark.parametrize("prefix", ["new:", "NEW TOPIC:", "newtopic:"])
    @pytest.mark.parametrize("system", ["", "Use the supplied evidence."])
    def test_new_topic_prefix_is_not_part_of_model_input(self, prefix, system):
        msgs = [
            self._msg("system", system),
            self._msg("user", f"  {prefix} What animals are mentioned?"),
        ]
        expected = "What animals are mentioned?"
        if system:
            expected = f"{system}\n\n{expected}"
        assert _buildQuery(msgs) == expected

    def test_last_user_message_wins(self):
        msgs = [
            self._msg("user", "first question"),
            self._msg("assistant", "first answer"),
            self._msg("user", "second question"),
        ]
        assert _buildQuery(msgs) == "second question"

    def test_system_message_prepended(self):
        msgs = [
            self._msg("system", "You are a helpful assistant."),
            self._msg("user", "what is a hedgehog?"),
        ]
        result = _buildQuery(msgs)
        assert result.startswith("You are a helpful assistant.")
        assert "what is a hedgehog?" in result

    def test_system_plus_last_user(self):
        msgs = [
            self._msg("system", "System context."),
            self._msg("user", "Q1"),
            self._msg("assistant", "A1"),
            self._msg("user", "Q2"),
        ]
        result = _buildQuery(msgs)
        assert result.startswith("System context.")
        assert result.endswith("Q2")

    def test_only_system_no_user(self):
        msgs = [self._msg("system", "only system")]
        # system_text set but user_text is "" → "only system\n\n"
        result = _buildQuery(msgs)
        assert "only system" in result

    def test_empty_messages(self):
        assert _buildQuery([]) == ""

    def test_no_user_messages(self):
        msgs = [self._msg("assistant", "some prior answer")]
        assert _buildQuery(msgs) == ""

    def test_only_second_system_ignored(self):
        """Only the first system message is picked up."""
        msgs = [
            self._msg("system", "first system"),
            self._msg("system", "second system"),
            self._msg("user", "query"),
        ]
        result = _buildQuery(msgs)
        assert "first system" in result
        assert "second system" not in result

    def test_multipart_user_content(self):
        msg = ChatMessage(
            role="user", content=[{"text": "describe"}, {"text": "a fox"}]
        )
        result = _buildQuery([msg])
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

    def test_new_topic_preserves_raw_input_and_system_prompt(self):
        raw = "new: What animals are mentioned?"
        session = self._apply(
            _base_req(
                messages=[
                    {"role": "system", "content": "Use the supplied evidence."},
                    {"role": "user", "content": raw},
                ]
            )
        )
        assert session.user_query_input == raw
        assert (
            session.query == "Use the supplied evidence.\n\nWhat animals are mentioned?"
        )

        self._apply(
            _base_req(messages=[{"role": "user", "content": "What do they eat?"}]),
            session=session,
        )
        assert session.user_query_input == "What do they eat?"
        assert session.query == "What do they eat?"

    @pytest.mark.parametrize(
        "params, filename, path, expected_filter",
        [
            ({"file": " doc.pdf "}, "doc.pdf", None, {"FileName": {"$eq": "doc.pdf"}}),
            (
                {"path": r'"D:\docs\doc.pdf"'},
                None,
                r"D:\docs\doc.pdf",
                {"FilePath": {"$eq": r"D:\docs\doc.pdf"}},
            ),
            ({"file": ""}, None, None, None),
            ({"path": " "}, None, None, None),
        ],
    )
    def test_document_pin_sets_session_and_retrieval_filter(
        self,
        params,
        filename,
        path,
        expected_filter,
    ):
        session = self._apply(_base_req(**params))
        assert session.is_service is True
        assert session.file_name == filename
        assert session.file_path == path
        assert session.base_kwargs is not None
        assert session.base_kwargs.get("filter") == expected_filter

    @pytest.mark.parametrize(
        "params",
        [
            {"file": "doc.pdf", "path": r"D:\docs\other.pdf"},
            {"file": r"D:\docs\doc.pdf"},
            {"path": "D:\\docs\\"},
            {"file": ".."},
            {"path": "invalid\x00path"},
        ],
    )
    def test_invalid_document_selection_rejected(self, params):
        with pytest.raises(HTTPException) as exc:
            self._apply(_base_req(**params))
        assert exc.value.status_code == 400

    def test_document_pin_does_not_leak_between_requests(self):
        pinned = self._apply(_base_req(file="doc.pdf"))
        unpinned = self._apply(_base_req())
        assert pinned.file_name == "doc.pdf"
        assert unpinned.file_name is None
        assert unpinned.base_kwargs is not None
        assert "filter" not in unpinned.base_kwargs

    @pytest.mark.parametrize("mode", ["off", "heuristic", "llm", "both", " LLM "])
    def test_confidence_evaluation_override(self, mode):
        session = self._apply(_base_req(confidence_evaluation=mode))
        assert session.confidence_evaluation == mode.strip().lower()

    def test_invalid_confidence_mode_does_not_mutate_session(self):
        session = StubSession()
        session.collection_name = "TestCollection"
        session.orchestrator_flow = "ENGLISH_ALL_MULTILEG"
        session.confidence_evaluation = "heuristic"
        with pytest.raises(HTTPException) as exc:
            self._apply(_base_req(confidence_evaluation="invalid"), session=session)
        assert exc.value.status_code == 400
        assert session.confidence_evaluation == "heuristic"

    def test_collection_name_set_from_model(self):
        req = _base_req(model="DocsToLoadOrClassify")
        session = self._apply(req)
        assert session.collection_name == "DocsToLoadOrClassify"

    def test_collection_switch_keeps_overrides_and_clears_collection_scoped_filters(
        self,
    ):
        called_strategy = []
        called_flow = []

        class RecordingQP:
            def applyStrategyDefaults(self, strategy, session=None):
                _ = session
                called_strategy.append(strategy)
                return ""

            def applyOrchestrationFlowDefaults(self, orchestrator_flow, session=None):
                _ = session
                called_flow.append(orchestrator_flow)
                return orchestrator_flow

        session = StubSession()
        session.collection_name = "OldCollection"
        session.strategy = "NARROW"
        session.orchestrator_flow = "ENGLISH_ALL_MULTILEG"
        session.retriever_k = 77
        session.file_name = "doc.txt"
        session.file_path = "D:/OldCollection/doc.txt"
        session.metadata_filters = {"Author": "Ada"}

        req = _base_req(model="NewCollection")
        _applyRequestToSession(req, session, RecordingQP(), StubConfig())

        assert session.collection_name == "NewCollection"
        assert session.file_name is None
        assert session.file_path is None
        assert session.metadata_filters == {}
        assert session.retriever_k == 77
        assert called_strategy == []
        assert called_flow == []

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
        called_flow_with = []

        class RecordingQP:
            def applyStrategyDefaults(self, strategy, session=None):
                called_with.append(strategy)
                return ""

            def applyOrchestrationFlowDefaults(self, orchestrator_flow, session=None):
                called_flow_with.append(orchestrator_flow)
                return orchestrator_flow

        req = _base_req(strategy="WIDE")
        _applyRequestToSession(req, StubSession(), RecordingQP(), StubConfig())
        assert called_with == ["WIDE"]
        assert called_flow_with == ["ENGLISH_ALL_MULTILEG"]

    def test_valid_orchestrator_flow_applied(self):
        for flow in (
            "ENGLISH_ALL_MULTILEG",
            "ENGLISH_ALL",
            "ORIGINAL_VECTOR",
        ):
            session = StubSession()
            req = _base_req(orchestrator_flow=flow)
            self._apply(req, session=session)
            assert session.orchestrator_flow == flow

    def test_orchestrator_flow_case_insensitive(self):
        req = _base_req(orchestrator_flow="english_all")
        session = self._apply(req)
        assert session.orchestrator_flow == "ENGLISH_ALL"

    def test_legacy_orchestrator_flow_alias_resolved(self):
        req = _base_req(orchestrator_flow="thorough_query_rewrite")
        session = self._apply(req)
        assert session.orchestrator_flow == "ENGLISH_ALL_MULTILEG"

    def test_removed_orchestrator_flow_raises_400(self):
        req = _base_req(orchestrator_flow="TRANSLATION_FOCUSED")
        with pytest.raises(HTTPException) as exc_info:
            self._apply(req)
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize(
        "source", ["REWRITTEN_QUERY", "translated_query", "ORIGINAL_QUERY"]
    )
    def test_rerank_query_source_override(self, source):
        session = self._apply(_base_req(rerank_query_source=source))
        assert session.rerank_query_source == source.upper()

    def test_invalid_rerank_query_source_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            self._apply(_base_req(rerank_query_source="BOGUS"))
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize("language", ["english", "NATIVE", " Native "])
    def test_retrieval_language_override(self, language):
        session = self._apply(_base_req(retrieval_language=language))
        assert session.retrieval_language_override == language.strip().upper()

    def test_invalid_retrieval_language_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            self._apply(_base_req(retrieval_language="german"))
        assert exc_info.value.status_code == 400

    def test_invalid_orchestrator_flow_raises_400(self):
        req = _base_req(orchestrator_flow="CUSTOM_FLOW")
        with pytest.raises(HTTPException) as exc_info:
            self._apply(req)
        assert exc_info.value.status_code == 400
        assert "CUSTOM_FLOW" in str(exc_info.value.detail)

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


@pytest.mark.parametrize(
    "text",
    [
        "Generate a concise title summarizing the chat history.",
        "### Task:\nSummarize this document.",
        "Explain this prompt: ### Task:\nGenerate a concise title summarizing the chat history.",
        "summarize: document.pdf",
    ],
)
def test_title_detection_does_not_swallow_regular_queries(text):
    assert not _isTitleRequest(_base_req(messages=[{"role": "user", "content": text}]))


def test_title_detection_uses_last_user_message_only():
    req = _base_req(
        messages=[
            {
                "role": "user",
                "content": "### Task:\nGenerate a concise title summarizing the chat history.",
            },
            {"role": "assistant", "content": '{"title": "RAG Chat"}'},
            {"role": "user", "content": "summarize: document.pdf"},
        ]
    )
    assert not _isTitleRequest(req)


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

    @pytest.mark.parametrize(
        "prompt, expected",
        [
            (
                "### Task:\nGenerate a concise, 3-5 word title for the chat.",
                {"title": "RAG Chat"},
            ),
            (
                "### Task:\nGenerate a concise title summarizing the chat history.\n"
                "### Guidelines:\nKeep it short: 2-4 words is best.\n"
                '### Output:\nJSON format: {"title": "your concise title here"}\n'
                "### Chat History:\n<chat_history>\n"
                "USER: summarize: document.pdf\nASSISTANT: \n</chat_history>",
                {"title": "RAG Chat"},
            ),
            ("### Task:\nGenerate 1-3 broad tags for the chat.", {"tags": ["General"]}),
            (
                "### Task:\nSuggest 3-5 relevant follow-up questions for the chat.",
                {"follow_ups": []},
            ),
        ],
    )
    @pytest.mark.parametrize("stream", [False, True])
    def test_housekeeping_bypasses_rag_and_session_updates(
        self, prompt, expected, stream, monkeypatch
    ):
        import Api.ChatCompletionHandler as module

        apply_request = MagicMock(
            side_effect=AssertionError("Unexpected session update")
        )
        monkeypatch.setattr(module, "_applyRequestToSession", apply_request)
        chatter = MagicMock()
        rag = MagicMock()
        ai_helpers = MagicMock()
        req = _base_req(
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            max_tokens=1000,
        )

        response = self._run_handle(
            req, chatter=chatter, rag=rag, ai_helpers=ai_helpers
        )

        if stream:
            assert isinstance(response, StreamingResponse)

            async def collect():
                parts = []
                async for part in response.body_iterator:
                    parts.append(part.decode() if isinstance(part, bytes) else part)
                return "".join(parts)

            events = _run(collect())
            payloads = [
                json.loads(line[len("data: ") :])
                for line in events.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = "".join(
                item["choices"][0]["delta"].get("content", "") for item in payloads
            )
            assert "data: [DONE]" in events
        else:
            assert isinstance(response, JSONResponse)
            content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert json.loads(content) == expected
        apply_request.assert_not_called()
        assert chatter.mock_calls == []
        assert rag.mock_calls == []
        assert ai_helpers.mock_calls == []

    def test_summary_settings_reach_api_session(self):
        chatter = StubChatter()
        chatter.run = MagicMock(wraps=chatter.run)
        req = _base_req(stream=False)
        req.summary_length = "very long"
        req.summary_temperature = 0.0
        req.chunk_overlap = 0
        self._run_handle(req, chatter=chatter)
        session = chatter.run.call_args.args[0]
        assert session.summary_length == "very_long"
        assert session.summary_temperature == 0.0
        assert session.chunk_overlap == 0

    @pytest.mark.parametrize("stream", [False, True])
    def test_markdown_table_reaches_openwebui_unchanged(self, stream, monkeypatch):
        monkeypatch.setenv("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", "0")
        table = (
            "| Animal | Mammal |\n| --- | --- |\n" "| Hedgehogs | yes |\n| Fish | N/A |"
        )

        class MarkdownChatter(StubChatter):
            cfg = StubConfig()

            def run(self, session, *, apiChunkHandler=None, is_streaming=None):
                if apiChunkHandler is not None:
                    apiChunkHandler(self._answer)
                return True, self._answer

        async def collect():
            with ThreadPoolExecutor(max_workers=1) as executor:
                response = await handleRequest(
                    req=_base_req(stream=stream),
                    chatter=MarkdownChatter(answer=table),
                    rag=StubRAG(),
                    aiHelpers=StubAIHelpers(),
                    queryParts=StubQueryParts(),
                    cfg=StubConfig(),
                    lock=asyncio.Lock(),
                    executor=executor,
                )
                if isinstance(response, StreamingResponse):
                    events = [
                        json.loads(chunk.removeprefix("data: ").strip())
                        async for chunk in response.body_iterator
                        if isinstance(chunk, str) and "[DONE]" not in chunk
                    ]
                    return "".join(
                        event["choices"][0]["delta"].get("content", "")
                        for event in events
                    )
                return json.loads(response.body)["choices"][0]["message"]["content"]

        assert _run(collect()) == table

    @pytest.mark.parametrize(
        "query, override",
        [
            ("What does it say?", None),
            ("summarize:", None),
            ("summarize: other.pdf", "other.pdf"),
            ("summary:", None),
            ("summary: other.pdf", "other.pdf"),
        ],
    )
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "selection, filename, path, expected_filter",
        [
            ({"file": "doc.pdf"}, "doc.pdf", None, {"FileName": {"$eq": "doc.pdf"}}),
            (
                {"path": r"D:\docs\doc.pdf"},
                None,
                r"D:\docs\doc.pdf",
                {"FilePath": {"$eq": r"D:\docs\doc.pdf"}},
            ),
        ],
    )
    def test_pinned_document_reaches_chat_and_summary_with_visible_notice(
        self,
        monkeypatch,
        query,
        override,
        stream,
        selection,
        filename,
        path,
        expected_filter,
    ):
        monkeypatch.setenv("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", "0")

        class RecordingChatter(StubChatter):
            cfg = StubConfig()

            def run(self, session, *, apiChunkHandler=None, is_streaming=None):
                if apiChunkHandler is not None:
                    apiChunkHandler(self._answer)
                return True, self._answer

        chatter = RecordingChatter(answer="Document answer.")
        chatter.run = MagicMock(wraps=chatter.run)
        req = _base_req(
            **selection,
            stream=stream,
            messages=[ChatMessage(role="user", content=query)],
        )

        async def collect():
            with ThreadPoolExecutor(max_workers=1) as executor:
                response = await handleRequest(
                    req=req,
                    chatter=chatter,
                    rag=StubRAG(),
                    aiHelpers=StubAIHelpers(),
                    queryParts=StubQueryParts(),
                    cfg=StubConfig(),
                    lock=asyncio.Lock(),
                    executor=executor,
                )
                if isinstance(response, StreamingResponse):
                    events = [
                        json.loads(chunk.removeprefix("data: ").strip())
                        async for chunk in response.body_iterator
                        if isinstance(chunk, str) and "[DONE]" not in chunk
                    ]
                    return "".join(
                        event["choices"][0]["delta"].get("content", "")
                        for event in events
                    )
                return json.loads(response.body)["choices"][0]["message"]["content"]

        content = _run(collect())
        session = chatter.run.call_args.args[0]
        assert session.query == query
        assert session.file_name == filename
        assert session.file_path == path
        assert session.base_kwargs["filter"] == expected_filter
        assert content.startswith(
            f"Pinned document (chat and summarize): {path or filename}"
        )
        assert "Document answer." in content
        if override:
            assert (
                f"Summary target for this turn: {override} (pin unchanged)" in content
            )
        else:
            assert "Summary target for this turn" not in content

    @pytest.mark.parametrize("command", ["b", "b:", " B: "])
    @pytest.mark.parametrize("stream", [False, True])
    def test_terminal_navigation_is_not_sent_to_pipeline(self, command, stream):
        chatter = MagicMock()
        rag = MagicMock()
        ai_helpers = MagicMock()
        req = _base_req(
            stream=stream,
            messages=[ChatMessage(role="user", content=command)],
        )
        response = self._run_handle(
            req, chatter=chatter, rag=rag, ai_helpers=ai_helpers
        )
        if not stream:
            content = json.loads(response.body)["choices"][0]["message"]["content"]
            assert "not available in OpenWebUI" in content
        chatter.run.assert_not_called()
        rag.set_vector_store.assert_not_called()
        ai_helpers.check_user_prompt_with_filter_chain.assert_not_called()

    def test_summary_lookup_failure_preserves_pin_notice(self):
        class MissingDocumentChatter(StubChatter):
            def run(self, session, **kwargs):
                session.indexed_query_error = (
                    "Summarize: no indexed chunks found for missing.pdf."
                )
                return False, session.indexed_query_error

        response = self._run_handle(
            _base_req(
                file="missing.pdf",
                messages=[ChatMessage(role="user", content="summarize:")],
            ),
            chatter=MissingDocumentChatter(),
        )
        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert content.startswith("Pinned document (chat and summarize): missing.pdf")
        assert "no indexed chunks found" in content
        assert "compliance check" not in content

    def test_summary_without_pin_lists_grounded_documents(self):
        class IndexedRag(StubRAG):
            def indexed_file_metadata(self, session, file_names):
                assert file_names == ["doc.pdf", "other.pdf"]
                return [
                    {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf"},
                    {"FileName": "other.pdf", "FilePath": r"D:\docs\other.pdf"},
                ]

        chatter = StubChatter()
        chatter.run = MagicMock(wraps=chatter.run)
        req = _base_req(
            messages=[
                {"role": "user", "content": "What is in these docs?"},
                {
                    "role": "assistant",
                    "content": "Answer\n\n---\n📄 **Sources**\n- doc.pdf\n- other.pdf\n",
                },
                {"role": "user", "content": "summarize:"},
            ]
        )

        response = self._run_handle(req, chatter=chatter, rag=IndexedRag())

        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert "1. 👉 `D:\\docs\\doc.pdf`" in content
        assert "2. 👉 `D:\\docs\\other.pdf`" in content
        assert "`cancel`" in content
        chatter.run.assert_not_called()

    def test_service_numbered_selection_runs_summary_for_selected_path(self):
        selected_path = r"D:\docs\other.pdf"
        listed = ServiceFileSelector.format_choices([r"D:\docs\doc.pdf", selected_path])
        chatter = StubChatter()
        chatter.run = MagicMock(wraps=chatter.run)
        ai_helpers = MagicMock(spec=StubAIHelpers)
        req = _base_req(
            messages=[
                {"role": "user", "content": "What is in these docs?"},
                {"role": "assistant", "content": listed},
                {"role": "user", "content": "2"},
            ]
        )

        response = self._run_handle(req, chatter=chatter, ai_helpers=ai_helpers)

        session = chatter.run.call_args.args[0]
        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert session.query == f"summarize: {selected_path}"
        assert "The hedgehog answer." in content
        ai_helpers.check_user_prompt_with_filter_chain.assert_not_called()

    def test_service_cancel_reply_does_not_run_summary(self):
        listed = ServiceFileSelector.format_choices([r"D:\docs\doc.pdf"])
        chatter = StubChatter()
        chatter.run = MagicMock(wraps=chatter.run)
        req = _base_req(
            messages=[
                {"role": "assistant", "content": listed},
                {"role": "user", "content": "cancel"},
            ]
        )

        response = self._run_handle(req, chatter=chatter)

        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert content == "Summarization cancelled. Nothing was summarized."
        chatter.run.assert_not_called()

    @pytest.mark.parametrize(
        "name, value",
        [
            ("summary_length", "huge"),
            ("summary_temperature", -1.0),
            ("chunk_overlap", -1),
        ],
    )
    def test_invalid_summary_settings_rejected(self, name, value):
        req = _base_req(stream=False)
        setattr(req, name, value)
        with pytest.raises(HTTPException) as exc:
            _applyRequestToSession(req, StubSession(), StubQueryParts(), StubConfig())
        assert exc.value.status_code == 400

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

    def test_grounded_document_link_precedes_answer(self, monkeypatch):
        monkeypatch.setenv("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", "1")
        link_block = "[Grounded document](https://example.test/marked/source.pdf)\n\n"
        monkeypatch.setattr(
            "Api.ChatCompletionHandler.register_marked_documents",
            lambda *_args: link_block,
        )

        response = self._run_handle(
            _base_req(stream=True), answer="The grounded answer."
        )

        assert isinstance(response, JSONResponse)
        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert content.startswith(link_block)
        assert content.index("Grounded document") < content.index(
            "The grounded answer."
        )

    def test_grounded_html_answer_link_is_included(self, monkeypatch):
        monkeypatch.setenv("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", "1")
        link_block = (
            "\n\n---\n**Grounded answer**\n"
            "- [Open formatted HTML answer](https://example.test/marked/token.html)\n"
        )
        monkeypatch.setattr(
            "Api.ChatCompletionHandler.register_grounded_answer",
            lambda *_args: link_block,
        )

        response = self._run_handle(
            _base_req(stream=False), answer="The grounded answer."
        )

        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert link_block in content
        assert content.index("Open formatted HTML answer") < content.index(
            "The grounded answer."
        )

    def test_non_streaming_response_shape(self):
        resp = self._run_handle(_base_req(stream=False))
        body = json.loads(resp.body)
        assert "id" in body
        assert body["object"] == "chat.completion"
        assert "created" in body
        assert "usage" in body

    @pytest.mark.parametrize(
        "command",
        [
            "summarize:",
            "summarize: doc.pdf",
            r"SuMmArIzE: D:\docs\doc.pdf",
        ],
    )
    @pytest.mark.parametrize("stream", [False, True])
    def test_summary_skips_prompt_checks_and_preserves_command(self, command, stream):
        class SummaryChatter(StubChatter):
            cfg = StubConfig()

            def run(self, session, *, apiChunkHandler=None, is_streaming=None):
                if apiChunkHandler is not None:
                    apiChunkHandler(self._answer)
                return True, self._answer

        chatter = SummaryChatter(answer="Document summary with confidence.")
        chatter.run = MagicMock(wraps=chatter.run)
        ai_helpers = MagicMock(spec=StubAIHelpers)
        rag = StubRAG()
        rag.intent_filter = MagicMock()
        req = ChatCompletionRequest(
            model="AnimalDocs",
            stream=stream,
            messages=[
                ChatMessage(role="system", content="Operator instructions."),
                ChatMessage(role="user", content=command),
            ],
        )

        async def collect():
            with ThreadPoolExecutor(max_workers=1) as executor:
                response = await handleRequest(
                    req=req,
                    chatter=chatter,
                    rag=rag,
                    aiHelpers=ai_helpers,
                    queryParts=StubQueryParts(),
                    cfg=StubConfig(),
                    lock=asyncio.Lock(),
                    executor=executor,
                    serviceCfgDebugLevel=0,
                )
                if isinstance(response, StreamingResponse):
                    return "".join(
                        [
                            chunk if isinstance(chunk, str) else bytes(chunk).decode()
                            async for chunk in response.body_iterator
                        ]
                    )
                return bytes(response.body).decode()

        result = _run(collect())

        assert "Document summary with confidence." in result
        chatter.run.assert_called_once()
        assert chatter.run.call_args.args[0].query == command
        ai_helpers.check_user_prompt_with_filter_chain.assert_not_called()
        ai_helpers.check_prompt_with_llm_guard.assert_not_called()
        rag.intent_filter.score_query.assert_not_called()

    # --- happy path: streaming ----------------------------------------------

    def test_streaming_returns_streaming_response(self):
        # Ensure grounding doesn't force buffered reply (which disables streaming)
        old_val = os.environ.get("SERVE_RAGCHATSERVICE_GROUNDED_DOCS")
        os.environ["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"] = "0"
        try:
            req = _base_req(stream=True)
            resp = self._run_handle(req)
            assert isinstance(resp, StreamingResponse)
            assert resp.media_type == "text/event-stream"
        finally:
            if old_val is None:
                os.environ.pop("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", None)
            else:
                os.environ["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"] = old_val

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
                raise CollectionNotFoundError("internal collection detail")

        req = _base_req(stream=False)
        resp = self._run_handle(req, rag=MissingRAG())
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "collection unavailable" in content.lower()
        assert "internal collection detail" not in content
        assert "Reference: ragchat-" in content

    @pytest.mark.parametrize("stream", [False, True])
    def test_embed_model_mismatch_advises_selecting_compatible_collection(self, stream):
        from Commons.Exceptions import EmbedModelMismatch

        class MismatchedRAG:
            def set_vector_store(self, session):
                raise EmbedModelMismatch("private stored and current model details")

        req = _base_req(stream=stream)
        resp = self._run_handle(req, rag=MismatchedRAG())
        if stream:
            assert isinstance(resp, StreamingResponse)

            async def collect():
                chunks: list[str] = []
                async for chunk in resp.body_iterator:
                    chunks.append(
                        chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                    )
                return "".join(chunks)

            response_text = _run(collect())
            payloads = [
                json.loads(line[len("data: ") :])
                for line in response_text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = "".join(
                item["choices"][0]["delta"].get("content", "") for item in payloads
            )
            assert "data: [DONE]" in response_text
        else:
            content = json.loads(resp.body)["choices"][0]["message"]["content"]

        assert "\u26a0\ufe0f" in content
        assert "select a collection built with the active embedder" in content
        assert "private stored and current model details" not in content
        assert "Reference: ragchat-" in content

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
        assert "request could not be completed" in content.lower()
        assert "LLM timed out" not in content

    def test_translation_error_is_not_reported_as_compliance_refusal(self):
        class _TranslationFailure(StubChatter):
            def run(self, session, **kwargs):
                session.indexed_query_error = (
                    "Retrieval stopped: Cannot translate document language 'fr'."
                )
                return False, session.indexed_query_error

        response = self._run_handle(
            _base_req(stream=False), chatter=_TranslationFailure()
        )
        content = json.loads(response.body)["choices"][0]["message"]["content"]
        assert "Retrieval stopped" in content and "'fr'" in content
        assert "compliance check" not in content

    def test_streamed_translation_error_is_not_followed_by_compliance_refusal(self):
        from types import SimpleNamespace
        from Api.ChatCompletionHandler import _streamGenerator

        class _TranslationFailure(StubChatter):
            cfg = StubConfig()

            def run(self, session, *, apiChunkHandler=None, **kwargs):
                session.indexed_query_error = (
                    "Retrieval stopped: Cannot translate document language 'fr'."
                )
                apiChunkHandler(session.indexed_query_error)
                return False, session.indexed_query_error

        async def collect():
            lock = asyncio.Lock()
            await lock.acquire()
            session = SimpleNamespace(indexed_query_error=None)
            with ThreadPoolExecutor(max_workers=1) as executor:
                chunks = [
                    chunk
                    async for chunk in _streamGenerator(
                        _TranslationFailure(),
                        session,
                        "request",
                        "model",
                        asyncio.get_running_loop(),
                        executor,
                        lock,
                    )
                ]
            assert not lock.locked()
            return "".join(chunks)

        result = _run(collect())
        assert "Retrieval stopped" in result and "'fr'" in result
        assert "compliance check" not in result
        assert '"finish_reason":"stop"' in result
        assert "[DONE]" in result

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
        assert "http://example/v1/models" not in body["detail"]
        assert "Reference: ragchat-" in body["detail"]

    # --- generic exception → 500 --------------------------------------------

    def test_unexpected_exception_returns_friendly_message(self):
        class BrokenRAG:
            def set_vector_store(self, session):
                raise RuntimeError("disk full")

        req = _base_req(stream=False)
        resp = self._run_handle(req, rag=BrokenRAG())
        body = json.loads(resp.body)
        content = body["choices"][0]["message"]["content"]
        assert "request could not be completed" in content.lower()
        assert "disk full" not in content.lower()
        assert "Reference: ragchat-" in content
