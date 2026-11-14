# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Focused decomposition, local dispatch, coverage and answer-wiring regressions."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from Chat.Chatter import Chatter
from Chat.KeywordActionSplitter import KeywordActionSplit, KeywordActionSplitter
from Chat.RAGChatImpl import RAGChatImpl
from Config.Config import Config
from Configuration.Config_KeywordActionSplitter import (
    _KEYWORD_ACTION_SPLITTER,
    _PROMPT_KEYWORD_ACTION_SPLITTER,
)
from Configuration.Config_RAGChat_Prompts import (
    _PROMPT_KEYWORD_ACTION_ANSWER,
    _PROMPT_KEYWORD_ACTION_OPERATION,
)
from Globals.Session import Session
from Gui.Colors import BROWN
from Helpers.FileUtils import FileUtils
from Orchestrator.Orchestrator import Orchestrator
from Orchestrator.OrchestratorTypes import LocalStageInputs, OriginalQueryLeg
from Strategies.HomeBrewChunkSelector import ScoreRankedSelector, PerFileCapSelector


def _payload() -> dict[str, Any]:
    return {
        "retrieval_queries": ["Where do hedgehogs live?", "What do hedgehogs eat?"],
        "action_part": "Create a list of foods hedgehogs eat.",
        "answer_instruction": "Using only the retrieved context, describe where hedgehogs live and list foods hedgehogs eat.",
        "entities": ["hedgehogs"],
        "key_words": ["hedgehog", "habitat", "diet"],
    }


@pytest.fixture
def splitter() -> KeywordActionSplitter:
    cfg: Any = object.__new__(Config)
    cfg._initialized = True
    cfg._lock = threading.RLock()
    cfg.cfg = {
        "_KEYWORD_ACTION_SPLITTER": _KEYWORD_ACTION_SPLITTER,
        "_PROMPT_KEYWORD_ACTION_SPLITTER": _PROMPT_KEYWORD_ACTION_SPLITTER,
        "_PROMPT_KEYWORD_ACTION_ANSWER": _PROMPT_KEYWORD_ACTION_ANSWER,
        "_PROMPT_KEYWORD_ACTION_OPERATION": _PROMPT_KEYWORD_ACTION_OPERATION,
    }
    cfg.args = {}
    cfg.pretty = MagicMock()
    result = object.__new__(KeywordActionSplitter)
    result.cfg = cfg
    result.pretty = MagicMock()
    result.pretty_always = MagicMock()
    result.file_utils = object.__new__(FileUtils)
    result.helpers = MagicMock()
    result.helpers.get_model_args.return_value = {"MODEL": "test"}
    result.llm_caller = MagicMock()
    result.token_budget = MagicMock()
    result.token_budget.get_effective_context_limit.return_value = 4096
    result.perf_logger = MagicMock()
    return result


def _session(split: KeywordActionSplit | None = None) -> Any:
    return SimpleNamespace(
        query="Where do hedgehogs live and what do they eat? Create a list of foods.",
        user_query_original="Where do hedgehogs live and what do they eat? Create a list of foods.",
        orig_translated_query_en="Where do hedgehogs live and what do they eat? Create a list of foods.",
        retrieval_language="english",
        enable_pronoun_substitution=True,
        debug_level=30,
        debug_mode="is",
        keyword_action_split=split,
        keyword_action_query_ids={},
        keyword_action_scores={},
        keyword_action_language_queries={},
        keyword_action_status="not_run",
        keyword_action_local_needs=0,
        keyword_action_keyword_leg=False,
    )


def test_shared_json_parser_and_complete_instruction(
    splitter: KeywordActionSplitter,
) -> None:
    result = splitter.parse("```json\n" + json.dumps(_payload()) + "\n```")
    assert result.retrieval_queries == _payload()["retrieval_queries"]
    assert "live" in result.answer_instruction and "foods" in result.answer_instruction
    assert splitter.parse(str(_payload())) == result  # shared parser's json5 support


@pytest.mark.parametrize(
    "update",
    [
        {"intent": "retrieve_and_transform"},
        {"key_words": ["hedgehog"]},
        {"key_words": ["compare", "habitat", "diet"]},
        {"action_part": ""},
        {"retrieval_queries": [123]},
        {"answer_instruction": None},
        {"answer_instruction": "Hedgehogs live in forests and eat insects."},
        {"retrieval_queries": ["need " + str(i) for i in range(9)]},
        {"retrieval_queries": []},
    ],
)
def test_invalid_contract_rejected(
    splitter: KeywordActionSplitter, update: dict[str, Any]
) -> None:
    payload = _payload()
    payload.update(update)
    with pytest.raises(ValueError):
        splitter.parse(json.dumps(payload))


def test_generic_keywords_are_removed_without_losing_valid_decomposition(
    splitter: KeywordActionSplitter,
) -> None:
    payload = _payload()
    payload["key_words"] = [
        "hedgehog",
        " LIST ",
        "habitat",
        "Create",
        "diet",
        "card list",
    ]
    result = splitter.parse(json.dumps(payload))
    assert result.key_words == ["hedgehog", "habitat", "diet", "card list"]
    assert result.retrieval_queries == payload["retrieval_queries"]
    assert result.action_part == payload["action_part"]
    assert result.answer_instruction == payload["answer_instruction"]
    assert splitter.pretty.write.call_args.args[0] == "W"
    assert "['LIST', 'Create']" in splitter.pretty.write.call_args.args[2]


def test_keyword_cleanup_does_not_invent_replacement_keywords(
    splitter: KeywordActionSplitter,
) -> None:
    payload = _payload()
    payload["key_words"] = ["animal", "mammal", "list"]
    with pytest.raises(ValueError, match=r"after removing.*\['list'\]; got 2"):
        splitter.parse(json.dumps(payload))


@pytest.mark.parametrize("level,mode", [(0, "is"), (10, "is"), (30, "ge"), (40, "le")])
def test_call_language_pronoun_contract_and_unconditional_messages(
    splitter: KeywordActionSplitter, level: int, mode: str
) -> None:
    session = _session()
    session.debug_level = level
    session.debug_mode = mode
    session.retrieval_language = "german"
    session.enable_pronoun_substitution = False
    splitter.llm_caller.call_llm.return_value = {"content": json.dumps(_payload())}
    assert splitter.split(session) is not None
    request = splitter.llm_caller.call_llm.call_args.kwargs
    assert "MUST be in german" in request["prompt"]
    assert "Pronoun substitution enabled: False" in request["prompt"]
    assert request["streaming"] is False
    assert request["model_role"] == "_ACTIVE_LLM_REWRITE_PROMPT"
    messages = splitter.pretty_always.write.call_args_list
    assert len(messages) == 3
    assert [call.args[0] for call in messages] == ["I", "I", "OK"]
    assert session.query == session.user_query_original


@pytest.mark.parametrize("action_part", [None, "Create a list of foods hedgehogs eat."])
def test_decomposition_has_readable_fields(
    splitter: KeywordActionSplitter, action_part: str | None
) -> None:
    payload = _payload()
    payload["action_part"] = action_part
    splitter.llm_caller.call_llm.return_value = {"content": json.dumps(payload)}
    session = _session()
    session.debug_level = 30
    assert splitter.split(session) is not None
    message = splitter.pretty_always.write.call_args.args[2]
    assert message.splitlines() == [
        "Validated decomposition:",
        "Retrieval queries (2):",
        "[query_id=0] Where do hedgehogs live?",
        "[query_id=1] What do hedgehogs eat?",
        f"Action: {action_part or '(none - no transformation requested)'}",
        f"Answer instruction: {payload['answer_instruction']}",
        "Entities: hedgehogs",
        "Keywords: hedgehog, habitat, diet",
    ]


@pytest.mark.parametrize("response", [{"content": "not JSON"}, {"error": "offline"}])
def test_failures_warn_even_with_debug_off(
    splitter: KeywordActionSplitter, response: dict[str, str]
) -> None:
    session = _session()
    session.debug_level = 0
    splitter.llm_caller.call_llm.return_value = response
    assert splitter.split(session) is None
    assert splitter.pretty.write.call_args.args[0] == "W"
    assert "unchanged query pipeline" in splitter.pretty.write.call_args.args[2]


def test_live_model_answer_entities_are_rejected(
    splitter: KeywordActionSplitter,
) -> None:
    payload = _payload()
    payload["entities"] = ["hedgehogs", "forests", "insects"]
    splitter.llm_caller.call_llm.return_value = {"content": json.dumps(payload)}
    assert splitter.split(_session()) is None
    assert "not grounded" in splitter.pretty.write.call_args.args[2]


def test_no_retrieval_and_no_action_contract(splitter: KeywordActionSplitter) -> None:
    payload = _payload()
    payload.update(retrieval_queries=[], key_words=[], action_part=None)
    assert splitter.parse(json.dumps(payload)).retrieval_queries == []


def test_split_dispatch_reuses_real_local_stages(
    splitter: KeywordActionSplitter,
) -> None:
    from test_orchestrator import LocalHostStub

    class Host(LocalHostStub):
        def _run_indexed_local_retrievers(self, *args: Any, **kwargs: Any) -> Any:
            docs, seed, final = super()._run_indexed_local_retrievers(*args, **kwargs)
            return (
                {
                    label: [
                        Document(
                            page_content=f"{kwargs['primary_query']} {text}",
                            metadata={
                                "chunk_id": f"{label}-{kwargs['primary_query']}-{text}"
                            },
                        )
                        for text in texts
                    ]
                    for label, texts in docs.items()
                },
                seed,
                final,
            )

    host = Host(use_guardrail=True, mode_flags=(False, True, True, True))
    host.bm25_retriever = SimpleNamespace(rrf_k=60)
    split = splitter.parse(json.dumps(_payload()))
    session = _session(split)
    session.vector_weight = 0
    session.bm25_weight = 1
    session.graph_weight = 1
    session.regex_weight = 1
    session.retriever_k = 8
    session.collection_name = "TestCollection"
    session.web_search = False
    orchestrator = Orchestrator(host)
    inputs = LocalStageInputs(
        False,
        True,
        True,
        True,
        True,
        "legacy",
        "guardrail",
        True,
        [],
        None,
        OriginalQueryLeg(False, "", None, "not needed"),
    )
    result = orchestrator._run_split_local_stages(session, inputs)
    keywords = "hedgehog habitat diet"
    assert keywords in host.run_indexed_primary_queries
    assert host.run_indexed_primary_queries.count(split.retrieval_queries[0]) == 3
    assert host.run_indexed_primary_queries.count(split.retrieval_queries[1]) == 3
    assert host.indexed_stage_labels[-1] == ["BM25"]
    assert host.run_indexed_guardrail_queries[-1] == ""
    assert result.local_docs.bm25_docs and result.local_docs.graph_docs
    assert session.keyword_action_query_ids
    assert session.query == session.user_query_original


@pytest.mark.parametrize("selector_class", [ScoreRankedSelector, PerFileCapSelector])
def test_existing_selection_is_not_replaced_by_query_quotas(
    splitter: KeywordActionSplitter, selector_class: Any
) -> None:
    from test_chunk_selector import _StubSession

    session = _StubSession(threshold=0.5, boost=1.0)
    session.keyword_action_split = splitter.parse(json.dumps(_payload()))
    session.final_chunks_to_llm = 2
    session.per_file_limit = 1
    session.keyword_action_scores = {}
    first = Document(
        page_content="A RAM",
        metadata={
            "chunk_id": "a",
            "FilePath": "A",
            "raw_rerank_score": 4.0,
            "rerank_score": 1.0,
        },
    )
    second = Document(
        page_content="A more RAM",
        metadata={
            "chunk_id": "a2",
            "FilePath": "A",
            "raw_rerank_score": 3.0,
            "rerank_score": 0.9,
        },
    )
    other = Document(
        page_content="B RAM",
        metadata={
            "chunk_id": "b",
            "FilePath": "B",
            "raw_rerank_score": 2.0,
            "rerank_score": 0.5,
        },
    )
    missing = Document(
        page_content="unrelated C",
        metadata={
            "chunk_id": "c",
            "FilePath": "C",
            "raw_rerank_score": -5.0,
            "rerank_score": 0.0,
        },
    )
    session.keyword_action_query_ids = {
        "id:a": [0],
        "id:a2": [0],
        "id:b": [1],
        "id:c": [2],
    }
    selector = selector_class(session)
    selector.fileHist = MagicMock()
    expected = (
        [first, second] if selector_class is ScoreRankedSelector else [first, other]
    )
    assert selector.select([first, second, other, missing]) == expected


def test_reranking_uses_need_language_and_keeps_web_query(
    splitter: KeywordActionSplitter,
) -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.pretty = MagicMock()
    impl.cross_encoder_query_instruction = ""
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.x_encoder = MagicMock()
    impl.x_encoder.predict.side_effect = [[4.0, 0.3], [2.0, 5.0], [1.0]]
    session = _session(splitter.parse(json.dumps(_payload())))
    session.keyword_action_query_ids = {"id:a": [0], "id:b": [1]}
    session.keyword_action_language_queries = {
        0: {"de": "Lebensraum Igel"},
        1: {"de": "Nahrung Igel"},
    }
    docs = [
        Document(page_content="habitat", metadata={"chunk_id": "a", "Language": "de"}),
        Document(page_content="diet", metadata={"chunk_id": "b", "Language": "de"}),
        Document(
            page_content="keyword-only hit",
            metadata={"chunk_id": "c", "Language": "de"},
        ),
        Document(page_content="web", metadata={"Source": "Web"}),
    ]
    pairs = [("unchanged web query", doc.page_content) for doc in docs]
    assert impl._predict_keyword_action_scores(session, docs, pairs) == [
        4.0,
        2.0,
        5.0,
        1.0,
    ]
    assert session.keyword_action_scores["id:c"] == {0: 0.3, 1: 5.0}
    assert [call.args[0][0][0] for call in impl.x_encoder.predict.call_args_list] == [
        "Lebensraum Igel",
        "Nahrung Igel",
        "unchanged web query",
    ]


@pytest.mark.parametrize(
    "prompt_case", ["default", "custom", "no_action", "disabled", "invalid"]
)
def test_answer_prompt_uses_complete_request_and_history_keeps_original(
    splitter: KeywordActionSplitter, monkeypatch: pytest.MonkeyPatch, prompt_case: str
) -> None:
    payload = _payload()
    if prompt_case == "no_action":
        payload["action_part"] = None
    session = _session(
        splitter.parse(json.dumps(payload)) if prompt_case != "disabled" else None
    )
    session.debug_level = 0
    if prompt_case == "custom":
        splitter.cfg.cfg["_PROMPT_KEYWORD_ACTION_ANSWER"] = (
            "Custom request: {user_query}\nInstruction: {answer_instruction}{action_instruction}"
        )
        splitter.cfg.cfg["_PROMPT_KEYWORD_ACTION_OPERATION"] = (
            "\nCustom operation: {action_part}"
        )
    elif prompt_case == "invalid":
        splitter.cfg.cfg["_PROMPT_KEYWORD_ACTION_ANSWER"] = "{unknown_placeholder}"
    session.query = "normalized request"
    session.retriever_k = 5
    session.max_output_tokens = 256
    session.temperature = 0.1
    session.top_k = 10
    session.top_p = 0.9
    session.preferred_response_language = "english"
    session.user_language = "english"
    session.max_output_tokens_override = None
    session.context_size_override = None
    session.extraOllamaOptions = None
    session.mark_text = False
    session.web_search = False
    session.ollamaTopLevelParams = None
    chatter: Any = object.__new__(Chatter)
    chatter.cfg = splitter.cfg
    chatter.prompt = "Context: {context}\nQuery: {input}\nLanguage: {response_language}"
    chatter.prompt_name = "test"
    chatter.pretty = MagicMock()
    chatter.pretty_always = MagicMock()
    chatter.llm_model = "test"
    chatter.is_streaming = False
    chatter.use_ollama_gpu = False
    chatter.rag = MagicMock()
    chatter.rag.parse_chat_command.return_value = ("", "")
    chatter.rag.retrieve.return_value = ("habitat and food evidence", 2)
    chatter._build_query_notice = MagicMock(return_value="")
    chatter._resolve_token_params = MagicMock(return_value=(4096, 256))
    chatter.llmCaller = MagicMock()
    chatter.modelOutputAdapter = MagicMock()
    chatter.modelOutputAdapter.interpret.side_effect = RuntimeError(
        "stop after capture"
    )
    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    if prompt_case == "invalid":
        with pytest.raises(KeyError, match="unknown_placeholder"):
            Chatter.run(chatter, session)
        chatter.llmCaller.call_llm.assert_not_called()
        return
    with pytest.raises(RuntimeError, match="stop after capture"):
        Chatter.run(chatter, session)
    prompt = chatter.llmCaller.call_llm.call_args.args[1]
    if prompt_case != "disabled":
        assert payload["answer_instruction"] in prompt
        trace = chatter.pretty_always.write.call_args.args
        assert trace[:2] == ("I", "KeywordActionSplitter")
        assert "Final answer instruction:" in trace[2]
        assert chatter.pretty_always.write.call_args.kwargs["color"] == BROWN
    else:
        chatter.pretty_always.write.assert_not_called()
    if prompt_case == "custom":
        assert "Custom request: normalized request" in prompt
        assert "Custom operation: Create a list of foods hedgehogs eat." in prompt
        assert "Requested operation to execute on the retrieved evidence:" not in prompt
        assert "For a requested list, use Markdown bullets" not in prompt
    elif prompt_case == "default":
        assert "Requested operation to execute on the retrieved evidence:" in prompt
        assert "Create a list of foods hedgehogs eat." in prompt
        assert "For a requested list, use Markdown bullets" in prompt
    else:
        assert "Requested operation to execute on the retrieved evidence:" not in prompt
        assert "Create a list of foods hedgehogs eat." not in prompt
    assert "normalized request" in prompt
    assert session.query == "normalized request"


def test_reset_clears_decomposition_not_feature_override() -> None:
    session: Any = object.__new__(Session)
    session.enable_keyword_action_splitter = True
    session.keyword_action_split = _payload()
    session.keyword_action_query_ids = {"id:a": [0]}
    session.keyword_action_scores = {"id:a": {0: 1.0}}
    Session.reset_turn_state(session)
    assert session.keyword_action_split is None
    assert session.keyword_action_query_ids == {}
    assert session.keyword_action_scores == {}
    assert session.enable_keyword_action_splitter is True


@pytest.mark.parametrize("emoji", [True, False])
def test_splitter_output_is_brown_and_unindented_with_debug_off(
    splitter: KeywordActionSplitter,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    emoji: bool,
) -> None:
    from Gui.Colors import BRIGHT_BLUE, GREEN, RESET
    from Gui.PrettyWriter import PrettyWriter
    from Gui.Symbols import Symbols

    splitter.cfg.cfg.update(DEBUG_LEVEL=0, TERMINAL_LINE_SIZE=2000)
    monkeypatch.setattr(Symbols, "_use_emoji", staticmethod(lambda: emoji))
    monkeypatch.setattr(
        "Chat.KeywordActionSplitter.FileUtils",
        MagicMock(return_value=splitter.file_utils),
    )
    monkeypatch.setattr(
        "Chat.KeywordActionSplitter.TokenBudget",
        MagicMock(return_value=splitter.token_budget),
    )
    monkeypatch.setattr(
        "Chat.KeywordActionSplitter.PerfLogger",
        MagicMock(return_value=splitter.perf_logger),
    )
    normal_writer = PrettyWriter(cfg=splitter.cfg)
    actual = KeywordActionSplitter(
        cfg=splitter.cfg,
        pretty=normal_writer,
        helpers=splitter.helpers,
        llm_caller=splitter.llm_caller,
    )
    splitter.llm_caller.call_llm.return_value = {"content": json.dumps(_payload())}
    session = _session()
    session.debug_level = 0

    assert actual.split(session) is not None

    lines = capsys.readouterr().out.splitlines()
    messages = [line for line in lines if "KeywordActionSplitter" in line]
    blue_bullet = "🔵 " if emoji else f"{BRIGHT_BLUE}●{BROWN}  "
    green_bullet = "🟢 " if emoji else f"{GREEN}●{BROWN}  "
    assert len(messages) == 3
    for line, bullet in zip(messages, [blue_bullet, blue_bullet, green_bullet]):
        assert line.startswith(f"{BROWN}{bullet}")
    assert lines[-1].endswith(RESET)
    assert all(not line.startswith(" ") for line in lines)
    assert normal_writer.always_on is False
