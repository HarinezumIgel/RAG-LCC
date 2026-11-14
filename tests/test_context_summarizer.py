# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false
"""Document-summary command, unlimited retrieval, and token-budget contracts."""

from __future__ import annotations

import ast
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import Chat.ContextSummarizer as summary_module
from Chat.FileSelector import (
    CANCELLED_MESSAGE,
    CliFileSelector,
    FileSelection,
    FileSelector,
    ServiceFileSelector,
)
from Commons.Exceptions import CollectionNotFoundError, LLMResultError
from Config.Config import Config
from Configuration import Config_ContextSummarizer, Config_Grounder
from Chat.ConfidenceEvaluator import ConfidenceEvaluator
from Gui.Colors import GREEN, ORANGE, RESET, YELLOW
from Gui.Symbols import Symbols
from Grounding.SourceReferences import (
    SUMMARY_SCHEMA,
    SummaryStatement,
    parse_summary,
    summary_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = "Summarizing document: `D:\\docs\\doc.pdf`\n\n"


def _response(text="The resulting summary.", ids=()):
    return {"content": summary_payload([SummaryStatement(text, tuple(ids))])}


def _shell(
    relative_path: str, class_name: str, methods: set[str], **namespace: Any
) -> Any:
    source = ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))
    original = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    if class_name == "RAGChatImpl" and "_chroma_where" in methods:
        executor = ast.parse(
            (ROOT / "src/Chat/LocalRetrieverExecutor.py").read_text(encoding="utf-8")
        )
        executor_class = next(
            node for node in executor.body if isinstance(node, ast.ClassDef)
        )
        selected.extend(
            node
            for node in executor_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_chroma_where"
        )
    assert len(selected) == len(methods)
    cls = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[class_name]()


@pytest.fixture
def rag() -> Any:
    import ntpath

    source = ast.parse(
        (ROOT / "src/Chat/RAGChatImpl.py").read_text(encoding="utf-8-sig")
    )
    commands = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_CHAT_COMMANDS"
    )
    shell = _shell(
        "src/Chat/RAGChatImpl.py",
        "RAGChatImpl",
        {
            "parse_chat_command",
            "summary_selection_warning",
            "retrieve_summary_documents",
            "_chroma_where",
            "_prepare_session",
            "indexed_file_metadata",
        },
        ntpath=ntpath,
        Document=Document,
        FileSelector=FileSelector,
        _CHAT_COMMANDS=commands,
        CYAN="",
    )
    shell._lock = threading.Lock()
    shell.pretty = MagicMock()
    shell.collection = MagicMock()
    shell.collection_name = "selected-collection"
    shell._set_vector_store = MagicMock(return_value=True)
    shell.collection.get.return_value = {
        "documents": ["last page", "first page"],
        "metadatas": [
            {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf", "PageNumber": 2},
            {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf", "PageNumber": 1},
        ],
    }
    return shell


@pytest.fixture
def session() -> Any:
    return SimpleNamespace(
        file_name="doc.pdf",
        file_path=None,
        query="summarize:",
        collection_name="selected-collection",
        reset_turn_state=MagicMock(),
        retriever_k=1,
        final_chunks_to_llm=1,
        metadata_filters={"Language": "german"},
        base_kwargs={"k": 1},
        web_search=True,
        chroma_threshold=0.99,
        per_file_limit=1,
        preferred_response_language=None,
        mark_text=True,
        summary_length=None,
        summary_temperature=None,
        chunk_overlap=None,
        use_chat_context=True,
    )


@pytest.fixture
def summarizer(monkeypatch: pytest.MonkeyPatch) -> Any:
    values = {
        "CHUNK_OVERLAP": 5,
        "SUMMARY_PROMPT_TEMPLATE": "Summarize faithfully.",
        "_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE": "Combine faithfully.",
        "SUMMARY_TEMPERATURE": 0.4,
        "SUMMARY_LENGTH": "medium",
        "SUMMARY_WORD_LIMITS": {
            preset: dict(limits)
            for preset, limits in Config_Grounder.SUMMARY_WORD_LIMITS.items()
        },
        "DEBUG_LEVEL": "none",
        "_ACTIVE_ENDPOINT": "vllm",
    }
    cfg = MagicMock()

    def config_value(key: str, *args: object) -> Any:
        return values[key]

    cfg.get_int.side_effect = config_value
    cfg.get_float.side_effect = config_value
    cfg.get_str.side_effect = config_value
    cfg.get.side_effect = config_value
    helpers = MagicMock()
    helpers.get_model_args.return_value = {"MODEL": "configured-mistral"}
    helpers.get_active_endpoint_args.return_value = {"USE_GPU": False}
    monkeypatch.setattr(summary_module, "Helpers", lambda: helpers)
    monkeypatch.setattr(summary_module, "PrettyWriter", MagicMock())
    caller = MagicMock()
    caller.call_llm.return_value = _response()
    budget = MagicMock()
    budget.get_context_limit.return_value = 1024
    budget.get_effective_context_limit.return_value = 1024
    budget.reserved_system = 5
    budget.reserved_output = 10

    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    budget.count_tokens_approx.side_effect = count_tokens
    budget.compute_dynamic_max_tokens.return_value = 10
    return summary_module.ContextSummarizer(
        cfg=cfg, llm_caller=caller, token_budget=budget
    )


@pytest.mark.parametrize(
    "query, expected",
    [
        ("summarize:", ("summarize", "")),
        ("  SuMmArIzE: doc.pdf", ("summarize", "doc.pdf")),
        ("summary:", ("summarize", "")),
        ("  SuMmArY: doc.pdf", ("summarize", "doc.pdf")),
        (r"summary: D:\docs\doc.pdf", ("summarize", r"D:\docs\doc.pdf")),
        ("summary: cancel", ("summarize", "cancel")),
        ("explain summary: please", ("", "explain summary: please")),
        ("new: a question", ("new", "a question")),
        ("new topic: question", ("new", "question")),
        ("newtopic: question", ("new", "question")),
        ("explain summarize: please", ("", "explain summarize: please")),
    ],
)
def test_command_parser(rag, query, expected):
    assert rag.parse_chat_command(query) == expected


def test_metadata_only_retrieval_ignores_all_ranking_limits(rag, session):
    documents = rag.retrieve_summary_documents(session)
    rag.collection.get.assert_called_once_with(
        where={"FileName": {"$eq": "doc.pdf"}}, include=["documents", "metadatas"]
    )
    assert [doc.page_content for doc in documents] == ["first page", "last page"]
    assert session.retriever_k == 1
    assert session.metadata_filters == {"Language": "german"}
    rag._set_vector_store.assert_not_called()


@pytest.mark.parametrize("loaded", [None, "previous-collection"])
def test_summary_loads_settings_collection_without_picker(rag, session, loaded):
    collection = rag.collection
    rag.collection_name = loaded
    if loaded is None:
        rag.collection = None

    def load_collection(selected):
        assert selected.collection_name == "selected-collection"
        rag.collection_name = selected.collection_name
        rag.collection = collection
        return True

    rag._set_vector_store.side_effect = load_collection
    assert len(rag.retrieve_summary_documents(session)) == 2
    rag._set_vector_store.assert_called_once_with(session)
    collection.get.assert_called_once()


def test_summary_loads_configured_collection_when_session_has_no_override(rag, session):
    session.collection_name = None
    collection = rag.collection
    rag.collection = None

    def load_collection(selected):
        assert selected.collection_name is None
        rag.collection = collection
        return True

    rag._set_vector_store.side_effect = load_collection
    assert len(rag.retrieve_summary_documents(session)) == 2
    rag._set_vector_store.assert_called_once_with(session)


def test_summary_collection_load_failure_does_not_read_previous_collection(
    rag, session
):
    rag.collection_name = "previous-collection"
    rag._set_vector_store.return_value = False
    with pytest.raises(
        ValueError, match="could not load the selected indexed collection"
    ):
        rag.retrieve_summary_documents(session)
    rag.collection.get.assert_not_called()


def test_path_pin_adds_filename_and_path_only(rag, session):
    session.file_name = None
    session.file_path = r"D:\docs\doc.pdf"
    rag.retrieve_summary_documents(session)
    assert rag.collection.get.call_args.kwargs["where"] == {
        "$and": [
            {"FileName": {"$eq": "doc.pdf"}},
            {"FilePath": {"$eq": r"D:\docs\doc.pdf"}},
        ]
    }


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "stored_path",
    [
        "D:/docs/doc.pdf",
        r"d:\docs\doc.pdf",
        "D:/docs/./doc.pdf",
    ],
)
def test_summary_path_fallback_normalizes_without_losing_chunks(
    rag,
    session,
    pinned,
    stored_path,
):
    candidates = {
        "documents": ["last page", "wrong file", "first page"],
        "metadatas": [
            {"FileName": "doc.pdf", "FilePath": stored_path, "PageNumber": 2},
            {"FileName": "doc.pdf", "FilePath": "D:/other/doc.pdf", "PageNumber": 1},
            {"FileName": "doc.pdf", "FilePath": stored_path, "PageNumber": 1},
        ],
    }
    rag.collection.get.side_effect = [
        {"documents": [], "metadatas": []},
        candidates,
    ]
    if pinned:
        session.file_name = None
        session.file_path = r"D:\docs\doc.pdf"
    documents = rag.retrieve_summary_documents(
        session, "" if pinned else r"D:\docs\doc.pdf"
    )
    assert [doc.page_content for doc in documents] == ["first page", "last page"]
    assert all(doc.metadata["FilePath"] == stored_path for doc in documents)
    assert rag.collection.get.call_args.kwargs == {
        "where": {"FileName": {"$eq": "doc.pdf"}},
        "include": ["documents", "metadatas"],
    }


def test_summary_path_fallback_never_uses_another_directory(rag, session):
    rag.collection.get.side_effect = [
        {"documents": [], "metadatas": []},
        {
            "documents": ["wrong file"],
            "metadatas": [
                {"FileName": "doc.pdf", "FilePath": "D:/other/doc.pdf"},
            ],
        },
    ]
    with pytest.raises(ValueError, match="no indexed chunks found"):
        rag.retrieve_summary_documents(session, r"D:\docs\doc.pdf")


@pytest.mark.parametrize(
    "target, path",
    [
        ("other.pdf", None),
        ("   other.pdf   ", None),
        ('"other.pdf"', None),
        ("'other.pdf'", None),
        ('"D:\\elsewhere\\other.pdf"', r"D:\elsewhere\other.pdf"),
        ("'D:\\elsewhere\\other.pdf'", r"D:\elsewhere\other.pdf"),
        (r"D:\elsewhere\other.pdf", r"D:\elsewhere\other.pdf"),
        (r"\\server\share\other.pdf", r"\\server\share\other.pdf"),
    ],
)
def test_explicit_target_is_one_turn_only(rag, session, target, path):
    session.file_path = r"D:\docs\doc.pdf"
    rag.retrieve_summary_documents(session, target)
    where = rag.collection.get.call_args.kwargs["where"]
    assert where == (
        {"FileName": {"$eq": "other.pdf"}}
        if path is None
        else {"$and": [{"FileName": {"$eq": "other.pdf"}}, {"FilePath": {"$eq": path}}]}
    )
    assert session.file_name == "doc.pdf"
    assert session.file_path == r"D:\docs\doc.pdf"


@pytest.mark.parametrize(
    "target",
    [
        "my document.pdf",
        '"my document.pdf"',
        r"D:\my documents\my document.pdf",
        '"D:\\my documents\\my document.pdf"',
    ],
)
def test_target_with_spaces_needs_no_pin(rag, session, target):
    session.file_name = session.file_path = None
    rag.retrieve_summary_documents(session, target)
    if "D:" in target:
        assert rag.collection.get.call_args.kwargs["where"] == {
            "$and": [
                {"FileName": {"$eq": "my document.pdf"}},
                {"FilePath": {"$eq": r"D:\my documents\my document.pdf"}},
            ]
        }
    else:
        assert rag.collection.get.call_args.kwargs["where"] == {
            "FileName": {"$eq": "my document.pdf"}
        }


@pytest.mark.parametrize(
    "target, expected",
    [
        ('"doc.pdf', '"doc.pdf'),
        ("doc.pdf'", "doc.pdf'"),
        ("\"doc.pdf'", "\"doc.pdf'"),
        ("'doc.pdf\"", "'doc.pdf\""),
        ("\"'doc.pdf'\"", "'doc.pdf'"),
        ("doc'ument.pdf", "doc'ument.pdf"),
    ],
)
def test_summary_only_removes_one_matching_outer_quote_pair(
    rag, session, target, expected
):
    rag.retrieve_summary_documents(session, target)
    assert rag.collection.get.call_args.kwargs["where"] == {
        "FileName": {"$eq": expected}
    }


def test_missing_pin_stops_before_database_access(rag, session):
    session.file_name = session.file_path = None
    with pytest.raises(ValueError, match="requires a file to be selected"):
        rag.retrieve_summary_documents(session)
    rag.collection.get.assert_not_called()


def test_ambiguous_filename_requires_path(rag, session):
    rag.collection.get.return_value["metadatas"][1]["FilePath"] = r"D:\other\doc.pdf"
    with pytest.raises(ValueError, match="multiple files"):
        rag.retrieve_summary_documents(session)


def test_no_indexed_results_is_explicit(rag, session):
    rag.collection.get.return_value = {"documents": [], "metadatas": []}
    with pytest.raises(ValueError, match="no indexed chunks"):
        rag.retrieve_summary_documents(session)


@pytest.mark.parametrize("stored_path", [r"D:\docs\doc.pdf", "D:/docs/doc.pdf"])
def test_actual_chroma_fetches_more_than_session_top_k(
    rag, session, tmp_path, stored_path
):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "summary-index"))
    collection = client.create_collection("summary-test")
    try:
        collection.add(
            ids=[str(i) for i in range(31)],
            documents=[f"chunk {i}" for i in range(31)],
            metadatas=[
                {
                    "FileName": "doc.pdf",
                    "FilePath": stored_path,
                    "Language": "english" if i % 2 else "german",
                }
                for i in range(31)
            ],
            embeddings=[[1.0, 0.0] for _ in range(31)],
        )
        rag.collection = collection
        assert len(rag.retrieve_summary_documents(session)) == 31
        assert len(rag.retrieve_summary_documents(session, r"D:\docs\doc.pdf")) == 31
    finally:
        client.delete_collection("summary-test")


def test_budgeted_batches_include_every_chunk_and_whole_chunk_overlap(summarizer):
    documents = [Document(page_content=f"CHUNK{i} " + "word " * 22) for i in range(10)]
    assert summarizer.summarize_documents(documents) == "The resulting summary."
    calls = summarizer.llm_caller.call_llm.call_args_list
    prompts = [call.kwargs["prompt"] for call in calls[:-1]]
    for i in range(10):
        assert any(f"CHUNK{i} " in prompt for prompt in prompts)
    assert "CHUNK1 " in prompts[0] and "CHUNK1 " in prompts[1]
    assert len(calls) <= len(documents) + 1
    for call in calls:
        kwargs = call.kwargs
        assert kwargs["model"] == "configured-mistral"
        assert kwargs["model_role"] == "_ACTIVE_LLM"
        assert kwargs["answer_is_json"] is True
        assert (
            kwargs["top_level_params"]["response_format"]["json_schema"]["schema"]
            == SUMMARY_SCHEMA
        )
        assert kwargs["ollama_options"]["num_ctx"] == 1024
        assert kwargs["ollama_options"]["num_predict"] == 10
        assert kwargs["ollama_options"]["num_gpu"] == 0
        assert len(kwargs["prompt"].encode("utf-8")) + 15 <= 1024
    assert calls[0].kwargs["ollama_options"]["temperature"] == 0.4
    assert calls[-1].kwargs["ollama_options"]["temperature"] == 0.4
    assert summarizer.chunk_summaries
    assert summarizer.summary == "The resulting summary."


def test_oversized_chunk_is_split_without_losing_tail(summarizer):
    text = " ".join(f"WORD{i}" for i in range(350))
    summarizer.summarize_documents([Document(page_content=text)])
    prompts = [
        " ".join(
            item.text
            for item in parse_summary(call.kwargs["prompt"].split("\n\n", 1)[1])
        )
        for call in summarizer.llm_caller.call_llm.call_args_list
        if call.kwargs["stage"] == "Summarize document chunks"
    ]
    for word in text.split():
        assert any(word in prompt.split() for prompt in prompts)
    map_calls = [
        call
        for call in summarizer.llm_caller.call_llm.call_args_list
        if call.kwargs["stage"] == "Summarize document chunks"
    ]
    assert all(
        all(
            item.context_chunk_ids == (1,)
            for item in parse_summary(call.kwargs["prompt"].split("\n\n", 1)[1])
        )
        for call in map_calls
    )


def test_chunk_ids_survive_map_reduce_and_final_summary(summarizer):
    def cited_response(**kwargs):
        body = kwargs["prompt"].split("\n\n", 1)[1]
        ids = {i for item in parse_summary(body) for i in item.context_chunk_ids}
        assert ids
        return _response(
            "Memory allocation reserves storage for program variables.", [min(ids)]
        )

    summarizer.llm_caller.call_llm.side_effect = cited_response
    docs = [
        Document(page_content=f"Source {i} " + "memory storage " * 10)
        for i in range(40)
    ]
    result = summarizer.summarize_documents(docs)
    assert summarizer.reduction_summaries
    assert result == "Memory allocation reserves storage for program variables."
    assert summarizer.statements[0].context_chunk_ids == (1,)
    assert [doc.metadata["SummaryChunkId"] for doc in docs] == list(range(1, 41))
    for call in summarizer.llm_caller.call_llm.call_args_list:
        assert "never invent IDs" in call.kwargs["prompt"]
        assert len(call.kwargs["prompt"].encode("utf-8")) + 15 <= 1024


@pytest.mark.parametrize(
    "final, input_description",
    [
        (False, "this input batch"),
        (True, "input summaries"),
    ],
)
def test_invented_chunk_id_is_discarded_with_warning(
    summarizer, final, input_description
):
    summarizer.llm_caller.call_llm.side_effect = [
        _response("A source-backed statement.", [1] if final else [999]),
        _response("A summary statement.", [999] if final else []),
    ]
    result = summarizer.summarize_documents(
        [Document(page_content="some source content")]
    )
    assert result == "A summary statement."
    assert summarizer.statements[0].context_chunk_ids == ()
    assert any(
        f"context chunk IDs absent from {input_description}: [999]" in str(call)
        for call in summarizer.pretty.write.call_args_list
    )


def test_plain_text_summary_fails_explicitly(summarizer):
    summarizer.llm_caller.call_llm.return_value = {"content": "A plain text summary."}
    with pytest.raises(LLMResultError, match="Invalid structured"):
        summarizer.summarize_documents([Document(page_content="some source content")])


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("ollama", {"format": SUMMARY_SCHEMA}),
        (
            "vllm",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "document_summary",
                        "strict": True,
                        "schema": SUMMARY_SCHEMA,
                    },
                }
            },
        ),
    ],
)
def test_summary_requests_backend_json_mode(summarizer, endpoint, expected):
    original = summarizer.cfg.get_str.side_effect
    summarizer.cfg.get_str.side_effect = lambda key, *args: (
        endpoint if key == "_ACTIVE_ENDPOINT" else original(key, *args)
    )
    summarizer.summarize_documents([Document(page_content="some source content")])
    assert all(
        call.kwargs["top_level_params"] == expected
        for call in summarizer.llm_caller.call_llm.call_args_list
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "null",
        "[]",
        "{}",
        '{"statements":[]}',
        '{"statements":[{"text":"", "context_chunk_ids":[1]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":["1"]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[true]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[1.5]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[0]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[-1]}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":null}]}',
        '{"statements":[{"text":"Summary"}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[1], "extra":2}]}',
        '{"statements":[{"text":"Summary", "context_chunk_ids":[1]}], "extra":2}',
    ],
)
def test_invalid_structured_summary_stops_explicitly(summarizer, session, payload):
    session.summary_statements = [SummaryStatement("Stale summary", (42,))]
    summarizer.llm_caller.call_llm.return_value = {"content": payload}
    with pytest.raises(LLMResultError, match="Invalid structured document summary"):
        summarizer.summarize_documents(
            [Document(page_content="source")], session=session
        )
    assert session.summary_statements == []
    assert summarizer.summary == ""
    assert summarizer.llm_caller.call_llm.call_count == 1


def test_structured_summary_reaches_grounder_without_inline_ids(
    summarizer, session, tmp_path
):
    from Grounding.Grounder import Grounder

    text = "Memory allocation reserves storage for program variables."
    summarizer.llm_caller.call_llm.return_value = _response(text, [1, 1, 999])
    path = tmp_path / "source.pdf"
    path.touch()
    documents = [
        Document(
            page_content="Storage for program variables is reserved by memory allocation.",
            metadata={"FilePath": str(path), "PageNumber": 3},
        )
    ]
    answer = summarizer.summarize_documents(documents, session=session)
    assert answer == text
    assert session.summary_statements == [SummaryStatement(text, (1,))]
    session.summary_grounding = True
    session.last_answer_content = answer
    session.chunk_texts_for_grounding = [documents[0].page_content]
    session.grounding_proxy_text = ""
    session.grounding_proxy_sentences = {}
    Grounder._reset()
    try:
        snippets = Grounder().build_grounded_snippets(session, documents, "orange")
        assert snippets[str(path)][0].text == documents[0].page_content
        assert snippets[str(path)][0].page_number == 3
    finally:
        Grounder._reset()


def test_recursive_reduction_fits_every_prompt(summarizer):
    summarizer.token_budget.get_context_limit.return_value = 640
    summarizer.llm_caller.call_llm.return_value = _response(
        "one two three four five six seven eight"
    )
    docs = [Document(page_content=f"chunk{i} " + "word " * 13) for i in range(60)]
    summarizer.summarize_documents(docs)
    assert len(summarizer.reduction_summaries) >= 2
    for call in summarizer.llm_caller.call_llm.call_args_list:
        assert len(call.kwargs["prompt"].encode("utf-8")) + 15 <= 640


def test_progress_counts_successful_calls_per_stage(summarizer, monkeypatch):
    bars: list[Any] = []

    def make_progress(**kwargs: Any) -> Any:
        bar = MagicMock()
        bar.total = kwargs["total"]
        bar.desc = kwargs["desc"]
        bar.__enter__.return_value = bar
        bars.append(bar)
        return bar

    monkeypatch.setattr(summary_module, "tqdm", make_progress)
    summarizer.token_budget.get_context_limit.return_value = 640
    summarizer.llm_caller.call_llm.return_value = _response(
        "one two three four five six seven eight"
    )
    summarizer.summarize_documents(
        [Document(page_content=f"chunk{i} " + "word " * 13) for i in range(60)]
    )

    assert bars[0].desc == "Summarize batches"
    assert bars[0].total == len(summarizer.chunk_summaries)
    assert len(bars) == len(summarizer.reduction_summaries) + 2
    for pass_number, summaries in enumerate(summarizer.reduction_summaries, 1):
        assert bars[pass_number].desc == f"Reduce summaries (pass {pass_number})"
        assert bars[pass_number].total == len(summaries)
    assert bars[-1].desc == "Final summary" and bars[-1].total == 1
    for bar in bars:
        completed_before_bar = 1 if bar is bars[0] else 0
        assert bar.update.call_args_list == [((1,), {})] * (
            bar.total - completed_before_bar
        )
        bar.__exit__.assert_called_once()
    assert sum(bar.total for bar in bars) == summarizer.llm_caller.call_llm.call_count
    summarizer.pretty.write.assert_called_once_with(
        "W",
        "Summarize",
        "Summary contains no context chunk IDs; grounding will search all chunks.",
    )


def test_progress_renders_counts(summarizer, monkeypatch):
    from functools import partial
    from io import StringIO

    output = StringIO()
    monkeypatch.setattr(
        summary_module,
        "tqdm",
        partial(summary_module.tqdm, file=output, mininterval=0, ascii=True),
    )
    summarizer.summarize_documents([Document(page_content="some source content")])
    rendered = output.getvalue()
    assert "Summarize batches:" in rendered
    assert "Final summary:" in rendered
    assert "1/1" in rendered
    assert "100%" in rendered


def test_failed_model_call_does_not_advance_progress(summarizer, monkeypatch):
    factory = MagicMock()
    bar = factory.return_value.__enter__.return_value
    monkeypatch.setattr(summary_module, "tqdm", factory)
    summarizer.llm_caller.call_llm.return_value = {"error": "backend failed"}

    with pytest.raises(LLMResultError, match="backend failed"):
        summarizer.summarize_documents([Document(page_content="some source content")])

    factory.assert_not_called()
    bar.update.assert_not_called()
    factory.return_value.__exit__.assert_not_called()


def test_only_first_summary_call_emits_diagnostics(summarizer, monkeypatch, capsys):
    from Gui.PrettyWriter import PrettyWriter

    writer = PrettyWriter(always_on=True)
    monkeypatch.setattr(summary_module, "PrettyWriter", PrettyWriter)

    def generate(**kwargs: Any) -> dict[str, str]:
        writer.write("I", "Call LLM", "inference diagnostic", max_line_length=160)
        return _response()

    summarizer.llm_caller.call_llm.side_effect = generate
    summarizer.summarize_documents(
        [Document(page_content="source " * 22) for _ in range(6)]
    )
    assert summarizer.llm_caller.call_llm.call_count > 1
    assert capsys.readouterr().out.count("inference diagnostic") == 1
    writer.write("I", "Other", "normal output restored", max_line_length=160)
    assert "normal output restored" in capsys.readouterr().out


def test_quiet_summary_context_preserves_warnings_and_restores_after_error(capsys):
    from Gui.PrettyWriter import PrettyWriter

    writer = PrettyWriter(always_on=True)
    with pytest.raises(ValueError):
        with writer.suppress_info():
            writer.write("I", "Hidden", "hidden diagnostic", max_line_length=160)
            writer.write("W", "Warning", "visible warning", max_line_length=160)
            writer.write("E", "Error", "visible error", max_line_length=160)
            raise ValueError("failed")
    writer.write("I", "Restored", "restored diagnostic", max_line_length=160)
    output = capsys.readouterr().out
    assert "hidden diagnostic" not in output
    assert "visible warning" in output and "visible error" in output
    assert "restored diagnostic" in output


@pytest.mark.parametrize("response", [{"error": "backend failed"}, {"content": ""}])
def test_llm_failure_is_not_returned_as_source_text(summarizer, response):
    summarizer.llm_caller.call_llm.return_value = response
    with pytest.raises(LLMResultError):
        summarizer.summarize_documents([Document(page_content="some source content")])
    assert summarizer.summary == ""


def test_non_shrinking_reduction_raises_instead_of_looping(summarizer):
    summarizer.token_budget.get_context_limit.return_value = 640

    def echo_prompt(**kwargs: Any) -> dict[str, str]:
        return {"content": str(kwargs["prompt"]).split("\n\n", 1)[1]}

    summarizer.llm_caller.call_llm.side_effect = echo_prompt
    with pytest.raises(LLMResultError, match="did not shrink"):
        summarizer.summarize_documents(
            [Document(page_content="word " * 14) for _ in range(10)]
        )


def test_empty_source_never_calls_llm(summarizer):
    with pytest.raises(ValueError, match="no indexed text"):
        summarizer.summarize_documents([Document(page_content=" ")])
    summarizer.llm_caller.call_llm.assert_not_called()


def test_session_context_override_is_used(summarizer, session):
    summarizer.token_budget.get_effective_context_limit.return_value = 640
    summarizer.summarize_documents([Document(page_content="source")], session=session)
    summarizer.token_budget.get_effective_context_limit.assert_called_once_with(
        "configured-mistral", session
    )
    assert (
        summarizer.llm_caller.call_llm.call_args.kwargs["ollama_options"]["num_ctx"]
        == 640
    )


@pytest.mark.parametrize(
    "detected,override,expected",
    [
        (65536, None, 32768),
        (16384, None, 16384),
        (32768, 8192, 8192),
    ],
)
def test_dense_pdf_batches_respect_model_and_dynamic_context_limits(
    summarizer,
    session,
    detected,
    override,
    expected,
):
    summarizer.helpers.get_model_args.return_value = {
        "MODEL": "mistral_7b",
        "TOKEN_BUDGET_CONTEXT_CAP": 32768,
        "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
        "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
    }
    summarizer.token_budget.get_context_limit.return_value = detected
    summarizer.token_budget.get_effective_context_limit.return_value = min(
        detected, override if override is not None else detected
    )
    summarizer.token_budget.compute_dynamic_max_tokens.return_value = 2048
    session.context_size_override = override
    # One "word" can represent thousands of model tokens in a PDF code listing.
    source = "0123456789abcdef" * 6602
    summarizer.summarize_documents([Document(page_content=source)], session=session)

    calls = summarizer.llm_caller.call_llm.call_args_list
    assert len(calls) > 2
    mapped_source = "".join(
        "".join(
            item.text
            for item in parse_summary(call.kwargs["prompt"].split("\n\n", 1)[1])
        )
        for call in calls
        if call.kwargs["stage"] == "Summarize document chunks"
    )
    assert mapped_source == source
    for call in calls:
        options = call.kwargs["ollama_options"]
        prompt = call.kwargs["prompt"]
        assert options["num_ctx"] == expected
        assert options["num_predict"] == 2048
        assert len(prompt.encode("utf-8")) + 1024 + 2048 <= expected
        assert "summarize" in prompt.lower() or "combine" in prompt.lower()


def test_oversized_splitting_preserves_whitespace_and_unicode(summarizer):
    summarizer.context_limit = 480
    summarizer.prompt_limit = 465
    source = ("  MOV eax, 0x1234\n\t" + "\u65e5\u672c\u8a9e " * 20) * 20
    parts = summarizer._split_statement(
        SummaryStatement(source, (7,)), "Summarize faithfully."
    )
    assert len(parts) > 1
    assert "".join(part.text for part in parts) == source
    assert all(part.context_chunk_ids == (7,) for part in parts)
    assert all(summarizer._fits([part], "Summarize faithfully.") for part in parts)


@pytest.mark.parametrize(
    "reserved_system,reserved_output", [(1100, 10), (-1, 10), (5, 0)]
)
def test_invalid_or_exhausted_model_reserves_fail_before_inference(
    summarizer,
    reserved_system,
    reserved_output,
):
    summarizer.helpers.get_model_args.return_value = {
        "MODEL": "configured-mistral",
        "TOKEN_BUDGET_RESERVED_SYSTEM": reserved_system,
        "TOKEN_BUDGET_RESERVED_OUTPUT": reserved_output,
    }
    with pytest.raises(ValueError, match="reserves"):
        summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.assert_not_called()


def test_oversized_final_instruction_fails_before_map_calls(summarizer):
    original_config = summarizer.cfg.get_str.side_effect

    def config_value(key: str, *args: object) -> Any:
        if key == "_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE":
            return "instruction " * 100
        return original_config(key, *args)

    summarizer.cfg.get_str.side_effect = config_value
    with pytest.raises(ValueError, match="cannot fit the summary prompt"):
        summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.assert_not_called()


def test_dynamic_output_budget_is_respected(summarizer):
    summarizer.token_budget.compute_dynamic_max_tokens.return_value = 3
    summarizer.summarize_documents([Document(page_content="source")])
    for call in summarizer.llm_caller.call_llm.call_args_list:
        assert call.kwargs["ollama_options"]["num_predict"] == 3


def test_nonpositive_dynamic_output_budget_fails_before_inference(summarizer):
    summarizer.token_budget.compute_dynamic_max_tokens.return_value = 0
    with pytest.raises(ValueError, match="no output tokens"):
        summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.assert_not_called()


def test_last_moment_guard_rejects_an_oversized_prompt(summarizer):
    summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.reset_mock()
    with pytest.raises(ValueError, match="prompt exceeds"):
        summarizer._generate_summary(
            [SummaryStatement("0x0123456789abcdef" * 100, (1,))],
            summarizer.final_prompt_template,
            final=True,
        )
    summarizer.llm_caller.call_llm.assert_not_called()


@pytest.mark.parametrize(
    "length,intermediate_words,final_words",
    [
        ("short", 150, 400),
        ("medium", 300, 1000),
        ("long", 600, 2000),
        ("very_long", 1000, 4000),
    ],
)
def test_summary_settings_apply_to_all_stages(
    summarizer,
    session,
    length,
    intermediate_words,
    final_words,
):
    session.summary_length = length
    session.summary_temperature = 0.0
    session.chunk_overlap = 0
    summarizer.summarize_documents(
        [Document(page_content="source " * 22) for _ in range(6)], session=session
    )
    instruction = summary_module.ContextSummarizer.LENGTH_INSTRUCTIONS[length]
    calls = summarizer.llm_caller.call_llm.call_args_list
    for call in calls:
        assert instruction in call.kwargs["prompt"]
        assert call.kwargs["ollama_options"]["temperature"] == 0.0
        words = final_words if call is calls[-1] else intermediate_words
        assert f"at most {words} words" in call.kwargs["prompt"]
    assert summarizer.chunk_overlap == 0
    assert summarizer.llm_caller.call_llm.call_count > 1


def test_configured_word_budgets_are_used_for_recursive_reduction(summarizer):
    summarizer.cfg.get("SUMMARY_WORD_LIMITS")["medium"] = {
        "intermediate": 37,
        "final": 91,
    }
    summarizer.token_budget.get_context_limit.return_value = 640
    summarizer.llm_caller.call_llm.return_value = _response(
        "one two three four five six seven eight"
    )
    summarizer.summarize_documents(
        [Document(page_content=f"chunk{i} " + "word " * 13) for i in range(60)]
    )
    assert len(summarizer.reduction_summaries) >= 2
    calls = summarizer.llm_caller.call_llm.call_args_list
    assert len(calls) == (
        len(summarizer.chunk_summaries)
        + sum(len(results) for results in summarizer.reduction_summaries)
        + 1
    )
    for call in calls[:-1]:
        assert "at most 37 words" in call.kwargs["prompt"]
        assert "at most 91 words" not in call.kwargs["prompt"]
    assert "at most 91 words" in calls[-1].kwargs["prompt"]


def test_prompt_templates_resolve_numeric_word_placeholder(summarizer):
    original_config = summarizer.cfg.get_str.side_effect

    def config_value(key: str, *args: object) -> Any:
        if key == "SUMMARY_PROMPT_TEMPLATE":
            return Config_ContextSummarizer.SUMMARY_PROMPT_TEMPLATE
        if key == "_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE":
            return Config_ContextSummarizer._PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE
        return original_config(key, *args)

    summarizer.cfg.get_str.side_effect = config_value
    summarizer.token_budget.get_context_limit.return_value = 2000
    summarizer.summarize_documents([Document(page_content="some source content")])
    calls = summarizer.llm_caller.call_llm.call_args_list
    assert len(calls) == 2
    assert (
        calls[0]
        .kwargs["prompt"]
        .startswith(
            Config_ContextSummarizer.SUMMARY_PROMPT_TEMPLATE.replace(
                "{max_words}", "300"
            )
        )
    )
    assert (
        calls[-1]
        .kwargs["prompt"]
        .startswith(
            Config_ContextSummarizer._PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE.replace(
                "{max_words}", "1000"
            )
        )
    )
    assert all("{max_words}" not in call.kwargs["prompt"] for call in calls)


@pytest.mark.parametrize(
    "limits",
    [
        None,
        [],
        {},
        {"medium": None},
        {"medium": 150},
        {"medium": {"intermediate": 150}},
        {"medium": {"final": 400}},
    ],
)
def test_missing_or_malformed_word_budgets_fail_before_inference(summarizer, limits):
    original_config = summarizer.cfg.get.side_effect

    def config_value(key: str, *args: object) -> Any:
        if key == "SUMMARY_WORD_LIMITS":
            return limits
        return original_config(key, *args)

    summarizer.cfg.get.side_effect = config_value
    with pytest.raises(ValueError, match="SUMMARY_WORD_LIMITS"):
        summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.assert_not_called()


@pytest.mark.parametrize("stage", ["intermediate", "final"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "150", None])
def test_invalid_word_budget_fails_before_inference(summarizer, stage, value):
    summarizer.cfg.get("SUMMARY_WORD_LIMITS")["medium"][stage] = value
    with pytest.raises(ValueError, match=rf"SUMMARY_WORD_LIMITS.medium.{stage}"):
        summarizer.summarize_documents([Document(page_content="source")])
    summarizer.llm_caller.call_llm.assert_not_called()


def test_cleared_summary_overrides_restore_configuration(summarizer, session):
    session.summary_length = "short"
    session.summary_temperature = 0.0
    session.chunk_overlap = 0
    summarizer.summarize_documents([Document(page_content="source")], session=session)
    session.summary_length = session.summary_temperature = session.chunk_overlap = None
    summarizer.summarize_documents([Document(page_content="source")], session=session)
    assert summarizer.summary_temperature == 0.4
    assert summarizer.chunk_overlap == 5
    assert "at most 300 words" in summarizer.summary_prompt_template
    assert "at most 1000 words" in summarizer.final_prompt_template


@pytest.mark.parametrize(
    "name, value",
    [
        ("summary_length", "huge"),
        ("summary_temperature", -0.1),
        ("summary_temperature", 1.1),
        ("summary_temperature", "nan"),
        ("chunk_overlap", -1),
        ("chunk_overlap", 1.5),
    ],
)
def test_invalid_summary_setting_is_rejected(name, value):
    with pytest.raises(ValueError):
        summary_module.ContextSummarizer.validate_setting(name, value)


def test_config_module_is_loaded_by_config_manager():
    cfg = Config()
    assert cfg.get_int("CHUNK_OVERLAP") == Config_ContextSummarizer.CHUNK_OVERLAP
    assert cfg.get_str("_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE") == (
        Config_ContextSummarizer._PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE
    )
    assert cfg.get_dict("SUMMARY_WORD_LIMITS") == Config_Grounder.SUMMARY_WORD_LIMITS
    assert not any(
        "TOKEN" in key or "RESERVATION" in key for key in vars(Config_ContextSummarizer)
    )


def _chatter(rag):
    shell = _shell(
        "src/Chat/Chatter.py",
        "Chatter",
        {
            "run",
            "_run_document_summary",
            "_build_answer_confidence_notice",
            "_build_answer_confidence_notice_cli",
            "_resolve_answer_confidence_payload",
        },
        YELLOW=YELLOW,
        RED="",
        ORANGE=ORANGE,
        GREEN=GREEN,
        RESET=RESET,
        Symbols=Symbols,
        ModelOutput=SimpleNamespace,
        LLMResultError=LLMResultError,
        CollectionNotFoundError=CollectionNotFoundError,
        ContextSummarizer=MagicMock(),
        FileSelector=FileSelector,
    )
    shell.rag = rag
    shell.is_streaming = False
    shell.pretty = MagicMock()
    shell.pretty_always = MagicMock()
    shell.print_llm_answer = MagicMock()
    shell.terminal_line_size = 100
    shell.cfg = MagicMock()
    shell.llmCaller = MagicMock()
    shell.tokenBudget = MagicMock()
    shell.confidence_logger = MagicMock()
    shell.fileUtils = MagicMock()
    shell.chatContext = MagicMock()
    shell.masker = SimpleNamespace(mask=str)
    shell._check_answer_compliance = MagicMock(return_value=False)

    def grounded_answer(
        _session: Any, answer_text: str, *args: Any, **kwargs: Any
    ) -> str:
        return answer_text

    shell.grounder = MagicMock()
    shell.grounder.is_answer_too_short.return_value = False
    shell.grounder.has_grounded_evidence.return_value = True
    shell.grounder.has_local_spans.return_value = True
    shell.grounder.mark_answer_cli.side_effect = grounded_answer
    shell.grounder.short_answer_notice.return_value = "Short answer.\n"
    shell._grounding_language_detector = MagicMock(return_value="detector")
    shell._evaluate_answer_quality = MagicMock()
    shell._open_marked_documents = MagicMock()
    rag.confidenceEvaluator = ConfidenceEvaluator(
        confidence_logger=MagicMock(), pretty=MagicMock()
    )
    rag._mark_sources = MagicMock()
    return shell


def test_chat_missing_pin_prints_exact_yellow_warning(rag, session):
    session.file_name = session.file_path = None
    chatter = _chatter(rag)
    callback = MagicMock()
    success, message = chatter.run(session, apiChunkHandler=callback)
    assert success is False
    assert session.indexed_query_error == message
    assert message == (
        "Summarize: requires a file to be selected. Use path! in session "
        "settings to select the document, then enter summarize:.\n"
        "If no file can be selected, enter a file name contained in the "
        "collection or issue a query which fills the picker"
    )
    assert chatter.pretty.write.call_args_list == [
        (("W", "", line), {"color": YELLOW}) for line in message.splitlines()
    ]
    callback.assert_called_once_with(message)
    chatter.llmCaller.call_llm.assert_not_called()
    rag.collection.get.assert_not_called()


def test_service_missing_pin_never_suggests_terminal_pickers(rag, session):
    session.is_service = True
    session.file_name = session.file_path = None
    warning = rag.summary_selection_warning(session)
    assert "Advanced Parameters" in warning
    assert "summarize:" in warning
    assert all(token not in warning for token in ("b:", "file!", "path!", "picker"))
    with pytest.raises(ValueError, match="Advanced Parameters"):
        rag.retrieve_summary_documents(session)
    rag.collection.get.assert_not_called()


def test_service_ambiguous_filename_suggests_api_path_not_picker(rag, session):
    session.is_service = True
    rag.collection.get.return_value["metadatas"][1]["FilePath"] = r"D:\other\doc.pdf"
    with pytest.raises(ValueError) as exc:
        rag.retrieve_summary_documents(session)
    assert "Advanced Parameters" in str(exc.value)
    assert "path!" not in str(exc.value)


@pytest.mark.parametrize("query", ["What does it say?", "summarize:"])
@pytest.mark.parametrize("pin", ["file", "path"])
def test_cli_prompt_identifies_pin_before_query(session, query, pin):
    import os

    session.file_name = "doc.pdf" if pin == "file" else None
    session.file_path = r"D:\docs\doc.pdf" if pin == "path" else None
    session.file_path_select = pin
    session.chat_name = "test"
    session.web_search = False
    session.metadata_filters = {}
    processor = _shell(
        "src/Chat/CommandProcessor.py",
        "CommandProcessor",
        {"configure_and_query"},
        os=os,
        HistoryManager=SimpleNamespace(key=lambda *args: "test"),
        **{
            color: ""
            for color in (
                "MAGENTA",
                "ORANGE",
                "RESET",
                "BOLD",
                "RED",
                "TURQUOISE",
                "YELLOW",
            )
        },
    )
    processor.session = session
    processor.queryParts = MagicMock()
    processor._first_configure = False
    processor.hist = MagicMock()
    processor.hist.prompt.side_effect = ["", query]
    assert processor.configure_and_query() is session
    prompt = processor.hist.prompt.call_args.args[0]
    assert (
        f"Pinned document (chat and summarize): {session.file_path or session.file_name}"
        in prompt
    )
    assert prompt.index("Pinned document") < prompt.index("Your actual query")
    assert session.query == query


def test_summary_unavailable_settings_collection_reports_error_without_model_call(
    rag, session
):
    rag.collection_name = "previous-collection"
    rag._set_vector_store.side_effect = CollectionNotFoundError(
        "Selected collection unavailable"
    )
    chatter = _chatter(rag)
    callback = MagicMock()
    success, message = chatter.run(session, apiChunkHandler=callback)
    assert success is False
    assert message == "Selected collection unavailable"
    callback.assert_called_once_with(message)
    rag.collection.get.assert_not_called()
    chatter._run_document_summary.__func__.__globals__[
        "ContextSummarizer"
    ].assert_not_called()


def test_cli_missing_pin_picker_return_skips_summary_and_prompt_checks(rag, session):
    import re

    session.file_name = session.file_path = None
    chatter = _chatter(rag)
    app = _shell("src/Apps/RAGChat.py", "RAGChat", {"_process_query"}, re=re, CYAN="")
    app.cp = SimpleNamespace(configure_and_query=MagicMock(return_value=session))
    app.chatter = chatter
    path_picker = MagicMock(return_value=False)
    app.fileSelector = CliFileSelector(path_picker)
    app.pretty = MagicMock()
    app.aiHelpers = MagicMock()
    app.csvWriter = MagicMock()

    assert app._process_query() is True
    path_picker.assert_called_once_with(session)
    app.pretty.write.assert_called_once_with(
        "I", "Summarize", CANCELLED_MESSAGE, color=""
    )
    chatter.pretty.write.assert_not_called()
    app.aiHelpers.check_user_prompt_with_filter_chain.assert_not_called()
    app.aiHelpers.check_prompt_with_llm_guard.assert_not_called()
    app.csvWriter.write_json2csv.assert_not_called()
    rag.collection.get.assert_not_called()


def test_cli_path_picker_return_goes_back_to_session_settings(rag, session):
    import re

    session.file_name = session.file_path = None
    app = _shell(
        "src/Apps/RAGChat.py",
        "RAGChat",
        {"_process_query", "_run_query"},
        re=re,
        CYAN="",
        GREEN="",
        RESET="",
    )
    app.cp = SimpleNamespace(
        configure_and_query=MagicMock(
            side_effect=[
                session,
                SimpleNamespace(query=""),
            ]
        )
    )
    app.chatter = _chatter(rag)
    app.fileSelector = CliFileSelector(MagicMock(return_value=False))
    app.pretty = MagicMock()

    app._run_query()

    assert app.cp.configure_and_query.call_count == 2
    app.pretty.write.assert_called_once_with(
        "I", "Summarize", CANCELLED_MESSAGE, color=""
    )
    app.chatter.llmCaller.call_llm.assert_not_called()
    rag.collection.get.assert_not_called()


def test_cli_unpinned_summary_uses_path_picker_and_continues(rag, session, monkeypatch):
    import re

    import Gui.FileList as file_module
    from Chat.QueryParts import QueryParts
    from Gui.FileList import FileEntry, FileList

    session = _unpinned(session)
    entry = FileEntry("selected-collection_chat", "doc.pdf", r"D:\docs\doc.pdf", "File")
    files = object.__new__(FileList)
    files.entries = [entry]
    files.cur = None
    files.current_store = None
    files.usage = "File"
    parts = object.__new__(QueryParts)
    parts.fileList = files
    prompt = MagicMock()
    prompt.select.return_value.execute.return_value = entry
    monkeypatch.setattr(file_module, "inquirer", prompt)
    app = _shell("src/Apps/RAGChat.py", "RAGChat", {"_process_query"}, re=re, CYAN="")
    app.cp = SimpleNamespace(configure_and_query=MagicMock(return_value=session))
    app.fileSelector = CliFileSelector(parts.pick_path)
    app.chatter = SimpleNamespace(rag=rag, run=MagicMock())
    app.pretty = MagicMock()

    assert app._process_query() is True

    prompt.select.assert_called_once()
    assert prompt.select.call_args.kwargs["choices"][0]["value"] is entry
    assert session.file_path == entry.path
    assert session.file_name is None
    assert session.file_path_select == "path"
    assert session.query == "summarize:"
    app.chatter.run.assert_called_once_with(session, original_query="summarize:")


@pytest.mark.parametrize(
    "file_name, file_path, target",
    [
        ("doc.pdf", None, ""),
        (None, r"D:\docs\doc.pdf", ""),
        (None, None, "doc.pdf"),
    ],
)
def test_summary_selection_accepts_pin_or_explicit_target(
    rag, session, file_name, file_path, target
):
    session.file_name, session.file_path = file_name, file_path
    assert rag.summary_selection_warning(session, target) is None


def test_chat_calls_existing_summarizer_without_normal_retrieval(rag, session):
    chatter = _chatter(rag)
    callback = MagicMock()
    handler = chatter._run_document_summary.__func__
    factory = handler.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"
    success, answer = chatter.run(session, apiChunkHandler=callback)
    assert success is True
    assert answer.startswith(CONFIRMATION + "## Summary of doc.pdf\n\ndocument summary")
    assert "**Answer confidence:**" in answer and "C_final=0.20" in answer
    assert "```text\n" in answer
    assert "C_final=0.20" in chatter.print_llm_answer.call_args.args[0]
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] is None
    documents = factory.return_value.summarize_documents.call_args.args[0]
    assert len(documents) == 2
    assert session.last_chosen_chunks == documents
    callback.assert_called_once_with(answer)
    chatter._check_answer_compliance.assert_called_once()
    chatter.pretty_always.write.assert_called_once_with(
        "I", "Summarize", "Summarizing all 2 indexed chunks."
    )
    chatter.grounder.has_grounded_evidence.assert_called_once_with(
        session, "document summary", ["first page", "last page"], "english", "detector"
    )
    chatter._evaluate_answer_quality.assert_called_once()
    chatter.grounder.mark_answer_cli.assert_called_once()
    rag._mark_sources.assert_called_once_with(session, documents, summary=True)
    assert session.answer_confidence_result is not None
    assert "relevance scores are unavailable" in session.answer_confidence_result.note
    assert session.answer_confidence_result.score == pytest.approx(0.2)


@pytest.mark.parametrize("streaming", [False, True])
def test_grounded_document_link_is_opened_before_cli_answer(
    rag, session, capsys, streaming
):
    from langchain_core.prompts import PromptTemplate

    from Chat.Chatter import Chatter
    from Commons.Exceptions import QueryTranslationError
    from Helpers.SourcePathLinkifier import SourcePathLinkifier

    events = []
    chatter = _chatter(rag)
    chatter.run.__func__.__globals__.update(
        PromptTemplate=PromptTemplate,
        QueryTranslationError=QueryTranslationError,
        DebugHelper=SimpleNamespace(check_session=lambda *_args: False),
        SourcePathLinkifier=SourcePathLinkifier,
        os=os,
    )
    chatter._build_query_notice = MagicMock(return_value="")
    chatter._resolve_token_params = MagicMock(return_value=(1024, 128))
    chatter._evaluate_answer_quality = MagicMock()
    chatter._build_answer_confidence_notice = MagicMock(return_value="")
    chatter._build_answer_confidence_notice_cli = MagicMock(return_value="")
    chatter._show_web_sources = MagicMock()
    chatter.helpers = SimpleNamespace(build_document_metadata_md=lambda _chosen: "")
    chatter.use_ollama_gpu = True
    chatter.llm_model = "test-model"
    chatter.prompt = "{context}\n{input}\n{response_language}"
    chatter.prompt_name = None
    chatter.modelOutputAdapter = SimpleNamespace(
        interpret=lambda *_args, **_kwargs: SimpleNamespace(content="Grounded answer.")
    )
    chatter.llmCaller.make_on_chunk.return_value = MagicMock()
    chatter.llmCaller.call_llm.return_value = {}
    chatter.grounder.collect_web_texts.return_value = []
    chatter.grounder.mark_answer_cli.return_value = "Grounded answer."

    def print_answer(*args, **kwargs):
        events.append("answer")
        Chatter.print_llm_answer(chatter, *args, **kwargs)

    chatter.print_llm_answer.side_effect = print_answer
    chatter._open_marked_documents.side_effect = lambda *_args: events.append("link")

    session.query = "question"
    session.retriever_k = 10
    session.max_output_tokens = 128
    session.temperature = 0.1
    session.top_k = 40
    session.top_p = 0.9
    session.max_output_tokens_override = None
    session.context_size_override = None
    session.extraOllamaOptions = None
    session.ollamaTopLevelParams = None
    session.preferred_response_language = "english"
    session.user_language = None
    session.mark_text = True
    session.enable_grounding = True
    session.use_chat_context = False
    session.last_chosen_chunks = [SimpleNamespace(metadata={})]
    session.chunk_texts_for_grounding = ["Grounded evidence."]
    session.grounding_proxy_sentences = {}
    session.answer_confidence_result = SimpleNamespace(
        level="HIGH", score=1.0, summary="supported"
    )
    session.marked_documents = []

    def retrieve(current):
        current.query = "rewritten query"
        return "Grounded evidence.", 1

    rag.retrieve = MagicMock(side_effect=retrieve)
    rag._mark_sources.side_effect = lambda current, *_args: setattr(
        current, "marked_documents", [("document.pdf", b"marked")]
    )
    chatter.grounder.is_answer_too_short.return_value = False
    chatter.grounder.has_grounded_evidence.return_value = True
    chatter.grounder.has_local_spans.return_value = True

    success, answer = chatter.run(session, is_streaming=streaming)

    assert success is True
    assert "Grounded answer." in answer
    assert events == ["link", "answer"]
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] == "question"
    output = capsys.readouterr().out
    assert output.startswith(f"{YELLOW}Original query: question{RESET}\n")
    assert "Grounded answer." in output.splitlines()[1]


def test_summary_displays_input_before_document_selection(rag, session):
    session.query = "summarize: doc.pdf"
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"

    success, answer = chatter.run(session, original_query="summary:")

    assert success and answer
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] == "summary:"


def test_cli_preserves_summary_input_when_selection_changes_command(rag, session):
    import re

    session.query = "summary:"
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"
    app = _shell("src/Apps/RAGChat.py", "RAGChat", {"_process_query"}, re=re, CYAN="")
    app.cp = SimpleNamespace(configure_and_query=MagicMock(return_value=session))
    app.chatter = chatter
    app.fileSelector = SimpleNamespace(
        select=MagicMock(return_value=FileSelection("picked", target="doc.pdf"))
    )

    assert app._process_query() is True
    assert session.query == "summarize: doc.pdf"
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] == "summary:"


@pytest.mark.parametrize(
    "target",
    [
        "doc.pdf",
        r"D:\docs\doc.pdf",
        '"D:\\docs\\doc.pdf"',
        "'D:\\docs\\doc.pdf'",
    ],
)
def test_chat_explicit_target_without_pin(rag, session, target):
    session.file_name = session.file_path = None
    session.query = f"summarize:   {target}"
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"

    success, answer = chatter.run(session)

    assert success is True
    assert answer.startswith(CONFIRMATION + "## Summary of doc.pdf\n\ndocument summary")
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] == session.query
    assert "**Answer confidence:**" in answer and "C_final=0.20" in answer
    assert session.file_name is None and session.file_path is None
    history_query = chatter.chatContext.add_chat_turn.call_args.args[1]
    assert history_query == session.query
    assert session.user_query_original == session.query
    assert chatter.chatContext.add_chat_turn.call_args.args[2].startswith(
        "## Summary of doc.pdf\n\ndocument summary"
    )


@pytest.mark.parametrize(
    "file_name, file_path",
    [
        ("doc.pdf", None),
        (None, r"D:\docs\doc.pdf"),
    ],
)
def test_pinned_summary_history_includes_resolved_file(
    rag, session, file_name, file_path
):
    session.file_name, session.file_path = file_name, file_path
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"

    success, _ = chatter.run(session)

    assert success is True
    assert chatter.chatContext.add_chat_turn.call_count == 1
    assert (
        chatter.chatContext.add_chat_turn.call_args.args[1]
        == r"summarize: D:\docs\doc.pdf"
    )
    assert (session.file_name, session.file_path) == (file_name, file_path)
    assert session.query == "summarize:"


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_summary_path_is_unquoted_for_lookup_and_kept_in_history(
    rag, session, quote
):
    path = r"D:\RAG-LCC\DocsToLoadOrClassify\ReverseEngineeringForBeginners.en.pdf"
    session.file_name = session.file_path = None
    session.query = f"summarize: {quote}{path}{quote}"
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"

    success, _ = chatter.run(session)

    assert success is True
    assert rag.collection.get.call_args.kwargs["where"] == {
        "$and": [
            {"FileName": {"$eq": "ReverseEngineeringForBeginners.en.pdf"}},
            {"FilePath": {"$eq": path}},
        ]
    }
    assert chatter.chatContext.add_chat_turn.call_args.args[1] == session.query


def test_summary_does_not_save_history_when_disabled(rag, session):
    session.use_chat_context = False
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"

    success, _ = chatter.run(session)

    assert success is True
    chatter.chatContext.add_chat_turn.assert_not_called()


@pytest.mark.parametrize("failure", ["lookup", "model", "compliance"])
def test_failed_summary_does_not_save_history(rag, session, failure):
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"
    if failure == "lookup":
        rag.collection.get.return_value["documents"] = []
    elif failure == "model":
        factory.return_value.summarize_documents.side_effect = LLMResultError(
            "model failed"
        )
    else:
        chatter._check_answer_compliance.return_value = True

    success, _ = chatter.run(session)

    assert success is False
    chatter.chatContext.add_chat_turn.assert_not_called()


def test_chat_context_stamps_embedding_identity_before_writing():
    context = _shell(
        "src/Chat/ChatContext.py",
        "ChatContext",
        {"_init_chat_collection"},
    )
    context.chromaDBHelper = MagicMock()
    context.chromaDBHelper.chroma_coll_name_and_mkdir_or_del.return_value = (
        "test_ChatContext",
        "unused",
    )
    context.chromaDBHelper.get_chroma_client_and_collection.return_value = (
        MagicMock(),
        MagicMock(),
    )
    context.initalized_collections = []
    context._delete_collection = MagicMock()
    context._init_chat_collection(SimpleNamespace(collection_name="test"))
    context.chromaDBHelper.get_chroma_client_and_collection.assert_called_once_with(
        "unused",
        "test_ChatContext",
        stamp=True,
    )


@pytest.mark.parametrize(
    "query, pinned",
    [
        ("summarize:", True),
        ("summary:", True),
        ("summary: doc.pdf", False),
        ("summarize: doc.pdf", False),
        (r"summarize: D:\docs\doc.pdf", False),
    ],
)
def test_cli_summary_skips_prompt_checks_with_selected_file(
    rag, session, query, pinned
):
    import re

    session.query = query
    if not pinned:
        session.file_name = session.file_path = None
    app = _shell("src/Apps/RAGChat.py", "RAGChat", {"_process_query"}, re=re, CYAN="")
    app.cp = SimpleNamespace(configure_and_query=MagicMock(return_value=session))
    app.chatter = SimpleNamespace(rag=rag, run=MagicMock())
    app.fileSelector = CliFileSelector(
        MagicMock(side_effect=AssertionError("Unexpected picker"))
    )
    app.pretty = MagicMock()
    app.aiHelpers = MagicMock()
    app.csvWriter = MagicMock()
    assert app._process_query() is True
    app.chatter.run.assert_called_once_with(session, original_query=query)
    app.aiHelpers.check_user_prompt_with_filter_chain.assert_not_called()
    app.aiHelpers.check_prompt_with_llm_guard.assert_not_called()
    app.csvWriter.write_json2csv.assert_not_called()


def test_unsupported_summary_has_zero_confidence_and_no_highlights(rag, session):
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "unsupported summary"
    chatter.grounder.has_grounded_evidence.return_value = False

    success, answer = chatter.run(session)

    assert success is True
    assert "couldn't find supporting evidence" in answer
    assert "unsupported summary" not in answer
    assert session.answer_confidence_result.score == 0
    assert session.last_chosen_chunks == []
    assert session.chunk_texts_for_grounding == []
    chatter.grounder.mark_answer_cli.assert_not_called()
    rag._mark_sources.assert_not_called()


def test_short_summary_has_explicit_notice_and_zero_confidence(rag, session):
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "short"
    chatter.grounder.is_answer_too_short.return_value = True

    success, answer = chatter.run(session)

    assert success is True
    assert answer.startswith(CONFIRMATION + "Short answer.")
    assert session.answer_confidence_result.score == 0
    chatter.grounder.has_grounded_evidence.assert_not_called()
    rag._mark_sources.assert_not_called()


@pytest.mark.parametrize("prefix", ["new:", "NEW TOPIC:", "newtopic:"])
def test_new_command_still_resets_topic_and_skips_rewrite(rag, session, prefix):
    session.query = f"{prefix} another question"
    session.last_web_search = session.last_fetch_page_content = None
    session.fetch_page_content = False
    rag.chatContext = MagicMock()
    rag.confidence_logger = MagicMock()
    rag._fileUtils = MagicMock()
    rag.pretty = MagicMock()
    assert rag._prepare_session(session) == "another question"
    assert session.force_skip_rewrite is True
    assert session.last_topic_referents is None
    rag.chatContext.reset_conversation.assert_called_once()
    rag.pretty.write.assert_any_call(
        "I",
        "UserQuery",
        f"Original user query: '{prefix} another question'",
        color="",
    )


def _file_entries(*paths, store="selected-collection_chat"):
    return SimpleNamespace(
        entries=[
            SimpleNamespace(store=store, usage="File", file=Path(path).name, path=path)
            for path in paths
        ]
    )


def _unpinned(session):
    session.file_name = session.file_path = None
    session.chat_name = "chat"
    return session


@pytest.mark.parametrize("target", ["cancel", "CANCEL", "c", "0"])
def test_file_selector_cancel_never_summarizes(rag, session, target):
    path_picker = MagicMock(side_effect=AssertionError("Unexpected picker"))
    selector = CliFileSelector(path_picker)

    selection = selector.select(_unpinned(session), target)

    assert selection.status == "cancelled" and selection.stops_summary
    assert selection.message == CANCELLED_MESSAGE
    path_picker.assert_not_called()
    rag.collection.get.assert_not_called()


def test_file_selector_explicit_and_pinned_selections_take_precedence(rag, session):
    path_picker = MagicMock(side_effect=AssertionError("Unexpected picker"))
    selector = CliFileSelector(path_picker)

    explicit = selector.select(session, " doc.pdf ")
    pinned = selector.select(session, "")

    assert (explicit.status, explicit.target) == ("explicit", "doc.pdf")
    assert pinned.status == "pinned"
    path_picker.assert_not_called()


@pytest.mark.parametrize(
    "paths", [(), (r"D:\docs\doc.pdf",), (r"D:\a.pdf", r"D:\b.pdf")]
)
def test_cli_selector_uses_settings_path_picker_without_auto_selection(
    session, paths, monkeypatch
):
    history = _file_entries(*paths)
    history_lookup = MagicMock(return_value=history)
    monkeypatch.setattr("Gui.FileList.FileList", history_lookup)
    prompt = MagicMock(side_effect=AssertionError("Unexpected inline picker"))
    monkeypatch.setattr("InquirerPy.inquirer.select", prompt)
    session = _unpinned(session)
    session.last_chosen_chunks = [
        Document(page_content="source", metadata={"FilePath": path}) for path in paths
    ]

    path_picker = MagicMock(return_value=False)
    selection = CliFileSelector(path_picker).select(session, "")

    assert selection.status == "cancelled"
    assert selection.target == ""
    assert selection.candidates == ()
    history_lookup.assert_not_called()
    prompt.assert_not_called()
    path_picker.assert_called_once_with(session)


def test_indexed_file_metadata_returns_raw_collection_metadata(rag, session):
    rag.collection.get.return_value = {
        "metadatas": [
            {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf"},
            {"FileName": "doc.pdf", "FilePath": "D:/docs/doc.pdf"},
        ]
    }

    metadata = rag.indexed_file_metadata(session, ["doc.pdf", "doc.pdf"])

    assert metadata == rag.collection.get.return_value["metadatas"]
    assert rag.collection.get.call_args.kwargs == {
        "where": {"FileName": {"$eq": "doc.pdf"}},
        "include": ["metadatas"],
    }


def test_indexed_file_metadata_switches_collection_under_lock(rag, session):
    rag.collection_name = "previous-collection"
    rag._set_vector_store.side_effect = (
        lambda current: setattr(rag, "collection_name", current.collection_name) or True
    )
    rag.collection.get.return_value = {"metadatas": []}

    assert rag.indexed_file_metadata(session, ["doc.pdf"]) == []
    rag._set_vector_store.assert_called_once_with(session)
    rag.collection.get.assert_called_once()


def test_indexed_file_metadata_reports_unavailable_collection(rag, session):
    rag.collection_name = "previous-collection"
    rag._set_vector_store.return_value = False

    with pytest.raises(
        ValueError, match="could not load the selected indexed collection"
    ):
        rag.indexed_file_metadata(session, ["doc.pdf"])

    rag.collection.get.assert_not_called()


_MARKED_ANSWER = (
    "Hedgehogs hibernate.\n\n---\n🔵 **Marked sources** *(links expire in 30 min)*\n"
    "- 🟢 [doc.pdf (highlighted)](http://host/marked/abc.pdf)\n"
    "- 🟢 [other.pdf (highlighted)](http://host/marked/def.pdf)\n"
)


def test_service_extracts_grounded_document_references():
    extract = ServiceFileSelector.extract_document_references
    assert extract(_MARKED_ANSWER) == ["doc.pdf", "other.pdf"]
    assert extract("Answer\n\n---\n📄 **Sources**\n- a.pdf\n- b.md\n\nAfter") == [
        "a.pdf",
        "b.md",
    ]
    assert extract(
        "### Document metadata\n\n- **doc.pdf**\n"
        "    - FilePath: [D:\\docs\\doc.pdf](http://host/doc)\n"
    ) == [r"D:\docs\doc.pdf"]
    assert extract(CONFIRMATION + "## Summary of doc.pdf") == [r"D:\docs\doc.pdf"]
    assert extract("Plain answer\n- not a source") == []


def test_service_auto_selects_single_indexed_path_from_history(rag, session):
    rag.collection.get.return_value = {
        "metadatas": [
            {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf"},
            {"FileName": "doc.pdf", "FilePath": "D:/docs/doc.pdf"},
        ]
    }
    messages = [
        ("user", "What do hedgehogs eat?"),
        ("assistant", "Insects.\n\n---\n📄 **Sources**\n- doc.pdf\n"),
        ("user", "summarize:"),
    ]

    selection = ServiceFileSelector(rag, messages).select(_unpinned(session), "")

    assert (selection.status, selection.target) == ("auto", r"D:\docs\doc.pdf")
    assert rag.collection.get.call_args.kwargs["where"] == {
        "FileName": {"$eq": "doc.pdf"}
    }


def test_service_does_not_reuse_sources_from_an_older_answer(rag, session):
    messages = [
        ("assistant", "Older answer\n\n---\n📄 **Sources**\n- doc.pdf\n"),
        ("user", "A later question"),
        ("assistant", "No indexed sources are cited in this answer."),
        ("user", "summarize:"),
    ]

    selection = ServiceFileSelector(rag, messages).select(_unpinned(session), "")

    assert selection.status == "none"
    rag.collection.get.assert_not_called()


def test_service_offers_paths_for_multiple_sources(rag, session):
    rag.collection.get.return_value = {
        "metadatas": [
            {"FileName": "doc.pdf", "FilePath": r"D:\docs\doc.pdf"},
            {"FileName": "other.pdf", "FilePath": r"D:\docs\other.pdf"},
        ]
    }
    messages = [
        ("assistant", _MARKED_ANSWER),
        ("user", "summarize:"),
    ]

    selection = ServiceFileSelector(rag, messages).select(_unpinned(session), "")

    assert selection.status == "choose" and selection.stops_summary
    assert selection.candidates == (r"D:\docs\doc.pdf", r"D:\docs\other.pdf")
    assert "1. 👉 `D:\\docs\\doc.pdf`" in selection.message
    assert "2. 👉 `D:\\docs\\other.pdf`" in selection.message
    assert "`cancel`" in selection.message
    assert rag.collection.get.call_args.kwargs["where"] == {
        "FileName": {"$in": ["doc.pdf", "other.pdf"]}
    }


def _listed_service_reply(reply):
    listed = ServiceFileSelector.format_choices(
        [r"D:\docs\doc.pdf", r"D:\docs\other.pdf"]
    )
    return ServiceFileSelector(
        MagicMock(),
        [("user", "summarize:"), ("assistant", listed), ("user", reply)],
    )


def test_service_numbered_reply_selects_the_listed_path():
    selection = _listed_service_reply("2").resolve_reply(" 2 ")
    assert selection is not None
    assert (selection.status, selection.target) == ("picked", r"D:\docs\other.pdf")
    assert selection.summary_query("2") == r"summarize: D:\docs\other.pdf"


def test_service_choices_mark_both_names_and_paths():
    listed = ServiceFileSelector.format_choices(["doc.pdf", r"D:\docs\other.pdf"])
    assert "1. 👉 `doc.pdf`" in listed
    assert "2. 👉 `D:\\docs\\other.pdf`" in listed


def test_service_numbered_reply_accepts_legacy_list_without_icons():
    listed = ServiceFileSelector.format_choices(
        [r"D:\docs\doc.pdf", r"D:\docs\other.pdf"]
    ).replace("👉 ", "")
    selector = ServiceFileSelector(MagicMock(), [("assistant", listed), ("user", "2")])
    selection = selector.resolve_reply("2")
    assert selection is not None
    assert (selection.status, selection.target) == ("picked", r"D:\docs\other.pdf")


@pytest.mark.parametrize("reply", ["cancel", "0", "c"])
def test_service_selection_reply_can_cancel(reply):
    selection = _listed_service_reply(reply).resolve_reply(reply)
    assert selection is not None
    assert selection.status == "cancelled"
    assert selection.message == CANCELLED_MESSAGE


def test_service_invalid_number_repeats_the_document_list():
    selection = _listed_service_reply("7").resolve_reply("7")
    assert selection is not None
    assert selection.status == "choose"
    assert selection.message.startswith("7 is not in the list.")
    assert "1. 👉 `D:\\docs\\doc.pdf`" in selection.message
    assert "2. 👉 `D:\\docs\\other.pdf`" in selection.message


def test_service_non_numeric_reply_does_not_consume_the_conversation():
    assert (
        _listed_service_reply("What about cats?").resolve_reply("What about cats?")
        is None
    )
    unrelated = ServiceFileSelector(
        MagicMock(), [("assistant", "A normal answer."), ("user", "2")]
    )
    assert unrelated.resolve_reply("2") is None


def test_confirmation_precedes_summary_reply_but_not_chat_history(rag, session):
    chatter = _chatter(rag)
    factory = chatter._run_document_summary.__func__.__globals__["ContextSummarizer"]
    factory.return_value.summarize_documents.return_value = "document summary"
    callback = MagicMock()

    success, answer = chatter.run(session, apiChunkHandler=callback)

    assert success is True
    assert answer.startswith(CONFIRMATION)
    callback.assert_called_once_with(answer)
    assert chatter.print_llm_answer.call_args.args[0].startswith(CONFIRMATION)
    assert (
        "Summarizing document"
        not in chatter.chatContext.add_chat_turn.call_args.args[2]
    )
