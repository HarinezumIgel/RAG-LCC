"""Tests for language-aware live reranking and its diagnostics."""

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Chat.RAGChatImpl import RAGChatImpl
from Gui.Colors import CYAN, GREEN, YELLOW


class _PrettyStub:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []
        self.colors: list[Any] = []

    def write(self, severity: str, stage: str, message: str, **kwargs: Any) -> None:
        self.messages.append((severity, stage, message))
        self.colors.append(kwargs.get("color"))


class _EncoderStub:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        show_progress_bar: bool,
    ) -> list[float]:
        assert show_progress_bar is False
        self.calls.append(pairs)
        return [0.9 if "Hedgehog" in passage else 0.1 for _, passage in pairs]


def _doc(file_name: str, text: str, language: str) -> Any:
    return SimpleNamespace(
        metadata={"FileName": file_name, "Language": language},
        page_content=text,
    )


def test_live_reranking_uses_language_queries_for_matching_candidates() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[
            {"language": "de", "query": "German query"},
            {"language": "fr", "query": "French query"},
        ]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.x_encoder = _EncoderStub()
    impl.pretty = _PrettyStub()
    impl.cross_encoder_query_instruction = ""
    vector_doc = _doc("german.pdf", "German facts", "de")
    vector_doc.metadata["retriever_sources"] = "Vector"
    indexed_doc = _doc("french.pdf", "French facts", "fr")
    indexed_doc.metadata["retriever_sources"] = "BM25"
    candidates = [vector_doc, indexed_doc, _doc("unknown.pdf", "Unknown facts", "")]
    pairs = [
        ("Original query", "German facts"),
        ("Original query", "French facts"),
        ("Original query", "Unknown facts"),
    ]

    scores = impl._predict_rerank_scores(candidates, pairs)

    assert scores == [0.1, 0.1, 0.1]
    assert len(impl.x_encoder.calls) == 3
    assert [call[0][0] for call in impl.x_encoder.calls] == [
        "German query",
        "French query",
        "Original query",
    ]
    assert [call[0][1] for call in impl.x_encoder.calls] == [
        "German facts",
        "French facts",
        "Unknown facts",
    ]
    assert any(
        "1 candidate(s) had no matching language query" in message
        for _, _, message in impl.pretty.messages
    )
    passing_messages = [
        (message, color)
        for (_, _, message), color in zip(impl.pretty.messages, impl.pretty.colors)
        if message.startswith("Passing language=")
    ]
    assert len(passing_messages) == 3
    assert all(color == YELLOW for _, color in passing_messages)
    assert "language=de chunks=1" in passing_messages[0][0]
    summary_messages = [
        message
        for _, stage, message in impl.pretty.messages
        if stage == "Rerank summary"
    ]
    assert summary_messages[0].split() == [
        "Language",
        "Chunks",
        "Top",
        "raw",
        "Mean",
        "top3",
        "Mean",
        "raw",
    ]
    assert summary_messages[1].split() == ["de", "1", "0.1000", "0.1000", "0.1000"]
    assert summary_messages[2].split() == ["fr", "1", "0.1000", "0.1000", "0.1000"]
    assert any(
        "language=de top5=[1:german.pdf (raw=0.1000)]" in message
        for _, _, message in impl.pretty.messages
    )
    assert all(
        color == CYAN
        for (_, stage, _), color in zip(impl.pretty.messages, impl.pretty.colors)
        if stage == "Rerank summary"
    )


def test_live_summary_uses_real_scores_and_logs_inputs_before_prediction() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[{"language": "en", "query": "English query"}]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.pretty = _PrettyStub()
    impl.cross_encoder_query_instruction = ""
    calls: list[list[tuple[str, str]]] = []

    def predict(
        pairs: list[tuple[str, str]], *, show_progress_bar: bool
    ) -> list[float]:
        assert show_progress_bar is False
        assert impl.pretty.messages[-1][2].startswith("Passing language=en")
        assert impl.pretty.colors[-1] == YELLOW
        calls.append(pairs)
        return [0.2, 0.9, -0.5, 0.6]

    impl.x_encoder = SimpleNamespace(predict=predict)
    candidates = [_doc(f"{index}.pdf", f"Chunk {index}", "en") for index in range(4)]
    pairs = [("Original query", doc.page_content) for doc in candidates]

    assert impl._predict_rerank_scores(candidates, pairs) == [0.2, 0.9, -0.5, 0.6]
    assert len(calls) == 1
    assert impl.pretty.messages[-1][2].split() == [
        "en",
        "4",
        "0.9000",
        "0.5667",
        "0.3000",
    ]
    assert any(
        "top5=[1:1.pdf (raw=0.9000), 2:3.pdf (raw=0.6000)" in message
        for _, _, message in impl.pretty.messages
    )


def test_live_reranking_preserves_legacy_query_without_language_queries() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(_rerank_language_queries=[])
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.x_encoder = _EncoderStub()
    candidates = [
        _doc("german.pdf", "German facts", "de"),
        _doc("french.pdf", "French facts", "fr"),
    ]
    pairs = [
        ("Original query", "German facts"),
        ("Original query", "French facts"),
    ]

    scores = impl._predict_rerank_scores(candidates, pairs)

    assert scores == [0.1, 0.1]
    assert len(impl.x_encoder.calls) == 1
    assert impl.x_encoder.calls[0] == pairs


@pytest.mark.parametrize("limit", [4, None, 2, 0, -1])
def test_candidate_fusion_is_language_scoped_before_the_pool_cap(
    limit: int | None,
) -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[
            {"language": "de", "query": "German query"},
            {"language": "en", "query": "English query"},
            {"language": "fr", "query": "French query"},
        ]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.pretty = _PrettyStub()
    vector_results = [
        _doc("german.pdf", "Shared German chunk", "de"),
        _doc("english.pdf", "English vector chunk", "en"),
    ]
    indexed_results = [
        _doc("german.pdf", "Shared German chunk", "de"),
        _doc("german.pdf", "Second German chunk", "de"),
        _doc("french.pdf", "French indexed chunk", "fr"),
    ]

    merged = impl._merge_local_docs_by_language(
        [vector_results, indexed_results],
        ["Vector", "BM25"],
        [1.0, 1.0],
        rrf_k=60,
        limit=limit,
    )

    assert merged is not None
    expected_count = 4 if limit is None else max(0, min(limit, 4))
    assert len(merged) == expected_count
    assert [doc.metadata["Language"] for doc in merged] == ["de", "en", "fr", "de"][
        :expected_count
    ]
    if expected_count == 0:
        return
    assert merged[0].metadata["retriever_sources"] == "Vector,BM25"
    assert vector_results[1].metadata["rrf_score"] == 1 / 61
    assert indexed_results[2].metadata["rrf_score"] == 1 / 61
    assert vector_results[1].metadata["retriever_sources"] == "Vector"
    assert indexed_results[2].metadata["retriever_sources"] == "BM25"
    assert any("languages=3" in message for _, _, message in impl.pretty.messages)


def test_best_language_selection_passes_only_that_language_to_chunk_selection() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[
            {"language": "de", "query": "German query"},
            {"language": "en", "query": "English query"},
        ]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.pretty = _PrettyStub()
    candidates = [
        _doc("german-best.pdf", "German best chunk", "de"),
        _doc("english-best.pdf", "English best chunk", "en"),
        _doc("german-second.pdf", "German second chunk", "de"),
    ]

    selected, selected_scores, language = impl._select_best_rerank_language(
        candidates,
        [0.81, 0.84, 0.22],
    )

    assert language == "en"
    assert selected == [candidates[1]]
    assert selected_scores == [0.84]
    assert "Selected language=en" in impl.pretty.messages[-1][2]
    assert impl.pretty.colors[-1] == GREEN
    assert "English best chunk" not in impl.pretty.messages[-1][2]


def test_best_language_selection_keeps_legacy_pool_without_language_queries() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(_rerank_language_queries=[])
    impl._shared = SimpleNamespace(lang_name_to_code={})
    candidates = [
        _doc("german.pdf", "German facts", "de"),
        _doc("english.pdf", "English facts", "en"),
    ]
    scores = [0.8, 0.7]

    selected, selected_scores, language = impl._select_best_rerank_language(
        candidates,
        scores,
    )

    assert selected is candidates
    assert selected_scores is scores
    assert language is None


def test_web_results_keep_original_query_and_cannot_choose_local_language() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[
            {"language": "de", "query": "German query"},
            {"language": "en", "query": "English query"},
        ]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.pretty = _PrettyStub()
    impl.cross_encoder_query_instruction = ""
    impl.x_encoder = _EncoderStub()
    german = _doc("german.pdf", "German facts", "de")
    english = _doc("english.pdf", "English facts", "en")
    web = _doc("web", "Full web page", "en")
    web.metadata.update(Source="Web", snippet="Hedgehog snippet")
    unknown_web = _doc("unknown-web", "Other page", "")
    unknown_web.metadata["Source"] = "Web"
    candidates = [german, english, web, unknown_web]
    pairs = [
        ("Original query", "German facts"),
        ("Original query", "English facts"),
        ("Original query", "Hedgehog snippet"),
        ("Original query", "Other page"),
    ]

    impl._predict_rerank_scores(candidates, pairs)

    assert impl.x_encoder.calls[-1] == pairs[2:]
    selected, scores, language = impl._select_best_rerank_language(
        candidates, [0.8, 0.6, 0.99, 0.9]
    )
    assert language == "de"
    assert selected == [german, web, unknown_web]
    assert scores == [0.8, 0.99, 0.9]
    assert not any(
        "had no matching language query" in message
        for _, _, message in impl.pretty.messages
    )


def test_web_only_pool_bypasses_local_language_selection() -> None:
    impl: Any = object.__new__(RAGChatImpl)
    impl.orchestrator = SimpleNamespace(
        _rerank_language_queries=[{"language": "en", "query": "English query"}]
    )
    impl._shared = SimpleNamespace(lang_name_to_code={})
    impl.pretty = _PrettyStub()
    web = _doc("web", "Web text", "en")
    web.metadata["Source"] = "Web"
    candidates = [web]
    selected, scores, language = impl._select_best_rerank_language(candidates, [0.9])
    assert selected is candidates
    assert scores == [0.9]
    assert language is None
    assert impl.pretty.messages == []
