# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""Tests for the retrieval-language contract in RAGChatImpl.

Covers:
- _normalize_query tracks user_language vs retrieval_language and
    orig/post-rewrite English retrieval queries
- post-rewrite non-English handling (strict retry and translation fallback)
- _fetch_local_docs runs both post-rewrite and orig-translated retrieval legs
    and fuses them
"""

import os
import re
import sys
import textwrap
import time
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Chat.RetrievalOrchestrator import RetrievalOrchestrator

_RAG_IMPL_SRC = os.path.join(
    os.path.dirname(__file__), "..", "src", "Chat", "RAGChatImpl.py"
)


class _DebugHelperStub:
    @staticmethod
    def check_session(_session: Any, _level: int) -> bool:
        return False


class StubPrettyWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write(self, *a: Any, **k: Any) -> None:
        self.calls.append((a, k))


class StubSession:
    def __init__(self, query: str) -> None:
        self.query = query
        self.use_chat_context = True
        self.force_skip_rewrite = False
        self.preferred_response_language: str | None = None

        self.current_query_lang: str | None = None
        self.user_language: str | None = None
        self.retrieval_language: str | None = None
        self.orig_translated_query_en: str | None = None
        self.post_rewrite_query_en: str | None = None
        self.retrieval_top_k_orig_query_en: int | None = None
        self.retrieval_top_k_post_rewrite_query_en: int | None = None
        self.t1_query: str | None = None
        self.rewritten_query: str | None = None
        self.rewrite_language: str | None = None
        self.t2_query: str | None = None
        self.retrieval_top_k_before_t2: int | None = None
        self.retrieval_top_k_after_t2: int | None = None
        self.effective_query: str | None = None
        self.effective_query_reason: str | None = None

        self.vector_weight: float | None = None
        self.bm25_weight: float | None = None
        self.graph_weight: float | None = None
        self.regex_weight: float | None = None
        self.retriever_k: int | None = 5
        self.base_kwargs: dict[str, Any] | None = None
        self.debug_level: int | None = 0
        self.debug_mode: str = "ge"


class StubFileUtils:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def get_user_text_language(
        self,
        text: str,
        output: str = "nltk",
        native_lang: str | None = None,
    ) -> str:
        _ = output
        _ = native_lang
        return self._mapping.get(text, "english")


class StubTranslator:
    def __init__(self, mapping: dict[tuple[str, str], str]) -> None:
        self._mapping = mapping
        self.calls: list[tuple[str, str, str]] = []

    def translate_text(
        self, text: str, target_lang: str, source_lang: str = "auto"
    ) -> str:
        self.calls.append((text, target_lang, source_lang))
        return self._mapping.get((text, source_lang), text)


class StubSharedTranslate:
    def __init__(self, mapping: dict[tuple[str, str, str], str]) -> None:
        self._mapping = mapping
        self.calls: list[tuple[str, str, str]] = []
        self.lang_name_to_code: dict[str, str] = {
            "english": "en",
            "german": "de",
        }

    def translate_text(
        self, text: str, target_lang: str, source_lang: str = "auto"
    ) -> str:
        self.calls.append((text, target_lang, source_lang))
        return self._mapping.get((text, target_lang, source_lang), text)


class StubPromptRewrite:
    def __init__(self, normal: str, strict: str | None = None) -> None:
        self.normal = normal
        self.strict = strict if strict is not None else normal
        self.calls: list[bool] = []

    def rewrite(self, _session: StubSession, *, strict_english: bool = False) -> str:
        self.calls.append(strict_english)
        return self.strict if strict_english else self.normal


class StubDoc:
    def __init__(self, doc_id: str, text: str = "") -> None:
        self.page_content = text or doc_id
        self.metadata: dict[str, Any] = {"id": doc_id}


class StubBM25RetrieverClass:
    @staticmethod
    def reciprocal_rank_fusion(
        *ranked_lists: list[StubDoc],
        k: int = 60,
        labels: list[str] | None = None,
        weights: list[float] | None = None,
    ) -> list[StubDoc]:
        _ = k
        _ = labels
        _ = weights
        seen: set[str] = set()
        merged: list[StubDoc] = []
        for docs in ranked_lists:
            for d in docs:
                doc_id = str(d.metadata.get("id", d.page_content))
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(d)
        return merged


class StubBM25Runtime:
    def __init__(self, query_map: dict[str, list[StubDoc]]) -> None:
        self.query_map = query_map
        self.queries: list[str] = []
        self.rrf_k = 60

    def get_bm25_dir(self, _collection_name: str) -> str:
        return "bm25"

    def get_index_dir(self, collection_name: str) -> str:
        return self.get_bm25_dir(collection_name)

    def load_or_rebuild(self, *a: Any, **k: Any) -> None:
        _ = a
        _ = k

    def query(
        self, query: str, k: int, file_filter: dict[str, Any] | None = None
    ) -> list[StubDoc]:
        _ = k
        _ = file_filter
        self.queries.append(query)
        return list(self.query_map.get(query, []))


class StubGraphRuntime:
    def get_graph_dir(self, _collection_name: str) -> str:
        return "graph"

    def get_index_dir(self, collection_name: str) -> str:
        return self.get_graph_dir(collection_name)

    def load_or_rebuild(self, *a: Any, **k: Any) -> None:
        _ = a
        _ = k

    def query(
        self, query: str, k: int, file_filter: dict[str, Any] | None = None
    ) -> list[StubDoc]:
        _ = query
        _ = k
        _ = file_filter
        return []


class StubPerfLogger:
    def log(self, *_a: Any, **_k: Any) -> None:
        return None


def _extract_method(method_name: str) -> str:
    with open(_RAG_IMPL_SRC, encoding="utf-8") as fh:
        source = fh.read()
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    @|\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in RAGChatImpl.py"
    return textwrap.dedent(match.group(1))


def _load_method(name: str) -> Any:
    ns: dict[str, Any] = {
        "Any": Any,
        "cast": cast,
        "Session": StubSession,
        "CYAN": "CYAN",
        "time": time,
        "DebugHelper": _DebugHelperStub,
        "BM25Retriever": StubBM25RetrieverClass,
    }
    exec(compile(_extract_method(name), _RAG_IMPL_SRC, "exec"), ns)
    return ns[name]


_normalize_query = _load_method("_normalize_query")
_resolve_turn_translation_backend = _load_method("_resolve_turn_translation_backend")
_detect_raw_query_language = _load_method("_detect_raw_query_language")
_fetch_local_docs = _load_method("_fetch_local_docs")
_resolve_file_filter = _load_method("_resolve_file_filter")
_normalize_language_bucket = _load_method("_normalize_language_bucket")
_query_language_label = _load_method("_query_language_label")
_run_indexed_retriever_with_guardrail = _load_method(
    "_run_indexed_retriever_with_guardrail"
)
_mode_flags_for_local_retrievers = _load_method("_mode_flags_for_local_retrievers")
_is_guardrail_query_active = _load_method("_is_guardrail_query_active")
_resolve_guardrail_queries = _load_method("_resolve_guardrail_queries")
_accumulate_topk_counts = _load_method("_accumulate_topk_counts")
_indexed_retriever_specs = _load_method("_indexed_retriever_specs")
_run_indexed_local_retrievers = _load_method("_run_indexed_local_retrievers")
_store_guardrail_retrieval_topk = _load_method("_store_guardrail_retrieval_topk")
_apply_post_rewrite_translation = _load_method("_apply_post_rewrite_translation")
_rewrite_query_with_strict_retry = _load_method("_rewrite_query_with_strict_retry")
_log_query_language_drift = _load_method("_log_query_language_drift")
_set_effective_query_metadata = _load_method("_set_effective_query_metadata")
_translate_query_to_english = _load_method("_translate_query_to_english")
_shape_graph_regex_stage_queries = _load_method("_shape_graph_regex_stage_queries")


class NormalizeShell:
    _normalize_query = _normalize_query
    _resolve_turn_translation_backend = _resolve_turn_translation_backend
    _detect_raw_query_language = _detect_raw_query_language
    _apply_post_rewrite_translation = _apply_post_rewrite_translation
    _rewrite_query_with_strict_retry = _rewrite_query_with_strict_retry
    _log_query_language_drift = _log_query_language_drift
    _set_effective_query_metadata = _set_effective_query_metadata
    _translate_query_to_english = _translate_query_to_english

    def __init__(
        self,
        *,
        language_map: dict[str, str],
        translations: dict[tuple[str, str], str],
        rewrite_normal: str,
        rewrite_strict: str,
    ) -> None:
        self._translation_backend = "argos"
        self._fileUtils = StubFileUtils(language_map)
        self.pretty = StubPrettyWriter()
        self.promptRewrite = StubPromptRewrite(rewrite_normal, rewrite_strict)
        self._translator = StubTranslator(translations)

    def _get_translator(self, backend: str) -> StubTranslator | None:
        _ = backend
        return self._translator

    def _generate_alternate_queries(
        self, _query: str, _session: StubSession
    ) -> list[str]:
        return []


class FetchShell:
    _fetch_local_docs = _fetch_local_docs
    _shape_graph_regex_stage_queries = _shape_graph_regex_stage_queries
    _normalize_language_bucket = _normalize_language_bucket
    _query_language_label = _query_language_label
    _resolve_file_filter = staticmethod(_resolve_file_filter)
    _run_indexed_retriever_with_guardrail = _run_indexed_retriever_with_guardrail
    _mode_flags_for_local_retrievers = staticmethod(_mode_flags_for_local_retrievers)
    _is_guardrail_query_active = staticmethod(_is_guardrail_query_active)
    _resolve_guardrail_queries = _resolve_guardrail_queries
    _accumulate_topk_counts = staticmethod(_accumulate_topk_counts)
    _indexed_retriever_specs = _indexed_retriever_specs
    _run_indexed_local_retrievers = _run_indexed_local_retrievers
    _store_guardrail_retrieval_topk = _store_guardrail_retrieval_topk

    def __init__(self, bm25_map: dict[str, list[StubDoc]]) -> None:
        self.pretty = StubPrettyWriter()
        self.persist_directory = "db"
        self.collection_name = "Test"
        self.collection = object()
        self.perf_logger = StubPerfLogger()
        self.bm25_retriever = StubBM25Runtime(bm25_map)
        self.graph_retriever = StubGraphRuntime()
        self.regex_retriever = StubGraphRuntime()
        self._translation_backend = "off"
        self._shared = StubSharedTranslate({})
        self.retrieval_orchestrator = RetrievalOrchestrator(self)

    def _vector_kwargs(self, _session: StubSession) -> dict[str, Any]:
        return {}

    def _print_bm25_debug(self, _docs: list[Any]) -> None:
        return None

    def _print_graph_debug(self, _docs: list[Any]) -> None:
        return None

    def _print_regex_debug(self, _docs: list[Any]) -> None:
        return None


class TestNormalizeQueryContract:
    def test_tracks_user_and_retrieval_languages(self) -> None:
        raw = "was essen igel"
        orig_translated_query_en = "what do hedgehogs eat"
        rewritten_non_en = "was essen igel im winter"
        rewritten_strict_en = "what do hedgehogs eat in winter?"

        shell = NormalizeShell(
            language_map={
                raw: "german",
                orig_translated_query_en: "english",
                rewritten_non_en: "german",
                rewritten_strict_en: "english",
            },
            translations={(raw, "german"): orig_translated_query_en},
            rewrite_normal=rewritten_non_en,
            rewrite_strict=rewritten_strict_en,
        )
        session = StubSession(raw)

        final_query, alt = shell._normalize_query(session, user_query_original=raw)

        assert final_query == rewritten_strict_en
        assert alt == []
        assert session.user_language == "german"
        assert session.current_query_lang == "german"
        assert session.retrieval_language == "english"
        assert session.orig_translated_query_en == orig_translated_query_en
        assert session.t1_query == orig_translated_query_en
        assert session.rewritten_query == rewritten_strict_en
        assert session.rewrite_language == "english"
        assert session.post_rewrite_query_en == rewritten_strict_en
        assert session.t2_query == rewritten_strict_en
        assert session.effective_query_reason == "translated+rewritten"
        assert shell.promptRewrite.calls == [False, True]
        assert shell._translator.calls == [(raw, "en", "german")]

    def test_translates_rewrite_when_strict_retry_is_still_non_english(self) -> None:
        raw = "was essen igel"
        orig_translated_query_en = "what do hedgehogs eat"
        rewritten_non_en = "was essen igel im winter"
        post_rewrite_query_en = "what do hedgehogs eat in winter"

        shell = NormalizeShell(
            language_map={
                raw: "german",
                orig_translated_query_en: "english",
                rewritten_non_en: "german",
                post_rewrite_query_en: "english",
            },
            translations={
                (raw, "german"): orig_translated_query_en,
                (rewritten_non_en, "german"): post_rewrite_query_en,
            },
            rewrite_normal=rewritten_non_en,
            rewrite_strict=rewritten_non_en,
        )
        session = StubSession(raw)

        final_query, _ = shell._normalize_query(session, user_query_original=raw)

        assert final_query == post_rewrite_query_en
        assert session.rewrite_language == "german"
        assert session.post_rewrite_query_en == post_rewrite_query_en
        assert session.t2_query == post_rewrite_query_en
        assert shell.promptRewrite.calls == [False, True]
        assert shell._translator.calls == [
            (raw, "en", "german"),
            (rewritten_non_en, "en", "german"),
        ]


class TestDualQueryFusion:
    def test_runs_post_rewrite_and_orig_translated_bm25_legs_and_logs_topk(
        self,
    ) -> None:
        post_rewrite_query_en = "what do hedgehogs eat in winter"
        orig_translated_query_en = "what do hedgehogs eat"

        shell = FetchShell(
            bm25_map={
                post_rewrite_query_en: [StubDoc("A"), StubDoc("B")],
                orig_translated_query_en: [StubDoc("B"), StubDoc("C")],
            }
        )
        session = StubSession(query=post_rewrite_query_en)

        vector_docs, bm25_docs, graph_docs, regex_docs = shell._fetch_local_docs(
            session,
            retrieve_mode="BM25",
            bm25_query=post_rewrite_query_en,
            alternate_queries=[],
            orig_translated_query_en=orig_translated_query_en,
        )

        assert vector_docs == []
        assert graph_docs == []
        assert regex_docs == []
        assert shell.bm25_retriever.queries == [
            post_rewrite_query_en,
            orig_translated_query_en,
        ]
        assert session.retrieval_top_k_orig_query_en == 2
        assert session.retrieval_top_k_post_rewrite_query_en == 2
        assert session.retrieval_top_k_after_t2 == 2
        assert session.retrieval_top_k_before_t2 == 2
        assert {d.metadata["id"] for d in bm25_docs} == {"A", "B", "C"}

    def test_skips_guardrail_leg_when_queries_match(self) -> None:
        query = "what do hedgehogs eat"
        shell = FetchShell(bm25_map={query: [StubDoc("A")]})
        session = StubSession(query=query)

        _, bm25_docs, _, regex_docs = shell._fetch_local_docs(
            session,
            retrieve_mode="BM25",
            bm25_query=query,
            alternate_queries=[],
            orig_translated_query_en=query,
        )

        assert shell.bm25_retriever.queries == [query]
        assert len(bm25_docs) == 1
        assert regex_docs == []
        assert session.retrieval_top_k_orig_query_en is None
        assert session.retrieval_top_k_post_rewrite_query_en is None
        assert session.retrieval_top_k_before_t2 is None
        assert session.retrieval_top_k_after_t2 is None


class TestGraphRegexStageQueryShaping:
    def test_translates_non_english_bucket_when_argos_enabled(self) -> None:
        shell = FetchShell(bm25_map={})
        shell._translation_backend = "argos"
        shared = StubSharedTranslate(
            {
                ("Q2", "de", "en"): "Q2|de",
                ("Q1", "de", "en"): "Q1|de",
            }
        )
        shell._shared = shared

        primary_query, guardrail_query = shell._shape_graph_regex_stage_queries(
            StubSession(query="Q2"),
            language_bucket="german",
            primary_query="Q2",
            guardrail_query="Q1",
        )

        assert primary_query == "Q2|de"
        assert guardrail_query == "Q1|de"
        assert shared.calls == [
            ("Q2", "de", "en"),
            ("Q1", "de", "en"),
        ]

    def test_keeps_queries_for_english_bucket(self) -> None:
        shell = FetchShell(bm25_map={})
        shell._translation_backend = "argos"
        shared = StubSharedTranslate(
            {
                ("Q2", "de", "en"): "Q2|de",
                ("Q1", "de", "en"): "Q1|de",
            }
        )
        shell._shared = shared

        primary_query, guardrail_query = shell._shape_graph_regex_stage_queries(
            StubSession(query="Q2"),
            language_bucket="english",
            primary_query="Q2",
            guardrail_query="Q1",
        )

        assert primary_query == "Q2"
        assert guardrail_query == "Q1"
        assert shared.calls == []


class TestLocalRetrieverModeFlags:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("ALL", (True, True, True, True)),
            ("VECTOR", (True, False, False, False)),
            ("BM25_GRAPH", (False, True, True, False)),
            ("regex_vector", (True, False, False, True)),
            ("WEB", (False, False, False, False)),
        ],
    )
    def test_mode_flags(
        self, mode: str, expected: tuple[bool, bool, bool, bool]
    ) -> None:
        assert FetchShell._mode_flags_for_local_retrievers(mode) == expected


class TestGuardrailActivation:
    @pytest.mark.parametrize(
        ("primary", "guardrail", "expected"),
        [
            ("what do hedgehogs eat", "what do hedgehogs eat", False),
            ("what do hedgehogs eat", "what do hedgehogs eat in winter", True),
            ("What Do Hedgehogs Eat", "what do hedgehogs eat", False),
            ("what do hedgehogs eat", "", False),
        ],
    )
    def test_is_guardrail_query_active(
        self,
        primary: str,
        guardrail: str,
        expected: bool,
    ) -> None:
        assert FetchShell._is_guardrail_query_active(primary, guardrail) is expected


class TestGuardrailQueryResolution:
    @pytest.mark.parametrize(
        ("bm25_query", "orig_translated_query_en", "expected"),
        [
            (
                "what do hedgehogs eat",
                "what do hedgehogs eat",
                ("what do hedgehogs eat", "what do hedgehogs eat", False),
            ),
            (
                "what do hedgehogs eat in winter",
                "what do hedgehogs eat",
                ("what do hedgehogs eat in winter", "what do hedgehogs eat", True),
            ),
            (
                "what do hedgehogs eat",
                "  what do hedgehogs eat in winter  ",
                ("what do hedgehogs eat", "what do hedgehogs eat in winter", True),
            ),
        ],
    )
    def test_resolve_guardrail_queries(
        self,
        bm25_query: str,
        orig_translated_query_en: str,
        expected: tuple[str, str, bool],
    ) -> None:
        shell = FetchShell(bm25_map={})
        assert (
            shell._resolve_guardrail_queries(
                bm25_query=bm25_query,
                orig_translated_query_en=orig_translated_query_en,
            )
            == expected
        )


class TestTopKAccumulator:
    @pytest.mark.parametrize(
        ("start_orig", "start_post", "add_orig", "add_post", "expected"),
        [
            (0, 0, 2, 3, (2, 3)),
            (2, 3, 1, 4, (3, 7)),
            (10, 12, 0, 0, (10, 12)),
        ],
    )
    def test_accumulate_topk_counts(
        self,
        start_orig: int,
        start_post: int,
        add_orig: int,
        add_post: int,
        expected: tuple[int, int],
    ) -> None:
        assert (
            FetchShell._accumulate_topk_counts(
                start_orig,
                start_post,
                add_orig,
                add_post,
            )
            == expected
        )
