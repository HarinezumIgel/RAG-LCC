# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""Tests for Chat.RetrievalOrchestrator.

These tests verify orchestration sequencing, early-exit branches, and debug
trace emission at level 28.
"""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Chat.RetrievalOrchestrator import RetrievalOrchestrator


class StubPretty:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write(self, *a: Any, **k: Any) -> None:
        self.calls.append((a, k))


class StubPerfLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def log(self, *a: Any, **k: Any) -> None:
        self.calls.append((a, k))


class StubSession:
    def __init__(self) -> None:
        self.retrieve_mode: str = "ALL"
        self.query: str = "Q2"
        self.orig_translated_query_en: str | None = "Q1"
        self.seed_retrieval_query: str | None = None
        self.t1_query: str | None = None
        self.debug_level: int = 0
        self.debug_mode: str = "ge"


class LocalSessionStub:
    def __init__(self) -> None:
        self.vector_weight: float | None = None
        self.bm25_weight: float | None = None
        self.graph_weight: float | None = None
        self.regex_weight: float | None = None
        self.base_kwargs: dict[str, Any] | None = None
        self.retriever_k: int | None = 5
        self.current_query_lang: str | None = "english"
        self.user_language: str | None = "english"
        self.retrieval_language: str | None = "english"
        self.debug_level: int = 0
        self.debug_mode: str = "ge"

        self.retrieval_top_k_orig_query_en: int | None = None
        self.retrieval_top_k_post_rewrite_query_en: int | None = None
        self.retrieval_top_k_seed_query: int | None = None
        self.retrieval_top_k_final_query: int | None = None
        self.retrieval_top_k_before_t2: int | None = None
        self.retrieval_top_k_after_t2: int | None = None


class WebSessionStub:
    def __init__(self, *, web_search: bool, query: str = "QWEB") -> None:
        self.web_search = web_search
        self.query = query
        self.fetch_page_content = False
        self.collection_name = "Test"
        self.debug_level = 0
        self.debug_mode = "ge"


class WebDocStub:
    def __init__(self, url: str) -> None:
        self.page_content = f"snippet for {url}"
        self.metadata: dict[str, Any] = {
            "FilePath": url,
            "chroma_score": 0.5,
        }


class WebCfgStub:
    def __init__(
        self,
        *,
        max_results: int = 5,
        bm25_pre_filter: float = 0.0,
        cosine_pre_filter: float = 0.0,
    ) -> None:
        self.max_results = max_results
        self.bm25_pre_filter = bm25_pre_filter
        self.cosine_pre_filter = cosine_pre_filter

    def get_int(self, key: str) -> int | None:
        if key == "_WEB_SEARCH.max_results":
            return self.max_results
        return None

    def get_float(self, key: str) -> float | None:
        if key == "_WEB_SEARCH.bm25_pre_filter":
            return self.bm25_pre_filter
        if key == "_WEB_SEARCH.cosine_pre_filter":
            return self.cosine_pre_filter
        return None


class WebRetrieverStub:
    def __init__(self, docs: list[WebDocStub]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int, bool, str, str]] = []

    def query(
        self,
        query: str,
        *,
        k: int,
        fetch_page_content: bool,
        original_query: str,
        collection: str,
    ) -> list[WebDocStub]:
        self.calls.append((query, k, fetch_page_content, original_query, collection))
        return list(self.docs)


class WebPreFilterStub:
    def __init__(
        self,
        *,
        bm25_out: list[WebDocStub] | None = None,
        cosine_out: list[WebDocStub] | None = None,
    ) -> None:
        self.bm25_out = bm25_out
        self.cosine_out = cosine_out
        self.bm25_calls: list[tuple[list[WebDocStub], str]] = []
        self.cosine_calls: list[tuple[list[WebDocStub], str, list[float]]] = []

    def bm25_prefilter(self, docs: list[WebDocStub], query: str) -> list[WebDocStub]:
        self.bm25_calls.append((list(docs), query))
        if self.bm25_out is not None:
            return list(self.bm25_out)
        return list(docs)

    def cosine_prefilter(
        self,
        docs: list[WebDocStub],
        query: str,
        *,
        query_vec: list[float],
    ) -> list[WebDocStub]:
        self.cosine_calls.append((list(docs), query, list(query_vec)))
        if self.cosine_out is not None:
            return list(self.cosine_out)
        return list(docs)


class EmbedderStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return [0.1, 0.2]


class HostStub:
    def __init__(
        self, *, set_vector_ok: bool = True, gate_result: bool = False
    ) -> None:
        self.set_vector_ok = set_vector_ok
        self.gate_result = gate_result

        self.pretty = StubPretty()
        self.perf_logger = StubPerfLogger()

        self.calls: list[str] = []
        self.merge_args: tuple[Any, ...] | None = None

    def _set_vector_store(self, session: StubSession) -> bool:
        _ = session
        self.calls.append("_set_vector_store")
        return self.set_vector_ok

    def _prepare_session(self, session: StubSession) -> str:
        _ = session
        self.calls.append("_prepare_session")
        return "USER_ORIG"

    def _normalize_query(
        self, session: StubSession, user_query_original: str
    ) -> tuple[str, list[str]]:
        _ = session
        _ = user_query_original
        self.calls.append("_normalize_query")
        return "Q2", ["ALT1", "ALT2"]

    def _check_gates(self, session: StubSession, final_query: str) -> bool:
        _ = session
        _ = final_query
        self.calls.append("_check_gates")
        return self.gate_result

    def _resolve_guardrail_queries(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> tuple[str, str, bool]:
        self.calls.append("_resolve_guardrail_queries")
        primary_query = bm25_query or ""
        guardrail_query = (orig_translated_query_en or "").strip()
        use_guardrail = bool(
            guardrail_query and guardrail_query.lower() != primary_query.strip().lower()
        )
        return primary_query, guardrail_query, use_guardrail

    def _merge_and_select(
        self,
        session: StubSession,
        vector_docs: list[Any],
        bm25_docs: list[Any],
        graph_docs: list[Any],
        regex_docs: list[Any],
        web_docs: list[Any],
    ) -> list[Any]:
        _ = session
        self.calls.append("_merge_and_select")
        self.merge_args = (
            list(vector_docs),
            list(bm25_docs),
            list(graph_docs),
            list(regex_docs),
            list(web_docs),
        )
        return ["chosen1", "chosen2"]

    def _build_context(
        self, session: StubSession, chosen: list[Any]
    ) -> tuple[str, int]:
        _ = session
        self.calls.append("_build_context")
        return (f"CTX::{','.join(chosen)}", len(chosen))


class _LocalCollectionStub:
    def __init__(self, metadatas: list[dict[str, Any]] | None = None) -> None:
        self.metadatas = metadatas or [{"Language": "en"}]

    def get(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {"metadatas": list(self.metadatas)}


class _LocalCfgStub:
    def __init__(
        self,
        pairs: list[list[str]] | None = None,
        active_languages: list[str] | None = None,
    ) -> None:
        self.pairs = pairs or [["en", "de"]]
        self.active_languages = active_languages or ["en", "de"]

    def get_list(self, key: str, default: Any, silent: bool = False) -> Any:
        _ = (default, silent)
        if key == "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES":
            return list(self.active_languages)
        if key == "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES":
            return list(self.pairs)
        return []


class _LocalSharedStub:
    def __init__(self) -> None:
        self.lang_name_to_code: dict[str, str] = {
            "english": "en",
            "german": "de",
        }


class _LocalTranslatingSharedStub(_LocalSharedStub):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, str]] = []

    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> str:
        self.calls.append((text, target_lang, source_lang))
        return f"{text}|{target_lang}"


class LocalHostStub:
    def __init__(
        self,
        *,
        use_guardrail: bool,
        mode_flags: tuple[bool, bool, bool, bool] = (False, True, False, False),
    ) -> None:
        self.use_guardrail = use_guardrail
        self.mode_flags = mode_flags
        self.calls: list[str] = []
        self.last_store_args: tuple[int, int] | None = None
        self.last_run_indexed_args: tuple[str, bool, dict[str, Any] | None] | None = (
            None
        )
        self.indexed_stage_labels: list[list[str]] = []
        self.run_indexed_file_filters: list[dict[str, Any] | None] = []
        self.run_indexed_primary_queries: list[str] = []
        self.run_indexed_guardrail_queries: list[str] = []
        self.collection = _LocalCollectionStub()
        self.cfg = _LocalCfgStub()
        self._shared = _LocalSharedStub()
        self.pretty = StubPretty()
        self.collection_name = "TestCollection"

    def _mode_flags_for_local_retrievers(
        self, retrieve_mode: str
    ) -> tuple[bool, bool, bool, bool]:
        self.calls.append("_mode_flags_for_local_retrievers")
        _ = retrieve_mode
        return self.mode_flags

    def _resolve_guardrail_queries(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> tuple[str, str, bool]:
        self.calls.append("_resolve_guardrail_queries")
        return bm25_query, orig_translated_query_en.strip(), self.use_guardrail

    def _normalize_language_bucket(self, language: Any) -> str | None:
        text = str(language or "").strip().lower()
        if not text:
            return None

        mapping_obj = getattr(self._shared, "lang_name_to_code", None)
        if isinstance(mapping_obj, dict):
            mapped = mapping_obj.get(text)
            if mapped is not None:
                mapped_text = str(mapped).strip().lower()
                if mapped_text:
                    return mapped_text

        return text

    def _shape_graph_regex_stage_queries(
        self,
        mySession: LocalSessionStub,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
    ) -> tuple[str, str]:
        _ = mySession
        target_language = self._normalize_language_bucket(language_bucket)
        if not target_language:
            return primary_query, guardrail_query

        if not target_language:
            return primary_query, guardrail_query
        if target_language == "en" or target_language.startswith("en-"):
            return primary_query, guardrail_query

        backend = str(getattr(self, "_translation_backend", "off") or "off")
        if backend.strip().lower() != "argos":
            return primary_query, guardrail_query

        translate_text = getattr(self._shared, "translate_text", None)
        if not callable(translate_text):
            return primary_query, guardrail_query

        def _translate_query(query: str) -> str:
            if not query:
                return query
            try:
                translated_obj = translate_text(query, target_language, "en")
            except Exception:
                return query
            translated = str(translated_obj or "").strip()
            return translated or query

        return _translate_query(primary_query), _translate_query(guardrail_query)

    def _resolve_file_filter(self, session: LocalSessionStub) -> dict[str, Any] | None:
        self.calls.append("_resolve_file_filter")
        _ = session
        return {"FileName": "A.txt"}

    def _run_vector_retriever_with_guardrail(
        self,
        *,
        mySession: LocalSessionStub,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> tuple[list[Any], int, int]:
        self.calls.append("_run_vector_retriever_with_guardrail")
        _ = mySession
        _ = primary_query
        _ = guardrail_query
        _ = use_guardrail
        _ = alternate_queries
        return (["v1"], 1, 2)

    @staticmethod
    def _accumulate_topk_counts(
        current_orig_query_en: int,
        current_post_rewrite_query_en: int,
        add_orig_query_en: int,
        add_post_rewrite_query_en: int,
    ) -> tuple[int, int]:
        return (
            current_orig_query_en + add_orig_query_en,
            current_post_rewrite_query_en + add_post_rewrite_query_en,
        )

    def _indexed_retriever_specs(
        self,
        mySession: LocalSessionStub,
        *,
        bm25_enabled: bool,
        graph_enabled: bool,
        regex_enabled: bool,
    ) -> list[tuple[str, bool, float, Any, int, Any]]:
        self.calls.append("_indexed_retriever_specs")
        _ = mySession

        def _noop_debug_printer(docs: list[Any]) -> None:
            _ = docs

        return [
            ("BM25", bm25_enabled, 1.0, object(), 10, _noop_debug_printer),
            ("Graph", graph_enabled, 1.0, object(), 30, _noop_debug_printer),
            ("Regex", regex_enabled, 1.0, object(), 30, _noop_debug_printer),
        ]

    def _run_indexed_local_retrievers(
        self,
        mySession: LocalSessionStub,
        *,
        indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]:
        self.calls.append("_run_indexed_local_retrievers")
        _ = mySession
        active_labels = [
            label
            for label, enabled, weight, *_ in indexed_specs
            if enabled and weight != 0.0
        ]
        self.indexed_stage_labels.append(active_labels)
        self.run_indexed_file_filters.append(file_filter)
        self.run_indexed_primary_queries.append(primary_query)
        self.run_indexed_guardrail_queries.append(guardrail_query)
        self.last_run_indexed_args = (primary_query, use_guardrail, file_filter)

        docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        top_k_orig = 0
        top_k_post = 0

        if "BM25" in active_labels:
            if guardrail_query and use_guardrail:
                docs["BM25"] = ["b1", "b2"]
                top_k_orig += 2
                top_k_post += 3
            else:
                docs["BM25"] = ["b1"]
                top_k_post += 1

        if "Graph" in active_labels:
            docs["Graph"] = ["g1"]

        if "Regex" in active_labels:
            docs["Regex"] = ["r1"]

        return docs, top_k_orig, top_k_post

    def _store_guardrail_retrieval_topk(
        self,
        mySession: LocalSessionStub,
        *,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None:
        self.calls.append("_store_guardrail_retrieval_topk")
        self.last_store_args = (top_k_orig_query_en, top_k_post_rewrite_query_en)
        mySession.retrieval_top_k_orig_query_en = top_k_orig_query_en
        mySession.retrieval_top_k_post_rewrite_query_en = top_k_post_rewrite_query_en


class WebHostStub:
    def __init__(
        self,
        docs: list[WebDocStub],
        *,
        cfg: WebCfgStub,
        bm25_out: list[WebDocStub] | None = None,
        cosine_out: list[WebDocStub] | None = None,
    ) -> None:
        self.pretty = StubPretty()
        self.perf_logger = StubPerfLogger()
        self.cfg = cfg
        self.web_retriever = WebRetrieverStub(docs)
        self.web_pre_filter = WebPreFilterStub(
            bm25_out=bm25_out,
            cosine_out=cosine_out,
        )
        self.embedder = EmbedderStub()

        self.web_debug_calls: list[list[Any]] = []
        self.web_prefilter_debug_calls: list[tuple[str, float, int, int]] = []

    def _print_web_debug(self, docs: list[Any]) -> None:
        self.web_debug_calls.append(list(docs))

    def _print_web_prefilter_debug(
        self,
        kept: list[Any],
        dropped: list[Any],
        filter_name: str,
        threshold: float,
    ) -> None:
        self.web_prefilter_debug_calls.append(
            (filter_name, threshold, len(kept), len(dropped))
        )


class TestRetrievalOrchestrator:
    def test_run_happy_path_sequences_stages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()

        orchestrator = RetrievalOrchestrator(host)
        captured_local_args: tuple[str, str, list[str], str] | None = None
        captured_web_args: tuple[str, str] | None = None
        captured_guardrail_use: bool | None = None

        def _fake_run_local_docs_stage(
            my_session: StubSession,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            nonlocal captured_local_args, captured_guardrail_use
            _ = my_session
            captured_local_args = (
                retrieve_mode,
                bm25_query,
                list(alternate_queries),
                orig_translated_query_en,
            )
            if resolved_guardrail is not None:
                captured_guardrail_use = bool(
                    getattr(resolved_guardrail, "use_guardrail", False)
                )
            return type(
                "_Docs",
                (),
                {
                    "vector_docs": ["v1"],
                    "bm25_docs": ["b1", "b2"],
                    "graph_docs": ["g1"],
                    "regex_docs": ["r1"],
                },
            )()

        def _fake_run_web_retriever(
            my_session: StubSession,
            retrieve_mode: str,
            user_query_original: str,
        ) -> list[Any]:
            nonlocal captured_web_args
            _ = my_session
            captured_web_args = (retrieve_mode, user_query_original)
            return ["w1"]

        monkeypatch.setattr(
            orchestrator,
            "_run_local_docs_stage",
            _fake_run_local_docs_stage,
        )
        monkeypatch.setattr(
            orchestrator,
            "run_web_retriever",
            _fake_run_web_retriever,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("CTX::chosen1,chosen2", 2)
        assert host.calls == [
            "_set_vector_store",
            "_prepare_session",
            "_normalize_query",
            "_check_gates",
            "_resolve_guardrail_queries",
            "_merge_and_select",
            "_build_context",
        ]
        assert captured_local_args == ("ALL", "Q2", ["ALT1", "ALT2"], "Q1")
        assert captured_guardrail_use is True
        assert captured_web_args == ("ALL", "USER_ORIG")
        assert host.merge_args == (["v1"], ["b1", "b2"], ["g1"], ["r1"], ["w1"])
        assert len(host.perf_logger.calls) == 1

    def test_run_routes_through_post_gate_pipeline_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        orchestrator = RetrievalOrchestrator(host)

        captured: tuple[Any, Any] | None = None

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            nonlocal captured
            captured = (mySession, pipeline_inputs)
            return "CTX::PIPE", 9

        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("CTX::PIPE", 9)
        assert host.calls == [
            "_set_vector_store",
            "_prepare_session",
            "_normalize_query",
            "_check_gates",
            "_resolve_guardrail_queries",
        ]
        assert captured is not None
        assert captured[0] is session
        pipeline_inputs = captured[1]
        assert pipeline_inputs.retrieve_mode == "ALL"
        assert pipeline_inputs.bm25_query == "Q2"
        assert pipeline_inputs.alternate_queries == ["ALT1", "ALT2"]
        assert pipeline_inputs.orig_translated_query_en == "Q1"
        assert pipeline_inputs.user_query_original == "USER_ORIG"
        assert bool(getattr(pipeline_inputs.resolved_guardrail, "use_guardrail", False))

    def test_run_routes_through_pre_gate_pipeline_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        orchestrator = RetrievalOrchestrator(host)

        captured_pre_gate_session: StubSession | None = None
        captured_post_gate_inputs: Any = None

        class _Guardrail:
            def __init__(self) -> None:
                self.primary_query = "Q2"
                self.guardrail_query = "Q1"
                self.use_guardrail = True

        class _PipelineInputs:
            def __init__(self) -> None:
                self.retrieve_mode = "ALL"
                self.bm25_query = "Q2"
                self.alternate_queries = ["ALT1"]
                self.orig_translated_query_en = "Q1"
                self.user_query_original = "USER_ORIG"
                self.resolved_guardrail = _Guardrail()

        class _PreGateResult:
            def __init__(self) -> None:
                self.should_abort = False
                self.post_gate_inputs = _PipelineInputs()

        def _fake_run_pre_gate_pipeline(mySession: StubSession) -> Any:
            nonlocal captured_pre_gate_session
            captured_pre_gate_session = mySession
            return _PreGateResult()

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            nonlocal captured_post_gate_inputs
            _ = mySession
            captured_post_gate_inputs = pipeline_inputs
            return "CTX::POST", 4

        monkeypatch.setattr(
            orchestrator,
            "_run_pre_gate_pipeline",
            _fake_run_pre_gate_pipeline,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("CTX::POST", 4)
        assert captured_pre_gate_session is session
        assert captured_post_gate_inputs is not None
        assert captured_post_gate_inputs.retrieve_mode == "ALL"
        assert captured_post_gate_inputs.bm25_query == "Q2"
        assert host.calls == ["_set_vector_store"]

    def test_run_aborts_when_pre_gate_pipeline_requests_abort(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        orchestrator = RetrievalOrchestrator(host)

        post_gate_called = False

        class _PreGateAbort:
            def __init__(self) -> None:
                self.should_abort = True
                self.post_gate_inputs = None

        def _fake_run_pre_gate_pipeline(mySession: StubSession) -> Any:
            _ = mySession
            return _PreGateAbort()

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            nonlocal post_gate_called
            _ = (mySession, pipeline_inputs)
            post_gate_called = True
            return "CTX::UNEXPECTED", 99

        monkeypatch.setattr(
            orchestrator,
            "_run_pre_gate_pipeline",
            _fake_run_pre_gate_pipeline,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("", 0)
        assert post_gate_called is False
        assert host.calls == ["_set_vector_store"]

    def test_run_early_exit_when_vector_store_setup_fails(self) -> None:
        host = HostStub(set_vector_ok=False, gate_result=False)
        session = StubSession()

        orchestrator = RetrievalOrchestrator(host)
        context, count = orchestrator.run(session)

        assert (context, count) == ("", 0)
        assert host.calls == ["_set_vector_store"]
        assert len(host.perf_logger.calls) == 0

    def test_run_early_exit_when_gate_blocks(self) -> None:
        host = HostStub(set_vector_ok=True, gate_result=True)
        session = StubSession()

        orchestrator = RetrievalOrchestrator(host)
        context, count = orchestrator.run(session)

        assert (context, count) == ("", 0)
        assert host.calls == [
            "_set_vector_store",
            "_prepare_session",
            "_normalize_query",
            "_check_gates",
        ]

    def test_run_web_only_skips_local_retrieval_stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        session.retrieve_mode = "WEB"

        orchestrator = RetrievalOrchestrator(host)

        local_stage_called = False

        def _fail_if_local_stage_called(
            my_session: StubSession,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            nonlocal local_stage_called
            _ = (
                my_session,
                retrieve_mode,
                bm25_query,
                alternate_queries,
                orig_translated_query_en,
                resolved_guardrail,
            )
            local_stage_called = True
            raise AssertionError("_run_local_docs_stage should not run in WEB mode")

        def _fake_run_web_retriever(
            my_session: StubSession,
            retrieve_mode: str,
            user_query_original: str,
        ) -> list[Any]:
            _ = (my_session, user_query_original)
            assert retrieve_mode == "WEB"
            return ["w1"]

        monkeypatch.setattr(
            orchestrator,
            "_run_local_docs_stage",
            _fail_if_local_stage_called,
        )
        monkeypatch.setattr(
            orchestrator,
            "run_web_retriever",
            _fake_run_web_retriever,
        )

        context, count = orchestrator.run(session)

        assert local_stage_called is False
        assert (context, count) == ("CTX::chosen1,chosen2", 2)

        trace_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]
        assert not any(
            "retrieve local candidates" in message for message in trace_messages
        )
        assert not any("local docs fetched" in message for message in trace_messages)
        assert not any("idx_stage l=" in message for message in trace_messages)
        assert any("retrieve web candidates" in message for message in trace_messages)

    def test_trace_emitted_without_debug_level_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        session.debug_level = 0

        orchestrator = RetrievalOrchestrator(host)

        def _fake_run_local_docs_stage(
            my_session: StubSession,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            _ = (
                my_session,
                retrieve_mode,
                bm25_query,
                alternate_queries,
                orig_translated_query_en,
                resolved_guardrail,
            )
            return type(
                "_Docs",
                (),
                {
                    "vector_docs": [],
                    "bm25_docs": [],
                    "graph_docs": [],
                    "regex_docs": [],
                },
            )()

        def _fake_run_web_retriever(
            my_session: StubSession,
            retrieve_mode: str,
            user_query_original: str,
        ) -> list[Any]:
            _ = (my_session, retrieve_mode, user_query_original)
            return []

        monkeypatch.setattr(
            orchestrator,
            "_run_local_docs_stage",
            _fake_run_local_docs_stage,
        )
        monkeypatch.setattr(
            orchestrator,
            "run_web_retriever",
            _fake_run_web_retriever,
        )

        orchestrator.run(session)

        trace_msgs = [
            call[0][2]
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]
        assert any(
            len(call[0]) >= 2
            and call[0][0] == "I"
            and call[0][1] == "Retrieval Orchestration"
            for call in host.pretty.calls
        )
        assert any("start retrieval turn" in msg for msg in trace_msgs)
        assert any("prepare session context" in msg for msg in trace_msgs)
        assert any("normalize user query" in msg for msg in trace_msgs)
        assert any("retrieve local candidates" in msg for msg in trace_msgs)
        assert any("retrieval completed" in msg for msg in trace_msgs)
        local_candidate_messages = [
            msg for msg in trace_msgs if "retrieve local candidates" in msg
        ]
        assert local_candidate_messages
        assert all("mode=ALL" in msg for msg in local_candidate_messages)
        assert all("primary_query=" not in msg for msg in local_candidate_messages)
        assert all("query:" not in msg for msg in local_candidate_messages)

    def test_run_emits_query_rewrite_status_before_gate_checks(self) -> None:
        host = HostStub(set_vector_ok=True, gate_result=True)
        host._shared = type(
            "_Shared",
            (),
            {"lang_name_to_code": {"english": "en", "german": "de"}},
        )()

        session = StubSession()
        session.user_language = "german"  # type: ignore[attr-defined]
        session.current_query_lang = "german"  # type: ignore[attr-defined]
        session.rewrite_language = "english"  # type: ignore[attr-defined]
        session.retrieval_language = "english"  # type: ignore[attr-defined]
        session.rewritten_query = "what do hedgehogs eat"  # type: ignore[attr-defined]
        session.post_rewrite_query_en = "what do hedgehogs eat"  # type: ignore[attr-defined]
        session.orig_translated_query_en = "what eat hedgehogs"  # type: ignore[attr-defined]

        orchestrator = RetrievalOrchestrator(host)

        _ = orchestrator.run(session)

        info_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3
            and call[0][0] == "I"
            and call[0][1] == "Retrieval Orchestration"
        ]
        assert not any("query language flow:" in msg for msg in info_messages)
        assert any(
            "query rewrite flow:" in msg
            and "\nseed_en='what eat hedgehogs'" in msg
            and "\nrewritten='what do hedgehogs eat'" in msg
            and "\nfinal_en='what do hedgehogs eat'" in msg
            for msg in info_messages
        )


class TestLocalRetrievalOrchestration:
    def test_run_local_retrievers_stores_guardrail_counts_when_active(self) -> None:
        host = LocalHostStub(use_guardrail=True)
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="BM25",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b2"]
        assert graph_docs == []
        assert regex_docs == []
        assert host.last_run_indexed_args == ("Q2", True, {"FileName": "A.txt"})
        assert host.indexed_stage_labels == [["BM25"]]
        assert host.last_store_args == (2, 3)
        assert session.retrieval_top_k_orig_query_en == 2
        assert session.retrieval_top_k_post_rewrite_query_en == 3

    def test_run_local_retrievers_skips_guardrail_store_when_inactive(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="BM25",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1"]
        assert graph_docs == []
        assert regex_docs == []
        assert host.last_run_indexed_args == ("Q2", False, {"FileName": "A.txt"})
        assert host.indexed_stage_labels == [["BM25"]]
        assert host.last_store_args is None
        assert session.retrieval_top_k_orig_query_en is None
        assert session.retrieval_top_k_post_rewrite_query_en is None

    def test_run_local_retrievers_splits_indexed_stages(self) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1"]
        assert graph_docs == ["g1"]
        assert regex_docs == ["r1"]
        assert host.indexed_stage_labels == [["BM25"], ["Graph"], ["Regex"]]
        assert host.run_indexed_file_filters == [
            {"FileName": "A.txt"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "en"},
        ]

    def test_run_local_retrievers_routes_calls_through_stage_seams(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        bm25_stage_labels_seen: list[list[str]] = []
        graph_regex_stage_labels_seen: list[list[str]] = []

        original_bm25_stage = orchestrator._run_bm25_indexed_stage
        original_graph_regex_stage = orchestrator._run_graph_regex_indexed_stage

        def _wrapped_bm25_stage(
            mySession: LocalSessionStub,
            *,
            bm25_specs: list[tuple[str, bool, float, Any, int, Any]],
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            file_filter: dict[str, Any] | None,
        ) -> tuple[dict[str, list[Any]], int, int]:
            _ = (mySession, primary_query, guardrail_query, use_guardrail, file_filter)
            bm25_stage_labels_seen.append([spec[0] for spec in bm25_specs])
            return original_bm25_stage(
                mySession,
                bm25_specs=bm25_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )

        def _wrapped_graph_regex_stage(
            mySession: LocalSessionStub,
            *,
            graph_regex_specs: list[tuple[str, bool, float, Any, int, Any]],
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            file_filter: dict[str, Any] | None,
        ) -> tuple[dict[str, list[Any]], int, int]:
            _ = (mySession, primary_query, guardrail_query, use_guardrail, file_filter)
            graph_regex_stage_labels_seen.append(
                [spec[0] for spec in graph_regex_specs]
            )
            return original_graph_regex_stage(
                mySession,
                graph_regex_specs=graph_regex_specs,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )

        monkeypatch.setattr(
            orchestrator,
            "_run_bm25_indexed_stage",
            _wrapped_bm25_stage,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_graph_regex_indexed_stage",
            _wrapped_graph_regex_stage,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1"]
        assert graph_docs == ["g1"]
        assert regex_docs == ["r1"]
        assert bm25_stage_labels_seen == [["BM25"]]
        assert graph_regex_stage_labels_seen == [["Graph", "Regex"]]

    def test_run_local_retrievers_bm25_stage_uses_invocation_planner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=True)
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        class _Invocation:
            def __init__(
                self,
                file_filter: dict[str, Any] | None,
                primary_query: str,
                guardrail_query: str,
            ) -> None:
                self.file_filter = file_filter
                self.primary_query = primary_query
                self.guardrail_query = guardrail_query

        def _fake_bm25_stage_invocation(
            mySession: LocalSessionStub,
            *,
            file_filter: dict[str, Any] | None,
            primary_query: str,
            guardrail_query: str,
        ) -> Any:
            _ = (mySession, file_filter, primary_query, guardrail_query)
            return _Invocation(
                {"FileName": "B.txt"},
                "Q2|bm25",
                "Q1|bm25",
            )

        monkeypatch.setattr(
            orchestrator,
            "_bm25_stage_invocation",
            _fake_bm25_stage_invocation,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="BM25",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b2"]
        assert graph_docs == []
        assert regex_docs == []
        assert host.indexed_stage_labels == [["BM25"]]
        assert host.run_indexed_file_filters == [{"FileName": "B.txt"}]
        assert host.run_indexed_primary_queries == ["Q2|bm25"]
        assert host.run_indexed_guardrail_queries == ["Q1|bm25"]

    def test_run_local_retrievers_routes_through_indexed_stage_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        captured_args: tuple[Any, ...] | None = None

        class _IndexedResult:
            def __init__(self) -> None:
                self.bm25_docs = ["bX"]
                self.graph_docs = ["gX"]
                self.regex_docs = ["rX"]
                self.top_k_orig_query_en = 4
                self.top_k_post_rewrite_query_en = 7

        def _fake_run_indexed_retrieval_stages(
            mySession: LocalSessionStub,
            *,
            indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            file_filter: dict[str, Any] | None,
        ) -> Any:
            nonlocal captured_args
            captured_args = (
                mySession,
                [spec[0] for spec in indexed_specs],
                primary_query,
                guardrail_query,
                use_guardrail,
                file_filter,
            )
            return _IndexedResult()

        monkeypatch.setattr(
            orchestrator,
            "_run_indexed_retrieval_stages",
            _fake_run_indexed_retrieval_stages,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["bX"]
        assert graph_docs == ["gX"]
        assert regex_docs == ["rX"]
        assert captured_args is not None
        assert captured_args[0] is session
        assert captured_args[1] == ["BM25", "Graph", "Regex"]
        assert captured_args[2] == "Q2"
        assert captured_args[3] == "Q1"
        assert captured_args[4] is True
        assert captured_args[5] == {"FileName": "A.txt"}
        assert host.last_store_args == (4, 7)

    def test_run_local_retrievers_routes_through_vector_stage_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(True, True, False, False),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        captured_vector_args: tuple[Any, ...] | None = None

        class _VectorResult:
            def __init__(self) -> None:
                self.vector_docs = ["vX"]
                self.top_k_orig_query_en = 2
                self.top_k_post_rewrite_query_en = 5

        class _IndexedResult:
            def __init__(self) -> None:
                self.bm25_docs = ["bX"]
                self.graph_docs: list[Any] = []
                self.regex_docs: list[Any] = []
                self.top_k_orig_query_en = 3
                self.top_k_post_rewrite_query_en = 4

        def _fake_run_vector_stage(
            mySession: LocalSessionStub,
            *,
            vector_enabled: bool,
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            alternate_queries: list[str],
        ) -> Any:
            nonlocal captured_vector_args
            captured_vector_args = (
                mySession,
                vector_enabled,
                primary_query,
                guardrail_query,
                use_guardrail,
                list(alternate_queries),
            )
            return _VectorResult()

        def _fake_run_indexed_retrieval_stages(
            mySession: LocalSessionStub,
            *,
            indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            file_filter: dict[str, Any] | None,
        ) -> Any:
            _ = (
                mySession,
                indexed_specs,
                primary_query,
                guardrail_query,
                use_guardrail,
                file_filter,
            )
            return _IndexedResult()

        monkeypatch.setattr(
            orchestrator,
            "_run_vector_stage",
            _fake_run_vector_stage,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_indexed_retrieval_stages",
            _fake_run_indexed_retrieval_stages,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=["ALT1"],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == ["vX"]
        assert bm25_docs == ["bX"]
        assert graph_docs == []
        assert regex_docs == []
        assert captured_vector_args is not None
        assert captured_vector_args[0] is session
        assert captured_vector_args[1] is True
        assert captured_vector_args[2] == "Q2"
        assert captured_vector_args[3] == "Q1"
        assert captured_vector_args[4] is True
        assert captured_vector_args[5] == ["ALT1"]
        assert host.last_store_args == (5, 9)

    def test_run_local_retrievers_routes_through_local_stage_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(True, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        captured_local_stage_inputs: Any = None

        class _LocalInputs:
            def __init__(self) -> None:
                self.vector_enabled = True
                self.bm25_enabled = True
                self.graph_enabled = True
                self.regex_enabled = True
                self.primary_query = "Q2"
                self.guardrail_query = "Q1"
                self.use_guardrail = True
                self.alternate_queries = ["ALT1", "ALT2"]
                self.file_filter = {"FileName": "A.txt"}

        class _LocalStageResult:
            def __init__(self) -> None:
                self.local_docs = type(
                    "_Docs",
                    (),
                    {
                        "vector_docs": ["vZ"],
                        "bm25_docs": ["bZ"],
                        "graph_docs": ["gZ"],
                        "regex_docs": ["rZ"],
                    },
                )()
                self.top_k_orig_query_en = 6
                self.top_k_post_rewrite_query_en = 8

        def _fake_resolve_local_stage_inputs(
            mySession: LocalSessionStub,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            _ = (
                mySession,
                retrieve_mode,
                bm25_query,
                alternate_queries,
                orig_translated_query_en,
                resolved_guardrail,
            )
            return _LocalInputs()

        def _fake_run_local_retrieval_stages(
            mySession: LocalSessionStub,
            *,
            local_inputs: Any,
        ) -> Any:
            nonlocal captured_local_stage_inputs
            _ = mySession
            captured_local_stage_inputs = local_inputs
            return _LocalStageResult()

        monkeypatch.setattr(
            orchestrator,
            "_resolve_local_stage_inputs",
            _fake_resolve_local_stage_inputs,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_local_retrieval_stages",
            _fake_run_local_retrieval_stages,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=["ALT1", "ALT2"],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == ["vZ"]
        assert bm25_docs == ["bZ"]
        assert graph_docs == ["gZ"]
        assert regex_docs == ["rZ"]
        assert captured_local_stage_inputs is not None
        assert captured_local_stage_inputs.vector_enabled is True
        assert captured_local_stage_inputs.bm25_enabled is True
        assert captured_local_stage_inputs.graph_enabled is True
        assert captured_local_stage_inputs.regex_enabled is True
        assert captured_local_stage_inputs.primary_query == "Q2"
        assert captured_local_stage_inputs.guardrail_query == "Q1"
        assert captured_local_stage_inputs.use_guardrail is True
        assert captured_local_stage_inputs.alternate_queries == ["ALT1", "ALT2"]
        assert captured_local_stage_inputs.file_filter == {"FileName": "A.txt"}
        assert host.last_store_args == (6, 8)

    def test_run_local_retrievers_delegates_to_local_docs_stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        captured_args: tuple[Any, ...] | None = None

        class _Docs:
            def __init__(self) -> None:
                self.vector_docs = ["vD"]
                self.bm25_docs = ["bD"]
                self.graph_docs = ["gD"]
                self.regex_docs = ["rD"]

        def _fake_run_local_docs_stage(
            mySession: LocalSessionStub,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            nonlocal captured_args
            captured_args = (
                mySession,
                retrieve_mode,
                bm25_query,
                list(alternate_queries),
                orig_translated_query_en,
                resolved_guardrail,
            )
            return _Docs()

        monkeypatch.setattr(
            orchestrator,
            "_run_local_docs_stage",
            _fake_run_local_docs_stage,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=["ALT1"],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == ["vD"]
        assert bm25_docs == ["bD"]
        assert graph_docs == ["gD"]
        assert regex_docs == ["rD"]
        assert captured_args is not None
        assert captured_args[0] is session
        assert captured_args[1] == "ALL"
        assert captured_args[2] == "Q2"
        assert captured_args[3] == ["ALT1"]
        assert captured_args[4] == "Q1"
        assert captured_args[5] is None

    def test_run_local_docs_stage_routes_guardrail_persistence_via_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=True)
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        captured_guardrail_store: tuple[Any, bool, int, int] | None = None

        class _LocalInputs:
            def __init__(self) -> None:
                self.vector_enabled = True
                self.bm25_enabled = True
                self.graph_enabled = True
                self.regex_enabled = True
                self.primary_query = "Q2"
                self.guardrail_query = "Q1"
                self.use_guardrail = True
                self.alternate_queries = []
                self.file_filter = {"FileName": "A.txt"}

        class _LocalStageResult:
            def __init__(self) -> None:
                self.local_docs = type(
                    "_Docs",
                    (),
                    {
                        "vector_docs": ["vY"],
                        "bm25_docs": ["bY"],
                        "graph_docs": ["gY"],
                        "regex_docs": ["rY"],
                    },
                )()
                self.top_k_orig_query_en = 11
                self.top_k_post_rewrite_query_en = 13

        def _fake_resolve_local_stage_inputs(
            mySession: LocalSessionStub,
            *,
            retrieve_mode: str,
            bm25_query: str,
            alternate_queries: list[str],
            orig_translated_query_en: str,
            resolved_guardrail: Any | None,
        ) -> Any:
            _ = (
                mySession,
                retrieve_mode,
                bm25_query,
                alternate_queries,
                orig_translated_query_en,
                resolved_guardrail,
            )
            return _LocalInputs()

        def _fake_run_local_retrieval_stages(
            mySession: LocalSessionStub,
            *,
            local_inputs: Any,
        ) -> Any:
            _ = (mySession, local_inputs)
            return _LocalStageResult()

        def _fake_store_guardrail_topk_if_enabled(
            mySession: LocalSessionStub,
            *,
            use_guardrail: bool,
            top_k_orig_query_en: int,
            top_k_post_rewrite_query_en: int,
        ) -> None:
            nonlocal captured_guardrail_store
            captured_guardrail_store = (
                mySession,
                use_guardrail,
                top_k_orig_query_en,
                top_k_post_rewrite_query_en,
            )

        monkeypatch.setattr(
            orchestrator,
            "_resolve_local_stage_inputs",
            _fake_resolve_local_stage_inputs,
        )
        monkeypatch.setattr(
            orchestrator,
            "_run_local_retrieval_stages",
            _fake_run_local_retrieval_stages,
        )
        monkeypatch.setattr(
            orchestrator,
            "_store_guardrail_topk_if_enabled",
            _fake_store_guardrail_topk_if_enabled,
        )

        local_docs = orchestrator._run_local_docs_stage(
            session,
            retrieve_mode="ALL",
            bm25_query="Q2",
            alternate_queries=[],
            orig_translated_query_en="Q1",
            resolved_guardrail=None,
        )

        assert local_docs.vector_docs == ["vY"]
        assert local_docs.bm25_docs == ["bY"]
        assert local_docs.graph_docs == ["gY"]
        assert local_docs.regex_docs == ["rY"]
        assert captured_guardrail_store is not None
        assert captured_guardrail_store[0] is session
        assert captured_guardrail_store[1] is True
        assert captured_guardrail_store[2] == 11
        assert captured_guardrail_store[3] == 13

    def test_run_local_retrievers_graph_regex_stage_iterates_filters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        def _fake_graph_regex_stage_filters(
            mySession: LocalSessionStub,
            file_filter: dict[str, Any] | None,
        ) -> list[dict[str, Any] | None]:
            _ = (mySession, file_filter)
            return [
                {"FileName": "A.txt"},
                {"FileName": "B.txt"},
            ]

        monkeypatch.setattr(
            orchestrator,
            "_graph_regex_stage_filters",
            _fake_graph_regex_stage_filters,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b1"]
        assert graph_docs == ["g1", "g1"]
        assert regex_docs == ["r1", "r1"]
        assert host.indexed_stage_labels == [
            ["BM25"],
            ["BM25"],
            ["Graph"],
            ["Regex"],
            ["Graph"],
            ["Regex"],
        ]
        assert host.run_indexed_file_filters == [
            {"FileName": "A.txt"},
            {"FileName": "B.txt"},
            {"FileName": "A.txt"},
            {"FileName": "A.txt"},
            {"FileName": "B.txt"},
            {"FileName": "B.txt"},
        ]

    def test_run_local_retrievers_graph_regex_stage_uses_language_bucket_planner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        def _fake_plan_graph_regex_language_buckets(
            mySession: LocalSessionStub,
            file_filter: dict[str, Any] | None,
        ) -> list[str | None]:
            _ = (mySession, file_filter)
            return ["en", "de"]

        def _fake_build_graph_regex_stage_filter(
            base_file_filter: dict[str, Any] | None,
            language_bucket: str | None,
        ) -> dict[str, Any] | None:
            staged_filter = dict(base_file_filter or {})
            if language_bucket is not None:
                staged_filter["Language"] = language_bucket
            return staged_filter or None

        monkeypatch.setattr(
            orchestrator,
            "_plan_graph_regex_language_buckets",
            _fake_plan_graph_regex_language_buckets,
        )
        monkeypatch.setattr(
            orchestrator,
            "_build_graph_regex_stage_filter",
            _fake_build_graph_regex_stage_filter,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b1"]
        assert graph_docs == ["g1", "g1"]
        assert regex_docs == ["r1", "r1"]
        assert host.indexed_stage_labels == [
            ["BM25"],
            ["BM25"],
            ["Graph"],
            ["Regex"],
            ["Graph"],
            ["Regex"],
        ]
        assert host.run_indexed_file_filters == [
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "de"},
        ]

    def test_graph_regex_stage_iterations_plans_filters_and_shaped_queries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        stage_filters = [
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "en"},
        ]

        def _fake_graph_regex_stage_filters(
            mySession: LocalSessionStub,
            file_filter: dict[str, Any] | None,
        ) -> list[dict[str, Any] | None]:
            _ = (mySession, file_filter)
            return list(stage_filters)

        def _fake_shape_graph_regex_stage_queries(
            mySession: LocalSessionStub,
            *,
            language_bucket: str | None,
            primary_query: str,
            guardrail_query: str,
        ) -> tuple[str, str]:
            _ = mySession
            suffix = language_bucket or "none"
            return (
                f"{primary_query}|{suffix}",
                f"{guardrail_query}|{suffix}",
            )

        monkeypatch.setattr(
            orchestrator,
            "_graph_regex_stage_filters",
            _fake_graph_regex_stage_filters,
        )
        monkeypatch.setattr(
            orchestrator,
            "_shape_graph_regex_stage_queries",
            _fake_shape_graph_regex_stage_queries,
        )

        iterations = orchestrator._graph_regex_stage_iterations(
            session,
            file_filter={"FileName": "A.txt"},
            primary_query="Q2",
            guardrail_query="Q1",
        )

        assert [it.language_bucket for it in iterations] == ["de", "en"]
        assert [it.file_filter for it in iterations] == stage_filters
        assert [it.primary_query for it in iterations] == ["Q2|de", "Q2|en"]
        assert [it.guardrail_query for it in iterations] == ["Q1|de", "Q1|en"]

    def test_run_local_retrievers_graph_regex_stage_shapes_queries_per_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        def _fake_plan_graph_regex_language_buckets(
            mySession: LocalSessionStub,
            file_filter: dict[str, Any] | None,
        ) -> list[str | None]:
            _ = (mySession, file_filter)
            return ["en", "de"]

        def _fake_shape_graph_regex_stage_queries(
            mySession: LocalSessionStub,
            *,
            language_bucket: str | None,
            primary_query: str,
            guardrail_query: str,
        ) -> tuple[str, str]:
            _ = mySession
            suffix = language_bucket or "none"
            return (
                f"{primary_query}|{suffix}",
                f"{guardrail_query}|{suffix}",
            )

        monkeypatch.setattr(
            orchestrator,
            "_plan_graph_regex_language_buckets",
            _fake_plan_graph_regex_language_buckets,
        )
        monkeypatch.setattr(
            orchestrator,
            "_shape_graph_regex_stage_queries",
            _fake_shape_graph_regex_stage_queries,
        )

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b2", "b1", "b2"]
        assert graph_docs == ["g1", "g1"]
        assert regex_docs == ["r1", "r1"]
        assert host.indexed_stage_labels == [
            ["BM25"],
            ["BM25"],
            ["Graph"],
            ["Regex"],
            ["Graph"],
            ["Regex"],
        ]
        assert host.run_indexed_file_filters == [
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "de"},
        ]
        assert host.run_indexed_primary_queries == [
            "Q2|en",
            "Q2|de",
            "Q2|en",
            "Q2|en",
            "Q2|de",
            "Q2|de",
        ]
        assert host.run_indexed_guardrail_queries == [
            "Q1|en",
            "Q1|de",
            "Q1|en",
            "Q1|en",
            "Q1|de",
            "Q1|de",
        ]

    def test_query_dispatch_trace_includes_language_bucket(self) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(False, True, True, True),
        )
        host.collection = _LocalCollectionStub(  # type: ignore[attr-defined]
            [{"Language": "en"}, {"Language": "de"}]
        )
        session = LocalSessionStub()
        session.debug_level = 28
        session.current_query_lang = "german"
        session.user_language = "german"
        session.retrieval_language = "english"

        orchestrator = RetrievalOrchestrator(host)
        _ = orchestrator.run_local_retrievers(
            session,
            retrieve_mode="ALL",
            bm25_query="Q2",
            alternate_queries=[],
            orig_translated_query_en="Q1",
        )

        trace_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]

        assert any(
            "retrievers='BM25/Graph/Regex'" in message
            and "language: de" in message
            and "query: 'Q2'" in message
            and "guardrail_query='Q1'" in message
            for message in trace_messages
        )
        assert any(
            "retrievers='BM25/Graph/Regex'" in message
            and "language: en" in message
            and "query: 'Q2'" in message
            and "guardrail_query='Q1'" in message
            for message in trace_messages
        )

    def test_vector_dispatch_trace_omits_language_field(self) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(True, False, False, False),
        )
        session = LocalSessionStub()
        session.debug_level = 28
        session.current_query_lang = "german"
        session.user_language = "german"
        session.retrieval_language = "english"

        orchestrator = RetrievalOrchestrator(host)
        _ = orchestrator.run_local_retrievers(
            session,
            retrieve_mode="ALL",
            bm25_query="Q2",
            alternate_queries=["ALT1", "ALT2"],
            orig_translated_query_en="Q1",
        )

        trace_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]
        multiquery_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "MultiQuery"
        ]

        assert any(
            "retriever: Vector" in message
            and "dispatch query" in message
            and "query: 'Q2'" in message
            and "language:" not in message
            for message in trace_messages
        )
        assert any(
            "query language flow: user=de current=de rewrite=unknown retrieval=en"
            in message
            for message in trace_messages
        )
        assert any(
            "2 Alternate queries for vector retrieval:" in message
            and "\n1: 'ALT1'" in message
            and "\n2: 'ALT2'" in message
            for message in multiquery_messages
        )

    def test_shape_graph_regex_stage_queries_translates_non_english_with_argos(
        self,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        host._translation_backend = "argos"  # type: ignore[attr-defined]
        shared = _LocalTranslatingSharedStub()
        host._shared = shared  # type: ignore[attr-defined]
        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        stage_primary_query, stage_guardrail_query = (
            orchestrator._shape_graph_regex_stage_queries(
                session,
                language_bucket="de",
                primary_query="Q2",
                guardrail_query="Q1",
            )
        )

        assert stage_primary_query == "Q2|de"
        assert stage_guardrail_query == "Q1|de"
        assert shared.calls == [
            ("Q2", "de", "en"),
            ("Q1", "de", "en"),
        ]

    def test_shape_graph_regex_stage_queries_logs_original_and_translated_queries(
        self,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        host._translation_backend = "argos"  # type: ignore[attr-defined]
        shared = _LocalTranslatingSharedStub()
        host._shared = shared  # type: ignore[attr-defined]
        session = LocalSessionStub()
        session.debug_level = 0
        orchestrator = RetrievalOrchestrator(host)

        stage_primary_query, stage_guardrail_query = (
            orchestrator._shape_graph_regex_stage_queries(
                session,
                language_bucket="de",
                primary_query="Q2",
                guardrail_query="Q1",
            )
        )

        assert stage_primary_query == "Q2|de"
        assert stage_guardrail_query == "Q1|de"
        orchestration_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]
        assert any(
            "idx_stage l='de'" in message
            and "r='B/G/R'" in message
            and "tr=1" in message
            and "\nprimary_query='Q2'" in message
            and "\nstage_primary_query='Q2|de'" in message
            and "\nguardrail_query='Q1'" in message
            and "\nstage_guardrail_query='Q1|de'" in message
            for message in orchestration_messages
        )

    def test_run_local_retrievers_graph_regex_stage_translates_queries_with_argos(
        self,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(False, True, True, True),
        )
        host._translation_backend = "argos"  # type: ignore[attr-defined]
        shared = _LocalTranslatingSharedStub()
        host._shared = shared  # type: ignore[attr-defined]
        host.collection = _LocalCollectionStub(  # type: ignore[attr-defined]
            [{"Language": "en"}, {"Language": "de"}]
        )

        session = LocalSessionStub()
        session.current_query_lang = "german"
        session.user_language = "german"
        session.retrieval_language = "english"
        orchestrator = RetrievalOrchestrator(host)

        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q1",
            )
        )

        assert vector_docs == []
        assert bm25_docs == ["b1", "b2", "b1", "b2"]
        assert graph_docs == ["g1", "g1"]
        assert regex_docs == ["r1", "r1"]
        assert host.indexed_stage_labels == [
            ["BM25"],
            ["BM25"],
            ["Graph"],
            ["Regex"],
            ["Graph"],
            ["Regex"],
        ]
        assert host.run_indexed_file_filters == [
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "de"},
            {"FileName": "A.txt", "Language": "en"},
            {"FileName": "A.txt", "Language": "en"},
        ]
        assert host.run_indexed_primary_queries == [
            "Q2|de",
            "Q2",
            "Q2|de",
            "Q2|de",
            "Q2",
            "Q2",
        ]
        assert host.run_indexed_guardrail_queries == [
            "Q1|de",
            "Q1",
            "Q1|de",
            "Q1|de",
            "Q1",
            "Q1",
        ]
        assert shared.calls == [
            ("Q2", "de", "en"),
            ("Q1", "de", "en"),
        ]

    def test_normalize_language_bucket_prefers_host_seam(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        orchestrator = RetrievalOrchestrator(host)

        calls: list[Any] = []

        def _fake_normalize_language_bucket(language: Any) -> str | None:
            calls.append(language)
            return "zz"

        host._normalize_language_bucket = _fake_normalize_language_bucket  # type: ignore[attr-defined]

        normalized = orchestrator._normalize_language_bucket("german")

        assert normalized == "zz"
        assert calls == ["german"]

    def test_discover_graph_regex_languages_normalizes_from_collection(self) -> None:
        class _CollectionStub:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def get(self, **kwargs: Any) -> dict[str, Any]:
                self.calls.append(dict(kwargs))
                return {
                    "metadatas": [
                        {"Language": "english"},
                        {"Language": "de"},
                        {"Language": "DE"},
                        {"Language": ""},
                        {},
                        None,
                    ]
                }

        class _SharedStub:
            def __init__(self) -> None:
                self.lang_name_to_code: dict[str, str] = {
                    "english": "en",
                    "german": "de",
                }

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[attr-defined]
        host._shared = _SharedStub()  # type: ignore[attr-defined]
        orchestrator = RetrievalOrchestrator(host)

        languages = orchestrator._discover_graph_regex_languages(
            {"FileName": "A.txt", "Language": "en"}
        )

        assert languages == ["de", "en"]
        assert host.collection.calls == [  # type: ignore[attr-defined]
            {
                "include": ["metadatas"],
                "where": {"FileName": "A.txt"},
                "limit": 5000,
            }
        ]

    def test_plan_graph_regex_language_buckets_honors_explicit_language_filter(
        self,
    ) -> None:
        class _SharedStub:
            def __init__(self) -> None:
                self.lang_name_to_code: dict[str, str] = {
                    "english": "en",
                    "german": "de",
                }

        host = LocalHostStub(use_guardrail=False)
        host._shared = _SharedStub()  # type: ignore[attr-defined]
        host.collection = _LocalCollectionStub(  # type: ignore[attr-defined]
            [{"Language": "en"}, {"Language": "de"}]
        )
        session = LocalSessionStub()
        session.current_query_lang = "german"
        orchestrator = RetrievalOrchestrator(host)

        buckets_plain = orchestrator._plan_graph_regex_language_buckets(
            session,
            {"Language": "german", "FileName": "A.txt"},
        )
        buckets_eq = orchestrator._plan_graph_regex_language_buckets(
            session,
            {"Language": {"$eq": "english"}},
        )

        assert buckets_plain == ["de"]
        assert buckets_eq == ["en"]

    def test_plan_graph_regex_language_buckets_intersects_active_and_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        session.current_query_lang = "german"
        orchestrator = RetrievalOrchestrator(host)

        def _fake_discover_graph_regex_languages(
            file_filter: dict[str, Any] | None,
        ) -> list[str]:
            _ = file_filter
            return ["de", "en", "fr"]

        monkeypatch.setattr(
            orchestrator,
            "_discover_graph_regex_languages",
            _fake_discover_graph_regex_languages,
        )

        buckets = orchestrator._plan_graph_regex_language_buckets(
            session,
            {"FileName": "A.txt"},
        )

        assert buckets == ["de", "en"]

    def test_plan_graph_regex_language_buckets_keeps_non_query_languages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        session.current_query_lang = "english"
        session.user_language = "english"
        session.retrieval_language = "english"
        orchestrator = RetrievalOrchestrator(host)

        def _fake_discover_graph_regex_languages(
            file_filter: dict[str, Any] | None,
        ) -> list[str]:
            _ = file_filter
            return ["de", "en", "fr"]

        monkeypatch.setattr(
            orchestrator,
            "_discover_graph_regex_languages",
            _fake_discover_graph_regex_languages,
        )

        buckets = orchestrator._plan_graph_regex_language_buckets(
            session,
            {"FileName": "A.txt"},
        )

        assert buckets == ["en", "de"]

    def test_plan_graph_regex_language_buckets_skips_explicit_not_allowed(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        session.current_query_lang = "french"
        orchestrator = RetrievalOrchestrator(host)

        buckets = orchestrator._plan_graph_regex_language_buckets(
            session,
            {"Language": "fr", "FileName": "A.txt"},
        )

        assert buckets == []

    def test_emit_vector_store_language_status_emits_orchestrator_message_when_all_active(
        self,
    ) -> None:
        class _CollectionStub:
            def get(self, **kwargs: Any) -> dict[str, Any]:
                _ = kwargs
                return {
                    "metadatas": [
                        {"Language": "english"},
                        {"Language": "de"},
                    ]
                }

        class _CfgStub:
            def get_list(self, key: str, default: Any, silent: bool = False) -> Any:
                _ = (default, silent)
                if key == "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES":
                    return ["en", "de"]
                if key == "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES":
                    return [["en", "de"]]
                return []

        class _SharedStub:
            def __init__(self) -> None:
                self.lang_name_to_code: dict[str, str] = {
                    "english": "en",
                    "german": "de",
                }

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[attr-defined]
        host.cfg = _CfgStub()  # type: ignore[attr-defined]
        host.pretty = StubPretty()  # type: ignore[attr-defined]
        host._shared = _SharedStub()  # type: ignore[attr-defined]
        host.collection_name = "TestCollection"  # type: ignore[attr-defined]

        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        orchestrator._emit_vector_store_language_status(session)

        assert any(
            len(call[0]) >= 3
            and call[0][0] == "I"
            and call[0][1] == "Retrieval Orchestration"
            and "discovered corpus language buckets" in str(call[0][2])
            and "[de, en]" in str(call[0][2])
            for call in host.pretty.calls  # type: ignore[attr-defined]
        )
        assert not any(
            len(call[0]) >= 2 and call[0][1] == "LanguageConfig"
            for call in host.pretty.calls  # type: ignore[attr-defined]
        )

    def test_emit_vector_store_language_status_warns_on_undefined_languages(
        self,
    ) -> None:
        class _CollectionStub:
            def get(self, **kwargs: Any) -> dict[str, Any]:
                _ = kwargs
                return {
                    "metadatas": [
                        {"Language": "en"},
                        {"Language": "fr"},
                    ]
                }

        class _CfgStub:
            def get_list(self, key: str, default: Any, silent: bool = False) -> Any:
                _ = (default, silent)
                if key == "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES":
                    return ["en", "de"]
                if key == "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES":
                    return [["en", "de"]]
                return []

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[attr-defined]
        host.cfg = _CfgStub()  # type: ignore[attr-defined]
        host.pretty = StubPretty()  # type: ignore[attr-defined]
        host.collection_name = "TestCollection"  # type: ignore[attr-defined]

        session = LocalSessionStub()
        orchestrator = RetrievalOrchestrator(host)

        orchestrator._emit_vector_store_language_status(session)

        assert any(
            len(call[0]) >= 3
            and call[0][0] == "W"
            and call[0][1] == "LanguageConfig"
            and "not active in _ARGOS_DEFINITIONS.ACTIVE_LANGUAGES" in str(call[0][2])
            and "fr" in str(call[0][2])
            for call in host.pretty.calls  # type: ignore[attr-defined]
        )


class TestWebRetrievalOrchestration:
    def test_run_web_retriever_returns_empty_when_not_triggered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_MODE", "1")
        host = WebHostStub([WebDocStub("https://a")], cfg=WebCfgStub())
        session = WebSessionStub(web_search=False)
        orchestrator = RetrievalOrchestrator(host)

        docs = orchestrator.run_web_retriever(
            session,
            retrieve_mode="BM25",
            user_query_original="original",
        )

        assert docs == []
        assert host.web_retriever.calls == []

    def test_run_web_retriever_blocks_when_mode_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_MODE", "0")
        host = WebHostStub([WebDocStub("https://a")], cfg=WebCfgStub())
        session = WebSessionStub(web_search=True)
        orchestrator = RetrievalOrchestrator(host)

        docs = orchestrator.run_web_retriever(
            session,
            retrieve_mode="WEB",
            user_query_original="original",
        )

        assert docs == []
        assert host.web_retriever.calls == []
        assert any(
            'WEB_SEARCH_MODE="0"' in call[0][2]
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Web"
        )

    def test_run_web_retriever_applies_prefilters_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_MODE", "1")
        d1 = WebDocStub("https://a")
        d2 = WebDocStub("https://b")
        d3 = WebDocStub("https://c")
        host = WebHostStub(
            [d1, d2, d3],
            cfg=WebCfgStub(max_results=7, bm25_pre_filter=0.2, cosine_pre_filter=0.3),
            bm25_out=[d1, d2],
            cosine_out=[d2],
        )
        session = WebSessionStub(web_search=True, query="hedgehog query")
        orchestrator = RetrievalOrchestrator(host)

        docs = orchestrator.run_web_retriever(
            session,
            retrieve_mode="ALL",
            user_query_original="original",
        )

        assert docs == [d2]
        assert host.web_retriever.calls == [
            ("hedgehog query", 7, False, "original", "Test")
        ]
        assert len(host.web_pre_filter.bm25_calls) == 1
        assert len(host.web_pre_filter.cosine_calls) == 1
        assert host.embedder.calls == ["hedgehog query"]
        assert len(host.perf_logger.calls) == 2
