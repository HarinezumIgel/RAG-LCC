# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""Tests for Orchestrator.Orchestrator.

These tests verify orchestration sequencing, early-exit branches, and debug
trace emission at level 28.
"""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Orchestrator.Orchestrator import Orchestrator
from Orchestrator.OrchestratorTypes import GuardrailInputs
from Commons.Exceptions import ConfigurationError, QueryTranslationError
from Gui.Colors import BRIGHT_GREEN, CYAN


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
        self.web_search: bool = False
        self.query: str = "Q2"
        self.orig_translated_query_en: str | None = "Q1"
        self.seed_retrieval_query: str | None = None
        self.t1_query: str | None = None
        self.enable_keyword_action_splitter: bool | None = False
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
        self.user_query_original: str | None = None
        self.debug_level: int = 0
        self.debug_mode: str = "ge"

        self.retrieval_top_k_orig_query_en: int | None = None
        self.retrieval_top_k_post_rewrite_query_en: int | None = None
        self.retrieval_top_k_seed_query: int | None = None
        self.retrieval_top_k_final_query: int | None = None
        self.retrieval_top_k_before_t2: int | None = None
        self.retrieval_top_k_after_t2: int | None = None

        self.original_query_leg_enabled: bool = False
        self.original_query_leg_query: str | None = None
        self.original_query_leg_language: str | None = None
        self.original_query_leg_reason: str | None = None
        self.original_query_leg_vector_hits: int | None = None
        self.original_query_leg_vector_added: int | None = None
        self.original_query_leg_vector_overlap: int | None = None
        self.original_query_leg_bm25_hits: int | None = None
        self.original_query_leg_bm25_added: int | None = None
        self.original_query_leg_bm25_overlap: int | None = None
        self.original_query_leg_graph_hits: int | None = None
        self.original_query_leg_graph_added: int | None = None
        self.original_query_leg_graph_overlap: int | None = None
        self.original_query_leg_regex_hits: int | None = None
        self.original_query_leg_regex_added: int | None = None
        self.original_query_leg_regex_overlap: int | None = None


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


class RetrievalDocStub:
    def __init__(self, doc_id: str) -> None:
        self.page_content = doc_id
        self.metadata: dict[str, Any] = {
            "id": doc_id,
            "retriever_sources": "Vector",
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

    def _normalize_language_bucket(self, language: Any) -> str | None:
        return str(language or "").strip().lower() or None

    @staticmethod
    def _mode_flags_for_local_retrievers(
        retrieve_mode: str,
    ) -> tuple[bool, bool, bool, bool]:
        mode = str(retrieve_mode or "").strip().upper()
        if mode == "VECTOR":
            return (True, False, False, False)
        if mode == "BM25":
            return (False, True, False, False)
        if mode == "GRAPH":
            return (False, False, True, False)
        if mode == "REGEX":
            return (False, False, False, True)
        if mode == "VECTOR_BM25":
            return (True, True, False, False)
        if mode == "VECTOR_GRAPH":
            return (True, False, True, False)
        if mode == "BM25_GRAPH":
            return (False, True, True, False)
        if mode == "VECTOR_REGEX":
            return (True, False, False, True)
        if mode == "BM25_REGEX":
            return (False, True, False, True)
        if mode == "GRAPH_REGEX":
            return (False, False, True, True)
        if mode == "WEB":
            return (False, False, False, False)
        return (True, True, True, True)

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
        indexed_settings: dict[str, Any] | None = None,
    ) -> None:
        self.pairs = pairs or [["en", "de"]]
        self.active_languages = active_languages or ["en", "de"]
        self.indexed_settings = indexed_settings or {}

    def get_dict(self, key: str, default: Any, silent: bool = False) -> Any:
        return dict(self.indexed_settings) if key == "_INDEXED_QUERY" else default

    def get_list(self, key: str, default: Any, silent: bool = False) -> Any:
        _ = (default, silent)
        if key == "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES":
            return list(self.active_languages)
        if key == "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES":
            return list(self.pairs)
        return []

    indirect_list = get_list
    indirect_dict = get_dict


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
        enable_translation: bool = True,
        enable_synonym_expansion: bool = False,
    ) -> tuple[str, str]:
        _ = mySession
        target_language = self._normalize_language_bucket(language_bucket)
        if not target_language or not enable_translation:
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


class TestOrchestrator:
    def test_run_happy_path_sequences_stages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        session.web_search = True

        orchestrator = Orchestrator(host)
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
            "_merge_and_select",
            "_build_context",
        ]
        assert captured_local_args == ("ALL", "Q2", ["ALT1", "ALT2"], "Q1")
        assert captured_guardrail_use is True
        assert captured_web_args == ("ALL", "USER_ORIG")
        assert host.merge_args == (["v1"], ["b1", "b2"], ["g1"], ["r1"], ["w1"])
        assert len(host.perf_logger.calls) == 1

    def test_run_applies_flow_rewrite_and_pronoun_switches_before_normalize(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _CfgStub:
            def get_dict(self, key: str, default: Any = None) -> dict[str, Any]:
                if key == "_ORCHESTRATION_FLOWS":
                    return {
                        "FLOW_OFF": {
                            "use_query_rewrite": False,
                            "use_pronoun_substitution": False,
                            "run_low_score_fallback": False,
                            "run_low_recall_rescue": False,
                        }
                    }
                return default if isinstance(default, dict) else {}

            def get_list(self, key: str, default: Any = None) -> list[Any]:
                if key == "_ALLOWED_ORCHESTRATION_FLOWS":
                    return ["FLOW_OFF"]
                return default if isinstance(default, list) else []

            def get_str(self, key: str, default: str = "") -> str:
                if key == "_ACTIVE_ORCHESTRATION_FLOW":
                    return "FLOW_OFF"
                return default

        class _FlowControlHost(HostStub):
            def __init__(self) -> None:
                super().__init__(set_vector_ok=True, gate_result=False)
                self.cfg = _CfgStub()
                self.observed_rewrite_flag: bool | None = None
                self.observed_pronoun_flag: bool | None = None
                self.observed_low_score_fallback_flag: bool | None = None
                self.observed_low_recall_rescue_flag: bool | None = None

            def _normalize_query(
                self,
                session: StubSession,
                user_query_original: str,
            ) -> tuple[str, list[str]]:
                _ = user_query_original
                self.calls.append("_normalize_query")
                self.observed_rewrite_flag = bool(
                    getattr(session, "enable_query_rewrite", None)
                )
                self.observed_pronoun_flag = bool(
                    getattr(session, "enable_pronoun_substitution", None)
                )
                self.observed_low_score_fallback_flag = bool(
                    getattr(session, "enable_low_score_fallback", None)
                )
                self.observed_low_recall_rescue_flag = bool(
                    getattr(session, "enable_low_recall_rescue", None)
                )
                return "Q2", ["ALT1", "ALT2"]

        host = _FlowControlHost()
        session = StubSession()
        orchestrator = Orchestrator(host)

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            _ = (mySession, pipeline_inputs)
            return "CTX::FLOW", 1

        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("CTX::FLOW", 1)
        assert host.observed_rewrite_flag is False
        assert host.observed_pronoun_flag is False
        assert host.observed_low_score_fallback_flag is False
        assert host.observed_low_recall_rescue_flag is False
        assert bool(getattr(session, "enable_query_rewrite", True)) is False
        assert bool(getattr(session, "enable_pronoun_substitution", True)) is False
        assert bool(getattr(session, "enable_low_score_fallback", True)) is False
        assert bool(getattr(session, "enable_low_recall_rescue", True)) is False

    def test_pre_gate_disabled_stages_provide_explicit_empty_inputs(self) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        session.user_query_original = "Wie leben Igel?"
        session.user_language = "german"
        orchestrator = Orchestrator(host)
        profile = orchestrator.orchestratorOverrides.resolve_orchestration_flow_profile(
            session
        )
        profile = replace(
            profile,
            enable_keyword_action_splitter=False,
            enable_query_rewrite=False,
            enable_pronoun_substitution=False,
            enable_guardrail_leg=False,
            enable_original_language_vector_leg=False,
            enable_vector_alternate_queries=False,
            enable_indexed_query_shaping=False,
            enable_indexed_query_translation=False,
            enable_indexed_query_wordnet_expansion=False,
        )

        result = orchestrator._run_pre_gate_pipeline(session, flow_profile=profile)

        assert result.should_abort is False
        assert result.post_gate_inputs is not None
        assert result.post_gate_inputs.resolved_guardrail == GuardrailInputs(
            primary_query="Q2",
            guardrail_query="",
            use_guardrail=False,
        )
        assert result.post_gate_inputs.alternate_queries == []
        assert result.post_gate_inputs.enable_original_language_vector_leg is False
        assert result.post_gate_inputs.enable_indexed_query_shaping is False
        assert result.post_gate_inputs.enable_indexed_query_translation is False
        assert result.post_gate_inputs.enable_indexed_query_wordnet_expansion is False
        assert session.enable_query_rewrite is False
        assert session.enable_pronoun_substitution is False

    def test_run_emits_yellow_flow_stage_knob_status_lines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _CfgStub:
            def get_dict(self, key: str, default: Any = None) -> dict[str, Any]:
                if key == "_ORCHESTRATION_FLOWS":
                    return {
                        "FLOW_MIXED": {
                            "use_query_rewrite": True,
                            "use_pronoun_substitution": False,
                            "use_secondary_query": False,
                            "use_original_language_vector": True,
                            "shape_indexed_queries": True,
                            "use_vector_alternates": False,
                            "run_vector": True,
                            "run_bm25": False,
                            "run_graph": True,
                            "run_regex": False,
                            "run_low_score_fallback": False,
                            "run_low_recall_rescue": False,
                        }
                    }
                return default if isinstance(default, dict) else {}

            def get_list(self, key: str, default: Any = None) -> list[Any]:
                if key == "_ALLOWED_ORCHESTRATION_FLOWS":
                    return ["FLOW_MIXED"]
                return default if isinstance(default, list) else []

            def get_str(self, key: str, default: str = "") -> str:
                if key == "_ACTIVE_ORCHESTRATION_FLOW":
                    return "FLOW_MIXED"
                return default

        class _FlowMessageHost(HostStub):
            def __init__(self) -> None:
                super().__init__(set_vector_ok=True, gate_result=False)
                self.cfg = _CfgStub()

        host = _FlowMessageHost()
        session = StubSession()
        orchestrator = Orchestrator(host)

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            _ = (mySession, pipeline_inputs)
            return "CTX::FLOW", 1

        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)
        assert (context, count) == ("CTX::FLOW", 1)

        knob_messages = [
            (str(call[0][2]), call[1].get("color"))
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]

        assert any(
            "Query rewrite stage" in message
            and "Activated" in message
            and "knob=use_query_rewrite" in message
            and color == CYAN
            for message, color in knob_messages
        )
        assert any(
            "Pronoun substitution stage" in message
            and "Not activated" in message
            and "knob=use_pronoun_substitution" in message
            and color == CYAN
            for message, color in knob_messages
        )
        assert any(
            "BM25 retriever stage" in message
            and "Not activated" in message
            and "knob=run_bm25" in message
            and color == CYAN
            for message, color in knob_messages
        )
        assert any(
            "Low-score fallback action" in message
            and "Not activated" in message
            and "knob=run_low_score_fallback" in message
            and color == CYAN
            for message, color in knob_messages
        )
        assert any(
            "Low-recall rescue action" in message
            and "Not activated" in message
            and "knob=run_low_recall_rescue" in message
            and color == CYAN
            for message, color in knob_messages
        )

    def test_resolve_flow_profile_reads_indexed_translation_and_wordnet_flags(
        self,
    ) -> None:
        class _CfgStub:
            def get_dict(self, key: str, default: Any = None) -> dict[str, Any]:
                if key == "_ORCHESTRATION_FLOWS":
                    return {
                        "FLOW_FLAGS": {
                            "translate_indexed_queries": False,
                            "expand_indexed_queries_with_wordnet": False,
                            "confidence_evaluation": "llm",
                        }
                    }
                return default if isinstance(default, dict) else {}

            def get_list(self, key: str, default: Any = None) -> list[Any]:
                if key == "_ALLOWED_ORCHESTRATION_FLOWS":
                    return ["FLOW_FLAGS"]
                return default if isinstance(default, list) else []

            def get_str(self, key: str, default: str = "") -> str:
                if key == "_ACTIVE_ORCHESTRATION_FLOW":
                    return "FLOW_FLAGS"
                return default

        class _FlowHost(HostStub):
            def __init__(self) -> None:
                super().__init__(set_vector_ok=True, gate_result=False)
                self.cfg = _CfgStub()

        host = _FlowHost()
        session = StubSession()
        orchestrator = Orchestrator(host)

        profile = orchestrator.orchestratorOverrides.resolve_orchestration_flow_profile(
            session
        )

        assert profile.name == "FLOW_FLAGS"
        assert profile.enable_indexed_query_translation is False
        assert profile.enable_indexed_query_wordnet_expansion is False
        assert profile.confidence_evaluation == "llm"
        session.confidence_evaluation = "off"
        effective = orchestrator.orchestratorOverrides.resolve_effective_flow_profile_for_session(
            session, profile
        )
        assert effective.confidence_evaluation == "off"
        assert profile.confidence_evaluation == "llm"

    def test_run_disables_low_recall_rescue_when_rerank_cli_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _CfgStub:
            def get_dict(self, key: str, default: Any = None) -> dict[str, Any]:
                if key == "_ORCHESTRATION_FLOWS":
                    return {
                        "FLOW_ON": {
                            "run_rerank": True,
                            "run_low_score_fallback": True,
                            "run_low_recall_rescue": True,
                        }
                    }
                return default if isinstance(default, dict) else {}

            def get_list(self, key: str, default: Any = None) -> list[Any]:
                if key == "_ALLOWED_ORCHESTRATION_FLOWS":
                    return ["FLOW_ON"]
                return default if isinstance(default, list) else []

            def get_str(self, key: str, default: str = "") -> str:
                if key == "_ACTIVE_ORCHESTRATION_FLOW":
                    return "FLOW_ON"
                return default

        class _FlowToggleHost(HostStub):
            def __init__(self) -> None:
                super().__init__(set_vector_ok=True, gate_result=False)
                self.cfg = _CfgStub()
                self.observed_rerank_flag: bool | None = None
                self.observed_low_score_fallback_flag: bool | None = None
                self.observed_rescue_flag: bool | None = None

            def _normalize_query(
                self,
                session: StubSession,
                user_query_original: str,
            ) -> tuple[str, list[str]]:
                _ = user_query_original
                self.calls.append("_normalize_query")
                self.observed_rerank_flag = bool(
                    getattr(session, "enable_rerank", None)
                )
                self.observed_low_score_fallback_flag = bool(
                    getattr(session, "enable_low_score_fallback", None)
                )
                self.observed_rescue_flag = bool(
                    getattr(session, "enable_low_recall_rescue", None)
                )
                return "Q2", ["ALT1", "ALT2"]

        host = _FlowToggleHost()
        session = StubSession()
        session.rerank = 0
        orchestrator = Orchestrator(host)

        def _fake_run_post_gate_pipeline(
            mySession: StubSession,
            *,
            pipeline_inputs: Any,
        ) -> tuple[str, int]:
            _ = (mySession, pipeline_inputs)
            return "CTX::FLOW", 1

        monkeypatch.setattr(
            orchestrator,
            "_run_post_gate_pipeline",
            _fake_run_post_gate_pipeline,
        )

        context, count = orchestrator.run(session)

        assert (context, count) == ("CTX::FLOW", 1)
        assert host.observed_rerank_flag is True
        assert host.observed_low_score_fallback_flag is False
        assert host.observed_rescue_flag is False
        assert bool(getattr(session, "enable_rerank", False)) is True
        assert bool(getattr(session, "enable_low_score_fallback", True)) is False
        assert bool(getattr(session, "enable_low_recall_rescue", True)) is False

    def test_run_local_only_skips_web_retrieval_stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        session.web_search = False

        orchestrator = Orchestrator(host)

        web_stage_called = False

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
                    "vector_docs": ["v1"],
                    "bm25_docs": ["b1"],
                    "graph_docs": [],
                    "regex_docs": [],
                },
            )()

        def _fake_run_web_retriever(
            my_session: StubSession,
            retrieve_mode: str,
            user_query_original: str,
        ) -> list[Any]:
            nonlocal web_stage_called
            _ = (my_session, retrieve_mode, user_query_original)
            web_stage_called = True
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

        assert web_stage_called is False
        assert (context, count) == ("CTX::chosen1,chosen2", 2)

        trace_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3 and call[0][1] == "Retrieval Orchestration"
        ]
        assert any(
            "effective stage gates:" in message and "web_mode=local_only" in message
            for message in trace_messages
        )
        assert not any(
            "retrieve web candidates" in message for message in trace_messages
        )
        assert any(
            "web stage skipped " in message and "web_mode=local_only" in message
            for message in trace_messages
        )

    def test_run_routes_through_post_gate_pipeline_seam(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = HostStub(set_vector_ok=True, gate_result=False)
        session = StubSession()
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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

        def _fake_run_pre_gate_pipeline(
            mySession: StubSession,
            *,
            flow_profile: Any | None = None,
        ) -> Any:
            nonlocal captured_pre_gate_session
            _ = flow_profile
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
        orchestrator = Orchestrator(host)

        post_gate_called = False

        class _PreGateAbort:
            def __init__(self) -> None:
                self.should_abort = True
                self.post_gate_inputs = None

        def _fake_run_pre_gate_pipeline(
            mySession: StubSession,
            *,
            flow_profile: Any | None = None,
        ) -> Any:
            _ = mySession
            _ = flow_profile
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

        orchestrator = Orchestrator(host)
        context, count = orchestrator.run(session)

        assert (context, count) == ("", 0)
        assert host.calls == ["_set_vector_store"]
        assert len(host.perf_logger.calls) == 0

    def test_run_early_exit_when_gate_blocks(self) -> None:
        host = HostStub(set_vector_ok=True, gate_result=True)
        session = StubSession()

        orchestrator = Orchestrator(host)
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

        orchestrator = Orchestrator(host)

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
        gate_trace_calls = [
            call
            for call in host.pretty.calls
            if len(call[0]) >= 3
            and call[0][1] == "Retrieval Orchestration"
            and "effective stage gates:" in str(call[0][2])
        ]
        assert gate_trace_calls
        assert gate_trace_calls[0][1].get("color") == BRIGHT_GREEN
        assert "web_mode=web_only" in str(gate_trace_calls[0][0][2])
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

        orchestrator = Orchestrator(host)

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

        orchestrator = Orchestrator(host)

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
    @pytest.mark.parametrize(
        ("flow_flag", "docs_field"),
        [
            ("_flow_enable_vector_retriever", "vector_docs"),
            ("_flow_enable_bm25_retriever", "bm25_docs"),
            ("_flow_enable_graph_retriever", "graph_docs"),
            ("_flow_enable_regex_retriever", "regex_docs"),
        ],
    )
    def test_disabled_flow_retriever_contributes_no_docs(
        self,
        flow_flag: str,
        docs_field: str,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        orchestrator = Orchestrator(host)
        setattr(orchestrator, flow_flag, False)

        local_docs = orchestrator._run_local_docs_stage(
            session,
            retrieve_mode="ALL",
            bm25_query="question",
            alternate_queries=[],
            orig_translated_query_en="question",
            resolved_guardrail=GuardrailInputs("question", "", False),
        )

        assert getattr(local_docs, docs_field) == []

    def test_all_flow_retrievers_disabled_returns_no_results_without_answer_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from Chat.Chatter import Chatter

        monkeypatch.setenv("WEB_SEARCH_MODE", "0")
        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)
        for flow_flag in (
            "_flow_enable_vector_retriever",
            "_flow_enable_bm25_retriever",
            "_flow_enable_graph_retriever",
            "_flow_enable_regex_retriever",
        ):
            setattr(orchestrator, flow_flag, False)

        class _NoPathsRag:
            @staticmethod
            def parse_chat_command(_query: str) -> tuple[str, str]:
                return "", ""

            def retrieve(self, _session: Any) -> tuple[str, int]:
                local_docs = orchestrator._run_local_docs_stage(
                    LocalSessionStub(),
                    retrieve_mode="ALL",
                    bm25_query="question",
                    alternate_queries=[],
                    orig_translated_query_en="question",
                    resolved_guardrail=GuardrailInputs("question", "", False),
                )
                assert local_docs.vector_docs == []
                assert local_docs.bm25_docs == []
                assert local_docs.graph_docs == []
                assert local_docs.regex_docs == []
                return "", 0

        chatter = MagicMock()
        chatter.is_streaming = False
        chatter.rag = _NoPathsRag()
        chatter.prompt = "{context}\n{input}"
        chatter.cfg = SimpleNamespace(
            get_str=lambda key, *_args: "RAGChat" if key == "_FRIENDLY_NAME" else "",
            get=lambda key, default=None: (
                120 if key == "TERMINAL_LINE_SIZE" else default
            ),
        )
        chatter.pretty.write = MagicMock()
        chatter.print_llm_answer = MagicMock()
        chatter.confidence_logger.flush_turn_csv = MagicMock()
        chatter.llmCaller.call_llm = MagicMock()
        chatter._build_query_notice.return_value = ""
        chatter._handle_no_results.side_effect = (
            lambda *args: Chatter._handle_no_results(chatter, *args)
        )
        session = SimpleNamespace(
            query="question",
            retrieve_mode="ALL",
            retriever_k=5,
            max_output_tokens=128,
            temperature=0.1,
            top_k=20,
            top_p=0.9,
            clarification_response=None,
            debug_level=0,
            debug_mode="ge",
        )

        success, response = Chatter.run(chatter, session, is_streaming=False)

        assert success is True
        assert response is not None
        assert "context is empty" in response
        chatter.llmCaller.call_llm.assert_not_called()

    def test_run_local_retrievers_stores_guardrail_counts_when_active(self) -> None:
        host = LocalHostStub(use_guardrail=True)
        session = LocalSessionStub()
        orchestrator = Orchestrator(host)

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
        assert host.last_run_indexed_args == (
            "Q2",
            True,
            {"FileName": "A.txt", "Language": "en"},
        )
        assert host.indexed_stage_labels == [["BM25"]]
        assert host.last_store_args == (2, 3)
        assert session.retrieval_top_k_orig_query_en == 2
        assert session.retrieval_top_k_post_rewrite_query_en == 3

    def test_run_local_retrievers_skips_guardrail_store_when_inactive(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        orchestrator = Orchestrator(host)

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
        assert host.last_run_indexed_args == (
            "Q2",
            False,
            {"FileName": "A.txt", "Language": "en"},
        )
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
        orchestrator = Orchestrator(host)

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
            {"FileName": "A.txt", "Language": "en"},
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
        orchestrator = Orchestrator(host)

        bm25_stage_labels_seen: list[list[str]] = []
        graph_regex_stage_labels_seen: list[list[str]] = []

        original_indexed_stage_group = orchestrator._run_indexed_stage_group

        def _wrapped_indexed_stage_group(
            mySession: LocalSessionStub,
            *,
            stage_invocations: list[Any],
            specs: list[tuple[str, bool, float, Any, int, Any]],
            use_guardrail: bool,
            dispatch_per_spec: bool,
        ) -> tuple[dict[str, list[Any]], int, int]:
            _ = (mySession, stage_invocations, use_guardrail)
            labels = [spec[0] for spec in specs]
            if dispatch_per_spec:
                graph_regex_stage_labels_seen.append(labels)
            else:
                bm25_stage_labels_seen.append(labels)
            return original_indexed_stage_group(
                mySession,
                stage_invocations=stage_invocations,
                specs=specs,
                use_guardrail=use_guardrail,
                dispatch_per_spec=dispatch_per_spec,
            )

        monkeypatch.setattr(
            orchestrator,
            "_run_indexed_stage_group",
            _wrapped_indexed_stage_group,
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
        orchestrator = Orchestrator(host)

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
            return [
                _Invocation(
                    {"FileName": "B.txt"},
                    "Q2|bm25",
                    "Q1|bm25",
                )
            ]

        monkeypatch.setattr(
            orchestrator,
            "_bm25_stage_invocations",
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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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

        class LocalStageResult:
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
            return LocalStageResult()

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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

        class LocalStageResult:
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
            return LocalStageResult()

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)
        orchestrator._indexed_alternate_queries = ["ALT1", "ALT2"]

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

        assert [it.language_bucket for it in iterations] == [
            "de",
            "de",
            "de",
            "en",
            "en",
            "en",
        ]
        assert [it.file_filter for it in iterations] == [
            stage_filters[0],
            stage_filters[0],
            stage_filters[0],
            stage_filters[1],
            stage_filters[1],
            stage_filters[1],
        ]
        assert [it.primary_query for it in iterations] == [
            "Q2|de",
            "ALT1|de",
            "ALT2|de",
            "Q2|en",
            "ALT1|en",
            "ALT2|en",
        ]
        assert [it.guardrail_query for it in iterations] == [
            "Q1|de",
            "",
            "",
            "Q1|en",
            "",
            "",
        ]
        status_messages = [
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3
            and call[0][1] == "Retrieval Orchestration"
            and "idx_stage flow=" in str(call[0][2])
        ]
        assert len(status_messages) == 2
        assert "flow=N/A language=de" in status_messages[0]
        assert "primary_query source=REWRITTEN_QUERY lang=en->de" in status_messages[0]
        assert (
            "guardrail_query source=TRANSLATED_QUERY lang=en->de" in status_messages[0]
        )
        assert "original='Q1'" in status_messages[0]
        assert "staged='Q1|de'" in status_messages[0]
        assert "alternative_query='ALT1'" in status_messages[0]
        assert "stage_alternative_query='ALT1|de'" in status_messages[0]
        assert "alternative_query='ALT2'" in status_messages[0]
        assert "stage_alternative_query='ALT2|de'" in status_messages[0]
        assert "alternative_query='ALT1'" in status_messages[1]
        assert "stage_alternative_query='ALT1|en'" in status_messages[1]

    def test_indexed_stage_status_identifies_flow_disabled_guardrail(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)

        orchestrator.orchestratorMessages.write_indexed_stage_query_status(
            flow_name="NATIVE_ALL",
            target_language="de",
            primary_query_source="REWRITTEN_QUERY",
            primary_source_language="en",
            was_translated=True,
            primary_query="describe the device",
            stage_primary_query="Gerät beschreiben",
            guardrail_query_source="TRANSLATED_QUERY",
            guardrail_source_language="en",
            guardrail_query="",
            stage_guardrail_query="",
            guardrail_disabled_reason="disabled_by_flow",
            indexed_shaping_enabled=True,
        )

        message = next(
            str(call[0][2])
            for call in host.pretty.calls
            if len(call[0]) >= 3
            and call[0][1] == "Retrieval Orchestration"
            and "idx_stage flow=" in str(call[0][2])
        )
        assert "flow=NATIVE_ALL language=de" in message
        assert "guardrail_query source=TRANSLATED_QUERY lang=en->de" in message
        assert (
            "value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)" in message
        )

    def test_run_local_retrievers_graph_regex_stage_shapes_queries_per_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=True,
            mode_flags=(False, True, True, True),
        )
        session = LocalSessionStub()
        orchestrator = Orchestrator(host)

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

        orchestrator = Orchestrator(host)
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

        orchestrator = Orchestrator(host)
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
        orchestrator = Orchestrator(host)

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

    def test_shape_graph_regex_stage_queries_does_not_log_incomplete_query_set(
        self,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        host._translation_backend = "argos"  # type: ignore[attr-defined]
        shared = _LocalTranslatingSharedStub()
        host._shared = shared  # type: ignore[attr-defined]
        session = LocalSessionStub()
        session.debug_level = 0
        orchestrator = Orchestrator(host)

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
        assert not any(
            "idx_stage flow=" in message for message in orchestration_messages
        )

    def test_shape_graph_regex_stage_queries_passes_translation_and_synonym_toggles(
        self,
    ) -> None:
        class _ToggleHost(LocalHostStub):
            def __init__(self) -> None:
                super().__init__(use_guardrail=False)
                self.received_flags: list[tuple[bool, bool]] = []

            def _shape_graph_regex_stage_queries(
                self,
                mySession: LocalSessionStub,
                *,
                language_bucket: str | None,
                primary_query: str,
                guardrail_query: str,
                enable_translation: bool = True,
                enable_synonym_expansion: bool = True,
            ) -> tuple[str, str]:
                _ = (mySession, language_bucket)
                self.received_flags.append(
                    (enable_translation, enable_synonym_expansion)
                )
                return primary_query, guardrail_query

        host = _ToggleHost()
        session = LocalSessionStub()
        orchestrator = Orchestrator(host)
        orchestrator._flow_enable_indexed_query_translation = False
        orchestrator._flow_enable_indexed_query_wordnet_expansion = False

        stage_primary_query, stage_guardrail_query = (
            orchestrator._shape_graph_regex_stage_queries(
                session,
                language_bucket="de",
                primary_query="Q2",
                guardrail_query="Q1",
            )
        )

        assert stage_primary_query == "Q2"
        assert stage_guardrail_query == "Q1"
        assert host.received_flags == [(False, False)]

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
        orchestrator = Orchestrator(host)

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

    def test_run_local_retrievers_original_language_leg_runs_vector_and_indexed(
        self,
    ) -> None:
        host = LocalHostStub(
            use_guardrail=False,
            mode_flags=(True, True, True, True),
        )
        host.collection = _LocalCollectionStub(  # type: ignore[attr-defined]
            [{"Language": "en"}, {"Language": "de"}]
        )
        host._translation_backend = "argos"  # type: ignore[attr-defined]
        host._shared = _LocalTranslatingSharedStub()  # type: ignore[attr-defined]

        session = LocalSessionStub()
        session.current_query_lang = "german"
        session.user_language = "german"
        session.retrieval_language = "english"
        session.user_query_original = "Schenkung mit Auflage"

        def _fake_run_vector_retriever_with_guardrail(
            *,
            mySession: LocalSessionStub,
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            alternate_queries: list[str],
        ) -> tuple[list[Any], int, int]:
            _ = (mySession, guardrail_query, use_guardrail, alternate_queries)
            if primary_query == "Schenkung mit Auflage":
                return [RetrievalDocStub("v-de")], 0, 1
            return [RetrievalDocStub("v-en")], 0, 1

        def _fake_run_indexed_local_retrievers(
            mySession: LocalSessionStub,
            *,
            indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
            primary_query: str,
            guardrail_query: str,
            use_guardrail: bool,
            file_filter: dict[str, Any] | None,
        ) -> tuple[dict[str, list[Any]], int, int]:
            _ = mySession
            active_labels = [
                label
                for label, enabled, weight, *_ in indexed_specs
                if enabled and weight != 0.0
            ]
            host.indexed_stage_labels.append(active_labels)
            host.run_indexed_file_filters.append(file_filter)
            host.run_indexed_primary_queries.append(primary_query)
            host.run_indexed_guardrail_queries.append(guardrail_query)

            is_native_leg = primary_query == "Schenkung mit Auflage"
            docs: dict[str, list[Any]] = {
                "BM25": [
                    (
                        RetrievalDocStub("b-de")
                        if is_native_leg
                        else RetrievalDocStub("b-en")
                    )
                ],
                "Graph": [
                    (
                        RetrievalDocStub("g-de")
                        if is_native_leg
                        else RetrievalDocStub("g-en")
                    )
                ],
                "Regex": [
                    (
                        RetrievalDocStub("r-de")
                        if is_native_leg
                        else RetrievalDocStub("r-en")
                    )
                ],
            }
            return docs, 0, len(active_labels)

        host._run_vector_retriever_with_guardrail = _fake_run_vector_retriever_with_guardrail  # type: ignore[attr-defined]
        host._run_indexed_local_retrievers = _fake_run_indexed_local_retrievers  # type: ignore[attr-defined]

        orchestrator = Orchestrator(host)
        vector_docs, bm25_docs, graph_docs, regex_docs = (
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="ALL",
                bm25_query="Q2",
                alternate_queries=[],
                orig_translated_query_en="Q2",
            )
        )

        vector_ids = [doc.metadata.get("id") for doc in vector_docs]
        bm25_ids = [doc.metadata.get("id") for doc in bm25_docs]
        graph_ids = [doc.metadata.get("id") for doc in graph_docs]
        regex_ids = [doc.metadata.get("id") for doc in regex_docs]

        assert "v-en" in vector_ids
        assert "v-de" in vector_ids
        assert "b-en" in bm25_ids
        assert "b-de" in bm25_ids
        assert "g-en" in graph_ids
        assert "g-de" in graph_ids
        assert "r-en" in regex_ids
        assert "r-de" in regex_ids
        assert session.original_query_leg_enabled is True
        assert session.original_query_leg_language == "de"
        assert session.original_query_leg_vector_added == 1
        assert session.original_query_leg_bm25_added == 1
        assert session.original_query_leg_graph_added == 1
        assert session.original_query_leg_regex_added == 1

        native_stage_calls = [
            (filt, query)
            for filt, query in zip(
                host.run_indexed_file_filters,
                host.run_indexed_primary_queries,
            )
            if query == "Schenkung mit Auflage"
        ]
        assert native_stage_calls
        assert all(
            isinstance(filt, dict) and filt.get("Language") == "de"
            for filt, _query in native_stage_calls
        )

    def test_normalize_language_bucket_prefers_host_seam(self) -> None:
        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)

        calls: list[Any] = []

        def _fake_normalize_language_bucket(language: Any) -> str | None:
            calls.append(language)
            return "zz"

        host._normalize_language_bucket = _fake_normalize_language_bucket  # type: ignore[attr-defined]

        normalized = orchestrator._normalize_language_bucket("german")

        assert normalized == "zz"
        assert calls == ["german"]

    def test_normalize_language_bucket_propagates_host_error(self, monkeypatch) -> None:
        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)

        def fail(language: Any) -> str | None:
            raise ValueError("language normalization failed")

        monkeypatch.setattr(host, "_normalize_language_bucket", fail)

        with pytest.raises(ValueError, match="language normalization failed"):
            orchestrator._normalize_language_bucket("german")

    def test_query_shaping_type_error_is_logged_without_retry(
        self, monkeypatch
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)
        calls: list[dict[str, Any]] = []

        def fail(session: Any, **kwargs: Any) -> tuple[str, str]:
            calls.append(kwargs)
            raise TypeError("query shaping failed")

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", fail)

        with pytest.raises(TypeError, match="query shaping failed"):
            orchestrator._shape_graph_regex_stage_queries(
                LocalSessionStub(),
                language_bucket="de",
                primary_query="animal habitats",
                guardrail_query="",
            )

        assert len(calls) == 1
        assert any(
            args[0] == "E" and "query shaping failed" in str(args[2])
            for args, _ in host.pretty.calls
        )

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
        orchestrator = Orchestrator(host)

        languages = orchestrator._discover_graph_regex_languages(
            {"FileName": "A.txt", "Language": "en"}
        )

        assert languages == ["de", "en"]
        assert host.collection.calls == [  # type: ignore[attr-defined]
            {
                "include": ["metadatas"],
                "where": {"FileName": "A.txt"},
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
        orchestrator = Orchestrator(host)

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

    @pytest.mark.parametrize(
        ("file_filter", "expected_where", "expected_buckets"),
        [
            (None, [None], ["en", "de"]),
            ({}, [None], ["en", "de"]),
            ({"Language": "de"}, [None], ["de"]),
            ({"Language": {"$eq": "de"}}, [None], ["de"]),
            ({"FileName": "A.txt"}, [None, {"FileName": "A.txt"}], ["de"]),
            (
                {"Language": "de", "FileName": "A.txt"},
                [None, {"FileName": "A.txt"}],
                ["de"],
            ),
        ],
    )
    def test_language_planning_reuses_identical_discovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        file_filter: dict[str, Any] | None,
        expected_where: list[dict[str, Any] | None],
        expected_buckets: list[str],
    ) -> None:
        class _CollectionStub:
            def __init__(self) -> None:
                self.where_calls: list[dict[str, Any] | None] = []

            def get(self, **kwargs: Any) -> dict[str, Any]:
                where = kwargs.get("where")
                self.where_calls.append(where)
                languages = ["de"] if where else ["en", "de"]
                return {"metadatas": [{"Language": language} for language in languages]}

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[assignment]
        orchestrator = Orchestrator(host)
        active_calls: list[None] = []
        resolve_active = orchestrator._active_configured_language_codes

        def count_active() -> set[str]:
            active_calls.append(None)
            return resolve_active()

        monkeypatch.setattr(
            orchestrator, "_active_configured_language_codes", count_active
        )

        buckets = orchestrator._plan_graph_regex_language_buckets(
            LocalSessionStub(), file_filter
        )

        assert buckets == expected_buckets
        assert host.collection.where_calls == expected_where
        assert len(active_calls) == 1

    def test_plan_graph_regex_language_buckets_intersects_active_and_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        session = LocalSessionStub()
        session.current_query_lang = "german"
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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

        class _SharedStub:
            def __init__(self) -> None:
                self.lang_name_to_code: dict[str, str] = {
                    "english": "en",
                    "german": "de",
                }

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[attr-defined]
        host.cfg = _LocalCfgStub()
        host.pretty = StubPretty()  # type: ignore[attr-defined]
        host._shared = _SharedStub()  # type: ignore[attr-defined]
        host.collection_name = "TestCollection"  # type: ignore[attr-defined]

        session = LocalSessionStub()
        orchestrator = Orchestrator(host)

        orchestrator.orchestratorMessages.emit_vector_store_language_status(
            session,
            vector_store_languages=orchestrator._discover_graph_regex_languages(None),
            active_codes=sorted(orchestrator._active_configured_language_codes()),
        )

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

        host = LocalHostStub(use_guardrail=False)
        host.collection = _CollectionStub()  # type: ignore[attr-defined]
        host.cfg = _LocalCfgStub()
        host.pretty = StubPretty()  # type: ignore[attr-defined]
        host.collection_name = "TestCollection"  # type: ignore[attr-defined]

        session = LocalSessionStub()
        orchestrator = Orchestrator(host)

        orchestrator.orchestratorMessages.emit_vector_store_language_status(
            session,
            vector_store_languages=orchestrator._discover_graph_regex_languages(None),
            active_codes=sorted(orchestrator._active_configured_language_codes()),
        )

        assert any(
            len(call[0]) >= 3
            and call[0][0] == "W"
            and call[0][1] == "LanguageConfig"
            and "not active in _ARGOS_DEFINITIONS.ACTIVE_LANGUAGES" in str(call[0][2])
            and "fr" in str(call[0][2])
            for call in host.pretty.calls  # type: ignore[attr-defined]
        )


class TestMultilingualQueryPlan:
    @pytest.mark.parametrize("debug", [0, 30])
    def test_translation_warnings_deduplicate_per_turn_language_and_category(
        self,
        monkeypatch,
        debug,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        host.collection = _LocalCollectionStub([{"Language": "en"}, {"Language": "de"}])

        def shape(session, *, primary_query, guardrail_query, **kwargs):
            if primary_query.startswith("bad"):
                raise QueryTranslationError("translation failed")
            return primary_query, guardrail_query

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        orchestrator = Orchestrator(host)
        session = LocalSessionStub()
        session.debug_level = debug
        for turn in range(2):
            orchestrator.run_local_retrievers(
                session,
                retrieve_mode="BM25",
                bm25_query="original",
                alternate_queries=["bad1", "bad2", "bad3"],
            )
            warnings = [
                args
                for args, _ in host.pretty.calls
                if args[0] == "W" and "Skipping alternative" in str(args[2])
            ]
            assert len(warnings) == (turn + 1) * 2
            assert len(session.indexed_query_notices) == 2
        details = [
            args[2]
            for args, _ in host.pretty.calls
            if args[1] == "QueryPlan" and "translation failure;" in str(args[2])
        ]
        assert len(details) == (12 if debug == 30 else 0)
        assert {
            ("alternative", "en"),
            ("alternative", "de"),
        } == session.indexed_query_warning_keys

    @pytest.mark.parametrize("limit, expected", [(0, ["en", "fr"]), (5000, ["en"])])
    def test_discovers_languages_beyond_old_cutoff(self, limit, expected) -> None:
        class _Collection:
            def get(self, **kwargs):
                assert kwargs["where"] == {"FileName": "A.txt"}
                metas = [{"Language": "en"}] * 5000 + [{"Language": "fr"}]
                return {"metadatas": metas[: kwargs.get("limit", len(metas))]}

        host = LocalHostStub(use_guardrail=False)
        host.cfg = _LocalCfgStub(indexed_settings={"language_detection_limit": limit})
        host.collection = _Collection()
        assert (
            Orchestrator(host)._discover_graph_regex_languages(
                {"FileName": "A.txt", "Language": "en"}
            )
            == expected
        )

    @pytest.mark.parametrize(
        "settings",
        [
            {"language_detection_limit": -1},
            {"language_detection_limit": True},
            {"translation_failure_policy": "fallback"},
            {"use_alternates": "false"},
        ],
    )
    def test_rejects_invalid_settings(self, settings) -> None:
        host = LocalHostStub(use_guardrail=False)
        host.cfg = _LocalCfgStub(indexed_settings=settings)
        with pytest.raises(ConfigurationError):
            Orchestrator(host)._indexed_query_settings()

    def test_shares_four_by_four_matrix_across_three_retrievers(
        self, monkeypatch
    ) -> None:
        host = LocalHostStub(use_guardrail=False, mode_flags=(False, True, True, True))
        host.cfg = _LocalCfgStub(active_languages=["de", "en", "fr", "it"])
        host.collection = _LocalCollectionStub(
            [{"Language": language} for language in ["de", "en", "fr", "it"]]
        )
        translations = []

        def shape(
            session, *, language_bucket, primary_query, guardrail_query, **kwargs
        ):
            if language_bucket != "de":
                translations.append((primary_query, language_bucket))
                return f"{primary_query}|{language_bucket}", ""
            return primary_query, ""

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        session = LocalSessionStub()
        session.current_query_lang = session.retrieval_language = "de"
        orchestrator = Orchestrator(host)
        orchestrator.run_local_retrievers(
            session,
            retrieve_mode="ALL",
            bm25_query="original",
            alternate_queries=["alt1", "alt2", "alt3"],
        )
        assert len(translations) == 12
        assert len(set(translations)) == 12
        assert len(host.run_indexed_primary_queries) == 48
        for language in ["de", "en", "fr", "it"]:
            suffix = "" if language == "de" else f"|{language}"
            expected = {
                f"{query}{suffix}" for query in ["original", "alt1", "alt2", "alt3"]
            }
            for label in ["BM25", "Graph", "Regex"]:
                actual = set()
                for query, file_filter, labels in zip(
                    host.run_indexed_primary_queries,
                    host.run_indexed_file_filters,
                    host.indexed_stage_labels,
                ):
                    assert file_filter is not None
                    if file_filter["Language"] == language and label in labels:
                        actual.add(query)
                assert actual == expected

    @pytest.mark.parametrize("policy", ["stop", "skip_language"])
    def test_primary_failure_applies_policy_before_vector_search(
        self, monkeypatch, policy
    ) -> None:
        host = LocalHostStub(use_guardrail=False, mode_flags=(True, True, True, True))
        host.cfg = _LocalCfgStub(
            indexed_settings={"translation_failure_policy": policy}
        )
        host.collection = _LocalCollectionStub([{"Language": "en"}, {"Language": "de"}])

        def shape(
            session, *, language_bucket, primary_query, guardrail_query, **kwargs
        ):
            if language_bucket == "de":
                raise QueryTranslationError("missing en->de")
            return primary_query, guardrail_query

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        orchestrator = Orchestrator(host)
        arguments = dict(
            retrieve_mode="ALL",
            bm25_query="original",
            alternate_queries=[],
        )
        if policy == "stop":
            with pytest.raises(QueryTranslationError, match="document language 'de'"):
                orchestrator.run_local_retrievers(LocalSessionStub(), **arguments)
            assert "_run_vector_retriever_with_guardrail" not in host.calls
            assert host.run_indexed_primary_queries == []
        else:
            orchestrator.run_local_retrievers(LocalSessionStub(), **arguments)
            assert "_run_vector_retriever_with_guardrail" in host.calls
            assert all(
                f is not None and f["Language"] == "en"
                for f in host.run_indexed_file_filters
            )
            assert any("coverage is incomplete" in str(a) for a, _ in host.pretty.calls)
        assert orchestrator._prepared_indexed_stage_iterations is None

    def test_failed_alternative_does_not_abort_strict_policy(self, monkeypatch) -> None:
        host = LocalHostStub(use_guardrail=False)
        host.cfg = _LocalCfgStub(
            indexed_settings={"translation_failure_policy": "stop"}
        )

        def shape(
            session, *, language_bucket, primary_query, guardrail_query, **kwargs
        ):
            if primary_query == "bad":
                raise QueryTranslationError("alternative unavailable")
            return primary_query, guardrail_query

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        Orchestrator(host).run_local_retrievers(
            LocalSessionStub(),
            retrieve_mode="BM25",
            bm25_query="original",
            alternate_queries=["good", "bad"],
        )
        assert host.run_indexed_primary_queries == ["original", "good"]
        assert any("Skipping alternative" in str(a) for a, _ in host.pretty.calls)

    def test_equivalent_shaped_alternatives_are_reported_but_dispatched_once(
        self,
        monkeypatch,
    ) -> None:
        host = LocalHostStub(use_guardrail=False)
        shaped_queries: list[str] = []

        def shape(session, *, primary_query, guardrail_query, **kwargs):
            shaped_queries.append(primary_query)
            return (
                "normalized" if primary_query == "original" else "same alternative",
                guardrail_query,
            )

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        Orchestrator(host).run_local_retrievers(
            LocalSessionStub(),
            retrieve_mode="BM25",
            bm25_query="original",
            alternate_queries=["ORIGINAL", "first", "second"],
        )

        assert shaped_queries == ["original", "first", "second"]
        assert host.run_indexed_primary_queries == ["normalized", "same alternative"]
        reports = [
            str(args[2])
            for args, _ in host.pretty.calls
            if len(args) > 2 and "idx_stage flow=" in str(args[2])
        ]
        assert len(reports) == 1
        assert "alternative_query='first'" in reports[0]
        assert "alternative_query='second'" in reports[0]

    def test_all_failed_languages_without_vector_or_web_reports_error(
        self, monkeypatch
    ) -> None:
        host = LocalHostStub(use_guardrail=False)

        def shape(*args, **kwargs):
            raise QueryTranslationError("no translator")

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        with pytest.raises(QueryTranslationError, match="No usable"):
            Orchestrator(host).run_local_retrievers(
                LocalSessionStub(),
                retrieve_mode="BM25",
                bm25_query="original",
                alternate_queries=[],
            )
        assert host.run_indexed_primary_queries == []
        assert any("No usable" in str(a) for a, _ in host.pretty.calls)

    def test_failed_guardrail_is_omitted_even_under_stop_policy(
        self, monkeypatch
    ) -> None:
        host = LocalHostStub(use_guardrail=True)
        host.cfg = _LocalCfgStub(
            indexed_settings={"translation_failure_policy": "stop"}
        )

        def shape(session, *, primary_query, guardrail_query, **kwargs):
            if primary_query == "seed":
                raise QueryTranslationError("seed translation failed")
            return primary_query, guardrail_query

        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        Orchestrator(host).run_local_retrievers(
            LocalSessionStub(),
            retrieve_mode="BM25",
            bm25_query="original",
            alternate_queries=[],
            orig_translated_query_en="seed",
        )
        assert host.last_run_indexed_args == (
            "original",
            False,
            {"FileName": "A.txt", "Language": "en"},
        )
        assert any("Skipping guardrail" in str(a) for a, _ in host.pretty.calls)

    def test_zero_weight_indexed_retrievers_do_not_require_translation(
        self, monkeypatch
    ) -> None:
        host = LocalHostStub(use_guardrail=False, mode_flags=(True, True, True, True))
        host.cfg = _LocalCfgStub(
            indexed_settings={"translation_failure_policy": "stop"}
        )
        original_specs = host._indexed_retriever_specs

        def specs(*args, **kwargs):
            return [
                (label, enabled, 0.0, retriever, level, printer)
                for label, enabled, weight, retriever, level, printer in original_specs(
                    *args, **kwargs
                )
            ]

        def shape(*args, **kwargs):
            pytest.fail("Disabled indexed retrievers must not translate queries")

        monkeypatch.setattr(host, "_indexed_retriever_specs", specs)
        monkeypatch.setattr(host, "_shape_graph_regex_stage_queries", shape)
        session = LocalSessionStub()
        session.bm25_weight = session.graph_weight = session.regex_weight = 0.0
        Orchestrator(host).run_local_retrievers(
            session,
            retrieve_mode="ALL",
            bm25_query="original",
            alternate_queries=[],
        )
        assert "_run_vector_retriever_with_guardrail" in host.calls
        assert host.run_indexed_primary_queries == []

    @pytest.mark.parametrize("debug, enabled", [(0, True), (30, True), (30, False)])
    def test_debug_30_steps_and_alternates_switch(self, debug, enabled) -> None:
        host = LocalHostStub(use_guardrail=False)
        host.cfg = _LocalCfgStub(indexed_settings={"use_alternates": enabled})
        session = LocalSessionStub()
        session.debug_level = debug
        Orchestrator(host).run_local_retrievers(
            session,
            retrieve_mode="BM25",
            bm25_query="original",
            alternate_queries=["alternative"],
        )
        assert host.run_indexed_primary_queries == (
            ["original", "alternative"] if enabled else ["original"]
        )
        messages = [
            str(a[2])
            for a, _ in host.pretty.calls
            if len(a) > 2 and a[1] == "QueryPlan"
        ]
        if debug == 30:
            assert any(
                "Step 4" in message and "whole collection" in message
                for message in messages
            )
            assert not any("Step 5" in message for message in messages)
            assert any("Step 6" in message for message in messages)
        else:
            assert messages == []
        indexed_messages = [
            str(a[2])
            for a, _ in host.pretty.calls
            if len(a) > 2
            and a[1] == "Retrieval Orchestration"
            and "idx_stage flow=" in str(a[2])
        ]
        assert len(indexed_messages) == 1
        assert "original='original'" in indexed_messages[0]
        assert ("alternative_query='alternative'" in indexed_messages[0]) is enabled

    def test_repeated_alternative_hits_do_not_outvote_original(
        self, monkeypatch
    ) -> None:
        from Orchestrator.OrchestratorTypes import GraphRegexStageIteration

        host = LocalHostStub(use_guardrail=False)
        orchestrator = Orchestrator(host)
        original = RetrievalDocStub("anchor")

        def search(session, *, primary_query, **kwargs):
            docs = (
                [original]
                if primary_query == "original"
                else [RetrievalDocStub("alternate")]
            )
            return {"BM25": docs, "Graph": [], "Regex": []}, 0, len(docs)

        monkeypatch.setattr(orchestrator, "_run_indexed_specs_stage", search)
        docs, _, _ = orchestrator._run_indexed_stage_group(
            LocalSessionStub(),
            stage_invocations=[
                GraphRegexStageIteration("en", {"Language": "en"}, query, "", index > 0)
                for index, query in enumerate(["original", "alt1", "alt2", "alt3"])
            ],
            specs=[("BM25", True, 1.0, None, 10, None)],
            use_guardrail=False,
            dispatch_per_spec=False,
        )
        assert [doc.metadata["id"] for doc in docs["BM25"]] == ["anchor", "alternate"]


class TestWebRetrievalOrchestration:
    def test_run_web_retriever_returns_empty_when_not_triggered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_MODE", "1")
        host = WebHostStub([WebDocStub("https://a")], cfg=WebCfgStub())
        session = WebSessionStub(web_search=False)
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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
        orchestrator = Orchestrator(host)

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
