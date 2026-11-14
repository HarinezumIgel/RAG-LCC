"""Regression tests for orchestration helper ownership and state restoration."""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Gui.Colors import BRIGHT_MAGENTA, CYAN
from Orchestrator.Orchestrator import Orchestrator
from Orchestrator.OrchestratorHelper import OrchestratorHelper
from Orchestrator.OrchestratorMessages import OrchestratorMessages
from Orchestrator.OrchestratorOverrides import OrchestratorOverrides
from Orchestrator.OrchestratorTypes import (
    GuardrailInputs,
    IndexedStageResult,
    LocalDocs,
    OriginalQueryLeg,
    PostGatePipelineInputs,
    RetrievalPlan,
)


class _PrettyStub:
    def __init__(self) -> None:
        self.always_on = False
        self.fail = False
        self.calls: list[tuple[str, str, str, bool]] = []
        self.options: list[dict[str, Any]] = []

    def write(self, severity: str, stage: str, message: str, **kwargs: Any) -> None:
        self.calls.append((severity, stage, message, self.always_on))
        self.options.append(kwargs)
        if self.fail:
            raise RuntimeError("writer failed")


def _normalize_language(value: Any) -> str | None:
    return str(value).strip().lower() if value else None


def _helpers() -> tuple[OrchestratorOverrides, OrchestratorMessages, Any]:
    host = SimpleNamespace(cfg=None, pretty=_PrettyStub(), collection_name="test")
    overrides = OrchestratorOverrides(host)
    messages = OrchestratorMessages(
        host,
        overrides=overrides,
        normalize_language_bucket=_normalize_language,
    )
    return overrides, messages, host


def _query_status(messages: OrchestratorMessages, query: str) -> None:
    messages.write_indexed_stage_query_status(
        flow_name="test",
        target_language="de",
        primary_query_source="REWRITTEN_QUERY",
        primary_source_language="en",
        was_translated=False,
        primary_query=query,
        stage_primary_query=query,
        guardrail_query_source="TRANSLATED_QUERY",
        guardrail_source_language="en",
        guardrail_query="",
        stage_guardrail_query="",
        guardrail_disabled_reason="disabled_by_flow",
        indexed_shaping_enabled=False,
    )


def test_orchestrator_composes_named_helpers() -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)

    assert isinstance(orchestrator.orchestratorMessages, OrchestratorMessages)
    assert isinstance(orchestrator.orchestratorOverrides, OrchestratorOverrides)
    assert isinstance(orchestrator.orchestratorHelper, OrchestratorHelper)
    assert not hasattr(orchestrator, "_emit_flow_knob_stage_statuses")
    assert not hasattr(orchestrator, "_apply_query_rewrite_flow_controls")


def test_session_overrides_preserve_baseline_and_gate_fallbacks() -> None:
    overrides, _, _ = _helpers()
    session: Any = SimpleNamespace(
        enable_rerank=False,
        force_retrieve_mode="WEB",
        main_query_source="ORIGINAL_QUERY",
    )
    baseline = overrides.resolve_orchestration_flow_profile(session)
    effective = overrides.resolve_effective_flow_profile_for_session(session, baseline)
    overrides.apply_query_rewrite_flow_controls(session, flow_profile=effective)

    assert baseline.enable_rerank is True
    assert effective.enable_rerank is False
    assert effective.retrieve_mode_override == "WEB"
    assert effective.primary_query_source == "ORIGINAL_QUERY"
    assert session.enable_low_score_fallback is False
    assert session.enable_low_recall_rescue is False
    assert session.retrieval_language_mode == "ENGLISH"


def test_rerank_query_source_final_clears_previous_override() -> None:
    overrides, _, _ = _helpers()
    session: Any = SimpleNamespace(user_query_original="native question")
    baseline = overrides.resolve_orchestration_flow_profile(session)
    plan = RetrievalPlan(
        user_query_original="original question",
        final_query="rewritten question",
        alternate_queries=[],
        retrieve_mode="ALL",
        bm25_query="retrieval question",
        orig_translated_query_en="translated question",
    )

    overrides.apply_rerank_query_source(
        session,
        plan=plan,
        flow_profile=replace(baseline, rerank_query_source="ORIGINAL_QUERY"),
    )
    assert session.rerank_query == "native question"

    overrides.apply_rerank_query_source(session, plan=plan, flow_profile=baseline)
    assert session.rerank_query is None


@pytest.mark.parametrize(
    "writer", ["write_retrieval_orchestration", "write_multiquery_status"]
)
@pytest.mark.parametrize("fail", [False, True])
def test_always_on_is_restored_even_when_writing_fails(writer: str, fail: bool) -> None:
    _, messages, host = _helpers()
    host.pretty.fail = fail
    write = getattr(messages, writer)

    if fail:
        with pytest.raises(RuntimeError, match="writer failed"):
            write(severity="I", message="status")
    else:
        write(severity="I", message="status")

    assert host.pretty.calls[0][3] is True
    assert host.pretty.always_on is False


@pytest.mark.parametrize(
    ("writer", "stage", "color"),
    [
        ("write_retrieval_orchestration", "Retrieval Orchestration", BRIGHT_MAGENTA),
        ("write_multiquery_status", "MultiQuery", CYAN),
    ],
)
@pytest.mark.parametrize("initial_always_on", [False, True])
def test_always_on_writers_preserve_stage_color_and_previous_state(
    writer: str, stage: str, color: str, initial_always_on: bool
) -> None:
    _, messages, host = _helpers()
    host.pretty.always_on = initial_always_on

    getattr(messages, writer)(severity="W", message="status")

    assert host.pretty.calls == [("W", stage, "status", True)]
    assert host.pretty.options == [{"color": color}]
    assert host.pretty.always_on is initial_always_on


@pytest.mark.parametrize(
    "writer", ["write_retrieval_orchestration", "write_multiquery_status"]
)
def test_always_on_writers_allow_an_unset_or_read_only_flag(writer: str) -> None:
    _, messages, host = _helpers()
    calls: list[tuple[str, str, str, dict[str, Any]]] = []

    class _ReadOnlyPretty:
        @property
        def always_on(self) -> bool:
            return False

        def write(self, severity: str, stage: str, message: str, **kwargs: Any) -> None:
            calls.append((severity, stage, message, kwargs))

    host.pretty = _ReadOnlyPretty()
    getattr(messages, writer)(severity="I", message="read-only", color=None)
    assert calls[0][2:] == ("read-only", {"color": None})

    host.pretty = SimpleNamespace(
        write=lambda severity, stage, message, **kwargs: calls.append(
            (severity, stage, message, kwargs)
        )
    )
    getattr(messages, writer)(severity="I", message="unset", color=None)
    assert calls[1][2:] == ("unset", {"color": None})


@pytest.mark.parametrize(
    ("flow_rerank", "session_rerank", "disabled_action", "cli_knob"),
    [
        (True, None, None, None),
        (True, 0, "Deactivated in session CLI", "rerank=0"),
        (False, None, "Deactivated because rerank stage is off", None),
        (False, 1, "Deactivated in session CLI", "rerank=1"),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("rescue", [False, True])
def test_rerank_fallback_statuses_preserve_flow_and_session_details(
    flow_rerank: bool,
    session_rerank: int | None,
    disabled_action: str | None,
    cli_knob: str | None,
    fallback: bool,
    rescue: bool,
) -> None:
    overrides, messages, host = _helpers()
    session: Any = SimpleNamespace(rerank=session_rerank)
    baseline = overrides.resolve_orchestration_flow_profile(session)
    profile = replace(
        baseline,
        enable_rerank=flow_rerank,
        enable_low_score_fallback=fallback,
        enable_low_recall_rescue=rescue,
    )

    messages.emit_primary_flow_stage_statuses(mySession=session, flow_profile=profile)

    for stage_name, configured in (
        ("Low-score fallback action", fallback),
        ("Low-recall rescue action", rescue),
    ):
        line = next(call[2] for call in host.pretty.calls if stage_name in call[2])
        effective = configured and flow_rerank and session_rerank != 0
        action = (
            disabled_action
            if configured and not effective
            else "Activated" if effective else "Not activated"
        )
        assert action in line
        assert f"value={'on' if effective else 'off'}" in line
        assert session.orchestration_step_confidence[stage_name] == (
            "PENDING" if effective else "N/A"
        )
        assert ("cli_knob=" in line) == bool(configured and not effective and cli_knob)
        if configured and not effective and cli_knob:
            assert f"cli_knob={cli_knob}" in line


@pytest.mark.parametrize(
    ("scope", "language", "include_language", "query", "expected"),
    [
        ("orchestrate", None, False, None, "status"),
        ("orchestrate", "en", True, None, f"language: {'en':<7} status"),
        (
            "Vector",
            "de",
            True,
            "question",
            f"retriever: {'Vector':<10} language: {'de':<7} query: 'question' status",
        ),
        ("Vector", None, False, None, f"retriever: {'Vector':<10} status"),
        ("  ", None, True, None, f"retriever: {'':<10} status"),
    ],
)
def test_trace_preserves_scope_language_and_spacing(
    scope: str,
    language: str | None,
    include_language: bool,
    query: str | None,
    expected: str,
) -> None:
    _, messages, host = _helpers()
    session: Any = SimpleNamespace()

    messages.trace(
        session,
        "status",
        retriever_scope=scope,
        language_bucket=language,
        query_text=query,
        include_language=include_language,
    )

    assert host.pretty.calls[0][2] == expected


@pytest.mark.parametrize(
    ("retrieve_mode", "web_search", "web_mode", "local_enabled", "web_enabled"),
    [
        ("VECTOR", False, "local_only", True, False),
        ("VECTOR", True, "local_and_web", True, True),
        ("WEB", False, "web_only", False, True),
        (" web ", True, "web_only", False, True),
    ],
)
def test_stage_gates_preserve_all_web_modes(
    retrieve_mode: str,
    web_search: bool,
    web_mode: str,
    local_enabled: bool,
    web_enabled: bool,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    session: Any = SimpleNamespace(web_search=web_search)

    gates = orchestrator._resolve_stage_execution_gates(
        session, retrieve_mode=retrieve_mode
    )

    assert (gates.web_mode, gates.local_stage_enabled, gates.web_stage_enabled) == (
        web_mode,
        local_enabled,
        web_enabled,
    )


def test_indexed_doc_merge_preserves_order_keys_and_independent_lists() -> None:
    first = {"BM25": ["first"], "Graph": ["graph"], "Other": ["ignored"]}
    second = {"BM25": ["second"], "Regex": ["regex"]}

    merged = OrchestratorHelper.merge_indexed_stage_docs(first, second)

    assert list(merged) == ["BM25", "Graph", "Regex"]
    assert merged == {
        "BM25": ["first", "second"],
        "Graph": ["graph"],
        "Regex": ["regex"],
    }
    assert merged["BM25"] is not first["BM25"]
    assert merged["Graph"] is not first["Graph"]
    assert merged["Regex"] is not second["Regex"]


@pytest.mark.parametrize(
    ("metadata", "document_id", "content", "expected"),
    [
        ({"chunk_id": "chunk", "id": "metadata"}, "document", "text", "id:chunk"),
        ({"id": "metadata"}, "document", "text", "id:document"),
        ({"id": "metadata"}, None, "text", "id:metadata"),
        ({}, None, "text", "text:text"),
    ],
)
def test_candidate_identity_preserves_identifier_priority(
    metadata: dict[str, str],
    document_id: str | None,
    content: str,
    expected: str,
) -> None:
    document = SimpleNamespace(metadata=metadata, id=document_id, page_content=content)

    assert OrchestratorHelper.doc_identity(document) == expected


def test_candidate_identity_uses_object_identity_without_ids_or_text() -> None:
    document = SimpleNamespace(metadata=None, page_content="")

    assert OrchestratorHelper.doc_identity(document) == f"obj:{id(document)}"


def test_native_leg_merge_preserves_objects_order_markers_and_counts() -> None:
    helper = OrchestratorHelper()
    original = SimpleNamespace(
        metadata={"chunk_id": "shared", "retriever_sources": "Vector"}
    )
    overlap = SimpleNamespace(metadata={"chunk_id": "shared"})
    added = SimpleNamespace(
        metadata={"chunk_id": "new", "retriever_sources": "VectorRaw"}
    )
    duplicate = SimpleNamespace(metadata={"chunk_id": "new"})
    base = [original]

    merged, added_count, overlap_count = helper.merge_docs_with_native_leg(
        base_docs=base,
        native_docs=[overlap, added, duplicate],
        source_marker="VectorRaw",
    )

    assert merged == [original, added]
    assert merged[0] is original
    assert merged[1] is added
    assert base == [original]
    assert (added_count, overlap_count) == (1, 2)
    assert original.metadata["retriever_sources"] == "Vector,VectorRaw"
    assert added.metadata["retriever_sources"] == "VectorRaw"


def test_variant_merge_uses_best_rank_instead_of_summing_alternate_hits() -> None:
    helper = OrchestratorHelper()
    first = SimpleNamespace(metadata={"chunk_id": "first"})
    second = SimpleNamespace(metadata={"chunk_id": "second"})
    alternate = SimpleNamespace(metadata={"chunk_id": "alternate"})
    second_variant = SimpleNamespace(metadata={"chunk_id": "second"})

    merged = helper.merge_ranked_variants(
        [
            ([first, second], 1.0),
            ([alternate, second_variant], 0.9),
            ([alternate], 0.9),
            ([alternate], 0.9),
        ],
        rrf_k=60,
    )

    assert merged == [first, second, alternate]
    assert merged[1] is second


def test_variant_merge_keeps_first_object_on_ties_and_replaces_on_improvement() -> None:
    helper = OrchestratorHelper()
    first = SimpleNamespace(metadata={"chunk_id": "shared"})
    tied = SimpleNamespace(metadata={"chunk_id": "shared"})
    other = SimpleNamespace(metadata={"chunk_id": "other"})

    tied_result = helper.merge_ranked_variants(
        [([first], 1.0), ([tied, other], 1.0)], rrf_k=60
    )
    assert tied_result[0] is first

    improved_result = helper.merge_ranked_variants(
        [([first], 1.0), ([tied, other], 2.0)], rrf_k=60
    )
    assert improved_result[0] is tied
    assert improved_result[1] is other


def _post_gate_inputs() -> PostGatePipelineInputs:
    return PostGatePipelineInputs(
        flow_name="test",
        primary_query_source="REWRITTEN_QUERY",
        guardrail_query_source="TRANSLATED_QUERY",
        enable_guardrail_leg=False,
        retrieve_mode="ALL",
        bm25_query="question",
        alternate_queries=[],
        orig_translated_query_en="seed",
        user_query_original="question",
        resolved_guardrail=GuardrailInputs("question", "", False),
        enable_indexed_query_shaping=True,
        enable_indexed_query_translation=True,
        enable_indexed_query_wordnet_expansion=True,
        enable_original_language_vector_leg=True,
        enable_vector_retriever=True,
        enable_bm25_retriever=True,
        enable_graph_retriever=True,
        enable_regex_retriever=True,
    )


def test_local_pipeline_stage_restores_legacy_mode_flag_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    session: Any = SimpleNamespace(web_search=False)

    def fail_local(session: Any, **kwargs: Any) -> LocalDocs:
        assert orchestrator._respect_legacy_retrieve_mode is False
        raise RuntimeError("local retrieval failed")

    host._normalize_language_bucket = (
        lambda language: str(language or "").strip().lower() or None
    )
    monkeypatch.setattr(orchestrator, "_run_local_docs_stage", fail_local)
    with pytest.raises(RuntimeError, match="local retrieval failed"):
        orchestrator._run_post_gate_pipeline(
            session, pipeline_inputs=_post_gate_inputs()
        )

    assert orchestrator._respect_legacy_retrieve_mode is True


def test_post_gate_pipeline_configures_then_runs_local_web_and_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    session: Any = SimpleNamespace(web_search=True)
    inputs = _post_gate_inputs()
    local_docs = LocalDocs([], [], [], [])
    web_docs: list[Any] = [object()]
    calls: list[str] = []

    def run_local(session: Any, **kwargs: Any) -> LocalDocs:
        assert orchestrator._indexed_flow_name == inputs.flow_name
        assert kwargs["stage_gates"].local_stage_enabled is True
        calls.append("local")
        return local_docs

    def run_web(session: Any, **kwargs: Any) -> list[Any]:
        assert kwargs["stage_gates"].web_stage_enabled is True
        calls.append("web")
        return web_docs

    def merge(session: Any, local: LocalDocs, web: list[Any]) -> tuple[str, int]:
        assert local is local_docs
        assert web is web_docs
        calls.append("merge")
        return "context", 1

    monkeypatch.setattr(orchestrator, "_run_local_pipeline_stage", run_local)
    monkeypatch.setattr(orchestrator, "_run_web_pipeline_stage", run_web)
    monkeypatch.setattr(orchestrator, "_run_merge_and_context_stage", merge)

    assert orchestrator._run_post_gate_pipeline(session, pipeline_inputs=inputs) == (
        "context",
        1,
    )
    assert calls == ["local", "web", "merge"]


@pytest.mark.parametrize(
    ("leg_enabled", "weight", "should_run"),
    [(False, 1.0, False), (True, 0.0, False), (True, 1.0, True)],
)
def test_original_indexed_leg_preserves_gate_filter_and_unshaped_query(
    monkeypatch: pytest.MonkeyPatch,
    leg_enabled: bool,
    weight: float,
    should_run: bool,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    session: Any = SimpleNamespace()
    calls: list[dict[str, Any]] = []
    expected_result = IndexedStageResult([], [], [], 0, 0)

    def run_indexed(session: Any, **kwargs: Any) -> IndexedStageResult:
        calls.append(kwargs)
        return expected_result

    monkeypatch.setattr(
        orchestrator, "_run_indexed_retrieval_with_shape_mode", run_indexed
    )

    result = orchestrator._run_original_query_indexed_leg(
        session,
        original_query_leg=OriginalQueryLeg(leg_enabled, "raw query", "de", "test"),
        indexed_specs=[("BM25", True, weight, None, 10, None)],
        file_filter={"FileName": "A.txt", "Language": "en"},
    )

    assert len(calls) == int(should_run)
    if should_run:
        assert result is expected_result
        assert calls[0]["primary_query"] == "raw query"
        assert calls[0]["guardrail_query"] == ""
        assert calls[0]["use_guardrail"] is False
        assert calls[0]["shape_queries"] is False
        assert calls[0]["file_filter"] == {"FileName": "A.txt", "Language": "de"}
    else:
        assert result is None
        assert bool(host.pretty.calls) is leg_enabled


def test_nested_deferred_reports_restore_outer_buffer_after_failure() -> None:
    _, messages, host = _helpers()

    with messages.deferred_indexed_status():
        _query_status(messages, "outer query")
        assert host.pretty.calls == []
        with pytest.raises(RuntimeError, match="preflight failed"):
            with messages.deferred_indexed_status():
                _query_status(messages, "inner query")
                raise RuntimeError("preflight failed")
        messages.flush_indexed_stage_query_status()

    assert len(host.pretty.calls) == 1
    assert "outer query" in host.pretty.calls[0][2]
    assert "inner query" not in host.pretty.calls[0][2]
    _query_status(messages, "immediate query")
    assert len(host.pretty.calls) == 2
    assert "immediate query" in host.pretty.calls[1][2]


def test_corpus_language_status_cache_is_owned_by_messages() -> None:
    _, messages, host = _helpers()
    session: Any = SimpleNamespace()

    for _ in range(2):
        messages.emit_vector_store_language_status(
            session, vector_store_languages=["de", "en"], active_codes=["de", "en"]
        )
    assert len(host.pretty.calls) == 1

    host.collection_name = "another collection"
    messages.emit_vector_store_language_status(
        session, vector_store_languages=["de", "en"], active_codes=["de", "en"]
    )
    assert len(host.pretty.calls) == 2


def test_confidence_logging_preserves_event_details() -> None:
    _, messages, host = _helpers()
    session: Any = SimpleNamespace()
    events: list[dict[str, Any]] = []

    def log_step(session: Any, **kwargs: Any) -> None:
        events.append(kwargs)

    host.confidence_logger = SimpleNamespace(log_step=log_step)
    messages.log_step(
        session,
        step_name="retrieve local candidates",
        status="executed",
        confidence_level="HIGH",
        confidence_score=1.0,
        detail="vector=2 bm25=1",
        source="orchestrator",
    )

    assert events == [
        {
            "step_name": "retrieve local candidates",
            "status": "executed",
            "confidence_level": "HIGH",
            "confidence_score": 1.0,
            "detail": "vector=2 bm25=1",
            "source": "orchestrator",
        }
    ]


@pytest.mark.parametrize(
    ("web_mode", "retrieve_mode", "allowed", "warning"),
    [
        ("1", "HYBRID", True, None),
        ("0", "WEB", False, 'retrieve_mode=WEB requested but WEB_SEARCH_MODE="0"'),
        ("0", "HYBRID", False, "blocked by administrator"),
        ("2", "HYBRID", False, "WEB_SEARCH_MODE = '2'"),
    ],
)
def test_web_search_gate_preserves_mode_warnings(
    monkeypatch: pytest.MonkeyPatch,
    web_mode: str,
    retrieve_mode: str,
    allowed: bool,
    warning: str | None,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    monkeypatch.setenv("WEB_SEARCH_MODE", web_mode)

    assert orchestrator._is_web_search_allowed(retrieve_mode) is allowed
    if warning is None:
        assert host.pretty.calls == []
    else:
        assert host.pretty.calls[-1][0] == "W"
        assert warning in host.pretty.calls[-1][2]


def test_web_prefilters_run_bm25_then_cosine_and_report_dropped_docs() -> None:
    _, _, host = _helpers()
    thresholds = {
        "_WEB_SEARCH.bm25_pre_filter": 0.2,
        "_WEB_SEARCH.cosine_pre_filter": 0.5,
    }
    host.cfg = SimpleNamespace(get_float=thresholds.get)
    host.embedder = SimpleNamespace(embed_query=lambda query: [1.0])
    filter_calls: list[tuple[str, list[str]]] = []

    def bm25_prefilter(docs: list[str], query: str) -> list[str]:
        filter_calls.append(("bm25", list(docs)))
        return docs[:2]

    def cosine_prefilter(
        docs: list[str], query: str, query_vec: list[float]
    ) -> list[str]:
        filter_calls.append(("cosine", list(docs)))
        return docs[:1]

    host.web_pre_filter = SimpleNamespace(
        bm25_prefilter=bm25_prefilter,
        cosine_prefilter=cosine_prefilter,
    )
    debug_calls: list[tuple[list[str], list[str], str, float]] = []
    host._print_web_prefilter_debug = (
        lambda kept, dropped, label, threshold: debug_calls.append(
            (kept, dropped, label, threshold)
        )
    )
    orchestrator = Orchestrator(host)
    session: Any = SimpleNamespace(query="bees", debug_level=30, debug_mode="ge")

    result = orchestrator._apply_web_prefilters(session, ["a", "b", "c"])

    assert result == ["a"]
    assert filter_calls == [("bm25", ["a", "b", "c"]), ("cosine", ["a", "b"])]
    assert debug_calls == [
        (["a", "b"], ["c"], "BM25", 0.2),
        (["a"], ["b"], "Cosine", 0.5),
    ]
    messages = [call[2] for call in host.pretty.calls]
    assert "After BM25 pre-filter: 2/3 kept" in messages
    assert "After cosine pre-filter: 1/3 kept" in messages


def _doc(chunk_id: str) -> Any:
    return SimpleNamespace(metadata={"chunk_id": chunk_id}, page_content=chunk_id)


def test_native_indexed_leg_merge_returns_docs_and_stats_per_retriever() -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    base = IndexedStageResult(
        bm25_docs=[_doc("b1")],
        graph_docs=[],
        regex_docs=[_doc("r1")],
        top_k_orig_query_en=0,
        top_k_post_rewrite_query_en=0,
    )

    merged, stats = orchestrator._merge_native_indexed_leg(base, None)
    assert merged == {"bm25": base.bm25_docs, "graph": [], "regex": base.regex_docs}
    assert set(stats.values()) == {0} and len(stats) == 9
    assert host.pretty.calls == []

    native = IndexedStageResult(
        bm25_docs=[_doc("b1"), _doc("b2")],
        graph_docs=[_doc("g1")],
        regex_docs=[],
        top_k_orig_query_en=0,
        top_k_post_rewrite_query_en=0,
    )
    merged, stats = orchestrator._merge_native_indexed_leg(base, native)
    assert [doc.metadata["chunk_id"] for doc in merged["bm25"]] == ["b1", "b2"]
    assert [doc.metadata["chunk_id"] for doc in merged["graph"]] == ["g1"]
    assert (
        stats["bm25_hits"] == 2
        and stats["bm25_added"] == 1
        and stats["bm25_overlap"] == 1
    )
    assert stats["graph_added"] == 1 and stats["regex_hits"] == 0
    assert (
        "bm25(h=2,a=1,o=1) graph(h=1,a=1,o=0) regex(h=0,a=0,o=0)"
        in host.pretty.calls[-1][2]
    )


def test_expand_stage_iterations_dedupes_alternates_case_insensitively() -> None:
    iterations = Orchestrator._expand_stage_iterations(
        language_bucket="de",
        file_filter={"Language": "de"},
        primary_query="Bienen",
        guardrail_query="g",
        stage_alternatives=[("x", "bienen"), ("y", "Wespen"), ("z", "WESPEN")],
    )
    assert [
        (it.primary_query, it.guardrail_query, it.is_alternate) for it in iterations
    ] == [
        ("Bienen", "g", False),
        ("Wespen", "", True),
    ]


@pytest.mark.parametrize(
    ("enabled", "primary", "guardrail", "reason"),
    [
        (False, "a", "b", "disabled_by_flow"),
        (True, "Bees ", "bees", "same as primary"),
        (True, "bees", "", None),
        (True, "bees", "wasps", None),
    ],
)
def test_indexed_guardrail_disabled_reason(
    enabled: bool, primary: str, guardrail: str, reason: str | None
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    orchestrator._indexed_guardrail_enabled = enabled
    assert (
        orchestrator._indexed_guardrail_disabled_reason(
            primary_query=primary, guardrail_query=guardrail
        )
        == reason
    )


@pytest.mark.parametrize(
    ("primary", "guardrail", "expected_guardrail", "optional_calls"),
    [
        ("bees", "", "", 0),
        ("", "", "", 0),
        ("bees", "bees", "DE:bees", 0),
        ("bees", "wasps", "DE:wasps", 1),
    ],
)
def test_shape_stage_iteration_queries_guardrail_branches(
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    guardrail: str,
    expected_guardrail: str,
    optional_calls: int,
) -> None:
    _, _, host = _helpers()
    orchestrator = Orchestrator(host)
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "_shape_primary_indexed_query",
        lambda s, **kw: f"DE:{kw['query']}",
    )

    def shape_optional(session: Any, **kwargs: Any) -> str:
        calls.append(kwargs["category"])
        return f"DE:{kwargs['query']}"

    monkeypatch.setattr(orchestrator, "_shape_optional_indexed_query", shape_optional)
    shaped = orchestrator._shape_stage_iteration_queries(
        SimpleNamespace(),
        language_bucket="de",
        primary_query=primary,
        guardrail_query=guardrail,
        alternatives=[],
        failure_policy="skip_language",
    )
    assert shaped == (f"DE:{primary}", expected_guardrail, [])
    assert len(calls) == optional_calls
