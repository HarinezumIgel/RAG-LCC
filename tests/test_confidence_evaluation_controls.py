"""Confidence evaluation session controls without constructing the chat pipeline."""

import re
from unittest.mock import Mock

import pytest

from Chat.QueryParts import QueryParts
from Config.Config import Config
from Configuration.Config_Orchestrator import _ORCHESTRATION_FLOWS
from Globals.Session import Session
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator


@pytest.fixture
def query_parts():
    qp = object.__new__(QueryParts)
    qp.session = Session()
    values = {
        "_ORCHESTRATION_FLOWS": _ORCHESTRATION_FLOWS,
        "_ALLOWED_RETRIEVE_MODES": ["ALL", "VECTOR", "BM25", "GRAPH", "REGEX", "WEB"],
    }
    qp.cfg = Mock(spec=Config)
    for method in (qp.cfg.get, qp.cfg.get_dict, qp.cfg.get_list, qp.cfg.get_str):
        method.side_effect = lambda key, default=None: values.get(key, default)
    qp.tokenBudget = Mock()
    qp.prompt_map = QueryParts.COMMAND_SPECS
    qp.allowed_orchestration_flows = list(_ORCHESTRATION_FLOWS)
    return qp


@pytest.mark.parametrize("mode", QualityDimensionEvaluator.MODES)
def test_cli_set_show_and_clear(query_parts, mode, capsys):
    query_parts._set("confidence_evaluation", mode.upper())
    assert query_parts.session.confidence_evaluation == mode
    query_parts._show("confidence_evaluation")
    assert f"[confidence_evaluation] {mode}" in capsys.readouterr().out
    query_parts._clear("confidence_evaluation")
    assert query_parts.session.confidence_evaluation is None
    assert (
        QualityDimensionEvaluator.resolve_mode(query_parts.cfg, query_parts.session)
        == "both"
    )


def test_invalid_cli_mode_preserves_previous_value(query_parts, capsys):
    query_parts.session.confidence_evaluation = "llm"
    query_parts._set("confidence_evaluation", "unknown")
    assert "Invalid confidence_evaluation" in capsys.readouterr().out
    assert query_parts.session.confidence_evaluation == "llm"


def test_picker_offers_exact_modes(query_parts, monkeypatch):
    picker = Mock()
    picker.select.return_value.execute.return_value = "llm"
    monkeypatch.setattr("Chat.QueryParts.inquirer", picker)
    query_parts._ask("confidence_evaluation")
    choices = picker.select.call_args.kwargs["choices"]
    assert choices[:-1] == [
        "off",
        "heuristic",
        "llm",
        "both",
    ]
    assert choices[-1].value is None
    assert query_parts.session.confidence_evaluation == "llm"


def test_settings_include_confidence_line(query_parts, capsys):
    query_parts.session.confidence_evaluation = "heuristic"
    query_parts.print_values()
    output = capsys.readouterr().out
    lines = output.splitlines()
    overrides_index = lines.index(
        next(line for line in lines if "Orchestrator overrides" in line)
    )
    confidence_index = lines.index(
        next(line for line in lines if "confidence_evaluation='heuristic'" in line)
    )
    assert confidence_index > overrides_index
    assert "▶ Confidence" in re.sub(r"\x1b\[[0-9;]*m", "", lines[confidence_index])


def test_snapshot_matches_cli_and_includes_all_setting_commands(query_parts, capsys):
    query_parts.cfg = Config()
    query_parts.session.summary_temperature = 0.0
    query_parts.session.metadata_filters = {"Author": "Example"}
    query_parts.session.preferred_response_language = "german"
    query_parts.session.enable_grounding = False
    query_parts.session.max_output_tokens_override = 512
    query_parts.session.context_size_override = 4096
    query_parts.print_values()
    cli = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out).strip()

    snapshot = query_parts.format_session_snapshot()

    assert cli in snapshot
    assert "\x1b" not in snapshot
    values = query_parts.session_values()
    assert set(QueryParts.COMMAND_SPECS) - {"help"} <= set(values)
    for token in set(QueryParts.COMMAND_SPECS) - {"help", "strategy"}:
        assert f"{token}=" in snapshot
    assert "summary_temperature=0.0" in snapshot
    assert "preferred_response_language='german'" in snapshot
    assert "grounding=False" in snapshot
    assert "override(api: max_tokens)=512" in snapshot
    assert "override(api: num_ctx)=4096" in snapshot


def test_snapshot_excludes_internal_state(query_parts):
    query_parts.cfg = Config()
    session = query_parts.session
    session.query = "INTERNAL_QUERY"
    session.user_language = "INTERNAL_LANGUAGE"
    session.grounding_proxy_text = "INTERNAL_PROXY"
    session.last_answer_content = "INTERNAL_ANSWER"
    session.chunk_texts_for_grounding = ["INTERNAL_CHUNK"]
    session.indexed_query_error = "INTERNAL_ERROR"

    snapshot = query_parts.format_session_snapshot()

    assert "INTERNAL_" not in snapshot
    for key in (
        "grounding_proxy_text",
        "last_answer_content",
        "summary_statements",
        "current_query_lang",
        "user_language",
        "retrieval_language_mode",
        "file_path_select",
        "base_kwargs",
        "marked_documents",
        "answer_confidence_result",
        "confidence_step_events",
    ):
        assert f"{key}=" not in snapshot


@pytest.mark.parametrize("flow_name", _ORCHESTRATION_FLOWS)
def test_flow_selection_restores_enabled_default(query_parts, flow_name):
    assert _ORCHESTRATION_FLOWS[flow_name]["confidence_evaluation"] == "both"
    query_parts.session.confidence_evaluation = "off"
    query_parts.applyOrchestrationFlowDefaults(flow_name)
    assert query_parts.session.confidence_evaluation is None
    assert (
        QualityDimensionEvaluator.resolve_mode(query_parts.cfg, query_parts.session)
        == "both"
    )
