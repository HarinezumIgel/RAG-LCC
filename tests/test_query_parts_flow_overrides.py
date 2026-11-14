"""CLI knobs for flow overrides (rerank_query_source, retrieval_language, flow aliases)."""

from unittest.mock import Mock
from unittest.mock import MagicMock

import pytest

from Chat.QueryParts import QueryParts
from Config.Config import Config
from Configuration.Config_Orchestrator import (
    _ORCHESTRATION_FLOW_ALIASES,
    _ORCHESTRATION_FLOWS,
)
from Globals.Session import Session
from Gui.Picker import RETURN_LABEL
from Helpers.Helpers import Helpers


@pytest.fixture
def query_parts():
    qp = object.__new__(QueryParts)
    qp.session = Session()
    values = {
        "_ORCHESTRATION_FLOWS": _ORCHESTRATION_FLOWS,
        "_ORCHESTRATION_FLOW_ALIASES": _ORCHESTRATION_FLOW_ALIASES,
        "_DEFAULT_ORCHESTRATION_FLOW": "NATIVE_ALL",
        "_ALLOWED_RETRIEVE_MODES": ["ALL", "VECTOR", "BM25", "GRAPH", "REGEX", "WEB"],
    }
    qp.cfg = Mock(spec=Config)
    for method in (qp.cfg.get, qp.cfg.get_dict, qp.cfg.get_list, qp.cfg.get_str):
        method.side_effect = lambda key, default=None: values.get(key, default)
    qp.tokenBudget = Mock()
    qp.prompt_map = QueryParts.COMMAND_SPECS
    qp.allowed_orchestration_flows = list(_ORCHESTRATION_FLOWS)
    return qp


def test_rerank_query_source_set_show_clear(query_parts, capsys):
    query_parts._set("rerank_query_source", "original_query")
    assert query_parts.session.rerank_query_source == "ORIGINAL_QUERY"
    query_parts._show("rerank_query_source")
    assert "[rerank_query_source] ORIGINAL_QUERY" in capsys.readouterr().out
    query_parts._clear("rerank_query_source")
    assert query_parts.session.rerank_query_source is None


def test_invalid_rerank_query_source_keeps_previous(query_parts, capsys):
    query_parts.session.rerank_query_source = "REWRITTEN_QUERY"
    query_parts._set("rerank_query_source", "bogus")
    assert "Invalid rerank_query_source" in capsys.readouterr().out
    assert query_parts.session.rerank_query_source == "REWRITTEN_QUERY"


def test_retrieval_language_set_show_clear(query_parts, capsys):
    query_parts._set("retrieval_language", "native")
    assert query_parts.session.retrieval_language_override == "NATIVE"
    query_parts._show("retrieval_language")
    assert "[retrieval_language] NATIVE" in capsys.readouterr().out
    query_parts._clear("retrieval_language")
    assert query_parts.session.retrieval_language_override is None


def test_invalid_retrieval_language_keeps_previous(query_parts, capsys):
    query_parts.session.retrieval_language_override = "ENGLISH"
    query_parts._set("retrieval_language", "german")
    assert "Invalid retrieval_language" in capsys.readouterr().out
    assert query_parts.session.retrieval_language_override == "ENGLISH"


def test_keyword_action_splitter_alias_set_clear_and_help(query_parts, capsys):
    token = query_parts._normalize_token("keyword_action_splitter")
    query_parts._set(token, "true")
    assert query_parts.session.enable_keyword_action_splitter is True
    query_parts.help()
    assert "keyword_action_splitter:" in capsys.readouterr().out
    query_parts._clear("enable_keyword_action_splitter")
    assert query_parts.session.enable_keyword_action_splitter is None


def test_help_matches_startup_sections_and_keeps_all_commands(query_parts, capsys):
    query_parts.help()
    output = capsys.readouterr().out
    headings = [
        "Chat settings",
        "Summarize settings",
        "Retrieval settings",
        "Orchestration settings",
    ]
    positions = [output.index(heading) for heading in headings]
    assert positions == sorted(positions)
    for key, spec in QueryParts.COMMAND_SPECS.items():
        assert f"      ▶ {key}:" in output
        assert spec["prompt"] in output
    assert output.index("      ▶ rerank_query_source:") > positions[3]
    assert output.index("      ▶ summary_length:") > positions[1]
    assert output.index("      ▶ threshold:") > positions[2]
    assert "show? displays grouped current values" in output
    assert "summarize: name or full path" in output
    assert "strategy*name / orchestrator_flow*FLOW" in output
    for key in (
        "query_rewrite",
        "pronoun_subst",
        "rerank_stage",
        "low_recall_rescue",
        "low_score_fallback",
        "grounding",
        "vector",
        "bm25",
        "graph",
        "regex",
    ):
        assert f"      ▶ {key}:" in output


def test_help_keeps_new_commands_visible(query_parts, capsys):
    query_parts.prompt_map = {
        **QueryParts.COMMAND_SPECS,
        "future_setting": {
            "section": "New section",
            "prompt": "Future command explanation.",
        },
    }
    query_parts.help()
    output = capsys.readouterr().out
    assert "Other commands" in output
    assert "future_setting: Future command explanation." in output


def test_shared_settings_layout_preserves_values_and_deduplicates_help(
    query_parts, capsys
):
    values = {
        "file_cap": 15,
        "use_chat_context": True,
        "summary_length": "medium",
        "strategy": "DEFAULT",
        "query_rewrite": False,
        "orchestrator_flow": "NATIVE_ALL",
    }
    query_parts._print_settings(values)
    output = capsys.readouterr().out
    for section in QueryParts._SETTINGS_LAYOUT:
        assert section in output
    assert "retrieval strategy='DEFAULT'" in output
    assert "query_rewrite=False" in output
    assert output.count("file_cap=15") == 2
    query_parts.help()
    help_output = capsys.readouterr().out
    assert help_output.count("      ▶ file_cap:") == 1
    assert help_output.count("      ▶ use_chat_context:") == 1


@pytest.mark.parametrize(
    "flow, language",
    [
        ("NATIVE_ALL", "NATIVE"),
        ("ENGLISH_ALL", "ENGLISH"),
        ("ORIGINAL_VECTOR", "ENGLISH"),
    ],
)
def test_flow_defaults_seed_language_and_rerank_source(query_parts, flow, language):
    query_parts.session.retrieval_language_override = "something-stale"
    query_parts.session.rerank_query_source = "ORIGINAL_QUERY"
    query_parts.session.enable_keyword_action_splitter = False
    query_parts.applyOrchestrationFlowDefaults(flow)
    assert query_parts.session.enable_keyword_action_splitter is True
    assert query_parts.session.retrieval_language_override == language
    assert query_parts.session.rerank_query_source == (
        _ORCHESTRATION_FLOWS[flow]["rerank_query_source"]
    )


@pytest.mark.parametrize(
    "legacy, current",
    [
        ("BGE_ENC_XENC", "NATIVE_ALL"),
        ("thorough_query_rewrite", "ENGLISH_ALL_MULTILEG"),
    ],
)
def test_legacy_flow_name_resolves(query_parts, legacy, current):
    assert query_parts.applyOrchestrationFlowDefaults(legacy) == current
    assert query_parts.session.orchestrator_flow == current


def test_invalid_orchestrator_flow_keeps_previous(query_parts, capsys):
    query_parts.applyOrchestrationFlowDefaults("ENGLISH_ALL")
    query_parts._set("orchestrator_flow", "TRANSLATION_FOCUSED")
    assert "Invalid orchestrator_flow" in capsys.readouterr().out
    assert query_parts.session.orchestrator_flow == "ENGLISH_ALL"


@pytest.mark.parametrize(
    "name, raw, expected",
    [
        ("summary_length", "very long", "very_long"),
        ("summary_temperature", "0", 0.0),
        ("chunk_overlap", "0", 0),
    ],
)
def test_summarize_settings_set_show_clear(
    query_parts: QueryParts,
    capsys: pytest.CaptureFixture[str],
    name: str,
    raw: str,
    expected: str | float | int,
) -> None:
    query_parts._set(name, raw)  # pyright: ignore[reportPrivateUsage]
    assert getattr(query_parts.session, name) == expected
    query_parts._show(name)  # pyright: ignore[reportPrivateUsage]
    assert f"[{name}] {expected}" in capsys.readouterr().out
    query_parts._clear(name)  # pyright: ignore[reportPrivateUsage]
    assert getattr(query_parts.session, name) is None
    assert query_parts.COMMAND_SPECS[name]["section"] == "Summarize settings"


@pytest.mark.parametrize(
    "name, raw",
    [
        ("summary_length", "huge"),
        ("summary_temperature", "nan"),
        ("summary_temperature", "2"),
        ("chunk_overlap", "-1"),
        ("chunk_overlap", "1.5"),
    ],
)
def test_invalid_summarize_setting_reports_error(
    query_parts: QueryParts,
    capsys: pytest.CaptureFixture[str],
    name: str,
    raw: str,
) -> None:
    query_parts._set(name, raw)  # pyright: ignore[reportPrivateUsage]
    assert getattr(query_parts.session, name) is None
    assert name in capsys.readouterr().out


@pytest.mark.parametrize(
    "name, selection",
    [
        ("summary_length", "long"),
        ("summary_temperature", "0.5"),
        ("chunk_overlap", "3"),
    ],
)
def test_summary_pickers_apply_values(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    selection: str,
) -> None:
    import Chat.QueryParts as module

    picker = MagicMock()
    picker.select.return_value.execute.return_value = selection
    picker.text.return_value.execute.return_value = selection
    monkeypatch.setattr(module, "inquirer", picker)
    query_parts._ask(name)  # pyright: ignore[reportPrivateUsage]
    expected = {
        "summary_length": "long",
        "summary_temperature": 0.5,
        "chunk_overlap": 3,
    }
    assert getattr(query_parts.session, name) == expected[name]


def test_summarize_section_shows_effective_defaults(
    query_parts: QueryParts,
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_parts.cfg = Config()
    query_parts.print_values()
    output = capsys.readouterr().out
    assert "Summarize settings" in output
    assert "summary_length='medium'" in output
    assert "summary_temperature=0.3" in output
    assert "chunk_overlap=5" in output


def test_retriever_tuning_values_stay_on_one_line(query_parts: QueryParts) -> None:
    output = query_parts._format_settings(
        {"fetch_k": 100, "context_chunks": 50, "threshold": 0.5}
    )
    tuning_line = next(
        line for line in output.splitlines() if "Retriever tuning" in line
    )
    assert "fetch_k=100  context_chunks=50  threshold=0.5" in tuning_line


@pytest.fixture
def strategy_warning_parts(query_parts):
    helpers = Mock(spec=Helpers)
    helpers.get_chunker_max_size.return_value = 256
    helpers.get_model_args.return_value = {
        "MODEL": "test-model",
        "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
        "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
    }
    budget = Mock()
    budget.get_effective_context_limit.return_value = 32768
    budget.count_tokens_approx.return_value = 332
    pretty = Mock()
    query_parts.helpers = helpers
    query_parts.tokenBudget = budget
    query_parts.pretty = pretty
    query_parts.session.strategy = "ULTRA_WIDE"
    query_parts.session.retriever_k = 3000
    return query_parts, budget, pretty


@pytest.mark.parametrize(
    "chunks, context_limit, warns",
    [
        (20, 32768, False),
        (89, 32768, False),
        (90, 32768, True),
        (1500, 32768, True),
        (150, 65536, False),
    ],
)
def test_strategy_warning_uses_active_session_chunk_count(
    strategy_warning_parts,
    chunks: int,
    context_limit: int,
    warns: bool,
) -> None:
    query_parts, budget, _ = strategy_warning_parts
    query_parts.session.final_chunks_to_llm = chunks
    budget.get_effective_context_limit.return_value = context_limit
    warning = query_parts.strategy_context_warning(query_parts.session)
    assert bool(warning) is warns
    if warning:
        assert f"context_chunks={chunks:,}" in warning
        assert f"({context_limit:,} tokens)" in warning
        assert "larger LLM context window" in warning
        assert "Actual fit depends" in warning
    assert query_parts.session.final_chunks_to_llm == chunks
    assert query_parts.session.retriever_k == 3000
    budget.get_effective_context_limit.assert_called_once_with(
        "test-model", query_parts.session
    )


@pytest.mark.parametrize("chunks, warns", [(20, False), (1500, True)])
def test_strategy_picker_warns_using_active_chunks(
    strategy_warning_parts,
    monkeypatch: pytest.MonkeyPatch,
    chunks: int,
    warns: bool,
) -> None:
    import Chat.QueryParts as module

    query_parts, _, pretty = strategy_warning_parts
    query_parts.session.final_chunks_to_llm = chunks
    query_parts.allowed_strategies = ["ULTRA_WIDE"]
    query_parts._base_defaults = Mock(return_value="")
    monkeypatch.setattr(module, "pick_value", Mock(return_value="ULTRA_WIDE"))
    query_parts._ask("strategy")
    assert bool(pretty.write.call_args_list) is warns
    if warns:
        assert pretty.write.call_args.args[0] == "W"
        assert f"context_chunks={chunks:,}" in pretty.write.call_args.args[2]
    assert query_parts.session.final_chunks_to_llm == chunks


def test_context_chunk_override_updates_warning_with_new_value(
    strategy_warning_parts,
) -> None:
    query_parts, _, pretty = strategy_warning_parts
    query_parts.session.final_chunks_to_llm = 20
    query_parts._set("context_chunks", "1500")
    assert "context_chunks=1,500" in pretty.write.call_args.args[2]
    assert query_parts.session.final_chunks_to_llm == 1500


@pytest.mark.parametrize(
    "name",
    [
        "summary_length",
        "confidence_evaluation",
        "enable_rerank",
        "strategy",
        "orchestrator_flow",
        "force_retrieve_mode",
        "main_query_source",
        "secondary_query_source",
        "rerank_query_source",
        "retrieval_language",
        "debug_level",
        "debug_mode",
        "chat_name",
        "web_search",
        "fetch_page_content",
    ],
)
def test_return_from_setting_picker_preserves_session_and_config(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    import Chat.QueryParts as module

    query_parts.allowed_strategies = ["DEFAULT"]
    query_parts.allowed_debug_levels = {"off": 0, "normal": 30, "verbose": 40}
    query_parts.session.debug_level = 30
    query_parts.session.debug_mode = "ge"
    query_parts.session.enable_rerank = True
    config_set = Mock()
    monkeypatch.setattr(query_parts.cfg, "set", config_set)
    monkeypatch.setattr(module, "get_active_language_codes", lambda cfg: ["en"])
    monkeypatch.setattr(module, "get_lang_code_to_name", lambda cfg: {"en": "English"})
    monkeypatch.setattr(
        QueryParts, "_list_chat_names", lambda self, collection: ["chat"]
    )
    monkeypatch.setattr(
        module.QualityDimensionEvaluator, "resolve_mode", lambda cfg, s: "off"
    )
    picker = MagicMock()
    picker.select.return_value.execute.return_value = None
    monkeypatch.setattr(module, "inquirer", picker)
    before = dict(vars(query_parts.session))

    query_parts._ask(name)  # pyright: ignore[reportPrivateUsage]

    assert vars(query_parts.session) == before
    config_set.assert_not_called()
    choices = picker.select.call_args.kwargs["choices"]
    assert choices[-1].value is None
    assert choices[-1].name == RETURN_LABEL


def test_return_from_debug_comparison_rolls_back_level_choice(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import Chat.QueryParts as module

    query_parts.allowed_debug_levels = {"normal": 30, "verbose": 40}
    query_parts.session.debug_level = 30
    query_parts.session.debug_mode = "ge"
    config_set = Mock()
    monkeypatch.setattr(query_parts.cfg, "set", config_set)
    picker = MagicMock()
    picker.select.return_value.execute.side_effect = [40, None]
    monkeypatch.setattr(module, "inquirer", picker)

    query_parts._ask("debug_level")  # pyright: ignore[reportPrivateUsage]

    assert query_parts.session.debug_level == 30
    assert query_parts.session.debug_mode == "ge"
    config_set.assert_not_called()


@pytest.mark.parametrize(
    "level, mode, combined",
    [
        (0, "ge", "none"),
        (30, "ge", "ge 30"),
        (40, "is", "is 40"),
    ],
)
def test_debug_picker_applies_level_and_mode_together(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    mode: str,
    combined: str,
) -> None:
    import Chat.QueryParts as module

    query_parts.allowed_debug_levels = {"off": 0, "normal": 30, "verbose": 40}
    config_set = Mock()
    monkeypatch.setattr(query_parts.cfg, "set", config_set)
    picker = MagicMock()
    picker.select.return_value.execute.side_effect = [level, mode]
    monkeypatch.setattr(module, "inquirer", picker)

    query_parts._ask("debug_level")  # pyright: ignore[reportPrivateUsage]

    assert query_parts.session.debug_level == level
    assert query_parts.session.debug_mode == mode
    config_set.assert_called_once_with("DEBUG_LEVEL", combined, force=True)


@pytest.mark.parametrize("enabled", [True, False])
def test_boolean_picker_applies_true_and_false(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    import Chat.QueryParts as module

    picker = MagicMock()
    picker.select.return_value.execute.return_value = enabled
    monkeypatch.setattr(module, "inquirer", picker)

    query_parts._ask("enable_rerank")  # pyright: ignore[reportPrivateUsage]

    assert query_parts.session.enable_rerank is enabled
    assert query_parts.session.rerank is enabled


@pytest.mark.parametrize("name", ["strategy", "orchestrator_flow"])
def test_return_from_defaults_picker_does_not_reset_session(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    import Chat.QueryParts as module

    query_parts.allowed_strategies = ["DEFAULT"]
    config_set = Mock()
    monkeypatch.setattr(query_parts.cfg, "set", config_set)
    picker = MagicMock()
    picker.select.return_value.execute.return_value = None
    monkeypatch.setattr(module, "inquirer", picker)
    before = dict(vars(query_parts.session))

    query_parts._defaults(name, "")  # pyright: ignore[reportPrivateUsage]

    assert vars(query_parts.session) == before
    config_set.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "summary_temperature",
        "chunk_overlap",
        "temperature",
        "rerank",
        "use_chat_context",
        "mark_text",
        "terminal_line_size",
        "context_size",
    ],
)
def test_empty_typed_setting_keeps_current_value(
    query_parts: QueryParts,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    import Chat.QueryParts as module

    picker = MagicMock()
    picker.text.return_value.execute.return_value = ""
    config_set = Mock()
    monkeypatch.setattr(query_parts.cfg, "set", config_set)
    monkeypatch.setattr(module, "inquirer", picker)
    before = dict(vars(query_parts.session))

    query_parts._ask(name)  # pyright: ignore[reportPrivateUsage]

    assert picker.text.call_args.kwargs["validate"]("") is True
    assert vars(query_parts.session) == before
    config_set.assert_not_called()


def test_collection_return_does_not_switch_store_or_clear_filters(
    query_parts: QueryParts,
) -> None:
    query_parts.collectionPicker = MagicMock()
    query_parts.collectionPicker.pick_collection.return_value = None
    query_parts.ragChatImpl = MagicMock()
    query_parts.hist = MagicMock()
    query_parts.session.file_path = r"D:\docs\current.pdf"
    query_parts.session.metadata_filters = {"Author": "A"}
    before = dict(vars(query_parts.session))

    query_parts._ask("collection")  # pyright: ignore[reportPrivateUsage]

    assert vars(query_parts.session) == before
    query_parts.ragChatImpl.set_vector_store.assert_not_called()
    query_parts.hist.save.assert_not_called()


@pytest.mark.parametrize("name", ["file", "path"])
def test_file_picker_return_preserves_pin(
    query_parts: QueryParts,
    name: str,
) -> None:
    query_parts.fileList = MagicMock()
    query_parts.fileList.select.return_value = None
    query_parts.session.file_path = r"D:\docs\current.pdf"
    query_parts.session.file_path_select = "path"
    before = dict(vars(query_parts.session))

    query_parts._ask(name)  # pyright: ignore[reportPrivateUsage]

    assert vars(query_parts.session) == before
    query_parts.fileList.set.assert_not_called()
