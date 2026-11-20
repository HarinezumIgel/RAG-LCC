from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Config.Config import Config
from Globals.Session import Session
from Gui.Informer import Informer
from Helpers.DebugHelper import DebugHelper


class TestDebugHelperCheckSession:
    def test_check_session_accepts_string_numeric_level(self):
        session = SimpleNamespace(debug_level="30", debug_mode="ge")

        assert DebugHelper.check_session(session, 30) is True
        assert DebugHelper.check_session(session, 29) is False
        assert DebugHelper.check_session(session, 31) is True

    def test_check_session_defaults_to_le_when_mode_is_missing(self):
        session = SimpleNamespace(debug_level=10, debug_mode=None)

        assert DebugHelper.check_session(session, 5) is True
        assert DebugHelper.check_session(session, 11) is False


@pytest.mark.parametrize("raw", [30, "30", " 30 "])
def test_bare_level_defaults_to_le_for_config_guards(raw: str | int) -> None:
    assert DebugHelper.parse(raw) == (30, "le")
    cfg = Mock(spec=Config)
    cfg.get_str.return_value = str(raw)
    helper = DebugHelper(cfg)
    for level, visible in ((29, True), (30, True), (31, False), (40, False)):
        assert helper.active(level) is visible
        assert DebugHelper.check(cfg, level) is visible


@pytest.mark.parametrize("mode", [None, "le", "ge", "is"])
def test_api_numeric_debug_level_defaults_to_le(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
) -> None:
    from Api.ChatCompletionHandler import ChatCompletionRequest
    from Api.ChatRequestSessionMapper import RequestSessionMapper
    from Chat.QueryParts import QueryParts

    cfg = Config()
    config_set = Mock()
    monkeypatch.setattr(cfg, "set", config_set)
    session = Session()
    session.debug_mode = "ge"
    request = ChatCompletionRequest(
        model="documents",
        messages=[],
        debug_level=30,
        debug_mode=mode,
    )

    RequestSessionMapper(Mock(spec=QueryParts), cfg).apply(request, session)

    expected_mode = mode or "le"
    assert session.debug_mode == expected_mode
    assert DebugHelper.check_session(session, 40) is (expected_mode == "ge")
    config_set.assert_called_once_with("DEBUG_LEVEL", f"{expected_mode} 30")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("le", [True, True, False]),
        ("ge", [False, True, True]),
        ("is", [False, True, False]),
        ("<=", [True, True, False]),
        (">=", [False, True, True]),
        ("==", [False, True, False]),
    ],
)
def test_comparison_modes_match_message_level(mode, expected):
    cfg = Mock(spec=Config)
    cfg.get_str.return_value = f"{mode} 30"
    session = SimpleNamespace(debug_level=30, debug_mode=mode)
    helper = DebugHelper(cfg)

    for level, visible in zip((29, 30, 31), expected):
        assert helper.active(level) is visible
        assert DebugHelper.check(cfg, level) is visible
        assert DebugHelper.check_session(session, level) is visible


@pytest.mark.parametrize("mode", ["ge", "le", "is"])
def test_zero_level_is_silent_in_config_and_session(mode):
    cfg = Mock(spec=Config)
    cfg.get_str.return_value = f"{mode} 0"
    session = SimpleNamespace(debug_level=0, debug_mode=mode)

    assert not DebugHelper(cfg).active(30)
    assert not DebugHelper.check(cfg, 30)
    assert not DebugHelper.check_session(session, 30)


@pytest.mark.parametrize(
    ("level", "mode", "visible"),
    [
        (30, "le", False),
        (30, "is", False),
        (30, "ge", True),
        (40, "le", True),
        (40, "is", True),
        (0, "ge", False),
        (None, "le", True),
    ],
)
def test_informer_config_dump_uses_session_not_global(
    monkeypatch: pytest.MonkeyPatch,
    level: int | None,
    mode: str,
    visible: bool,
) -> None:
    informer = object.__new__(Informer)
    informer.cfg = Mock(spec=Config)
    informer.cfg.get_str.return_value = "ge 30"
    informer.pretty = Mock()
    informer.piplineSummarizer = Mock()
    informer.friendly_name = "RAGChat"
    startup_calls = [Mock(), Mock(), Mock()]
    for method, callback in zip(
        ("_check_ollama_is_running", "_inform_cuda", "_llm_info"), startup_calls
    ):
        monkeypatch.setattr(informer, method, callback)
    session = object.__new__(Session)
    session.debug_level = level
    session.debug_mode = mode

    informer.inform(session)

    assert informer.cfg.print_config_values.called is visible
    labels = [call.args[1] for call in informer.pretty.write.call_args_list]
    assert ("Actual config values:" in labels) is visible
    informer.piplineSummarizer.display.assert_called_once()
    for callback in startup_calls:
        callback.assert_called_once()


def test_command_processor_settings_and_queries_share_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import Chat.CommandProcessor as module

    query_parts = Mock()
    query_parts.session = object.__new__(Session)
    for name in ("RAGChatImpl", "ChromaDBHelper", "HistoryManager"):
        monkeypatch.setattr(module, name, Mock())
    monkeypatch.setattr(module, "QueryParts", Mock(return_value=query_parts))
    cfg = Mock(spec=Config)
    cfg.get.return_value = 160
    monkeypatch.setattr(module, "Config", Mock(return_value=cfg))

    processor = module.CommandProcessor()

    assert processor.session is processor.queryParts.session
