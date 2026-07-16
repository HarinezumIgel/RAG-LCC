from types import SimpleNamespace

from Helpers.DebugHelper import DebugHelper


class TestDebugHelperCheckSession:
    def test_check_session_accepts_string_numeric_level(self):
        session = SimpleNamespace(debug_level="30", debug_mode="ge")

        assert DebugHelper.check_session(session, 30) is True
        assert DebugHelper.check_session(session, 31) is False

    def test_check_session_defaults_to_ge_when_mode_is_missing(self):
        session = SimpleNamespace(debug_level=10, debug_mode=None)

        assert DebugHelper.check_session(session, 5) is True
        assert DebugHelper.check_session(session, 11) is False
