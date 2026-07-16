# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnusedImport=false
"""Tests for Chatter._supports_osc8 and Chatter._open_marked_documents.

Heavy transitive imports from Chatter.__init__ are avoided by testing
the two methods directly on a lightweight shell object that inherits only
the methods under test.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Lightweight shell — carries only the two methods under test
# ---------------------------------------------------------------------------

# Import the real methods without triggering Chatter.__init__
import importlib
import importlib.util

_chatter_src = os.path.join(
    os.path.dirname(__file__), "..", "src", "Chat", "Chatter.py"
)

# We compile just enough to get the two method objects without running __init__:
# load the module source and grab the unbound functions from the class body.
_chatter_mod = None

# Attributes each stub module must expose so that Chatter.py's top-level
# "from X import Y" statements succeed when the real packages are not yet
# loaded (e.g. when this test file is run in isolation).
_STUB_ATTRS: dict[str, dict[str, Any]] = {
    "langchain_core.prompts": {"PromptTemplate": type("PromptTemplate", (), {})},
    "AI.AIHelpers": {"AIHelpers": type("AIHelpers", (), {})},
    "AI.LLMCaller": {"LLMCaller": type("LLMCaller", (), {})},
    "AI.ModelOutputAdapter": {
        "ModelOutput": type("ModelOutput", (), {}),
        "ModelOutputAdapter": type("ModelOutputAdapter", (), {}),
    },
    "AI.ModelsCache": {"ModelsCache": type("ModelsCache", (), {})},
    "AI.TensorHelpers": {"TensorHelpers": type("TensorHelpers", (), {})},
    "AI.TokenBudget": {"TokenBudget": type("TokenBudget", (), {})},
    "Algos.Masker": {"Masker": type("Masker", (), {})},
    "Chat.ChatContext": {"ChatContext": type("ChatContext", (), {})},
    "Chat.CommandProcessor": {"CommandProcessor": type("CommandProcessor", (), {})},
    "Chat.QueryParts": {"QueryParts": type("QueryParts", (), {})},
    "Chat.RAGChatImpl": {"RAGChatImpl": type("RAGChatImpl", (), {})},
    "Commons.Exceptions": {"LLMResultError": type("LLMResultError", (Exception,), {})},
    "Compliance.BannedPhraseCollector": {
        "BannedPhraseCollector": type("BannedPhraseCollector", (), {})
    },
    "Config.Config": {"Config": type("Config", (), {})},
    "Globals.Globals": {"Globals": type("Globals", (), {})},
    "Globals.Session": {"Session": type("Session", (), {})},
    "Gui.Colors": {"CYAN": "", "ORANGE": "", "RED": "", "RESET": ""},
    "Gui.PrettyWriter": {"PrettyWriter": type("PrettyWriter", (), {})},
    "Helpers.CSVWriter": {"CSVWriter": type("CSVWriter", (), {})},
    "Helpers.FileUtils": {"FileUtils": type("FileUtils", (), {})},
    "Helpers.Helpers": {"Helpers": type("Helpers", (), {})},
}

_STUBS = [
    "langchain_core",
    "langchain_core.prompts",
    "AI",
    "AI.AIHelpers",
    "AI.LLMCaller",
    "AI.ModelOutputAdapter",
    "AI.ModelsCache",
    "AI.TensorHelpers",
    "AI.TokenBudget",
    "Algos",
    "Algos.Masker",
    "Chat",
    "Chat.ChatContext",
    "Chat.CommandProcessor",
    "Chat.QueryParts",
    "Chat.RAGChatImpl",
    "Commons",
    "Commons.Exceptions",
    "Compliance",
    "Compliance.BannedPhraseCollector",
    "Config",
    "Config.Config",
    "Globals",
    "Globals.Globals",
    "Globals.Session",
    "Gui",
    "Gui.Colors",
    "Gui.PrettyWriter",
    "Helpers",
    "Helpers.CSVWriter",
    "Helpers.FileUtils",
    "Helpers.Helpers",
]


def _get_chatter_class():
    global _chatter_mod
    if _chatter_mod is not None:
        return _chatter_mod
    saved: dict[str, Any] = {}
    for name in _STUBS:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for attr, val in _STUB_ATTRS.get(name, {}).items():
                setattr(mod, attr, val)
            sys.modules[name] = mod
            saved[name] = None
    try:
        spec = importlib.util.spec_from_file_location("Chat.Chatter", _chatter_src)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        _chatter_mod = m.Chatter
        return _chatter_mod
    finally:
        for name in saved:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Typed stubs used in place of untyped inline lambdas
# ---------------------------------------------------------------------------


def _noop(*args: Any, **kwargs: Any) -> None:
    """No-op; used as a typed stand-in for atexit.register patches."""


def _noop_str(*args: Any) -> str:
    """Returns empty string; used as a typed stand-in for builtins.input patches."""
    return ""


# ---------------------------------------------------------------------------
# Minimal shell object — avoids __init__ entirely
# ---------------------------------------------------------------------------


def _pretty_write(*args: Any, **kwargs: Any) -> None:
    pass


def _cfg_get_str(key: str, default: str = "") -> str:
    return default


class _Shell:
    """Minimal object that carries only _supports_osc8 and _open_marked_documents."""

    pretty = types.SimpleNamespace(write=_pretty_write)
    cfg = types.SimpleNamespace(get_str=_cfg_get_str)

    @staticmethod
    def _supports_osc8() -> bool:
        Chatter = _get_chatter_class()
        return Chatter._supports_osc8()

    def _open_marked_documents(self, marked):
        Chatter = _get_chatter_class()
        # Bind the real method to self
        Chatter._open_marked_documents(self, marked)

    @staticmethod
    def _cleanup_marked_dir(path: str) -> None:
        pass  # suppress atexit cleanup in tests


# ===========================================================================
# _supports_osc8
# ===========================================================================

_OSC8_POSITIVE_ENVS: list[dict[str, str]] = [
    {"WT_SESSION": "some-uuid"},  # Windows Terminal
    {"TERM_PROGRAM": "iTerm.app"},
    {"TERM_PROGRAM": "vscode"},
    {"TERM_PROGRAM": "WezTerm"},
    {"TERM_PROGRAM": "JetBrains-JediTerm"},
    {"WEZTERM_EXECUTABLE": "/usr/bin/wezterm"},
    {"KITTY_WINDOW_ID": "1"},
    {"TERM": "xterm-kitty"},
    {"TERM": "xterm-ghostty"},
    {"TERM": "foot"},
    {"GHOSTTY_RESOURCES_DIR": "/usr/share/ghostty"},
    {"VTE_VERSION": "5000"},
    {"VTE_VERSION": "9999"},
]

_OSC8_NEGATIVE_ENVS: list[dict[str, str]] = [
    {},  # bare environment
    {"TERM": "xterm-256color"},  # generic xterm — no OSC 8
    {"TERM": "screen"},
    {"VTE_VERSION": "4999"},  # too old
    {"VTE_VERSION": "0"},
    {"VTE_VERSION": "bad"},  # non-numeric — shouldn't crash
    {"TERM_PROGRAM": "someunknownterminal"},
]


class TestSupportsOsc8:
    @pytest.mark.parametrize("env", _OSC8_POSITIVE_ENVS)
    def test_returns_true_for_known_osc8_terminals(self, env, monkeypatch):
        for key in (
            "WT_SESSION",
            "TERM_PROGRAM",
            "WEZTERM_EXECUTABLE",
            "KITTY_WINDOW_ID",
            "TERM",
            "GHOSTTY_RESOURCES_DIR",
            "VTE_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert _Shell._supports_osc8() is True

    @pytest.mark.parametrize("env", _OSC8_NEGATIVE_ENVS)
    def test_returns_false_for_unsupported_terminals(self, env, monkeypatch):
        for key in (
            "WT_SESSION",
            "TERM_PROGRAM",
            "WEZTERM_EXECUTABLE",
            "KITTY_WINDOW_ID",
            "TERM",
            "GHOSTTY_RESOURCES_DIR",
            "VTE_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert _Shell._supports_osc8() is False


# ===========================================================================
# _open_marked_documents — OSC 8 path
# ===========================================================================


class TestOpenMarkedDocumentsOsc8:
    def _make_marked(self, tmp_path: Path) -> list[tuple[str, bytes]]:
        return [
            (str(tmp_path / "doc1.pdf"), b"%PDF-1.4 fake"),
            (str(tmp_path / "doc2.pdf"), b"%PDF-1.4 fake2"),
        ]

    def test_osc8_path_prints_links_not_picker(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("WT_SESSION", "test-uuid")
        monkeypatch.setattr("atexit.register", _noop)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))

        out = capsys.readouterr().out
        # Plain file:// URI must be present
        assert "file://" in out
        # Picker prompt must NOT appear
        assert "Open file [" not in out

    def test_osc8_path_does_not_call_input(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WT_SESSION", "test-uuid")
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input() called on OSC 8 path"))  # type: ignore[reportUnknownParameterType]

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))

    def test_osc8_path_includes_filename_in_link_text(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("WT_SESSION", "test-uuid")
        monkeypatch.setattr("atexit.register", _noop)

        shell = _Shell()
        shell._open_marked_documents([(str(tmp_path / "Animals.pdf"), b"fake")])

        out = capsys.readouterr().out
        assert "Animals.pdf" in out


# ===========================================================================
# _open_marked_documents — picker path
# ===========================================================================


class TestOpenMarkedDocumentsPicker:
    def _clean_osc8_env(self, monkeypatch):
        for key in (
            "WT_SESSION",
            "TERM_PROGRAM",
            "WEZTERM_EXECUTABLE",
            "KITTY_WINDOW_ID",
            "TERM",
            "GHOSTTY_RESOURCES_DIR",
            "VTE_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)

    def _make_marked(self, tmp_path: Path) -> list[tuple[str, bytes]]:
        return [(str(tmp_path / "doc1.pdf"), b"fake")]

    def test_picker_path_shows_numbered_list(self, tmp_path, monkeypatch, capsys):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", _noop_str)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))

        out = capsys.readouterr().out
        assert "[1]" in out
        assert "doc1.pdf" in out

    def test_picker_no_osc8_escapes(self, tmp_path, monkeypatch, capsys):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", _noop_str)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))

        out = capsys.readouterr().out
        assert "\033]8;;" not in out

    def test_picker_enter_skips_without_opening(self, tmp_path, monkeypatch):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        opened: list[str] = []
        monkeypatch.setattr("builtins.input", _noop_str)
        import Chat.MarkedDocsViewer as _mdv

        def _record(p: Path) -> None:
            opened.append(str(p))

        monkeypatch.setattr(_mdv, "_open_with_os", _record)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))
        assert opened == []

    def test_picker_valid_choice_opens_file(self, tmp_path, monkeypatch):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        opened: list[str] = []

        responses = iter(["1", ""])  # choose 1, then Enter to exit
        monkeypatch.setattr("builtins.input", lambda *a: next(responses))  # type: ignore[reportUnknownParameterType]
        import Chat.MarkedDocsViewer as _mdv

        def _record(p: Path) -> None:
            opened.append(str(p))

        monkeypatch.setattr(_mdv, "_open_with_os", _record)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))
        assert len(opened) == 1
        assert "doc1_marked.pdf" in opened[0]

    def test_picker_invalid_then_valid_choice(self, tmp_path, monkeypatch, capsys):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        opened: list[str] = []

        responses = iter(["99", "abc", "1", ""])
        monkeypatch.setattr("builtins.input", lambda *a: next(responses))  # type: ignore[reportUnknownParameterType]
        import Chat.MarkedDocsViewer as _mdv

        def _record(p: Path) -> None:
            opened.append(str(p))

        monkeypatch.setattr(_mdv, "_open_with_os", _record)

        shell = _Shell()
        shell._open_marked_documents(self._make_marked(tmp_path))
        assert len(opened) == 1

    def test_picker_eof_exits_gracefully(self, tmp_path, monkeypatch):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))  # type: ignore[reportUnknownParameterType]

        shell = _Shell()
        # Should not raise
        shell._open_marked_documents(self._make_marked(tmp_path))

    def test_txt_converted_note_shown(self, tmp_path, monkeypatch, capsys):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", _noop_str)

        shell = _Shell()
        shell._open_marked_documents([(str(tmp_path / "notes.txt"), b"plain text")])

        out = capsys.readouterr().out
        assert "txt.md" in out
        assert "notes.txt" in out

    def test_empty_marked_list_does_nothing(self, tmp_path, monkeypatch):
        self._clean_osc8_env(monkeypatch)
        monkeypatch.setattr("atexit.register", _noop)
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("input() called on empty list"))  # type: ignore[reportUnknownParameterType]

        shell = _Shell()
        shell._open_marked_documents([])  # should return immediately


# ===========================================================================
# _cleanup_dir — jailbreak guard and existence check
# ===========================================================================


class TestCleanupDir:
    def test_raises_when_project_root_is_empty(self, tmp_path):
        import Chat.MarkedDocsViewer as _mdv

        d = tmp_path / "rag_marked_test"
        d.mkdir()
        with pytest.raises(RuntimeError, match="project_root is not set"):
            _mdv._cleanup_dir(str(d), project_root="")

    def test_raises_when_project_root_is_missing(self, tmp_path):
        import Chat.MarkedDocsViewer as _mdv

        d = tmp_path / "rag_marked_test"
        d.mkdir()
        with pytest.raises(RuntimeError, match="project_root is not set"):
            _mdv._cleanup_dir(str(d))

    def test_raises_when_path_outside_project_root(self, tmp_path):
        import Chat.MarkedDocsViewer as _mdv

        project_root = tmp_path / "project"
        project_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        with pytest.raises(RuntimeError, match="outside the project root"):
            _mdv._cleanup_dir(str(outside), project_root=str(project_root))

    def test_raises_when_path_does_not_exist(self, tmp_path):
        import Chat.MarkedDocsViewer as _mdv

        project_root = tmp_path / "project"
        project_root.mkdir()
        missing = project_root / "rag_marked_gone"
        with pytest.raises(RuntimeError, match="does not exist"):
            _mdv._cleanup_dir(str(missing), project_root=str(project_root))

    def test_deletes_dir_within_project_root(self, tmp_path):
        import Chat.MarkedDocsViewer as _mdv

        project_root = tmp_path / "project"
        project_root.mkdir()
        d = project_root / "tmp" / "rag_marked_xyz"
        d.mkdir(parents=True)
        (d / "file.pdf").write_bytes(b"data")
        _mdv._cleanup_dir(str(d), project_root=str(project_root))
        assert not d.exists()
