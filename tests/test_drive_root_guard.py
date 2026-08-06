# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportReturnType=false
"""
Tests for the drive/filesystem-root guards that refuse to run when the project
root collapses to a drive root (e.g. 'C:\\' or '/'), where the deletion
path-guards would be disabled.

Covered (real implementations — no reimplemented copies):
  * Commons.DriveRootGuard.is_drive_root / assert_not_drive_root / drive_root_message
  * Helpers.Helpers.is_in_drive_root
  * Commons.StartupCommons._ensure_started_from_project_root

Everything runs in-process (fast). The drive/filesystem root is derived from
this file's own drive, so the suite is identical on Windows and POSIX.
"""

import os

import pytest

from Commons.DriveRootGuard import (
    assert_not_drive_root,
    drive_root_message,
    is_drive_root,
)
from Commons.Exceptions import DriveRootExecutionError

# Platform-appropriate drive/filesystem root ('C:\' / 'D:\' … or '/').
_DRIVE, _ = os.path.splitdrive(os.path.abspath(__file__))
_FS_ROOT = (_DRIVE + os.sep) if _DRIVE else os.sep


# ---------------------------------------------------------------------------
# Lightweight stubs — avoid the heavy Helpers.__init__ (NLTK, torch, …).
# ---------------------------------------------------------------------------
class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[object, ...]] = []

    def write(self, *a, **kw):
        self.messages.append((*a, kw))


class StubConfig:
    def __init__(self, root: str):
        self._root = root

    def get_str(self, key: str, default: str = "", **_kw) -> str:
        return self._root if key == "_ABSOLUTE_PATH" else default

    def get(self, key, default=None):
        return self._root if key == "_ABSOLUTE_PATH" else default


def _make_helpers(root: str):
    """Construct a Helpers instance with injected stubs, bypassing __init__."""
    from Helpers.Helpers import Helpers

    h = object.__new__(Helpers)
    h.cfg = StubConfig(root)
    h.pretty = StubPrettyWriter()
    return h


# ---------------------------------------------------------------------------
# Commons.DriveRootGuard.is_drive_root
# ---------------------------------------------------------------------------
class TestIsDriveRoot:
    def test_empty_is_drive_root(self):
        assert is_drive_root("") is True

    def test_filesystem_root_is_drive_root(self):
        assert is_drive_root(_FS_ROOT) is True

    def test_project_dir_is_not_drive_root(self, tmp_path):
        assert is_drive_root(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Commons.DriveRootGuard.assert_not_drive_root
#   Resolves the project root as two levels above the given script_file
#   (the src/Scripts/X.py convention).
# ---------------------------------------------------------------------------
class TestAssertNotDriveRoot:
    def test_aborts_when_two_levels_up_is_drive_root(self):
        script = os.path.join(_FS_ROOT, "src", "Scripts", "probe.py")
        with pytest.raises(SystemExit) as exc:
            assert_not_drive_root(script)
        assert exc.value.code == 1

    def test_allows_when_two_levels_up_is_a_named_dir(self, tmp_path):
        script = tmp_path / "src" / "Scripts" / "probe.py"
        assert assert_not_drive_root(str(script)) is None


class TestDriveRootMessage:
    def test_message_mentions_path_and_is_a_refusal(self):
        msg = drive_root_message(_FS_ROOT)
        assert "EXECUTION BLOCKED" in msg
        assert _FS_ROOT in msg

    def test_message_handles_unresolved_root(self):
        assert "<UNRESOLVED>" in drive_root_message("")


# ---------------------------------------------------------------------------
# Helpers.Helpers.is_in_drive_root
# ---------------------------------------------------------------------------
class TestHelpersIsInDriveRoot:
    def test_raises_when_required_and_root_is_drive_root(self):
        with pytest.raises(DriveRootExecutionError):
            _make_helpers(_FS_ROOT).is_in_drive_root(required=True)

    def test_returns_true_without_raising_when_not_required(self):
        assert _make_helpers(_FS_ROOT).is_in_drive_root(required=False) is True

    def test_returns_false_for_named_project_dir(self, tmp_path):
        assert _make_helpers(str(tmp_path)).is_in_drive_root(required=True) is False

    def test_unresolved_root_counts_as_drive_root(self):
        assert _make_helpers("").is_in_drive_root(required=False) is True


# ---------------------------------------------------------------------------
# Commons.StartupCommons startup path guards
#   _ensure_started_from_project_root compares configured root vs cwd.
#   _ensure_safe_startup_root additionally rejects drive roots.
# ---------------------------------------------------------------------------
class TestEnsureStartedFromProjectRoot:
    def test_passes_when_cwd_equals_configured_root(self, tmp_path, monkeypatch):
        from Commons.StartupCommons import StartupCommons

        monkeypatch.chdir(tmp_path)
        cfg = StubConfig(str(tmp_path))
        assert StartupCommons._ensure_started_from_project_root(cfg) is None

    def test_aborts_when_cwd_differs_from_configured_root(self, tmp_path, monkeypatch):
        from Commons.StartupCommons import StartupCommons

        # _die() normally calls os._exit(); swap it for a catchable SystemExit.
        monkeypatch.setattr(
            StartupCommons,
            "_die",
            staticmethod(lambda code=1: (_ for _ in ()).throw(SystemExit(code))),
        )
        monkeypatch.chdir(tmp_path)
        cfg = StubConfig(str(tmp_path / "elsewhere"))
        with pytest.raises(SystemExit):
            StartupCommons._ensure_started_from_project_root(cfg)

    def test_aborts_when_configured_root_is_drive_root(self, tmp_path, monkeypatch):
        from Commons.StartupCommons import StartupCommons

        monkeypatch.setattr(
            StartupCommons,
            "_die",
            staticmethod(lambda code=1: (_ for _ in ()).throw(SystemExit(code))),
        )
        monkeypatch.chdir(tmp_path)
        cfg = StubConfig(_FS_ROOT)
        with pytest.raises(SystemExit):
            StartupCommons._ensure_safe_startup_root(cfg)
