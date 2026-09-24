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
    assert_app_project_root,
    assert_not_drive_root,
    assert_script_project_root,
    drive_root_message,
    is_path_like_slot,
    is_drive_root,
    resolve_guard_path,
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


class StubIndirectRootConfig:
    def get_str(self, key: str, default: str = "", **_kw) -> str:
        if key == "_ABSOLUTE_PATH":
            return "$_ROOT_ALIAS"
        return default

    def get(self, key, default=None):
        if key == "_ABSOLUTE_PATH":
            return "$_ROOT_ALIAS"
        return default, key


def _raise_system_exit(code: int = 1) -> None:
    raise SystemExit(code)


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

    @pytest.mark.parametrize(
        "candidate",
        ["C:/", "C:\\", "c:/", "c:", "c::/", "C::\\"],
    )
    def test_windows_root_like_paths_are_rejected_cross_platform(self, candidate):
        assert is_drive_root(candidate) is True

    @pytest.mark.parametrize("candidate", ["C:/RAG-LCC", "C:\\RAG-LCC", "c::/RAG-LCC"])
    def test_windows_named_paths_are_not_drive_roots(self, candidate):
        assert is_drive_root(candidate) is False

    def test_resolve_guard_path_preserves_windows_drive_inputs(self, tmp_path):
        resolved = resolve_guard_path("c::/", base_dir=str(tmp_path))
        assert is_drive_root(resolved) is True

    def test_traversal_that_collapses_to_root_is_blocked(self):
        assert is_drive_root("..", base_dir=_FS_ROOT) is True
        assert is_drive_root("../../..", base_dir=_FS_ROOT) is True

    def test_can_skip_traversal_resolution(self):
        assert is_drive_root("../../..", resolve_traversal=False) is False

    def test_traversal_that_stays_non_root_remains_allowed(self, tmp_path):
        assert is_drive_root("..", base_dir=str(tmp_path)) is False


class TestPathSlotHeuristics:
    def test_bare_dotdot_is_treated_as_path_for_path_hinted_slot(self):
        assert is_path_like_slot("LOG_FILE", "..") is True

    def test_bare_windows_drive_designator_is_path_for_path_hinted_slot(self):
        assert is_path_like_slot("LOG_FILE", "c:") is True

    def test_bare_dot_is_treated_as_path_for_path_hinted_slot(self):
        assert is_path_like_slot("DOC_DIR", ".") is True

    def test_bare_dotdot_stays_non_path_for_non_path_slot(self):
        assert is_path_like_slot("MODEL", "..") is False


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


class TestAssertScriptProjectRoot:
    def test_returns_script_derived_root_when_config_matches(self, tmp_path):
        script = tmp_path / "src" / "Scripts" / "probe.py"
        derived = assert_script_project_root(
            str(script),
            configured_project_root=str(tmp_path),
        )
        assert derived == str(tmp_path)

    def test_rejects_when_script_not_under_src_scripts(self, tmp_path):
        script = tmp_path / "Scripts" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_script_project_root(
                str(script), configured_project_root=str(tmp_path)
            )
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_drive_root(self, tmp_path):
        script = tmp_path / "src" / "Scripts" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_script_project_root(str(script), configured_project_root=_FS_ROOT)
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_backslash_root_like(self, tmp_path):
        script = tmp_path / "src" / "Scripts" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_script_project_root(str(script), configured_project_root="\\")
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_unrelated(self, tmp_path):
        script_root = tmp_path / "repo_a"
        script = script_root / "src" / "Scripts" / "probe.py"
        unrelated = tmp_path / "repo_b"
        with pytest.raises(SystemExit) as exc:
            assert_script_project_root(
                str(script),
                configured_project_root=str(unrelated),
            )
        assert exc.value.code == 1

    def test_allows_when_derived_root_is_nested_under_config_root(self, tmp_path):
        script_root = tmp_path / "repo_a"
        script = script_root / "src" / "Scripts" / "probe.py"
        derived = assert_script_project_root(
            str(script),
            configured_project_root=str(tmp_path),
        )
        assert derived == str(script_root)

    def test_rejects_when_cwd_mismatch_required(self, tmp_path, monkeypatch):
        script_root = tmp_path / "repo_a"
        script = script_root / "src" / "Scripts" / "probe.py"
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        with pytest.raises(SystemExit) as exc:
            assert_script_project_root(
                str(script),
                configured_project_root=str(tmp_path),
                require_cwd_match=True,
            )

        assert exc.value.code == 1

    def test_allows_when_cwd_matches_configured_root(self, tmp_path, monkeypatch):
        script_root = tmp_path / "repo_a"
        script = script_root / "src" / "Scripts" / "probe.py"
        monkeypatch.chdir(tmp_path)

        derived = assert_script_project_root(
            str(script),
            configured_project_root=str(tmp_path),
            require_cwd_match=True,
        )
        assert derived == str(script_root)


class TestAssertAppProjectRoot:
    def test_returns_app_derived_root_when_config_matches(self, tmp_path):
        app = tmp_path / "src" / "Apps" / "probe.py"
        derived = assert_app_project_root(
            str(app),
            configured_project_root=str(tmp_path),
        )
        assert derived == str(tmp_path)

    def test_rejects_when_app_not_under_src_apps(self, tmp_path):
        app = tmp_path / "Apps" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_app_project_root(str(app), configured_project_root=str(tmp_path))
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_drive_root(self, tmp_path):
        app = tmp_path / "src" / "Apps" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_app_project_root(str(app), configured_project_root=_FS_ROOT)
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_backslash_root_like(self, tmp_path):
        app = tmp_path / "src" / "Apps" / "probe.py"
        with pytest.raises(SystemExit) as exc:
            assert_app_project_root(str(app), configured_project_root="\\")
        assert exc.value.code == 1

    def test_rejects_when_config_root_is_unrelated(self, tmp_path):
        app_root = tmp_path / "repo_a"
        app = app_root / "src" / "Apps" / "probe.py"
        unrelated = tmp_path / "repo_b"
        with pytest.raises(SystemExit) as exc:
            assert_app_project_root(
                str(app),
                configured_project_root=str(unrelated),
            )
        assert exc.value.code == 1

    def test_allows_when_derived_root_is_nested_under_config_root(self, tmp_path):
        app_root = tmp_path / "repo_a"
        app = app_root / "src" / "Apps" / "probe.py"
        derived = assert_app_project_root(
            str(app),
            configured_project_root=str(tmp_path),
        )
        assert derived == str(app_root)


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
            staticmethod(_raise_system_exit),
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
            staticmethod(_raise_system_exit),
        )
        monkeypatch.chdir(tmp_path)
        cfg = StubConfig(_FS_ROOT)
        with pytest.raises(SystemExit):
            StartupCommons._ensure_safe_startup_root(cfg)

    def test_aborts_when_configured_root_uses_indirection(self, tmp_path, monkeypatch):
        from Commons.StartupCommons import StartupCommons

        monkeypatch.setattr(
            StartupCommons,
            "_die",
            staticmethod(_raise_system_exit),
        )
        monkeypatch.chdir(tmp_path)
        cfg = StubIndirectRootConfig()
        with pytest.raises(SystemExit):
            StartupCommons._ensure_safe_startup_root(cfg)
