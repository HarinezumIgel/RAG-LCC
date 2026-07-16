# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportReturnType=false
"""
Tests for FileUtils.delete_file_or_dir — path jailbreak guard.

All operations are scoped to pytest tmp_path; nothing outside that
directory is ever created or deleted.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.FileUtils import FileUtils

# ---------------------------------------------------------------------------
# Minimal stubs — no heavy deps needed
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[object, ...]] = []

    def write(self, *a, **kw):
        self.messages.append((*a, kw))

    def logged_labels(self) -> list[str]:
        """Return all 'label' (second positional arg) strings that were written."""
        return [m[1] for m in self.messages if len(m) > 1]


class StubConfig:
    def __init__(self, project_root: str):
        self._root = project_root

    def get_str(self, key: str, default: str = "") -> str:
        if key == "_ABSOLUTE_PATH":
            return self._root
        return default

    def get(self, key, default=None):
        return default


def _make_file_utils(project_root: str):
    """
    Construct a FileUtils with injected stubs, bypassing the heavy __init__
    (NLTK, stopwords, etc.).  Only the attributes used by delete_file_or_dir
    are set.
    """
    fu = object.__new__(FileUtils)
    fu.pretty = StubPrettyWriter()
    fu.cfg = StubConfig(project_root)
    return fu


# ---------------------------------------------------------------------------
# Safety fixture — patches the two real deletion calls used inside
# delete_file_or_dir so that if the guard ever has a bug and lets a
# dangerous path through, nothing on disk is actually touched.
# Applied to every guard test via autouse on the class or explicit use.
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_real_deletion(monkeypatch):
    """Monkeypatch shutil.rmtree and os.remove inside FileUtils so they
    raise AssertionError if ever reached.  Proves the guard blocked the
    call AND prevents accidental damage if the guard has a bug."""
    import Helpers.FileUtils as fu_mod

    def _safe_rmtree(path, *a, **kw):
        raise AssertionError(f"shutil.rmtree reached with '{path}' — guard failed!")

    def _safe_remove(path):
        raise AssertionError(f"os.remove reached with '{path}' — guard failed!")

    monkeypatch.setattr(fu_mod.shutil, "rmtree", _safe_rmtree)
    monkeypatch.setattr(fu_mod.os, "remove", _safe_remove)


# ---------------------------------------------------------------------------
# Guard: empty path
# ---------------------------------------------------------------------------


class TestEmptyPath:
    def test_returns_false_and_logs_error(self, tmp_path, no_real_deletion):
        fu = _make_file_utils(str(tmp_path))
        result = fu.delete_file_or_dir("")
        assert result is False
        assert any("E" in m[0] for m in fu.pretty.messages)


# ---------------------------------------------------------------------------
# Guard: unresolvable project root
# ---------------------------------------------------------------------------


class TestUnresolvableProjectRoot:
    def test_refuses_when_absolute_path_is_empty(self, tmp_path, no_real_deletion):
        """If _ABSOLUTE_PATH returns empty string the guard must refuse."""
        fu = object.__new__(FileUtils)
        fu.pretty = StubPrettyWriter()

        class EmptyRootConfig:
            def get_str(self, key, default=""):
                return ""  # simulate missing config

            def get(self, key, default=None):
                return default

        fu.cfg = EmptyRootConfig()
        target = str(tmp_path / "safe_subdir")
        os.makedirs(target)

        result = fu.delete_file_or_dir(target)

        assert result is False
        assert target not in fu.pretty.logged_labels()  # not logged as deleted
        assert os.path.isdir(target), "directory must not have been deleted"


class TestDriveRootProjectRoot:
    """Guard: _ABSOLUTE_PATH itself resolves to a drive or filesystem root."""

    @pytest.mark.parametrize("bad_root", ["C:/", "C:\\", "/"])
    def test_refuses_when_project_root_is_drive_root(
        self, tmp_path, no_real_deletion, bad_root
    ):
        """Deletion must be refused if the configured project root is a drive root."""
        fu = object.__new__(FileUtils)
        fu.pretty = StubPrettyWriter()
        fu.cfg = StubConfig(bad_root)

        target = str(tmp_path / "subdir")
        os.makedirs(target)

        result = fu.delete_file_or_dir(target)

        assert result is False
        assert os.path.isdir(target), "directory must not have been deleted"


# ---------------------------------------------------------------------------
# Guard: drive-root / very-short paths
# ---------------------------------------------------------------------------


class TestDriveRootPaths:
    @pytest.mark.parametrize(
        "dangerous_path",
        [
            "C:\\",
            "C:/",
            "/",
            "D:\\",
        ],
    )
    def test_refuses_drive_or_fs_root(self, tmp_path, no_real_deletion, dangerous_path):
        fu = _make_file_utils(str(tmp_path))
        result = fu.delete_file_or_dir(dangerous_path)
        assert result is False
        assert "Path Guard" in fu.pretty.logged_labels()


# ---------------------------------------------------------------------------
# Guard: path outside project root
# ---------------------------------------------------------------------------


class TestPathOutsideProjectRoot:
    def test_refuses_sibling_directory(self, tmp_path, no_real_deletion):
        """A sibling of tmp_path must be blocked, even if it exists."""
        project = tmp_path / "project"
        sibling = tmp_path / "other_dir"
        project.mkdir()
        sibling.mkdir()

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(sibling))

        assert result is False
        assert "Path Guard" in fu.pretty.logged_labels()
        assert sibling.is_dir(), "sibling directory must not have been deleted"

    def test_refuses_parent_directory(self, tmp_path, no_real_deletion):
        """The parent of the project root must be blocked."""
        project = tmp_path / "project"
        project.mkdir()

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(tmp_path))

        assert result is False
        assert "Path Guard" in fu.pretty.logged_labels()
        assert tmp_path.is_dir(), "parent directory must not have been deleted"

    def test_refuses_path_traversal_attempt(self, tmp_path, no_real_deletion):
        """Path traversal via '../../' must be caught after normpath."""
        project = tmp_path / "project"
        victim = tmp_path / "victim"
        project.mkdir()
        victim.mkdir()

        # Construct a path that starts inside the project but escapes via ..
        traversal = str(project / ".." / "victim")

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(traversal)

        assert result is False
        assert "Path Guard" in fu.pretty.logged_labels()
        assert victim.is_dir(), "victim directory must not have been deleted"

    def test_refuses_prefix_trick(self, tmp_path, no_real_deletion):
        """
        A path whose string starts with the project root but is not actually
        inside it (e.g. /project_evil vs /project).
        """
        project = tmp_path / "project"
        trick = tmp_path / "project_evil"
        project.mkdir()
        trick.mkdir()

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(trick))

        assert result is False
        assert "Path Guard" in fu.pretty.logged_labels()
        assert trick.is_dir(), "trick directory must not have been deleted"


# ---------------------------------------------------------------------------
# Happy path: inside project root
# ---------------------------------------------------------------------------


class TestInsideProjectRoot:
    def test_deletes_file_inside_project(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        target_file = project / "data" / "store.bin"
        target_file.parent.mkdir(parents=True)
        target_file.write_bytes(b"data")

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(target_file))

        assert result is True
        assert not target_file.exists()

    def test_deletes_directory_inside_project(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        target_dir = project / "chromadb" / "bm25" / "Test"
        target_dir.mkdir(parents=True)
        (target_dir / "index.pkl").write_bytes(b"index")

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(target_dir))

        assert result is True
        assert not target_dir.exists()

    def test_returns_true_when_path_does_not_exist(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        nonexistent = project / "chromadb" / "ghost"

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(nonexistent))

        assert result is True

    def test_deletes_project_root_itself(self, tmp_path):
        """Deleting exactly the project root is permitted by the guard (abs_fp == project_root)."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "file.txt").write_text("hi")

        fu = _make_file_utils(str(project))
        result = fu.delete_file_or_dir(str(project))

        assert result is True
        assert not project.exists()
