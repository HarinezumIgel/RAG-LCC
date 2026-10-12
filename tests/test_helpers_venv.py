import sys
from pathlib import Path

import pytest

from Configuration import Config_Global
from Commons.Exceptions import NoVirtualEnvError
from Helpers.Helpers import Helpers


class _ProjectConfig:
    def __init__(self, project_root: Path):
        self.project_root = str(project_root)

    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        if key == "_ABSOLUTE_PATH":
            return self.project_root
        return default


def _helpers_for_project(project_root: Path) -> Helpers:
    helpers = Helpers.__new__(Helpers)
    helpers.cfg = _ProjectConfig(project_root)
    return helpers


class TestHelpersVenvGuard:
    def test_is_in_venv_raises_when_not_in_virtual_environment(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "base_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "exec_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(
            sys, "base_exec_prefix", "/tmp/system-prefix", raising=False
        )
        monkeypatch.delattr(sys, "real_prefix", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

        with pytest.raises(NoVirtualEnvError):
            Helpers().is_in_venv(required=True)

    def test_is_in_venv_can_return_false_without_raising(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "base_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "exec_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(
            sys, "base_exec_prefix", "/tmp/system-prefix", raising=False
        )
        monkeypatch.delattr(sys, "real_prefix", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

        assert Helpers().is_in_venv(required=False) is False

    def test_is_in_venv_detects_virtual_env_env_var(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "base_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(sys, "exec_prefix", "/tmp/system-prefix", raising=False)
        monkeypatch.setattr(
            sys, "base_exec_prefix", "/tmp/system-prefix", raising=False
        )
        monkeypatch.delattr(sys, "real_prefix", raising=False)
        monkeypatch.setenv("VIRTUAL_ENV", "/workspaces/RAG-LCC/.venv")
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

        assert Helpers().is_in_venv(required=True) is True


class TestArgosRuntimeDirectories:
    def test_uses_project_local_cache_directories(self, tmp_path, monkeypatch):
        project_root = tmp_path / "RAG-LCC"
        project_root.mkdir()
        monkeypatch.setattr(Config_Global, "_ABSOLUTE_PATH", str(project_root))

        _helpers_for_project(project_root).prepare_argos_runtime_dirs()

        assert Path(__import__("os").environ["ARGOS_PACKAGES_DIR"]) == (
            project_root / ".caches" / "argos" / "data" / "packages"
        )
        assert (project_root / ".caches" / "argos" / "config").is_dir()
        assert (project_root / ".caches" / "argos" / "cache").is_dir()

    def test_rejects_cache_path_resolving_outside_project(self, tmp_path, monkeypatch):
        project_root = tmp_path / "RAG-LCC"
        project_root.mkdir()
        outside_root = tmp_path / "outside"
        outside_root.mkdir()
        monkeypatch.setattr(Config_Global, "_ABSOLUTE_PATH", str(project_root))

        original_resolve = Path.resolve

        def _resolve_with_escape(path: Path, strict: bool = False) -> Path:
            if path == project_root / ".caches" / "argos" / "data":
                return outside_root / "data"
            return original_resolve(path, strict=strict)

        monkeypatch.setattr(Path, "resolve", _resolve_with_escape)

        with pytest.raises(RuntimeError, match="escapes the project root"):
            _helpers_for_project(project_root).prepare_argos_runtime_dirs()
