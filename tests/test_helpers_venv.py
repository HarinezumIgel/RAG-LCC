import sys

import pytest

from Commons.Exceptions import NoVirtualEnvError
from Helpers.Helpers import Helpers


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
