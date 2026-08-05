import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if not (Path(__file__).resolve().parents[1] / "deploy").exists():
    pytest.skip("deploy directory not present", allow_module_level=True)


def test_ensure_classgraph_dependencies_installs_pip_packages(monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[1] / "deploy" / "scripts" / "CreateClassGraphs.py"
    calls = []

    def fake_run(command, capture_output=True, text=True):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_find_spec(name: str):
        return None if name in ("graphviz", "pylint") else object()

    def fake_which(name: str):
        if name == "dot":
            return "/usr/bin/dot"
        return "/usr/bin/" + name

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(shutil, "which", fake_which)

    spec = importlib.util.spec_from_file_location("create_classgraphs_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == [[sys.executable, "-m", "pip", "install", "--upgrade", "graphviz", "pylint"]]
    assert os.environ["PATH"].startswith("/usr/bin")
