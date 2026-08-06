import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("deploy")
import deploy.scripts.deploy as deploy


def test_should_run_install_helper_returns_true_without_requirements_file(
    tmp_path: Path,
) -> None:
    venv_path = tmp_path / ".venv" / "bin"
    venv_path.mkdir(parents=True)
    (venv_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    assert deploy.should_run_install_helper(tmp_path, ".venv") is True


def test_should_run_install_helper_returns_false_when_requirements_are_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv_path = tmp_path / ".venv" / "bin"
    venv_path.mkdir(parents=True)
    (venv_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()
    (requirements_dir / "requirements_final.txt").write_text(
        "pytest\nsentencepiece>=0.2.1\n", encoding="utf-8"
    )

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], capture_output: bool = True, text: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="Version: 1.0\n", stderr=""
        )

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)

    assert deploy.should_run_install_helper(tmp_path, ".venv") is False
    assert calls == [
        [str(venv_path / "python"), "-m", "pip", "show", "pytest"],
        [str(venv_path / "python"), "-m", "pip", "show", "sentencepiece"],
    ]


def test_should_run_install_helper_returns_true_when_any_requirement_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv_path = tmp_path / ".venv" / "bin"
    venv_path.mkdir(parents=True)
    (venv_path / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()
    (requirements_dir / "requirements_final.txt").write_text(
        "pytest\n", encoding="utf-8"
    )

    def fake_run(
        command: list[str], capture_output: bool = True, text: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not found")

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)

    assert deploy.should_run_install_helper(tmp_path, ".venv") is True


def test_should_recreate_target_venv_uses_the_cleanup_flags() -> None:
    assert (
        deploy.should_recreate_target_venv(
            skip_module_install=False, no_target_env_cleanup=False
        )
        is True
    )
    assert (
        deploy.should_recreate_target_venv(
            skip_module_install=True, no_target_env_cleanup=False
        )
        is False
    )
    assert (
        deploy.should_recreate_target_venv(
            skip_module_install=False, no_target_env_cleanup=True
        )
        is False
    )
    assert (
        deploy.should_recreate_target_venv(
            skip_module_install=True, no_target_env_cleanup=True
        )
        is False
    )


def test_ensure_classgraph_tools_skips_install_when_tools_are_already_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev_python = tmp_path / ".venv" / "bin" / "python"
    dev_python.parent.mkdir(parents=True)
    dev_python.write_text("#!/bin/sh\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_module_available(python_exe: Path, module_name: str) -> bool:
        return True

    def fake_run(
        command: list[str], capture_output: bool = True, text: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(deploy, "module_available", fake_module_available)
    monkeypatch.setattr(deploy, "run", fake_run)

    assert deploy.ensure_classgraph_tools(tmp_path) == dev_python
    assert calls == []


def test_ensure_classgraph_tools_installs_missing_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dev_python = tmp_path / ".venv" / "bin" / "python"
    dev_python.parent.mkdir(parents=True)
    dev_python.write_text("#!/bin/sh\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_module_available(python_exe: Path, module_name: str) -> bool:
        return False

    def fake_run(
        command: list[str], capture_output: bool = True, text: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(deploy, "module_available", fake_module_available)
    monkeypatch.setattr(deploy, "run", fake_run)

    assert deploy.ensure_classgraph_tools(tmp_path) == dev_python
    assert calls == [
        [str(dev_python), "-m", "pip", "install", "--upgrade", "graphviz", "pylint"]
    ]
