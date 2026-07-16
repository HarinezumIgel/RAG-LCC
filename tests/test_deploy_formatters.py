import os
import subprocess
from pathlib import Path

import deploy.scripts.deploy as deploy_module


def test_run_formatters_uses_existing_dev_python_without_pip_install(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_run(command, cwd=None, capture=False):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(deploy_module, "run", fake_run)
    monkeypatch.setattr(
        deploy_module, "module_available", lambda *_args, **_kwargs: True
    )

    dev_python = (
        tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    dev_python.parent.mkdir(parents=True)
    dev_python.write_text("", encoding="utf-8")

    src_dir = tmp_path / "src"
    tests_dir = tmp_path / "tests"

    deploy_module.run_formatters(
        project_path=tmp_path, src_dir=src_dir, tests_dir=tests_dir
    )

    assert calls == [
        [str(dev_python), "-m", "black", str(src_dir), str(tests_dir)],
        [str(dev_python), "-m", "isort", str(src_dir)],
    ]
    assert all("install" not in command for command in calls)


def test_run_formatters_installs_missing_tools_in_dev_env(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_run(command, cwd=None, capture=False):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(deploy_module, "run", fake_run)
    monkeypatch.setattr(
        deploy_module, "module_available", lambda *_args, **_kwargs: False
    )

    dev_python = (
        tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    dev_python.parent.mkdir(parents=True)
    dev_python.write_text("", encoding="utf-8")

    src_dir = tmp_path / "src"
    tests_dir = tmp_path / "tests"

    deploy_module.run_formatters(
        project_path=tmp_path, src_dir=src_dir, tests_dir=tests_dir
    )

    assert calls[0] == [
        str(dev_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "black",
        "isort",
    ]
    assert calls[1:] == [
        [str(dev_python), "-m", "black", str(src_dir), str(tests_dir)],
        [str(dev_python), "-m", "isort", str(src_dir)],
    ]
