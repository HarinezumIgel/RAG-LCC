import subprocess
from pathlib import Path

import deploy.scripts.install_required as install_required


def test_resolve_install_spec_uses_windows_marker_for_win32com() -> None:
    assert (
        install_required.resolve_install_spec("win32com")
        == 'pywin32; sys_platform == "win32"'
    )


def test_resolve_package_name_keeps_plain_name_for_requirements() -> None:
    assert install_required.resolve_package_name("win32com") == "pywin32"


def test_install_packages_streams_output(monkeypatch, tmp_path: Path) -> None:
    python_exe = tmp_path / "venv" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("#!/bin/sh\n", encoding="utf-8")

    calls = []

    def fake_run(command, *, capture_output=True, cwd=None, stdout=None, stderr=None):
        calls.append((command, capture_output))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(install_required, "run", fake_run)

    install_required.install_packages(python_exe, ["argostranslate"])

    assert calls == [
        ([str(python_exe), "-m", "pip", "install", "argostranslate"], False)
    ]
