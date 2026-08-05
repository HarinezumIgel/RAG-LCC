import subprocess
from pathlib import Path

import pytest

pytest.importorskip("deploy")
import deploy.scripts.install_required as install_required
import deploy.scripts.show_3rd_party_licenses as show_licenses


def test_scan_imports_uses_target_python(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, *, capture_output=True, cwd=None):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="spacy\n", stderr="")

    monkeypatch.setattr(install_required, "run", fake_run)

    scan_script = tmp_path / "scan_imports.py"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target_python = tmp_path / "venv" / "bin" / "python"

    install_required.scan_imports(scan_script, src_dir, python_exe=target_python)

    assert calls == [[str(target_python), str(scan_script), str(src_dir)]]


def test_generate_license_report_delegates_to_license_helper(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_generate_license_report(*, project_path, venv_name, containing_license_directory_name):
        calls.append((project_path, venv_name, containing_license_directory_name))

    monkeypatch.setattr(show_licenses, "generate_license_report", fake_generate_license_report)

    install_required.generate_license_report(tmp_path, tmp_path / "venv", tmp_path / "licenses")

    assert calls == [(tmp_path, "venv", "licenses")]
