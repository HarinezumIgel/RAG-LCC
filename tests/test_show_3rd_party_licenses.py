from pathlib import Path

import pytest

pytest.importorskip("deploy")
from deploy.scripts.show_3rd_party_licenses import _normalize_license_file_path


def test_normalize_license_file_path_replaces_venv_prefix(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    license_file = str(venv_path / "Lib" / "site-packages" / "demo" / "LICENSE")

    normalized = _normalize_license_file_path(license_file, venv_path)

    assert normalized == "<your-venv>/Lib/site-packages/demo/LICENSE"


def test_normalize_license_file_path_keeps_unknown_paths_unchanged(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    license_file = str(tmp_path / "somewhere" / "LICENSE.txt")

    assert _normalize_license_file_path(license_file, venv_path) == license_file
