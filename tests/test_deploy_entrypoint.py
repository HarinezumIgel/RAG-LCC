import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("deploy")
import deploy.scripts.deploy as deploy_entry


def test_main_exits_with_error_when_project_not_found(
    monkeypatch, tmp_path: Path
) -> None:
    """main() should return 1 immediately when --project-path does not exist."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["deploy.py", "--project-path", str(tmp_path / "nonexistent")],
    )

    exit_code = deploy_entry.main()

    assert exit_code == 1
