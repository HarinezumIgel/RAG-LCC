import shutil
from pathlib import Path

import pytest

pytest.importorskip("deploy")
import deploy.scripts.deploy as deploy


def test_copy_tree_falls_back_when_copy2_is_not_permitted(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"
    source.mkdir()
    file_path = source / "alpha.txt"
    file_path.write_text("hello", encoding="utf-8")

    real_copy2 = shutil.copy2
    real_copyfile = shutil.copyfile

    def fake_copy2(src, dst, *args, **kwargs):
        if Path(src) == file_path and Path(dst) == destination / "alpha.txt":
            raise PermissionError(1, "Operation not permitted", str(dst))
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fake_copy2)
    monkeypatch.setattr(shutil, "copyfile", real_copyfile)

    deploy.copy_tree(source, destination)

    assert (destination / "alpha.txt").read_text(encoding="utf-8") == "hello"
