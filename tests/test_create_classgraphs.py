import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

if not (Path(__file__).resolve().parents[1] / "deploy").exists():
    pytest.skip("deploy directory not present", allow_module_level=True)


def test_ensure_classgraph_dependencies_installs_pip_packages(monkeypatch) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "scripts"
        / "CreateClassGraphs.py"
    )
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

    spec = importlib.util.spec_from_file_location(
        "create_classgraphs_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == [
        [sys.executable, "-m", "pip", "install", "--upgrade", "graphviz", "pylint"]
    ]
    assert os.environ["PATH"].startswith("/usr/bin")


@pytest.fixture
def classgraph_module(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    pytest.importorskip("graphviz")
    pytest.importorskip("pylint")
    monkeypatch.setenv(
        "PATH",
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    if shutil.which("dot") is None or shutil.which("pyreverse") is None:
        pytest.skip("Graphviz and pyreverse executables are required")
    script = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "scripts"
        / "CreateClassGraphs.py"
    )
    spec = importlib.util.spec_from_file_location("classgraph_bom_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("use_detector", [True, False])
@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"], ids=["utf8", "utf8-bom"])
def test_read_text_preserves_content_without_leading_bom(
    classgraph_module: ModuleType, tmp_path: Path, use_detector: bool, bom: bytes
) -> None:
    source = 'class Example:\n    text = "caf\u00e9 and embedded \ufeff"\n'
    path = tmp_path / "example.py"
    path.write_bytes(bom + source.encode("utf-8"))
    assert (
        classgraph_module.read_text_fallback(path, use_detector=use_detector) == source
    )


def test_bom_class_discovery_and_import_scanning(
    classgraph_module: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "model.py").write_bytes(b"\xef\xbb\xbfclass Example:\n    pass\n")
    (tmp_path / "consumer.py").write_bytes(
        b"\xef\xbb\xbffrom model import Example\ninstance = Example()\n"
    )
    classes = classgraph_module.gather_all_classes(tmp_path)
    assert classes == {"model.Example": Path("model.py")}
    imports = classgraph_module.scan_imports(tmp_path, classes)
    assert imports == {"model.Example": {Path("consumer.py")}}


def test_invalid_python_still_raises(
    classgraph_module: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "invalid.py").write_bytes(b"\xef\xbb\xbfclass Broken(\n")
    with pytest.raises(SyntaxError):
        classgraph_module.gather_all_classes(tmp_path)


def test_cli_renders_bom_sources(classgraph_module: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "src"
    package = source / "models"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (package / "example.py").write_bytes(b"\xef\xbb\xbfclass Example:\n    pass\n")
    apps = source / "apps"
    apps.mkdir()
    (apps / "demo.py").write_bytes(
        b"\xef\xbb\xbffrom models.example import Example\ninstance = Example()\n"
    )
    output = tmp_path / "graphs"
    assert classgraph_module.__file__ is not None
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            classgraph_module.__file__,
            "--source-path",
            str(source),
            "--output-dir",
            str(output),
            "--global-diagram",
            "--diagram",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ["demo_imports.png", "classes_models.png"]:
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
