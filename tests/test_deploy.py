import ast
import subprocess
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("deploy")
import deploy.scripts.deploy as deploy
import deploy.scripts.sign_files as signing


def _use_backup_dir(monkeypatch: pytest.MonkeyPatch, backup: Path) -> None:
    def mkdtemp(**kwargs: object) -> str:
        return str(backup)

    monkeypatch.setattr(deploy.tempfile, "mkdtemp", mkdtemp)


def _noop(*args: object, **kwargs: object) -> None:
    pass


def _successful_run(
    *args: object, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0)


def _no_api_key_clears(config: Path) -> list[dict[str, str]]:
    return []


def _configuration_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "source"
    config = project / "src" / "Configuration"
    config.mkdir(parents=True)
    slots = [
        "Config_Models",
        "Config_Banned_Detection",
        "Config_Banned_Content",
        "Config_Banned_Prompts",
        "Config_Load_Retrievers",
        "Config_Load_Chunkers",
        "Config_WebSearch",
        "Config_Internet_Env",
    ]
    for slot in slots:
        (config / f"{slot}.py").write_bytes(b"# original config\r\n")
    (config / "Config_WebSearch.py").write_bytes(b"_OPENWEB_UI_WEBSEARCH = True\r\n")
    (config / "Config_Internet_Env.py").write_bytes(
        b'os.environ["SERVE_RAGCHAT_GROUNDED_DOCS"] = "1"\r\n'
    )
    (config / "Config_Global.py").write_bytes(
        f"_CRITICAL_CONFIG_HASHES = {dict.fromkeys(slots, 'source-hash')!r}\r\n".replace(
            "'", '"'
        ).encode(
            "utf-8"
        )
    )
    (config / "unchanged.dat").write_bytes(b"\x00original\xff")
    scripts = project / "src" / "Scripts"
    scripts.mkdir()
    for relative in (
        Path("Scripts") / "RecalcConfigHashes.py",
        Path("Scripts") / "VerifySignatures.py",
        Path("Commons") / "DriveRootGuard.py",
        Path("Gui") / "Colors.py",
    ):
        destination = project / "src" / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(deploy.REPO_ROOT / "src" / relative, destination)
    return project, config, tmp_path / "deployment" / "src" / "Configuration"


def _assert_empty_configuration_hashes(config: Path) -> None:
    text = (config / "Config_Global.py").read_text(encoding="utf-8")
    hashes = ast.literal_eval(text.partition("=")[2].strip())
    assert len(hashes) == 8
    assert set(hashes.values()) == {""}


def test_configuration_packaging_restores_source_and_removes_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, source, destination = _configuration_project(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    backup = tmp_path / "config-backup"
    _use_backup_dir(monkeypatch, backup)
    backup.mkdir()

    with deploy.prepare_deployment_configuration(
        project,
        destination,
        [
            {
                "conf": "Config_WebSearch.py",
                "keyname": "_OPENWEB_UI_WEBSEARCH",
                "value": "False",
            },
        ],
    ):
        _assert_empty_configuration_hashes(source)
        assert backup.exists()

    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert "_OPENWEB_UI_WEBSEARCH = False" in (
        destination / "Config_WebSearch.py"
    ).read_text(encoding="utf-8")
    _assert_empty_configuration_hashes(destination)
    assert (destination / "unchanged.dat").read_bytes() == before["unchanged.dat"]
    assert not backup.exists()
    assert not (destination.parents[1] / "Examples").exists()


def test_update_config_values_preserves_inline_comments(tmp_path: Path) -> None:
    config_file = tmp_path / "Config_Models.py"
    config_file.write_text(
        '_ACTIVE_ENDPOINT = "vllm" # ollama vllm\n'
        '_LABEL = "keep # inside string"  # label note\n',
        encoding="utf-8",
    )
    script = deploy.REPO_ROOT / "src" / "Scripts" / "UpdateConfigValues.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--conf",
            str(config_file),
            "--keyname",
            "_ACTIVE_ENDPOINT",
            "--value",
            '"ollama"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert config_file.read_text(encoding="utf-8") == (
        '_ACTIVE_ENDPOINT = "ollama" # ollama vllm\n'
        '_LABEL = "keep # inside string"  # label note\n'
    )


@pytest.mark.parametrize(
    "failure_stage", ["update", "validate", "hash", "copy", "sign"]
)
def test_configuration_packaging_restores_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    project, source, destination = _configuration_project(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    backup = tmp_path / "config-backup"
    backup.mkdir()
    _use_backup_dir(monkeypatch, backup)

    def update(config: Path, updates: list[dict[str, str]]) -> None:
        (config / "Config_WebSearch.py").write_text("edited", encoding="utf-8")
        if failure_stage == "update":
            raise RuntimeError("update failed")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"{failure_stage} failed")

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1 if failure_stage == "hash" else 0)

    monkeypatch.setattr(deploy, "update_config_values", update)
    monkeypatch.setattr(
        deploy,
        "validate_config_updates",
        fail if failure_stage == "validate" else _noop,
    )
    monkeypatch.setattr(deploy, "run", run)
    if failure_stage == "copy":
        monkeypatch.setattr(deploy, "copy_tree", fail)

    with pytest.raises(RuntimeError):
        with deploy.prepare_deployment_configuration(project, destination, []):
            fail()

    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert not backup.exists()


def test_configuration_backup_failure_does_not_restore_partial_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, source, destination = _configuration_project(tmp_path)
    original = (source / "Config_WebSearch.py").read_bytes()
    backup = tmp_path / "config-backup"
    backup.mkdir()
    _use_backup_dir(monkeypatch, backup)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("backup failed")

    monkeypatch.setattr(deploy.shutil, "copytree", fail)
    with pytest.raises(OSError, match="backup failed"):
        with deploy.prepare_deployment_configuration(project, destination, []):
            pytest.fail("Backup failure must prevent packaging")
    assert (source / "Config_WebSearch.py").read_bytes() == original
    assert not backup.exists()


def test_failed_configuration_restore_retains_recoverable_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, source, destination = _configuration_project(tmp_path)
    original = (source / "Config_WebSearch.py").read_bytes()
    backup = tmp_path / "config-backup"
    backup.mkdir()
    _use_backup_dir(monkeypatch, backup)
    real_copytree = shutil.copytree

    def copytree(src: Path, dst: Path, *, dirs_exist_ok: bool = False) -> Path:
        if dst == source:
            raise OSError("restore failed")
        return real_copytree(src, dst, dirs_exist_ok=dirs_exist_ok)

    monkeypatch.setattr(deploy.shutil, "copytree", copytree)
    monkeypatch.setattr(deploy, "update_config_values", _noop)
    monkeypatch.setattr(deploy, "validate_config_updates", _noop)
    monkeypatch.setattr(deploy, "run", _successful_run)

    with pytest.raises(OSError, match="restore failed"):
        with deploy.prepare_deployment_configuration(project, destination, []):
            pass

    assert (backup / "Configuration" / "Config_WebSearch.py").read_bytes() == original


def test_main_packages_configuration_without_examples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project, source, destination = _configuration_project(tmp_path)
    original = (source / "Config_WebSearch.py").read_bytes()
    target = destination.parents[1]
    (project / "EXAMPLES.md").write_text("obsolete", encoding="utf-8")
    (project / "README.md").write_text("documentation", encoding="utf-8")
    monkeypatch.setattr(
        deploy,
        "parse_args",
        lambda: SimpleNamespace(
            project_path=str(project),
            target_dir=str(target),
            host_target_dir="",
            venv_name=".venv",
            resign=False,
        ),
    )
    monkeypatch.setattr(deploy, "_build_api_key_clears", _no_api_key_clears)

    def update(config: Path, updates: list[dict[str, str]]) -> None:
        assert config == source
        assert any(item["keyname"] == "_ACTIVE_ENDPOINT" for item in updates)
        (config / "Config_WebSearch.py").write_text(
            "DEPLOYMENT = True\n", encoding="utf-8"
        )

    monkeypatch.setattr(deploy, "update_config_values", update)
    monkeypatch.setattr(deploy, "validate_config_updates", _noop)
    monkeypatch.setattr(deploy, "run", _successful_run)

    def stop_before_venv(*args: object) -> None:
        raise RuntimeError("configuration packaging complete")

    monkeypatch.setattr(deploy, "create_venv", stop_before_venv)

    with pytest.raises(RuntimeError, match="configuration packaging complete"):
        deploy.main()

    assert (source / "Config_WebSearch.py").read_bytes() == original
    assert (destination / "Config_WebSearch.py").read_text() == "DEPLOYMENT = True\n"
    assert not (target / "Examples").exists()
    assert not (target / "EXAMPLES.md").exists()
    assert (target / "README.md").read_text(encoding="utf-8") == "documentation"


@pytest.mark.parametrize("failure_stage", [None, "sign", "verify"])
def test_main_ships_clean_signed_configs_before_restoring_development(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str | None,
) -> None:
    monkeypatch.setenv("PYTHONUTF8", "1")
    project, source, destination = _configuration_project(tmp_path)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    target = destination.parents[1]
    host = tmp_path / "host"
    keys = project / "verify_sign"
    keys.mkdir()
    signing.load_or_create_keypair(
        keys / "RAG_LCC_private.pem",
        keys / "RAG_LCC_public.pem",
    )
    monkeypatch.setattr(
        deploy,
        "parse_args",
        lambda: SimpleNamespace(
            project_path=str(project),
            target_dir=str(target),
            host_target_dir=str(host),
            venv_name=".venv",
            resign=False,
            skip_module_install=True,
            no_target_env_cleanup=True,
            skip_graphviz=True,
        ),
    )
    monkeypatch.setattr(deploy, "_build_api_key_clears", _no_api_key_clears)
    update = deploy.update_config_values
    updates = [
        {
            "conf": "Config_WebSearch.py",
            "keyname": "_OPENWEB_UI_WEBSEARCH",
            "value": "False",
        },
        {
            "conf": "Config_Internet_Env.py",
            "keyname": 'os.environ["SERVE_RAGCHAT_GROUNDED_DOCS"]',
            "value": '"0"',
        },
    ]

    def update_config(config: Path, defaults: list[dict[str, str]]) -> None:
        assert updates[1] in defaults
        update(config, updates)

    monkeypatch.setattr(deploy, "update_config_values", update_config)
    validate = deploy.validate_config_updates

    def validate_config(config: Path, defaults: list[dict[str, str]]) -> None:
        validate(config, updates)

    monkeypatch.setattr(deploy, "validate_config_updates", validate_config)

    def create_venv(root: Path, name: str) -> None:
        binaries = root / name / ("Scripts" if deploy.os.name == "nt" else "bin")
        binaries.mkdir(parents=True)
        for executable in ("python", "pip"):
            (
                binaries / f"{executable}{'.exe' if deploy.os.name == 'nt' else ''}"
            ).touch()

    monkeypatch.setattr(deploy, "create_venv", create_venv)

    def formatters(*args: object) -> None:
        _assert_empty_configuration_hashes(source)
        # A later packaging step must not leave populated hashes in the release.
        (destination / "Config_Global.py").write_bytes(original["Config_Global.py"])

    monkeypatch.setattr(deploy, "run_formatters", formatters)

    def signing_python(path: Path) -> Path:
        return Path(sys.executable)

    monkeypatch.setattr(deploy, "make_scripts_executable", _noop)
    monkeypatch.setattr(deploy, "ensure_signing_tools", signing_python)
    sign = deploy.sign_files
    verify = deploy.verify_signatures
    stages: list[str] = []

    def sign_configs(root: Path, development: Path) -> None:
        _assert_empty_configuration_hashes(source)
        _assert_empty_configuration_hashes(destination)
        assert "_OPENWEB_UI_WEBSEARCH = False" in (
            destination / "Config_WebSearch.py"
        ).read_text(encoding="utf-8")
        assert 'os.environ["SERVE_RAGCHAT_GROUNDED_DOCS"] = "0"' in (
            destination / "Config_Internet_Env.py"
        ).read_text(encoding="utf-8")
        stages.append("sign")
        sign(root, development)
        if failure_stage == "sign":
            raise RuntimeError("sign failed")

    def verify_configs(root: Path) -> None:
        _assert_empty_configuration_hashes(source)
        stages.append("verify")
        if failure_stage == "verify":
            with (destination / "Config_Global.py").open("a", encoding="utf-8") as file:
                file.write("# changed after signing\n")
        verify(root)

    monkeypatch.setattr(deploy, "sign_files", sign_configs)
    monkeypatch.setattr(deploy, "verify_signatures", verify_configs)
    if failure_stage:
        with pytest.raises(
            RuntimeError, match="sign failed|Signature verification failed"
        ):
            deploy.main()
    else:
        assert deploy.main() == 0

    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert stages == (["sign"] if failure_stage == "sign" else ["sign", "verify"])
    if failure_stage is None:
        _assert_empty_configuration_hashes(destination)
        _assert_empty_configuration_hashes(host / "src" / "Configuration")
        assert (destination / "Config_Global.py").read_bytes() != original[
            "Config_Global.py"
        ]
        verify(target)
        verify(host)
        assert (host / "src" / "Configuration" / "Config_Global.py").read_bytes() == (
            destination / "Config_Global.py"
        ).read_bytes()
    else:
        assert not host.exists()


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
