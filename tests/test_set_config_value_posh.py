import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts_posh" / "private"
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PWSH is None, reason="PowerShell 7 is required")


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert PWSH is not None
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(script), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("slot", "original", "value", "expected"),
    [
        (
            'os.environ["WEB_SEARCH_MODE"]',
            'os.environ["WEB_SEARCH_MODE"] = "1"  # keep\n',
            '"0"',
            'os.environ["WEB_SEARCH_MODE"] = "0"  # keep\n',
        ),
        (
            "os.environ['KEY.WITH.DOTS']",
            "os.environ['KEY.WITH.DOTS'] = 'old#value'\n",
            "'new'",
            "os.environ['KEY.WITH.DOTS'] = 'new'\n",
        ),
        ("_FLAG", "_FLAG: bool = True\n", "False", "_FLAG: bool = False\n"),
        (
            "_DOCS.enabled",
            '_DOCS = {\n    "enabled": True,\n}\n',
            "False",
            '_DOCS = {\n    "enabled": False,\n}\n',
        ),
        (
            "_MODELS.ollama._OLLAMA.BASE_URL",
            '_MODELS = {\n "ollama": {\n  "_OLLAMA": {\n'
            '   "BASE_URL": "old", # keep\n  }\n }\n}\n',
            '"http://localhost:11434/api/generate"',
            '_MODELS = {\n "ollama": {\n  "_OLLAMA": {\n'
            '   "BASE_URL": "http://localhost:11434/api/generate", # keep\n'
            "  }\n }\n}\n",
        ),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_assignment_and_idempotence(
    tmp_path: Path, slot: str, original: str, value: str, expected: str, newline: str
) -> None:
    config = tmp_path / "config.py"
    config.write_bytes(original.replace("\n", newline).encode("utf-8"))
    for _ in range(2):
        result = run_script(
            SCRIPTS / "Set-ConfigValue.ps1",
            "-ConfigFile",
            str(config),
            "-SlotName",
            slot,
            "-Value",
            value,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert config.read_bytes() == expected.replace("\n", newline).encode("utf-8")


@pytest.mark.parametrize(
    "slot", ['os.environ["MISSING"]', "_MISSING", "_MISSING.enabled", "_DOCS.missing"]
)
def test_missing_slot_fails_without_modifying_file(tmp_path: Path, slot: str) -> None:
    config = tmp_path / "config.py"
    original = b'_DOCS = {\n    "enabled": True,\n}\n'
    config.write_bytes(original)
    result = run_script(
        SCRIPTS / "Set-ConfigValue.ps1",
        "-ConfigFile",
        str(config),
        "-SlotName",
        slot,
        "-Value",
        "False",
    )
    assert result.returncode == 1
    assert "not found" in result.stderr
    assert config.read_bytes() == original


def _run_hardening(
    config_dir: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    deploy = (SCRIPTS / "Deploy.ps1").read_text(encoding="utf-8-sig")
    block = deploy.split("# Harden web-search defaults for deployment.", 1)[1]
    block = block.split("# Clear config hashes", 1)[0]
    runner = tmp_path / "harden.ps1"
    runner.write_text(
        "param([string]$configDir, [string]$scriptsDir_Priv_ProjectPath)\n"
        "$currentDir = Get-Location\n" + block,
        encoding="utf-8",
    )
    return run_script(runner, str(config_dir), str(SCRIPTS))


def test_deploy_hardening_on_current_configs(tmp_path: Path) -> None:
    config_dir = tmp_path / "Configuration"
    shutil.copytree(ROOT / "src" / "Configuration", config_dir)
    originals = {path.name: path.read_bytes() for path in config_dir.glob("*.py")}
    result = _run_hardening(config_dir, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    replacements = {
        "Config_Internet_Env.py": [
            (r'(os\.environ\["WEB_SEARCH_MODE"\]\s*=\s*)"[^"]*"', r'\g<1>"0"'),
            (
                r'(os\.environ\["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"\]\s*=\s*)"[^"]*"',
                r'\g<1>"0"',
            ),
        ],
        "Config_WebSearch.py": [
            (r"(_OPENWEB_UI_WEBSEARCH:\s*bool\s*=\s*)(True|False)", r"\g<1>False"),
        ],
        "Config_Models.py": [
            (r'("BASE_URL":\s*)"[^"]*"', r'\g<1>"http://localhost:11434/api/generate"'),
        ],
    }
    for name, original in originals.items():
        expected = original.decode("utf-8")
        for pattern, replacement in replacements.get(name, []):
            expected, count = re.subn(pattern, replacement, expected, count=1)
            assert count == 1, (name, pattern)
        assert (config_dir / name).read_bytes() == expected.encode("utf-8"), name


def test_deploy_hardening_reports_missing_pdf_serving_setting(tmp_path: Path) -> None:
    config_dir = tmp_path / "Configuration"
    shutil.copytree(ROOT / "src" / "Configuration", config_dir)
    env_config = config_dir / "Config_Internet_Env.py"
    original = env_config.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'^os\.environ\["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"\]\s*=.*$',
        "",
        original,
        flags=re.MULTILINE,
    )
    assert count == 1
    env_config.write_text(updated, encoding="utf-8")

    result = _run_hardening(config_dir, tmp_path)

    assert result.returncode != 0
    assert "SERVE_RAGCHATSERVICE_GROUNDED_DOCS" in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("answer", "kept", "prompts"),
    [
        ("", True, 2),
        ("   ", True, 2),
        ("no", False, 1),
        ("yes", True, 1),
        (" YES ", True, 1),
        ("invalid", True, 2),
    ],
)
def test_deploy_existing_directory_prompt(
    tmp_path: Path, answer: str, kept: bool, prompts: int
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    stale = target / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    protected = [".git", ".github", ".gitignore", ".gitattributes"]
    for name in protected:
        (target / name).write_text("keep", encoding="utf-8")
    deploy = (SCRIPTS / "Deploy.ps1").read_text(encoding="utf-8-sig")
    block = (
        "if (Test-Path -LiteralPath $targetDir) {"
        + deploy.split("if (Test-Path -LiteralPath $targetDir) {", 1)[1].split(
            "# Normalize target dir", 1
        )[0]
    )
    runner = tmp_path / "prompt.ps1"
    runner.write_text(
        "param([string]$targetDir, [string]$answerValue)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$script:prompts = 0\n"
        "function Read-Host {\n"
        "    param([string]$Prompt)\n"
        "    Write-Host $Prompt\n"
        "    $script:prompts++\n"
        "    if ($script:prompts -gt 2) { throw 'Unexpected repeated prompt' }\n"
        "    if ($script:prompts -eq 1) { return $answerValue }\n"
        "    return 'yes'\n"
        "}\n" + block + "\n"
        'Write-Output "Prompts: $script:prompts"\n',
        encoding="utf-8",
    )
    result = run_script(runner, str(target), answer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Do you want to keep the directory? (yes/no)" in result.stdout
    assert f"Prompts: {prompts}" in result.stdout
    assert stale.exists() is kept
    for name in protected:
        assert (target / name).read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("mode", ["return", "throw", "interrupt"])
def test_deploy_restores_starting_directory(tmp_path: Path, mode: str) -> None:
    project = tmp_path / "project"
    private = project / "scripts_posh" / "private"
    private.mkdir(parents=True)
    keys = project / "verify_sign"
    keys.mkdir()
    (keys / "RAG_LCC_private.pem").write_text("test-only", encoding="utf-8")
    (keys / "mode.txt").write_text(mode, encoding="utf-8")
    target = tmp_path / "target"
    public = target / "scripts_posh"
    public.mkdir(parents=True)
    (private / "sign_verify.ps1").write_text(
        "param($InputDir, $KeyDir, $Mode, $IncludeDirs, $IncludeFiles)\n"
        "Set-Location -LiteralPath $InputDir\n"
        "New-Item -Path (Join-Path $InputDir 'started') -ItemType File | Out-Null\n"
        "$testMode = Get-Content -LiteralPath (Join-Path $KeyDir 'mode.txt')\n"
        "if ($testMode -eq 'throw') { throw 'Expected test failure' }\n"
        "if ($testMode -eq 'interrupt') { Start-Sleep -Seconds 60 }\n",
        encoding="utf-8",
    )
    (public / "VerifySignature.ps1").write_text(
        "param($InputDir, $KeyDir, $IncludeDirs, $IncludeFiles)\n",
        encoding="utf-8",
    )
    runner = tmp_path / "restore.ps1"
    runner.write_text(
        "param($Deploy, $Project, $Target, $Start, $Mode)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$runspace = [runspacefactory]::CreateRunspace()\n"
        "$runspace.Open()\n"
        "$ps = [powershell]::Create()\n"
        "$ps.Runspace = $runspace\n"
        "try {\n"
        "    $runspace.SessionStateProxy.Path.SetLocation($Start) | Out-Null\n"
        "    $null = $ps.AddCommand($Deploy).AddParameter('ProjectPath', $Project)"
        ".AddParameter('TargetDir', $Target).AddParameter('Resign')\n"
        "    $pending = $ps.BeginInvoke()\n"
        "    if ($Mode -eq 'interrupt') {\n"
        "        $deadline = [datetime]::UtcNow.AddSeconds(10)\n"
        "        while (-not (Test-Path -LiteralPath (Join-Path $Target 'started'))) {\n"
        "            if ([datetime]::UtcNow -gt $deadline) { throw 'Deployment did not start' }\n"
        "            Start-Sleep -Milliseconds 20\n"
        "        }\n"
        "        $ps.Stop()\n"
        "        if ($ps.InvocationStateInfo.State -ne 'Stopped') { throw 'Not stopped' }\n"
        "    } else {\n"
        "        try { $null = $ps.EndInvoke($pending) } catch {\n"
        "            if ($Mode -ne 'throw') { throw }\n"
        "        }\n"
        "        if ($Mode -eq 'return' -and $ps.HadErrors) { throw $ps.Streams.Error[0] }\n"
        "        if ($Mode -eq 'throw' -and -not $ps.HadErrors) { throw 'Missing failure' }\n"
        "    }\n"
        "    $actual = $runspace.SessionStateProxy.Path.CurrentLocation.Path\n"
        '    if ($actual -ne $Start) { throw "Not restored: $actual" }\n'
        "    Write-Output 'Starting directory restored'\n"
        "} finally { $ps.Dispose(); $runspace.Dispose() }\n",
        encoding="utf-8",
    )
    result = run_script(
        runner,
        str(SCRIPTS / "Deploy.ps1"),
        str(project),
        str(target),
        str(tmp_path),
        mode,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "started").exists()
    assert "Starting directory restored" in result.stdout
