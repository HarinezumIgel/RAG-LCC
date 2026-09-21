# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import Scripts.Setup as Setup
from Commons.StartupCommons import StartupCommons


@pytest.fixture
def _restore_argv() -> Any:
    original = list(sys.argv)
    try:
        yield
    finally:
        sys.argv = original


def _patch_setup_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Setup, "_banner", lambda: None)
    monkeypatch.setattr(Setup, "_print_execution_plan", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_project_root", lambda: None)
    monkeypatch.setattr(Setup, "_init_setup_log", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_venv", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_cache_ownership", lambda: None)
    monkeypatch.setattr(Setup, "_write_setup_log", lambda *a, **k: None)
    monkeypatch.setattr(Setup, "_install_apt_packages", lambda: None)
    monkeypatch.setattr(Setup, "_show_licenses_pager", lambda: True)
    monkeypatch.setattr(Setup, "_enrich_pip_install_meta_with_consent", lambda: None)
    monkeypatch.setattr(Setup, "_make_scripts_executable", lambda _p: None)
    monkeypatch.setattr(Setup, "_print_gpu_notice", lambda w=70: None)
    monkeypatch.setattr(Setup, "_print_setup_notice", lambda w=70: None)

    def _fake_subprocess_run(*a: Any, **k: Any) -> Any:
        _ = (a, k)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(Setup.subprocess, "run", _fake_subprocess_run)


class TestSetupFlow:
    def test_normal_mode_runs_spacy_step_before_rehash(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _restore_argv: Any,
    ) -> None:
        _patch_setup_common(monkeypatch)

        step_calls: list[int] = []
        questions_called = 0

        def _fake_run_step(
            step: int,
            label: str,
            script: str,
            description: str,
            extra_argv: list[str] | None = None,
            required: bool = True,
        ) -> None:
            _ = (label, script, description, extra_argv, required)
            step_calls.append(step)

        def _fake_run_setup_questions() -> None:
            nonlocal questions_called
            questions_called += 1

        monkeypatch.setattr(Setup, "_run_step", _fake_run_step)
        monkeypatch.setattr(Setup, "_run_setup_questions", _fake_run_setup_questions)
        monkeypatch.setattr(Setup, "_confirm", lambda _prompt: True)

        sys.argv = [
            "Setup.py",
            "--no-examples-copy",
            "--skip-signature-verification",
        ]
        Setup.main()

        assert questions_called == 1
        assert step_calls == [2, 4, 5, 6, 7]
        assert step_calls.index(6) < step_calls.index(7)

    def test_config_values_only_mode_without_rehash_skips_installers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _restore_argv: Any,
    ) -> None:
        _patch_setup_common(monkeypatch)

        step_calls: list[int] = []
        questions_called = 0

        def _fail_if_called() -> None:
            raise AssertionError("Installer path should be skipped in config-only mode")

        def _fake_run_step(
            step: int,
            label: str,
            script: str,
            description: str,
            extra_argv: list[str] | None = None,
            required: bool = True,
        ) -> None:
            _ = (label, script, description, extra_argv, required)
            step_calls.append(step)

        def _fake_run_setup_questions() -> None:
            nonlocal questions_called
            questions_called += 1

        monkeypatch.setattr(Setup, "_install_apt_packages", _fail_if_called)
        monkeypatch.setattr(Setup, "_run_step", _fake_run_step)
        monkeypatch.setattr(Setup, "_run_setup_questions", _fake_run_setup_questions)

        sys.argv = ["Setup.py", "--set-config-values-only", "--no-config-rehash"]
        Setup.main()

        assert questions_called == 1
        assert step_calls == []

    def test_config_values_only_mode_runs_only_rehash_step_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _restore_argv: Any,
    ) -> None:
        _patch_setup_common(monkeypatch)

        step_calls: list[int] = []
        questions_called = 0

        def _fake_run_step(
            step: int,
            label: str,
            script: str,
            description: str,
            extra_argv: list[str] | None = None,
            required: bool = True,
        ) -> None:
            _ = (label, script, description, extra_argv, required)
            step_calls.append(step)

        def _fake_run_setup_questions() -> None:
            nonlocal questions_called
            questions_called += 1

        monkeypatch.setattr(Setup, "_run_step", _fake_run_step)
        monkeypatch.setattr(Setup, "_run_setup_questions", _fake_run_setup_questions)

        sys.argv = ["Setup.py", "--set-config-values-only"]
        Setup.main()

        assert questions_called == 1
        assert step_calls == [7]


class TestStartupEnvironmentChecks:
    def test_environment_checks_match_expected_keys(self) -> None:
        checks = StartupCommons._environment_checks()
        assert set(checks.keys()) == {
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "LICENSE_DOWNLOAD",
            "RAG_LCC_NW_TRACE",
            "RAG_LCC_STACK_TRACE",
            "ARGOS_MODEL_PROVIDER",
            "HF_HUB_DISABLE_PROGRESS_BARS",
        }
