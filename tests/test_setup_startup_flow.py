# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
# pyright: reportUnusedFunction=false

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import Scripts.Setup as Setup
from Commons.Exceptions import ConfigurationError
from Commons.StartupCommons import StartupCommons
from Config.Config import Config


@pytest.fixture
def _restore_argv() -> Any:
    original = list(sys.argv)
    try:
        yield
    finally:
        sys.argv = original


def _patch_setup_common(monkeypatch: pytest.MonkeyPatch) -> None:
    def _write_setup_log_noop(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def _make_scripts_executable_noop(path: Any) -> None:
        _ = path

    monkeypatch.setattr(Setup, "_banner", lambda: None)
    monkeypatch.setattr(Setup, "_print_execution_plan", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_project_root", lambda: None)
    monkeypatch.setattr(Setup, "_init_setup_log", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_venv", lambda: None)
    monkeypatch.setattr(Setup, "_ensure_cache_ownership", lambda: None)
    monkeypatch.setattr(Setup, "_write_setup_log", _write_setup_log_noop)
    monkeypatch.setattr(Setup, "_install_apt_packages", lambda: None)
    monkeypatch.setattr(Setup, "_show_licenses_pager", lambda: True)
    monkeypatch.setattr(Setup, "_enrich_pip_install_meta_with_consent", lambda: None)
    monkeypatch.setattr(
        Setup,
        "_make_scripts_executable",
        _make_scripts_executable_noop,
    )
    monkeypatch.setattr(Setup, "_print_gpu_notice", lambda w=70: None)
    monkeypatch.setattr(Setup, "_print_setup_notice", lambda w=70: None)

    def _fake_subprocess_run(*a: Any, **k: Any) -> Any:
        _ = (a, k)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(Setup.subprocess, "run", _fake_subprocess_run)


class TestSetupFlow:
    @pytest.mark.parametrize("runtime", ["docker", "windows", "host"])
    def test_listener_guidance_orders_options_by_exposure(
        self,
        runtime: Literal["docker", "windows", "host"],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        Setup._print_ragchatservice_listener_guidance(runtime)
        output = capsys.readouterr().out
        assert (
            output.index("127.0.0.1")
            < output.index("specific")
            < output.index("0.0.0.0")
        )
        assert "remote browsers" in output
        assert "_SERVE_DOCS.public_base_url" in output
        assert f"{Setup._PALE_WHITE}    Document links share" in output
        assert "Loopback-only listeners are reachable only from the" in output
        assert "same host (or through port forwarding/tunneling)." in output
        assert "an HTTPS reverse proxy" in output
        assert f"{Setup._TURQUOISE}127.0.0.1" in output
        assert f"{Setup._TURQUOISE}0.0.0.0" in output

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("docs_port", ["11436", "12436"])
    def test_setup_persists_local_listener_independently_of_openwebui(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        enabled: bool,
        docs_port: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = tmp_path / "Config_Internet_Env.py"
        flags = (
            "HF_HUB_OFFLINE",
            "LICENSE_DOWNLOAD",
            "WEB_SEARCH_MODE",
            "SERVE_OPENWEBUI_CHAT",
            "SERVE_RAGCHATSERVICE_GROUNDED_DOCS",
            "SERVE_RAGCHAT_GROUNDED_DOCS",
            "RAG_LCC_NW_TRACE",
        )
        config.write_text(
            "\n".join(f'os.environ["{flag}"] = "0"' for flag in flags) + "\n",
            encoding="utf-8",
        )
        chat_config = tmp_path / "Config_RAGChat.py"
        chat_config.write_text(
            '_LOCAL_SERVE_DOCS = {"port": 11436}\n', encoding="utf-8"
        )
        monkeypatch.setattr(Setup, "_CFG_DIR", tmp_path)
        monkeypatch.setattr(Setup, "_detect_runtime_network_context", lambda: "windows")
        monkeypatch.setattr(Setup, "_write_setup_log", lambda *args, **kwargs: None)
        monkeypatch.setattr("builtins.input", lambda *args: "")
        choices = iter(["2", "1"])
        port_answers = iter(["invalid", "0", "65536", docs_port, docs_port])
        port_corrections: list[str | None] = []

        def choice(
            prompt: str,
            options: list[str],
            default: str,
            **kwargs: Any,
        ) -> str:
            return next(choices) if prompt.startswith("Choose:") else default

        def text(prompt: str, default: str, **kwargs: Any) -> str:
            if prompt == "Set _LOCAL_SERVE_DOCS.port":
                assert default == "11436"
                port_corrections.append(kwargs.get("correction_value"))
                return default if docs_port == "11436" else next(port_answers)
            if "BASE_URL" in prompt:
                return "http://localhost:11434"
            if prompt.endswith(".HOST"):
                assert default == "127.0.0.1"
                return default
            return default

        corrections: list[bool | None] = []

        def boolean(prompt: str, default: bool, **kwargs: Any) -> bool:
            if prompt == "Enable SERVE_RAGCHAT_GROUNDED_DOCS":
                corrections.append(kwargs.get("correction_value"))
                return enabled
            return default

        monkeypatch.setattr(Setup, "_prompt_choice", choice)
        monkeypatch.setattr(Setup, "_prompt_text", text)
        monkeypatch.setattr(Setup, "_prompt_bool", boolean)

        Setup._run_setup_questions()

        saved = config.read_text(encoding="utf-8")
        assert f'os.environ["SERVE_RAGCHAT_GROUNDED_DOCS"] = "{int(enabled)}"' in saved
        assert 'os.environ["SERVE_OPENWEBUI_CHAT"] = "0"' in saved
        assert 'os.environ["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"] = "0"' in saved
        assert corrections == [None, enabled]
        namespace: dict[str, Any] = {}
        exec(chat_config.read_text(encoding="utf-8"), namespace)
        assert namespace["_LOCAL_SERVE_DOCS"]["port"] == int(docs_port)
        assert port_corrections[-1] == docs_port
        output = capsys.readouterr().out
        if docs_port != "11436":
            assert output.count("Enter a port number between 1 and 65535.") == 3

    def test_config_review_uses_installed_configs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = tmp_path / "src" / "Configuration"
        config.mkdir(parents=True)
        expected = {
            "Config_Models.py",
            "Config_Banned_Detection.py",
            "Config_Banned_Content.py",
            "Config_Banned_Prompts.py",
            "Config_Load_Retrievers.py",
            "Config_Load_Chunkers.py",
            "Config_WebSearch.py",
            "Config_Internet_Env.py",
        }
        for name in expected:
            (config / name).write_text("# installed config\n", encoding="utf-8")
        (config / "Config_Global.py").write_text("# not reviewed\n", encoding="utf-8")
        monkeypatch.setattr(Setup, "_PROJECT_ROOT", tmp_path)

        files = Setup._collect_config_files()

        assert {name for name, _, _ in files} == expected
        assert all(
            path.parent == config and status == "critical" for _, status, path in files
        )
        assert not (tmp_path / "Examples").exists()

    def test_missing_installed_configs_are_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Setup, "_PROJECT_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="Config_Models.py"):
            Setup._collect_config_files()

    def test_review_step_does_not_run_copy_script(
        self, monkeypatch: pytest.MonkeyPatch, _restore_argv: Any
    ) -> None:
        _patch_setup_common(monkeypatch)
        scripts: list[str] = []
        monkeypatch.setattr(
            Setup, "_run_step", lambda **kwargs: scripts.append(kwargs["script"])
        )
        monkeypatch.setattr(Setup, "_run_setup_questions", lambda: None)
        monkeypatch.setattr(Setup, "_confirm", lambda prompt: True)
        files = [
            ("Config_Models.py", "critical", Path("Configuration") / "Config_Models.py")
        ]
        monkeypatch.setattr(Setup, "_collect_config_files", lambda: files)
        reviewed: list[object] = []
        monkeypatch.setattr(
            Setup,
            "_review_config_files_interactive",
            lambda selected: reviewed.append(selected) or True,
        )
        sys.argv = ["Setup.py", "--skip-signature-verification"]

        Setup.main()

        assert reviewed == [files]
        assert "CopyExampleConfigs.py" not in scripts
        assert "RecalcConfigHashes.py" in scripts

    def test_vllm_selection_prints_orchestration_hint(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        Setup._print_vllm_orchestration_hint("vllm")

        output = capsys.readouterr().out
        assert "configuration-driven orchestration" in output
        assert "models, LLMs, embedders, and rerankers" in output
        assert "https://github.com/HarinezumIgel/harinezumigel-llm-stack" in output

    def test_ollama_selection_does_not_print_vllm_hint(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        Setup._print_vllm_orchestration_hint("ollama")

        assert capsys.readouterr().out == ""

    def test_required_command_masks_credential_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            Setup.subprocess,
            "run",
            lambda command: SimpleNamespace(returncode=0),
        )

        Setup._run_required_command(
            ["client", "--api-key", "command-line-secret"],
            "client setup",
        )

        output = capsys.readouterr().out
        assert "command-line-secret" not in output
        assert "******" in output

    @pytest.mark.parametrize(
        "skip_review", ["--no-config-review", "--no-examples-copy"]
    )
    def test_normal_mode_runs_spacy_step_before_rehash(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _restore_argv: Any,
        skip_review: str,
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

        def _always_confirm(prompt: str) -> bool:
            _ = prompt
            return True

        monkeypatch.setattr(Setup, "_run_step", _fake_run_step)
        monkeypatch.setattr(Setup, "_run_setup_questions", _fake_run_setup_questions)
        monkeypatch.setattr(Setup, "_confirm", _always_confirm)

        sys.argv = [
            "Setup.py",
            skip_review,
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
    @pytest.mark.parametrize("enabled", [False, True])
    def test_ragchat_listener_setting_is_reported_in_orange_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, enabled: bool
    ) -> None:
        from Gui.Colors import BRIGHT_BLUE, ORANGE
        from Gui.PrettyWriter import PrettyWriter

        calls: list[tuple[str, str | None]] = []
        monkeypatch.setenv("SERVE_RAGCHAT_GROUNDED_DOCS", "1" if enabled else "0")
        monkeypatch.setenv("WEB_SEARCH_MODE", "0")

        def record(
            self: PrettyWriter,
            severity: str,
            label: str,
            message: str,
            *args: Any,
            color: str | None = None,
            **kwargs: Any,
        ) -> str:
            calls.append((message, color))
            return message

        monkeypatch.setattr(PrettyWriter, "write", record)
        StartupCommons._report_environment_settings(
            PrettyWriter(always_on=True), Config(), "RAGChat"
        )
        assert (
            f"SERVE_RAGCHAT_GROUNDED_DOCS={int(enabled)}",
            ORANGE if enabled else BRIGHT_BLUE,
        ) in calls

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


class _StartupCfgStub(Config):
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get(
        self,
        key: str,
        default: Any = None,
        allow_indirect: bool = True,
        *,
        silent: bool = False,
    ) -> Any:
        _ = (allow_indirect, silent)
        return self.values.get(key, default)

    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        _ = silent
        return str(self.values.get(key, default))

    def get_bool(
        self,
        key: str,
        default: bool = False,
        *,
        silent: bool = False,
    ) -> bool:
        _ = silent
        return bool(self.values.get(key, default))

    def get_int(self, key: str, default: int = 0, *, silent: bool = False) -> int:
        _ = silent
        return int(self.values.get(key, default))


class TestPipelineCheckStartupValidation:
    def _base_values(self) -> dict[str, Any]:
        slot = "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE"
        return {
            "_ACTIVE_DETECTION_CONFIG": "STRICT_DETECT_CONFIG",
            "_FRIENDLY_NAME": "RAGChat",
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.Check": True,
            f"{slot}.REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
            f"{slot}.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 3,
            f"{slot}.ALGOS_TO_PROCESS": {
                "Regex": True,
                "Jaccard": True,
                "BM25": True,
                "Keybert": True,
            },
        }

    def test_allows_valid_pipeline_check_requirements(self) -> None:
        cfg = _StartupCfgStub(self._base_values())
        StartupCommons._validate_pipeline_check_requirements(cfg)  # no exception

    def test_skips_validation_when_pipeline_check_disabled(self) -> None:
        values = self._base_values()
        values["_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.Check"] = (
            False
        )
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 999
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE"
        ] = 999
        cfg = _StartupCfgStub(values)
        StartupCommons._validate_pipeline_check_requirements(cfg)  # no exception

    def test_rejects_non_positive_required_thresholds(self) -> None:
        values = self._base_values()
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 0
        cfg = _StartupCfgStub(values)

        with pytest.raises(ConfigurationError, match="both values must be >= 1"):
            StartupCommons._validate_pipeline_check_requirements(cfg)

    def test_rejects_requirements_above_enabled_algo_count(self) -> None:
        values = self._base_values()
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE.ALGOS_TO_PROCESS"
        ] = {
            "Regex": True,
            "Jaccard": False,
            "BM25": False,
            "Keybert": False,
        }
        cfg = _StartupCfgStub(values)

        with pytest.raises(
            ConfigurationError,
            match="Invalid PIPELINE_CHECK consensus requirements",
        ):
            StartupCommons._validate_pipeline_check_requirements(cfg)


class TestPromptCheckStartupValidation:
    def _base_values(self) -> dict[str, Any]:
        slot = "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE"
        return {
            "_ACTIVE_DETECTION_CONFIG": "STRICT_DETECT_CONFIG",
            "_FRIENDLY_NAME": "RAGChat",
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.Check": True,
            f"{slot}.REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
            f"{slot}.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 3,
            f"{slot}.ALGOS_TO_PROCESS": {
                "Regex": True,
                "Jaccard": True,
                "BM25": True,
                "Keybert": True,
            },
        }

    def test_allows_valid_prompt_check_requirements(self) -> None:
        cfg = _StartupCfgStub(self._base_values())
        StartupCommons._validate_prompt_check_requirements(cfg)  # no exception

    def test_skips_validation_when_prompt_check_disabled(self) -> None:
        values = self._base_values()
        values["_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.Check"] = False
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 999
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE"
        ] = 999
        cfg = _StartupCfgStub(values)
        StartupCommons._validate_prompt_check_requirements(cfg)  # no exception

    def test_rejects_non_positive_prompt_required_thresholds(self) -> None:
        values = self._base_values()
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 0
        cfg = _StartupCfgStub(values)

        with pytest.raises(ConfigurationError, match="both values must be >= 1"):
            StartupCommons._validate_prompt_check_requirements(cfg)

    def test_rejects_prompt_requirements_above_enabled_algo_count(self) -> None:
        values = self._base_values()
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE.ALGOS_TO_PROCESS"
        ] = {
            "Regex": True,
            "Jaccard": False,
            "BM25": False,
            "Keybert": False,
        }
        cfg = _StartupCfgStub(values)

        with pytest.raises(
            ConfigurationError,
            match="Invalid PROMPT_CHECK consensus requirements",
        ):
            StartupCommons._validate_prompt_check_requirements(cfg)


class TestCombinedStageStartupValidation:
    def _base_values(self) -> dict[str, Any]:
        prompt_slot = (
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE"
        )
        pipeline_slot = (
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE"
        )
        return {
            "_ACTIVE_DETECTION_CONFIG": "STRICT_DETECT_CONFIG",
            "_FRIENDLY_NAME": "RAGChat",
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.Check": True,
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.Check": True,
            f"{prompt_slot}.REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
            f"{prompt_slot}.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 2,
            f"{pipeline_slot}.REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
            f"{pipeline_slot}.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 2,
            f"{prompt_slot}.ALGOS_TO_PROCESS": {
                "Regex": True,
                "Jaccard": True,
            },
            f"{pipeline_slot}.ALGOS_TO_PROCESS": {
                "Regex": True,
                "Jaccard": True,
            },
        }

    def test_raises_combined_error_for_both_invalid_stages(self) -> None:
        values = self._base_values()
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PROMPT_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 0
        values[
            "_BANNED_DETECT.STRICT_DETECT_CONFIG.RAGChat.PIPELINE_CHECK.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        ] = 0

        cfg = _StartupCfgStub(values)
        with pytest.raises(ConfigurationError) as exc_info:
            StartupCommons._validate_all_stage_check_requirements(cfg)

        error_text = str(exc_info.value)
        assert "Invalid PROMPT_CHECK thresholds" in error_text
        assert "Invalid PIPELINE_CHECK thresholds" in error_text
