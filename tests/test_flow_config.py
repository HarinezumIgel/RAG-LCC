"""Tests for orchestration flow catalog helpers, overrides and retriever narrowing."""

import os
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Commons.Exceptions import ConfigurationError
from Configuration import Config_KeywordActionSplitter, Config_Orchestrator
from Globals.Session import Session
from Orchestrator import FlowConfig
from Orchestrator.OrchestratorOverrides import OrchestratorOverrides


class _CfgStub:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_dict(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_list(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_str(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


def _shipped_cfg(**overrides: Any) -> _CfgStub:
    values = {
        "_ORCHESTRATION_FLOWS": Config_Orchestrator._ORCHESTRATION_FLOWS,
        "_ORCHESTRATION_FLOW_ALIASES": Config_Orchestrator._ORCHESTRATION_FLOW_ALIASES,
        "_ALLOWED_ORCHESTRATION_FLOWS": Config_Orchestrator._ALLOWED_ORCHESTRATION_FLOWS,
        "_DEFAULT_ORCHESTRATION_FLOW": Config_Orchestrator._DEFAULT_ORCHESTRATION_FLOW,
        "_ALLOWED_RETRIEVE_MODES": [
            "VECTOR",
            "BM25",
            "GRAPH",
            "REGEX",
            "VECTOR_BM25",
            "ALL",
            "WEB",
        ],
    }
    values.update(overrides)
    return _CfgStub(values)


class TestShippedCatalog:
    def test_config_read_errors_propagate(self, monkeypatch):
        cfg = _shipped_cfg()

        def fail(key: str, default: Any = None) -> Any:
            raise ValueError("invalid flow configuration")

        monkeypatch.setattr(cfg, "get_dict", fail)

        with pytest.raises(ValueError, match="invalid flow configuration"):
            FlowConfig.load_flow_map(cfg)

    def test_shipped_config_validates(self):
        FlowConfig.validate_flow_config(_shipped_cfg())

    def test_every_flow_lists_every_known_key(self):
        for name, profile in Config_Orchestrator._ORCHESTRATION_FLOWS.items():
            missing = FlowConfig.KNOWN_FLOW_KEYS - set(profile)
            assert not missing, f"{name} missing {sorted(missing)}"

    def test_keyword_action_splitter_is_enabled_in_shipped_flows(self):
        assert all(
            profile["use_keyword_action_splitter"] is True
            for profile in Config_Orchestrator._ORCHESTRATION_FLOWS.values()
        )
        assert "enabled" not in Config_KeywordActionSplitter._KEYWORD_ACTION_SPLITTER

    def test_default_flow_is_native_all(self):
        assert FlowConfig.default_flow_name(_shipped_cfg()) == "NATIVE_ALL"

    @pytest.mark.parametrize(
        "legacy, current",
        [
            ("BGE_ENC_XENC", "NATIVE_ALL"),
            ("thorough_query_rewrite", "ENGLISH_ALL_MULTILEG"),
            ("ORIGINAL_LANGUAGE_VECTOR_ONLY", "ORIGINAL_VECTOR"),
        ],
    )
    def test_legacy_names_resolve(self, legacy, current):
        assert FlowConfig.select_flow_name(_shipped_cfg(), legacy) == current

    @pytest.mark.parametrize(
        "removed", ["TRANSLATION_FOCUSED", "ORIGINAL_LANGUAGE_ALL_RETRIEVERS"]
    )
    def test_removed_flows_fall_back_to_default(self, removed):
        assert FlowConfig.select_flow_name(_shipped_cfg(), removed) == "NATIVE_ALL"


class TestValidation:
    def test_unknown_key_and_bad_values_are_reported_together(self):
        flows = {
            "X": {
                "run_vectr": True,
                "run_bm25": "yes",
                "main_query_source": "FOO",
                "retrieval_language": "german",
                "force_retrieve_mode": "NOPE",
            }
        }
        with pytest.raises(ConfigurationError) as exc_info:
            FlowConfig.validate_flow_profiles(
                flows,
                allowed_retrieve_modes={"ALL"},
                allowed_query_sources=list(FlowConfig.QUERY_SOURCES),
            )
        message = str(exc_info.value)
        for fragment in (
            "run_vectr",
            "run_bm25",
            "main_query_source",
            "retrieval_language",
            "force_retrieve_mode",
        ):
            assert fragment in message

    def test_unknown_default_flow_rejected(self):
        with pytest.raises(ConfigurationError):
            FlowConfig.validate_flow_config(
                _shipped_cfg(_DEFAULT_ORCHESTRATION_FLOW="NOPE")
            )

    def test_alias_to_unknown_flow_rejected(self):
        with pytest.raises(ConfigurationError):
            FlowConfig.validate_flow_config(
                _shipped_cfg(_ORCHESTRATION_FLOW_ALIASES={"OLD": "MISSING"})
            )

    def test_alias_shadowing_real_flow_rejected(self):
        with pytest.raises(ConfigurationError):
            FlowConfig.validate_flow_config(
                _shipped_cfg(_ORCHESTRATION_FLOW_ALIASES={"ENGLISH_ALL": "NATIVE_ALL"})
            )


class TestValueHelpers:
    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("ALL", (True, True, True, True)),
            ("VECTOR", (True, False, False, False)),
            ("bm25", (False, True, False, False)),
            ("VECTOR_BM25", (True, True, False, False)),
        ],
    )
    def test_mode_flags(self, mode, expected):
        assert FlowConfig.mode_flags_for_local_retrievers(mode) == expected

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("", False),
            (None, False),
            ("ALL", False),
            ("WEB", False),
            ("VECTOR", True),
            ("bm25", True),
        ],
    )
    def test_narrows_local_retrievers(self, mode, expected):
        assert FlowConfig.narrows_local_retrievers(mode) is expected

    @pytest.mark.parametrize(
        "value, expected",
        [("english", "ENGLISH"), (" Native ", "NATIVE"), ("de", None), (None, None)],
    )
    def test_normalize_retrieval_language(self, value, expected):
        assert FlowConfig.normalize_retrieval_language(value) == expected


def _overrides() -> OrchestratorOverrides:
    return OrchestratorOverrides(SimpleNamespace(cfg=_shipped_cfg()))


def _session(**attrs: Any) -> SimpleNamespace:
    base: dict[str, Any] = {"orchestrator_flow": "ENGLISH_ALL"}
    base.update(attrs)
    return SimpleNamespace(**base)


class TestOverrides:
    def test_flow_baseline(self):
        profile = _overrides().resolve_orchestration_flow_profile(_session())
        assert profile.name == "ENGLISH_ALL"
        assert profile.retrieval_language_mode == "ENGLISH"
        assert profile.rerank_query_source == "REWRITTEN_QUERY"
        assert profile.retrieve_mode_override == ""

    def test_missing_keyword_action_splitter_switch_defaults_to_enabled(self):
        flows = {
            **Config_Orchestrator._ORCHESTRATION_FLOWS,
            "ENGLISH_ALL": {
                key: value
                for key, value in Config_Orchestrator._ORCHESTRATION_FLOWS[
                    "ENGLISH_ALL"
                ].items()
                if key != "use_keyword_action_splitter"
            },
        }
        cfg = _shipped_cfg(_ORCHESTRATION_FLOWS=flows)
        profile = OrchestratorOverrides(
            SimpleNamespace(cfg=cfg)
        ).resolve_orchestration_flow_profile(cast(Session, _session()))
        assert profile.enable_keyword_action_splitter is True

    def test_legacy_flow_name_is_persisted_as_new_name(self):
        session = _session(orchestrator_flow="THOROUGH_QUERY_REWRITE")
        profile = _overrides().resolve_orchestration_flow_profile(session)
        assert profile.name == "ENGLISH_ALL_MULTILEG"
        assert session.orchestrator_flow == "ENGLISH_ALL_MULTILEG"

    def test_session_rerank_source_and_language_override_flow(self):
        overrides = _overrides()
        session = _session(
            rerank_query_source="original_query", retrieval_language_override="native"
        )
        profile = overrides.resolve_orchestration_flow_profile(session)
        effective = overrides.resolve_effective_flow_profile_for_session(
            session, profile
        )
        assert effective.rerank_query_source == "ORIGINAL_QUERY"
        assert effective.retrieval_language_mode == "NATIVE"

    def test_invalid_session_values_are_ignored(self):
        overrides = _overrides()
        session = _session(
            rerank_query_source="BOGUS", retrieval_language_override="german"
        )
        profile = overrides.resolve_orchestration_flow_profile(session)
        effective = overrides.resolve_effective_flow_profile_for_session(
            session, profile
        )
        assert effective.rerank_query_source == profile.rerank_query_source
        assert effective.retrieval_language_mode == profile.retrieval_language_mode


class TestNarrowRetrievers:
    def _profile(self, forced: str):
        overrides = _overrides()
        session = _session(force_retrieve_mode=forced)
        profile = overrides.resolve_orchestration_flow_profile(session)
        return overrides.resolve_effective_flow_profile_for_session(session, profile)

    def test_forced_bm25_disables_other_local_retrievers(self):
        profile = self._profile("BM25")
        narrowed = OrchestratorOverrides.narrow_retrievers_for_mode(profile, "BM25")
        assert (
            narrowed.enable_vector_retriever,
            narrowed.enable_bm25_retriever,
            narrowed.enable_graph_retriever,
            narrowed.enable_regex_retriever,
        ) == (False, True, False, False)

    def test_narrowing_never_reenables_a_disabled_retriever(self):
        overrides = _overrides()
        session = _session(
            orchestrator_flow="ENGLISH_BM25", force_retrieve_mode="VECTOR_BM25"
        )
        profile = overrides.resolve_effective_flow_profile_for_session(
            session, overrides.resolve_orchestration_flow_profile(session)
        )
        narrowed = OrchestratorOverrides.narrow_retrievers_for_mode(
            profile, "VECTOR_BM25"
        )
        assert narrowed.enable_vector_retriever is False
        assert narrowed.enable_bm25_retriever is True

    @pytest.mark.parametrize("forced", ["ALL", "WEB"])
    def test_all_and_web_do_not_narrow(self, forced):
        profile = self._profile(forced)
        assert (
            OrchestratorOverrides.narrow_retrievers_for_mode(profile, forced) == profile
        )

    def test_without_forced_mode_nothing_narrows(self):
        profile = self._profile("")
        assert (
            OrchestratorOverrides.narrow_retrievers_for_mode(profile, "VECTOR")
            == profile
        )
