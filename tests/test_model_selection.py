"""Tests for Helpers.ModelSelection - the shared _ACTIVE_* -> _MODELS lookup."""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import Configuration.Config_Models as Config_Models
from Helpers.ModelSelection import (
    active_model_entries,
    apply_provider_slot,
    model_selectors,
    resolve_model_entry,
)


class _Cfg:
    def __init__(self, values: dict[str, Any]):
        self.values = values

    def get_str(self, key: str, default: str = "") -> str:
        return str(self.values.get(key, default))

    def get_dict(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def _real_cfg(**overrides: Any) -> _Cfg:
    values = {name: getattr(Config_Models, name) for name in model_selectors()}
    values["_MODELS"] = Config_Models._MODELS
    values.update(overrides)
    return _Cfg(values)


def test_selectors_come_from_config_models():
    names = model_selectors()
    assert "_ACTIVE_EMBED" in names and "_ACTIVE_ENDPOINT" in names
    assert all(name.startswith("_ACTIVE_") for name in names)
    assert set(names) == {n for n in vars(Config_Models) if n.startswith("_ACTIVE_")}


def test_resolve_selector_derives_role():
    impl, role, entry = resolve_model_entry(_real_cfg(), "_ACTIVE_EMBED")
    assert (impl, role) == (Config_Models._ACTIVE_EMBED, "_EMBED")
    assert entry is Config_Models._MODELS[impl]["_EMBED"]


def test_resolve_direct_impl_needs_role():
    with pytest.raises(ValueError, match="explicit lookup key"):
        resolve_model_entry(_real_cfg(), "vllm")
    assert resolve_model_entry(_real_cfg(), "vllm", "_VLLM")[1] == "_VLLM"


def test_resolve_unknown_impl_and_role():
    with pytest.raises(ValueError, match="Unknown impl"):
        resolve_model_entry(_real_cfg(_ACTIVE_LLM="nope"), "_ACTIVE_LLM")
    with pytest.raises(ValueError, match="Role '_CROSS' not found"):
        resolve_model_entry(_real_cfg(_ACTIVE_CROSS="mistral"), "_ACTIVE_CROSS")


@pytest.mark.parametrize("endpoint", ["vllm", "ollama"])
def test_active_entries_include_endpoint_provider_role(endpoint):
    entries = active_model_entries(_real_cfg(_ACTIVE_ENDPOINT=endpoint))
    endpoint_keys = {k for k in entries if k.startswith(f"{endpoint}.")}
    assert endpoint_keys == {f"{endpoint}._{endpoint.upper()}"}


def test_endpoint_impls_have_only_the_provider_role():
    for endpoint in ("ollama", "vllm"):
        assert set(Config_Models._MODELS[endpoint]) == {f"_{endpoint.upper()}"}


def test_active_entries_skip_inactive_models_and_empty_selectors():
    entries = active_model_entries(_real_cfg(_ACTIVE_LLM_JUDGE=""))
    assert not any(k.endswith("._LLM_JUDGE") for k in entries)
    impls = {k.split(".")[0] for k in entries}
    assert impls == {
        getattr(Config_Models, s) for s in model_selectors() if s != "_ACTIVE_LLM_JUDGE"
    } | {"openwebui", "ragchatservice"}


def test_service_entries_are_always_active_without_selector():
    assert "_ACTIVE_OPENWEBUI" not in model_selectors()
    assert "_ACTIVE_RAGCHATSERVICE" not in model_selectors()
    entries = active_model_entries(_real_cfg())
    assert (
        entries["openwebui._OPENWEBUI"]
        is Config_Models._MODELS["openwebui"]["_OPENWEBUI"]
    )
    assert "ragchatservice._RAGCHATSERVICE" in entries
    # Unselected provider entries stay inactive.
    assert not any(
        k.startswith("ollama.")
        for k in active_model_entries(_real_cfg(_ACTIVE_ENDPOINT="vllm"))
    )


def test_active_entries_report_all_problems():
    with pytest.raises(ValueError) as exc:
        active_model_entries(_real_cfg(_ACTIVE_LLM="nope", _ACTIVE_ENDPOINT="nix"))
    assert "_ACTIVE_LLM = 'nope'" in str(exc.value)
    assert "_ACTIVE_ENDPOINT = 'nix'" in str(exc.value)


@pytest.mark.parametrize(
    ("endpoint", "model"),
    [("ollama", "mistral:7b"), ("vllm", "mistral_7b"), ("VLLM ", "mistral_7b")],
)
def test_provider_slot_overrides_role_keys(endpoint, model):
    entry = {
        "PROMPT": "p",
        "TIMEOUT": 1,
        "OLLAMA": {"MODEL": "mistral:7b"},
        "VLLM": {"MODEL": "mistral_7b", "TIMEOUT": 2},
    }
    flat = apply_provider_slot(entry, endpoint)
    assert flat["MODEL"] == model and flat["PROMPT"] == "p"
    assert flat["TIMEOUT"] == (2 if model == "mistral_7b" else 1)
    assert "OLLAMA" not in flat and "VLLM" not in flat
    assert "OLLAMA" in entry  # input untouched


def test_provider_slot_missing_keeps_role_keys():
    entry = {"MODEL": "hf/id", "VLLM": {"MODEL": "alias"}}
    assert apply_provider_slot(entry, "ollama") == {"MODEL": "hf/id"}
    assert apply_provider_slot(entry, "vllm") == {"MODEL": "alias"}


def test_no_legacy_endpoint_model_keys_in_config():
    for impl, roles in Config_Models._MODELS.items():
        for role, entry in roles.items():
            assert not {"MODEL_OLLAMA", "MODEL_VLLM"} & set(entry), f"{impl}.{role}"


@pytest.mark.parametrize("role", ["_LLM", "_LLM_REWRITE_PROMPT", "_LLM_CHK"])
def test_qwen_7b_provider_endpoint_and_license(role):
    _, resolved_role, entry = resolve_model_entry(
        _real_cfg(),
        "qwen2_5_7b_instruct",
        role,
    )
    assert resolved_role == role
    for provider in ("vllm", "ollama"):
        model = apply_provider_slot(entry, provider)["MODEL"]
        assert model == entry[provider.upper()]["MODEL"]
        assert isinstance(model, str) and model.strip()
    assert entry["LICENSE"] == "Apache-2.0"
    assert entry["LICENSE_URL"] == (
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/LICENSE"
    )
