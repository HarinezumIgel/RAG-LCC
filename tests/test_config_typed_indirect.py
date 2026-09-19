# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
"""Tests for Config typed indirect accessors."""

import threading
from typing import Any

import pytest

import Config.Config as cfg_mod


class _StubPrettyWriter:
    def write(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        return None


def _build_config(raw_cfg: dict[str, Any]) -> cfg_mod.Config:
    cfg = object.__new__(cfg_mod.Config)
    cfg._initialized = True
    cfg._lock = threading.RLock()
    cfg.cfg = raw_cfg
    cfg.args = {}
    cfg.pretty = _StubPrettyWriter()
    return cfg


def test_indirect_list_resolves_nested_value() -> None:
    cfg = _build_config(
        {
            "_ARGOS_DEFINITIONS": {
                "ACTIVE_LANGUAGES": ["en", "de"],
            }
        }
    )

    assert cfg.indirect_list("_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES") == ["en", "de"]


def test_indirect_dict_follows_plain_alias_chain() -> None:
    cfg = _build_config(
        {
            "_LANG_CODE_TO_NAME": {"en": "english", "de": "german"},
            "_LANG_ALIAS": "_LANG_CODE_TO_NAME",
            "_ARGOS_DEFINITIONS": {
                "LANG_CODE_TO_NAME": "_LANG_ALIAS",
            },
        }
    )

    got = cfg.indirect_dict("_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME")

    assert got == {"en": "english", "de": "german"}


def test_indirect_dict_follows_prefixed_alias_chain() -> None:
    cfg = _build_config(
        {
            "_LANG_CODE_TO_NAME": {"en": "english"},
            "_LANG_ALIAS": "$_LANG_CODE_TO_NAME",
            "_ARGOS_DEFINITIONS": {
                "LANG_CODE_TO_NAME": "_LANG_ALIAS",
            },
        }
    )

    got = cfg.indirect_dict("_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME")

    assert got == {"en": "english"}


def test_get_list_rejects_dollar_prefixed_key() -> None:
    cfg = _build_config(
        {
            "_ARGOS_DEFINITIONS": {
                "ACTIVE_LANGUAGES": ["en"],
            }
        }
    )

    with pytest.raises(TypeError, match="indirect_list"):
        cfg.get_list("$_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES")


def test_get_dict_rejects_dollar_prefixed_key() -> None:
    cfg = _build_config(
        {
            "_ARGOS_DEFINITIONS": {
                "LANG_CODE_TO_NAME": {"en": "english"},
            }
        }
    )

    with pytest.raises(TypeError, match="indirect_dict"):
        cfg.get_dict("$_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME")


def test_indirect_get_resolves_dotted_alias_target() -> None:
    cfg = _build_config(
        {
            "_SPACY": {
                "GRAPH": {
                    "spacy_model": "en_core_web_sm",
                }
            },
            "_GRAPH_INDEX": {
                "spacy_model": "$_SPACY.GRAPH.spacy_model",
            },
        }
    )

    value, slot = cfg.indirect_get("_GRAPH_INDEX.spacy_model")

    assert value == "en_core_web_sm"
    assert slot == "_SPACY.GRAPH.spacy_model"


def test_indirect_dict_resolves_dotted_alias_target() -> None:
    cfg = _build_config(
        {
            "_SPACY": {
                "REGEX": {
                    "spacy_models_by_language": {"de": "de_core_news_sm"},
                }
            },
            "_REGEX_INDEX": {
                "spacy_models_by_language": "$_SPACY.REGEX.spacy_models_by_language",
            },
        }
    )

    value = cfg.indirect_dict("_REGEX_INDEX.spacy_models_by_language")

    assert value == {"de": "de_core_news_sm"}
