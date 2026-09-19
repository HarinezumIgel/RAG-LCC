# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
"""Tests for Helpers.LanguageConfig active-language resolution helpers."""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.LanguageConfig import get_active_argos_pairs, get_active_language_codes


class StubCfg:
    def __init__(
        self,
        *,
        active_languages: list[str] | None = None,
        pairs: list[tuple[str, str]] | None = None,
        code_to_name: dict[str, str] | None = None,
    ) -> None:
        self._active = active_languages
        self._pairs = pairs or []
        self._map = code_to_name or {
            "en": "english",
            "de": "german",
            "fr": "french",
        }

    def get_list(self, key: str, default: Any = None, silent: bool = False) -> Any:
        _ = silent
        if key == "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES":
            if self._active is None:
                return default if default is not None else []
            return list(self._active)
        if key == "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES":
            return [list(pair) for pair in self._pairs]
        return default if default is not None else []

    def get_dict(self, key: str, default: Any = None, silent: bool = False) -> Any:
        _ = silent
        if key == "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME":
            return dict(self._map)
        return default if default is not None else {}


class TestActiveLanguageCodes:
    def test_normalizes_codes_and_names(self) -> None:
        cfg = StubCfg(active_languages=["english", "de", "FR"])

        result = get_active_language_codes(cfg)

        assert result == {"en", "de", "fr"}

    def test_defaults_to_en_when_missing(self) -> None:
        cfg = StubCfg(active_languages=None, pairs=[("en", "de")])

        result = get_active_language_codes(cfg)

        assert result == {"en"}


class TestActiveArgosPairs:
    def test_filters_pairs_to_active_languages(self) -> None:
        cfg = StubCfg(
            active_languages=["en", "de"],
            pairs=[("de", "en"), ("en", "de"), ("fr", "en")],
        )

        result = get_active_argos_pairs(cfg)

        assert result == [("de", "en"), ("en", "de")]

    def test_normalizes_name_pairs(self) -> None:
        cfg = StubCfg(
            active_languages=["en", "de"],
            pairs=[("german", "english")],
        )

        result = get_active_argos_pairs(cfg)

        assert result == [("de", "en")]
