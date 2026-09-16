# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for SharedHelpers translation-pair selection edge cases."""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.SharedHelpers import SharedHelpers


class _StubPretty:
    def write(self, *a: Any, **k: Any) -> None:
        _ = a
        _ = k


class _Translator:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def translate(self, text: str) -> str:
        return f"{self.tag}:{text}"


class _Lang:
    def __init__(self, code: str, mapping: dict[str, _Translator | None]) -> None:
        self.code = code
        self._mapping = mapping

    def get_translation(self, tgt_lang: "_Lang") -> _Translator | None:
        return self._mapping.get(tgt_lang.code)


def _make_shared(
    installed_langs: dict[str, _Lang],
    *,
    lang_name_to_code: dict[str, str] | None = None,
) -> SharedHelpers:
    shared = SharedHelpers.__new__(SharedHelpers)
    shared.lang_load_failed = False
    shared.installed_langs = installed_langs
    shared.warned_langs = set()
    shared.warned_pairs = set()
    shared.pretty = _StubPretty()
    shared.lang_name_to_code = lang_name_to_code or {}
    return shared


def test_get_translation_does_not_use_identity_fallback_for_concrete_source() -> None:
    en_identity = _Translator("identity")
    en = _Lang("en", {"en": en_identity})
    de = _Lang("de", {"en": None})
    shared = _make_shared({"de": de, "en": en})

    tr = shared._get_translation("de", "en")

    assert tr is None


def test_get_translation_allows_source_fallback_when_source_is_auto() -> None:
    en_identity = _Translator("identity")
    fr_to_en = _Translator("fr_to_en")
    en = _Lang("en", {"en": en_identity})
    fr = _Lang("fr", {"en": fr_to_en})
    shared = _make_shared({"fr": fr, "en": en})

    tr = shared._get_translation("auto", "en")

    assert tr is fr_to_en
