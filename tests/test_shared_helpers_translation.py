# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for SharedHelpers translation-pair selection edge cases."""

import os
import sys
from typing import Any, cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.SharedHelpers import SharedHelpers


class _StubPretty:
    def write(self, *a: Any, **k: Any) -> None:
        _ = a
        _ = k


class _CollectingPretty:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write(self, *a: Any, **k: Any) -> None:
        self.calls.append((a, k))


class _Translator:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def translate(self, text: str) -> str:
        return f"{self.tag}:{text}"


class _FailingTranslator:
    def translate(self, _text: str) -> str:
        raise RuntimeError("missing model files")


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
    pretty: Any | None = None,
) -> SharedHelpers:
    shared = SharedHelpers.__new__(SharedHelpers)
    shared.lang_load_failed = False
    shared.installed_langs = installed_langs
    shared._warned_by_category = {}
    shared.warned_langs = shared._warning_bucket("argos.target-language-missing")
    shared.warned_unsupported_langs = shared._warning_bucket(
        "argos.runtime-language-unsupported"
    )
    shared.warned_pairs = shared._warning_bucket("argos.translation-pair-missing")
    shared.warned_runtime_pairs = shared._warning_bucket(
        "argos.translation-runtime-failure"
    )
    shared.warned_spacy_model_unavailable = shared._warning_bucket(
        "spacy.model-unavailable"
    )
    shared.translation_cache = {}
    shared.pretty = cast(Any, pretty or _StubPretty())
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


def test_translate_text_warns_when_runtime_translation_fails_once() -> None:
    failing = _FailingTranslator()
    en = _Lang("en", {"de": failing})
    de = _Lang("de", {})
    pretty = _CollectingPretty()
    shared = _make_shared({"en": en, "de": de}, pretty=pretty)

    first = shared.translate_text("hello", target_lang="de", source_lang="en")
    second = shared.translate_text("world", target_lang="de", source_lang="en")

    assert first == "hello"
    assert second == "world"

    messages = [
        str(args[2])
        for args, _kwargs in pretty.calls
        if len(args) >= 3 and args[1] == "Translate"
    ]
    runtime_warnings = [
        message
        for message in messages
        if "Argos translation failed for en->de" in message
    ]
    assert len(runtime_warnings) == 1


def test_translate_text_can_suppress_fallback_warning() -> None:
    en = _Lang("en", {})
    pretty = _CollectingPretty()
    shared = _make_shared({"en": en}, pretty=pretty)

    translated = shared.translate_text(
        "what do hedgehogs eat",
        target_lang="de",
        source_lang="en",
        warn_on_fallback=False,
    )

    assert translated == "what do hedgehogs eat"

    warning_messages = [
        str(args[2])
        for args, _kwargs in pretty.calls
        if len(args) >= 3 and args[1] == "Translate"
    ]
    assert warning_messages == []
