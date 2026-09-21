# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for Chatter grounding guard helper methods.

These tests compile selected methods from ``src/Chat/Chatter.py`` into a
lightweight shell class so we can verify grounding behavior without importing
full runtime dependencies.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "Chat", "Chatter.py")


def _extract_method(source: str, method_name: str) -> str:
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    @(?:staticmethod|classmethod|property)|\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in Chatter.py"
    return textwrap.dedent(match.group(1))


with open(_SRC, encoding="utf-8") as _fh:
    _CHATTER_SOURCE = _fh.read()

_EXTRACT_ANSWER_SECTION_SRC = _extract_method(
    _CHATTER_SOURCE, "_extract_answer_section"
)
_BUILD_NO_EVIDENCE_MESSAGE_SRC = _extract_method(
    _CHATTER_SOURCE, "_build_no_evidence_message"
)
_BUILD_SHORT_NOTICE_SRC = _extract_method(
    _CHATTER_SOURCE, "_build_grounding_skipped_short_answer_notice"
)
_COLLECT_WEB_GROUNDING_TEXTS_SRC = _extract_method(
    _CHATTER_SOURCE, "_collect_web_grounding_texts"
)
_IS_SHORT_FOR_GROUNDING_SRC = _extract_method(
    _CHATTER_SOURCE, "_is_answer_too_short_for_grounding"
)
_HAS_GROUNDED_EVIDENCE_SRC = _extract_method(_CHATTER_SOURCE, "_has_grounded_evidence")


class _StubFileUtils:
    def __init__(self, lang: str = "english") -> None:
        self.lang = lang

    def get_user_text_language(
        self,
        _text: str,
        output: str = "nltk",
        native_lang: str | None = None,
    ) -> str:
        _ = output
        _ = native_lang
        return self.lang


def _compile_methods() -> tuple[Any, Any, Any, Any, Any, Any]:
    class _SymbolsStub:
        @staticmethod
        def sym_warning() -> str:
            return "⚠ "

    ns: dict[str, Any] = {"Session": object, "Symbols": _SymbolsStub}
    exec(compile(_EXTRACT_ANSWER_SECTION_SRC, _SRC, "exec"), ns)
    exec(compile(_BUILD_NO_EVIDENCE_MESSAGE_SRC, _SRC, "exec"), ns)
    exec(compile(_BUILD_SHORT_NOTICE_SRC, _SRC, "exec"), ns)
    exec(compile(_COLLECT_WEB_GROUNDING_TEXTS_SRC, _SRC, "exec"), ns)
    exec(compile(_IS_SHORT_FOR_GROUNDING_SRC, _SRC, "exec"), ns)
    exec(compile(_HAS_GROUNDED_EVIDENCE_SRC, _SRC, "exec"), ns)
    return (
        ns["_extract_answer_section"],
        ns["_build_no_evidence_message"],
        ns["_build_grounding_skipped_short_answer_notice"],
        ns["_collect_web_grounding_texts"],
        ns["_is_answer_too_short_for_grounding"],
        ns["_has_grounded_evidence"],
    )


(
    _extract_answer_section,
    _build_no_evidence_message,
    _build_short_notice,
    _collect_web_grounding_texts,
    _is_short_for_grounding,
    _has_grounded_evidence,
) = _compile_methods()


class _Shell:
    _collect_web_grounding_texts = _collect_web_grounding_texts
    _extract_answer_section = staticmethod(_extract_answer_section)
    _build_no_evidence_message = staticmethod(_build_no_evidence_message)
    _build_grounding_skipped_short_answer_notice = staticmethod(_build_short_notice)
    _is_answer_too_short_for_grounding = _is_short_for_grounding
    _has_grounded_evidence = _has_grounded_evidence

    def __init__(self, lang: str = "english") -> None:
        self.fileUtils = _StubFileUtils(lang=lang)


class _SessionStub:
    def __init__(
        self,
        retrieval_language: str = "english",
        last_chosen_chunks: list[Any] | None = None,
    ) -> None:
        self.retrieval_language = retrieval_language
        self.last_chosen_chunks = last_chosen_chunks or []


def _install_grounder_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    vm_pkg = types.ModuleType("VisualMarkers")
    vm_mod = types.ModuleType("VisualMarkers.AnswerGrounder")

    class _Grounder:
        def find_grounded_sentences(self, answer: str, chunks: list[str]) -> list[str]:
            joined_chunks = " ".join(chunks)
            if "direct_hit" in answer:
                return ["direct_hit"]
            if "translated_hit" in answer:
                return ["translated_hit"]
            if "web_hit" in joined_chunks and (
                "web_answer" in answer or "translated_web_answer" in answer
            ):
                return ["web_hit"]
            return []

    vm_mod.AnswerGrounder = _Grounder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "VisualMarkers", vm_pkg)
    monkeypatch.setitem(sys.modules, "VisualMarkers.AnswerGrounder", vm_mod)


def _install_grounder_no_match_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    vm_pkg = types.ModuleType("VisualMarkers")
    vm_mod = types.ModuleType("VisualMarkers.AnswerGrounder")

    class _Grounder:
        def find_grounded_sentences(
            self, _answer: str, _chunks: list[str]
        ) -> list[str]:
            return []

    vm_mod.AnswerGrounder = _Grounder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "VisualMarkers", vm_pkg)
    monkeypatch.setitem(sys.modules, "VisualMarkers.AnswerGrounder", vm_mod)


def _install_shared_stub(monkeypatch: pytest.MonkeyPatch, translated: str) -> None:
    c_pkg = types.ModuleType("Compliance")
    c_mod = types.ModuleType("Compliance.SharedHelpers")

    class _SharedHelpers:
        def translate_text(
            self,
            _text: str,
            target_lang: str,
            source_lang: str = "auto",
        ) -> str:
            _ = target_lang
            _ = source_lang
            return translated

    c_mod.SharedHelpers = _SharedHelpers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Compliance", c_pkg)
    monkeypatch.setitem(sys.modules, "Compliance.SharedHelpers", c_mod)


def _install_grounder_threshold_stub(
    monkeypatch: pytest.MonkeyPatch,
    min_sentence_tokens: int,
) -> None:
    vm_pkg = types.ModuleType("VisualMarkers")
    vm_mod = types.ModuleType("VisualMarkers.AnswerGrounder")

    class _Grounder:
        def __init__(self) -> None:
            self.min_sentence_tokens = min_sentence_tokens

    vm_mod.AnswerGrounder = _Grounder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "VisualMarkers", vm_pkg)
    monkeypatch.setitem(sys.modules, "VisualMarkers.AnswerGrounder", vm_mod)


class TestGroundingGuardHelpers:
    def test_extract_answer_section_removes_sources_block(self) -> None:
        text = "### Answer\nCats sleep.\n\n### Sources\n- file.md"
        body = _Shell._extract_answer_section(text)
        assert body == "### Answer\nCats sleep."

    def test_build_no_evidence_message_contains_context_phrase(self) -> None:
        msg = _Shell._build_no_evidence_message()
        assert "couldn't find evidence" in msg.lower()
        assert "retrieved context" in msg.lower()

    def test_build_grounding_skipped_short_answer_notice_mentions_too_short(
        self,
    ) -> None:
        msg = _Shell._build_grounding_skipped_short_answer_notice()
        assert msg.startswith("⚠")
        assert "grounding skipped" in msg.lower()
        assert "too short" in msg.lower()

    def test_is_answer_too_short_for_grounding_true_for_short_answer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_grounder_threshold_stub(monkeypatch, min_sentence_tokens=5)
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\nCats, dogs, foxes"
        )

        assert is_short is True

    def test_is_answer_too_short_for_grounding_false_for_long_sentence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_grounder_threshold_stub(monkeypatch, min_sentence_tokens=5)
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\n"
            "Hedgehogs are small nocturnal mammals with protective spines and adaptable foraging habits."
        )

        assert is_short is False

    def test_is_answer_too_short_for_grounding_true_for_brief_bullet_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_grounder_threshold_stub(monkeypatch, min_sentence_tokens=5)
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\n- cats\n- dogs\n- apes (gorillas and orangutans mentioned)"
        )

        assert is_short is True

    def test_has_grounded_evidence_true_when_direct_overlap_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        shell = _Shell(lang="english")
        session = _SessionStub(retrieval_language="english")

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\ndirect_hit\n\n### Sources\n- A.txt",
            ["chunk text"],
            "english",
        )

        assert ok is True

    def test_has_grounded_evidence_uses_translated_proxy_when_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="german")
        session = _SessionStub(retrieval_language="english")

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAntwort ohne overlap\n\n### Sources\n- A.txt",
            ["chunk text"],
            "german",
        )

        assert ok is True

    def test_has_grounded_evidence_false_when_no_overlap_and_no_translation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        _install_shared_stub(monkeypatch, translated="")
        shell = _Shell(lang="german")
        session = _SessionStub(retrieval_language="english")

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAntwort ohne overlap\n\n### Sources\n- A.txt",
            ["chunk text"],
            "german",
        )

        assert ok is False

    def test_has_grounded_evidence_true_when_no_chunk_texts(self) -> None:
        shell = _Shell(lang="english")
        session = _SessionStub(retrieval_language="english")
        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAny answer\n\n### Sources\n- A.txt",
            [],
            "english",
        )
        assert ok is True

    def test_has_grounded_evidence_true_when_web_chunk_overlaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        shell = _Shell(lang="english")
        web_doc = types.SimpleNamespace(
            metadata={"Source": "Web", "snippet": "web_hit snippet"},
            page_content="web_hit full page",
        )
        session = _SessionStub(
            retrieval_language="english",
            last_chosen_chunks=[web_doc],
        )

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nweb_answer\n\n### Sources\n- https://example.com",
            ["unrelated local chunk"],
            "english",
        )

        assert ok is True

    def test_has_grounded_evidence_uses_translated_proxy_for_web_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        _install_shared_stub(monkeypatch, translated="translated_web_answer")
        shell = _Shell(lang="german")
        web_doc = types.SimpleNamespace(
            metadata={"Source": "Web", "snippet": "web_hit snippet"},
            page_content="web_hit full page",
        )
        session = _SessionStub(
            retrieval_language="english",
            last_chosen_chunks=[web_doc],
        )

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAntwort ohne overlap\n\n### Sources\n- https://example.com",
            ["unrelated local chunk"],
            "german",
        )

        assert ok is True

    def test_has_grounded_evidence_relaxed_overlap_paraphrase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_no_match_stub(monkeypatch)
        shell = _Shell(lang="english")
        session = _SessionStub(retrieval_language="english")

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nHedgehogs inhabit suburban gardens and farm edges.\n\n### Sources\n- A.txt",
            ["Hedgehogs are often found in suburban gardens and agricultural edges."],
            "english",
        )

        assert ok is True

    def test_has_grounded_evidence_relaxed_overlap_after_translation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_no_match_stub(monkeypatch)
        _install_shared_stub(
            monkeypatch,
            translated="Hedgehogs inhabit suburban gardens and farm edges.",
        )
        shell = _Shell(lang="german")
        session = _SessionStub(retrieval_language="english")

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAntwort ohne overlap\n\n### Sources\n- A.txt",
            ["Hedgehogs are often found in suburban gardens and agricultural edges."],
            "german",
        )

        assert ok is True
