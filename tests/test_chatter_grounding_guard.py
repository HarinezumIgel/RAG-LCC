# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for the grounding guard (``Grounding.Grounder``) and Chatter confidence notices.

Grounder reads its thresholds through ``Config`` lookups; the sentence
matcher is stubbed where a test controls which sentences are grounded.
Confidence-notice methods are compiled from ``src/Chat/Chatter.py``.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Grounding.Grounder import Grounder
from Config.Config import Config

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

_RESOLVE_ANSWER_CONFIDENCE_PAYLOAD_SRC = _extract_method(
    _CHATTER_SOURCE, "_resolve_answer_confidence_payload"
)
_BUILD_ANSWER_CONFIDENCE_NOTICE_SRC = _extract_method(
    _CHATTER_SOURCE, "_build_answer_confidence_notice"
)


def _compile_confidence_methods() -> tuple[Any, Any]:
    class _SymbolsStub:
        @staticmethod
        def sym_icon(kind: str) -> str:
            icon_map = {"HIGH": "🟢 ", "MEDIUM": "🟡 ", "LOW": "🟠 "}
            return icon_map.get(str(kind or "").upper(), "ℹ ")

    ns: dict[str, Any] = {
        "Session": object,
        "Symbols": _SymbolsStub,
        "AnswerConfidenceResult": object,
        "Any": Any,
    }
    exec(compile(_RESOLVE_ANSWER_CONFIDENCE_PAYLOAD_SRC, _SRC, "exec"), ns)
    exec(compile(_BUILD_ANSWER_CONFIDENCE_NOTICE_SRC, _SRC, "exec"), ns)
    ns["Chatter"] = types.SimpleNamespace(
        _resolve_answer_confidence_payload=ns["_resolve_answer_confidence_payload"]
    )
    return (
        ns["_resolve_answer_confidence_payload"],
        ns["_build_answer_confidence_notice"],
    )


_resolve_answer_confidence_payload, _build_answer_confidence_notice = (
    _compile_confidence_methods()
)


class _ConfidenceShell:
    _resolve_answer_confidence_payload = staticmethod(
        _resolve_answer_confidence_payload
    )
    _build_answer_confidence_notice = staticmethod(_build_answer_confidence_notice)


def test_answer_uses_rich_markdown_renderer(capsys: pytest.CaptureFixture[str]) -> None:
    from Chat.Chatter import Chatter

    cfg = types.SimpleNamespace(get=lambda _key: {"bg": "ANSWER_BG", "fg": "ANSWER_FG"})
    chatter = types.SimpleNamespace(cfg=cfg)

    Chatter.print_llm_answer(
        chatter, "## Answer\n\n- **Supported**\n- Unsupported", 80, prefix="> "
    )

    output = capsys.readouterr().out
    assert ">" in output
    assert "Answer" in output
    assert "Supported" in output
    assert "Unsupported" in output


def test_default_answer_colors_cover_body_and_confidence() -> None:
    import io

    from rich.color import Color
    from rich.console import Console
    from rich.text import Text

    from Chat.Chatter import Chatter
    from Configuration.Config_Global import ANSWER_DISPLAY
    from Gui.RichAnswerRenderer import ANSWER_META_SEPARATOR, RichAnswerRenderer

    output = io.StringIO()
    console = Console(
        file=output, width=60, force_terminal=True, color_system="truecolor"
    )
    chatter = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            get=lambda key: {"ANSWER_DISPLAY": ANSWER_DISPLAY}[key]
        ),
        answer_renderer=RichAnswerRenderer(console),
    )
    Chatter.print_llm_answer(
        chatter,
        "Answer body." + ANSWER_META_SEPARATOR + "**Answer confidence:** MEDIUM",
        60,
        prefix="",
    )

    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert "Answer body." in lines[0]
    assert "MEDIUM" in lines[1]
    for line in lines:
        decoded = Text.from_ansi(line)
        assert decoded.cell_len == 60
        for offset in range(len(decoded)):
            style = decoded.get_style_at_offset(console, offset)
            assert style.bgcolor == Color.from_ansi(238)
            assert style.color == Color.from_ansi(252)


def test_rich_renderer_preserves_confidence_markdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter

    cfg = types.SimpleNamespace(get=lambda _key: {"bg": "ANSWER_BG", "fg": "ANSWER_FG"})
    chatter = types.SimpleNamespace(cfg=cfg)
    confidence = (
        "**Answer confidence:**\n\n"
        "🟠 LOW\n"
        "Accuracy, answer completeness, reasoning validity, and citation support: not assessed."
    )

    Chatter.print_llm_answer(chatter, confidence, 100)

    output = capsys.readouterr().out
    assert "Answer confidence:" in output
    assert "LOW" in output
    assert "citation support: not assessed." in " ".join(output.split())


def test_rich_renderer_wraps_markdown_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter

    chatter = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            get=lambda _key: {"bg": "ANSWER_BG", "fg": "ANSWER_FG"}
        )
    )
    summary = (
        "## Summary of Hedgehogs.pdf\n\n"
        "Hedgehogs are small nocturnal mammals with distinctive spines. "
        "They eat invertebrates and live in gardens. "
        "Ongoing research explores their role in ecosystems."
    )

    Chatter.print_llm_answer(chatter, summary, 60)

    output = capsys.readouterr().out
    assert "Summary of Hedgehogs.pdf" in output
    assert "ecosystems." in output
    assert "## Summary" not in output


def test_answer_renders_nested_bullets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter

    chatter = types.SimpleNamespace(cfg=types.SimpleNamespace(get=lambda _key: {}))
    answer = "- **Cats.md**\n" "  * cats\n" "    + kittens"

    Chatter.print_llm_answer(chatter, answer, 80, prefix="> ")

    output = capsys.readouterr().out
    assert "Cats.md" in output
    assert "cats" in output
    assert "kittens" in output


def test_rich_renderer_wraps_long_bullet_and_keeps_answer_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter

    chatter = types.SimpleNamespace(cfg=types.SimpleNamespace(get=lambda _key: {}))
    answer = "  - Hedgehogs live in gardens and eat insects and other small animals."

    Chatter.print_llm_answer(chatter, answer, 30, prefix="> ")

    output = capsys.readouterr().out
    for word in ("Hedgehogs", "insects", "animals."):
        assert word in output


@pytest.mark.parametrize(
    ("opening", "inner_fence", "closing"),
    [
        ("```text", "~~~", "```"),
        ("~~~~text", "~~~", "~~~~"),
        ("````text", "```", "````"),
    ],
)
def test_rich_renderer_preserves_list_markers_inside_fenced_code(
    capsys: pytest.CaptureFixture[str],
    opening: str,
    inner_fence: str,
    closing: str,
) -> None:
    from Chat.Chatter import Chatter

    chatter = types.SimpleNamespace(cfg=types.SimpleNamespace(get=lambda _key: {}))
    code = [opening, "- literal", inner_fence, "  * literal", "+ literal", closing]
    answer = "\n".join(["- before", *code, "+ after"])

    Chatter.print_llm_answer(chatter, answer, 80, prefix="> ")

    output = capsys.readouterr().out
    assert "before" in output
    assert "after" in output
    assert "- literal" in output
    assert "* literal" in output
    assert "+ literal" in output


def test_rich_renderer_preserves_non_list_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter

    chatter = types.SimpleNamespace(cfg=types.SimpleNamespace(get=lambda _key: {}))
    lines = ["---", "- - -", "* * *", "-10", "+5", "1. Dogs", "Use - cats inline."]

    Chatter.print_llm_answer(chatter, "\n".join(lines), 80, prefix="> ")

    output = capsys.readouterr().out
    for line in ("-10", "+5", "Dogs", "Use - cats inline."):
        assert line in output


class _Shell:
    """A real Grounder with recorded output and a configurable language detector."""

    def __init__(self, lang: str = "english") -> None:
        self.lang = lang
        self.messages: list[str] = []
        self.detected: list[str] = []
        Grounder._reset()
        self.grounder = Grounder(cfg=Config())
        Grounder._reset()
        writer = types.SimpleNamespace(
            write=lambda _severity, _stage, message, **_kwargs: self.messages.append(
                message
            )
        )
        self.grounder.pretty = writer  # type: ignore[assignment]
        self.grounder.pretty_always = writer  # type: ignore[assignment]

    def detect(self, _text: str, _native: str, stage: str) -> str:
        self.detected.append(stage)
        return self.lang

    def _has_grounded_evidence(
        self, session: Any, answer: str, chunks: list[str], response_language: str
    ) -> bool:
        return self.grounder.has_grounded_evidence(
            session, answer, chunks, response_language, self.detect
        )

    def _has_local_grounding_spans(self, session: Any) -> bool:
        return self.grounder.has_local_spans(session)

    def _is_answer_too_short_for_grounding(self, answer: str) -> bool:
        return self.grounder.is_answer_too_short(answer)


class _SessionStub:
    def __init__(
        self,
        retrieval_language: str = "english",
        last_chosen_chunks: list[Any] | None = None,
    ) -> None:
        self.retrieval_language = retrieval_language
        self.last_chosen_chunks = last_chosen_chunks or []
        self.debug_level = 0
        self.debug_mode = "ge"
        self.grounding_proxy_text: str = ""
        self.grounding_proxy_sentences: dict[str, str] = {}


@pytest.mark.parametrize(
    ("answer", "chunks", "expected"),
    [
        ("### Answer\ndirect_hit", ["local evidence"], True),
        ("### Answer\nGerman paraphrase", ["English evidence"], False),
        ("### Answer\nNo match\n### Sources\ndirect_hit", ["local evidence"], False),
        ("### Answer\ndirect_hit", [], False),
        ("", ["local evidence"], False),
    ],
)
def test_source_marking_requires_local_answer_spans(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    chunks: list[str],
    expected: bool,
) -> None:
    _install_grounder_stub(monkeypatch)
    session = types.SimpleNamespace(
        last_answer_content=answer, chunk_texts_for_grounding=chunks
    )
    assert _Shell()._has_local_grounding_spans(session) is expected


class _ConfidenceSessionStub:
    def __init__(
        self,
        level: str | None = None,
        score: float | None = None,
        summary: str | None = None,
    ) -> None:
        self.level = level or ""
        self.score = score
        self.summary = summary or ""


def _install_grounder_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _find(_self: Any, answer: str, chunks: list[str]) -> list[str]:
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

    monkeypatch.setattr(Grounder, "find_grounded_sentences", _find)


def _install_grounder_no_match_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Grounder, "find_grounded_sentences", lambda _self, _answer, _chunks: []
    )


def _install_shared_stub(
    monkeypatch: pytest.MonkeyPatch,
    translated: str,
) -> list[tuple[str, str, str]]:
    c_pkg = types.ModuleType("Compliance")
    c_mod = types.ModuleType("Compliance.SharedHelpers")
    calls: list[tuple[str, str, str]] = []

    class _SharedHelpers:
        def translate_text(
            self,
            _text: str,
            target_lang: str,
            source_lang: str = "auto",
        ) -> str:
            calls.append((_text, target_lang, source_lang))
            return translated

    c_mod.SharedHelpers = _SharedHelpers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Compliance", c_pkg)
    monkeypatch.setitem(sys.modules, "Compliance.SharedHelpers", c_mod)
    return calls


@pytest.mark.parametrize(
    "body, short, unverified",
    [
        (
            "| Animal | Mammal Status |\n| --- | --- |\n"
            "| Cats | yes |\n| Dogs | yes |\n| Hedgehogs | yes |\n"
            "| Elephants | yes |\n| Fish | N/A |",
            True,
            False,
        ),
        (
            "- **Cats** - yes\n- **Dogs** - yes\n- **Hedgehogs** - yes\n- **Fish** - N/A",
            False,
            False,
        ),
        (
            "Purple spacecraft manufacture impossible engines on distant planets every night without any fuel.",
            False,
            False,
        ),
        ("- Purple spacecraft manufacture impossible engines.", False, True),
        ("1. Purple spacecraft manufacture impossible engines.", False, True),
        (
            "Purple spacecraft manufacture impossible engines.\n\n"
            "- Hedgehogs are small nocturnal mammals.",
            False,
            True,
        ),
        (
            "```mermaid\nflowchart LR\n"
            "  A[Purple spacecraft] -->|manufacture| B[Impossible engines]\n```",
            False,
            True,
        ),
        (
            "Purple spacecraft manufacture impossible engines.\n\n"
            "```mermaid\nflowchart LR\n  A[Hedgehogs] -->|are| B[Small nocturnal mammals]\n```",
            False,
            True,
        ),
        ("```mermaid\nnot a valid diagram\n```", False, True),
        (
            "```text\n- Purple spacecraft manufacture impossible engines.\n```",
            False,
            False,
        ),
        (
            "| Slot Number | Type | Width | Length | Speed | Power (W) |\n"
            "|-------------|------|-------|--------|-------|-----------|\n"
            "| 1 | PCIe 4.0 x16 | Double | Full | 4x | 75 |\n"
            "| 2 | PCIe 4.0 x16 | Double | Full | 4x | 75 |\n"
            "| 3 | PCIe 4.0 x8 | Single | Full | 2x | - |\n"
            "| 4 | PCIe 4.0 x8 | Single | Full | 2x | - |",
            False,
            True,
        ),
        (
            "Slot | Unsupported specifications\n"
            "--- | ---\n"
            "First slot | Purple spacecraft manufacture impossible engines.",
            False,
            True,
        ),
        ("| Animal | Mammal |\n| --- | --- |\n| Cats | Yes |", True, False),
    ],
)
@pytest.mark.parametrize("service", [False, True])
def test_answer_guard_displays_unverified_structures_but_withholds_prose(
    body: str,
    short: bool,
    unverified: bool,
    service: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Chat.Chatter import Chatter
    from Chat.ConfidenceEvaluator import ConfidenceEvaluator

    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    answer = (
        body + "\n\nMetadata used:\n"
        "- FileName: Hedgehogs.pdf\n"
        "- FilePath: /workspaces/RAG-LCC/TestDocs/Hedgehogs.pdf\n"
        "- PageNumber: 1"
    )
    evidence = (
        "Cats are mammals."
        if body.startswith("| Animal | Mammal |")
        else "Hedgehogs are small nocturnal mammals."
    )
    chatter = MagicMock()
    chatter.rag.parse_chat_command.return_value = ("", "")
    chatter.rag.retrieve.return_value = ("Hedgehogs are small nocturnal mammals.", 1)
    chatter.prompt = "Context: {context}\nQuery: {input}"
    chatter._build_query_notice.return_value = ""
    chatter._resolve_token_params.return_value = (8192, 512)
    chatter.llmCaller.call_llm.return_value = {"content": answer}
    chatter.modelOutputAdapter.interpret.return_value = types.SimpleNamespace(
        content=answer
    )
    chatter._check_answer_compliance.return_value = False
    chatter.masker.mask.side_effect = lambda text: text
    chatter._build_answer_confidence_notice.return_value = ""
    chatter.helpers.build_selected_sources_md.return_value = ""
    chatter._grounding_language_detector.return_value = lambda *_args: "english"
    chatter.grounder = _Shell().grounder
    monkeypatch.setitem(
        chatter.grounder.cfg.cfg, "_MARKED_DOCS_COLORS", {"answer_ansi": "48;5;214"}
    )
    session = types.SimpleNamespace(
        query="Create a table of animals and indicate whether they are mammals. Use yes, no, N/A.",
        retriever_k=20,
        max_output_tokens=512,
        max_output_tokens_override=None,
        context_size_override=None,
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        preferred_response_language="english",
        keyword_action_split=None,
        extraOllamaOptions=None,
        ollamaTopLevelParams=None,
        mark_text=True,
        enable_grounding=True,
        use_chat_context=True,
        web_search=False,
        answer_confidence_result=MagicMock(),
        chunk_texts_for_grounding=[evidence],
        last_chosen_chunks=[],
        debug_level=0,
        debug_mode="ge",
    )
    selected = types.SimpleNamespace(
        page_content=evidence,
        metadata={"Source": "local", "rerank_score": 0.8},
    )
    session.last_chosen_chunks = [selected]
    session.final_chunks_to_llm = 1
    evaluator = ConfidenceEvaluator(confidence_logger=MagicMock(), pretty=MagicMock())
    evaluator.store_answer_confidence(session, [selected], emit=False)
    chatter.rag.confidenceEvaluator.store_answer_confidence.side_effect = (
        evaluator.store_answer_confidence
    )
    delivered: list[str] = []

    success, content = Chatter.run(
        chatter,
        session,
        apiChunkHandler=delivered.append if service else None,
        is_streaming=False,
    )

    assert success
    if service:
        assert content == "".join(delivered)
    else:
        assert session.browser_answer_content == answer
        mark_supported = not short
        chatter._grounded_answer_browser_document.assert_called_once_with(
            session,
            mark_supported=mark_supported,
        )
        document = Chatter._grounded_answer_browser_document(
            chatter,
            session,
            mark_supported=mark_supported,
        )
        assert document is not None
        page = document[1].decode("utf-8")
        if not session.answer_grounding_verified:
            expected_title = (
                "Not fully grounded"
                if mark_supported and session.chunk_texts_for_grounding
                else "Unverified answer preview"
            )
            assert f"<title>{expected_title}</title>" in page
            if short:
                assert '<mark class="grounded">' not in page
            assert '<mark class="grounded">Purple spacecraft' not in page
        assert Grounder.no_evidence_message() not in page
        if body.startswith("| Slot Number"):
            assert page.count("<table>") == 1
            assert page.count("<th>") == 6
            assert page.count("<tr>") == 5
    if short:
        assert content == chatter.grounder.short_answer_notice() + answer
        assert not session.answer_grounding_verified
        chatter.rag.confidenceEvaluator.store_answer_confidence.assert_not_called()
        chatter.rag._mark_sources.assert_not_called()
    elif unverified:
        assert content == Grounder.unverified_structure_notice() + answer
        assert not session.answer_grounding_verified
        assert session.last_chosen_chunks == [selected]
        assert session.chunk_texts_for_grounding == [selected.page_content]
        chatter.chatContext.add_chat_turn.assert_called_once_with(
            session, session.query, content
        )
        cli_answer = chatter.print_llm_answer.call_args.args[0]
        assert cli_answer.startswith(Grounder.unverified_structure_notice())
        assert re.sub(r"\033\[[0-9;]*m", "", cli_answer) == content
        for line in cli_answer.splitlines():
            if "Purple spacecraft" in line or "Impossible engines" in line:
                assert "\033[" not in line
        if "Hedgehogs" in body:
            assert "\033[" in cli_answer
        chatter.confidence_logger.flush_turn_csv.assert_called_once_with(
            session, outcome="grounding_unverified", answer_text=content
        )
    elif body.startswith("- "):
        assert content == answer
        assert session.answer_grounding_verified
        chatter.rag.confidenceEvaluator.store_answer_confidence.assert_not_called()
    else:
        assert content is not None
        assert content == Grounder.no_evidence_message()
        assert "Grounding failed" in content
        assert "withheld" in content
        assert session.last_chosen_chunks == []
        assert session.chunk_texts_for_grounding == []
    if not short and not session.answer_grounding_verified:
        chatter.rag.confidenceEvaluator.store_answer_confidence.assert_called_once()
        confidence = session.answer_confidence_result
        assert confidence.level == "LOW"
        assert confidence.score == 0.0
        assert confidence.components["evidence_chunks"] == 1
        assert confidence.components["top_score"] == 0.8
        assert confidence.components["coverage"] == 1.0
        assert "grounding failed" in confidence.summary
        assert "evidence_chunks=1" in confidence.summary


def test_flow_disabled_grounding_does_not_require_grounding_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Chat.Chatter import Chatter

    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    answer = "A response without grounding."
    chatter = MagicMock()
    chatter.rag.parse_chat_command.return_value = ("", "")
    chatter.rag.retrieve.return_value = ("retrieved context", 1)
    chatter.prompt = "Context: {context}\nQuery: {input}"
    chatter._build_query_notice.return_value = ""
    chatter._resolve_token_params.return_value = (8192, 512)
    chatter.llmCaller.call_llm.return_value = {"content": answer}
    chatter.modelOutputAdapter.interpret.return_value = types.SimpleNamespace(
        content=answer
    )
    chatter._check_answer_compliance.return_value = False
    chatter.masker.mask.side_effect = lambda text: text
    chatter._build_answer_confidence_notice.return_value = ""
    chatter.helpers.build_selected_sources_md.return_value = ""
    chatter.grounder = MagicMock()
    session = types.SimpleNamespace(
        query="question",
        retriever_k=20,
        max_output_tokens=512,
        max_output_tokens_override=None,
        context_size_override=None,
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        preferred_response_language="english",
        keyword_action_split=None,
        extraOllamaOptions=None,
        ollamaTopLevelParams=None,
        mark_text=True,
        enable_grounding=False,
        use_chat_context=False,
        web_search=False,
        answer_confidence_result=MagicMock(),
        chunk_texts_for_grounding=[],
        last_chosen_chunks=[],
        debug_level=0,
        debug_mode="ge",
    )

    success, _ = Chatter.run(chatter, session, is_streaming=False)

    assert success
    chatter.grounder.is_answer_too_short.assert_not_called()
    chatter.grounder.has_grounded_evidence.assert_not_called()
    chatter.rag._mark_sources.assert_not_called()


def test_web_leg_shows_answer_and_preserves_confidence_when_grounding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Chat.Chatter import Chatter
    from Chat.RAGChatImpl import RAGChatImpl

    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    answer = "### Answer\nThe web result mentions an unsupported claim."
    selected_result = types.SimpleNamespace(
        page_content="A local result unrelated to the generated claim.",
        metadata={"Source": "Local", "rerank_score": 0.8},
    )
    chatter = MagicMock()
    chatter.rag.parse_chat_command.side_effect = RAGChatImpl.parse_chat_command
    chatter.rag.retrieve.return_value = (selected_result.page_content, 1)
    chatter.prompt = "Context: {context}\nQuery: {input}"
    chatter._build_query_notice.side_effect = lambda current, current_query: (
        Chatter._build_query_notice(chatter, current, current_query)
    )
    chatter._resolve_token_params.return_value = (8192, 512)
    chatter.llmCaller.call_llm.return_value = {"content": answer}
    chatter.modelOutputAdapter.interpret.return_value = types.SimpleNamespace(
        content=answer
    )
    chatter._check_answer_compliance.return_value = False
    chatter.masker.mask.side_effect = lambda text: text
    chatter._build_answer_confidence_notice.return_value = ""
    chatter.helpers.build_selected_sources_md.return_value = ""
    chatter._grounding_language_detector.return_value = lambda *_args: "english"
    chatter.grounder = _Shell().grounder
    monkeypatch.setattr(chatter.grounder, "has_grounded_evidence", lambda *_args: False)

    evaluated_confidence = types.SimpleNamespace(
        level="MEDIUM", score=0.65, components={}, summary="medium retrieval confidence"
    )
    session = types.SimpleNamespace(
        query="What does the web say?",
        user_query_input="new: What does the web say?",
        retriever_k=20,
        max_output_tokens=512,
        max_output_tokens_override=None,
        context_size_override=None,
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        preferred_response_language="english",
        keyword_action_split=None,
        extraOllamaOptions=None,
        ollamaTopLevelParams=None,
        mark_text=True,
        enable_grounding=True,
        use_chat_context=True,
        web_search=True,
        answer_confidence_result=evaluated_confidence,
        chunk_texts_for_grounding=[selected_result.page_content],
        last_chosen_chunks=[selected_result],
        final_chunks_to_llm=1,
        debug_level=0,
        debug_mode="ge",
    )
    delivered: list[str] = []
    query_notice = Chatter._build_query_notice(chatter, session, session.query)

    success, content = Chatter.run(
        chatter, session, apiChunkHandler=delivered.append, is_streaming=False
    )

    assert success
    assert content is not None
    assert 'Query: "What does the web say?"' in content
    assert "new:" not in content
    assert content == query_notice + Grounder.unverified_structure_notice() + answer
    assert "".join(delivered) == content
    assert "Grounding failed" not in content
    assert session.answer_confidence_result is evaluated_confidence
    assert session.answer_confidence_result.level == "MEDIUM"
    assert session.answer_confidence_result.score == 0.65
    assert session.last_chosen_chunks == [selected_result]
    chatter.rag.confidenceEvaluator.store_answer_confidence.assert_not_called()


@pytest.mark.parametrize("service", [False, True])
@pytest.mark.parametrize("split_enabled", [False, True])
def test_new_topic_resets_history_without_leaking_command_into_answer_prompt(
    service: bool,
    split_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Chat.Chatter import Chatter
    from Chat.KeywordActionSplitter import KeywordActionSplit
    from Chat.RAGChatImpl import RAGChatImpl
    from Configuration.Config_RAGChat_Prompts import _PROMPT_KEYWORD_ACTION_ANSWER
    from Gui.Colors import CYAN

    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    raw = "new: List the animals."
    clean = "List the animals."
    effective = f"Use the evidence.\n\n{clean}" if service else clean
    chatter = MagicMock()
    chatter.rag.parse_chat_command.side_effect = RAGChatImpl.parse_chat_command
    chatter.prompt = "{context}\nQuery: {input}"
    chatter._build_query_notice.return_value = ""
    chatter._resolve_token_params.return_value = (8192, 512)
    chatter.llmCaller.call_llm.return_value = {}
    chatter.modelOutputAdapter.interpret.return_value = types.SimpleNamespace(
        content="Cats."
    )
    chatter._check_answer_compliance.return_value = False
    chatter.masker.mask.side_effect = lambda text: text
    chatter._build_answer_confidence_notice.return_value = ""
    chatter.grounder.is_answer_too_short.return_value = True
    chatter.grounder.short_answer_notice.return_value = "Grounding skipped.\n\n"
    chatter.cfg.get_str.return_value = _PROMPT_KEYWORD_ACTION_ANSWER
    session = types.SimpleNamespace(
        query=effective if service else raw,
        user_query_input=raw if service else None,
        is_service=service,
        reset_turn_state=MagicMock(),
        last_web_search=None,
        last_fetch_page_content=None,
        last_topic_referents=["previous animal"],
        fetch_page_content=False,
        retriever_k=20,
        max_output_tokens=512,
        max_output_tokens_override=None,
        context_size_override=None,
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        preferred_response_language="english",
        extraOllamaOptions=None,
        ollamaTopLevelParams=None,
        mark_text=True,
        enable_grounding=True,
        use_chat_context=True,
        web_search=False,
        answer_confidence_result=MagicMock(),
        chunk_texts_for_grounding=["Cats are mammals."],
        last_chosen_chunks=[],
        debug_level=0,
        debug_mode="ge",
    )

    def retrieve(current: Any) -> tuple[str, int]:
        assert RAGChatImpl._prepare_session(chatter.rag, current) == effective
        current.keyword_action_split = (
            KeywordActionSplit([clean], None, clean, ["animals"], ["animals"])
            if split_enabled
            else None
        )
        return "Cats are mammals.", 1

    chatter.rag.retrieve.side_effect = retrieve

    success, _ = Chatter.run(
        chatter,
        session,
        apiChunkHandler=(lambda _text: None) if service else None,
        is_streaming=False,
    )

    assert success
    prompt = chatter.llmCaller.call_llm.call_args.args[1]
    assert clean in prompt
    assert "new:" not in prompt.lower()
    if service:
        assert "Use the evidence." in prompt
    assert session.force_skip_rewrite is True
    assert session.last_topic_referents is None
    chatter.rag.chatContext.reset_conversation.assert_called_once()
    assert chatter.chatContext.add_chat_turn.call_args.args[1] == effective
    chatter.rag.pretty.write.assert_any_call(
        "I",
        "UserQuery",
        f"Original user query: {raw!r}",
        color=CYAN,
    )
    assert chatter.print_llm_answer.call_args.kwargs["original_query"] == (
        None if service else raw
    )


class TestGroundingGuardHelpers:
    @pytest.mark.parametrize(
        "answer",
        [
            "The slot | description is ordinary prose.",
            "```text\n| Slot | Type |\n| --- | --- |\n| 1 | PCIe x16 |\n```",
            "Unsupported prose.\n\n### Sources\n| File | Page |\n| --- | --- |\n| Guide | 6 |",
        ],
    )
    def test_table_fallback_ignores_prose_code_and_source_metadata(
        self,
        answer: str,
    ) -> None:
        assert not _Shell().grounder.has_structured_answer(answer)

    def test_extract_answer_section_removes_sources_block(self) -> None:
        text = "### Answer\nCats sleep.\n\n### Sources\n- file.md"
        body = Grounder.extract_answer_section(text)
        assert body == "### Answer\nCats sleep."

    @pytest.mark.parametrize(
        "heading", ["Metadata used:", "### Sources", "## Sources:"]
    )
    def test_metadata_footer_does_not_make_terse_table_groundable(
        self, heading: str
    ) -> None:
        body = (
            "| Animal | Mammal Status |\n"
            "| --- | --- |\n"
            "| Cats | yes |\n"
            "| Dogs | yes |\n"
            "| Hedgehogs | yes |\n"
            "| Elephants | yes |\n"
            "| Fish | N/A |"
        )
        answer = (
            f"{body}\n\n{heading}\n"
            "- FileName: Hedgehogs.pdf\n"
            "- FilePath: /workspaces/RAG-LCC/TestDocs/Hedgehogs.pdf\n"
            "- PageNumber: 1"
        )
        grounder = _Shell().grounder

        assert Grounder.extract_answer_section(answer) == body
        assert grounder.is_answer_too_short(answer)

    def test_metadata_overlap_does_not_ground_unsupported_answer(self) -> None:
        answer = (
            "Purple spacecraft manufacture impossible engines on distant planets every night without any fuel.\n\n"
            "Metadata used:\n"
            "- FilePath: /workspaces/RAG-LCC/TestDocs/Hedgehogs.pdf"
        )
        shell = _Shell()
        assert not shell.grounder.is_answer_too_short(answer)
        assert not shell._has_grounded_evidence(
            _SessionStub(),
            answer,
            [
                "- FilePath: /workspaces/RAG-LCC/TestDocs/Hedgehogs.pdf\n"
                "Hedgehogs are small nocturnal mammals."
            ],
            "english",
        )

    def test_build_no_evidence_message_contains_context_phrase(self) -> None:
        msg = Grounder.no_evidence_message()
        assert "grounding failed" in msg.lower()
        assert "couldn't verify" in msg.lower()
        assert "withheld" in msg.lower()
        assert "retrieved context" in msg.lower()
        assert "`new:`" in msg
        assert "different collection" in msg
        assert "session parameters" in msg
        assert "retriever_k" not in msg
        assert "top_k" not in msg

    def test_build_grounding_skipped_short_answer_notice_mentions_too_short(
        self,
    ) -> None:
        msg = Grounder.short_answer_notice()
        assert msg.startswith("⚠")
        assert "grounding skipped" in msg.lower()
        assert "too short" in msg.lower()

    def test_build_answer_confidence_notice_from_summary(self) -> None:
        session = _ConfidenceSessionStub(
            summary="HIGH (score=0.91; evidence_chunks=7; local_share=1.00)",
        )
        notice = _ConfidenceShell._build_answer_confidence_notice(session)
        assert notice.startswith("\n\n---\n\n**Answer confidence:**")
        assert "HIGH (score=0.91" in notice

    def test_build_answer_confidence_notice_from_level_and_score(self) -> None:
        session = _ConfidenceSessionStub(level="medium", score=0.638)
        notice = _ConfidenceShell._build_answer_confidence_notice(session)
        assert notice.startswith("\n\n---\n\n**Answer confidence:**")
        assert "MEDIUM" in notice
        assert "C_final=0.64" in notice

    def test_build_answer_confidence_notice_empty_when_unset(self) -> None:
        session = _ConfidenceSessionStub()
        notice = _ConfidenceShell._build_answer_confidence_notice(session)
        assert notice == ""

    def test_short_prose_still_requires_grounding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\nCats, dogs, foxes"
        )

        assert is_short is False

    def test_is_answer_too_short_for_grounding_false_for_long_sentence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\n"
            "Hedgehogs are small nocturnal mammals with protective spines and adaptable foraging habits."
        )

        assert is_short is False

    def test_brief_bullet_list_still_requires_grounding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shell = _Shell(lang="english")

        is_short = shell._is_answer_too_short_for_grounding(
            "### Answer\n- cats\n- dogs\n- apes (gorillas and orangutans mentioned)"
        )

        assert is_short is False

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

    def test_has_grounded_evidence_translates_english_answer_to_local_evidence_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_grounder_stub(monkeypatch)
        _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="english")
        local_doc = types.SimpleNamespace(
            metadata={"Source": "Local", "Language": "de"},
            page_content="deutscher chunk",
        )
        session = _SessionStub(
            retrieval_language="english",
            last_chosen_chunks=[local_doc],
        )

        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nNo lexical overlap here.\n\n### Sources\n- Pferde.pdf",
            ["deutscher chunk"],
            "english",
        )

        assert ok is True

    def test_has_grounded_evidence_false_when_no_chunk_texts(self) -> None:
        shell = _Shell(lang="english")
        session = _SessionStub(retrieval_language="english")
        ok = shell._has_grounded_evidence(
            session,
            "### Answer\nAny answer\n\n### Sources\n- A.txt",
            [],
            "english",
        )
        assert ok is False

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
            metadata={"Source": "Web", "Language": "en", "snippet": "web_hit snippet"},
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

    @pytest.mark.parametrize("evidence_language", ["en", "english"])
    def test_native_german_query_grounds_german_answer_against_english_evidence(
        self, monkeypatch: pytest.MonkeyPatch, evidence_language: str
    ) -> None:
        _install_grounder_no_match_stub(monkeypatch)
        calls = _install_shared_stub(
            monkeypatch,
            translated="Hedgehogs inhabit suburban gardens and farm edges.",
        )
        shell = _Shell(lang="german")
        evidence = (
            "Hedgehogs are often found in suburban gardens and agricultural edges."
        )
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[
                types.SimpleNamespace(metadata={"Language": evidence_language})
            ],
        )
        answer = "### Answer\nIgel leben in Vorstadtgaerten und an Feldraendern.\n\n### Sources\n- Igel.pdf"

        assert shell._has_grounded_evidence(session, answer, [evidence], "german")
        assert calls == [
            ("### Answer", "en", "de"),
            ("Igel leben in Vorstadtgaerten und an Feldraendern.", "en", "de"),
        ]
        assert session.last_chosen_chunks

    def test_german_answer_uses_french_evidence_not_english_retrieval_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls = _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="english",
            last_chosen_chunks=[
                types.SimpleNamespace(metadata={"Language": "fr"}),
                types.SimpleNamespace(metadata={"Language": "french"}),
            ],
        )

        assert shell._has_grounded_evidence(
            session, "Antwort ohne overlap", ["texte francais"], "german"
        )
        assert calls == [("Antwort ohne overlap", "fr", "de")]

    def test_matching_language_does_not_translate_failed_grounding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls = _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="english",
            last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "de"})],
        )

        assert not shell._has_grounded_evidence(
            session, "Antwort ohne overlap", ["unrelated"], "german"
        )
        assert calls == []

    def test_failed_proxy_does_not_accept_unsupported_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls = _install_shared_stub(monkeypatch, translated="")
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "en"})],
        )

        assert not shell._has_grounded_evidence(
            session, "Antwort ohne overlap", ["unrelated"], "german"
        )
        assert calls == [("Antwort ohne overlap", "en", "de")]
        assert any("empty or unchanged" in message for message in shell.messages)

    def test_translation_failure_logs_and_continues_to_next_evidence_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls: list[tuple[str, str]] = []
        _install_shared_stub(monkeypatch, translated="")
        shared_module = sys.modules["Compliance.SharedHelpers"]

        class _Translator:
            def translate_text(
                self, _text: str, target_lang: str, source_lang: str
            ) -> str:
                calls.append((source_lang, target_lang))
                if target_lang == "fr":
                    raise RuntimeError("translation unavailable")
                return "translated_hit"

        monkeypatch.setattr(shared_module, "SharedHelpers", _Translator)
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[
                types.SimpleNamespace(metadata={"Language": "fr"}),
                types.SimpleNamespace(metadata={"Language": "en", "Source": "Web"}),
            ],
        )

        assert shell._has_grounded_evidence(
            session, "Antwort ohne overlap", ["retrieved evidence"], "german"
        )
        assert calls == [("de", "fr"), ("de", "en")]
        assert any("de->fr failed" in message for message in shell.messages)

    def test_unknown_web_language_is_detected_for_native_query_grounding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls = _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="german")

        def detect_language(_text: str, _native: str, stage: str) -> str:
            shell.detected.append(stage)
            return "english" if stage == "web_grounding_language" else "german"

        shell.detect = detect_language  # type: ignore[method-assign]
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[
                types.SimpleNamespace(
                    metadata={"Source": "Web", "snippet": "English evidence"},
                    page_content="English page",
                )
            ],
        )
        assert shell._has_grounded_evidence(
            session, "Antwort ohne overlap", [], "german"
        )
        assert calls == [("Antwort ohne overlap", "en", "de")]
        assert "web_grounding_language" in shell.detected

    def test_foreign_answer_translates_first_and_stores_sentence_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        calls = _install_shared_stub(monkeypatch, translated="translated_hit")
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "en"})],
        )

        # "direct_hit" would pass the direct check; translate-first skips it.
        assert shell._has_grounded_evidence(
            session, "Erster Satz direct_hit. Zweiter Satz.", ["evidence"], "german"
        )
        assert calls == [
            ("Erster Satz direct_hit.", "en", "de"),
            ("Zweiter Satz.", "en", "de"),
        ]
        assert session.grounding_proxy_text == "translated_hit translated_hit"
        assert session.grounding_proxy_sentences == {
            "Erster Satz direct_hit.": "translated_hit",
            "Zweiter Satz.": "translated_hit",
        }
        assert any("grounded via translation de->en" in m for m in shell.messages)

    def test_foreign_answer_falls_back_to_direct_check_after_failed_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_grounder_stub(monkeypatch)
        _install_shared_stub(monkeypatch, translated="")
        shell = _Shell(lang="german")
        session = _SessionStub(
            retrieval_language="german",
            last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "en"})],
        )

        assert shell._has_grounded_evidence(
            session, "Name direct_hit bleibt gleich.", ["evidence"], "german"
        )
        assert not getattr(session, "grounding_proxy_text", "")


def test_source_marking_uses_translated_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_grounder_stub(monkeypatch)
    session = types.SimpleNamespace(
        last_answer_content="### Answer\nGerman paraphrase",
        grounding_proxy_text="translated_hit",
        chunk_texts_for_grounding=["English evidence"],
    )
    assert _Shell()._has_local_grounding_spans(session) is True


@pytest.mark.parametrize(
    ("level", "mode", "visible"),
    [
        (0, "is", False),
        (30, "ge", False),
        (33, "is", True),
        (33, "ge", True),
        (34, "is", False),
        (34, "ge", True),
    ],
)
@pytest.mark.parametrize("supported", [True, False])
def test_translated_grounding_text_respects_debug_mode(
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    mode: str,
    visible: bool,
    supported: bool,
) -> None:
    _install_grounder_stub(monkeypatch)
    translated = "translated_hit" if supported else "unsupported_translation"
    calls = _install_shared_stub(monkeypatch, translated=translated)
    shell = _Shell(lang="german")
    session = _SessionStub(
        last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "en"})]
    )
    session.debug_level = level
    session.debug_mode = mode

    assert (
        shell._has_grounded_evidence(
            session, "Antwort ohne overlap", ["evidence"], "german"
        )
        is supported
    )
    dumps = [
        m for m in shell.messages if m.startswith("Translated answer for grounding")
    ]
    assert dumps == (
        [
            f"Translated answer for grounding (de->en, document language en):\n{translated}"
        ]
        if visible
        else []
    )
    assert calls == [("Antwort ohne overlap", "en", "de")]


@pytest.mark.parametrize(
    ("level", "mode", "visible"),
    [
        (0, "is", False),
        (30, "ge", False),
        (33, "is", True),
        (33, "ge", True),
        (34, "is", False),
        (34, "ge", True),
    ],
)
@pytest.mark.parametrize("supported", [True, False])
def test_matching_document_language_reports_no_translation(
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    mode: str,
    visible: bool,
    supported: bool,
) -> None:
    _install_grounder_stub(monkeypatch)
    calls = _install_shared_stub(monkeypatch, translated="translated_hit")
    shell = _Shell(lang="german")
    session = _SessionStub(
        last_chosen_chunks=[types.SimpleNamespace(metadata={"Language": "german"})]
    )
    session.debug_level = level
    session.debug_mode = mode

    assert (
        shell._has_grounded_evidence(
            session,
            "direct_hit" if supported else "unsupported_answer",
            ["evidence"],
            "german",
        )
        is supported
    )
    notices = [
        m for m in shell.messages if m.startswith("Answer already in document language")
    ]
    assert notices == (
        [
            "Answer already in document language (de); no translation required "
            "for grounding against this language."
        ]
        if visible
        else []
    )
    assert not any(
        m.startswith("Translated answer for grounding") for m in shell.messages
    )
    assert calls == []


def test_unknown_document_language_does_not_claim_language_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_grounder_stub(monkeypatch)
    shell = _Shell(lang="german")
    session = _SessionStub(retrieval_language="")
    session.debug_level = 33
    session.debug_mode = "is"

    assert shell._has_grounded_evidence(session, "direct_hit", ["evidence"], "german")
    assert [m for m in shell.messages if not m.startswith("Timing ")] == [
        "Document language unavailable; checking the original answer "
        "without a grounding translation."
    ]
    assert any(m.startswith("Timing evidence validation:") for m in shell.messages)
