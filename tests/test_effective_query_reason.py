# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for the effective_query_reason feature.

Covers:
  * Session.effective_query_reason defaults to None
  * Chatter notice label mapping:
      translated / rewritten / translated+rewritten / unknown / None / ""
  * Chatter clarification path — no notice when query unchanged
  * Chatter clarification path — notice prepended for each reason value
  * Chatter clarification path — notice chunk arrives before clarification chunk
  * Chatter clarification path — no notice when effective_query equals original
"""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Globals.Session import Session

# ===========================================================================
# Session defaults
# ===========================================================================


class TestSessionDefaults:
    def test_effective_query_reason_defaults_to_none(self):
        s = Session()
        assert s.effective_query_reason is None

    def test_effective_query_defaults_to_none(self):
        s = Session()
        assert s.effective_query is None

    def test_both_reset_independently(self):
        """Verify the two fields are distinct attributes, not aliases."""
        s = Session()
        s.effective_query = "translated query"
        assert s.effective_query_reason is None
        s.effective_query_reason = "translated"
        assert s.effective_query == "translated query"


# ===========================================================================
# Notice label mapping — mirrors the dict in Chatter.run()
# ===========================================================================

_NOTICE_LABELS: dict[str, str] = {
    "translated": "Translated query",
    "rewritten": "Rewritten query",
    "translated+rewritten": "Translated & rewritten query",
}


def _resolve_label(reason: str | None) -> str:
    """Replicate the exact label-resolution logic from Chatter.run()."""
    r = (reason or "").strip() or "changed"
    return _NOTICE_LABELS.get(r, "Query (changed)")


class TestNoticeLabelMapping:
    @pytest.mark.parametrize(
        "reason,expected",
        [
            ("translated", "Translated query"),
            ("rewritten", "Rewritten query"),
            ("translated+rewritten", "Translated & rewritten query"),
            ("changed", "Query (changed)"),
            ("other_unknown", "Query (changed)"),
            (None, "Query (changed)"),
            ("", "Query (changed)"),
        ],
    )
    def test_label_for_reason(self, reason: str | None, expected: str):
        assert _resolve_label(reason) == expected

    def test_notice_string_contains_label_and_query(self):
        """Verify the notice string format matches what Chatter produces."""
        label = _resolve_label("translated")
        effective_q = "do hedgehogs have spines?"
        notice = f'\U0001f50d *{label}: "{effective_q}"*\n\n---\n\n'
        assert notice.startswith("\U0001f50d")
        assert "Translated query" in notice
        assert effective_q in notice
        assert notice.endswith("---\n\n")

    def test_notice_string_for_rewritten(self):
        label = _resolve_label("rewritten")
        effective_q = "does the hedgehog have spines?"
        notice = f'\U0001f50d *{label}: "{effective_q}"*\n\n---\n\n'
        assert "Rewritten query" in notice

    def test_notice_string_for_translated_plus_rewritten(self):
        label = _resolve_label("translated+rewritten")
        effective_q = "do hedgehogs have spines?"
        notice = f'\U0001f50d *{label}: "{effective_q}"*\n\n---\n\n'
        assert "Translated & rewritten query" in notice


# ===========================================================================
# Stubs for Chatter.run() tests
# ===========================================================================


class _StubConfig:
    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get(self, key: str, default: Any = None) -> Any:
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default


class _StubPretty:
    def write(self, *a: Any, **kw: Any) -> None:
        pass


class _StubSession:
    """Minimal Session-like object satisfying Chatter.run() field checks."""

    def __init__(self, query: str = "original query") -> None:
        self.query = query
        self.retriever_k = 10
        self.max_output_tokens = 512
        self.temperature = 0.1
        self.top_k = 40
        self.top_p = 0.9
        self.effective_query: str | None = None
        self.effective_query_reason: str | None = None
        self.clarification_response: str | None = None
        self.web_search = False
        self.debug_level = 0
        self.debug_mode = "ge"
        self.ollamaTopLevelParams: dict | None = None


class _RagStub:
    """Simulates rag.retrieve(): sets session state and returns (context, 0)."""

    def __init__(
        self,
        effective_q: str | None,
        reason: str | None,
        clarification: str,
    ) -> None:
        self._effective_q = effective_q
        self._reason = reason
        self._clarification = clarification

    def retrieve(self, session: _StubSession) -> tuple[str, int]:
        session.effective_query = self._effective_q
        session.effective_query_reason = self._reason
        session.clarification_response = self._clarification
        return ("", 0)


def _make_chatter(rag_stub: _RagStub) -> Any:
    """Build an uninitialised Chatter instance with only the fields run() needs."""
    from Chat.Chatter import Chatter

    chatter = object.__new__(Chatter)
    chatter.is_streaming = False
    chatter.llm_model = "stub-model"
    chatter.prompt = "{input}"
    chatter.prompt_name = None
    chatter.rag = rag_stub
    chatter.cfg = _StubConfig()
    chatter.pretty = _StubPretty()
    # terminal_line_size is a @property — it reads from self.cfg (returns 160 when None)
    return chatter


# ===========================================================================
# Chatter clarification-path tests
# ===========================================================================


class TestChatterClarificationNotice:
    def _call(
        self, session: _StubSession, rag_stub: _RagStub
    ) -> tuple[bool, str | None, list[str]]:
        chunks: list[str] = []
        chatter = _make_chatter(rag_stub)
        success, msg = chatter.run(session, apiChunkHandler=chunks.append)
        return success, msg, chunks

    # -------------------------------------------------------------------
    # No notice when query is unchanged
    # -------------------------------------------------------------------

    def test_no_notice_when_effective_query_is_none(self):
        """effective_query=None → clarification only, no notice prefix."""
        s = _StubSession("what are mammals?")
        rag = _RagStub(None, None, "Could you clarify?")
        success, msg, chunks = self._call(s, rag)
        assert success is True
        assert msg == "❔  Could you clarify?"
        assert "\U0001f50d" not in (msg or "")
        assert len(chunks) == 1
        assert chunks[0] == "❔  Could you clarify?"

    def test_no_notice_when_effective_query_equals_original(self):
        """effective_query same as session.query → condition false → no notice."""
        s = _StubSession("what are mammals?")
        rag = _RagStub("what are mammals?", "translated", "Could you clarify?")
        success, msg, chunks = self._call(s, rag)
        assert "\U0001f50d" not in (msg or "")
        assert msg == "❔  Could you clarify?"

    # -------------------------------------------------------------------
    # Notice is shown for each reason value
    # -------------------------------------------------------------------

    def test_notice_translated(self):
        """Translated query: notice says 'Translated query'."""
        s = _StubSession("haben igel stacheln?")
        rag = _RagStub("Is it igel?", "translated", "I'm not sure what 'it' refers to.")
        success, msg, chunks = self._call(s, rag)
        assert success is True
        assert msg is not None
        assert msg.startswith("\U0001f50d *Translated query:")
        assert "Is it igel?" in msg
        assert "I'm not sure what 'it' refers to." in msg

    def test_notice_rewritten(self):
        """Rewritten query: notice says 'Rewritten query'."""
        s = _StubSession("does it eat fish?")
        rag = _RagStub("does the dolphin eat fish?", "rewritten", "Could you clarify?")
        success, msg, chunks = self._call(s, rag)
        assert "\U0001f50d *Rewritten query:" in (msg or "")
        assert "does the dolphin eat fish?" in (msg or "")

    def test_notice_translated_plus_rewritten(self):
        """Translated+rewritten: notice says 'Translated & rewritten query'."""
        s = _StubSession("hat es stacheln?")
        rag = _RagStub(
            "do hedgehogs have spines?", "translated+rewritten", "Clarify please."
        )
        success, msg, chunks = self._call(s, rag)
        assert "\U0001f50d *Translated & rewritten query:" in (msg or "")
        assert "do hedgehogs have spines?" in (msg or "")

    def test_notice_unknown_reason_uses_fallback_label(self):
        """Unrecognised reason → fallback label 'Query (changed)'."""
        s = _StubSession("original?")
        rag = _RagStub("changed query", "some_other_reason", "Clarify?")
        success, msg, chunks = self._call(s, rag)
        assert "\U0001f50d *Query (changed):" in (msg or "")

    # -------------------------------------------------------------------
    # Streaming order: notice chunk arrives before clarification chunk
    # -------------------------------------------------------------------

    def test_chunks_order_notice_then_clarification(self):
        """apiChunkHandler receives the notice chunk first, clarification second."""
        s = _StubSession("haben igel stacheln?")
        rag = _RagStub("Is it igel?", "translated", "Please clarify.")
        success, msg, chunks = self._call(s, rag)
        assert len(chunks) == 2
        assert "\U0001f50d" in chunks[0]
        assert "Translated query" in chunks[0]
        assert "Please clarify" in chunks[1]

    def test_chunks_single_when_no_notice(self):
        """No notice → only one chunk (the clarification) is sent."""
        s = _StubSession("what are mammals?")
        rag = _RagStub(None, None, "Please clarify.")
        success, msg, chunks = self._call(s, rag)
        assert len(chunks) == 1

    # -------------------------------------------------------------------
    # Return value: notice is part of msg regardless of API vs CLI
    # -------------------------------------------------------------------

    def test_notice_in_return_value_for_cli(self):
        """run() return value includes notice even with no apiChunkHandler."""
        s = _StubSession("haben igel stacheln?")
        rag = _RagStub("Is it igel?", "translated", "Please clarify.")
        chatter = _make_chatter(rag)
        success, msg = chatter.run(session=s, apiChunkHandler=None)
        assert success is True
        assert msg is not None
        assert "\U0001f50d *Translated query:" in msg
        assert "Is it igel?" in msg
        assert "Please clarify" in msg
