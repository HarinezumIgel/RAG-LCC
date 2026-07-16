# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""
Tests for language-scoped chat context (ChatContext.add_chat_turn / _fetch_context_docs).

Verifies that:
  - add_chat_turn stores 'query_lang' in turn metadata
  - _fetch_context_docs filters by 'query_lang' in the WHERE clause
  - German turns are invisible to English sessions and vice versa
  - Sessions with current_query_lang=None default to 'english'
  - Switching back to a language restores its original context
"""

import os
import re
import sys
import textwrap
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document as LangchainDocument
from Helpers.DebugHelper import DebugHelper as _DebugHelper

# ---------------------------------------------------------------------------
# In-memory ChromaDB collection stub
# ---------------------------------------------------------------------------


def _where_matches(meta: dict, where: dict) -> bool:
    """Recursive evaluation of ChromaDB-style $and / field-equality filters."""
    if "$and" in where:
        return all(_where_matches(meta, cond) for cond in where["$and"])
    for key, value in where.items():
        if meta.get(key) != value:
            return False
    return True


class FakeCollection:
    def __init__(self):
        self.stored: list[dict] = []
        self.last_where: Any = None  # captured from the most recent get()

    def add(self, ids, embeddings, metadatas, documents):
        for doc, meta in zip(documents, metadatas):
            self.stored.append({"document": doc, "metadata": dict(meta)})

    def get(self, where, include, limit=None):
        self.last_where = where
        matched = [
            item for item in self.stored if _where_matches(item["metadata"], where)
        ]
        if limit:
            matched = matched[:limit]
        return {
            "documents": [r["document"] for r in matched],
            "metadatas": [r["metadata"] for r in matched],
            "ids": [str(i) for i in range(len(matched))],
        }


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self):
        self.calls: list = []

    def write(self, *a, **k):
        self.calls.append((a, k))


class StubSession:
    def __init__(
        self,
        chat_name: str = "MyChat",
        collection_name: str = "TestCol",
        current_query_lang: str | None = None,
        file_name: str | None = None,
        file_path: str | None = None,
        turns: int = 20,
        prune_batch: int = 2,
        debug_level: int = 0,
    ):
        self.chat_name = chat_name
        self.collection_name = collection_name
        self.current_query_lang = current_query_lang
        self.file_name = file_name
        self.file_path = file_path
        self.turns = turns
        self.prune_batch = prune_batch
        self.debug_level = debug_level


# ---------------------------------------------------------------------------
# Load add_chat_turn and _fetch_context_docs from source without heavy imports
# ---------------------------------------------------------------------------

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "Chat", "ChatContext.py")


def _extract_method(source: str, method_name: str) -> str:
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in ChatContext.py"
    return textwrap.dedent(match.group(1))


with open(_SRC, encoding="utf-8") as _f:
    _CHAT_SOURCE = _f.read()

_add_chat_turn_src = _extract_method(_CHAT_SOURCE, "add_chat_turn")
_fetch_context_docs_src = _extract_method(_CHAT_SOURCE, "_fetch_context_docs")


def _compile_methods():
    ns: dict[str, Any] = {
        "datetime": datetime,
        "timezone": timezone,
        "uuid": _uuid,
        "Any": Any,
        "List": List,
        "LangchainDocument": LangchainDocument,
        "Session": StubSession,  # satisfies type annotation at def time
        "getattr": getattr,
        "DebugHelper": _DebugHelper,
    }
    exec(compile(_add_chat_turn_src, _SRC, "exec"), ns)
    exec(compile(_fetch_context_docs_src, _SRC, "exec"), ns)
    return ns["add_chat_turn"], ns["_fetch_context_docs"]


_add_chat_turn, _fetch_context_docs = _compile_methods()


# ---------------------------------------------------------------------------
# ContextShell — lightweight ChatContext stand-in
# ---------------------------------------------------------------------------


class ContextShell:
    """Binds the real add_chat_turn / _fetch_context_docs to a stub body."""

    add_chat_turn = _add_chat_turn
    _fetch_context_docs = _fetch_context_docs

    def __init__(self, conversation_id: str = "conv-1"):
        self.conversation_id: str = conversation_id
        self.collection_name: str = "TestCol_ChatContext"
        self.turn_index: int = 0
        self.pretty = StubPrettyWriter()
        self._fake: FakeCollection = FakeCollection()
        self._upserted: list[dict] = []  # captured upsert calls

    def _init_chat_collection(self, session):
        return self._fake

    def _prune_chat_context(self, session, summary_model=None):
        pass  # not under test

    def _start_conversation(self):
        pass  # conversation_id already set by __init__

    def _upsert_to_collection(self, session, docs, ids, metadatas):
        self._upserted.append({"docs": list(docs), "metadatas": list(metadatas)})
        self._fake.add(
            ids=ids,
            embeddings=[[0.0] * 3] * len(docs),
            metadatas=metadatas,
            documents=docs,
        )


# ---------------------------------------------------------------------------
# Tests — add_chat_turn stores query_lang
# ---------------------------------------------------------------------------


class TestAddChatTurnQueryLang:

    def test_stores_english_when_lang_is_none(self):
        ctx = ContextShell()
        session = StubSession(current_query_lang=None)
        ctx.add_chat_turn(session, "hello", "hi")
        assert ctx._upserted[0]["metadatas"][0]["query_lang"] == "english"

    def test_stores_german_when_lang_is_german(self):
        ctx = ContextShell()
        session = StubSession(current_query_lang="german")
        ctx.add_chat_turn(session, "haben igel flügel?", "Nein.")
        assert ctx._upserted[0]["metadatas"][0]["query_lang"] == "german"

    def test_stores_english_when_lang_is_empty_string(self):
        ctx = ContextShell()
        session = StubSession(current_query_lang="")
        ctx.add_chat_turn(session, "q", "a")
        assert ctx._upserted[0]["metadatas"][0]["query_lang"] == "english"

    def test_stores_french(self):
        ctx = ContextShell()
        session = StubSession(current_query_lang="french")
        ctx.add_chat_turn(session, "Bonjour", "Salut")
        assert ctx._upserted[0]["metadatas"][0]["query_lang"] == "french"

    def test_stores_chat_name_and_file_tag(self):
        ctx = ContextShell()
        session = StubSession(
            chat_name="MyChatName",
            file_name="report.pdf",
            current_query_lang="english",
        )
        ctx.add_chat_turn(session, "q", "a")
        meta = ctx._upserted[0]["metadatas"][0]
        assert meta["chat_name"] == "MyChatName"
        assert meta["file_tag"] == "report.pdf"

    def test_combined_text_includes_user_and_assistant(self):
        ctx = ContextShell()
        session = StubSession()
        ctx.add_chat_turn(session, "do bees sting?", "Yes, bees sting.")
        doc = ctx._upserted[0]["docs"][0]
        assert "USER: do bees sting?" in doc
        assert "ASSISTANT: Yes, bees sting." in doc

    def test_turn_index_increments_per_turn(self):
        ctx = ContextShell()
        session = StubSession()
        ctx.add_chat_turn(session, "q1", "a1")
        ctx.add_chat_turn(session, "q2", "a2")
        metas = [u["metadatas"][0] for u in ctx._upserted]
        assert metas[0]["turn_index"] == 1
        assert metas[1]["turn_index"] == 2

    def test_conversation_id_present_in_metadata(self):
        ctx = ContextShell(conversation_id="my-unique-conv")
        session = StubSession()
        ctx.add_chat_turn(session, "q", "a")
        assert ctx._upserted[0]["metadatas"][0]["conversation_id"] == "my-unique-conv"


# ---------------------------------------------------------------------------
# Tests — _fetch_context_docs filters by query_lang
# ---------------------------------------------------------------------------


class TestFetchContextDocsQueryLang:

    def _ctx_with_turns(
        self,
        turns: list[tuple[str, str]],  # (query_lang, content)
        conv_id: str = "conv-1",
        chat_name: str = "MyChat",
    ) -> ContextShell:
        """Pre-populate a ContextShell with turns tagged by language."""
        ctx = ContextShell(conversation_id=conv_id)
        for i, (lang, content) in enumerate(turns):
            ctx._fake.stored.append(
                {
                    "document": content,
                    "metadata": {
                        "conversation_id": conv_id,
                        "chat_name": chat_name,
                        "file_tag": "",
                        "query_lang": lang,
                        "turn_index": i + 1,
                    },
                }
            )
        return ctx

    def test_where_filter_includes_query_lang_english_by_default(self):
        ctx = self._ctx_with_turns([])
        session = StubSession(current_query_lang=None)
        ctx._fetch_context_docs(session)
        conditions = ctx._fake.last_where.get("$and", [])
        lang_conds = [c for c in conditions if "query_lang" in c]
        assert len(lang_conds) == 1
        assert lang_conds[0]["query_lang"] == "english"

    def test_where_filter_reflects_current_query_lang(self):
        ctx = self._ctx_with_turns([])
        session = StubSession(current_query_lang="german")
        ctx._fetch_context_docs(session)
        conditions = ctx._fake.last_where.get("$and", [])
        lang_conds = [c for c in conditions if "query_lang" in c]
        assert lang_conds[0]["query_lang"] == "german"

    def test_english_session_sees_only_english_turns(self):
        ctx = self._ctx_with_turns(
            [
                ("english", "[No file filter]\nUSER: do bees sting?\nASSISTANT: Yes."),
                (
                    "german",
                    "[No file filter]\nUSER: haben igel flügel?\nASSISTANT: Nein.",
                ),
            ]
        )
        docs = ctx._fetch_context_docs(StubSession(current_query_lang="english"))
        assert len(docs) == 1
        assert "do bees sting?" in docs[0].page_content

    def test_german_session_sees_only_german_turns(self):
        ctx = self._ctx_with_turns(
            [
                ("english", "[No file filter]\nUSER: do bees sting?\nASSISTANT: Yes."),
                (
                    "german",
                    "[No file filter]\nUSER: haben igel flügel?\nASSISTANT: Nein.",
                ),
            ]
        )
        docs = ctx._fetch_context_docs(StubSession(current_query_lang="german"))
        assert len(docs) == 1
        assert "igel" in docs[0].page_content

    def test_none_lang_session_sees_only_english_turns(self):
        ctx = self._ctx_with_turns(
            [
                ("english", "english turn"),
                ("german", "german turn"),
            ]
        )
        docs = ctx._fetch_context_docs(StubSession(current_query_lang=None))
        assert len(docs) == 1
        assert docs[0].page_content == "english turn"

    def test_no_conversation_id_returns_empty(self):
        ctx = ContextShell(conversation_id="")
        docs = ctx._fetch_context_docs(StubSession())
        assert docs == []

    def test_turns_sorted_by_turn_index(self):
        ctx = self._ctx_with_turns([])
        for turn_index, content in [(3, "third"), (1, "first"), (2, "second")]:
            ctx._fake.stored.append(
                {
                    "document": content,
                    "metadata": {
                        "conversation_id": "conv-1",
                        "chat_name": "MyChat",
                        "file_tag": "",
                        "query_lang": "english",
                        "turn_index": turn_index,
                    },
                }
            )
        docs = ctx._fetch_context_docs(StubSession(current_query_lang="english"))
        assert [d.page_content for d in docs] == ["first", "second", "third"]

    def test_multi_turn_same_language_all_returned(self):
        turns = [("german", f"german turn {i}") for i in range(4)]
        ctx = self._ctx_with_turns(turns)
        docs = ctx._fetch_context_docs(
            StubSession(current_query_lang="german", turns=10)
        )
        assert len(docs) == 4

    def test_language_switch_and_back_recovers_original_context(self):
        """Switching EN → DE → EN re-exposes the original English turns."""
        ctx = ContextShell(conversation_id="conv-1")
        session_en = StubSession(current_query_lang="english", chat_name="MyChat")
        session_de = StubSession(current_query_lang="german", chat_name="MyChat")

        ctx.add_chat_turn(session_en, "do bees sting?", "Yes.")
        ctx.add_chat_turn(session_de, "haben igel flügel?", "Nein.")
        ctx.add_chat_turn(session_en, "do ants bite?", "Yes.")

        # English session — must see exactly 2 English turns, no German
        docs = ctx._fetch_context_docs(session_en)
        content = " ".join(d.page_content for d in docs)
        assert "bees" in content
        assert "ants" in content
        assert "igel" not in content

    def test_german_history_invisible_to_english_query_rewriter(self):
        """The bee/hedgehog cross-contamination bug: German question must not
        see English bee history when the rewriter calls fetch_context_docs."""
        ctx = ContextShell(conversation_id="conv-1")
        session_en = StubSession(current_query_lang="english", chat_name="MyChat")
        session_de = StubSession(current_query_lang="german", chat_name="MyChat")

        # English conversation about bees
        ctx.add_chat_turn(
            session_en, "do bees have stingers?", "Yes, bees have stingers."
        )

        # German question — must see an empty history (no bee contamination)
        docs = ctx._fetch_context_docs(session_de)
        assert docs == []
