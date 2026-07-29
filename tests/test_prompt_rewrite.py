# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""
Tests for Chat.PromptRewrite:

  - rewrite() returns original query when disabled
  - rewrite() returns original query when no history
  - rewrite() returns rewritten query on success
  - rewrite() returns original query on empty LLM response
  - rewrite() returns original query on LLM exception
  - rewrite() logs purple output on success
  - rewrite() logs debug prompt at debug_level >= 60
  - ollama_options built correctly (incl. num_gpu when GPU disabled)

All LLM calls are stubbed — no real Ollama required.
"""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# We cannot import Chat.PromptRewrite directly because its transitive import
# chain (ChatContext → ModelsCache → sentence_transformers/torch) is
# incompatible with the test environment.  Instead we load only the module
# source, extract the rewrite() method, and bind it to a lightweight stand-in.
# ---------------------------------------------------------------------------

from Gui.Colors import BRIGHT_MAGENTA, ORANGE  # lightweight — always importable
from Helpers.DebugHelper import DebugHelper as _DebugHelper

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self, *a, **k):
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def write(self, *a, **k):
        self.calls.append((a, k))
        return ""


class StubPerfLogger:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def log(self, category: str, subcategory: str, message: str) -> None:
        self.calls.append((category, subcategory, message))


# ---------------------------------------------------------------------------
# Minimal spaCy stubs — no real model needed in tests
# ---------------------------------------------------------------------------


class SpacyMorph:
    """Minimal spaCy Morphology stub supporting .get()."""

    def __init__(self, features: dict[str, list[str]] | None = None):
        self._features = features or {}

    def get(self, feature: str, default: list[str] | None = None) -> list[str]:
        return self._features.get(feature, default or [])


class SpacyToken:
    """Minimal spaCy token stub."""

    def __init__(
        self,
        text: str,
        pos_: str,
        lemma_: str | None = None,
        is_stop: bool = False,
        is_punct: bool = False,
        morph: SpacyMorph | None = None,
        ent_type_: str = "",
    ):
        self.text = text
        self.pos_ = pos_
        self.lemma_ = lemma_ or text.lower()
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.morph = morph or SpacyMorph()
        self.ent_type_ = ent_type_


class SpacyDoc:
    """Minimal spaCy Doc stub."""

    def __init__(self, tokens: list[SpacyToken]):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class StubNlp:
    """Minimal spaCy model stub. Sets Person=3 morph on 3rd-person pronouns,
    Person=1/2 on 1st/2nd-person pronouns, and tags content words as NOUN."""

    _PRON_3RD = frozenset(
        {
            "it",
            "its",
            "itself",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "that",
            "those",
            "these",
        }
    )
    _PRON_1ST_2ND = frozenset(
        {
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
        }
    )
    _STOP = frozenset(
        {
            "what",
            "are",
            "is",
            "the",
            "a",
            "an",
            "of",
            "for",
            "to",
            "do",
            "does",
            "have",
            "has",
            "about",
            "with",
            "in",
            "on",
            "at",
            "and",
            "or",
            "but",
            "so",
            "as",
            "be",
            "been",
            "being",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "not",
            "no",
            "tell",
        }
    )

    def __call__(self, text: str) -> SpacyDoc:
        import re as _re

        tokens: list[SpacyToken] = []
        for word in _re.findall(r"\w+|[^\w\s]", text):
            lw = word.lower()
            if not word.isalnum() and len(word) == 1:
                tokens.append(SpacyToken(word, "PUNCT", lemma_=lw, is_punct=True))
            elif lw in self._PRON_3RD:
                tokens.append(
                    SpacyToken(
                        word, "PRON", lemma_=lw, morph=SpacyMorph({"Person": ["3"]})
                    )
                )
            elif lw in self._PRON_1ST_2ND:
                tokens.append(
                    SpacyToken(
                        word, "PRON", lemma_=lw, morph=SpacyMorph({"Person": ["1"]})
                    )
                )
            elif lw in self._STOP:
                tokens.append(SpacyToken(word, "AUX", lemma_=lw, is_stop=True))
            else:
                tokens.append(SpacyToken(word, "NOUN", lemma_=lw))
        return SpacyDoc(tokens)


class StubConfig:
    def __init__(self, overrides: dict[str, Any] | None = None):
        self._data: dict[str, Any] = {
            "_QUERY_REWRITE.enabled": True,
            "_QUERY_REWRITE.LLM_PARAM.temperature": 0.05,
            "_QUERY_REWRITE.LLM_PARAM.top_k": 10,
            "_QUERY_REWRITE.LLM_PARAM.top_p": 0.9,
            "_QUERY_REWRITE.LLM_PARAM.num_predict": 256,
            "_QUERY_REWRITE.LLM_PARAM.use_ollama_gpu": True,
            "_QUERY_REWRITE.LLM_PARAM.streaming": False,
            "_QUERY_REWRITE.topic_confidence_threshold": 0.5,
            "_QUERY_REWRITE.TOPIC_SUMMARY_MODE": "last",
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None, allow_indirect=True, *, silent=False):
        if key.startswith("$"):
            template = (
                "Previous user utterance : {previous_user_utterance}\n"
                "Rolling topic summary   : {rolling_topic_summary}\n"
                "Current user utterance  : {current_user_utterance}\n"
            )
            return template, "_PROMPT_TOPIC_DETECT"
        return self._data.get(key, default)

    def get_str(self, key, default="") -> str:
        return str(self._data.get(key, default))

    def get_int(self, key, default=0) -> int:
        return int(self._data.get(key, default))

    def get_bool(self, key, default=False) -> bool:
        return bool(self._data.get(key, default))

    def get_float(self, key, default=0.0) -> float:
        return float(self._data.get(key, default))

    def set(self, key, value):
        self._data[key] = value


class StubHelpers:
    def get_model_args(self, key):
        return {
            "MODEL": "mistral:7b",
            "PROMPT_TOPIC_DETECT": "_PROMPT_TOPIC_DETECT",
        }


class StubTokenBudget:
    def get_context_limit(self, model=None):
        return 8192

    def get_effective_context_limit(self, model_name=None, session=None) -> int:
        ctx_limit = self.get_context_limit(model_name)
        ctx_override = getattr(session, "context_size_override", None)
        return min(ctx_override, ctx_limit) if ctx_override is not None else ctx_limit


class StubDoc:
    """Mimics a LangchainDocument."""

    def __init__(self, content: str):
        self.page_content = content
        self.metadata: dict[str, Any] = {}


class StubChatContext:
    def __init__(self, docs: list[StubDoc] | None = None):
        self.docs = docs or []

    def fetch_context_docs(self, session):
        return self.docs


class StubSharedHelpers:
    """Lightweight stand-in for Compliance.SharedHelpers."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        import re as _re

        if not text:
            return []
        toks = _re.findall(r"[A-Za-z0-9]+", text)
        return [t.lower() for t in toks if t.strip()]

    @staticmethod
    def jaccard(a: list[str], b: list[str]) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if A and B else 0.0


class StubFileUtils:
    def get_text_language(self, text: str, fmt: str = "iso") -> str:
        return "en"

    def get_user_text_language(
        self,
        text: str,
        output: str = "nltk",
        native_lang: str | None = None,
        installed_codes=None,
    ) -> str:
        return "en"


class StubLLMCaller:
    def __init__(self, response: str = "", raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_kwargs: dict[str, Any] = {}

    def call_llm(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_exc:
            raise self.raise_exc
        return {"content": self.response, "thinking": "", "raw": ""}


class StubSession:
    def __init__(
        self,
        query: str = "are they mammals",
        debug_level: int = 0,
        turns: int = 10,
        chat_name: str = "TestChat",
        collection_name: str = "TestCol",
        max_history_turns: int = 3,
        file_name: str | None = None,
        file_path: str | None = None,
        context_size_override: int | None = None,
    ):
        self.query: str | None = query
        self.debug_level: int | None = debug_level
        self.turns: int | None = turns
        self.chat_name: str = chat_name
        self.collection_name: str | None = collection_name
        self.use_chat_context: bool | None = True
        self.max_history_turns: int | None = max_history_turns
        self.topic_summary_mode: str | None = "last"
        self.last_topic_referents: list[str] | None = None
        self.file_name: str | None = file_name
        self.file_path: str | None = file_path
        self.force_skip_rewrite: bool = False
        self.rewrite_was_underspecified: bool = False
        self.clarification_response: str | None = None
        self.context_size_override: int | None = context_size_override


# ---------------------------------------------------------------------------
# Load the rewrite() method without importing the module
# ---------------------------------------------------------------------------

import importlib.util
import types

_spec = importlib.util.spec_from_file_location(
    "Chat.PromptRewrite",
    os.path.join(os.path.dirname(__file__), "..", "src", "Chat", "PromptRewrite.py"),
    submodule_search_locations=[],
)


def _load_rewrite_func():
    """Extract the rewrite function source and compile it in an isolated namespace."""
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "Chat", "PromptRewrite.py"
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    # We only need the rewrite() method body.  Extract it by finding the class
    # and compiling it in a namespace that provides BRIGHT_MAGENTA.
    # Simpler approach: exec the method def in isolation.
    import textwrap, re, json, time

    # Pull out the rewrite method source
    match = re.search(
        r"(    def rewrite\(self.*?)(?=\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, "Could not find rewrite() method in PromptRewrite.py"
    method_src = textwrap.dedent(match.group(1))

    from Gui.Colors import VIOLET

    # Module-level constants from PromptRewrite that rewrite() references as globals
    _FIRST_SECOND_PERSON: frozenset[str] = frozenset(
        {
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
        }
    )
    _CONTENT_POS: frozenset[str] = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})
    ns: dict[str, Any] = {
        "BRIGHT_MAGENTA": BRIGHT_MAGENTA,
        "ORANGE": ORANGE,
        "VIOLET": VIOLET,
        "Session": StubSession,
        "Any": Any,
        "SharedHelpers": StubSharedHelpers,
        "re": re,
        "json": json,
        "time": time,
        "_FIRST_SECOND_PERSON": _FIRST_SECOND_PERSON,
        "_CONTENT_POS": _CONTENT_POS,
        "DebugHelper": _DebugHelper,
    }
    exec(compile(method_src, src_path, "exec"), ns)
    return ns["rewrite"]


_rewrite_func = _load_rewrite_func()


class _RewriterShell:
    """Lightweight stand-in for PromptRewrite with rewrite() bound."""

    rewrite = _rewrite_func

    # Default attributes safely attached to avoid missing-attribute errors
    fileUtils = None
    _shared = None


# ---------------------------------------------------------------------------
# Factory — build a rewriter shell with all deps replaced
# ---------------------------------------------------------------------------


def _make_rewriter(
    *,
    cfg_overrides: dict[str, Any] | None = None,
    history_docs: list[StubDoc] | None = None,
    llm_response: str = "",
    llm_exception: Exception | None = None,
    use_gpu: bool = True,
):
    pw = _RewriterShell()

    pw.cfg = StubConfig(cfg_overrides)
    pw.pretty = StubPrettyWriter()
    pw.helpers = StubHelpers()
    pw.tokenBudget = StubTokenBudget()
    pw.chatContext = StubChatContext(history_docs)
    pw.llmCaller = StubLLMCaller(llm_response, llm_exception)
    pw.fileUtils = StubFileUtils()
    pw._shared = StubSharedHelpers()
    pw.perf_logger = StubPerfLogger()

    # Populate fields that __init__ would compute from config
    pw.llm_model = "mistral:7b"
    prompt_cfg = pw.cfg.get("$_PROMPT_TOPIC_DETECT")
    assert isinstance(prompt_cfg, tuple)
    template, name = prompt_cfg
    pw.prompt_template = template
    pw.prompt_name = name
    pw.enabled = pw.cfg.get_bool("_QUERY_REWRITE.enabled")
    pw.temperature = pw.cfg.get_float("_QUERY_REWRITE.LLM_PARAM.temperature")
    pw.top_k = pw.cfg.get_int("_QUERY_REWRITE.LLM_PARAM.top_k")
    pw.top_p = pw.cfg.get_float("_QUERY_REWRITE.LLM_PARAM.top_p")
    pw.num_predict = pw.cfg.get_int("_QUERY_REWRITE.LLM_PARAM.num_predict")
    pw.use_gpu = use_gpu
    pw.streaming = pw.cfg.get_bool("_QUERY_REWRITE.LLM_PARAM.streaming")
    pw.topic_confidence_threshold = pw.cfg.get_float(
        "_QUERY_REWRITE.topic_confidence_threshold"
    )
    pw.topic_summary_mode = (
        pw.cfg.get_str("_QUERY_REWRITE.TOPIC_SUMMARY_MODE") or "last"
    )
    pw._nlp = StubNlp()

    return pw


def _pretty_calls(pw) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    return pw.pretty.calls


# ---------------------------------------------------------------------------
# Tests — disabled / no history early returns
# ---------------------------------------------------------------------------
# Helper JSON builder
# ---------------------------------------------------------------------------


def _json_resp(
    depends: bool = True,
    confidence: float = 0.9,
    reasoning: str = "follow-up",
    contextual: str | None = "contextual rewrite",
    standalone: str = "standalone rewrite",
    referents: list | None = None,
) -> str:
    import json

    return json.dumps(
        {
            "depends_on_previous_turn": depends,
            "confidence": confidence,
            "reasoning": reasoning,
            "contextual_rewrite": contextual,
            "standalone_rewrite": standalone,
            "salient_referents": referents or [],
        }
    )


# ---------------------------------------------------------------------------


class TestRewriteEarlyReturn:
    def test_disabled_returns_original(self):
        rw = _make_rewriter(cfg_overrides={"_QUERY_REWRITE.enabled": False})
        session = StubSession()
        assert rw.rewrite(session) == "are they mammals"
        assert len(_pretty_calls(rw)) == 0

    def test_no_history_returns_original(self):
        rw = _make_rewriter(history_docs=[])
        session = StubSession()
        result = rw.rewrite(session)
        assert result == "are they mammals"
        calls = _pretty_calls(rw)
        assert any("skipping rewrite" in str(c) for c in calls)

    def test_none_query_returns_empty(self):
        rw = _make_rewriter()
        session = StubSession(query=None)
        # No history, disabled path - query is ""
        assert rw.rewrite(session) == ""


# ---------------------------------------------------------------------------
# Tests - successful rewrite
# ---------------------------------------------------------------------------


class TestRewriteSuccess:
    def test_returns_contextual_rewrite(self):
        docs = [StubDoc("USER: what animals are described\nASSISTANT: cats, dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.9,
                contextual="Which of cats, dogs are mammals?",
                standalone="Which animals are mammals?",
            ),
        )
        session = StubSession()
        result = rw.rewrite(session)
        assert result == "Which of cats, dogs are mammals?"

    def test_strips_whitespace_from_contextual(self):
        docs = [StubDoc("USER: hi\nASSISTANT: hello")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                contextual="  What are mammals?  ", standalone="mammals?"
            ),
        )
        session = StubSession()
        assert rw.rewrite(session) == "What are mammals?"

    def test_logs_purple_on_success(self):
        from Gui.Colors import BRIGHT_MAGENTA

        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                contextual="rewritten", standalone="rewritten standalone"
            ),
        )
        session = StubSession()
        rw.rewrite(session)

        found = False
        for _, kwargs in _pretty_calls(rw):
            if kwargs.get("color") == BRIGHT_MAGENTA:
                found = True
                break
        assert found, "Expected a purple-colored log message on successful rewrite"

    def test_logs_debug_prompt_at_level_60(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                contextual="rewritten", standalone="rewritten standalone"
            ),
        )
        session = StubSession(debug_level=60)
        rw.rewrite(session)

        found = False
        for args, _ in _pretty_calls(rw):
            if len(args) >= 3 and "Topic-detect prompt:" in str(args[2]):
                found = True
                break
        assert found, "Expected debug-level prompt dump at debug_level=60"

    def test_no_debug_prompt_at_level_59(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                contextual="rewritten", standalone="rewritten standalone"
            ),
        )
        session = StubSession(debug_level=59)
        rw.rewrite(session)

        for args, _ in _pretty_calls(rw):
            if len(args) >= 3:
                assert "Topic-detect prompt:" not in str(args[2])


# ---------------------------------------------------------------------------
# Tests - failure paths
# ---------------------------------------------------------------------------


class TestRewriteFailure:
    def test_empty_response_returns_original(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(history_docs=docs, llm_response="")
        session = StubSession()
        result = rw.rewrite(session)
        assert result == "are they mammals"
        assert any("Empty response" in str(c) for c in _pretty_calls(rw))

    def test_whitespace_only_response_returns_original(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(history_docs=docs, llm_response="   \n  ")
        session = StubSession()
        assert rw.rewrite(session) == "are they mammals"

    def test_llm_exception_returns_original(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_exception=ConnectionError("timeout"),
        )
        session = StubSession()
        result = rw.rewrite(session)
        assert result == "are they mammals"
        assert any("LLM call failed" in str(c) for c in _pretty_calls(rw))


# ---------------------------------------------------------------------------
# Tests - ollama_options assembly
# ---------------------------------------------------------------------------


class TestOllamaOptions:
    def test_default_options(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(contextual="rewritten", standalone="r"),
        )
        session = StubSession()
        rw.rewrite(session)

        opts = rw.llmCaller.last_kwargs.get("ollama_options", {})
        assert opts["temperature"] == 0.05
        assert opts["top_k"] == 10
        assert opts["top_p"] == 0.9
        assert opts["num_predict"] == 256
        assert opts["num_ctx"] == 8192
        assert "num_gpu" not in opts

    def test_gpu_disabled_adds_num_gpu_0(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(contextual="r", standalone="r"),
            use_gpu=False,
        )
        session = StubSession()
        rw.rewrite(session)

        opts = rw.llmCaller.last_kwargs.get("ollama_options", {})
        assert opts["num_gpu"] == 0

    def test_call_llm_receives_correct_params(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession()
        rw.rewrite(session)

        kw = rw.llmCaller.last_kwargs
        assert kw["model"] == "mistral:7b"
        assert kw["answer_is_json"] is True
        assert kw["streaming"] is False
        assert kw["stage"] == "Topic detect / query rewrite"
        assert kw["template_name"] == "_PROMPT_TOPIC_DETECT"

    def test_context_size_override_applied(self):
        """session.context_size_override < auto → num_ctx uses override."""
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession(context_size_override=4000)
        rw.rewrite(session)

        opts = rw.llmCaller.last_kwargs.get("ollama_options", {})
        assert opts["num_ctx"] == 4000

    def test_context_size_override_clamped_to_auto(self):
        """session.context_size_override > auto (8192) → num_ctx clamped to auto."""
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession(context_size_override=99999)
        rw.rewrite(session)

        opts = rw.llmCaller.last_kwargs.get("ollama_options", {})
        assert opts["num_ctx"] == 8192  # clamped to StubTokenBudget limit

    def test_no_override_uses_auto(self):
        """session.context_size_override is None → num_ctx uses auto-detected limit."""
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession(context_size_override=None)
        rw.rewrite(session)

        opts = rw.llmCaller.last_kwargs.get("ollama_options", {})
        assert opts["num_ctx"] == 8192


# ---------------------------------------------------------------------------
# Tests - prompt formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    def test_prompt_contains_previous_utterance_and_query(self):
        docs = [
            StubDoc("USER: what animals\nASSISTANT: cats and dogs"),
            StubDoc("USER: tell me more\nASSISTANT: they are pets"),
        ]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        # max_history_turns=0 disables slicing
        session = StubSession(query="are they mammals", max_history_turns=0)
        rw.rewrite(session)

        prompt = rw.llmCaller.last_kwargs.get("prompt", "")
        assert "tell me more" in prompt  # previous_user_utterance from last turn
        assert "they are pets" in prompt  # rolling_topic_summary
        assert "are they mammals" in prompt  # current_user_utterance

    def test_prompt_sliced_to_max_history_turns(self):
        docs = [
            StubDoc("USER: what animals\nASSISTANT: cats and dogs"),
            StubDoc("USER: tell me more\nASSISTANT: they are pets"),
        ]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession(query="are they mammals", max_history_turns=1)
        rw.rewrite(session)

        prompt = rw.llmCaller.last_kwargs.get("prompt", "")
        # Only last turn
        assert "what animals" not in prompt
        assert "tell me more" in prompt
        assert "are they mammals" in prompt

    def test_info_log_shows_history_count(self):
        docs = [
            StubDoc("USER: t1\nASSISTANT: a1"),
            StubDoc("USER: t2\nASSISTANT: a2"),
            StubDoc("USER: t3\nASSISTANT: a3"),
        ]
        rw = _make_rewriter(
            history_docs=docs, llm_response=_json_resp(contextual="r", standalone="r")
        )
        session = StubSession(max_history_turns=0)
        rw.rewrite(session)

        assert any("3 history turns" in str(c) for c in _pretty_calls(rw))


# ---------------------------------------------------------------------------
# Tests - topic detection logic
# ---------------------------------------------------------------------------


class TestTopicDetection:
    def test_depends_false_no_pronoun_returns_original_query(self):
        # depends=False + no 3rd-person pronoun: the LLM standalone_rewrite is
        # discarded to prevent hallucinated entities; the original query is returned.
        docs = [StubDoc("USER: what animals\nASSISTANT: cats and dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=0.1,
                contextual=None,
                standalone="standalone answer",
            ),
        )
        session = StubSession(query="what are mammals")
        assert rw.rewrite(session) == "what are mammals"

    def test_depends_true_returns_contextual_rewrite(self):
        docs = [StubDoc("USER: what animals\nASSISTANT: cats and dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.9,
                contextual="Are cats and dogs mammals?",
                standalone="Are animals mammals?",
            ),
        )
        session = StubSession()
        assert rw.rewrite(session) == "Are cats and dogs mammals?"

    def test_low_confidence_returns_standalone(self):
        docs = [StubDoc("USER: what animals\nASSISTANT: cats and dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.3,  # below threshold 0.5
                contextual="Are cats and dogs mammals?",
                standalone="standalone fallback",
            ),
        )
        session = StubSession()
        assert rw.rewrite(session) == "standalone fallback"

    def test_json_parse_failure_falls_back_to_original(self):
        docs = [StubDoc("USER: q\nASSISTANT: a")]
        rw = _make_rewriter(history_docs=docs, llm_response="not valid json at all")
        session = StubSession()
        result = rw.rewrite(session)
        assert result == "are they mammals"
        assert any("JSON parse failed" in str(c) for c in _pretty_calls(rw))

    def test_null_contextual_rewrite_falls_back_to_standalone(self):
        docs = [StubDoc("USER: what animals\nASSISTANT: cats and dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.9,
                contextual=None,
                standalone="standalone fallback",
            ),
        )
        session = StubSession()
        assert rw.rewrite(session) == "standalone fallback"

    def test_logs_reasoning_and_referents(self):
        docs = [StubDoc("USER: what animals\nASSISTANT: cats and dogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.85,
                reasoning="query refers to previously listed animals",
                contextual="Are cats and dogs mammals?",
                standalone="Are animals mammals?",
                referents=["cats", "dogs"],
            ),
        )
        session = StubSession()
        rw.rewrite(session)

        all_msgs = str(_pretty_calls(rw))
        assert "depends=True" in all_msgs
        assert "0.85" in all_msgs
        assert "reasoning" in all_msgs
        assert "cats" in all_msgs

    def test_referents_saved_and_used_as_compact_summary_on_next_turn(self):
        """After turn 1 extracts referents, turn 2 should receive them as the
        rolling_topic_summary instead of the ASSISTANT prose block."""
        docs = [
            StubDoc(
                "USER: what animals\nASSISTANT: A very long prose answer about cats and dogs."
            )
        ]

        # Turn 1: LLM returns referents ["cats", "dogs"]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.9,
                contextual="Are cats and dogs mammals?",
                standalone="Are animals mammals?",
                referents=["cats", "dogs"],
            ),
        )
        session = StubSession()
        rw.rewrite(session)

        # Session should now have referents saved
        assert session.last_topic_referents == ["cats", "dogs"]

        # Turn 2: reuse same session (referents now set); check prompt content
        rw2 = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(contextual="r", standalone="r"),
        )
        rw2.rewrite(session)

        prompt = rw2.llmCaller.last_kwargs.get("prompt", "")
        assert "Key entities from previous turn: cats, dogs" in prompt
        assert "A very long prose answer" not in prompt

    def test_history_grounded_resolution_accepted(self):
        """Bee case: 'bee' in standalone is not in original query but IS in history.
        → true_hallucinations empty → rewrite accepted, NOT underspecified.

        Turn 1: 'do bees have stings?'
        Turn 2: 'can they fly?'   LLM standalone: 'can bees fly?'
        """
        docs = [StubDoc("USER: do bees have stings\nASSISTANT: Yes bees have stingers")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                contextual=None,
                standalone="can bees fly?",
                referents=[],
            ),
        )
        session = StubSession(query="can they fly?")
        result = rw.rewrite(session)

        assert result == "can bees fly?"
        assert session.rewrite_was_underspecified is False
        # Grounded-via-history log emitted (I-level)
        info = [c for c, _ in _pretty_calls(rw) if c[0] == "I"]
        assert any("Standalone grounded via history" in str(i) for i in info)

    def test_pronoun_dropped_without_entity_is_underspecified(self):
        """Hedgehog/RAM: LLM strips 'they' but adds no entity → underspecified.

        Turn 1: 'do hedgehogs hibernate?'
        Turn 2: 'do they have RAM?'   LLM standalone: 'do have RAM?'
        The pronoun was dropped but no referent was substituted → ask for clarification.
        """
        docs = [
            StubDoc(
                "USER: do hedgehogs hibernate\nASSISTANT: Yes hedgehogs hibernate in winter"
            )
        ]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                contextual=None,
                standalone="do have RAM?",
                referents=[],
            ),
        )
        session = StubSession(query="do they have RAM?")
        rw.rewrite(session)

        assert session.rewrite_was_underspecified is True

    def test_hedgehog_ram_pronoun_resolved_via_history_is_accepted(self):
        """Hedgehog/RAM: LLM correctly substitutes 'they' → 'hedgehogs'.
        'hedgehog' is in the history → no true hallucination → accepted.
        Retrieval proceeds (may return no results, but pronoun is resolved).
        """
        docs = [
            StubDoc(
                "USER: do hedgehogs hibernate\nASSISTANT: Yes hedgehogs hibernate in winter"
            )
        ]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                contextual=None,
                standalone="do hedgehogs have RAM?",
                referents=[],
            ),
        )
        session = StubSession(query="do they have RAM?")
        result = rw.rewrite(session)

        assert result == "do hedgehogs have RAM?"
        assert session.rewrite_was_underspecified is False

    def test_true_hallucination_not_in_history_is_rejected(self):
        """LLM invents 'computers' — not in hedgehog/hibernation history → rejected."""
        docs = [
            StubDoc(
                "USER: do hedgehogs hibernate\nASSISTANT: Yes hedgehogs hibernate in winter"
            )
        ]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                contextual=None,
                standalone="can computers fly?",
                referents=[],
            ),
        )
        session = StubSession(query="can they fly?")
        rw.rewrite(session)

        assert session.rewrite_was_underspecified is True
        warns = [c for c, _ in _pretty_calls(rw) if c[0] == "W"]
        assert any("Standalone rejected" in str(w) for w in warns)

    def test_referents_cleared_on_topic_switch(self):
        session = StubSession()
        session.last_topic_referents = ["cats", "dogs"]
        session.force_skip_rewrite = True

        rw = _make_rewriter(history_docs=[StubDoc("USER: q\nASSISTANT: a")])
        rw.rewrite(session)

        assert session.last_topic_referents is None


class TestGroundingCheck:
    """Post-hoc grounding check: reject standalone rewrites that invent entities
    when depends=False, referents=[], and the original query contains a pronoun."""

    def test_entity_invention_rejected_when_original_has_pronoun(self):
        """Standalone invents 'hedgehogs' — rejected because 'hedgehog' is NOT in history."""
        docs = [StubDoc("USER: what is the top speed\nASSISTANT: very fast")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                standalone="What are the RAM specifications of hedgehogs?",
                referents=[],
            ),
        )
        session = StubSession(query="What are its RAM specifications?")
        result = rw.rewrite(session)

        # Pronoun stripped, hallucinated entity not present
        assert "hedgehog" not in result.lower()
        assert "RAM" in result or "ram" in result.lower()

        # W-level rejection log emitted
        warns = [c for c, _ in _pretty_calls(rw) if c[0] == "W"]
        assert any("grounding_score" in str(w) for w in warns)
        assert any("hedgehog" in str(w) for w in warns)

    def test_depends_false_no_pronoun_discards_standalone_rewrite(self):
        """No pronoun in original + depends=False: standalone_rewrite is discarded
        (it may contain hallucinated entities such as 'Igel')."""
        docs = [StubDoc("USER: what animal\nASSISTANT: hedgehogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=0.9,
                standalone="What animals are commonly kept as pets?",
                referents=[],
            ),
        )
        session = StubSession(query="Tell me about animals.")
        result = rw.rewrite(session)

        # Standalone discarded — original query returned unchanged
        assert result == "Tell me about animals."
        warns = [c for c, _ in _pretty_calls(rw) if c[0] == "W"]
        assert not any("grounding_score" in str(w) for w in warns)

    def test_standalone_accepted_when_depends_true(self):
        """depends=True with referents — grounding check is skipped entirely."""
        docs = [StubDoc("USER: what animal\nASSISTANT: hedgehogs")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=True,
                confidence=0.9,
                contextual="What are hedgehog spines made of?",
                standalone="What are hedgehog spines made of?",
                referents=["hedgehogs"],
            ),
        )
        session = StubSession(query="What are its spines made of?")
        result = rw.rewrite(session)

        assert result == "What are hedgehog spines made of?"
        warns = [c for c, _ in _pretty_calls(rw) if c[0] == "W"]
        assert not any("grounding_score" in str(w) for w in warns)

    def test_standalone_kept_when_no_invention_despite_pronoun(self):
        """Original has 'its', standalone introduces no new content words — rewrite
        is kept, but rewrite_was_underspecified is still set because the pronoun
        had no resolvable referent."""
        docs = [StubDoc("USER: what spec\nASSISTANT: speed")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=0.8,
                standalone="What are the specifications?",
                referents=[],
            ),
        )
        session = StubSession(query="What are its specifications?")
        result = rw.rewrite(session)

        assert result == "What are the specifications?"
        # No W log because no entity was invented
        warns = [c for c, _ in _pretty_calls(rw) if c[0] == "W"]
        assert not any("grounding_score" in str(w) for w in warns)
        # Flag is set regardless — pronoun with no referent is always underspecified
        assert session.rewrite_was_underspecified is True


class TestUnderspecifiedFlag:
    """rewrite_was_underspecified is set when grounding check fires and rejects."""

    def test_flag_set_when_invention_rejected(self):
        docs = [StubDoc("USER: what is the top speed\nASSISTANT: very fast")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=1.0,
                standalone="What are the RAM specifications of hedgehogs?",
                referents=[],
            ),
        )
        session = StubSession(query="What are its RAM specifications?")
        rw.rewrite(session)
        assert session.rewrite_was_underspecified is True

    def test_flag_set_even_when_no_invention(self):
        """Pronoun + depends=False + referents=[] is underspecified even if rewrite is clean."""
        docs = [StubDoc("USER: what spec\nASSISTANT: speed")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=0.8,
                standalone="What are the specifications?",
                referents=[],
            ),
        )
        session = StubSession(query="Does it have spines?")
        rw.rewrite(session)
        assert session.rewrite_was_underspecified is True

    def test_flag_not_set_when_no_pronoun(self):
        docs = [StubDoc("USER: what animal\nASSISTANT: cats")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False,
                confidence=0.9,
                standalone="What animals are kept as pets?",
                referents=[],
            ),
        )
        session = StubSession(query="Tell me about animals.")
        rw.rewrite(session)
        assert session.rewrite_was_underspecified is False

    def test_flag_reset_at_start_of_each_rewrite(self):
        """Flag from a prior turn must be cleared even if this turn has no pronoun."""
        docs = [StubDoc("USER: what\nASSISTANT: a")]
        rw = _make_rewriter(
            history_docs=docs,
            llm_response=_json_resp(
                depends=False, standalone="normal query", referents=[]
            ),
        )
        session = StubSession(query="what are mammals")
        session.rewrite_was_underspecified = True  # stale from previous turn
        rw.rewrite(session)
        assert session.rewrite_was_underspecified is False
