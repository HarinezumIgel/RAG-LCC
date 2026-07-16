# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""
Tests for RAGChatImpl._generate_alternate_queries and _remove_similar_chunks.

Heavy transitive imports (torch, sentence-transformers, chromadb) are avoided
via source-extraction: only the two methods are compiled from RAGChatImpl.py
source and bound to lightweight shell objects.
"""

import json
import os
import re
import sys
import textwrap
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.DebugHelper import DebugHelper as _DebugHelper

_RAG_IMPL_SRC = os.path.join(
    os.path.dirname(__file__), "..", "src", "Chat", "RAGChatImpl.py"
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def write(self, *a, **k):
        self.calls.append((a, k))


class StubConfig:
    def __init__(self, overrides: dict[str, Any] | None = None):
        self._data: dict[str, Any] = {
            "_MULTI_QUERY.enabled": True,
            "_MULTI_QUERY.num_variants": 3,
            "_MULTI_QUERY.LLM_PARAM.temperature": 0.5,
            "_MULTI_QUERY.LLM_PARAM.top_k": 40,
            "_MULTI_QUERY.LLM_PARAM.top_p": 0.95,
            "_MULTI_QUERY.LLM_PARAM.num_predict": 256,
            "_MULTI_QUERY.LLM_PARAM.use_ollama_gpu": True,
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key: str, default: Any = None, **kw: Any) -> Any:
        if key.startswith("$"):
            # Simulate indirect config lookup — return a minimal template and name
            template = "Query: {query}\nnum_variants: {num_variants}\n"
            return template, "_PROMPT_QUERY_EXPAND"
        return self._data.get(key, default)

    def get_str(self, key: str, default: str = "") -> str:
        return str(self._data.get(key, default))

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self._data.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._data.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self._data.get(key, default))

    def indirect_get(
        self, key: str, default: Any = None, max_depth: int = 5
    ) -> tuple[Any, str | None]:
        template = "Query: {query}\nnum_variants: {num_variants}\n"
        return (template, key)


class StubTokenBudget:
    def get_effective_context_limit(self, model_name=None, session=None) -> int:
        ctx_override = getattr(session, "context_size_override", None)
        return min(ctx_override, 4096) if ctx_override is not None else 4096


class StubHelpers:
    def get_model_args(self, key: str) -> dict[str, str]:
        return {
            "MODEL": "mistral:7b",
            "PROMPT_QUERY_EXPAND": "_PROMPT_QUERY_EXPAND",
        }


class StubSharedHelpers:
    @staticmethod
    def jaccard(a: list[str], b: list[str]) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if A and B else 0.0


class StubDoc:
    def __init__(self, content: str):
        self.page_content = content
        self.metadata: dict[str, Any] = {}


class StubSession:
    def __init__(self, query: str = "test query", debug_level: int = 0):
        self.query = query
        self.debug_level: int | None = debug_level


# ---------------------------------------------------------------------------
# Stub AI.LLMCaller module — installed before the method source is loaded so
# the `from AI.LLMCaller import LLMCaller` local import inside
# _generate_alternate_queries() resolves to our stub without any disk load.
#
# A mutable dict (_llm_config) is shared between _StubLLMCallerClass and
# _configure_llm() so each test can set the desired response / exception.
# ---------------------------------------------------------------------------

_llm_config: dict[str, Any] = {
    "response": "",
    "raise_exc": None,
    "last_kwargs": {},
}


class _StubLLMCallerClass:
    def call_llm(self, **kwargs: Any) -> dict[str, str]:
        _llm_config["last_kwargs"] = kwargs
        exc = _llm_config["raise_exc"]
        if exc is not None:
            raise exc
        return {"content": _llm_config["response"], "thinking": "", "raw": ""}


_ai_llm_mod = types.ModuleType("AI.LLMCaller")
_ai_llm_mod.LLMCaller = _StubLLMCallerClass  # type: ignore[attr-defined]
sys.modules.setdefault("AI.LLMCaller", _ai_llm_mod)


def _configure_llm(response: str = "", raise_exc: Exception | None = None) -> None:
    """Set the stub LLM response or exception for the next test."""
    _llm_config["response"] = response
    _llm_config["raise_exc"] = raise_exc
    _llm_config["last_kwargs"] = {}


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------


def _extract_method(method_name: str) -> str:
    with open(_RAG_IMPL_SRC, encoding="utf-8") as fh:
        source = fh.read()
    # The signature may be single-line `def foo(self, ...)` or multi-line
    # `def foo(\n    self, ...)`.  Match from `def name(` to the next method /
    # class definition or end-of-file (lookahead so the boundary is not consumed).
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in RAGChatImpl.py"
    return textwrap.dedent(match.group(1))


# ---------------------------------------------------------------------------
# _remove_similar_chunks — compile once at module level
# ---------------------------------------------------------------------------


def _load_remove_func() -> Any:
    ns: dict[str, Any] = {"Any": Any}
    exec(compile(_extract_method("_remove_similar_chunks"), _RAG_IMPL_SRC, "exec"), ns)
    return ns["_remove_similar_chunks"]


_remove_func = _load_remove_func()


class _RemoveShell:
    _remove_similar_chunks = _remove_func

    def __init__(self) -> None:
        self._shared = StubSharedHelpers()


# ---------------------------------------------------------------------------
# _generate_alternate_queries — compile once at module level
# ---------------------------------------------------------------------------


def _load_gen_func() -> Any:
    ns: dict[str, Any] = {"Any": Any, "DebugHelper": _DebugHelper}
    exec(
        compile(_extract_method("_generate_alternate_queries"), _RAG_IMPL_SRC, "exec"),
        ns,
    )
    return ns["_generate_alternate_queries"]


_gen_func = _load_gen_func()


class _GenShell:
    _generate_alternate_queries = _gen_func

    def __init__(self, cfg_overrides: dict[str, Any] | None = None) -> None:
        self.cfg = StubConfig(cfg_overrides)
        self.helperInstance = StubHelpers()
        self.pretty = StubPrettyWriter()
        self.tokenBudget = StubTokenBudget()


# ---------------------------------------------------------------------------
# Tests — _remove_similar_chunks
# ---------------------------------------------------------------------------


class TestRemoveSimilarChunks:
    def test_empty_list_returns_empty(self):
        shell = _RemoveShell()
        assert shell._remove_similar_chunks([], 0.85) == []

    def test_single_doc_always_kept(self):
        shell = _RemoveShell()
        docs = [StubDoc("the quick brown fox")]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 1
        assert result[0] is docs[0]

    def test_distinct_docs_all_kept(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("hedgehogs are small mammals with spines"),
            StubDoc("blue whales are the largest animals on earth"),
            StubDoc("eagles are birds of prey with sharp talons"),
        ]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 3

    def test_exact_duplicate_dropped(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("hedgehogs are small mammals with spines"),
            StubDoc("hedgehogs are small mammals with spines"),
        ]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 1
        assert result[0] is docs[0]  # first (highest-ranked) kept

    def test_below_threshold_both_kept(self):
        shell = _RemoveShell()
        # "apple" vs "banana" — Jaccard = 0.0 → both kept
        docs = [StubDoc("apple"), StubDoc("banana")]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 2

    def test_ranking_order_preserved(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("alpha beta gamma"),  # rank 1 — kept
            StubDoc("delta epsilon zeta"),  # rank 2 — distinct, kept
            StubDoc("alpha beta gamma"),  # rank 3 — dup of rank 1, dropped
        ]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 2
        assert result[0] is docs[0]
        assert result[1] is docs[1]

    def test_threshold_1_drops_only_exact_duplicates(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("the quick brown fox"),
            StubDoc("the quick brown fox"),  # exact dup — Jaccard = 1.0
            StubDoc("the quick brown dog"),  # one word differs — Jaccard < 1
        ]
        result = shell._remove_similar_chunks(docs, 1.0)
        assert len(result) == 2
        assert result[0] is docs[0]
        assert result[1] is docs[2]

    def test_case_insensitive_comparison(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("Hedgehog Has Spines"),
            StubDoc("hedgehog has spines"),  # same tokens after .lower()
        ]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 1

    def test_third_doc_dup_of_second(self):
        shell = _RemoveShell()
        docs = [
            StubDoc("cats are popular pets"),
            StubDoc("dogs are loyal animals"),
            StubDoc("dogs are loyal animals"),  # dup of rank 2
        ]
        result = shell._remove_similar_chunks(docs, 0.85)
        assert len(result) == 2
        assert result[0] is docs[0]
        assert result[1] is docs[1]


# ---------------------------------------------------------------------------
# Tests — _generate_alternate_queries
# ---------------------------------------------------------------------------


class TestGenerateAlternateQueries:
    @pytest.fixture(autouse=True)
    def _stub_llm_caller(self):
        """Install the stub AI.LLMCaller before each test and restore after.

        When the full suite runs, another test module may have already imported
        the real AI.LLMCaller into sys.modules before this file is collected,
        making the module-level setdefault() a no-op.  This fixture guarantees
        the stub is in place for every test in this class, regardless of
        collection order, and restores the original entry afterwards so other
        tests are not affected.
        """
        _original = sys.modules.get("AI.LLMCaller")
        sys.modules["AI.LLMCaller"] = _ai_llm_mod
        yield
        if _original is None:
            sys.modules.pop("AI.LLMCaller", None)
        else:
            sys.modules["AI.LLMCaller"] = _original

    def test_disabled_returns_empty_list(self):
        _configure_llm('["alt1","alt2","alt3"]')
        shell = _GenShell({"_MULTI_QUERY.enabled": False})
        result = shell._generate_alternate_queries(
            "what animals are mammals", StubSession()
        )
        assert result == []

    def test_empty_query_returns_empty_list(self):
        _configure_llm('["alt1","alt2","alt3"]')
        shell = _GenShell()
        result = shell._generate_alternate_queries("", StubSession())
        assert result == []

    def test_valid_response_returns_variants(self):
        _configure_llm(
            '["which animals are classified as mammals?","name the mammals","list of mammals"]'
        )
        shell = _GenShell()
        result = shell._generate_alternate_queries(
            "what animals are mammals", StubSession()
        )
        assert result == [
            "which animals are classified as mammals?",
            "name the mammals",
            "list of mammals",
        ]

    def test_capped_to_num_variants(self):
        _configure_llm('["v1","v2","v3","v4","v5"]')
        shell = _GenShell({"_MULTI_QUERY.num_variants": 2})
        result = shell._generate_alternate_queries("test query", StubSession())
        assert len(result) == 2
        assert result == ["v1", "v2"]

    def test_whitespace_stripped_from_variants(self):
        _configure_llm('["  leading","trailing  ","  both  "]')
        shell = _GenShell()
        result = shell._generate_alternate_queries("test", StubSession())
        assert result == ["leading", "trailing", "both"]

    def test_empty_string_variants_skipped(self):
        _configure_llm('["valid query","","   ","another valid"]')
        shell = _GenShell({"_MULTI_QUERY.num_variants": 4})
        result = shell._generate_alternate_queries("test", StubSession())
        assert "" not in result
        assert "   " not in result
        assert "valid query" in result
        assert "another valid" in result

    def test_llm_exception_returns_empty_list(self):
        _configure_llm(raise_exc=ConnectionError("timeout"))
        shell = _GenShell()
        result = shell._generate_alternate_queries("test query", StubSession())
        assert result == []
        warns = [a for a, _ in shell.pretty.calls if "LLM call failed" in str(a)]
        assert warns

    def test_invalid_json_returns_empty_list_and_warns(self):
        _configure_llm("not json at all")
        shell = _GenShell()
        result = shell._generate_alternate_queries("test query", StubSession())
        assert result == []
        warns = [a for a, _ in shell.pretty.calls if "JSON parse failed" in str(a)]
        assert warns

    def test_non_list_json_returns_empty_list(self):
        _configure_llm('{"key": "value"}')
        shell = _GenShell()
        result = shell._generate_alternate_queries("test query", StubSession())
        assert result == []

    def test_empty_llm_response_returns_empty_list(self):
        _configure_llm("")
        shell = _GenShell()
        result = shell._generate_alternate_queries("test query", StubSession())
        assert result == []

    def test_gpu_disabled_adds_num_gpu_zero(self):
        _configure_llm('["v1"]')
        shell = _GenShell({"_MULTI_QUERY.LLM_PARAM.use_ollama_gpu": False})
        shell._generate_alternate_queries("test query", StubSession())
        opts = _llm_config["last_kwargs"].get("ollama_options", {})
        assert opts.get("num_gpu") == 0

    def test_gpu_enabled_no_num_gpu_key(self):
        _configure_llm('["v1"]')
        shell = _GenShell({"_MULTI_QUERY.LLM_PARAM.use_ollama_gpu": True})
        shell._generate_alternate_queries("test query", StubSession())
        opts = _llm_config["last_kwargs"].get("ollama_options", {})
        assert "num_gpu" not in opts

    def test_llm_call_receives_correct_params(self):
        _configure_llm('["v1","v2","v3"]')
        shell = _GenShell()
        shell._generate_alternate_queries("test query", StubSession())
        kw = _llm_config["last_kwargs"]
        assert kw["model"] == "mistral:7b"
        assert kw["answer_is_json"] is True
        assert kw["streaming"] is False
        assert kw["stage"] == "Multi-query expansion"
        assert kw["template_name"] == "_PROMPT_QUERY_EXPAND"

    def test_prompt_contains_query_and_num_variants(self):
        _configure_llm('["v1","v2","v3"]')
        shell = _GenShell({"_MULTI_QUERY.num_variants": 3})
        shell._generate_alternate_queries("hedgehog spines", StubSession())
        prompt = _llm_config["last_kwargs"].get("prompt", "")
        assert "hedgehog spines" in prompt
        assert "3" in prompt
