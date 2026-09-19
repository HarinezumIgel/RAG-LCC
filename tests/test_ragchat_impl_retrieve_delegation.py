# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
"""Delegation contract tests for RAGChatImpl._retrieve.

Keep this seam stable so orchestration internals can evolve without changing
RAGChatImpl's public behavior.
"""

import os
import re
import sys
import textwrap
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "Chat", "RAGChatImpl.py")


def _extract_method(source: str, method_name: str) -> str:
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    @|\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in RAGChatImpl.py"
    return textwrap.dedent(match.group(1))


with open(_SRC, encoding="utf-8") as _f:
    _RAG_SOURCE = _f.read()


def _compile_method(method_name: str) -> Any:
    method_src = _extract_method(_RAG_SOURCE, method_name)
    ns: dict[str, Any] = {
        "Session": object,
        "Tuple": tuple,
        "Any": Any,
    }
    exec(compile(method_src, _SRC, "exec"), ns)
    return ns[method_name]


_retrieve = _compile_method("_retrieve")
_fetch_local_docs = _compile_method("_fetch_local_docs")
_fetch_web_docs = _compile_method("_fetch_web_docs")


class _OrchestratorStub:
    def __init__(
        self,
        retrieve_result: tuple[str, int],
        local_result: tuple[list[Any], list[Any], list[Any], list[Any]],
        web_result: list[Any],
    ) -> None:
        self.retrieve_result = retrieve_result
        self.local_result = local_result
        self.web_result = web_result
        self.retrieve_calls: list[Any] = []
        self.local_calls: list[tuple[Any, str, str, list[str], str]] = []
        self.web_calls: list[tuple[Any, str, str]] = []

    def run(self, session: Any) -> tuple[str, int]:
        self.retrieve_calls.append(session)
        return self.retrieve_result

    def run_local_retrievers(
        self,
        session: Any,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        self.local_calls.append(
            (
                session,
                retrieve_mode,
                bm25_query,
                list(alternate_queries),
                orig_translated_query_en,
            )
        )
        return self.local_result

    def run_web_retriever(
        self,
        session: Any,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        self.web_calls.append((session, retrieve_mode, user_query_original))
        return self.web_result


class _Shell:
    _retrieve = _retrieve
    _fetch_local_docs = _fetch_local_docs
    _fetch_web_docs = _fetch_web_docs

    def __init__(self, orchestrator: _OrchestratorStub) -> None:
        self.retrieval_orchestrator = orchestrator


class TestRetrieveDelegation:
    def test_retrieve_delegates_to_orchestrator_and_returns_result(self) -> None:
        orchestrator = _OrchestratorStub(("ctx", 3), ([], [], [], []), [])
        shell = _Shell(orchestrator)
        session = object()

        out = shell._retrieve(session)

        assert out == ("ctx", 3)
        assert orchestrator.retrieve_calls == [session]

    def test_fetch_local_docs_delegates_to_orchestrator(self) -> None:
        local_result = (["v1"], ["b1"], ["g1"], ["r1"])
        orchestrator = _OrchestratorStub(("", 0), local_result, [])
        shell = _Shell(orchestrator)
        session = object()

        out = shell._fetch_local_docs(
            session,
            retrieve_mode="ALL",
            bm25_query="Q2",
            alternate_queries=["ALT1", "ALT2"],
            orig_translated_query_en="Q1",
        )

        assert out == local_result
        assert orchestrator.local_calls == [
            (session, "ALL", "Q2", ["ALT1", "ALT2"], "Q1")
        ]

    def test_fetch_web_docs_delegates_to_orchestrator(self) -> None:
        web_result = ["w1", "w2"]
        orchestrator = _OrchestratorStub(("", 0), ([], [], [], []), web_result)
        shell = _Shell(orchestrator)
        session = object()

        out = shell._fetch_web_docs(
            session,
            retrieve_mode="WEB",
            user_query_original="what happened",
        )

        assert out == web_result
        assert orchestrator.web_calls == [(session, "WEB", "what happened")]
