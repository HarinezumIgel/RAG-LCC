# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Tests for configurable dedup scope in RAGChatImpl._merge_and_select.

Heavy transitive imports are avoided via source extraction. Only
_merge_and_select is compiled from RAGChatImpl source and bound to a shell
object with lightweight stubs.
"""

import os
import re
import sys
import textwrap
import types
from typing import Any, cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_RAG_IMPL_SRC = os.path.join(
    os.path.dirname(__file__), "..", "src", "Chat", "RAGChatImpl.py"
)


def _extract_method(method_name: str) -> str:
    with open(_RAG_IMPL_SRC, encoding="utf-8") as fh:
        source = fh.read()
    match = re.search(
        rf"(    def {re.escape(method_name)}\(.*?)(?=\n    def |\nclass |\Z)",
        source,
        re.DOTALL,
    )
    assert match, f"Could not find {method_name}() in RAGChatImpl.py"
    return textwrap.dedent(match.group(1))


class _StubChunkSelectionService:
    def __init__(self, session: Any) -> None:
        _ = session

    def select_chunks(self, chunks: list[Any]) -> list[Any]:
        return list(chunks)


class _StubBM25Retriever:
    @staticmethod
    def reciprocal_rank_fusion(*args: Any, **kwargs: Any) -> list[Any]:
        _ = (args, kwargs)
        return []


class _StubDebugHelper:
    @staticmethod
    def check_session(session: Any, level: int) -> bool:
        _ = (session, level)
        return False


def _load_merge_func() -> Any:
    ns: dict[str, Any] = {
        "Any": Any,
        "Session": Any,
        "cast": cast,
        "ChunkSelectionService": _StubChunkSelectionService,
        "BM25Retriever": _StubBM25Retriever,
        "DebugHelper": _StubDebugHelper,
        "CYAN": "",
    }
    exec(compile(_extract_method("_merge_and_select"), _RAG_IMPL_SRC, "exec"), ns)
    return ns["_merge_and_select"]


_merge_func = _load_merge_func()


class _StubCfg:
    def __init__(self, include_web_chunks: bool) -> None:
        self._data = {
            "_CHUNK_DEDUP.enabled": True,
            "_CHUNK_DEDUP.threshold": 0.85,
            "_CHUNK_DEDUP.include_web_chunks": include_web_chunks,
        }

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._data.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self._data.get(key, default)
        return float(value) if value is not None else default


class _StubSession:
    def __init__(self) -> None:
        self.vector_weight: float | None = None
        self.bm25_weight: float | None = None
        self.graph_weight: float | None = None
        self.regex_weight: float | None = None
        self.retriever_k: int = 10
        self.rerank: int = 0
        self.debug_level: int = 0
        self.debug_mode: str = "ge"


class _StubPretty:
    def write(self, *a: Any, **k: Any) -> None:
        _ = (a, k)


class _Shell:
    _merge_and_select = _merge_func

    def __init__(self, include_web_chunks: bool) -> None:
        self.cfg = _StubCfg(include_web_chunks)
        self.pretty = _StubPretty()
        self.bm25_retriever = types.SimpleNamespace(rrf_k=60)

    def _remove_similar_chunks(self, docs: list[Any], threshold: float) -> list[Any]:
        _ = threshold
        seen_texts: set[str] = set()
        out: list[Any] = []
        for doc in docs:
            text = str(getattr(doc, "page_content", "") or "")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            out.append(doc)
        return out

    def _rerank(self, mySession: Any, docs: list[Any]) -> list[Any]:
        _ = mySession
        return list(docs)

    def _print_merged_debug(self, docs: list[Any]) -> None:
        _ = docs


def _mk_doc(text: str, *, source: str) -> Any:
    doc = types.SimpleNamespace()
    doc.page_content = text
    if source == "web":
        doc.metadata = {"Source": "Web", "retriever_sources": "Web"}
    else:
        doc.metadata = {"Source": "Local", "retriever_sources": "Vector"}
    return doc


class TestMergeDedupConfig:
    def test_include_web_chunks_true_dedups_local_and_web(self):
        shell = _Shell(include_web_chunks=True)
        session = _StubSession()
        local = _mk_doc("same content", source="local")
        web = _mk_doc("same content", source="web")

        result = shell._merge_and_select(
            session,
            [local],
            [],
            [],
            [],
            [web],
        )

        assert len(result) == 1
        assert result[0].metadata.get("Source") == "Local"

    def test_include_web_chunks_false_only_dedups_local(self):
        shell = _Shell(include_web_chunks=False)
        session = _StubSession()
        local = _mk_doc("same content", source="local")
        web = _mk_doc("same content", source="web")

        result = shell._merge_and_select(
            session,
            [local],
            [],
            [],
            [],
            [web],
        )

        assert len(result) == 2
        assert any(d.metadata.get("Source") == "Web" for d in result)
