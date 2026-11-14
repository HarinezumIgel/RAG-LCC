# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Chat.Chatter import Chatter
from Gui.Colors import ORANGE


def _make_chatter(friendly_name: str) -> Chatter:
    chatter = object.__new__(Chatter)
    chatter.cfg = SimpleNamespace(
        get_str=lambda key: friendly_name if key == "_FRIENDLY_NAME" else "",
        get_dict=lambda _key: {},
        get=lambda key: 120 if key == "TERMINAL_LINE_SIZE" else None,
    )
    chatter.pretty = SimpleNamespace(write=MagicMock())
    chatter.print_llm_answer = MagicMock()
    return chatter


def _make_session(retrieve_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        clarification_response=None,
        retrieve_mode=retrieve_mode,
        strategy="DEFAULT",
    )


def test_web_only_no_results_warns_in_service_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_MODE", "1")
    chatter = _make_chatter("RAGChatService")
    chunks: list[str] = []

    _, response = chatter._handle_no_results(
        _make_session("WEB"), "population of China", "", chunks.append, None
    )

    assert response is not None
    assert "\u26a0\ufe0f **Web-only search returned no usable results.**" in response
    assert "".join(chunks) == response
    assert any("\u26a0\ufe0f" in chunk for chunk in chunks)
    assert "to search the internet for an answer" not in response


def test_web_only_no_results_warns_orange_in_ragchat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_MODE", "1")
    chatter = _make_chatter("RAGChat")

    chatter._handle_no_results(
        _make_session("WEB"), "population of China", "", None, None
    )

    chatter.pretty.write.assert_any_call(
        "W",
        "Web",
        "\u26a0\ufe0f **Web-only search returned no usable results.** "
        "Check the web-search log for provider errors or filtering.",
        color=ORANGE,
    )


def test_non_web_only_no_results_has_no_web_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_MODE", "1")
    chatter = _make_chatter("RAGChat")

    _, response = chatter._handle_no_results(
        _make_session("ALL"), "population of China", "", None, None
    )

    assert response is not None
    assert "Web-only search returned no usable results" not in response
    assert not any(
        call.args[1] == "Web" for call in chatter.pretty.write.call_args_list
    )


@pytest.mark.parametrize("friendly_name", ["RAGChat", "RAGChatService"])
def test_no_results_suggests_new_topic_collection_and_session_settings(
    monkeypatch: pytest.MonkeyPatch,
    friendly_name: str,
) -> None:
    monkeypatch.setenv("WEB_SEARCH_MODE", "0")
    chatter = _make_chatter(friendly_name)

    _, response = chatter._handle_no_results(
        _make_session("ALL"),
        "Where do hedgehogs live?",
        "",
        None,
        None,
    )

    assert response is not None
    assert "`new:`" in response
    assert "Switch to a collection" in response
    assert "session parameters" in response
    assert "retriever_k" not in response
    assert "top_k" not in response
    assert "threshold" not in response
