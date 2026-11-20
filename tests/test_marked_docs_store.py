"""Tests for the in-memory marked-documents store and service helpers."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from Api.MarkedDocsStore import (
    MarkedDocsStore,
    configure_default,
    get_default,
    reset_default,
)
from Api.MarkedDocsService import (
    marked_document_response,
    register_grounded_answer,
    register_marked_documents,
)

# ---------------------------------------------------------------------------
# MarkedDocsStore
# ---------------------------------------------------------------------------


def test_put_and_get_round_trip():
    store = MarkedDocsStore(ttl_seconds=60, max_total_bytes=1024)
    token = store.put(b"hello pdf", "Hedgehogs.pdf")
    assert isinstance(token, str) and len(token) >= 32
    entry = store.get(token)
    assert entry is not None
    assert entry.data == b"hello pdf"
    assert entry.filename == "Hedgehogs.pdf"


def test_get_unknown_token_returns_none():
    store = MarkedDocsStore()
    assert store.get("nope") is None
    assert store.get("") is None


def test_tokens_are_unique_and_unguessable():
    store = MarkedDocsStore()
    tokens = {store.put(b"x", "f.pdf") for _ in range(50)}
    assert len(tokens) == 50  # all unique


def test_ttl_expiry():
    store = MarkedDocsStore(ttl_seconds=1)
    token = store.put(b"data", "f.pdf")
    # Force expiry by mutating expires_at directly (avoids real sleep).
    entry = store._entries[token]  # noqa: SLF001 — test-internal
    entry.expires_at = time.time() - 0.1
    assert store.get(token) is None
    assert len(store) == 0  # expired entry purged on read
    assert store.total_bytes == 0


def test_size_cap_evicts_oldest():
    store = MarkedDocsStore(ttl_seconds=60, max_total_bytes=20)
    t1 = store.put(b"A" * 10, "a.pdf")
    t2 = store.put(b"B" * 10, "b.pdf")
    assert store.get(t1) is not None
    assert store.get(t2) is not None
    # Adding a 10-byte entry forces eviction of the oldest (t1)
    t3 = store.put(b"C" * 10, "c.pdf")
    assert store.get(t1) is None
    assert store.get(t2) is not None
    assert store.get(t3) is not None
    assert store.total_bytes == 20


def test_oversized_payload_rejected():
    store = MarkedDocsStore(max_total_bytes=10)
    with pytest.raises(ValueError):
        store.put(b"X" * 11, "big.pdf")


def test_put_rejects_non_bytes():
    store = MarkedDocsStore()
    with pytest.raises(TypeError):
        store.put("not bytes", "f.pdf")  # type: ignore[arg-type]


def test_single_use_destroys_after_first_get():
    store = MarkedDocsStore(single_use=True)
    token = store.put(b"once", "f.pdf")
    assert store.get(token) is not None
    assert store.get(token) is None
    assert len(store) == 0


def test_default_single_use_false_allows_multiple_fetches():
    store = MarkedDocsStore(single_use=False)
    token = store.put(b"data", "f.pdf")
    assert store.get(token) is not None
    assert store.get(token) is not None


def test_store_preserves_per_entry_response_headers():
    store = MarkedDocsStore()
    token = store.put(
        b"page",
        "grounded-answer.html",
        response_headers={"Content-Security-Policy": "default-src 'none'"},
    )

    entry = store.get(token)
    assert entry is not None
    assert entry.response_headers == {"Content-Security-Policy": "default-src 'none'"}


def test_clear_drops_everything():
    store = MarkedDocsStore()
    store.put(b"1", "a.pdf")
    store.put(b"2", "b.pdf")
    store.clear()
    assert len(store) == 0
    assert store.total_bytes == 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_singleton_configure_and_get():
    reset_default()
    assert get_default() is None
    s = configure_default(ttl_seconds=30, max_total_bytes=100, single_use=True)
    assert get_default() is s
    assert s.ttl_seconds == 30
    assert s.max_total_bytes == 100
    assert s.single_use is True
    reset_default()
    assert get_default() is None


# ---------------------------------------------------------------------------
# register_marked_documents()
# ---------------------------------------------------------------------------


def _make_session(marked):
    return SimpleNamespace(marked_documents=list(marked))


def test_register_returns_empty_when_no_store():
    session = _make_session([("doc.pdf", b"data")])
    assert register_marked_documents(session, None, "http://x") == ""


def test_register_returns_empty_when_no_documents():
    store = MarkedDocsStore()
    session = _make_session([])
    assert register_marked_documents(session, store, "http://x") == ""


def test_register_returns_empty_without_base_url():
    store = MarkedDocsStore()
    session = _make_session([("doc.pdf", b"data")])
    assert register_marked_documents(session, store, "") == ""


def test_register_produces_clickable_markdown():
    store = MarkedDocsStore(ttl_seconds=600)
    session = _make_session(
        [
            ("/abs/path/Hedgehogs.pdf", b"%PDF-1\n..."),
            ("C:/docs/Animals.pdf", b"%PDF-1\n..."),
        ]
    )
    block = register_marked_documents(session, store, "https://host:11435/")
    assert "**Marked sources**" in block
    assert "Hedgehogs.pdf (highlighted)" in block
    assert "Animals.pdf (highlighted)" in block
    # Two registered tokens
    assert block.count("https://host:11435/marked/") == 2
    assert block.endswith(f".pdf?token={store.session_token})\n")
    assert len(store) == 2


def test_register_skips_oversized_silently():
    store = MarkedDocsStore(max_total_bytes=5)
    session = _make_session([("a.pdf", b"X" * 100), ("b.pdf", b"YY")])
    block = register_marked_documents(session, store, "http://h")
    # First entry is too big and dropped; second fits.
    assert "b.pdf" in block
    assert "a.pdf" not in block
    assert len(store) == 1


def test_register_strips_trailing_slash_from_base_url():
    store = MarkedDocsStore()
    session = _make_session([("x.pdf", b"d")])
    block = register_marked_documents(session, store, "http://h:1/")
    assert "http://h:1/marked/" in block
    assert "http://h:1//marked" not in block


def test_register_falls_back_to_original_sources(tmp_path):
    source = tmp_path / "Hedgehogs.pdf"
    source.write_bytes(b"%PDF-1\n...")
    store = MarkedDocsStore(ttl_seconds=600)
    session = SimpleNamespace(
        marked_documents=[],
        last_chosen_chunks=[
            SimpleNamespace(metadata={"Source": "local", "FilePath": str(source)})
        ],
    )

    block = register_marked_documents(session, store, "https://host/")

    assert "**Marked sources**" in block
    assert "Hedgehogs.pdf (source)" in block
    assert "https://host/marked/" in block
    assert len(store) == 1


class _Grounder:
    @staticmethod
    def has_grounded_parts(_session, answer_markdown, evidence):
        return "Hedgehogs eat insects." in answer_markdown and bool(evidence)

    @staticmethod
    def collect_web_texts(_session):
        return []

    @staticmethod
    def mark_answer_html(
        _session, answer_markdown, _evidence, sentence_translations=None
    ):
        return answer_markdown.replace(
            "Hedgehogs eat insects.",
            '<mark class="grounded">Hedgehogs eat insects.</mark>',
        )


def test_register_grounded_answer_creates_expiring_html_link():
    store = MarkedDocsStore(ttl_seconds=600)
    session = SimpleNamespace(
        answer_grounding_verified=True,
        chunk_texts_for_grounding=["Hedgehogs eat insects."],
        grounding_proxy_sentences={},
    )
    block = register_grounded_answer(
        session,
        _Grounder(),
        "<script>alert(1)</script>\n\nHedgehogs eat insects.",
        store,
        "https://host/",
    )

    assert "link expires in 10 min" in block
    assert block.startswith("\n\n---")
    assert "https://host/marked/" in block
    token = block.split("/marked/", 1)[1].split(".html", 1)[0]
    entry = store.get(token)
    assert entry is not None
    assert b'<mark class="grounded">Hedgehogs eat insects.</mark>' in entry.data
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in entry.data
    assert b"<script>alert(1)</script>" not in entry.data
    assert "nonce-" in entry.response_headers["Content-Security-Policy"]


def test_register_partially_grounded_answer_uses_cautious_label():
    store = MarkedDocsStore()
    session = SimpleNamespace(
        answer_grounding_verified=False,
        chunk_texts_for_grounding=["Hedgehogs eat insects."],
        grounding_proxy_sentences={},
        mark_text=True,
    )

    block = register_grounded_answer(
        session, _Grounder(), "Hedgehogs eat insects.", store, "https://host"
    )

    assert "**Not fully grounded**" in block
    assert "https://host/marked/" in block
    token = block.split("/marked/", 1)[1].split(".html", 1)[0]
    entry = store.get(token)
    assert entry is not None
    assert entry.filename == "AnswerPreview.html"


def test_register_unverified_answer_creates_answer_preview():
    store = MarkedDocsStore()
    session = SimpleNamespace(
        answer_grounding_verified=False,
        chunk_texts_for_grounding=["Hedgehogs eat insects."],
        grounding_proxy_sentences={},
        mark_text=True,
    )

    block = register_grounded_answer(
        session, _Grounder(), "Cats are reptiles.", store, "https://host"
    )

    assert "**Answer preview**" in block
    assert "**Not fully grounded**" not in block


def test_marked_document_response_serves_html_with_security_headers():
    store = MarkedDocsStore()
    token = store.put(
        b"<!doctype html>",
        "grounded-answer.html",
        response_headers={
            "Content-Security-Policy": "default-src 'none'",
            "X-Frame-Options": "DENY",
        },
    )

    response = marked_document_response(f"{token}.html", store, store.session_token)

    assert response.status_code == 200
    assert response.media_type == "text/html; charset=utf-8"
    assert response.body == b"<!doctype html>"
    assert response.headers["Content-Security-Policy"] == "default-src 'none'"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_marked_document_response_returns_404_for_expired_html():
    store = MarkedDocsStore(ttl_seconds=1)
    token = store.put(b"page", "grounded-answer.html")
    store._entries[token].expires_at = time.time() - 0.1  # noqa: SLF001

    response = marked_document_response(f"{token}.html", store, store.session_token)

    assert response.status_code == 404
