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
from Api.MarkedDocsService import register_marked_documents

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
    assert block.endswith(".pdf)\n")
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
