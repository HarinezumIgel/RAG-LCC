"""In-memory store for visually-marked documents served over HTTP.

Used by ``RAGChatService`` to expose highlighted PDFs to OpenWebUI (or
any HTTP client) via short-lived, unguessable URLs without ever
touching the filesystem.

Security model:
  * Tokens are 256-bit ``secrets.token_urlsafe`` values — unforgeable
    and uniformly distributed.  Acting as a capability, the token is
    the only credential needed to fetch the bytes.
  * Entries auto-expire after ``ttl_seconds`` (configurable).
  * Total store size is capped (``max_total_bytes``); oldest entries
    are evicted when the cap would be exceeded.
  * Optional single-use mode (``single_use``): the entry is destroyed
    after the first successful fetch.
  * No cross-user isolation is implemented at this layer — any
    request that knows a token can fetch the bytes.  Combine with the
    existing Bearer-token middleware (``_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY``) on the
    same FastAPI app, or place the service behind a trusted reverse
    proxy, when stricter controls are required.

Thread-safety:
  * All mutating operations are guarded by a ``threading.Lock`` so
    the store can be shared between FastAPI worker threads.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarkedEntry:
    """A single cached document."""

    data: bytes
    filename: str
    expires_at: float  # epoch seconds


class MarkedDocsStore:
    """Bounded, TTL-expiring in-memory store keyed by random tokens."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 1800,
        max_total_bytes: int = 200 * 1024 * 1024,
        single_use: bool = False,
    ) -> None:
        self.ttl_seconds: int = max(1, int(ttl_seconds))
        self.max_total_bytes: int = max(1, int(max_total_bytes))
        self.single_use: bool = bool(single_use)
        self._entries: "OrderedDict[str, MarkedEntry]" = OrderedDict()
        self._total_bytes: int = 0
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def put(self, data: bytes, filename: str) -> str:
        """Store *data* under a fresh token and return that token."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        size = len(data)
        if size > self.max_total_bytes:
            raise ValueError(
                f"document size {size} exceeds cache capacity {self.max_total_bytes}"
            )
        entry = MarkedEntry(
            data=bytes(data),
            filename=str(filename),
            expires_at=time.time() + self.ttl_seconds,
        )
        with self._lock:
            self._purge_expired_locked()
            self._evict_for_size_locked(size)
            # Generate token inside the lock and check uniqueness.
            # Collision chance is 2^-256 — the loop body effectively never runs.
            token = secrets.token_urlsafe(32)
            while token in self._entries:
                token = secrets.token_urlsafe(32)
            self._entries[token] = entry
            self._total_bytes += size
        return token

    def get(self, token: str) -> Optional[MarkedEntry]:
        """Return the entry for *token* or ``None`` if missing/expired.

        Honours ``single_use``: when enabled the entry is removed after
        a successful fetch.
        """
        if not token:
            return None
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(token)
            if entry is None:
                return None
            if entry.expires_at <= time.time():
                self._remove_locked(token)
                return None
            if self.single_use:
                self._remove_locked(token)
            return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    # ------------------------------------------------------------------
    # Introspection (for tests / diagnostics)
    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal — must be called while holding ``self._lock``
    # ------------------------------------------------------------------

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [t for t, e in self._entries.items() if e.expires_at <= now]
        for t in expired:
            self._remove_locked(t)

    def _evict_for_size_locked(self, incoming: int) -> None:
        # Evict oldest entries (FIFO via OrderedDict insertion order)
        # until the new entry fits within the cap.
        while self._total_bytes + incoming > self.max_total_bytes and self._entries:
            oldest_token, _ = next(iter(self._entries.items()))
            self._remove_locked(oldest_token)

    def _remove_locked(self, token: str) -> None:
        entry = self._entries.pop(token, None)
        if entry is not None:
            self._total_bytes -= len(entry.data)


# Module-level singleton, lazily configured by RAGChatService at startup.
_INSTANCE: Optional[MarkedDocsStore] = None
_INSTANCE_LOCK = threading.Lock()


def configure_default(
    *,
    ttl_seconds: int,
    max_total_bytes: int,
    single_use: bool,
) -> MarkedDocsStore:
    """Create / replace the module-level default store."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = MarkedDocsStore(
            ttl_seconds=ttl_seconds,
            max_total_bytes=max_total_bytes,
            single_use=single_use,
        )
        return _INSTANCE


def get_default() -> Optional[MarkedDocsStore]:
    """Return the configured singleton store, or ``None`` if disabled."""
    with _INSTANCE_LOCK:
        return _INSTANCE


def reset_default() -> None:
    """Drop the singleton (used by tests)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
