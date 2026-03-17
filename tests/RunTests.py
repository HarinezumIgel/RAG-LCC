# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Run tests using pytest.

Usage:
    python tests/RunTests.py
"""

import os
import socket
import sys
from datetime import datetime
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Lightweight network monitor for the test suite.
# Logs every outbound socket.connect() so operators can verify that no test
# silently contacts the internet.  RFC 5737 TEST-NET addresses
# (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) are annotated.
# ---------------------------------------------------------------------------

_RED = "\033[31m"
_RESET = "\033[0m"

_TEST_NETS = {
    "192.0.2.": "TEST-NET-1 (RFC 5737)",
    "198.51.100.": "TEST-NET-2 (RFC 5737)",
    "203.0.113.": "TEST-NET-3 (RFC 5737)",
}

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_bind = socket.socket.bind
_original_listen = socket.socket.listen

# Duplicate the real stderr fd *before* pytest's fd-level capture replaces it.
# os.write to this fd bypasses pytest capture so the annotation always appears.
_tty_fd: int = os.dup(2)


import traceback as _tb


def _test_traced_connect(self_sock: socket.socket, address: Any) -> Any:
    host: str | None = None
    port: int | None = None
    if isinstance(address, tuple) and len(address) >= 2:
        h: Any = address[0]
        p: Any = address[1]
        if isinstance(h, str):
            host = h
        if isinstance(p, int):
            port = p

    if host is not None:
        label = ""
        for prefix, net_name in _TEST_NETS.items():
            if host.startswith(prefix):
                label = f"  [RFC 5737 {net_name} — expected, non-routable]"
                break

        # For unlabeled (unexpected) connections, include a short stack trace
        # to help identify which test or import triggers them.
        trace = ""
        if not label:
            frames = _tb.extract_stack()
            # Keep only project frames (exclude site-packages and stdlib)
            proj = [
                f
                for f in frames
                if "RAG-LCC" in f.filename and "site-packages" not in f.filename
            ]
            if proj:
                trace = "\n   Stack (project frames):\n"
                for fr in proj[-6:]:
                    trace += f"     {fr.filename}:{fr.lineno} in {fr.name}\n"

        suffix = f":{port}" if port is not None else ""
        ts = datetime.now().isoformat(timespec="seconds")
        msg = (
            f"\n🔵 [Test Network Monitor] {ts}  "
            f"connect → {host}{suffix}{label}{trace}\n"
        )
        os.write(_tty_fd, msg.encode())

    return _original_connect(self_sock, address)


def _test_traced_connect_ex(self_sock: socket.socket, address: Any) -> int:
    host: str | None = None
    port: int | None = None
    if isinstance(address, tuple) and len(address) >= 2:
        h: Any = address[0]
        p: Any = address[1]
        if isinstance(h, str):
            host = h
        if isinstance(p, int):
            port = p

    if host is not None:
        label = ""
        for prefix, net_name in _TEST_NETS.items():
            if host.startswith(prefix):
                label = f"  [RFC 5737 {net_name} — expected, non-routable]"
                break

        trace = ""
        if not label:
            frames = _tb.extract_stack()
            proj = [
                f
                for f in frames
                if "RAG-LCC" in f.filename and "site-packages" not in f.filename
            ]
            if proj:
                trace = "\n   Stack (project frames):\n"
                for fr in proj[-6:]:
                    trace += f"     {fr.filename}:{fr.lineno} in {fr.name}\n"

        suffix = f":{port}" if port is not None else ""
        ts = datetime.now().isoformat(timespec="seconds")
        msg = (
            f"\n🔵 [Test Network Monitor] {ts}  "
            f"connect_ex → {host}{suffix}{label}{trace}\n"
        )
        os.write(_tty_fd, msg.encode())

    return _original_connect_ex(self_sock, address)


def _test_traced_bind(self_sock: socket.socket, address: Any) -> Any:
    host: str | None = None
    port: int | None = None
    if isinstance(address, tuple) and len(address) >= 2:
        h: Any = address[0]
        p: Any = address[1]
        if isinstance(h, str):
            host = h
        if isinstance(p, int):
            port = p

    suffix = f":{port}" if port is not None else ""
    display = f"{host}{suffix}" if host is not None else repr(address)

    frames = _tb.extract_stack()
    proj = [
        f
        for f in frames
        if "RAG-LCC" in f.filename and "site-packages" not in f.filename
    ]
    trace = ""
    if proj:
        trace = "\n   Stack (project frames):\n"
        for fr in proj[-6:]:
            trace += f"     {fr.filename}:{fr.lineno} in {fr.name}\n"

    ts = datetime.now().isoformat(timespec="seconds")
    msg = (
        f"\n{_RED}\U0001f534 [Test Network Monitor] {ts}  "
        f"bind \u2192 {display}{_RESET}{trace}\n"
    )
    os.write(_tty_fd, msg.encode())

    return _original_bind(self_sock, address)


def _test_traced_listen(self_sock: socket.socket, backlog: int = 1) -> None:
    try:
        local_addr = self_sock.getsockname()
    except Exception:
        local_addr = "(unknown)"

    frames = _tb.extract_stack()
    proj = [
        f
        for f in frames
        if "RAG-LCC" in f.filename and "site-packages" not in f.filename
    ]
    trace = ""
    if proj:
        trace = "\n   Stack (project frames):\n"
        for fr in proj[-6:]:
            trace += f"     {fr.filename}:{fr.lineno} in {fr.name}\n"

    ts = datetime.now().isoformat(timespec="seconds")
    msg = (
        f"\n{_RED}\U0001f534 [Test Network Monitor] {ts}  "
        f"listen \u2192 {local_addr!r}  backlog={backlog}{_RESET}{trace}\n"
    )
    os.write(_tty_fd, msg.encode())

    return _original_listen(self_sock, backlog)


def main() -> None:
    # Patch socket.connect, connect_ex, bind, and listen for the duration of the test run
    socket.socket.connect = _test_traced_connect  # type: ignore[assignment]
    socket.socket.connect_ex = _test_traced_connect_ex  # type: ignore[assignment]
    socket.socket.bind = _test_traced_bind  # type: ignore[assignment]
    socket.socket.listen = _test_traced_listen  # type: ignore[assignment]
    try:
        rc = pytest.main(["-q", "tests"])  # returns exit code
    finally:
        socket.socket.connect = _original_connect  # type: ignore[assignment]
        socket.socket.connect_ex = _original_connect_ex  # type: ignore[assignment]
        socket.socket.bind = _original_bind  # type: ignore[assignment]
        socket.socket.listen = _original_listen  # type: ignore[assignment]
        os.close(_tty_fd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
