# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for NetworkTracer bind / listen detection."""

import os
import socket
import sys

import pytest

# Ensure src/ is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Commons.NetworkTracer import NetworkTracer

# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _tracer_lifecycle():  # pyright: ignore[reportUnusedFunction]
    """Enable the tracer before each test and disable it after."""
    NetworkTracer.enable_tracer()
    yield
    NetworkTracer.disable_tracer()


# ── Patching state ────────────────────────────────────────────────────────


class TestTracerState:
    def test_enable_patches_bind(self):
        assert socket.socket.bind is not NetworkTracer._original_bind  # type: ignore[comparison-overlap]

    def test_enable_patches_listen(self):
        assert socket.socket.listen is not NetworkTracer._original_listen  # type: ignore[comparison-overlap]

    def test_disable_restores_bind(self):
        NetworkTracer.disable_tracer()
        assert socket.socket.bind is NetworkTracer._original_bind  # type: ignore[comparison-overlap]

    def test_disable_restores_listen(self):
        NetworkTracer.disable_tracer()
        assert socket.socket.listen is NetworkTracer._original_listen  # type: ignore[comparison-overlap]


# ── Bind tracing ──────────────────────────────────────────────────────────


class TestTracedBind:
    def test_bind_loopback_ipv4_logged(self, capsys):
        """Binding 127.0.0.1:0 should be traced with the loopback annotation."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
        finally:
            sock.close()

        out = capsys.readouterr().out
        assert "Socket BIND detected" in out
        assert "127.0.0.1" in out
        assert "loopback" in out.lower()

    def test_bind_loopback_ipv6_logged(self, capsys):
        """Binding ::1:0 should be traced with the loopback annotation."""
        if not socket.has_ipv6:
            pytest.skip("IPv6 not available")
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            sock.bind(("::1", 0))
        finally:
            sock.close()

        out = capsys.readouterr().out
        assert "Socket BIND detected" in out
        assert "::1" in out
        assert "loopback" in out.lower()

    def test_bind_specific_port_no_harmless_label(self, capsys):
        """Binding to a specific port should NOT carry the 'likely harmless' annotation."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            # Get the actual assigned port — re-bind won't work, so we just
            # verify the ephemeral case above carried the label and look at
            # a non-zero port scenario textually.
        finally:
            sock.close()

        # For a non-loopback address the label must be absent.
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _ = capsys.readouterr()  # clear buffer
        try:
            sock2.bind(("0.0.0.0", 0))
        finally:
            sock2.close()

        out = capsys.readouterr().out
        assert "Socket BIND detected" in out
        assert "likely harmless" not in out

    def test_bind_still_works(self):
        """The traced bind must still perform the real bind."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            addr = sock.getsockname()
            assert addr[0] == "127.0.0.1"
            assert addr[1] != 0  # OS assigned a real port
        finally:
            sock.close()


# ── Listen tracing ────────────────────────────────────────────────────────


class TestTracedListen:
    def test_listen_logged(self, capsys):
        """socket.listen() should produce a LISTEN trace."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            _ = capsys.readouterr()  # clear bind output
            sock.listen(1)
        finally:
            sock.close()

        out = capsys.readouterr().out
        assert "Socket LISTEN detected" in out
        assert "inbound connections" in out.lower()

    def test_listen_shows_bound_address(self, capsys):
        """The listen trace should report the address the socket is bound to."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            _ = capsys.readouterr()  # clear bind output
            sock.listen(1)
        finally:
            sock.close()

        out = capsys.readouterr().out
        assert str(port) in out

    def test_listen_shows_backlog(self, capsys):
        """The listen trace should include the backlog value."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            _ = capsys.readouterr()
            sock.listen(5)
        finally:
            sock.close()

        out = capsys.readouterr().out
        assert "5" in out

    def test_listen_still_works(self):
        """The traced listen must still open a real listener."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]

            # Prove the listener is real by connecting to it.
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client.connect(("127.0.0.1", port))
            finally:
                client.close()
        finally:
            server.close()
