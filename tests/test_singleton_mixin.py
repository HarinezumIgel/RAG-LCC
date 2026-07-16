# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for SingletonMixin."""

import threading
import sys
import os
import pytest  # type: ignore[reportUnusedImport]

# Ensure src/ is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Commons.SingletonMixin import SingletonMixin

# ── Test helpers ──────────────────────────────────────────────────────────


class ServiceA(SingletonMixin):
    """Minimal singleton with init guard."""

    def __init__(self, value=42):
        if self._initialized:
            return
        self._initialized = True
        self.value = value


class ServiceB(SingletonMixin):
    """Second singleton — must be independent of ServiceA."""

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.data = []


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSingletonMixin:
    def teardown_method(self):
        ServiceA._reset()  # type: ignore[reportPrivateUsage]
        ServiceB._reset()  # type: ignore[reportPrivateUsage]

    # identity ---------------------------------------------------------------

    def test_same_instance_returned(self):
        a1 = ServiceA()
        a2 = ServiceA()
        assert a1 is a2

    def test_init_runs_only_once(self):
        a1 = ServiceA(value=10)
        a2 = ServiceA(value=99)  # should be ignored
        assert a1.value == 10
        assert a2.value == 10

    # independence -----------------------------------------------------------

    def test_subclasses_are_independent(self):
        a = ServiceA()
        b = ServiceB()
        assert a is not b
        assert type(a) is ServiceA
        assert type(b) is ServiceB

    # reset ------------------------------------------------------------------

    def test_reset_allows_fresh_instance(self):
        a1 = ServiceA(value=1)
        assert a1.value == 1

        ServiceA._reset()  # type: ignore[reportPrivateUsage]

        a2 = ServiceA(value=2)
        assert a2.value == 2
        assert a1 is not a2

    def test_reset_one_does_not_affect_other(self):
        a = ServiceA()
        b = ServiceB()

        ServiceA._reset()  # type: ignore[reportPrivateUsage]
        a2 = ServiceA()

        assert a2 is not a  # new A
        assert ServiceB() is b  # B unchanged

    # thread safety ----------------------------------------------------------

    def test_thread_safety(self):
        """Multiple threads calling the constructor get the same instance."""
        results = []
        barrier = threading.Barrier(8)

        def create():
            barrier.wait()
            results.append(ServiceA())

        threads = [threading.Thread(target=create) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)
