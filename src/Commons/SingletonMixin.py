"""
Reusable singleton mixin for RAG-LCC.

Usage
-----
Inherit from ``SingletonMixin`` and guard your ``__init__`` with the
``_initialized`` flag that is set to ``False`` on first creation::

    class MyService(SingletonMixin):
        def __init__(self, dep=None):
            if self._initialized:          # already constructed
                return
            self._initialized = True
            self.dep = dep or RealDep()

Thread safety
-------------
Instance creation is protected by a per-class lock so that two threads
calling ``MyService()`` at the same time cannot create two instances.

Testing
-------
Call ``MyService._reset()`` in test teardown to destroy the singleton and
allow a fresh instance on the next call::

    def teardown_method(self):
        MyService._reset()
"""

import threading
from typing import Any, Self


class SingletonMixin:
    """Mixin that makes any subclass a thread-safe singleton with test-reset support."""

    _instance: "SingletonMixin | None" = None
    _instance_lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def _reset(cls):
        """Destroy the singleton instance so the next call creates a fresh one.

        Intended for test teardown only.
        """
        with cls._instance_lock:
            cls._instance = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every subclass its own ``_instance`` and ``_instance_lock``.

        Without this, all subclasses would share the base-class
        ``_instance``/``_instance_lock`` class variables and only a single
        singleton could exist across the entire hierarchy.
        """
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._instance_lock = threading.Lock()
