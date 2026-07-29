"""Lightweight performance event logger.

Writes timestamped start/stop (and other) events to ``logs/Performance/<AppName>_Performance_<YYYYMMDD_HHMMSS>.log``
so that wall-clock cost of every major pipeline stage can be compared across
runs without changing the main application logging.

Usage::

    from Helpers.PerfLogger import PerfLogger

    # In app __init__, after _FRIENDLY_NAME is set in Config:
    self.perf_logger = PerfLogger()

    # Later, wherever timing is needed:
    self.perf_logger.log("BM25Retriever.query", "bm25", "start bm25 query q='cats'")
    # ... do work ...
    self.perf_logger.log("BM25Retriever.query", "bm25", f"stop  bm25 query n={len(results)}")

Log format (one line per event)::

    2026-07-13T14:22:05.123Z | 1.234s | bm25   | BM25Retriever.query              | start bm25 query q='cats'

Enable/disable via ``PERFORMANCE_LOGGING`` in ``Configuration/Config_Global.py``.
The log file is created (with its directory) on first write.
The filename is derived from ``_FRIENDLY_NAME`` and a startup timestamp so each
application run (RAGChat, RAGChatService, …) produces its own dated file.
Instantiate PerfLogger only after ``_FRIENDLY_NAME`` has been set in Config so
the correct filename is used from the start.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from Commons.SingletonMixin import SingletonMixin

_LOG_DIR: str = os.path.join("logs", "Performance")
_CALLER_WIDTH: int = 45
_GROUP_WIDTH: int = 12


def _build_log_filename() -> str:
    """Return a dated log filename derived from the active app's friendly name.

    Pattern: ``<_FRIENDLY_NAME>_Performance_<YYYYMMDD_HHMMSS>.log``
    Falls back to the script stem from ``sys.argv[0]`` when ``_FRIENDLY_NAME``
    is absent (should not happen if PerfLogger is instantiated after Config
    is fully initialised).
    """
    try:
        from Config.Config import Config  # lazy — avoids circular imports

        friendly = Config().get_str("_FRIENDLY_NAME", silent=True) or ""
    except Exception:
        friendly = ""

    if not friendly:
        argv0 = sys.argv[0] if sys.argv else ""
        friendly = os.path.splitext(os.path.basename(argv0))[0] or "Unknown"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{friendly}_Performance_{stamp}.log"


class PerfLogger(SingletonMixin):
    """Singleton performance logger.

    Call :py:meth:`log` to emit a timestamped start/stop event.
    All output is gated by ``PERFORMANCE_LOGGING`` in Config_Global.py.
    """

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._file_logger: logging.Logger | None = None
        self._start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_file_logger(self) -> logging.Logger:
        """Return the file logger, creating it on first call."""
        if self._file_logger is None:
            log_path: str = os.path.join(_LOG_DIR, _build_log_filename())
            os.makedirs(_LOG_DIR, exist_ok=True)

            logger: logging.Logger = logging.getLogger("PerfLogger")
            logger.propagate = False
            logger.setLevel(logging.DEBUG)

            if not logger.handlers:
                fh: logging.FileHandler = logging.FileHandler(
                    log_path, encoding="utf-8"
                )
                fh.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(fh)

            self._file_logger = logger

        return self._file_logger

    @staticmethod
    def _is_enabled() -> bool:
        """Return True when PERFORMANCE_LOGGING is truthy in config."""
        try:
            from Config.Config import \
                Config  # lazy — avoids circular imports at module load

            return bool(Config().get_bool("PERFORMANCE_LOGGING"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, caller: str, group: Optional[str], detail: str) -> None:
        """Emit a timestamped performance event to ``logs/Performance/<AppName>_Performance_<stamp>.log``.

        No-op when ``PERFORMANCE_LOGGING = False`` in Config_Global.py.

        Parameters
        ----------
        caller:
            Module and function name, e.g. ``"BM25Retriever.query"``.
            Padded to a fixed width so columns align across callers.
        group:
            Logical group for the event, e.g. ``"bm25"``, ``"cache"``, ``"scorer"``.
            Padded to a fixed width. Use ``None`` or ``"-"`` for ungrouped events.
        detail:
            Free-form description, e.g.
            ``"start bm25 query q='cats'"`` or
            ``"stop  bm25 query n=42 elapsed=0.123s"``.
        """
        if not self._is_enabled():
            return

        now: float = time.perf_counter()
        elapsed_col: str = ""

        if detail.startswith("start"):
            self._start_times[caller] = now
        elif detail.startswith("stop"):
            t0 = self._start_times.pop(caller, None)
            if t0 is not None:
                elapsed_col = f"\u0394={now - t0:.3f}s"

        ts: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        group_str: str = group if group is not None else "-"
        self._get_file_logger().info(
            "%s | %-10s | %-12s | %-*s | %s",
            ts,
            elapsed_col,
            group_str,
            _CALLER_WIDTH,
            caller,
            detail,
        )
