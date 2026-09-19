"""Shared base class for retriever performance logging helpers."""

from __future__ import annotations

from typing import Any

from Helpers.PerfLogger import PerfLogger


class RetrieverBase:
    """Provide a shared PerfLogger and caller-name helper for retrievers."""

    def __init__(self, *, perf_component: str) -> None:
        self._perf_component = str(perf_component or self.__class__.__name__).strip()
        self.perf_logger: PerfLogger = PerfLogger()

    def _perf_log(
        self,
        method: str,
        detail: str,
        *,
        group: str = "retriever",
    ) -> None:
        """Write one performance event with a consistent retriever caller prefix."""
        logger_obj: Any = getattr(self, "perf_logger", None)
        if logger_obj is None or not hasattr(logger_obj, "log"):
            return

        component = str(
            getattr(self, "_perf_component", self.__class__.__name__)
            or self.__class__.__name__
        ).strip()
        method_name = str(method or "unknown").strip()
        caller = f"{component}.{method_name}" if method_name else component
        logger_obj.log(caller, group, detail)
