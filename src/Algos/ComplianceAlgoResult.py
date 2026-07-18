import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from Helpers.PerfLogger import PerfLogger


@dataclass
class ComplianceAlgoResult:
    algo: str | None
    phrase: str
    score: float | None
    threshold: float | None
    detail: str | None = None
    meta: Dict[str, Any] | None = None


class ScorerBase(ABC):
    @abstractmethod
    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        """Return detection results as a list of ComplianceAlgoResult."""
        raise NotImplementedError

    @abstractmethod
    def _verify_impl(self, *args: Any, **kwargs: Any) -> List[ComplianceAlgoResult]:
        """Scorer logic — implement in each subclass."""
        raise NotImplementedError

    def __init__(self) -> None:
        self.perf_logger: PerfLogger = PerfLogger()

    def verify(self, *args: Any, **kwargs: Any) -> List[ComplianceAlgoResult]:
        """Timing wrapper: emits perf start/stop events around _verify_impl()."""
        caller = f"{type(self).__name__}.verify"
        self.perf_logger.log(caller, "start verify")
        t0 = time.perf_counter()
        result = self._verify_impl(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self.perf_logger.log(
            caller, f"stop  verify n={len(result)} elapsed={elapsed:.3f}s"
        )
        return result


@dataclass
class InternalResult(ComplianceAlgoResult):
    matched_algos_count: int | None = None

    @classmethod
    def from_base(cls, base: ComplianceAlgoResult, **kwargs: Any) -> "InternalResult":
        return cls(
            algo=base.algo,
            phrase=base.phrase,
            score=base.score,
            threshold=base.threshold,
            detail=base.detail,
            **kwargs,
        )


@dataclass
class ResultsForPrint:
    algo: str | None
    phrase: str
    score: float | None
    score_str: str
    threshold: float | None
    detail: str | None
    matched_algos_count: int | None
    algos_matched: str | None = None


@dataclass
class PhraseMeta:
    depth_count: int
    depth_req: int
    breadth_count: int
    breadth_req: int

    def plain(self) -> str:
        return f"{self.depth_count}/{self.depth_req}  {self.breadth_count}/{self.breadth_req}"
