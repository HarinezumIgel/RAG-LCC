from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceAlgoResult:
    algo: Optional[str]
    phrase: str
    score: Optional[float]
    threshold: Optional[float]
    detail: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ScorerBase(ABC):
    @abstractmethod
    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        """Return detection results as a list of ComplianceAlgoResult."""
        raise NotImplementedError


@dataclass
class InternalResult(ComplianceAlgoResult):
    matched_algos_count: Optional[int] = None

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
    algo: Optional[str]
    phrase: str
    score: Optional[float]
    score_str: str
    threshold: Optional[float]
    detail: Optional[str]
    matched_algos_count: Optional[int]
    algos_matched: Optional[str] = None


@dataclass
class PhraseMeta:
    depth_count: int
    depth_req: int
    breadth_count: int
    breadth_req: int

    def plain(self) -> str:
        return f"{self.depth_count}/{self.depth_req}  {self.breadth_count}/{self.breadth_req}"
