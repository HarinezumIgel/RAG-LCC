from typing import Any, Dict, Optional, Protocol

from Strategies.StrategyType import StrategyType


class ProcessingStrategy(Protocol):
    strategy_type: StrategyType  # <-- declare here

    def process(self, doc: Optional[Dict[str, Any]]) -> None:
        """Processes a document dictionary."""
        ...
