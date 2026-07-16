# Standard library imports
import logging
from datetime import datetime
from typing import \
    Any  # Assuming documents is a list of some custom document objects.

from Commons.SingletonMixin import SingletonMixin


class Globals(SingletonMixin):

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.documents: list[dict[str, Any]] = []
        self.failed_docs: list[dict[str, Any]] = []
        self.date: datetime = datetime.now()
        self.logger: logging.Logger | None = None

    def get_logger(self) -> logging.Logger | None:
        return self.logger

    # Document mutators
    def add_document(self, doc: dict[str, Any]) -> None:
        self.documents.append(doc)

    def add_failed_doc(self, doc: dict[str, Any]) -> None:
        self.failed_docs.append(doc)

    # Date accessors
    def get_date(self) -> str:
        # Format: YYYYMMDD_HHMMSS — safe for filenames
        return self.date.strftime("%Y%m%d_%H%M%S")

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger
