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

        self._documents: list[dict[str, Any]] = []
        self._failed_docs: list[dict[str, Any]] = []
        self._date: datetime = datetime.now()
        self._logger: logging.Logger | None = None

    def get_logger(self) -> logging.Logger | None:
        return self._logger

    # Document mutators
    def add_document(self, doc: dict[str, Any]) -> None:
        self._documents.append(doc)

    def add_failed_doc(self, doc: dict[str, Any]) -> None:
        self._failed_docs.append(doc)

    # Date accessors
    def get_date(self) -> str:
        # Format: YYYYMMDD_HHMMSS — safe for filenames
        return self._date.strftime("%Y%m%d_%H%M%S")

    def set_logger(self, logger: logging.Logger) -> None:
        self._logger = logger
