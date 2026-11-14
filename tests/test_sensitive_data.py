import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Gui.PrettyWriter import PrettyWriter
from Helpers.SensitiveData import mask_sensitive_data


def test_masks_credentials_in_urls_and_text() -> None:
    text = (
        "Connecting to https://user:pass@example.test/v1"
        "?access_token=abc123&mode=fast "
        "Authorization: Bearer secret-token api_key='another-secret'"
        ' --api-key command-line-secret {"api_key": "json-secret"} '
        "https://example.test/?key=query-secret"
    )

    masked = mask_sensitive_data(text)

    assert "user" not in masked
    assert "pass@" not in masked
    assert "abc123" not in masked
    assert "secret-token" not in masked
    assert "another-secret" not in masked
    assert "command-line-secret" not in masked
    assert "json-secret" not in masked
    assert "query-secret" not in masked
    assert "mode=fast" in masked
    assert "******" in masked


def test_pretty_writer_masks_terminal_and_logger_output(capsys) -> None:
    logger = logging.getLogger("test_pretty_writer_masks_sensitive_data")
    logger.handlers.clear()
    logger.propagate = False
    records: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger.addHandler(CaptureHandler())
    try:
        PrettyWriter(logger=logger, always_on=True).write(
            "ERROR",
            "Endpoint",
            "https://user:password@example.test/?api_key=private-value",
            max_line_length=200,
        )
    finally:
        logger.handlers.clear()

    terminal_output = capsys.readouterr().out
    assert "password@" not in terminal_output
    assert "private-value" not in terminal_output
    assert "password@" not in records[0]
    assert "private-value" not in records[0]
