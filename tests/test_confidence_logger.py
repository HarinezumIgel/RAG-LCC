from types import SimpleNamespace
from typing import Any

import pytest

from Helpers.ConfidenceLogger import ConfidenceLogger


@pytest.fixture(autouse=True)
def _reset_confidence_logger_singleton() -> None:
    ConfidenceLogger._reset()
    yield
    ConfidenceLogger._reset()


def test_log_step_coerces_non_list_event_storage() -> None:
    logger = ConfidenceLogger()
    session = SimpleNamespace(confidence_step_events="not-a-list")

    logger.log_step(
        session,
        step_name="Retrieve",
        status="executed",
        confidence_level="high",
        confidence_score=0.91,
        detail="ok",
    )

    events = session.confidence_step_events
    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["step"] == "Retrieve"
    assert events[0]["confidence_level"] == "HIGH"


def test_log_language_detection_records_fallback_status() -> None:
    logger = ConfidenceLogger()
    session = SimpleNamespace(lang_detection_events=None, confidence_step_events=[])

    logger.log_language_detection(
        session,
        {
            "stage": "query",
            "language": "de",
            "confidence": 0.32,
            "threshold": 0.65,
            "fell_back": True,
            "override_applied": False,
            "level": "LOW",
        },
    )

    assert len(session.lang_detection_events) == 1
    assert len(session.confidence_step_events) == 1
    assert session.confidence_step_events[0]["status"] == "fallback"
    assert session.confidence_step_events[0]["source"] == "langdetect"


def test_stringify_handles_dict_and_sequence_branches() -> None:
    logger = ConfidenceLogger()

    assert logger._stringify({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
    assert logger._stringify([1, "x", None]) == '[1, "x", null]'
    assert logger._stringify([{"x": 1}]) == "<list len=1>"


def test_flush_turn_csv_emits_component_and_step_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = ConfidenceLogger()
    logger.enabled = True
    logger.csv_enabled = True

    captured: dict[str, Any] = {}

    def _capture(path: str, row: dict[str, Any]) -> None:
        captured["path"] = path
        captured["row"] = row

    monkeypatch.setattr(logger, "_confidence_csv_path", lambda: "dummy_confidence.csv")
    monkeypatch.setattr(logger.csvWriter, "append_dynamic_row", _capture)

    session = SimpleNamespace(
        confidence_step_events=[
            {"step": "My Step", "confidence_level": "HIGH", "status": "executed"},
            {"step": "Skipped Step", "confidence_level": "LOW", "status": "skipped"},
            "invalid-entry",
        ],
        lang_detection_events=[{"stage": "query"}, "invalid-entry"],
        answer_confidence_components={"c_final": 0.87, 42: 0.5},
        query="q",
        user_query_original="uq",
        answer_confidence_level="HIGH",
        answer_confidence_score=0.87,
        answer_confidence_summary="summary",
        orchestration_step_confidence={"retrieve": "HIGH"},
        marked_documents=[1, 2],
        last_chosen_chunks=[1],
        chunk_texts_for_grounding=[],
    )

    logger.flush_turn_csv(session, outcome="ok", answer_text="hello")

    assert captured["path"] == "dummy_confidence.csv"
    row = captured["row"]
    assert row["confidence_steps_total"] == "2"
    assert row["lang_detection_events_total"] == "1"
    assert row["step_my_step"] == "HIGH"
    assert row["answer_confidence_c_final"] == "0.87"
    assert row["answer_confidence_42"] == "0.5"
    assert row["session_marked_documents"] == "<list len=2>"
