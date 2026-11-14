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


@pytest.mark.parametrize("scorer", ["both", "heuristic", "llm"])
def test_flush_turn_csv_emits_component_and_step_columns(
    monkeypatch: pytest.MonkeyPatch,
    scorer: str,
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
        answer_confidence_result=SimpleNamespace(
            components={"c_final": 0.87, 42: 0.5},
            level="HIGH",
            score=0.87,
            summary="summary",
            quality_dimensions={
                "evidence_relevance": {"heuristic": 0.8},
                "internal_consistency": {"llm": 0.9},
            },
            quality_subsignals={"relevance_support": 0.85},
            quality_mode="both",
            quality_error=None,
        ),
        query="q",
        user_query_original="uq",
        orchestration_step_confidence={"retrieve": "HIGH"},
        marked_documents=[1, 2],
        last_chosen_chunks=[1],
        chunk_texts_for_grounding=[],
    )
    if scorer == "heuristic":
        del session.answer_confidence_result.quality_dimensions["internal_consistency"]
    elif scorer == "llm":
        del session.answer_confidence_result.quality_dimensions["evidence_relevance"]

    logger.flush_turn_csv(session, outcome="ok", answer_text="hello")

    assert captured["path"] == "dummy_confidence.csv"
    row = captured["row"]
    assert row["confidence_steps_total"] == "2"
    assert row["lang_detection_events_total"] == "1"
    assert row["step.My Step.1.confidence_level"] == "HIGH"
    assert "answer_confidence_c_final" not in row
    assert row["answer_confidence_42"] == "0.5"
    assert row["session_marked_documents_count"] == "2"
    assert row["answer_confidence_level"] == "HIGH"
    assert row["answer_confidence_score"] == "0.87"
    assert "answer_confidence_summary" not in row
    assert row["quality_mode"] == "both"
    assert row.get("quality_internal_consistency_llm") == (
        None if scorer == "heuristic" else "0.9"
    )
    assert row.get("quality_evidence_relevance_heuristic") == (
        None if scorer == "llm" else "0.8"
    )
    assert row["quality_retrieval_relevance_support"] == "0.85"
    assert not any(key.endswith("_avg") for key in row)
    assert not any(key.startswith("quality_accuracy") for key in row)
    assert "session_query" not in row
    assert "session_answer_confidence_result" not in row
    assert not any(key.endswith("_json") for key in row)


def test_flatten_columns_preserves_nested_values_long_text_and_key_paths() -> None:
    logger = ConfidenceLogger()
    text = "x" * 5000
    columns = logger._flatten_columns(
        "options",
        {
            "nested": {"enabled": True, "unset": None},
            "list": [{"score": 0.9}, "other"],
            "a.b": text,
            "a": {"b": "different"},
        },
    )

    assert columns == {
        "options.nested.enabled": "true",
        "options.nested.unset": "",
        "options.list.1.score": "0.9",
        "options.list.2": "other",
        "options.a~1b": text,
        "options.a.b": "different",
    }


def test_repeated_steps_are_not_overwritten_and_summaries_not_duplicated() -> None:
    logger = ConfidenceLogger()
    columns = logger._step_columns(
        [
            {"step": "Rerank", "status": "executed", "confidence_score": 0.2},
            {"step": "Rerank", "status": "fallback", "confidence_score": 0.1},
            {
                "step": "final answer confidence",
                "source": "answer",
                "detail": "duplicated rendered summary",
                "quality_dimensions": {"accuracy": {"llm": 1.0}},
            },
        ]
    )

    assert columns["step.Rerank.1.confidence_score"] == "0.2"
    assert columns["step.Rerank.2.confidence_score"] == "0.1"
    assert columns["step.Rerank.2.status"] == "fallback"
    assert not any("detail" in key or "quality_dimensions" in key for key in columns)


def test_flat_csv_round_trip_preserves_events_and_nested_settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv

    logger = ConfidenceLogger()
    logger.enabled = True
    logger.csv_enabled = True
    path = tmp_path / "confidence_flat.csv"
    monkeypatch.setattr(logger, "_confidence_csv_path", lambda: str(path))
    session = SimpleNamespace(
        query="q" * 600,
        confidence_step_events=[
            {"step": "Retrieve", "status": "executed", "detail": "x" * 5000},
            {"step": "Retrieve", "status": "fallback"},
        ],
        lang_detection_events=[
            {"stage": "query", "language": "english", "confidence": 0.9}
        ],
        extraOllamaOptions={"temperature": 0.2},
        orchestration_step_confidence={"Retrieve": "HIGH"},
    )
    logger.flush_turn_csv(session, outcome="ok")
    session.extraOllamaOptions["seed"] = 42
    logger.flush_turn_csv(session, outcome="ok")

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=logger.csvWriter.csv_delimiter))
    assert len(rows) == 2
    assert rows[0]["query"] == "q" * 600
    assert rows[0]["step.Retrieve.1.detail"] == "x" * 5000
    assert rows[0]["step.Retrieve.2.status"] == "fallback"
    assert rows[0]["language_detection.1.language"] == "english"
    assert rows[0]["session_extraOllamaOptions.temperature"] == "0.2"
    assert rows[0]["session_extraOllamaOptions.seed"] == ""
    assert rows[1]["session_extraOllamaOptions.seed"] == "42"
    assert rows[0]["orchestration.Retrieve"] == "HIGH"
    assert not any(value.startswith(("{", "[", "<list")) for value in rows[0].values())
