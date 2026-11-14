from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import build_csv_path

if TYPE_CHECKING:
    from Chat.ConfidenceEvaluator import AnswerConfidenceResult


class ConfidenceLogger(SingletonMixin):
    """Record per-step confidence signals and persist turn-level CSV rows."""

    def __init__(
        self,
        *,
        cfg: Config | None = None,
        pretty: PrettyWriter | None = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.globals: Globals = Globals()
        self.csvWriter: CSVWriter = CSVWriter(cfg=self.cfg, pretty=self.pretty)

        settings = self.cfg.get_dict("_CONFIDENCE_LOGGING", {}) or {}
        self.enabled: bool = bool(settings.get("enabled", True))
        self.emit_pretty: bool = bool(settings.get("emit_pretty", True))
        self.csv_enabled: bool = bool(settings.get("csv_enabled", True))

    @staticmethod
    def _sanitize_step_name(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "").strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "unknown"

    @staticmethod
    def _stringify(value: Any, *, limit: int = 512) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, str)):
            text = str(value)
            return text if len(text) <= limit else text[: limit - 3] + "..."
        if isinstance(value, bytes):
            return f"<bytes len={len(value)}>"
        if isinstance(value, dict):
            mapping = cast(dict[Any, Any], value)
            try:
                text = json.dumps(mapping, ensure_ascii=True, sort_keys=True)
            except Exception:
                text = str(mapping)
            return text if len(text) <= limit else text[: limit - 3] + "..."
        if isinstance(value, (list, tuple, set)):
            seq_like = cast(list[Any] | tuple[Any, ...] | set[Any], value)
            seq: list[Any] = list(seq_like)
            if not seq:
                return "[]"
            primitive = all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in seq
            )
            if primitive and len(seq) <= 20:
                try:
                    text = json.dumps(seq, ensure_ascii=True)
                except Exception:
                    text = str(seq)
                return text if len(text) <= limit else text[: limit - 3] + "..."
            return f"<{type(seq_like).__name__} len={len(seq)}>"
        text = str(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def _coerce_event_list(value: Any) -> list[dict[str, Any]]:
        """Return a list of event dicts, discarding incompatible entries."""
        if not isinstance(value, list):
            return []
        raw_items = cast(list[Any], value)
        coerced: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict):
                item_map = cast(dict[Any, Any], item)
                coerced.append({str(key): val for key, val in item_map.items()})
        return coerced

    def reset_turn(self, session: Any) -> None:
        if not self.enabled:
            return
        session.confidence_step_events = []
        session.lang_detection_events = []

    def log_step(
        self,
        session: Any,
        *,
        step_name: str,
        status: str,
        confidence_level: str,
        confidence_score: float | None = None,
        detail: str = "",
        source: str = "pipeline",
        quality_dimensions: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "step": step_name,
            "status": str(status or "").strip().lower() or "unknown",
            "confidence_level": str(confidence_level or "").strip().upper()
            or "UNKNOWN",
            "confidence_score": confidence_score,
            "detail": str(detail or "").strip(),
        }

        # Add quality dimensions if provided
        if quality_dimensions:
            event["quality_dimensions"] = quality_dimensions

        events = self._coerce_event_list(
            getattr(session, "confidence_step_events", None)
        )
        events.append(event)
        session.confidence_step_events = events

    def log_language_detection(self, session: Any, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        lang_events = self._coerce_event_list(
            getattr(session, "lang_detection_events", None)
        )
        lang_events.append(dict(event))
        session.lang_detection_events = lang_events

        stage = str(event.get("stage", "language_detection") or "language_detection")
        language = str(event.get("language", "") or "")
        confidence = event.get("confidence")
        threshold = event.get("threshold")
        fell_back = bool(event.get("fell_back", False))
        override_applied = bool(event.get("override_applied", False))
        level = str(event.get("level", "LOW") or "LOW").upper()

        status = "executed"
        if fell_back and not override_applied:
            status = "fallback"
        elif fell_back and override_applied:
            status = "fallback_override"

        detail = (
            f"lang={language} conf={self._stringify(confidence)} "
            f"threshold={self._stringify(threshold)}"
        )
        self.log_step(
            session,
            step_name=f"LangDetect/{stage}",
            status=status,
            confidence_level=level,
            confidence_score=(
                float(confidence) if isinstance(confidence, (int, float)) else None
            ),
            detail=detail,
            source="langdetect",
        )

    def _confidence_csv_path(self) -> str:
        log_dir = self.cfg.get_str("_LOG_DIRECTORY", "")
        if not log_dir:
            log_dir = "logs/RAGChat"
        friendly_name = self.cfg.get_str("_FRIENDLY_NAME", "RAGChat")
        stamp = self.globals.get_date()
        return build_csv_path(friendly_name, "CONFIDENCE_FLAT", stamp, log_dir)

    def _flatten_columns(self, prefix: str, value: Any) -> dict[str, str]:
        """Expand nested values into scalar columns without truncating results."""
        columns: dict[str, str] = {}
        if isinstance(value, dict):
            mapping = cast(dict[Any, Any], value)
            for key, item in mapping.items():
                segment = str(key).replace("~", "~0").replace(".", "~1")
                columns.update(self._flatten_columns(f"{prefix}.{segment}", item))
        elif isinstance(value, (list, tuple)):
            sequence = cast(list[Any] | tuple[Any, ...], value)
            for index, item in enumerate(sequence, start=1):
                columns.update(self._flatten_columns(f"{prefix}.{index}", item))
        else:
            columns[prefix] = (
                ""
                if value is None
                else str(value).lower() if isinstance(value, bool) else str(value)
            )
        return columns

    def _step_columns(self, events: list[dict[str, Any]]) -> dict[str, str]:
        columns: dict[str, str] = {}
        occurrences: dict[str, int] = {}
        for event in events:
            name = str(event.get("step", ""))
            occurrences[name] = occurrences.get(name, 0) + 1
            segment = name.replace("~", "~0").replace(".", "~1")
            prefix = f"step.{segment}.{occurrences[name]}"
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"step", "quality_dimensions"}
            }
            if (
                name in {"retrieval evidence confidence", "final answer confidence"}
                or event.get("source") == "langdetect"
            ):
                fields.pop("detail", None)
            columns.update(self._flatten_columns(prefix, fields))
        return columns

    def _session_snapshot_columns(self, session: Any) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        excluded = {
            "cfg",
            "pretty",
            "answer_confidence_result",
            "answer_confidence_components",
            "answer_confidence_level",
            "answer_confidence_score",
            "answer_confidence_summary",
            "quality_dimension_scores",
            "quality_dimension_subsignals",
            "quality_dimension_mode",
            "quality_dimension_error",
            "quality_dimension_judge_response",
            "confidence_step_events",
            "lang_detection_events",
            "orchestration_step_confidence",
            "query",
            "user_query_original",
            "last_answer_content",
        }
        for key in sorted(getattr(session, "__dict__", {}).keys()):
            if key in excluded:
                continue
            value = getattr(session, key, None)
            if key in {
                "marked_documents",
                "last_chosen_chunks",
                "chunk_texts_for_grounding",
            }:
                if isinstance(value, (list, tuple, set)):
                    seq_value = cast(list[Any] | tuple[Any, ...] | set[Any], value)
                    snapshot[f"session_{key}_count"] = str(len(seq_value))
                else:
                    snapshot.update(self._flatten_columns(f"session_{key}", value))
                continue
            snapshot.update(self._flatten_columns(f"session_{key}", value))
        return snapshot

    def flush_turn_csv(
        self,
        session: Any,
        *,
        outcome: str,
        answer_text: str = "",
    ) -> None:
        if not self.enabled or not self.csv_enabled:
            return

        try:
            confidence_steps = self._coerce_event_list(
                getattr(session, "confidence_step_events", [])
            )
            lang_events = self._coerce_event_list(
                getattr(session, "lang_detection_events", [])
            )
            confidence: AnswerConfidenceResult | None = getattr(
                session, "answer_confidence_result", None
            )
            components: dict[str, float] = (
                confidence.components if confidence is not None else {}
            )

            # Get quality dimension data
            quality_scores: dict[str, dict[str, float]] = (
                confidence.quality_dimensions if confidence is not None else {}
            )
            quality_subsignals: dict[str, float] = (
                confidence.quality_subsignals if confidence is not None else {}
            )
            quality_mode = (
                confidence.quality_mode if confidence is not None else "heuristic_only"
            )
            quality_error = confidence.quality_error if confidence is not None else None

            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "friendly_name": self.cfg.get_str("_FRIENDLY_NAME", "RAGChat"),
                "outcome": str(outcome or "unknown"),
                "query": getattr(session, "query", ""),
                "user_query_original": getattr(session, "user_query_original", ""),
                "answer_chars": len(answer_text or ""),
                "answer_confidence_level": self._stringify(
                    confidence.level if confidence is not None else ""
                ),
                "answer_confidence_score": self._stringify(
                    confidence.score if confidence is not None else ""
                ),
                "confidence_steps_total": len(confidence_steps),
                "confidence_steps_skipped": sum(
                    1
                    for event in confidence_steps
                    if str(event.get("status", "")).lower()
                    in {"skipped", "not_executed", "fallback"}
                ),
                "lang_detection_events_total": len(lang_events),
                # Quality dimension columns
                "quality_mode": quality_mode,
                "quality_error": quality_error,
                "quality_judge_model": getattr(confidence, "judge_model", None),
            }

            # Add individual answer confidence components to row
            for component_key, component_value in components.items():
                if component_key == "c_final":
                    continue  # Already exported as answer_confidence_score.
                row[f"answer_confidence_{component_key}"] = self._stringify(
                    component_value
                )

            # Add quality dimension scores to row
            for dimension, scores in quality_scores.items():
                for scorer in ("heuristic", "llm"):
                    if scorer in scores:
                        row[f"quality_{dimension}_{scorer}"] = scores[scorer]

            # Add quality subsignals to row
            for signal, value in quality_subsignals.items():
                row[f"quality_retrieval_{signal}"] = value

            row.update(self._step_columns(confidence_steps))
            row.update(self._flatten_columns("language_detection", lang_events))
            row.update(
                self._flatten_columns(
                    "orchestration",
                    getattr(session, "orchestration_step_confidence", {}),
                )
            )
            row.update(self._session_snapshot_columns(session))
            row = {
                key: "" if value is None else str(value) for key, value in row.items()
            }

            path = self._confidence_csv_path()
            self.csvWriter.append_dynamic_row(path, row)
        except Exception as exc:
            self.pretty.write(
                "W",
                "Confidence",
                f"Failed writing confidence CSV: {exc}",
                color=YELLOW,
            )
