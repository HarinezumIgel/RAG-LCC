from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, cast

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import build_csv_path


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
        return build_csv_path(friendly_name, "CONFIDENCE", stamp, log_dir)

    def _session_snapshot_columns(self, session: Any) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for key in sorted(getattr(session, "__dict__", {}).keys()):
            value = getattr(session, key, None)
            if key in {
                "marked_documents",
                "last_chosen_chunks",
                "chunk_texts_for_grounding",
            }:
                if isinstance(value, (list, tuple, set)):
                    seq_value = cast(list[Any] | tuple[Any, ...] | set[Any], value)
                    snapshot[f"session_{key}"] = (
                        f"<{type(seq_value).__name__} len={len(seq_value)}>"
                    )
                else:
                    snapshot[f"session_{key}"] = self._stringify(value)
                continue
            snapshot[f"session_{key}"] = self._stringify(value)
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
            components_obj = getattr(session, "answer_confidence_components", {})
            components: dict[str, Any] = {}
            if isinstance(components_obj, dict):
                component_map = cast(dict[Any, Any], components_obj)
                components = {
                    str(component_key): component_value
                    for component_key, component_value in component_map.items()
                }

            step_latest: dict[str, str] = {}
            for event in confidence_steps:
                name = self._sanitize_step_name(str(event.get("step", "")))
                level = str(event.get("confidence_level", "UNKNOWN") or "UNKNOWN")
                step_latest[f"step_{name}"] = level

            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "friendly_name": self.cfg.get_str("_FRIENDLY_NAME", "RAGChat"),
                "outcome": str(outcome or "unknown"),
                "query": self._stringify(getattr(session, "query", "")),
                "user_query_original": self._stringify(
                    getattr(session, "user_query_original", "")
                ),
                "answer_chars": len(answer_text or ""),
                "answer_confidence_level": self._stringify(
                    getattr(session, "answer_confidence_level", "")
                ),
                "answer_confidence_score": self._stringify(
                    getattr(session, "answer_confidence_score", "")
                ),
                "answer_confidence_summary": self._stringify(
                    getattr(session, "answer_confidence_summary", "")
                ),
                "answer_confidence_components_json": self._stringify(
                    components, limit=4000
                ),
                "confidence_steps_total": len(confidence_steps),
                "confidence_steps_skipped": sum(
                    1
                    for event in confidence_steps
                    if str(event.get("status", "")).lower()
                    in {"skipped", "not_executed", "fallback"}
                ),
                "lang_detection_events_total": len(lang_events),
                "confidence_steps_json": self._stringify(confidence_steps, limit=4000),
                "lang_detection_json": self._stringify(lang_events, limit=4000),
                "orchestration_step_confidence": self._stringify(
                    getattr(session, "orchestration_step_confidence", {})
                ),
            }
            for key, value in components.items():
                safe_key = self._sanitize_step_name(str(key))
                row[f"answer_confidence_{safe_key}"] = value

            row.update(step_latest)
            row.update(self._session_snapshot_columns(session))
            row = {
                key: self._stringify(value, limit=4000) for key, value in row.items()
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
