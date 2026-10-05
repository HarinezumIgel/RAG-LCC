from __future__ import annotations

import math
from typing import Any, cast

from Globals.Session import Session
from Gui.Colors import GREEN, ORANGE, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.ConfidenceLogger import ConfidenceLogger


class ConfidenceHelper:
    """Compute and persist answer confidence from selected evidence chunks."""

    def __init__(
        self,
        *,
        confidence_logger: ConfidenceLogger | None = None,
        pretty: PrettyWriter | None = None,
    ) -> None:
        self.confidence_logger: ConfidenceLogger = (
            confidence_logger or ConfidenceLogger()
        )
        self.pretty: PrettyWriter = pretty or PrettyWriter(always_on=True)

    @staticmethod
    def _confidence_color_from_level(level: str) -> str:
        """Return terminal color for the confidence level label."""
        normalized = str(level or "").strip().upper()
        if normalized == "HIGH":
            return GREEN
        if normalized == "MEDIUM":
            return YELLOW
        return ORANGE

    @staticmethod
    def _confidence_icon_from_level(level: str) -> str:
        """Return a level icon that is safe for CLI and service payloads."""
        normalized = str(level or "").strip().upper()
        if normalized in {"HIGH", "MEDIUM", "LOW"}:
            return Symbols.sym_icon(normalized)
        return Symbols.sym_icon("INFO")

    def _emit_selection_confidence(
        self,
        *,
        level: str,
        score: float,
        evidence_chunks: int,
        fallback_triggered: bool,
    ) -> None:
        """Emit a user-facing confidence line for the final chunk-selection step."""
        if not self.confidence_logger.emit_pretty:
            return
        icon = self._confidence_icon_from_level(level)
        self.pretty.write(
            "I",
            "Selection Confidence",
            (
                f"{icon}{level} C_final={score:.2f} "
                f"evidence_chunks={evidence_chunks} "
                f"rerank_fallback={'on' if fallback_triggered else 'off'}"
            ),
            color=self._confidence_color_from_level(level),
        )

    @staticmethod
    def _clamp_probability(value: float) -> float:
        """Clamp numeric confidence values into the [0.0, 1.0] interval."""
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def _confidence_level_from_score(score: float) -> str:
        """Map a 0..1 confidence score to a user-facing level label."""
        if score >= 0.78:
            return "HIGH"
        if score >= 0.56:
            return "MEDIUM"
        return "LOW"

    @classmethod
    def _doc_confidence_score(cls, doc: Any) -> float:
        """Estimate one chunk confidence on a normalized 0..1 scale."""
        metadata_obj = getattr(doc, "metadata", None)
        metadata: dict[str, Any] = (
            cast(dict[str, Any], metadata_obj) if isinstance(metadata_obj, dict) else {}
        )

        raw_logit = metadata.get("raw_rerank_score")
        if raw_logit is not None:
            try:
                return cls._clamp_probability(1.0 / (1.0 + math.exp(-float(raw_logit))))
            except (TypeError, ValueError, OverflowError):
                pass

        for key in ("rerank_score", "chroma_score"):
            candidate = metadata.get(key)
            if candidate is None:
                continue
            try:
                return cls._clamp_probability(float(candidate))
            except (TypeError, ValueError):
                continue

        # Unknown score scales (e.g. raw BM25 / RRF only): keep neutral instead
        # of forcing an artificially low confidence.
        for key in ("rrf_score", "bm25_score", "graph_score", "regex_score"):
            if metadata.get(key) is not None:
                return 0.5

        return 0.0

    @classmethod
    def build_confidence_summary(
        cls,
        *,
        level: str,
        score: float,
        c_top: float,
        c_mean_top3: float,
        c_coverage: float,
        c_local: float,
        c_fallback_penalty: float,
        top_score: float,
        mean_top3: float,
        coverage: float,
        local_share: float,
        evidence_chunks: int,
        fallback_triggered: bool,
        note: str = "",
    ) -> str:
        """Return a compact multi-line confidence payload with metric meaning."""
        normalized_level = str(level or "LOW").strip().upper()
        icon = cls._confidence_icon_from_level(normalized_level)
        lines: list[str] = [f"{icon}{normalized_level}"]

        metrics: list[tuple[str, str]] = [
            (
                f"C_final={score:.2f}",
                "final confidence in [0,1] after penalties.",
            ),
            (
                f"C_top={c_top:.2f}",
                "weighted best-chunk signal (0.50 x top_score).",
            ),
            (
                f"C_mean_top3={c_mean_top3:.2f}",
                "weighted top-3 stability (0.30 x mean_top3).",
            ),
            (
                f"C_coverage={c_coverage:.2f}",
                "weighted evidence coverage (0.15 x coverage).",
            ),
            (
                f"C_local={c_local:.2f}",
                "weighted local-source share (0.05 x local_share).",
            ),
            (
                f"C_fallback_penalty={c_fallback_penalty:.2f}",
                "subtraction when low-confidence rerank fallback was used.",
            ),
            (
                f"top_score={top_score:.2f}",
                "strongest selected chunk confidence.",
            ),
            (
                f"mean_top3={mean_top3:.2f}",
                "average confidence of the top 3 selected chunks.",
            ),
            (
                f"coverage={coverage:.2f}",
                "selected_chunks / final_chunks_to_llm, capped at 1.00.",
            ),
            (
                f"local_share={local_share:.2f}",
                "fraction of selected chunks from local files.",
            ),
            (
                f"evidence_chunks={evidence_chunks}",
                "number of selected evidence chunks.",
            ),
            (
                f"rerank_fallback={'on' if fallback_triggered else 'off'}",
                "whether fallback rerank logic was triggered.",
            ),
        ]

        rows: list[tuple[str, str]] = []
        if note:
            rows.append(("Note", note))
        rows.extend(metrics)

        # Keep at least one visible blank before ':' even for the longest key.
        pad = max(len(left) for left, _ in rows) + 1
        # Use non-breaking spaces for padding so alignment is preserved in
        # renderers that collapse normal spaces.
        pad_char = "\u00a0"
        lines.extend(
            [f"- {left.ljust(pad, pad_char)}: {right}" for left, right in rows]
        )
        return "\n".join(lines)

    def store_answer_confidence(self, session: Session, chosen: list[Any]) -> None:
        """Compute and persist final answer confidence from selected evidence."""
        if not chosen:
            components: dict[str, float] = {
                "c_final": 0.0,
                "c_top": 0.0,
                "c_mean_top3": 0.0,
                "c_coverage": 0.0,
                "c_local": 0.0,
                "c_fallback_penalty": 0.0,
                "c_raw_sum": 0.0,
                "top_score": 0.0,
                "mean_top3": 0.0,
                "coverage": 0.0,
                "local_share": 0.0,
                "evidence_chunks": 0.0,
            }
            session.answer_confidence_components = components
            session.answer_confidence_score = 0.0
            session.answer_confidence_level = "LOW"
            session.answer_confidence_summary = self.build_confidence_summary(
                level="LOW",
                score=0.0,
                c_top=0.0,
                c_mean_top3=0.0,
                c_coverage=0.0,
                c_local=0.0,
                c_fallback_penalty=0.0,
                top_score=0.0,
                mean_top3=0.0,
                coverage=0.0,
                local_share=0.0,
                evidence_chunks=0,
                fallback_triggered=False,
                note="no selected evidence chunks",
            )
            self._emit_selection_confidence(
                level="LOW",
                score=0.0,
                evidence_chunks=0,
                fallback_triggered=False,
            )
            return

        doc_scores = [self._doc_confidence_score(doc) for doc in chosen]
        top_score = max(doc_scores)
        head_count = min(3, len(doc_scores))
        mean_top = (
            sum(doc_scores[:head_count]) / float(head_count) if head_count > 0 else 0.0
        )

        target_chunks = max(1, int(session.final_chunks_to_llm or len(chosen)))
        coverage = min(1.0, len(chosen) / float(target_chunks))

        local_count = sum(
            1
            for doc in chosen
            if str((getattr(doc, "metadata", {}) or {}).get("Source", "")).lower()
            != "web"
        )
        local_ratio = local_count / float(len(chosen))
        fallback_triggered = bool(
            getattr(session, "rerank_low_confidence_fallback_triggered", False)
        )

        c_top = 0.50 * top_score
        c_mean_top3 = 0.30 * mean_top
        c_coverage = 0.15 * coverage
        c_local = 0.05 * local_ratio
        c_raw_sum = c_top + c_mean_top3 + c_coverage + c_local
        c_fallback_penalty = 0.15 if fallback_triggered else 0.0

        score = self._clamp_probability(max(0.0, c_raw_sum - c_fallback_penalty))
        level = self._confidence_level_from_score(score)
        components = {
            "c_final": score,
            "c_top": c_top,
            "c_mean_top3": c_mean_top3,
            "c_coverage": c_coverage,
            "c_local": c_local,
            "c_fallback_penalty": c_fallback_penalty,
            "c_raw_sum": c_raw_sum,
            "top_score": top_score,
            "mean_top3": mean_top,
            "coverage": coverage,
            "local_share": local_ratio,
            "evidence_chunks": float(len(chosen)),
        }
        summary = self.build_confidence_summary(
            level=level,
            score=score,
            c_top=c_top,
            c_mean_top3=c_mean_top3,
            c_coverage=c_coverage,
            c_local=c_local,
            c_fallback_penalty=c_fallback_penalty,
            top_score=top_score,
            mean_top3=mean_top,
            coverage=coverage,
            local_share=local_ratio,
            evidence_chunks=len(chosen),
            fallback_triggered=fallback_triggered,
        )

        session.answer_confidence_components = components
        session.answer_confidence_score = score
        session.answer_confidence_level = level
        session.answer_confidence_summary = summary
        self._emit_selection_confidence(
            level=level,
            score=score,
            evidence_chunks=len(chosen),
            fallback_triggered=fallback_triggered,
        )
        self.confidence_logger.log_step(
            session,
            step_name="retrieval evidence confidence",
            status="executed",
            confidence_level=level,
            confidence_score=score,
            detail=summary,
            source="retrieval",
        )
