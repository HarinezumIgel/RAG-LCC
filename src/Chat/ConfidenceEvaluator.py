from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

from Config.Config import Config
from Gui.Colors import GREEN, ORANGE, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.ConfidenceLogger import ConfidenceLogger
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator


@dataclass
class AnswerConfidenceResult:
    level: str
    score: float
    components: dict[str, float]
    summary: str
    note: str = ""
    quality_dimensions: dict[str, dict[str, float]] = field(default_factory=dict)
    quality_subsignals: dict[str, float] = field(default_factory=dict)
    quality_mode: str | None = None
    quality_error: str | None = None
    judge_response: str | None = None
    judge_model: str | None = None


class ConfidenceEvaluator:
    """Compute and persist answer confidence from selected evidence chunks."""

    def __init__(
        self,
        *,
        confidence_logger: ConfidenceLogger | None = None,
        pretty: PrettyWriter | None = None,
        cfg: Config | None = None,
    ) -> None:
        self.cfg: Config = cfg or Config()
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

    def _confidence_level_from_score(self, score: float) -> str:
        """Map a 0..1 confidence score to a user-facing level label."""
        if score >= self.cfg.get_float("_CONFIDENCE_EVALUATOR.level_high"):
            return "HIGH"
        if score >= self.cfg.get_float("_CONFIDENCE_EVALUATOR.level_medium"):
            return "MEDIUM"
        return "LOW"

    def _doc_confidence_score(self, doc: Any) -> float:
        """Estimate one chunk confidence on a normalized 0..1 scale."""
        metadata_obj = getattr(doc, "metadata", None)
        metadata: dict[str, Any] = (
            cast(dict[str, Any], metadata_obj) if isinstance(metadata_obj, dict) else {}
        )

        raw_logit = metadata.get("raw_rerank_score")
        if raw_logit is not None:
            try:
                return self._clamp_probability(
                    1.0 / (1.0 + math.exp(-float(raw_logit)))
                )
            except (TypeError, ValueError, OverflowError):
                pass

        for key in ("rerank_score", "chroma_score"):
            candidate = metadata.get(key)
            if candidate is None:
                continue
            try:
                return self._clamp_probability(float(candidate))
            except (TypeError, ValueError):
                continue

        # Unknown score scales (e.g. raw BM25 / RRF only): keep neutral instead
        # of forcing an artificially low confidence.
        for key in ("rrf_score", "bm25_score", "graph_score", "regex_score"):
            if metadata.get(key) is not None:
                return self.cfg.get_float("_CONFIDENCE_EVALUATOR.unknown_scale_score")

        return 0.0

    def build_confidence_summary(
        self,
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
        quality_dimensions: dict[str, dict[str, float]] | None = None,
        judge_model: str | None = None,
    ) -> str:
        """Return a compact multi-line confidence payload with metric meaning."""
        normalized_level = str(level or "LOW").strip().upper()
        icon = self._confidence_icon_from_level(normalized_level)
        cfg = self.cfg
        lines: list[str] = [
            f"{icon}{normalized_level}",
            "",
            "Heuristic Confidence:",
        ]

        metrics: list[tuple[str, str]] = [
            (
                f"C_final={score:.2f}",
                "final confidence in [0,1] after penalties.",
            ),
            (
                f"C_top={c_top:.2f}",
                "weighted best-chunk signal "
                f"({cfg.get_float('_CONFIDENCE_EVALUATOR.weight_top'):.2f} x top_score).",
            ),
            (
                f"C_mean_top3={c_mean_top3:.2f}",
                "weighted top-3 stability "
                f"({cfg.get_float('_CONFIDENCE_EVALUATOR.weight_mean_top'):.2f} x mean_top3).",
            ),
            (
                f"C_coverage={c_coverage:.2f}",
                "weighted evidence coverage "
                f"({cfg.get_float('_CONFIDENCE_EVALUATOR.weight_coverage'):.2f} x coverage).",
            ),
            (
                f"C_local={c_local:.2f}",
                "weighted local-source share "
                f"({cfg.get_float('_CONFIDENCE_EVALUATOR.weight_local'):.2f} x local_share).",
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
        if quality_dimensions:
            for scorer, dimension_metrics, header in (
                (
                    "heuristic",
                    QualityDimensionEvaluator.HEURISTIC_METRICS,
                    "Static retrieval diagnostics (not answer-quality verification):",
                ),
                (
                    "llm",
                    QualityDimensionEvaluator.JUDGE_METRICS,
                    f"LLM review - answer text only (judge model: {judge_model or 'unspecified'}):",
                ),
            ):
                if not any(scorer in scores for scores in quality_dimensions.values()):
                    continue
                lines.extend(("", header))
                width = max(len(label) for label, _ in dimension_metrics.values())
                for key, (label, explanation) in dimension_metrics.items():
                    value = quality_dimensions.get(key, {}).get(scorer)
                    rating = (
                        f"{scorer}={value:.2f}" if value is not None else "not assessed"
                    )
                    lines.append(f"- {label.ljust(width, pad_char)}: {rating}")
                    lines.append(f"  {explanation}")
                if scorer == "llm":
                    lines.append(
                        "Subjective ratings, not probabilities. The judge sees neither "
                        "the question nor source evidence and may share the generator's blind spots."
                    )
                    lines.append(
                        "Accuracy, answer completeness, reasoning validity, and citation "
                        "support: not assessed."
                    )
        return "\n".join(lines)

    def store_answer_confidence(
        self,
        session: Any,
        chosen: list[Any],
        *,
        note: str = "",
        emit: bool = True,
    ) -> AnswerConfidenceResult:
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
            summary = self.build_confidence_summary(
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
                note=note or "no selected evidence chunks",
            )
            result = AnswerConfidenceResult(
                level="LOW",
                score=0.0,
                components=components,
                summary=summary,
                note=note or "no selected evidence chunks",
            )
            session.answer_confidence_result = result
            if emit:
                self._emit_selection_confidence(
                    level="LOW",
                    score=0.0,
                    evidence_chunks=0,
                    fallback_triggered=False,
                )
            return result

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

        cfg = self.cfg
        c_top = cfg.get_float("_CONFIDENCE_EVALUATOR.weight_top") * top_score
        c_mean_top3 = cfg.get_float("_CONFIDENCE_EVALUATOR.weight_mean_top") * mean_top
        c_coverage = cfg.get_float("_CONFIDENCE_EVALUATOR.weight_coverage") * coverage
        c_local = cfg.get_float("_CONFIDENCE_EVALUATOR.weight_local") * local_ratio
        c_raw_sum = c_top + c_mean_top3 + c_coverage + c_local
        c_fallback_penalty = (
            cfg.get_float("_CONFIDENCE_EVALUATOR.fallback_penalty")
            if fallback_triggered
            else 0.0
        )

        # Apply quality dimension enhancements
        score = self._clamp_probability(max(0.0, c_raw_sum - c_fallback_penalty))

        # Use the base score (no quality dimension enhancement in this method)
        final_score = score

        level = self._confidence_level_from_score(final_score)
        components = {
            "c_final": final_score,
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
            score=final_score,
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
            note=note,
        )

        result = AnswerConfidenceResult(
            level=level,
            score=final_score,
            components=components,
            summary=summary,
            note=note,
        )
        session.answer_confidence_result = result
        if emit:
            self._emit_selection_confidence(
                level=level,
                score=final_score,
                evidence_chunks=len(chosen),
                fallback_triggered=fallback_triggered,
            )
        self.confidence_logger.log_step(
            session,
            step_name="retrieval evidence confidence",
            status="executed",
            confidence_level=level,
            confidence_score=final_score,
            detail=summary,
            source="retrieval",
        )
        return result

    def add_quality_dimensions(
        self,
        result: AnswerConfidenceResult,
        quality_dimensions: dict[str, dict[str, float]],
    ) -> None:
        """Add evaluated quality scores to the answer confidence result."""
        result.quality_dimensions = quality_dimensions
        components = result.components
        result.summary = self.build_confidence_summary(
            level=result.level,
            score=result.score,
            c_top=components["c_top"],
            c_mean_top3=components["c_mean_top3"],
            c_coverage=components["c_coverage"],
            c_local=components["c_local"],
            c_fallback_penalty=components["c_fallback_penalty"],
            top_score=components["top_score"],
            mean_top3=components["mean_top3"],
            coverage=components["coverage"],
            local_share=components["local_share"],
            evidence_chunks=int(components["evidence_chunks"]),
            fallback_triggered=components["c_fallback_penalty"] > 0.0,
            note=result.note,
            quality_dimensions=quality_dimensions,
            judge_model=result.judge_model,
        )
