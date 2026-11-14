"""Retrieval diagnostics and limited, answer-text-only LLM review."""

import json
import math
from typing import TYPE_CHECKING, Any, Dict, List

from Config.Config import Config
from Orchestrator import FlowConfig

if TYPE_CHECKING:
    from AI.LLMCaller import LLMCaller
    from Chat.ConfidenceEvaluator import (AnswerConfidenceResult,
                                          ConfidenceEvaluator)
    from Globals.Session import Session


class QualityDimensionEvaluator:
    """Keep retrieval signals separate from subjective answer-text ratings."""

    MODES = ("off", "heuristic", "llm", "both")
    HEURISTIC_METRICS = {
        "evidence_relevance": (
            "Evidence relevance",
            "Top retrieval scores and fallback status; does not verify answer accuracy.",
        ),
        "retrieval_coverage": (
            "Retrieval coverage",
            "Selected chunk count relative to the target, plus evidence quantity; "
            "does not measure answer completeness.",
        ),
        "evidence_availability": (
            "Evidence availability",
            "Retrieval relevance and chunk availability; does not verify citations "
            "or claim-to-source support.",
        ),
    }
    JUDGE_METRICS = {
        "internal_consistency": (
            "Internal consistency",
            "Judge rating of contradictions within the answer; "
            "does not verify factual correctness.",
        ),
        "explanation_clarity": (
            "Explanation clarity",
            "Judge rating of how understandable the explanation is; "
            "does not verify reasoning validity.",
        ),
    }

    @classmethod
    def validate_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in cls.MODES:
            raise ValueError(
                f"Invalid confidence_evaluation {value!r}. "
                f"Allowed: {', '.join(cls.MODES)}"
            )
        return mode

    @classmethod
    def resolve_mode(cls, cfg: Config, session: "Session") -> str:
        override = getattr(session, "confidence_evaluation", None)
        if override is not None:
            return cls.validate_mode(override)
        flow_name = getattr(session, "orchestrator_flow", None) or (
            FlowConfig.default_flow_name(cfg)
        )
        profile = FlowConfig.normalize_flow_map(
            cfg.get("_ORCHESTRATION_FLOWS", {})
        ).get(FlowConfig.resolve_flow_alias(flow_name, cfg), {})
        return cls.validate_mode(profile.get("confidence_evaluation", "both"))

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg if cfg is not None else Config()

    def evaluate_answer(
        self,
        session: "Session",
        confidence: "AnswerConfidenceResult",
        answer_text: str,
        *,
        llm_caller: "LLMCaller",
        model: str,
        confidence_evaluator: "ConfidenceEvaluator",
    ) -> None:
        """Run the optional text review, store diagnostics, and update reporting."""
        confidence.judge_response = None
        confidence.judge_model = None
        confidence.quality_dimensions = {}
        confidence.quality_subsignals = {}
        confidence.quality_mode = None
        confidence.quality_error = None
        mode = self.resolve_mode(self.cfg, session)
        if mode == "off":
            return

        judge_error: str | None = None
        if mode in ("llm", "both"):
            confidence.judge_model = model
            try:
                prompt_template = self.cfg.get("_PROMPT_QUALITY_DIMENSION_JUDGE")
                response = llm_caller.call_llm(
                    model=model,
                    prompt=prompt_template.format(answer=answer_text),
                    ollama_options={
                        "temperature": float(
                            self.cfg.get("_QUALITY_CONFIDENCE.judge_temperature", 0.0)
                        ),
                    },
                    answer_is_json=True,
                    streaming=False,
                    stage="Answer-text LLM review",
                    purpose="Judge",
                    model_role="_ACTIVE_LLM_JUDGE",
                )
                if response.get("error"):
                    raise ValueError(response["error"])
                content = response.get("content", "").strip()
                if not content:
                    raise ValueError("LLM judge returned an empty response")
                confidence.judge_response = content
            except Exception as exc:
                judge_error = f"LLM judge failed: {exc}"
                confidence_evaluator.pretty.write("W", "Judge", judge_error)

        result = self.evaluate(
            confidence,
            answer_text,
            session.last_chosen_chunks,
            session_data=session,
            mode=mode,
        )
        confidence.quality_dimensions = result["scores"]
        confidence.quality_subsignals = result["subsignals"]
        confidence.quality_mode = result["mode"]
        confidence.quality_error = judge_error or result["error"]
        if result["error"] and not judge_error:
            confidence_evaluator.pretty.write(
                "W", "Quality evaluation", result["error"]
            )
        if self.cfg.get("_QUALITY_CONFIDENCE.emit_in_answer_block", True):
            confidence_evaluator.add_quality_dimensions(
                confidence, confidence.quality_dimensions
            )
        confidence_evaluator.confidence_logger.log_step(
            session,
            step_name="quality_dimensions",
            status=(
                "fallback"
                if confidence.quality_error and confidence.quality_dimensions
                else "failed" if confidence.quality_error else "executed"
            ),
            confidence_level="UNKNOWN",
            confidence_score=None,
            detail=confidence.quality_error
            or "Retrieval diagnostics / answer-text review completed",
            source="quality_evaluator",
            quality_dimensions=confidence.quality_dimensions,
        )

    def _clamp(self, x: float) -> float:
        """Clamp value to [0.0, 1.0] range."""
        return min(1.0, max(0.0, x))

    def _bool01(self, flag: bool) -> float:
        """Convert boolean to 1.0 or 0.0."""
        return 1.0 if flag else 0.0

    def _extract_base_signals(
        self,
        answer_confidence_components: Dict[str, Any],
        chosen_chunks: List[Any],
        session_data: Any = None,
    ) -> Dict[str, float]:
        """Extract base signals from answer confidence components and chosen chunks."""
        # Extract signals from answer confidence components
        top_score = answer_confidence_components.get("top_score", 0.0) or 0.0
        mean_top3 = answer_confidence_components.get("mean_top3", 0.0) or 0.0
        coverage = answer_confidence_components.get("coverage", 0.0) or 0.0

        # Extract signals from chosen chunks
        evidence_chunks = answer_confidence_components.get(
            "evidence_chunks", 0.0
        ) or len(chosen_chunks)

        # Check if fallback was triggered
        fallback_triggered = (
            getattr(session_data, "rerank_low_confidence_fallback_triggered", False)
            if session_data
            else False
        )

        # Calculate derived signals
        evidence_presence = 1.0 if evidence_chunks > 0 else 0.0
        evidence_quantity = self._clamp(evidence_chunks / 6.0)

        return {
            "top_score": top_score,
            "mean_top3": mean_top3,
            "coverage": coverage,
            "evidence_chunks": evidence_chunks,
            "fallback_triggered": fallback_triggered,
            "evidence_presence": evidence_presence,
            "evidence_quantity": evidence_quantity,
            "relevance_support": self._clamp(0.5 * top_score + 0.5 * mean_top3),
        }

    def _compute_heuristics(
        self, signals: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compute retrieval proxies, not answer-quality measurements."""
        top_score = signals["top_score"]
        mean_top3 = signals["mean_top3"]
        coverage = signals["coverage"]
        fallback_triggered = signals["fallback_triggered"]
        evidence_presence = signals["evidence_presence"]
        evidence_quantity = signals["evidence_quantity"]
        if not evidence_presence:
            return {key: {"heuristic": 0.0} for key in self.HEURISTIC_METRICS}

        relevance = self._clamp(
            0.55 * top_score
            + 0.35 * mean_top3
            + 0.10 * (1.0 - self._bool01(fallback_triggered))
        )

        retrieval_coverage = self._clamp(0.70 * coverage + 0.30 * evidence_quantity)
        availability = self._clamp(
            0.70 * signals["relevance_support"]
            + 0.20 * evidence_presence
            + 0.10 * evidence_quantity
        )

        return {
            "evidence_relevance": {"heuristic": relevance},
            "retrieval_coverage": {"heuristic": retrieval_coverage},
            "evidence_availability": {"heuristic": availability},
        }

    def _parse_judge_json(self, judge_response: str) -> Dict[str, float]:
        """Parse LLM judge JSON response."""
        try:
            # Try to parse as JSON
            parsed = json.loads(judge_response)

            # Validate required keys
            required_keys = self.JUDGE_METRICS
            if not isinstance(parsed, dict) or set(parsed) != set(required_keys):
                return {}

            # Convert scores to floats in [0.0, 1.0]
            result = {}
            for key in required_keys:
                value = parsed[key]
                if value is None:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= value <= 1.0
                    or not math.isfinite(value)
                ):
                    return {}
                result[key] = float(value)

            return result
        except (json.JSONDecodeError, TypeError):
            return {}

    def evaluate(
        self,
        confidence: "AnswerConfidenceResult",
        answer_text: str,
        chosen_chunks: List[Any],
        *,
        session_data: Any = None,
        mode: str = "both",
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval proxies and parse separate answer-text ratings.

        Args:
            confidence: Answer confidence including evidence components and judge response
            answer_text: The generated answer text
            chosen_chunks: List of chosen chunks used for generation

        Returns:
            Dictionary containing quality dimension scores and metadata
        """
        mode = self.validate_mode(mode)
        result: Dict[str, Any] = {
            "scores": {},
            "subsignals": {},
            "mode": None,
            "error": None,
        }
        if mode == "off":
            return result

        if mode in ("heuristic", "both"):
            signals = self._extract_base_signals(
                confidence.components, chosen_chunks, session_data
            )
            heuristics = self._compute_heuristics(signals)
            result["scores"] = heuristics
            result["subsignals"] = {
                key: signals[key]
                for key in (
                    "relevance_support",
                    "evidence_presence",
                    "evidence_quantity",
                )
            }
            result["mode"] = "heuristic_only"

        if mode in ("llm", "both"):
            judge_scores = self._parse_judge_json(confidence.judge_response or "")
            if not judge_scores:
                result["error"] = (
                    "LLM review returned invalid or entirely unassessable text ratings"
                )
                if mode == "llm":
                    result["mode"] = "llm_failed"
                return result
            for dimension, score in judge_scores.items():
                scores = result["scores"].setdefault(dimension, {})
                scores["llm"] = score
            result["mode"] = "both" if mode == "both" else "llm_only"
        return result
