"""Tests for narrowly labelled retrieval diagnostics and answer-text ratings."""

from types import SimpleNamespace

import pytest

from Chat.ConfidenceEvaluator import AnswerConfidenceResult
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator


def test_static_formulas_are_retrieval_proxies():
    evaluator = QualityDimensionEvaluator()
    signals = evaluator._extract_base_signals(
        {"top_score": 0.8, "mean_top3": 0.7, "coverage": 0.6, "evidence_chunks": 5},
        [],
        SimpleNamespace(rerank_low_confidence_fallback_triggered=False),
    )
    assert signals["evidence_presence"] == 1.0
    assert signals["evidence_quantity"] == pytest.approx(5 / 6)
    assert signals["relevance_support"] == pytest.approx(0.75)
    scores = evaluator._compute_heuristics(signals)
    assert scores == {
        "evidence_relevance": {"heuristic": pytest.approx(0.785)},
        "retrieval_coverage": {"heuristic": pytest.approx(0.67)},
        "evidence_availability": {"heuristic": pytest.approx(0.8083333333)},
    }


def test_empty_evidence_has_no_positive_diagnostics():
    evaluator = QualityDimensionEvaluator()
    signals = evaluator._extract_base_signals({}, [])
    assert evaluator._compute_heuristics(signals) == {
        key: {"heuristic": 0.0} for key in evaluator.HEURISTIC_METRICS
    }


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "[]",
        "null",
        '{"accuracy":1,"completeness":1,"reasoning":1,"provenance":1}',
        '{"internal_consistency":1}',
        '{"internal_consistency":true,"explanation_clarity":0.5}',
        '{"internal_consistency":NaN,"explanation_clarity":0.5}',
        '{"internal_consistency":Infinity,"explanation_clarity":0.5}',
        '{"internal_consistency":-0.1,"explanation_clarity":0.5}',
        '{"internal_consistency":1.1,"explanation_clarity":0.5}',
        '{"internal_consistency":"0.5","explanation_clarity":0.5}',
        '{"internal_consistency":0.5,"explanation_clarity":0.5,"accuracy":1}',
    ],
)
def test_invalid_judge_contract_is_rejected(text):
    assert QualityDimensionEvaluator()._parse_judge_json(text) == {}


def test_judge_can_explicitly_decline_a_rating():
    assert QualityDimensionEvaluator()._parse_judge_json(
        '{"internal_consistency":0,"explanation_clarity":null}'
    ) == {"internal_consistency": 0.0}


def test_both_keeps_separate_metrics_without_average():
    confidence = AnswerConfidenceResult(
        level="HIGH",
        score=0.8,
        components={"evidence_chunks": 2},
        summary="",
        judge_response='{"internal_consistency":0.9,"explanation_clarity":0.6}',
    )
    result = QualityDimensionEvaluator().evaluate(confidence, "answer", [])
    assert result["mode"] == "both"
    assert set(result["scores"]) == {
        "evidence_relevance",
        "retrieval_coverage",
        "evidence_availability",
        "internal_consistency",
        "explanation_clarity",
    }
    assert result["scores"]["internal_consistency"] == {"llm": 0.9}
    assert all(
        len(values) == 1 and "avg" not in values for values in result["scores"].values()
    )
