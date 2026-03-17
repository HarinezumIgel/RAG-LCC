# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Unit tests for core algorithms and helpers.
Modernized for clarity and maintainability.
"""

import importlib
import types
import sys
import torch


def test_compliance_dataclass():
    """Test ComplianceAlgoResult dataclass fields."""
    from Algos.ComplianceAlgoResult import ComplianceAlgoResult

    r = ComplianceAlgoResult(algo="x", phrase="p", score=0.5, threshold=0.3)
    assert r.algo == "x"
    assert r.phrase == "p"
    assert r.score is not None and abs(r.score - 0.5) < 1e-8


def test_cosine_similarity():
    """Test CosineScorer._cosine_similarity for orthogonal and identical vectors."""
    from Algos.CosineScorer import CosineScorer

    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    v = torch.tensor([1.0, 0.0])
    assert abs(CosineScorer._cosine_similarity(a, b) - 0.0) < 1e-6  # type: ignore[reportPrivateUsage]
    assert abs(CosineScorer._cosine_similarity(a, v) - 1.0) < 1e-6  # type: ignore[reportPrivateUsage]


def test_regex_verify_basic(monkeypatch):
    """Test RegexScorer.verify with monkeypatched config and helpers."""
    # Use monkeypatch to inject a lightweight AI.AIHelpers module before instantiation
    fake_mod = types.ModuleType("AI.AIHelpers")

    class FakeAIHelpers:
        def get_banned_phrases_config_slot(self):
            return ["foo"]

        def get_compliance_config_slot(self, stage):
            return "_BANNED_DETECT.TEST.SITE"

    fake_mod.AIHelpers = FakeAIHelpers  # type: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "AI.AIHelpers", fake_mod)
    importlib.invalidate_caches()
    # Patch Config.Config.Config.get to return safe defaults for nested keys
    import Config.Config as cfg_mod

    def safe_get(self, key, default=None, allow_indirect=True, **kwargs):
        k = key or ""
        if ".BANNED" in k:
            return ["foo"]
        if "TOP_N" in k or "PREFIX_SUFFIX_LEN" in k:
            return 2
        if "THRESHOLD" in k or "SOFT_SCORE" in k or "FUZZY_REGEX" in k:
            return 0.0
        if k == "DEBUG_LEVEL":
            return 0
        return default

    monkeypatch.setattr(cfg_mod.Config, "get", safe_get, raising=False)
    from Algos.RegexScorer import RegexScorer

    r = RegexScorer()
    # set runtime defaults expected by RegexScorer.verify
    r.window_max_chars = 5
    r.pref_suf_len = 2
    r.sep_class = r"\s"
    # ensure sharedHelpers has expected maps
    r.sharedHelpers.confusables = {}
    r.sharedHelpers.leet_map = {}
    res = r.verify("this is foo bar", "en", "PIPELINE_CHECK")
    assert any(getattr(x, "phrase", None) and "foo" in x.phrase for x in res)
