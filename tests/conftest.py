import os
import sys
import types
import importlib
import warnings

import pytest

# ── filter noisy deprecation warnings (replaces pytest.ini) ──────────
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"importlib\._bootstrap"
)
warnings.filterwarnings(
    "ignore", message=r"builtin type [Ss]wig", category=DeprecationWarning
)

# Ensure `src/` is on sys.path so imports like `Algos.*` resolve during tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class StubScorer:
    def __init__(self, *a, **k):
        pass

    def verify(self, *args, **kwargs):
        return []


@pytest.fixture(autouse=False)
def stub_algos(monkeypatch):
    """Patch common Algos classes to lightweight stubs."""
    mods = [
        "Algos.RegexScorer",
        "Algos.JaccardScorer",
        "Algos.BM25Scorer",
        "Algos.CosineScorer",
        "Algos.KeyBertScorer",
        "Algos.LevenshteinScorer",
    ]
    for mod_name in mods:
        mod = importlib.import_module(mod_name)
        cls_name = mod_name.split(".")[-1]
        monkeypatch.setattr(mod, cls_name, StubScorer)
    yield


@pytest.fixture(autouse=False)
def fake_config(monkeypatch):
    """Provide a FakeConfig class that returns deterministic values for keys used in tests."""
    import Config.Config as cfg_mod

    class FakeConfig(cfg_mod.Config):
        def get(self, key, *a, **k):
            mapping = {
                "_KEYBERT": "KEYBERT",
                "_JACCARD": "JACCARD",
                "_REGEX": "REGEX",
                "_BM25": "BM25",
                "_COSINE": "COSINE",
                "DEBUG_LEVEL": 0,
                "_DETECTION_CONFIG": "TEST",
                "_FRIENDLY_NAME": "SITE",
                "_BANNED_CONFIG": "_BANNED",
                "_BANNED.BANNED": ["foo"],
            }
            return mapping.get(key, super().get(key, None))

    monkeypatch.setattr(cfg_mod, "Config", FakeConfig)
    yield


@pytest.fixture(autouse=False)
def patch_helpers(monkeypatch):
    import Gui.PrettyWriter as pw_mod
    import Helpers.Accumulator as acc_mod

    class PW:
        def write(self, *a, **k):
            return None

    class Acc:
        def add_results(self, results, stage):
            return (False, [])

        def show_accumulated(self, stage):
            return (False, [])

    monkeypatch.setattr(pw_mod, "PrettyWriter", PW)
    monkeypatch.setattr(acc_mod, "Accumulator", Acc)
    yield
