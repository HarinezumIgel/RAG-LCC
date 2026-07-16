# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
import importlib
import types
import sys
from typing import Any


def _make_stub_scorer():
    class StubScorer:
        def __init__(self, *a, **k):
            pass

        def verify(self, *args, **kwargs):
            return []

    return StubScorer


def test_run_ensemble_checks_smoke(monkeypatch):
    # Replace algorithm classes with lightweight stubs before instantiating AIHelpers
    Stub = _make_stub_scorer()

    # Import modules and patch their exported classes
    import Algos.RegexScorer as regex_mod
    import Algos.JaccardScorer as jaccard_mod
    import Algos.BM25Scorer as bm25_mod
    import Algos.CosineScorer as cosine_mod
    import Algos.KeyBertScorer as keybert_mod
    import Algos.LevenshteinScorer as lev_mod

    monkeypatch.setattr(regex_mod, "RegexScorer", Stub)
    monkeypatch.setattr(jaccard_mod, "JaccardScorer", Stub)
    monkeypatch.setattr(bm25_mod, "BM25Scorer", Stub)
    monkeypatch.setattr(cosine_mod, "CosineScorer", Stub)
    monkeypatch.setattr(keybert_mod, "KeyBertScorer", Stub)
    monkeypatch.setattr(lev_mod, "LevenshteinScorer", Stub)

    # Patch Helpers used by AIHelpers: PrettyWriter and Accumulator
    import Gui.PrettyWriter as pw_mod

    class PW:
        def write(self, *a, **k):
            return None

    monkeypatch.setattr(pw_mod, "PrettyWriter", PW)

    import Helpers.Accumulator as acc_mod

    class Acc:
        def add_results(self, results, stage):
            return (False, [])

        def show_accumulated(self, stage):
            return (False, [])

    monkeypatch.setattr(acc_mod, "Accumulator", Acc)

    # Patch the existing Config.get method to provide safe fallbacks
    import Config.Config as cfg_mod

    def fake_get(self, key, *a, **kw):
        mapping = {
            "_KEYBERT": "KEYBERT",
            "_JACCARD": "JACCARD",
            "_REGEX": "REGEX",
            "_BM25": "BM25",
            "_COSINE": "COSINE",
            "_LEVENSHTEIN": "LEVENSHTEIN",
            "_ACTIVE_EMBED": "test-impl",
            "DEBUG_LEVEL": 0,
            "TERMINAL_LINE_SIZE": 160,
            "_DETECTION_CONFIG": "TEST",
            "_FRIENDLY_NAME": "SITE",
            "_BANNED_CONFIG": "_BANNED",
            "_BANNED.BANNED": ["foo"],
            "_CUSTOM_NLTK_DATA_DIRECTORY": "",
            "_LABEL_ALIAS": {},
            "_HF_HOME": "",
            "_HF_HUB_CACHE": "",
            "_LOG_DIRECTORY": "logs",
            "_DEFAULT_ALGOS": ["KEYBERT", "JACCARD", "REGEX", "BM25", "COSINE"],
            "USE_CPU": False,
            "OLLAMA_STREAMING_REQ": False,
            "_KEY_BERT": {
                "TOP_N_FIRST": 2,
                "TOP_N_SECOND": 2,
            },
            "_ACTIVE_ENDPOINT": "ollama",
            "_MODELS": {
                "test-impl": {
                    "_EMBED": {
                        "MODEL": "test-model",
                        "REVISION": None,
                        "FRIENDLY_NAME": "test",
                        "SOURCE": "local",
                    },
                },
                "ollama": {
                    "_OLLAMA": {
                        "MODEL": "test-ollama",
                        "REVISION": None,
                        "FRIENDLY_NAME": "test-ollama",
                        "SOURCE": "local",
                        "STREAMING_REQ": False,
                        "BASE_URL": "http://localhost:11434",
                    },
                },
            },
        }
        if key in mapping:
            return mapping[key]
        k = key or ""
        if "BANNED" in k:
            return ["foo"]
        if "TOP_N" in k or "PREFIX_SUFFIX_LEN" in k:
            return 2
        if (
            "THRESHOLD" in k
            or "SOFT_SCORE" in k
            or "THRESHOLD_MIN" in k
            or "FUZZY_REGEX" in k
        ):
            return 0.0
        if "CHUNK_SIZE" in k or "CHUNK_OVERLAP" in k:
            return 512
        if "EMBEDDER_BITS" in k or ("EMBED" in k and "BITS" in k):
            return 32
        if "EMBED" in k or "MODEL" in k:
            return "test-model"
        # Return the caller-supplied default when no pattern matches
        if a:
            return a[0]
        return None

    monkeypatch.setattr(cfg_mod.Config, "get", fake_get, raising=False)
    # Ensure AIHelpers and ModelsCache are re-imported so they pick up
    # the patched Config and the fake langchain_huggingface module.
    for mod_name in ("AI.AIHelpers", "AI.ModelsCache"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    # Provide a lightweight stub for langchain_huggingface.HuggingFaceEmbeddings
    # so AIHelpers.get_hf_embeddings does not attempt to download models.
    fake_lch: Any = types.ModuleType("langchain_huggingface")

    class FakeHFE:
        def __init__(self, *a, **k):
            pass

        def embed_documents(self, docs):
            # return simple numeric vectors for each document
            return [[0.0, 0.0, 0.0] for _ in docs]

    fake_lch.HuggingFaceEmbeddings = FakeHFE
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_lch)
    importlib.invalidate_caches()
    from AI.AIHelpers import AIHelpers

    ah = AIHelpers()
    out = ah.run_ensemble_checks("some text", "en", "PIPELINE_CHECK")
    assert isinstance(out, tuple)
