# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false, reportUnusedVariable=false, reportMissingTypeArgument=false
"""
Tests for HfTranslator (Compliance.HfTranslator).

The real M2M-100 model (~1.7 GB) is never loaded.  The model and tokenizer
attributes are pre-injected so `_ensure_loaded()` short-circuits, exercising
only the translator's own logic: language normalisation, cache, no-op
short-circuits, empty/exception handling, and the loader-failure fallback.
"""

import os
import sys
import threading
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.HfTranslator import HfTranslator

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def __init__(self, overrides=None):
        self._data = overrides or {}

    def get_dict(self, key, default=None):
        if key == "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME":
            # Minimal mapping covering the test cases.
            return {
                "en": "english",
                "de": "german",
                "fr": "french",
                "ja": "japanese",
            }
        v = self._data.get(key, default if default is not None else {})
        return v if isinstance(v, dict) else {}


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple] = []

    def write(self, severity, label, message, **kw):
        self.messages.append((severity, label, message, kw))
        return None


class StubHelpers:
    def __init__(self, model_args=None):
        self._model_args = model_args or {
            "model_name": "facebook/m2m100_418M",
            "revision": "main",
            "USE_GPU": False,
        }

    def get_model_args(self, role):
        return dict(self._model_args)


# ---------------------------------------------------------------------------
# Fake torch / model / tokenizer used to exercise translate_text without
# touching transformers at all.
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, _device):
        return self


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    """Mimics the M2M100Tokenizer surface used by HfTranslator."""

    def __init__(self, output_text="ÜBERSETZT"):
        self.src_lang: str = ""
        self.output_text = output_text
        self.calls: list[tuple[str, str, str]] = []  # (text, src, tgt)
        self._last_target = ""

    def __call__(self, text, return_tensors=None):
        # token count = 8 by convention; doesn't actually matter for assertions.
        return FakeInputs(input_ids=FakeTensor((1, 8)))

    def get_lang_id(self, iso_code: str) -> int:
        self._last_target = iso_code
        return 12345

    def batch_decode(self, _outputs, skip_special_tokens=True):
        self.calls.append(("decode", self.src_lang, self._last_target))
        return [self.output_text]


class FakeModel:
    def __init__(self):
        self.generation_config = type(
            "GC", (), {"max_length": 200, "early_stopping": True}
        )()
        self.generated = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **kw):
        self.generated += 1
        return ["ignored"]


# ---------------------------------------------------------------------------
# Helper: build a fresh HfTranslator bypassing the singleton guard.
# ---------------------------------------------------------------------------


def _make_translator(model_args=None, *, model=None, tokenizer=None, loaded=True):
    """Construct an HfTranslator with stub deps, bypassing SingletonMixin."""
    t = HfTranslator.__new__(HfTranslator)
    t._initialized = False
    cfg = StubConfig()
    pretty = StubPrettyWriter()
    helpers = StubHelpers(model_args=model_args)

    # Manually run __init__ body.
    t.cfg = cfg
    t.pretty = pretty
    t.helpers = helpers
    code_to_name = cfg.get_dict("_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME")
    t._lang_name_to_code = {
        str(name).lower(): str(code).lower() for code, name in code_to_name.items()
    }
    t._cache = {}
    t._model = model if loaded else None
    t._tokenizer = tokenizer if loaded else None
    t._device = "cpu" if loaded else None
    t._load_lock = threading.Lock()
    t._warned_targets = set()
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLanguageNormalisation:
    def test_iso_passthrough(self):
        t = _make_translator()
        assert t._to_iso("de") == "de"
        assert t._to_iso("EN") == "en"

    def test_nltk_name_mapped_to_iso(self):
        t = _make_translator()
        assert t._to_iso("German") == "de"
        assert t._to_iso("japanese") == "ja"

    def test_unknown_lang_returned_lowercased(self):
        t = _make_translator()
        # Unknown names fall through unchanged (lowercased).
        assert t._to_iso("Klingon") == "klingon"

    def test_empty_returns_empty(self):
        t = _make_translator()
        assert t._to_iso("") == ""


class TestShortCircuits:
    def test_empty_text_returns_text(self):
        t = _make_translator()
        assert t.translate_text("", "de", "en") == ""

    def test_missing_target_returns_text(self):
        t = _make_translator()
        assert t.translate_text("hello", "", "en") == "hello"

    def test_unknown_target_normalises_to_empty_returns_text(self):
        t = _make_translator()
        # When target normalises to "" (empty input), we return as-is.
        assert t.translate_text("hello", "", "en") == "hello"

    def test_same_language_roundtrips(self):
        tok = FakeTokenizer()
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        assert t.translate_text("hello", "english", "en") == "hello"
        # Model must NOT be invoked.
        assert mdl.generated == 0


class TestCache:
    def test_repeated_call_uses_cache(self, monkeypatch):
        tok = FakeTokenizer(output_text="hallo")
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        # Inject a fake torch module used inside translate_text.
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert t.translate_text("hello", "german", "english") == "hallo"
        assert mdl.generated == 1
        # Second call hits cache.
        assert t.translate_text("hello", "german", "english") == "hallo"
        assert mdl.generated == 1

    def test_cache_key_includes_language(self, monkeypatch):
        tok = FakeTokenizer(output_text="x")
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        t.translate_text("hello", "german", "english")
        t.translate_text("hello", "french", "english")
        # Two distinct (text, tgt, src) keys → two real generations.
        assert mdl.generated == 2


class TestLoaderFailureFallback:
    def test_returns_original_when_load_fails(self, monkeypatch):
        t = _make_translator(loaded=False)
        monkeypatch.setattr(t, "_ensure_loaded", lambda: False)
        result = t.translate_text("hello", "german", "english")
        assert result == "hello"
        # Loader failure does NOT cache — leaves the door open for a retry
        # once the model is available.
        assert ("hello", "de", "en") not in t._cache


class TestEmptyTranslation:
    def test_empty_output_returns_original_and_warns(self, monkeypatch):
        tok = FakeTokenizer(output_text="   ")  # whitespace → empty after strip
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = t.translate_text("hello", "german", "english")
        assert result == "hello"
        # First empty translation should have produced a warning.
        warns = [m for m in t.pretty.messages if m[0] == "W"]
        assert warns, "expected at least one warning for empty translation"
        # Same target → warning emitted only once.
        assert "de" in t._warned_targets


class TestExceptionFallback:
    def test_generate_exception_returns_original_and_warns(self, monkeypatch):
        tok = FakeTokenizer()
        mdl = FakeModel()

        def boom(**kw):
            raise RuntimeError("kaboom")

        mdl.generate = boom
        t = _make_translator(model=mdl, tokenizer=tok)
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = t.translate_text("hello", "german", "english")
        assert result == "hello"
        warns = [m for m in t.pretty.messages if m[0] == "W"]
        assert warns


class TestSourceLanguageHint:
    def test_auto_source_defaults_to_en(self, monkeypatch):
        tok = FakeTokenizer(output_text="bonjour")
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        t.translate_text("hello", "french", "auto")
        assert tok.src_lang == "en"
        assert tok._last_target == "fr"

    def test_explicit_source_propagates(self, monkeypatch):
        tok = FakeTokenizer(output_text="hello")
        mdl = FakeModel()
        t = _make_translator(model=mdl, tokenizer=tok)
        fake_torch = type("FT", (), {"no_grad": lambda: _NullCtx()})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        t.translate_text("hallo", "english", "german")
        assert tok.src_lang == "de"
        assert tok._last_target == "en"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
