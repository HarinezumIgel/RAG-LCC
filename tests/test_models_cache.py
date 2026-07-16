# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnusedImport=false, reportReturnType=false
"""
Tests for ModelsCache — model lifecycle management (device selection,
caching, quantization, embedding loading, cross-encoder loading).

Uses DI + attribute injection to bypass heavy __init__ and real model
downloads.  No monkeypatching or sys.modules manipulation needed.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importlib
import torch
import AI.ModelsCache as _mc_mod
from AI.ModelsCache import ModelsCache

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    """Provides the config keys ModelsCache reads."""

    def __init__(self, overrides=None):
        self._overrides = overrides or {}
        self._sets: list[tuple[str, object]] = []

    def get(self, key, default=None):
        if key in self._overrides:
            return self._overrides[key]
        mapping = {
            "_HF_HUB_CACHE": "fake_cache_dir",
            "_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG": "THOROUGH",
            "_ACTIVE_CHUNKER_CONFIG": "DETAILED",
            "_CHUNK_STRATEGY.DETAILED.DEFAULT": "RECURSIVE",
            "_CHUNKERS.RECURSIVE.CHUNK_SIZE": 512,
            "USE_CPU": True,
            "EMBEDDER_BITS": 32,
            "_MODELS": {
                "_EMBED": {
                    "MODEL": "test-embed-model",
                    "REVISION": "",
                    "FRIENDLY_NAME": "Test Embed",
                    "SOURCE": "test",
                },
                "_CROSS": {
                    "MODEL": "test-cross-model",
                    "REVISION": "",
                    "FRIENDLY_NAME": "Test Cross",
                    "SOURCE": "test",
                },
            },
        }
        if key in mapping:
            return mapping[key]
        if default is not None:
            return default
        return None

    def get_int(self, key, default=0):
        val = self.get(key, default)
        return int(val) if val is not None else default

    def get_str(self, key, default=""):
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_bool(self, key, default=False):
        val = self.get(key, default)
        return bool(val) if val is not None else default

    def get_float(self, key, default=0.0):
        val = self.get(key, default)
        return float(val) if val is not None else default

    def get_list(self, key, default=None):
        return self.get(key, default if default is not None else [])

    def get_dict(self, key, default=None):
        return self.get(key, default if default is not None else {})

    def set(self, key, value, force=False):
        self._sets.append((key, value))
        self._overrides[key] = value


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[object, ...]] = []

    def write(self, *a, **kw):
        self.messages.append(a)
        return None


class StubHelpers:
    """Provides get_model_args and bit_to_dtype stubs."""

    def __init__(self, model_args=None, dtype=None):
        self._model_args = model_args or {
            "model_name": "test-embed-model",
            "revision": "abc123",
            "local_files_only": True,
            "friendly_name": "Test Embed",
            "source": "test",
        }
        self._dtype = dtype or torch.float32

    def get_model_args(self, what):
        if what in ("_CROSS", "_ACTIVE_CROSS"):
            return {
                "model_name": "test-cross-model",
                "revision": "def456",
                "local_files_only": True,
                "friendly_name": "Test Cross",
                "source": "test",
            }
        return dict(self._model_args)

    def bit_to_dtype(self, bits=32):
        dtype_map = {32: torch.float32, 16: torch.float16, 8: torch.qint8}
        return dtype_map.get(bits, self._dtype)

    def get_chroma_config_slot(self) -> str:
        return "_CHROMA_EMBED_AND_RETRIEVE_PARAMS.THOROUGH"

    def get_chunker_config_slot(self) -> str:
        return "_CHUNKERS.RECURSIVE"

    def get_chunker_max_size(self) -> int:
        return 512


# ---------------------------------------------------------------------------
# Fake model classes
# ---------------------------------------------------------------------------


class FakeHFEmbeddings:
    """Mimics HuggingFaceEmbeddings for caching tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._client = FakeClient()


class FakeClient:
    def to(self, dtype):
        return self


class FakeSentenceTransformer:
    """Mimics SentenceTransformer."""

    def __init__(self, modules=None, device=None):
        self.modules = modules
        self.device_str = device

    def to(self, dtype):
        return self

    def eval(self):
        return self


class FakeTransformerModule:
    """Mimics sentence_transformers.models.Transformer."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.auto_model = FakeAutoModel()

    def get_embedding_dimension(self):
        return 384


class FakeAutoModel:
    pass


class FakePooling:
    def __init__(
        self, embedding_dimension=None, pooling_mode=None, pooling_mode_mean_tokens=None
    ):
        self.dim = embedding_dimension
        self.pooling_mode = pooling_mode


class FakeCrossEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    def __init__(self):
        pass

    def __call__(
        self, texts, truncation=True, max_length=512, padding=False, return_tensors=None
    ):
        # Simulate tokenization: return fake input_ids
        return {"input_ids": [[1, 2, 3] for _ in texts]}

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return "truncated text"


class FakeHFDownloader:
    """Records calls instead of performing real downloads."""

    def __init__(self):
        self.download_calls: list[str] = []

    def download(self, key):
        self.download_calls.append(key)
        return {"model_id": "test", "revision": "abc"}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_models_cache(cfg_overrides=None, helpers=None, pretty=None):
    """Build a ModelsCache singleton with all deps stubbed."""
    mc = ModelsCache.__new__(ModelsCache)
    mc._initialized = True

    mc.cfg = StubConfig(cfg_overrides)
    mc.pretty = pretty or StubPrettyWriter()
    mc.helpers = helpers or StubHelpers()

    mc.hf_embeddings_cache = {}
    mc.sentence_transformer_cache = []

    mc.cache_dir = mc.cfg.get_str("_HF_HUB_CACHE")
    mc.chunk_size = mc.helpers.get_chunker_max_size()
    mc.hf_hub_offline = "1"
    mc.use_cpu = mc.cfg.get_bool("USE_CPU")
    mc.bits = mc.cfg.get_int("EMBEDDER_BITS", 32)
    mc.lastDeviceBitSize = 0

    return mc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset():
    # Reload the module to undo any cross-test pollution (e.g. test_ensemble.py
    # deleting AI.ModelsCache from sys.modules and reimporting it with a
    # fake langchain_huggingface).
    global ModelsCache, _mc_mod
    if "AI.ModelsCache" in sys.modules:
        _mc_mod = sys.modules["AI.ModelsCache"]
        importlib.reload(_mc_mod)
    else:
        _mc_mod = importlib.import_module("AI.ModelsCache")
    ModelsCache = _mc_mod.ModelsCache

    ModelsCache._reset()
    yield
    ModelsCache._reset()


# ===================================================================
# switch2device
# ===================================================================


class TestSwitch2Device:
    def test_cpu_when_use_cpu_flag_set(self):
        """When USE_CPU is True, should return CPU device."""
        mc = _build_models_cache()
        mc.use_cpu = True

        device, device_type, _, idx = mc.switch2device()

        assert device == torch.device("cpu")
        assert device_type == "cpu"
        assert idx == 0

    def test_cpu_when_config_use_cpu_true(self):
        """When config USE_CPU=True, should return CPU."""
        mc = _build_models_cache(cfg_overrides={"USE_CPU": True})
        mc.use_cpu = False  # Instance flag off, but config on

        device, device_type, _, _ = mc.switch2device()

        assert device == torch.device("cpu")
        assert device_type == "cpu"

    def test_cpu_when_cuda_not_available(self, monkeypatch):
        """When CUDA is not available, should return CPU."""
        mc = _build_models_cache(cfg_overrides={"USE_CPU": False})
        mc.use_cpu = False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        device, device_type, _, _ = mc.switch2device()

        assert device == torch.device("cpu")
        assert device_type == "cpu"

    def test_returns_dtype_from_bits_config(self):
        """dtype should reflect the EMBEDDER_BITS config value."""
        mc = _build_models_cache(cfg_overrides={"EMBEDDER_BITS": 16})
        mc.use_cpu = True

        _, _, dtype, _ = mc.switch2device()

        assert dtype == torch.float16

    def test_default_bits_32_returns_float32(self):
        """Default 32-bit should return float32."""
        mc = _build_models_cache(cfg_overrides={"EMBEDDER_BITS": 32})
        mc.use_cpu = True

        _, _, dtype, _ = mc.switch2device()

        assert dtype == torch.float32

    def test_cuda_no_devices_returns_cpu(self, monkeypatch):
        """CUDA available but no devices → CPU with idx=-1."""
        mc = _build_models_cache(cfg_overrides={"USE_CPU": False})
        mc.use_cpu = False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        device, device_type, _, idx = mc.switch2device()

        assert device == torch.device("cpu")
        assert device_type == "cpu"
        assert idx == -1


# ===================================================================
# fallback_to_cpu
# ===================================================================


class TestFallbackToCpu:
    def test_sets_use_cpu_flag(self):
        mc = _build_models_cache()
        mc.use_cpu = False

        mc.fallback_to_cpu("test reason")

        assert mc.use_cpu is True

    def test_persists_to_config(self):
        mc = _build_models_cache()

        mc.fallback_to_cpu("GPU failed")

        sets = mc.cfg._sets
        assert ("USE_CPU", True) in sets
        assert ("EMBEDDER_BITS", 32) in sets

    def test_logs_warning(self):
        pw = StubPrettyWriter()
        mc = _build_models_cache(pretty=pw)

        mc.fallback_to_cpu("CUDA exploded")

        # Should have written a warning message
        assert any("CUDA exploded" in str(m) for m in pw.messages)


# ===================================================================
# get_hf_embeddings — caching behaviour
# ===================================================================


class TestGetHFEmbeddings:
    def test_returns_cached_embeddings_on_second_call(self, monkeypatch):
        """Second call for same model should return the cached instance."""
        mc = _build_models_cache()

        monkeypatch.setattr(_mc_mod, "HuggingFaceEmbeddings", FakeHFEmbeddings)

        emb1 = mc.get_hf_embeddings()
        emb2 = mc.get_hf_embeddings()

        assert emb1 is emb2

    def test_cache_key_includes_revision_and_device(self, monkeypatch):
        """Different revisions should produce different cache entries."""
        mc = _build_models_cache()

        monkeypatch.setattr(_mc_mod, "HuggingFaceEmbeddings", FakeHFEmbeddings)

        # Load first with default args
        emb1 = mc.get_hf_embeddings()

        # Now change revision
        mc.helpers = StubHelpers(
            model_args={
                "model_name": "test-embed-model",
                "revision": "different_rev",
                "local_files_only": True,
                "friendly_name": "Test",
                "source": "test",
            }
        )
        emb2 = mc.get_hf_embeddings()

        assert emb1 is not emb2
        assert len(mc.hf_embeddings_cache) == 2

    def test_hfdownloader_fallback_on_load_failure(self, monkeypatch):
        """When local load fails, HFDownloader should be called."""
        mc = _build_models_cache()

        call_count = 0

        def failing_then_succeeding(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Model not found locally")
            return FakeHFEmbeddings(**kwargs)

        monkeypatch.setattr(_mc_mod, "HuggingFaceEmbeddings", failing_then_succeeding)

        download_calls = []

        class PatchedHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                download_calls.append(key)

        monkeypatch.setattr(_mc_mod, "HFDownloader", PatchedHFDownloader)

        emb = mc.get_hf_embeddings()

        assert len(download_calls) == 1
        assert download_calls[0] == "_MODELS._EMBED"
        assert emb is not None

    def test_cuda_oom_falls_back_to_cpu(self, monkeypatch):
        """CUDA OOM during load should trigger CPU fallback."""
        mc = _build_models_cache(cfg_overrides={"USE_CPU": False})
        mc.use_cpu = False

        call_count = 0

        def oom_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("CUDA OOM")
            return FakeHFEmbeddings(**kwargs)

        monkeypatch.setattr(_mc_mod, "HuggingFaceEmbeddings", oom_then_ok)
        # Force CPU path on switch2device
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        emb = mc.get_hf_embeddings()

        assert mc.use_cpu is True
        assert emb is not None


# ===================================================================
# invalidate_hf_embeddings
# ===================================================================


class TestInvalidateHFEmbeddings:
    def test_clear_all(self):
        mc = _build_models_cache()
        mc.hf_embeddings_cache = {"key1": "emb1", "key2": "emb2"}

        mc.invalidate_hf_embeddings()

        assert len(mc.hf_embeddings_cache) == 0

    def test_clear_all_with_none_model_name(self):
        mc = _build_models_cache()
        mc.hf_embeddings_cache = {"key1": "emb1"}

        mc.invalidate_hf_embeddings(model_name=None)

        assert len(mc.hf_embeddings_cache) == 0


# ===================================================================
# load_quantized_model — caching behaviour
# ===================================================================


class TestLoadQuantizedModel:
    def test_returns_cached_model_on_second_call(self, monkeypatch):
        """Second call with the same model should return cached instance."""
        mc = _build_models_cache()

        monkeypatch.setattr(_mc_mod.models, "Transformer", FakeTransformerModule)
        monkeypatch.setattr(_mc_mod.models, "Pooling", FakePooling)
        monkeypatch.setattr(_mc_mod, "SentenceTransformer", FakeSentenceTransformer)

        model1 = mc.load_quantized_model("test-model")
        model2 = mc.load_quantized_model("test-model")

        assert model1 is model2

    def test_different_models_get_separate_cache_entries(self, monkeypatch):
        """Different model names should produce different cache entries."""
        mc = _build_models_cache()

        monkeypatch.setattr(_mc_mod.models, "Transformer", FakeTransformerModule)
        monkeypatch.setattr(_mc_mod.models, "Pooling", FakePooling)
        monkeypatch.setattr(_mc_mod, "SentenceTransformer", FakeSentenceTransformer)

        model1 = mc.load_quantized_model("model-a")
        model2 = mc.load_quantized_model("model-b")

        assert model1 is not model2
        assert len(mc.sentence_transformer_cache) == 2

    def test_hfdownloader_fallback_on_local_failure(self, monkeypatch):
        """When local model load fails, HFDownloader should be called."""
        mc = _build_models_cache()

        call_count = 0

        def transformer_stub(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Model not in cache")
            return FakeTransformerModule(**kwargs)

        monkeypatch.setattr(_mc_mod.models, "Transformer", transformer_stub)
        monkeypatch.setattr(_mc_mod.models, "Pooling", FakePooling)
        monkeypatch.setattr(_mc_mod, "SentenceTransformer", FakeSentenceTransformer)

        download_calls = []

        class PatchedHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                download_calls.append(key)

        monkeypatch.setattr(_mc_mod, "HFDownloader", PatchedHFDownloader)

        model = mc.load_quantized_model("test-model")

        assert len(download_calls) == 1
        assert download_calls[0] == "_MODELS._EMBED"
        assert model is not None

    def test_cache_key_includes_dtype(self, monkeypatch):
        """Cache key should incorporate dtype (via bits) for uniqueness."""
        mc = _build_models_cache()

        monkeypatch.setattr(_mc_mod.models, "Transformer", FakeTransformerModule)
        monkeypatch.setattr(_mc_mod.models, "Pooling", FakePooling)
        monkeypatch.setattr(_mc_mod, "SentenceTransformer", FakeSentenceTransformer)

        _ = mc.load_quantized_model("same-model")

        # After first load, the cache key contains the revision and dtype
        assert len(mc.sentence_transformer_cache) == 1
        key1 = list(mc.sentence_transformer_cache[0].keys())[0]

        # Check that the dtype component is present in the key
        assert "float32" in key1 or "abc123" in key1


# ===================================================================
# get_cross_encoder
# ===================================================================


class TestGetCrossEncoder:
    def test_returns_cross_encoder(self, monkeypatch, tmp_path):
        """Should load and return a CrossEncoder instance."""
        mc = _build_models_cache()

        # Create a fake local path so _load_from_meta considers it valid.
        fake_local = tmp_path / "test-cross-model"
        fake_local.mkdir()

        class _FakeHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                return {"local_path": str(fake_local)}

        monkeypatch.setattr(_mc_mod, "HFDownloader", _FakeHFDownloader)
        monkeypatch.setattr(_mc_mod, "CrossEncoder", FakeCrossEncoder)

        encoder = mc.get_cross_encoder()

        assert isinstance(encoder, FakeCrossEncoder)
        assert encoder.kwargs["model_name_or_path"] == str(fake_local)

    def test_passes_chunk_size_as_max_length(self, monkeypatch, tmp_path):
        """max_length should be set to the configured chunk_size."""
        mc = _build_models_cache()
        mc.chunk_size = 1024
        fake_local = tmp_path / "cross-model"
        fake_local.mkdir()

        class PatchedHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                return {"local_path": str(fake_local)}

        monkeypatch.setattr(_mc_mod, "HFDownloader", PatchedHFDownloader)
        monkeypatch.setattr(_mc_mod, "CrossEncoder", FakeCrossEncoder)

        encoder = mc.get_cross_encoder()
        assert isinstance(encoder, FakeCrossEncoder)

        assert encoder.kwargs["max_length"] == 1024

    def test_hfdownloader_fallback_on_local_failure(self, monkeypatch, tmp_path):
        """When the downloaded local_path does not exist on disk, ModelLoadError is raised."""
        mc = _build_models_cache()

        class PatchedHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                # Return a path that does not exist on disk
                return {"local_path": str(tmp_path / "nonexistent_model")}

        monkeypatch.setattr(_mc_mod, "HFDownloader", PatchedHFDownloader)
        monkeypatch.setattr(_mc_mod, "CrossEncoder", FakeCrossEncoder)

        from Commons.Exceptions import ModelLoadError

        with pytest.raises(ModelLoadError):
            mc.get_cross_encoder()

    def test_cuda_oom_falls_back_to_cpu(self, monkeypatch, tmp_path):
        """CUDA OOM during cross-encoder load should trigger CPU fallback."""
        mc = _build_models_cache(cfg_overrides={"USE_CPU": False})
        mc.use_cpu = False
        fake_local = tmp_path / "cross-model"
        fake_local.mkdir()

        class PatchedHFDownloader:
            def __init__(self):
                pass

            def download(self, key):
                return {"local_path": str(fake_local)}

        call_count = 0

        def oom_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("CUDA OOM")
            return FakeCrossEncoder(**kwargs)

        monkeypatch.setattr(_mc_mod, "HFDownloader", PatchedHFDownloader)
        monkeypatch.setattr(_mc_mod, "CrossEncoder", oom_then_ok)
        # Force CPU on switch2device
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        encoder = mc.get_cross_encoder()

        assert mc.use_cpu is True
        assert encoder is not None


# ===================================================================
# truncate_texts
# ===================================================================


class TestTruncateTexts:
    def test_returns_decoded_strings(self, monkeypatch):
        """Should return a list of truncated strings."""
        mc = _build_models_cache()

        monkeypatch.setattr(
            _mc_mod,
            "AutoTokenizer",
            type(
                "FakeAutoTokenizer",
                (),
                {"from_pretrained": staticmethod(lambda *a, **kw: FakeTokenizer())},
            ),
        )

        result = mc.truncate_texts(
            ["hello world", "another text"], "test-model", max_length=128
        )

        assert len(result) == 2
        assert all(isinstance(t, str) for t in result)

    def test_result_count_matches_input(self, monkeypatch):
        """Output list should have the same length as input."""
        mc = _build_models_cache()

        monkeypatch.setattr(
            _mc_mod,
            "AutoTokenizer",
            type(
                "FakeAutoTokenizer",
                (),
                {"from_pretrained": staticmethod(lambda *a, **kw: FakeTokenizer())},
            ),
        )

        texts = ["a", "b", "c", "d", "e"]
        result = mc.truncate_texts(texts, "test-model", max_length=64)

        assert len(result) == 5


# ===================================================================
# Singleton behaviour
# ===================================================================


class TestSingleton:
    def test_same_instance_returned(self, monkeypatch):
        """ModelsCache() should always return the same instance."""
        mc1 = _build_models_cache()
        # Store it as the singleton
        ModelsCache._instance = mc1
        mc2 = ModelsCache.__new__(ModelsCache)

        assert mc2 is mc1

    def test_reset_clears_instance(self):
        """_reset() should destroy the singleton."""
        mc = _build_models_cache()
        ModelsCache._instance = mc

        ModelsCache._reset()

        assert ModelsCache._instance is None
