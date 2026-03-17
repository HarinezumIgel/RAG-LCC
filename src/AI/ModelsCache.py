# Helpers/ModelsCache.py
"""
Centralised model cache for HuggingFace embeddings, quantized
SentenceTransformers, and CrossEncoder models.

Extracted from AIHelpers so that any component can load / reuse
models without pulling in the full AIHelpers dependency graph.
"""

import os
from typing import Any, Dict, List, Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer, models
from transformers import AutoTokenizer
from transformers import logging as hf_logging

from Commons.Exceptions import ModelLoadError
from Commons.SingletonMixin import SingletonMixin
from Compliance.HFDownloader import HFDownloader
from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class ModelsCache(SingletonMixin):
    """Thread-safe singleton that owns every cached model instance.

    Public API
    ----------
    - :py:meth:`get_hf_embeddings`       – returns (cached) ``HuggingFaceEmbeddings``
    - :py:meth:`invalidate_hf_embeddings` – evicts cached embeddings
    - :py:meth:`load_quantized_model`     – returns (cached) quantized ``SentenceTransformer``
    - :py:meth:`get_cross_encoder`        – returns a ``CrossEncoder``
    - :py:meth:`truncate_texts`           – tokeniser-based text truncation
    - :py:meth:`switch2device`            – device / dtype selection
    - :py:meth:`fallback_to_cpu`          – persist GPU → CPU fallback
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # --- injected deps ---
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()

        # --- caches ---
        self.hf_embeddings_cache: Dict[str, HuggingFaceEmbeddings] = {}
        self.sentence_transformer_cache: List[Dict[str, SentenceTransformer]] = []

        # --- config values used for model loading ---
        self.cache_dir: str = self.cfg.get_str("_HF_HUB_CACHE")
        self.chunk_size: int = self.cfg.get_int(
            "_CHROMA_EMBED_AND_RETRIEVE_PARAMS.CHUNK_SIZE"
        )
        self.hf_hub_offline: str = os.environ.get("HF_HUB_OFFLINE", "1")
        self.use_cpu: bool = self.cfg.get_bool("USE_CPU")
        self.bits: int = self.cfg.get_int("EMBEDDER_BITS", 32)
        self.lastDeviceBitSize: int = 0

    # -----------------------------------------------------------------
    # GPU → CPU fallback
    # -----------------------------------------------------------------
    def fallback_to_cpu(self, reason: str) -> None:
        """Persist a GPU → CPU fallback.

        Called by:
          - switch2device()       (CUDA probe failure)
          - get_hf_embeddings()   (CUDA OOM)
          - get_cross_encoder()   (CUDA OOM)
          - Informer._fallback_to_cpu() performs the same steps externally.
        """
        self.use_cpu = True
        self.cfg.set("USE_CPU", True, force=True)
        self.cfg.set("EMBEDDER_BITS", 32, force=True)
        self.pretty.write("W", "HF", reason, color=ORANGE)

    # -----------------------------------------------------------------
    # Device selection
    # -----------------------------------------------------------------
    def switch2device(self) -> tuple[torch.device, str, torch.dtype, int]:
        """Select device for computation (CPU or CUDA), returning device info and dtype.

        Reads USE_CPU live from config so runtime changes (e.g. Informer
        fallback) are always respected regardless of call-order.
        """
        bits: int = self.cfg.get_int("EMBEDDER_BITS", 32)
        target_dtype: torch.dtype = self.helpers.bit_to_dtype(bits)

        use_cpu: bool = self.use_cpu or self.cfg.get_bool("USE_CPU")

        if use_cpu or not torch.cuda.is_available():
            return torch.device("cpu"), "cpu", target_dtype, 0

        if torch.cuda.device_count() == 0:
            return torch.device("cpu"), "cpu", target_dtype, -1

        try:
            torch.zeros(1, device="cuda")
        except RuntimeError:
            self.fallback_to_cpu("CUDA probe failed – falling back to CPU.")
            return torch.device("cpu"), "cpu", target_dtype, 0

        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
        target_dtype = torch.float16

        if self.lastDeviceBitSize != bits and self.lastDeviceBitSize != 0:
            self.pretty.write(
                "I",
                "Device",
                f"Found device: {device}, Device type: cuda, Current bits: {self.lastDeviceBitSize} Target bits: {bits}, Device index: {idx}",
            )
        self.lastDeviceBitSize = bits

        return device, "cuda", target_dtype, idx

    # -----------------------------------------------------------------
    # Quantized SentenceTransformer
    # -----------------------------------------------------------------
    def load_quantized_model(self, model_name: str) -> SentenceTransformer:
        """Load a quantized SentenceTransformer model, using cache if available."""
        device, device_type, target_dtype, _ = self.switch2device()
        hf_args: Dict[str, Any] = self.helpers.get_model_args("_EMBED")
        revision: str | None = hf_args.get("revision")

        cache_key: str = f"{model_name}_{revision}_{target_dtype}"
        for entry in self.sentence_transformer_cache:
            if cache_key in entry:
                return entry[cache_key]

        def _init_transformer(use_local_only: bool):
            return models.Transformer(
                model_name_or_path=model_name,
                cache_dir=self.cache_dir,
                model_args={
                    **({"revision": revision} if revision else {}),
                    "dtype": target_dtype,
                    "local_files_only": use_local_only,
                },
                config_args={
                    "local_files_only": use_local_only,
                },
                tokenizer_args={
                    "local_files_only": use_local_only,
                },
            )

        try:
            transformer = _init_transformer(True)
        except Exception as e:
            self.pretty.write(
                "W",
                "Embedder",
                f"Local load failed ({e}); requesting download via HFDownloader.",
            )
            HFDownloader().download("_MODELS._EMBED")
            hf_args = self.helpers.get_model_args("_EMBED")
            revision = hf_args.get("revision")
            cache_key = f"{model_name}_{revision}_{target_dtype}"
            try:
                transformer = _init_transformer(True)
            except Exception as retry_err:
                raise ModelLoadError(
                    f"Failed to load embedding model '{model_name}'"
                ) from retry_err

        if self.bits and device_type == "cpu" and target_dtype != torch.float32:
            transformer.auto_model = torch.quantization.quantize_dynamic(  # type: ignore[reportUnknownMemberType]
                transformer.auto_model, {torch.nn.Linear}, dtype=target_dtype  # type: ignore[reportUnknownMemberType]
            )
        pooling = models.Pooling(
            word_embedding_dimension=transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )

        model = SentenceTransformer(modules=[transformer, pooling], device=str(device))
        model.to(target_dtype).eval()

        self.sentence_transformer_cache.append({cache_key: model})
        return model

    # -----------------------------------------------------------------
    # HuggingFace Embeddings
    # -----------------------------------------------------------------
    def get_hf_embeddings(self) -> HuggingFaceEmbeddings:
        """Load HuggingFace embeddings, using cache if available."""
        model_args: Dict[str, Any] = self.helpers.get_model_args("_EMBED")
        model_name: str = model_args["model_name"]
        revision: str | None = model_args.get("revision")
        device, _, _, _ = self.switch2device()
        dtype: torch.dtype = self.helpers.bit_to_dtype(self.bits)

        # normalize device to stable string
        try:
            device_type = device.type
            device_index = getattr(device, "index", None)
            device_key = (
                f"{device_type}:{device_index}"
                if device_index is not None
                else device_type
            )
        except Exception:
            device_key = str(device)

        dtype_key = str(dtype)
        rev_key = revision if revision else "none"
        cache_key = f"{model_name}_{rev_key}_{device_key}_{dtype_key}"

        cached: HuggingFaceEmbeddings | None = self.hf_embeddings_cache.get(cache_key)
        if cached is not None:
            self.pretty.write(
                "I",
                "HF",
                f"Reusing cached embeddings for {model_name} rev='{revision}' device={device_key} dtype={dtype_key}",
            )
            return cached

        base_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "cache_folder": self.cache_dir,
            "encode_kwargs": {},
        }
        local_kwargs: dict[str, Any] = {
            "device": device,
            "local_files_only": True,
            "revision": revision,
        }

        try:
            self.pretty.write(
                "I",
                "HF",
                f"Try to load {model_name} revision '{revision}' key: {cache_key} from cache.",
            )
            embeddings = HuggingFaceEmbeddings(**base_kwargs, model_kwargs=local_kwargs)
        except torch.cuda.OutOfMemoryError:
            self.fallback_to_cpu(
                f"CUDA out of memory loading {model_name} – falling back to CPU."
            )
            device = torch.device("cpu")
            dtype = self.helpers.bit_to_dtype(32)
            cpu_kwargs: dict[str, Any] = {
                "device": device,
                "local_files_only": True,
                "revision": revision,
            }
            embeddings = HuggingFaceEmbeddings(**base_kwargs, model_kwargs=cpu_kwargs)
            cache_key = f"{model_name}_{rev_key}_cpu_{str(dtype)}"
        except Exception as e:
            self.pretty.write(
                "W",
                "HuggingFace embeddings",
                f"{type(e).__name__}: {e} – requesting download via HFDownloader.",
            )
            HFDownloader().download("_MODELS._EMBED")
            model_args = self.helpers.get_model_args("_EMBED")
            revision = model_args.get("revision")
            rev_key = revision if revision else "none"
            cache_key = f"{model_name}_{rev_key}_{device_key}_{dtype_key}"
            local_kwargs["revision"] = revision
            try:
                embeddings = HuggingFaceEmbeddings(
                    **base_kwargs, model_kwargs=local_kwargs
                )
            except Exception as retry_err:
                raise ModelLoadError(
                    f"Failed to load HuggingFace embeddings '{model_name}'"
                ) from retry_err

        try:
            embeddings._client.to(dtype)  # type: ignore[reportPrivateUsage, reportUnknownMemberType]
        except Exception:
            self.pretty.write(
                "W",
                "HuggingFace embeddings",
                f"Failed to convert embeddings client to {dtype}; leaving as-is",
            )

        self.hf_embeddings_cache[cache_key] = embeddings
        return embeddings

    def invalidate_hf_embeddings(
        self, model_name: Optional[str] = None, revision: Optional[str] = None
    ) -> None:
        """Invalidate cached embeddings. If model_name is None, clear entire cache."""
        if model_name is None:
            self.hf_embeddings_cache.clear()
            return
        keys_to_remove = [
            k
            for k in self.hf_embeddings_cache
            if k[0] == model_name and (revision is None or k[1] == (revision or ""))
        ]
        for k in keys_to_remove:
            del self.hf_embeddings_cache[k]

    # -----------------------------------------------------------------
    # CrossEncoder
    # -----------------------------------------------------------------
    def get_cross_encoder(self) -> CrossEncoder:
        """Load a CrossEncoder model, using cache and local files if possible."""
        args: Dict[str, Any] = self.helpers.get_model_args("_CROSS")
        model_name: str = args["model_name"]
        revision: str | None = args.get("revision")

        device, _, _, _ = self.switch2device()

        self.pretty.write(
            "I",
            "HF",
            f"Try to load cross-encoder '{model_name}' revision '{revision}' device={device} from cache.",
        )

        base_kwargs: dict[str, Any] = {
            "model_name_or_path": model_name,
            "device": device,
            "trust_remote_code": False,
            "tokenizer_kwargs": {"use_fast": True},
            "max_length": self.chunk_size,
            **({"revision": revision} if revision else {}),
        }

        local_kwargs: dict[str, Any] = {**base_kwargs, "local_files_only": True}

        prev_level: int = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            try:
                return CrossEncoder(**local_kwargs)
            except torch.cuda.OutOfMemoryError:
                self.fallback_to_cpu(
                    f"CUDA out of memory loading cross-encoder '{model_name}' – falling back to CPU."
                )
                cpu_kwargs: dict[str, Any] = {
                    **base_kwargs,
                    "device": torch.device("cpu"),
                    "local_files_only": True,
                }
                return CrossEncoder(**cpu_kwargs)
            except Exception:
                self.pretty.write(
                    "W",
                    "HF",
                    f"Local load failed for '{model_name}' revision: '{revision}' – requesting download via HFDownloader.",
                )
                HFDownloader().download("_MODELS._CROSS")
                args = self.helpers.get_model_args("_CROSS")
                revision = args.get("revision")
                base_kwargs = {
                    "model_name_or_path": model_name,
                    "device": device,
                    "trust_remote_code": False,
                    "tokenizer_kwargs": {"use_fast": True},
                    "max_length": self.chunk_size,
                    **({"revision": revision} if revision else {}),
                }
                local_kwargs = {**base_kwargs, "local_files_only": True}
                try:
                    return CrossEncoder(**local_kwargs)
                except Exception as retry_err:
                    raise ModelLoadError(
                        f"Failed to load cross-encoder '{model_name}' after download retry. Maybe the HF cache for this model is corrupted?"
                    ) from retry_err
        finally:
            hf_logging.set_verbosity(prev_level)

    # -----------------------------------------------------------------
    # Text truncation
    # -----------------------------------------------------------------
    def truncate_texts(
        self, texts: list[str], model_name: str, max_length: int, padding: bool = False
    ) -> list[str]:
        """Truncate and decode texts using a tokenizer for a given model."""
        tokenizer: Any = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # type: ignore[reportUnknownMemberType]
        enc: Any = tokenizer(  # type: ignore[reportUnknownVariableType]
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length" if padding else False,
            return_tensors=None,
        )
        return [
            tokenizer.decode(  # type: ignore[reportUnknownMemberType]
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for ids in enc["input_ids"]  # type: ignore[reportUnknownVariableType]
        ]
