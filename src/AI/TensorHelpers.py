# Helpers/TensorHelpers.py
from typing import Any, List

import numpy as np
import torch

from AI.ModelsCache import ModelsCache
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class TensorHelpers:
    """
    Small, focused utility that centralizes conversions and normalization
    between lists, numpy arrays, and torch tensors.
    Contains only tensor/embedding helpers — move any model-loading or
    algorithm-specific objects out of here.
    """

    def __init__(self) -> None:
        self.pretty: PrettyWriter = PrettyWriter()
        self.cfg: Config = Config()
        self.helpers: Helpers = Helpers()

        # Only keep configuration values actually used by tensor helpers
        self.bits: int = self.cfg.get_int("EMBEDDER_BITS", 32)

    # -------------------------
    # dtype helpers
    # -------------------------
    def dtype_from_bits(self, backend: str) -> Any:
        """
        Return a dtype object for the given bit width and backend ('torch' or 'numpy').
        """
        backend = str(backend).lower()
        if backend not in ("torch", "numpy"):
            raise ValueError("backend must be 'torch' or 'numpy'")

        np_map: dict[int, Any] = {
            8: __import__("numpy").uint8,
            16: __import__("numpy").float16,
            32: __import__("numpy").float32,
            64: __import__("numpy").float64,
        }
        torch_map: dict[int, Any] = {
            8: __import__("torch").uint8,
            16: __import__("torch").float16,
            32: __import__("torch").float32,
            64: __import__("torch").float64,
        }

        if self.bits not in np_map:
            raise ValueError("bits must be one of: 8, 16, 32, 64")

        return torch_map[self.bits] if backend == "torch" else np_map[self.bits]

    # -------------------------
    # Basic conversions
    # -------------------------
    def to_cpu_tensor(self, emb: Any) -> torch.Tensor:
        """
        Convert torch/numpy/list to 1-D CPU float32 torch.Tensor.
        """
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().float().view(-1)
        if isinstance(emb, np.ndarray):
            return torch.from_numpy(emb.astype("float32")).cpu().view(-1)  # type: ignore[reportUnknownMemberType]
        if isinstance(emb, (list, tuple)):
            return torch.tensor(list(emb), dtype=torch.float32).cpu().view(-1)  # type: ignore[reportUnknownArgumentType]
        raise TypeError(f"Unsupported embedding type: {type(emb)}")

    def to_tensor(self, vec: Any) -> torch.Tensor:
        """
        Convert input to a torch.Tensor on CPU float32.
        Preserves shape for 2D inputs. Reshapes 1D (D,) to (1, D).
        """
        if isinstance(vec, torch.Tensor):
            t = vec.detach().cpu().float()
        else:
            if isinstance(vec, np.ndarray):
                t = torch.from_numpy(vec).cpu().float()  # type: ignore[reportUnknownMemberType]
            elif isinstance(vec, (list, tuple)):
                t = torch.as_tensor(vec, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported embedding type: {type(vec)}")

        if t.dim() == 1:
            t = t.view(1, -1)
        return t

    def ensure_tensor(self, embeddings: Any) -> torch.Tensor:
        """
        Normalize embeddings to torch.Tensor for downstream processing.
        Accepts list, numpy.ndarray, or torch.Tensor.
        Ensures 2D shape (N, D) or reshapes 1D (D,) to (1, D).
        """
        if isinstance(embeddings, torch.Tensor):
            t = embeddings
        elif isinstance(embeddings, np.ndarray):
            t = torch.from_numpy(embeddings)  # type: ignore[reportUnknownMemberType]
        elif isinstance(embeddings, list):
            t = torch.tensor(embeddings)
        else:
            msg = f"Unsupported embedding type: {type(embeddings)}"
            self.pretty.write("E", "ensure_tensor", msg)
            raise TypeError(msg)

        if t.dim() == 1:
            t = t.view(1, -1)
        return t

    def switch_tensor_device(
        self, x: Any, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Move tensor or numpy array to specified device and dtype.
        """
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    # -------------------------
    # Normalization helpers
    # -------------------------
    def _ensure_1_dimension_cpu_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return 1-D CPU float32 tensor (shape (D,)).
        """
        if not isinstance(t, torch.Tensor):  # type: ignore[reportUnnecessaryIsInstance]
            t = torch.as_tensor(t, dtype=torch.float32)
        t = t.detach().cpu().float().view(-1)
        return t

    def normalize_cpu(self, t: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize a 1-D CPU tensor and return CPU float32 tensor.
        """
        t = self._ensure_1_dimension_cpu_tensor(t)
        norm = t.norm(p=2)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if norm.item() == 0.0:  # type: ignore[reportUnknownMemberType]
            return torch.zeros_like(t)
        return t / (norm + 1e-12)  # type: ignore[reportUnknownVariableType]

    def normalize_vector(self, v: Any) -> torch.Tensor:
        """
        Normalize a vector to L2 norm and convert to configured dtype.
        """
        dtype: torch.dtype = self.helpers.bit_to_dtype(
            self.bits
        )  # torch.float16 or torch.float32

        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().float().view(-1)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v.astype(np.float32)).cpu().view(-1)  # type: ignore[reportUnknownMemberType]
        elif isinstance(v, (list, tuple)):
            t = torch.tensor(list(v), dtype=torch.float32).cpu().view(-1)  # type: ignore[reportUnknownArgumentType]
        else:
            raise TypeError(f"Unsupported vector type: {type(v)}")

        norm: torch.Tensor = t.norm(p=2)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if norm.item() == 0.0:  # type: ignore[reportUnknownMemberType]
            return torch.zeros_like(t, dtype=dtype)

        t_norm: torch.Tensor = t / (norm + 1e-12)  # type: ignore[reportUnknownVariableType]

        # only convert if dtype differs
        if t_norm.dtype != dtype:  # type: ignore[reportUnknownMemberType]
            t_norm = t_norm.to(dtype)  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]

        return t_norm  # type: ignore[reportUnknownVariableType]

    def normalize_batch_embeddings(self, batch: Any) -> torch.Tensor:
        """
        Accepts 2-D tensor (N, D) or list/np and returns CPU float32 normalized (N, D).
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32)
        if batch.dim() == 1:
            batch = batch.view(1, -1)
        batch = batch.detach().cpu().float()
        norms: torch.Tensor = batch.norm(p=2, dim=1, keepdim=True)
        norms = norms.clamp_min(1e-12)
        return batch / norms

    # -------------------------
    # Embedding provider adapters
    # -------------------------
    def encode_to_tensor(self, model: Any, texts: str | List[str]) -> torch.Tensor:
        """
        Encode texts using a SentenceTransformer-like model and return normalized CPU tensor(s).
        Returns normalized CPU tensor (N, D) or (D,) for single input.
        """
        single: bool = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        # prefer tensor output
        try:
            emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        except TypeError:
            # fallback if model doesn't accept convert_to_tensor
            emb = model.encode(texts, show_progress_bar=False)

        # emb may be torch.Tensor or numpy array or list
        if isinstance(emb, torch.Tensor):
            batch = emb.detach().cpu().float()
        elif isinstance(emb, (list, tuple)):
            batch = torch.as_tensor(emb, dtype=torch.float32)
        else:
            # numpy
            batch = torch.from_numpy(np.asarray(emb, dtype="float32"))  # type: ignore[reportUnknownMemberType]

        # ensure 2D
        if batch.dim() == 1:
            batch = batch.view(1, -1)

        batch = self.normalize_batch_embeddings(batch)
        return batch[0] if single else batch

    def get_sentence_embedding_dimension(self, model: Any = None) -> int:
        """
        Return the embedding dimension (int).
        - If `model` is provided, try to call model.get_sentence_embedding_dimension().
        - Otherwise, try to use the AIHelpers singleton to load the configured embed model and query it.
        """
        # 1) If caller provided a model, prefer that
        if model is not None:
            try:
                # HFEmbedder has get_sentence_embedding_dimension()
                return int(model.get_sentence_embedding_dimension())
            except Exception:
                # try common alternatives
                try:
                    # SentenceTransformer-like
                    return int(model.get_sentence_embedding_dimension())
                except Exception:
                    pass

        # 2) Try to use ModelsCache to get the configured embed model
        try:
            mc = ModelsCache()
            embed_model_name: str = self.helpers.get_model_args("_EMBED")["MODEL"]
            model_instance: Any = mc.load_quantized_model(embed_model_name)
            return int(model_instance.get_sentence_embedding_dimension())
        except Exception as e:
            # 3) Last resort: try to infer from config or return a safe default (raise if you prefer)
            self.pretty.write(
                "W",
                "TensorHelpers",
                f"Could not determine embedding dimension automatically: {e}; returning 0",
            )
            return 0
