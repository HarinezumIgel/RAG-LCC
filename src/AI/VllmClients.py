# AI/VllmClients.py
"""
Remote controls for embedding and reranking models -- not a chat client.

Think of this module as a delivery service: RAG-LCC sends text to a model
server, the server does the expensive model computation, and these classes
bring back numbers. They do NOT load model weights on the RAG-LCC machine.
The server operator must install and serve the models separately.

There are two jobs:

1. Embedding: turn text into a list of numbers (a "vector") used to find
   related text. For example, "hedgehog food" and "hedgehogs eat insects"
   should have related vectors. The numbers below are illustrative only::

       RAG-LCC                         remote model server
       embed_query("hedgehog food")
           |---- POST /embeddings ----> embedding model
           |<--- [0.12, -0.34, ...] ---|
           |
           +--> vector search / compliance comparisons / keyword extraction

2. Reranking: score how relevant each retrieved passage is to a query::

       query: "What do hedgehogs eat?"
       passages: ["They eat insects.", "The train leaves at noon."]
           |---- POST /score or /rerank ----> reranker model
           |<--- one score per passage -----|
           |
           +--> RAG-LCC ranks/selects evidence before answer generation

   A higher score means more relevance for ranking, not a guarantee of
   factual correctness. This client preserves the input passage order when
   returning scores, even if the server sends results in ranked order.

Why wrap the HTTP calls?
The rest of the application already knows methods such as ``embed_query``
and ``predict`` from local model objects. These adapters expose compatible
methods, so callers do not need to implement network requests themselves:

- ``VllmEmbeddings``: ``embed_documents`` / ``embed_query`` call
  ``/embeddings``. ``embed`` supports KeyBERT; ``encode`` supports existing
  SentenceTransformer-style consumers. Converting returned numbers to a
  local tensor does not run a local embedding model.
- ``VllmCrossEncoder``: ``predict(pairs, show_progress_bar=False)`` calls
  either ``/score`` (query/passage pairs) or ``/rerank`` (one query plus
  passages). Requests are split into batches to limit request sizes.
- ``_VllmClient``: shared HTTP request handling, timeout/error handling,
  authentication-header setup, and optional diagnostic logging.

How it is selected:
``ModelsCache`` chooses these adapters when ``_ACTIVE_ENDPOINT = "vllm"``.
With ``"ollama"``, embedding/reranking uses local Hugging Face models instead;
this module does not send embeddings to Ollama. Model licence verification
remains separate from inference and is still required. Local tokenizer files
may still be used by ingestion to truncate text.

Server setup sketch (commands depend on the installed vLLM version):
Direct vLLM normally serves one model per server process, for example::

    vllm serve BAAI/bge-m3 --runner pooling --port 8001
    vllm serve BAAI/bge-reranker-v2-m3 --runner pooling --port 8002

Server settings come from ``_MODELS.vllm._VLLM`` (API_KEY, REQUEST_TIMEOUT,
...), superseded by the ``"VLLM"`` slot of the active embedder / cross-encoder
in ``Config_Models.py``. Without an own ``BASE_URL`` in that slot the client
uses the ``_VLLM`` server root (its chat URL without ``/chat/completions``).
For separate servers set the base URL per model, for example
``"VLLM": {"MODEL": "BAAI/bge-m3", "BASE_URL": "http://host:8001/v1"}``;
this client appends ``/embeddings``, ``/score``, or ``/rerank``.
Do not put the full chat-completions URL into a model's ``BASE_URL``.

Through LiteLLM, both can instead point to the proxy base URL, for example
``http://host:4000/v1``. The request model name is then the configured
LiteLLM ``model_name`` alias (for example ``bge-m3-embedder``), not necessarily
the Hugging Face repository name. Use ``API_STYLE="rerank"`` when the proxy
exposes the Cohere-style rerank API.
"""

from typing import Any, Callable

import numpy as np
import requests
import torch
from keybert.backend import \
    BaseEmbedder  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from Commons.Exceptions import ModelLoadError

# (stage, purpose, model) -> None; used to emit "Call LLM"-style trace lines.
CallLogger = Callable[[str, str, str], None]


class _VllmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        call_logger: CallLogger | None = None,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: float = timeout
        self._call_logger: CallLogger | None = call_logger
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _log_call(self, stage: str, purpose: str) -> None:
        if self._call_logger is not None:
            try:
                self._call_logger(stage, purpose, self.model)
            except Exception:
                pass

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.post(
                url, json=payload, headers=self._headers, timeout=self.timeout
            )
            resp.raise_for_status()
            data: Any = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise ModelLoadError(
                f"vLLM request to {url} (model '{self.model}') failed: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ModelLoadError(f"vLLM {url} returned an unexpected payload.")
        return data  # type: ignore[reportUnknownVariableType]


class VllmEmbeddings(_VllmClient, BaseEmbedder):
    """Embeddings from a vLLM server (OpenAI-compatible ``/embeddings``)."""

    def __init__(self, *args: Any, batch_size: int = 64, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size: int = max(1, batch_size)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._log_call(
            "Encode",
            f"Embedding {len(texts)} text(s) via {self.base_url}/embeddings",
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            data = self._post("embeddings", {"model": self.model, "input": batch})
            items: list[dict[str, Any]] = sorted(
                data.get("data", []), key=lambda d: d.get("index", 0)
            )
            if len(items) != len(batch):
                raise ModelLoadError(
                    f"vLLM returned {len(items)} embeddings for {len(batch)} inputs."
                )
            vectors.extend([[float(x) for x in it["embedding"]] for it in items])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed(
        self, documents: list[str] | NDArray[np.str_], verbose: bool = False
    ) -> NDArray[np.float32]:
        """KeyBERT backend; candidate words may arrive as a NumPy array."""
        return np.asarray(self.embed_documents(list(documents)), dtype=np.float32)

    def encode(
        self,
        sentences: str | list[str],
        *,
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        device: str | None = None,
    ) -> NDArray[np.float32] | torch.Tensor:
        single = isinstance(sentences, str)
        texts = [sentences] if isinstance(sentences, str) else sentences
        vectors = self.embed(texts)
        result = vectors[0] if single else vectors
        if convert_to_tensor:
            return torch.as_tensor(result, dtype=torch.float32, device=device or "cpu")
        return result

    def get_sentence_embedding_dimension(self) -> int:
        return len(self.embed_query("embedding dimension"))


class VllmCrossEncoder(_VllmClient):
    """Cross-encoder scoring from a vLLM server or a LiteLLM proxy.

    ``api_style="score"``  -> vLLM ``/score`` (text_1/text_2 pairs).
    ``api_style="rerank"`` -> Cohere-style ``/rerank`` (LiteLLM proxy, also
    served by vLLM); one request per distinct query.
    """

    def __init__(
        self,
        *args: Any,
        batch_size: int = 64,
        api_style: str = "score",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size: int = max(1, batch_size)
        self.api_style: str = (api_style or "score").strip().lower()

    def predict(
        self, pairs: list[tuple[str, str]], show_progress_bar: bool = False
    ) -> list[float]:
        endpoint = "rerank" if self.api_style == "rerank" else "score"
        self._log_call(
            "Cross-encode",
            f"Scoring {len(pairs)} (query, passage) pair(s) via {self.base_url}/{endpoint}",
        )
        if self.api_style == "rerank":
            return self._predict_rerank(pairs)
        return self._predict_score(pairs)

    def _predict_rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = [0.0] * len(pairs)
        positions_by_query: dict[str, list[int]] = {}
        for pos, (query, _) in enumerate(pairs):
            positions_by_query.setdefault(query, []).append(pos)

        for query, positions in positions_by_query.items():
            for start in range(0, len(positions), self.batch_size):
                batch_pos = positions[start : start + self.batch_size]
                data = self._post(
                    "rerank",
                    {
                        "model": self.model,
                        "query": query,
                        "documents": [pairs[p][1] for p in batch_pos],
                        "top_n": len(batch_pos),
                        "return_documents": False,
                    },
                )
                results: list[dict[str, Any]] = data.get("results", [])
                if len(results) != len(batch_pos):
                    raise ModelLoadError(
                        f"Rerank returned {len(results)} scores for {len(batch_pos)} documents."
                    )
                for item in results:
                    scores[batch_pos[int(item["index"])]] = float(
                        item["relevance_score"]
                    )
        return scores

    def _predict_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            data = self._post(
                "score",
                {
                    "model": self.model,
                    "text_1": [q for q, _ in batch],
                    "text_2": [d for _, d in batch],
                },
            )
            items: list[dict[str, Any]] = sorted(
                data.get("data", []), key=lambda d: d.get("index", 0)
            )
            if len(items) != len(batch):
                raise ModelLoadError(
                    f"vLLM returned {len(items)} scores for {len(batch)} pairs."
                )
            scores.extend(float(it["score"]) for it in items)
        return scores
