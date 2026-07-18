import time
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import cosine_similarity

from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class CosineScorer(SingletonMixin, ScorerBase):
    """
    Cosine similarity keyword detector using normalized tensors.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.perf_logger: PerfLogger = PerfLogger()

        # banned phrases
        self.banned: List[str] = self.helpers.get_banned_phrases_config_slot()
        self.embed_model_name: str = self.helpers.get_model_args("_ACTIVE_EMBED")[
            "MODEL"
        ]
        self.algo: str = self.cfg.get_str("_COSINE")

        self.device: torch.device | None = None

        # Embedding caches for performance (tensor and list formats)
        self.phrase_embedding_cache_tensor: Dict[str, torch.Tensor] = {}
        self.phrase_embedding_cache: Dict[str, List[float]] = {}

        self.phrase_vecs: list[tuple[str, torch.Tensor]] = []
        self.vec: Optional[torch.Tensor] = None
        self.threshold: float | None = None
        self.threshold_min: float | None = None

        # Heavy model loading is deferred until first verify() call
        # so that AIHelpers.__init__ (which instantiates this scorer)
        # does NOT trigger a download before HFDownloader has run.
        self.ready: bool = False

    def _ensure_ready(self) -> None:
        """Lazily load the model and pre-compute caches on first use."""
        if self.ready:
            return
        self.ready = True
        self.device, _, _, _ = self.models_cache.switch2device()
        self.build_cosine_cache()

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate cosine similarity between two tensors."""
        return cosine_similarity(a, b, dim=0).item()

    def build_cosine_cache(self) -> Dict[str, torch.Tensor]:
        """Build embedding cache for all banned phrases."""
        phrases: List[str] = self.banned or []
        if not phrases:
            self.pretty.write(
                "E", "Cosine Scorer", "No banned phrases configured; cache empty."
            )
            return {}

        self.perf_logger.log(
            "CosineScorer.build_cosine_cache", f"start cache build n={len(phrases)}"
        )
        _t0 = time.perf_counter()
        self.phrase_embedding_cache_tensor = self.build_phrase_cache(
            phrases,
            list_cache=self.phrase_embedding_cache,
        )
        self.perf_logger.log(
            "CosineScorer.build_cosine_cache",
            f"stop  cache build n={len(self.phrase_embedding_cache_tensor)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        self.pretty.write(
            "O",
            "Cosine Scorer",
            f"Built cache for {len(self.phrase_embedding_cache_tensor)} phrase(s).",
        )
        return self.phrase_embedding_cache_tensor

    def _verify_impl(
        self, embedding_to_compare_with: Any, stage: str = "PIPELINE_CHECK"
    ) -> List[ComplianceAlgoResult]:
        """
        Verify embedded text against banned phrases using cosine similarity thresholds.
        """
        self._ensure_ready()
        if embedding_to_compare_with is None:
            return []

        # Load compliance thresholds for this stage
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        self.threshold = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD"
        )
        self.threshold_min = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD_MIN"
        )
        self.helpers.require_set(
            threshold=self.threshold, threshold_min=self.threshold_min
        )

        # convert incoming embedding to tensor
        if isinstance(embedding_to_compare_with, torch.Tensor):
            vec: torch.Tensor = (
                embedding_to_compare_with.detach().cpu().float().view(-1)
            )
        else:
            vec = torch.tensor(embedding_to_compare_with, dtype=torch.float32).view(-1)

        # assume already normalized by your adapter
        self.vec = vec

        # collect phrase vectors
        self.phrase_vecs = []
        for phrase in self.banned:
            t: torch.Tensor | None = self.phrase_embedding_cache_tensor.get(phrase)
            if isinstance(t, torch.Tensor):
                self.phrase_vecs.append((phrase, t))

        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D",
                "Cosine Scorer",
                f"Running with threshold={self.threshold}, threshold_min={self.threshold_min}",
            )

        return self.return_algo_result()

    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        """Build result list for phrases matching the threshold."""
        results: List[ComplianceAlgoResult] = []
        assert self.vec is not None and self.threshold_min is not None

        for phrase, pvec in self.phrase_vecs:
            score: float = self._cosine_similarity(self.vec, pvec)
            if score >= self.threshold_min:
                results.append(
                    ComplianceAlgoResult(
                        algo=self.algo,
                        phrase=phrase,
                        score=score,
                        threshold=self.threshold,
                        detail=f"cosine={score:.3f}",
                    )
                )
        return results

    def build_phrase_cache(
        self,
        phrases: List[str],
        list_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, torch.Tensor]:

        if not phrases:
            self.pretty.write(
                "E", "Cache build Cosine Banned", "No phrases provided; cache empty."
            )
            return {}

        cache: Dict[str, torch.Tensor] = {}
        sbert_model: Any = self.models_cache.load_quantized_model(self.embed_model_name)
        for phrase in phrases:
            try:
                emb: torch.Tensor = self.tensorHelpers.encode_to_tensor(
                    sbert_model, phrase
                )
                emb = self.tensorHelpers.to_cpu_tensor(emb)  # already normalized

                cache[phrase] = emb

                if list_cache is not None:
                    list_cache[phrase] = emb.tolist()  # type: ignore[reportUnknownMemberType]

                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D",
                        "Cache build Cosine Banned",
                        f"Cached {phrase:<40} (shape={tuple(emb.shape)}, len={emb.numel()})",
                    )

            except Exception as e:
                self.pretty.write(
                    "E", "Cache build Cosine Banned", f"Failed to cache '{phrase}': {e}"
                )
                continue

        self.pretty.write(
            "O", "Cache build Cosine Banned", f"Built cache with {len(cache)} entries"
        )
        return cache
