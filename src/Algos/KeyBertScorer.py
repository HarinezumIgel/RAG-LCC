import time
from collections import OrderedDict
from typing import Any, List, Optional, cast

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class KeyBertScorer(SingletonMixin, ScorerBase):
    """
    Keyword detector using SBERT embeddings.
    Deterministic phrase ordering, defensive checks, and debug instrumentation.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        # singleton guard
        if self._initialized:
            return

        self._initialized = True
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.perf_logger: PerfLogger = PerfLogger()

        # phrases come from config key "BANNED"
        self.banlist_en: List[str] = self.helpers.get_banned_phrases_config_slot()
        self.embed_model_name: str = self.helpers.get_model_args("_ACTIVE_EMBED")[
            "MODEL"
        ]
        self.algo: str = self.cfg.get_str("_KEYBERT")

        # These are populated lazily by _ensure_ready() so that
        # AIHelpers.__init__ (which instantiates this scorer) does NOT
        # trigger a download before HFDownloader has run.
        self.embedder: HuggingFaceEmbeddings | None = None
        self.device: torch.device | None = None
        self.device_type: str = ""
        self.target_dtype: Any = None
        self.device_index: int = 0

        # runtime state
        self.num_queries: Optional[int] = None
        self.k: Optional[int] = None
        self.topk_scores: Optional[torch.Tensor] = None
        self.topk_idxs: Optional[torch.Tensor] = None
        self.threshold: float = 0.0

        # frozen phrase ordering and matrix built in build_keybert_cache
        self.phrases: List[str] = []
        self.phrase_embedding_cache_tensor: "OrderedDict[str, torch.Tensor]" = (
            OrderedDict()
        )
        self.pharase_cache_matrix: Optional[torch.Tensor] = None

        # build cache lazily on first verify()
        self.ready: bool = False

    def _ensure_ready(self) -> None:
        """Lazily load the model and pre-compute caches on first use."""
        if self.ready:
            return
        self.ready = True
        self.embedder = self.models_cache.get_hf_embeddings()
        self.device, self.device_type, self.target_dtype, self.device_index = (
            self.models_cache.switch2device()
        )
        self.build_keybert_cache()

    def build_keybert_cache(self) -> None:
        """
        Build an OrderedDict cache and a frozen phrase matrix for stable indexing.
        This ensures consistent index -> phrase mapping across runs.
        """
        phrases: list[str] = [p for p in self.banlist_en if p]

        self.perf_logger.log(
            "KeyBertScorer.build_keybert_cache",
            "scorer",
            f"start cache build n={len(phrases)}",
        )
        _t0 = time.perf_counter()
        # Use OrderedDict to preserve insertion order and make intent explicit
        cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        sbert_model: Any = self.models_cache.load_quantized_model(self.embed_model_name)

        for phrase in phrases:
            if phrase in cache:
                continue
            try:
                emb_tensor: torch.Tensor = self.tensorHelpers.encode_to_tensor(
                    sbert_model, phrase
                )  # returns (D,) tensor
                t_norm: torch.Tensor = self.tensorHelpers.to_cpu_tensor(
                    emb_tensor
                )  # optional: keep existing helper usage
                t_norm = self.tensorHelpers.normalize_cpu(t_norm)  # ensure normalized
                cache[phrase] = t_norm

                shape_str: str = str(tuple(t_norm.shape))
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D",
                        "Cache build Keybert Scorer",
                        f"Cached {phrase:<40} (shape={shape_str:<14} len={t_norm.numel():<6})",
                    )

            except Exception as e:
                self.pretty.write(
                    "E",
                    "Cache build Keybert Scorer",
                    f"Failed to cache '{phrase}': {e}",
                )
                continue

        # Freeze ordering and build matrix once
        self.phrase_embedding_cache_tensor = cache
        self.phrases = list(cache.keys())

        if self.phrases:
            try:
                self.pharase_cache_matrix = torch.stack(
                    [cache[p] for p in self.phrases], dim=0
                ).to(device=self.device, dtype=self.target_dtype)
            except Exception as e:
                self.pretty.write(
                    "E",
                    "Cache build Keybert Scorer",
                    f"Failed to build phrase matrix: {e}",
                )
                self.pharase_cache_matrix = None

        self.pretty.write(
            "O",
            "Cache build Keybert Scorer",
            f"Built KeyBert embeddings cache with {len(cache)} entries",
        )
        self.perf_logger.log(
            "KeyBertScorer.build_keybert_cache",
            "scorer",
            f"stop  cache build n={len(cache)} elapsed={time.perf_counter() - _t0:.3f}s",
        )

    def _verify_impl(
        self, embedding_to_compare_with: Any, stage: str
    ) -> List[ComplianceAlgoResult]:
        """
        Compute top-k phrase matches for the provided embedding(s).
        Returns a list of ComplianceAlgoResult objects ranked by similarity score.
        """
        self._ensure_ready()
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        self.threshold = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD"
        )
        self.threshold_min = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD_MIN"
        )
        self.top_k = self.cfg.get_int(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.TOP_K"
        )
        self.helpers.require_set(
            threshold=self.threshold, threshold_min=self.threshold_min, top_k=self.top_k
        )

        # defensive checks
        if not self.phrase_embedding_cache_tensor or embedding_to_compare_with is None:
            if DebugHelper.check(self.cfg, 40):
                self.pretty.write(
                    "D", "KeyBERTChk", "Empty phrase cache or missing query embedding"
                )
            return []

        if self.pharase_cache_matrix is None:
            self.pretty.write(
                "E", "KeyBERTChk", "Phrase cache matrix missing; rebuild required"
            )
            return []

        pvecs: torch.Tensor = self.pharase_cache_matrix  # shape (P, D)
        num_phrases: int
        embed_dim: int
        num_phrases, embed_dim = pvecs.shape

        # Normalize phrase vectors for cosine similarity
        _pvec_norms: torch.Tensor = pvecs.norm(p=2, dim=1, keepdim=True)  # type: ignore[reportUnknownMemberType]
        pvecs = cast(torch.Tensor, pvecs / (_pvec_norms + 1e-12))

        # Normalize input to tensor
        q: torch.Tensor = self.tensorHelpers.to_tensor(embedding_to_compare_with).to(
            self.device, self.target_dtype
        )
        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D",
                "KeyBERTChk",
                f"verify: q shape={tuple(q.shape)}, cache D={embed_dim}",
            )

        # Accept either a single vector (D,) or batch (N, D)
        if q.dim() == 1 and q.numel() == embed_dim:
            q_rows: torch.Tensor = q.view(1, embed_dim)
        elif q.dim() == 2 and q.shape[1] == embed_dim:
            q_rows = q
        else:
            self.pretty.write(
                "E",
                "KeyBERTChk",
                f"Dim mismatch: query shape={tuple(q.shape)} vs cache dim={embed_dim}",
            )
            return []

        # L2 normalize rows
        _q_norms: torch.Tensor = q_rows.norm(p=2, dim=1, keepdim=True)  # type: ignore[reportUnknownMemberType]
        q_norm: torch.Tensor = cast(torch.Tensor, q_rows / (_q_norms + 1e-12))

        # Similarity matrix: (P, N)
        sims_mat: torch.Tensor = torch.matmul(pvecs, q_norm.t())  # (P, N)

        # Choose top_k (cap at P)
        self.k = min(self.top_k, num_phrases)

        # For each query column, get top-k phrase indices and scores
        # sims_mat is (P, N) so we want topk over dim=0 -> returns (k, N)
        try:
            self.topk_scores, self.topk_idxs = sims_mat.topk(
                k=self.k, dim=0, largest=True, sorted=True
            )
        except Exception as e:
            self.pretty.write("E", "KeyBERTChk", f"topk failed: {e}")
            return []

        self.num_queries = q_norm.shape[0]

        # Instrumentation: full mapping for first query and global sanity
        if DebugHelper.check(self.cfg, 40):
            num_phrases = len(self.phrases)
            min_idx = int(self.topk_idxs.min().item())
            max_idx = int(self.topk_idxs.max().item())
            self.pretty.write(
                "D",
                "KeyBERTChk",
                f"P={num_phrases} min_idx={min_idx} max_idx={max_idx}",
            )
            assert (
                0 <= min_idx < num_phrases and 0 <= max_idx < num_phrases
            ), f"topk idx out of range: 0..{num_phrases-1} vs {min_idx}..{max_idx}"

            # print the actual phrases for the topk indices of the first query
            if self.num_queries > 0:
                idxs_first_q = [int(self.topk_idxs[r, 0].item()) for r in range(self.k)]
                mapped_phrases = [self.phrases[i] for i in idxs_first_q]
                self.pretty.write("D", "KeyBERTChk", f"topk idxs (q0)={idxs_first_q}")
                self.pretty.write(
                    "D", "KeyBERTChk", f"topk phrases (q0)={mapped_phrases}"
                )

        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D",
                "KeyBertWordDetect",
                f"Running with: top_k: {self.top_k} "
                f"threshold: {self.threshold} threshold min: {self.threshold_min}",
            )

        return self.return_algo_result()

    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        """
        Convert topk tensors into a list of ComplianceAlgoResult dataclass instances.
        Instrumented with assertions and debug logs to ensure index -> phrase mapping is correct.
        """
        results: List[ComplianceAlgoResult] = []

        P: int = len(self.phrases)

        assert self.num_queries is not None and self.k is not None
        assert self.topk_scores is not None and self.topk_idxs is not None

        for qi in range(self.num_queries):
            for rank in range(self.k):
                # extract score and index safely
                score_tensor: torch.Tensor = self.topk_scores[rank, qi]
                idx_tensor: torch.Tensor = self.topk_idxs[rank, qi]

                # defensive conversions
                try:
                    score: float = float(score_tensor.item())
                    idx: int = int(idx_tensor.item())
                    if score < self.threshold_min:
                        continue
                except Exception as e:
                    self.pretty.write(
                        "E", "KeyBERTChk", f"Failed to convert topk tensors: {e}"
                    )
                    continue

                # bounds check
                if not (0 <= idx < P):
                    self.pretty.write(
                        "E",
                        "KeyBERTChk",
                        f"Index out of range idx={idx} P={P} qi={qi} rank={rank}",
                    )
                    continue

                phrase: str = self.phrases[idx]

                # debug log each appended result when verbosity is high
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D",
                        "KeyBERTChk",
                        f"qi={qi} rank={rank} idx={idx} phrase={phrase!r} score={score:.4f}",
                    )

                results.append(
                    ComplianceAlgoResult(
                        algo=self.algo,
                        phrase=phrase,
                        score=score,
                        threshold=self.threshold,
                        detail=f"detail: keybert={score:.3f} rank: {rank + 1} query_index: {qi}",
                    )
                )
        return results
