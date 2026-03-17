# Helpers/AIHelpers.py
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch import Tensor

from AI.LLMCaller import LLMCaller
from AI.ModelOutputAdapter import ModelOutput, ModelOutputAdapter
from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from Algos.BM25Scorer import BM25Scorer
from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ResultsForPrint
from Algos.CosineScorer import CosineScorer
from Algos.JaccardScorer import JaccardScorer
from Algos.KeyBertScorer import KeyBertScorer
from Algos.LevenshteinScorer import LevenshteinScorer
from Algos.RegexScorer import RegexScorer
from Commons.Exceptions import (ComplianceViolationError,
                                LLMComplianceCheckError)
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import GREEN, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Strategies.ClassifyHelper import ClassifyHelper


class AIHelpers(SingletonMixin):
    """Singleton for compliance checks, ensemble orchestration and embedding utilities.

    Model loading, caching, device selection and quantization are delegated to
    :class:`Helpers.ModelsCache.ModelsCache` — a dedicated singleton that can
    also be used directly by components that only need model access.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
        accumulator: "Accumulator | None" = None,
        models_cache: "ModelsCache | None" = None,
    ):
        # Initialize only once (singleton pattern)
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # Core helpers used across methods
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.accumulator: Accumulator = accumulator or Accumulator()
        self.helpers: Helpers = helpers or Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.llmCaller: LLMCaller = LLMCaller()
        self.modelOutputAdapter: ModelOutputAdapter = ModelOutputAdapter()
        # Tensor helpers centralize tensor/embedding conversions
        self.tensor_helpers: TensorHelpers = TensorHelpers()

        # Delegate all model loading / caching / device logic to ModelsCache
        self.models_cache: ModelsCache = models_cache or ModelsCache(
            cfg=self.cfg, pretty=self.pretty, helpers=self.helpers
        )

        self.is_streaming: bool = self.cfg.get_bool("OLLAMA_STREAMING_REQ")

        # Embedding / KeyBERT related config used by embedding helpers
        self.embed_model_name: str = self.helpers.get_model_args("_EMBED")["MODEL"]
        kb_cfg = self.helpers._get_keybert_config()  # type: ignore[reportPrivateUsage]
        self.top_n_first: int = kb_cfg["TOP_N_FIRST"]
        self.top_n_second: int = kb_cfg["TOP_N_SECOND"]

        # Algorithm names from config
        self.keybert: str = self.cfg.get_str("_KEYBERT")
        self.jaccard: str = self.cfg.get_str("_JACCARD")
        self.regex: str = self.cfg.get_str("_REGEX")
        self.bm25: str = self.cfg.get_str("_BM25")
        self.cosine: str = self.cfg.get_str("_COSINE")

        # Detectors / collectors used by check and ensemble methods
        self.regexDetector: RegexScorer = RegexScorer()
        self.jaccardScorer: JaccardScorer = JaccardScorer()
        self.cosineScorer: CosineScorer = CosineScorer()
        self.keyBertWordDetect: KeyBertScorer = KeyBertScorer()
        self.levenshteinScorer: LevenshteinScorer = LevenshteinScorer()
        self.bm25Scorer: BM25Scorer = BM25Scorer()
        self.classifyHelper: ClassifyHelper = ClassifyHelper()

        # Main embedder instance (loaded via ModelsCache)
        self.embedder: HuggingFaceEmbeddings = self.models_cache.get_hf_embeddings()

    # -----------------------------------------------------------------
    # Thin delegation helpers  (keep the old API surface for callers)
    # -----------------------------------------------------------------
    def load_quantized_model(self, model_name: str) -> SentenceTransformer:
        """Delegate to :py:meth:`ModelsCache.load_quantized_model`."""
        return self.models_cache.load_quantized_model(model_name)

    def truncate_texts(
        self, texts: list[str], model_name: str, max_length: int, padding: bool = False
    ) -> list[str]:
        """Delegate to :py:meth:`ModelsCache.truncate_texts`."""
        return self.models_cache.truncate_texts(texts, model_name, max_length, padding)

    def get_hf_embeddings(self) -> HuggingFaceEmbeddings:
        """Delegate to :py:meth:`ModelsCache.get_hf_embeddings`."""
        return self.models_cache.get_hf_embeddings()

    def invalidate_hf_embeddings(
        self, model_name: Optional[str] = None, revision: Optional[str] = None
    ) -> None:
        """Delegate to :py:meth:`ModelsCache.invalidate_hf_embeddings`."""
        self.models_cache.invalidate_hf_embeddings(model_name, revision)

    def get_cross_encoder(self) -> CrossEncoder:
        """Delegate to :py:meth:`ModelsCache.get_cross_encoder`."""
        return self.models_cache.get_cross_encoder()

    def fallback_to_cpu(self, reason: str) -> None:
        """Delegate to :py:meth:`ModelsCache.fallback_to_cpu`."""
        self.models_cache.fallback_to_cpu(reason)

    def switch2device(self) -> tuple[torch.device, str, torch.dtype, int]:
        """Delegate to :py:meth:`ModelsCache.switch2device`."""
        return self.models_cache.switch2device()

    # -------------------------
    # Prompt compliance helpers
    # -------------------------
    def check_user_prompt_with_filter_chain(
        self, user_prompt: str, stage: str
    ) -> Tuple[bool, List[ResultsForPrint]]:
        """
        Compliance hook: run checks on the resolved prompt before invoking LLM.
        Returns (human_review_required, phrase_table).
        """
        language: str = self.fileUtils.get_text_language(user_prompt, "ntlk")

        # embed_documents may return list[list[float]]; convert via TensorHelpers
        embeddings = self.embedder.embed_documents([user_prompt])
        # normalize/convert to tensor using TensorHelpers
        emb_tensor: Tensor = self.tensor_helpers.to_tensor(
            embeddings[0]
        )  # <-- normalization point

        human_review, _, phrase_table = self.run_ensemble_checks(
            user_prompt,
            language,
            stage=stage,
            accumulate=False,
            require_keybert=False,
            embedding=emb_tensor,
        )
        if human_review:
            self.pretty.write(
                "E",
                "CheckPrompt",
                f"⚠️   User provided prompt check failed (Filter chain).",
                color=RED,
            )
            return True, phrase_table
        return False, phrase_table

    def is_not_compliant_prompt(self, decision: ModelOutput, model: str) -> bool:
        """
        Determine if a prompt is not compliant based on model output.
        """
        if decision.decision == "error":
            self.pretty.write(
                "E",
                "CheckPrompt",
                f"⚠️  Compliance parsing error; treating as NOT compliant.",
                color=RED,
            )
            return True

        if decision.decision == "block":
            self.pretty.write(
                "E",
                "CheckPrompt",
                f"⚠️  Provided prompt is considered NOT compliant by: {model}. Reason: {decision.reason}",
                color=RED,
            )
            return True

        self.pretty.write(
            "O",
            "CheckPrompt",
            f"Provided prompt is considered compliant by: {model}. Reason: {decision.reason}",
            color=GREEN,
        )
        return False

    def check_provided_prompt(
        self,
        prompt: str,
        llm_model: str,
        temperature: float,
        top_k: int,
        top_p: float,
        max_output_tokens: int,
        answer_is_json: bool,
        use_ollama_gpu: bool,
        template_name: str,
        stage: str,
        context_size: int | None = None,
    ) -> None:
        """
        Run compliance check on provided prompt using LLM.
        Raises ComplianceViolationError if not compliant.
        """
        handler = self.llmCaller.make_on_chunk(
            temperature, max_output_tokens, top_k, top_p
        )
        # for cell in handler.__closure__: print(cell.cell_contents)
        llm_result = self.llmCaller.call_llm(
            model=llm_model,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            answer_is_json=answer_is_json,
            use_ollama_gpu=use_ollama_gpu,
            template_name=template_name,
            on_chunk=handler,
            streaming=self.is_streaming,
            stage=stage,
            context_size=context_size,
        )

        # handle errors
        if "error" in llm_result:
            self.pretty.write("E", "LLM", f"LLM error: {llm_result['error']}")
            raise LLMComplianceCheckError(
                f"LLM compliance check failed: {llm_result['error']}"
            )

        decision: ModelOutput = self.modelOutputAdapter.interpret(
            llm_result, llm_model, is_compliance=True, is_streaming=self.is_streaming
        )
        if self.is_not_compliant_prompt(decision, llm_model):
            raise ComplianceViolationError(
                "Prompt compliance check determined input is not compliant"
            )

    # -------------------------
    # Ensemble / detectors orchestration
    # -------------------------
    def run_ensemble_checks(
        self,
        cleaned_text: str,
        language: str,
        stage: str,
        accumulate: bool = False,
        require_keybert: bool = False,
        embedding: Any = None,
    ) -> tuple[bool, Tensor, list[ResultsForPrint]]:
        """
        Run ensemble compliance checks using configured algorithms.
        Returns (human_review_required, keyword_embeddings, phrase_table).
        """
        results: List[Any] = []
        algo_results: List[ComplianceAlgoResult] = []
        regex_results: List[ComplianceAlgoResult] = []
        keyword_embeddings: Tensor = torch.empty(0)
        filtered: List[Any] = []
        human_review: bool = False

        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        raw: Any = self.cfg.get(f"{compliance_config_slot}.PIPELINE.ALGOS_TO_PROCESS")

        algos_to_process: OrderedDict[str, bool] = self.helpers.make_ordered_dict(raw)

        if require_keybert and not algos_to_process.get(self.keybert, False):
            self.pretty.write(
                "E",
                "Keybert missing",
                "Keybert must be enabled in algos to process if keybert scores are requested. See flag require_keybert=True",
            )
            raise RuntimeError("Keybert must be enabled when require_keybert=True")

        algos_not_enabled: int = 0
        for name, enabled in algos_to_process.items():
            if not enabled:
                algos_not_enabled += 1
                continue
            try:
                algo_results.clear()
                if name == self.regex:
                    regex_results = self.regexDetector.verify(
                        cleaned_text, language, stage
                    )
                elif name == self.jaccard:
                    algo_results = self.jaccardScorer.verify(
                        cleaned_text, language, stage
                    )
                elif name == self.bm25:
                    algo_results = self.bm25Scorer.verify(cleaned_text, language, stage)
                elif name == self.cosine:
                    # requires embedding
                    if embedding is None:
                        embedding = self.embedder.embed_documents([cleaned_text])[0]
                    # normalize via TensorHelpers (canonical dtype/shape)
                    embedding = self.tensor_helpers.normalize_vector(
                        embedding
                    )  # <-- normalization point
                    algo_results = self.cosineScorer.verify(embedding, stage)
                elif name == self.keybert:
                    if embedding is None:
                        _, keyword_embeddings = (
                            self.classifyHelper.double_keybert_with_weights(
                                cleaned_text,
                                self.top_n_first,
                                self.top_n_second,
                                self.embed_model_name,
                                embedding,
                            )
                        )
                    else:
                        keyword_embeddings = embedding
                    try:
                        algo_results = self.keyBertWordDetect.verify(
                            keyword_embeddings, stage
                        )
                    except Exception:
                        algo_results = []
                else:
                    algo_results = []
            except Exception as e:
                self.pretty.write("E", "run_ensemble_checks", f"{name} failed: {e}")
                raise

            if algo_results:
                results.extend(algo_results)

        levenshtein_results: List[ComplianceAlgoResult] = self.levenshteinScorer.verify(
            regex_results, stage
        )
        algo_results = self.merge_algo_results(
            regex_results, levenshtein_results, merged_algo_name=self.regex
        )

        if algo_results:
            results.extend(algo_results)
        if algos_not_enabled == len(algos_to_process):
            print("\n")
            self.pretty.write(
                "W",
                "run_ensemble_checks",
                f"No check algorithm for {stage} defined !!!",
            )
            return (human_review, keyword_embeddings, filtered)
        human_review, filtered = self.accumulator.add_results(results, stage)
        if accumulate is True:
            # Accumulate path (RAGLoad): input is chunked document text.
            # Returns per-chunk depth-filtered ComplianceAlgoResult rows.
            # Breadth hits are stored internally and only surface when
            # show_accumulated() is called after the loop by the caller.
            # The merged view picks the best score per algo *across all chunks*,
            # so an algo failing in one chunk but passing in another will appear
            # as passing.  Breadth counts are also unioned across chunks.
            return (human_review, keyword_embeddings, filtered)

        # Non-accumulate path (DocClassify / Chat): input is KeyBERT n-grams or
        # a single text block — not multi-chunk.  show_accumulated() is called
        # immediately, so the phrase_table reflects only this invocation's data.
        # Returns presentation-ready ResultsForPrint objects (not raw algo results).
        human_review, phrase_table = self.accumulator.show_accumulated(stage)
        return (human_review, keyword_embeddings, phrase_table)

    # -------------------------
    # Merge helper
    # -------------------------
    def merge_algo_results(
        self,
        regex_results: List[ComplianceAlgoResult],
        levenshtein_results: List[ComplianceAlgoResult],
        merged_algo_name: Optional[str] = None,
    ) -> List[ComplianceAlgoResult]:
        """
        Merge results from regex and levenshtein algorithms for compliance checks.
        """
        regex_map: Dict[str, ComplianceAlgoResult] = {
            r.phrase: r for r in regex_results
        }
        lev_map: Dict[str, ComplianceAlgoResult] = {
            l.phrase: l for l in levenshtein_results
        }

        all_phrases: set[str] = set(regex_map.keys()) | set(lev_map.keys())
        merged: List[ComplianceAlgoResult] = []

        for phrase in sorted(all_phrases):
            r: ComplianceAlgoResult | None = regex_map.get(phrase)
            l: ComplianceAlgoResult | None = lev_map.get(phrase)

            score_r: float = r.score if r and r.score is not None else 0.0
            score_l: float = l.score if l and l.score is not None else 0.0
            combined_score: float = score_r + score_l

            threshold_r: float = r.threshold if r and r.threshold is not None else 0.0
            threshold_l: float = l.threshold if l and l.threshold is not None else 0.0
            combined_threshold: float = threshold_r + threshold_l

            detail_parts: list[str] = []
            if r:
                detail_parts.append(f"regex:{r.detail}")
            if l:
                detail_parts.append(f"lev:{l.detail}")

            merged.append(
                ComplianceAlgoResult(
                    algo=merged_algo_name,
                    phrase=phrase,
                    score=combined_score,
                    threshold=combined_threshold,
                    detail=" | ".join(detail_parts),
                )
            )

        return merged
