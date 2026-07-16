# Local module imports
# Standard library imports
import json
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any, Dict, cast

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from AI.AIHelpers import AIHelpers
from AI.LLMCaller import LLMCaller
from AI.ModelOutputAdapter import ModelOutput, ModelOutputAdapter
from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from AI.TokenBudget import TokenBudget
from Commons.Exceptions import LLMComplianceCheckError, PromptComplianceError
from Commons.SingletonMixin import SingletonMixin
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Compliance.Exclusions import Exclusions
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.CounterInstance import (FailedCount, HumanReviewCount,
                                     ProcessedCount)
from Globals.Globals import Globals
from Gui.Colors import ORANGE, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.CSVWriter import CSVWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Strategies.ClassifyHelper import ClassifyHelper
from Strategies.ProcessingStrategy import ProcessingStrategy
from Strategies.StrategyType import StrategyType

# Third-party imports
# Imaging, file detection & data processing libraries


class ClassifyStrategy(SingletonMixin, ProcessingStrategy):
    strategy_type = StrategyType.CLASSIFY  # enum identifier

    def process(self, doc: dict[str, Any] | None) -> None:
        if doc is None:
            return
        # self.ensure_compliance_checked()
        self._process_extract(doc)

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

        # Instantiate helper objects/singletons as instance attributes.
        self.failedCounter: FailedCount = FailedCount()
        self.processedCounter: ProcessedCount = ProcessedCount()
        self.humanReviewCount: HumanReviewCount = HumanReviewCount()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.modelOutputAdapter: ModelOutputAdapter = ModelOutputAdapter()
        self.llmCaller: LLMCaller = LLMCaller()
        self.classifyHelper: ClassifyHelper = ClassifyHelper()
        self.cfg: Config = cfg or Config()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.csvWriter: CSVWriter = CSVWriter()
        self.globalsInstance: Globals = Globals()
        self.bannedPhraseCollector: BannedPhraseCollector = BannedPhraseCollector()
        self.exclusions: Exclusions = Exclusions()
        self.doc_dir: str = self.cfg.get_str("DOC_DIR")
        self.embed_model_name: str = self.helpers.get_model_args("_ACTIVE_EMBED")[
            "MODEL"
        ]
        active_extraction: str = self.cfg.get_str("_ACTIVE_EXTRACTION_CONFIG")
        kb_cfg: dict[str, Any] = self.helpers.get_keybert_config()  # type: ignore[reportPrivateUsage]
        self.top_n_first: int = int(kb_cfg["TOP_N_FIRST"])
        self.top_n_second: int = int(kb_cfg["TOP_N_SECOND"])
        self.unwanted_char_map: dict[str, Any] = self.cfg.get_dict("_UNWANTED_CHAR_MAP")
        self.csv_delimiter: str = self.cfg.get_str("CSV_DELIMITER")

        # Device info
        self.device, self.device_type, self.target_dtype, self.device_index = (
            self.models_cache.switch2device()
        )

        # Core detection type
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(
            "PROMPT_CHECK"
        )

        self.is_streaming: bool = self.helpers.get_active_endpoint_args().get(
            "STREAMING_REQ", False
        )

        # LLM compliance parameters
        self.temperature_chk: float = self.cfg.get_float(
            f"{compliance_config_slot}.LLM_PARAM.temperature"
        )
        self.use_ollama_gpu: bool = self.helpers.get_active_endpoint_args().get(
            "USE_GPU", True
        )
        self.use_ollama_gpu_chk: bool = self.cfg.get_bool(
            f"{compliance_config_slot}.LLM_PARAM.use_ollama_gpu"
        )
        self.top_k_chk: int = self.cfg.get_int(
            f"{compliance_config_slot}.LLM_PARAM.top_k"
        )
        self.top_p_chk: float = self.cfg.get_float(
            f"{compliance_config_slot}.LLM_PARAM.top_p"
        )
        self.do_check_prompt: bool = self.cfg.get_bool(
            f"{compliance_config_slot}.Check", True
        )
        self.llm_model: str = self.helpers.get_model_args("_ACTIVE_LLM")["MODEL"]
        self.llm_chk_model: str = self.helpers.get_model_args("_ACTIVE_LLM_CHK")[
            "MODEL"
        ]
        self.prompt_chk: str
        self.prompt_chk_name: str | None
        # Indirect call — resolve prompt variable then follow indirection
        chk_prompt_var: str = self.helpers.get_model_args("_ACTIVE_LLM_CHK")[
            "PROMPT_CLASSIFY"
        ]
        self.prompt_chk, self.prompt_chk_name = self.cfg.indirect_get(chk_prompt_var)
        self.prompt: str
        self.prompt_name: str | None
        prompt_var: str = self.helpers.get_model_args("_ACTIVE_LLM")["PROMPT_CLASSIFY"]
        self.prompt, self.prompt_name = self.cfg.indirect_get(prompt_var)
        self.banlist_en: list[str] = self.helpers.get_banned_phrases_config_slot()
        self.user_classification_keys: list[str] = self.cfg.get_list(
            "_YOUR_CLASSIFICATION_KEYS"
        )
        self.classification_word_cnt: int = self.cfg.get_int("CLASSIFICATION_WORD_CNT")
        self.summary_sentence_cnt: int = self.cfg.get_int("SUMMARY_SENTENCE_CNT")
        self.classification_keys: list[str] = self.cfg.get_list("_CLASSIFICATION_KEYS")

        self.temperature_ext: float = self.cfg.get_float(
            f"_EXTRACTION_MODEL_PARAMS.{active_extraction}.TEMPERATURE_EXT"
        )
        self.top_k_ext: int = self.cfg.get_int(
            f"_EXTRACTION_MODEL_PARAMS.{active_extraction}.TOP_K_EXT"
        )
        self.top_p_ext: float = self.cfg.get_float(
            f"_EXTRACTION_MODEL_PARAMS.{active_extraction}.TOP_P_EXT"
        )

        self.keybert: str = self.cfg.get_str("_KEYBERT")
        self.jaccard: str = self.cfg.get_str("_JACCARD")
        self.regex: str = self.cfg.get_str("_REGEX")
        self.cosine: str = self.cfg.get_str("_COSINE")

        self.content: str
        self.file: str
        self.filePath: str
        self.creation_date: str
        self.fileType: str
        self.doc: Dict[str, Any]
        self.converted_obj: Any
        self.embedder: HuggingFaceEmbeddings = self.models_cache.get_hf_embeddings()
        self.tokenBudget: TokenBudget = TokenBudget()
        self.use_exclusions: bool = self.cfg.get_bool("USE_EXCLUSIONS")

        self._ensure_compliance_checked()

    def _ensure_compliance_checked(self) -> None:
        """Run the one-time LLM compliance check on the classify prompt."""

        user_classify_prompt: str = self.prompt.format(
            CLASSIFICATION_WORD_CNT=self.classification_word_cnt,
            SUMMARY_SENTENCE_CNT=self.summary_sentence_cnt,
        )
        prompt = f"{self.prompt_chk}\n\nINPUT:\n{user_classify_prompt}\n\nOutput:"
        # Check the user-provided prompt against the compliance filter chain
        ret, _ = self.aiHelpers.check_user_prompt_with_filter_chain(
            user_classify_prompt, "PROMPT_CHECK"
        )
        if ret is True:
            raise PromptComplianceError(
                "User-provided prompt failed compliance validation"
            )
        if self.do_check_prompt is False:
            self.pretty.write(
                "W",
                "CheckPrompt",
                "CHECK PROMPT was disabled in Config_Banned.py",
                color=ORANGE,
            )
        else:
            _fmt: defaultdict[str, str] = defaultdict(
                str,
                {
                    # key used by Mistral classify prompt
                    "CLASSIFICATION_KEYS": "\n".join(
                        f"  <key>{k}</key>" for k in self.user_classification_keys
                    ),
                    # key used by Llama-Guard classify prompt
                    "USER_DEFINED_CLASSIFICATION_KEYS": "\n".join(
                        f"  <key>{k}</key>" for k in self.user_classification_keys
                    ),
                    "BANNED_WORDS_ENGLISH": "\n".join(
                        f"  <word>{w}</word>" for w in self.banlist_en
                    ),
                },
            )
            prompt = self.prompt_chk.format_map(_fmt)
            compliance_options: dict[str, Any] = {
                "temperature": self.temperature_chk,
                "top_k": self.top_k_chk,
                "top_p": self.top_p_chk,
                "num_predict": self.tokenBudget.compute_dynamic_max_tokens(
                    prompt, self.llm_chk_model, model_role="_ACTIVE_LLM_CHK"
                ),
                "num_ctx": self.tokenBudget.get_context_limit(self.llm_chk_model),
            }
            if not self.use_ollama_gpu_chk:
                compliance_options["num_gpu"] = 0
            self.aiHelpers.check_provided_prompt(
                prompt=prompt,
                llm_model=self.llm_chk_model,
                ollama_options=compliance_options,
                answer_is_json=True,
                template_name=self.prompt_chk_name or "",
                stage="Check provided prompt",
            )

    def _process_extract(self, doc: dict[str, Any]):  # Load
        self.doc = doc
        self.pretty.write(
            "I",
            "Count",
            f"So far {self.processedCounter.get()} documents processed",
        )
        # Preprocess the document.
        self.pretty.write(
            "I",
            "Punctuation",
            f"Replacing unwanted chars and punctuations. E.g: {self.unwanted_char_map}",
        )
        cleaned_text: str = self.fileUtils.clean_text(
            doc.get("content", ""), cast(dict[str, str], self.unwanted_char_map)
        )

        language: str = self.fileUtils.get_text_language(cleaned_text, "iso-639")

        # --- unsupported-language gate ---
        file_path: str = doc.get("meta", {}).get("FilePath", "?")
        lang_action: str | None = SharedHelpers().check_language_support(
            language, file_path
        )
        if lang_action == "NOT_OK":
            doc.setdefault("meta", {}).update(
                {
                    "Status": "NOT_OK",
                    "Stage": "Language",
                    "Time": datetime.now().isoformat(),
                }
            )
            self.csvWriter.write_json2csv(doc["meta"], "NOT_OK")
            self.failedCounter.increment()
            return

        # Preprocess the document.
        self.pretty.write(
            "I",
            "Embedding",
            f"Embedding document with {self.embed_model_name} using {self.target_dtype}",
        )

        embeddings: list[float] = self.embedder.embed_documents([cleaned_text])[0]
        # embeddings is your list[float] from embed_documents([cleaned_text])[0]
        # 1) Convert to torch tensor (optional)
        emb_tensor: torch.Tensor = torch.tensor(
            embeddings, dtype=self.target_dtype
        )  # shape: (D,)

        # 2) Ensure batched shape (1, D)
        emb_tensor = emb_tensor.unsqueeze(0)  # shape: (1, D)

        # 3) Convert to CPU numpy float32 (KeyBERT-friendly)
        embeddings_for_keybert: Any

        if self.device_type == "cpu":
            embeddings_for_keybert = (
                emb_tensor.cpu()
                .numpy()
                .astype(self.tensorHelpers.dtype_from_bits("numpy"))
            )  # shape: (1, D)
        else:
            embeddings_for_keybert = emb_tensor.to(self.device_type)

        extraction_keywords: list[Any]
        keyword_embeddings: Any
        extraction_keywords, keyword_embeddings = (
            self.classifyHelper.double_keybert_with_weights(
                cleaned_text,
                self.top_n_first,
                self.top_n_second,
                self.embed_model_name,
                embeddings_for_keybert,
            )
        )

        human_review: bool
        human_review, _, phrase_table = self.aiHelpers.run_ensemble_checks(
            cleaned_text,
            language,
            stage="PIPELINE_CHECK",
            accumulate=False,
            require_keybert=True,
            embedding=embeddings,
        )

        # 2) Prepare keyword list and embeddings
        raw_keywords: list[str] = [
            kw for kw, _weight in extraction_keywords
        ]  # list[str]

        # Ensure keyword_embeddings is a tensor (N, D)
        if isinstance(keyword_embeddings, list):
            keyword_matrix: torch.Tensor = torch.stack(
                cast(list[torch.Tensor], keyword_embeddings)
            )  # (N, D)
        else:
            keyword_matrix: torch.Tensor = keyword_embeddings  # assume already (N, D)

        self.pretty.write(
            "I",
            "Classify strategy",
            f"Cosine similarity between the input vector and all keyword embeddings top_n_second: {self.top_n_second}",
        )

        # 4) Call the helper
        closest_keywords = self.classifyHelper.get_closest_word_with_weights(
            embeddings_for_keybert,
            raw_keywords,
            keyword_matrix,
            top_n=self.top_n_second,
        )

        self.pretty.write(
            "I",
            "Merge keywords and cosine",
            "Merging keywords obtained from double keyBERT and cosine similarity",
        )
        merged_keywords = self.classifyHelper.merge_keyword_weights(
            extraction_keywords, closest_keywords
        )

        self.pretty.write(
            "I", "Stemming", "Stemming the keywords with Snowball Stemmer"
        )
        stemmed_keywords, reverse_stem_map = (
            self.classifyHelper.stem_keywords_with_weights(merged_keywords)
        )

        self.pretty.write("I", "Sorting", "Sorting the keywords using their weights")
        stemmed_keywords.sort(key=lambda x: x[1], reverse=True)

        self.pretty.write(
            "I",
            "Formatting",
            f"Formatting the keyword/weight list: Number of items: {len(stemmed_keywords)}",
        )
        formatted_keywords: OrderedDict[str, float] = OrderedDict(
            (stem, float(weight)) for stem, weight in stemmed_keywords
        )
        # Ensure formatting consistency.
        formatted_keywords = OrderedDict(
            (stem, float(weight)) for stem, weight in formatted_keywords.items()
        )

        self.doc["meta"]["Keywords"] = formatted_keywords

        prompt = self.classifyHelper.build_classify_prompt(
            self.prompt, formatted_keywords
        )

        # Compute output-token budget from the actual assembled prompt
        max_output_tokens_dyn: int = self.tokenBudget.compute_dynamic_max_tokens(prompt)

        # Build the unified Ollama options dict
        ollama_options: dict[str, Any] = {
            "temperature": self.temperature_ext,
            "top_k": self.top_k_ext,
            "top_p": self.top_p_ext,
            "num_predict": max_output_tokens_dyn,
            "num_ctx": self.tokenBudget.get_context_limit(self.llm_model),
        }
        if not self.use_ollama_gpu:
            ollama_options["num_gpu"] = 0

        handler = self.llmCaller.make_on_chunk(ollama_options)
        # for cell in handler.__closure__: print(cell.cell_contents)
        llm_result = self.llmCaller.call_llm(
            self.llm_model,
            prompt,
            ollama_options,
            answer_is_json=True,
            template_name=self.prompt_name,
            on_chunk=handler,
            streaming=self.is_streaming,
            stage="Run classification prompt",
        )

        # handle errors
        if isinstance(llm_result, dict) and "error" in llm_result:  # type: ignore[reportUnnecessaryIsInstance]
            self.pretty.write(
                "E", "LLM", f"LLM error: {llm_result['error']}", color=RED
            )
            raise LLMComplianceCheckError(
                f"Classification LLM failed: {llm_result['error']}"
            )

        answer: ModelOutput = self.modelOutputAdapter.interpret(
            llm_result,
            self.llm_model,
            is_compliance=False,
            is_streaming=self.is_streaming,
        )
        # print(answer.raw)
        if answer.is_json is False:
            self.doc["meta"].update({"Temperature": f"{self.temperature_ext}"})
            self.globalsInstance.add_failed_doc(self.doc["meta"])
            meta = self.doc.get("meta", {})

            self.csvWriter.write_json2csv(
                meta,
                "NOT_OK",
            )
            self.failedCounter.increment()
            return

        # Ensure the classification is a dictionary.
        if answer.content is None:
            return

        parsed: dict[str, Any] = json.loads(answer.content)
        self.doc["meta"].update(parsed)

        # Apply reverse stemming once here so both the OK write and the
        # HUMAN_REVIEW write see the restored surface forms – no duplication.
        if reverse_stem_map and self.cfg.get_bool("REVERSE_STEMMING"):
            self.doc["meta"] = reverse_stem_map.apply_to_meta(
                self.doc["meta"], self.user_classification_keys
            )

        if "meta" in self.doc:  # type: ignore[reportUnnecessaryComparison]
            self.globalsInstance.add_document(self.doc["meta"])
        self.info()
        # Write the processed documents to file.
        self.doc["meta"].update({"Status": f"OK"})
        self.doc["meta"].update({"Temperature": f"{self.temperature_ext}"})
        self.doc["meta"].update({"Stage": "Summary"})
        self.doc["meta"].update({"Time": datetime.now().isoformat()})  # Must be string
        self.doc["meta"].update({"Temperature": f"{self.temperature_ext}"})
        self.csvWriter.write_json2csv(
            self.doc["meta"],
            "OK",
        )

        if human_review:
            self.humanReviewCount.increment()
            self.doc["meta"]["Status"] = "NOT_OK"
            hr_data: list[dict[str, Any]] | dict[str, Any]
            if phrase_table:
                hr_data = self.bannedPhraseCollector.prepare_for_csv_print(
                    phrase_table, self.doc["meta"]
                )
            else:
                hr_data = dict(self.doc["meta"])
            self.csvWriter.write_json2csv(hr_data, "HUMAN_REVIEW")
            orig_path = self.doc["meta"]["FilePath"]
            self.pretty.write(
                "W",
                "HUMAN_REVIEW",
                f"File {orig_path} selected for human review.",
            )
            if self.use_exclusions:
                self.exclusions.add(orig_path)

    def _pretty_meta(self, value: Any, indent: int = 0) -> str:
        """Return a human-readable string without Python dict/list syntax."""
        space: str = " " * indent

        # Case 1: dictionary-like
        if isinstance(value, dict):
            d_value: dict[str, Any] = cast(dict[str, Any], value)
            lines: list[str] = []
            for key, val in d_value.items():
                if isinstance(val, (dict, list)):
                    lines.append(f"{space}{key}:")
                    lines.append(self._pretty_meta(val, indent + 2))
                else:
                    lines.append(f"{space}{key}: {val}")
            return "\n".join(lines)

        # Case 2: list-like
        if isinstance(value, list):
            l_value: list[Any] = cast(list[Any], value)
            lines2: list[str] = []
            for item in l_value:
                if isinstance(item, (dict, list)):
                    lines2.append(f"{space}-")
                    lines2.append(self._pretty_meta(item, indent + 2))
                else:
                    lines2.append(f"{space}- {item}")
            return "\n".join(lines2)

        # Case 3: scalar
        return f"{space}{value}"

    def info(self):
        if DebugHelper.check(self.cfg, 10):
            self.pretty.write("D", "Result", f"{"<" * 60}")

            # Keys to display
            max_length: int = max(len(key) for key in self.classification_keys)
            # Print formatted key-value pairs
            for key in self.classification_keys:
                self.pretty.write(
                    "D",
                    "Result",
                    f"{key.ljust(max_length)} : {self.doc['meta'].get(key, 'N/A')}",
                )

            if DebugHelper.check(self.cfg, 70):
                line = self.doc.get("content", "N/A").strip()
            elif DebugHelper.check(self.cfg, 10):
                line = self.doc.get("content", "N/A").strip()[:60]
            else:
                line = ""

            meta_str = self._pretty_meta(self.doc.get("meta"))

            output = f"""Meta:
    {meta_str}

    Content: {line} ...
    """
            for line in output.splitlines():
                self.pretty.write("D", "Result", line)

            self.pretty.write("D", "Result", f"{"<" * 60}")
