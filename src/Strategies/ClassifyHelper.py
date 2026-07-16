# Local module imports
# Standard library imports
import json
from collections import OrderedDict
from typing import (  # Assuming documents is a list of some custom document objects.
    Any, List)

import numpy as np
# Third-party imports
import torch
from keybert import KeyBERT  # type: ignore[reportMissingTypeStubs]
from nltk.stem.snowball import \
    SnowballStemmer  # type: ignore[reportMissingTypeStubs]
from sentence_transformers import util

from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from Algos.ReverseStemmer import ReverseStemmer
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers


class ClassifyHelper:
    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        # Instantiate helper objects/singletons as instance attributes.
        # Cache for stopwords per language.
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.cfg: Config = cfg or Config()
        self.models_cache: ModelsCache = ModelsCache()
        self.bits: int = self.cfg.get_int("EMBEDDER_BITS", 32)
        self.device: Any
        self.device_type: str
        self.target_dtype: Any
        self.device, self.device_type, self.target_dtype, _ = (
            self.models_cache.switch2device()
        )

    def get_closest_word_with_weights(
        self,
        vector: Any,
        keyword_list: list[str],
        keyword_embeddings: Any,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Retrieve the top_n keywords closest to the given vector using cosine similarity,
        along with their similarity scores.

        Returns:
        - A list of tuples: (keyword, cosine_similarity)
        """

        # Option 1: Force everything to CPU
        keyword_embeddings = self.tensorHelpers.switch_tensor_device(
            keyword_embeddings, self.device_type, self.target_dtype
        )
        vector = self.tensorHelpers.switch_tensor_device(
            vector, self.device_type, self.target_dtype
        )

        similarities = util.cos_sim(vector, keyword_embeddings)  # type: ignore[reportUnknownMemberType]
        similarities_arr = similarities[0].cpu().numpy()

        # Get indices for top_n similar keywords
        closest_indices = np.argsort(similarities_arr)[::-1][:top_n]

        # Return list of (keyword, similarity score)
        return [(keyword_list[i], float(similarities_arr[i])) for i in closest_indices]

    def double_keybert_with_weights(
        self,
        text: str,
        top_n_first: int,
        top_n_second: int,
        embed_model_name: str,
        embedding: torch.Tensor,
    ) -> tuple[List[tuple[str, float]], torch.Tensor]:
        """
        Two-stage keyword extraction with KeyBERT and conversion into embeddings.
        Now returns sorted keywords with their weights.
        """
        kb_cfg = self.helpers.get_keybert_config()  # type: ignore[reportPrivateUsage]
        ngram_pass1: Any = kb_cfg.get("NGRAM_PASS1")
        ngram_pass2: Any = kb_cfg.get("NGRAM_PASS2")
        sbert_model = self.models_cache.load_quantized_model(embed_model_name)

        kw_model: Any = KeyBERT(model=sbert_model)  # type: ignore[arg-type]
        stop_words = self.fileUtils.get_stopwords(text)

        self.pretty.write(
            "I",
            "KeyBERT",
            f"Embed model: {embed_model_name}, bits: {self.bits} top_n_first/second {top_n_first} / {top_n_second} ngram_pass1/2 {ngram_pass1} / {ngram_pass2}",
        )
        self.pretty.write(
            "I",
            "KeyBERT",
            f"Extracting keywords with: {embed_model_name}, using {self.bits} bits. For large documents this will take its time...",
        )

        # Normalize embedding for KeyBERT (must be numpy on CPU)
        if isinstance(embedding, torch.Tensor):  # type: ignore[reportUnnecessaryIsInstance]
            embedding_input: Any = (
                embedding.detach()
                .cpu()
                .numpy()
                .astype(self.tensorHelpers.dtype_from_bits("numpy"))
            )
        else:
            embedding_input = embedding

        first_keywords: list[tuple[str, float]] = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_pass1,
            stop_words=stop_words,
            top_n=top_n_first,
            doc_embeddings=embedding_input,  # None is allowed in the KeyBERT API
        )
        refined_text = " ".join([keyword for keyword, _ in first_keywords])
        # Second pass to refine keywords along with their weights.
        second_keywords: list[tuple[str, float]] = kw_model.extract_keywords(
            refined_text,
            keyphrase_ngram_range=ngram_pass2,
            stop_words=stop_words,
            top_n=top_n_second,
            doc_embeddings=embedding_input,  # None is allowed in the KeyBERT API
        )

        sorted_keywords = sorted(
            second_keywords, key=lambda item: item[1], reverse=True
        )
        keyword_list: list[str] = [kw for kw, _ in sorted_keywords]

        keyword_embeddings = sbert_model.encode(  # type: ignore[reportUnknownMemberType]
            keyword_list,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device_type,
        )
        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D",
                "doubleKeyBert",
                f"Text: {text} Refined Text: {first_keywords} Second Keywords: {second_keywords}",
            )
        keyword_embeddings = self.tensorHelpers.ensure_tensor(keyword_embeddings)
        return sorted_keywords, keyword_embeddings

    def stem_keywords_with_weights(
        self, weighted_keywords: list[tuple[str, float]]
    ) -> tuple[list[tuple[str, float]], ReverseStemmer | None]:
        """
        Given a list of (keyword, weight) tuples, this function:
        1. Combines the keywords into a string to detect the language.
        2. Initializes a SnowballStemmer for the detected language.
        3. Applies stemming to each keyword.
        4. Merges the weights for identical stems, taking the maximum weight.
        5. If REVERSE_STEMMING is enabled, builds a ReverseStemmer so callers
           can recover the best original surface form for each stem (only the
           highest-weight original word is kept per stem – no weight is stored
           in the map itself).  Returns None when REVERSE_STEMMING is disabled.

        Parameters:
            weighted_keywords (list[tuple[str, float]]): A list where each element
                is a tuple containing a keyword (str) and its associated weight (float).

        Returns:
            Tuple of:
              - list of (stem, max_weight) tuples
              - ReverseStemmer mapping each stem → its best original word,
                or None when REVERSE_STEMMING is False
        """
        # Combine the keywords for language detection.
        combined_text = " ".join(keyword for keyword, _ in weighted_keywords)
        language = self.fileUtils.get_text_language(combined_text, "ntlk")

        # Initialize the stemmer, with fallback to English if needed.
        try:
            stemmer = SnowballStemmer(language)
        except ValueError:
            print(
                f"Warning: SnowballStemmer does not support language '{language}', falling back to English."
            )
            stemmer = SnowballStemmer("english")

        # Dictionary to merge stems and capture the maximum weight.
        stem_to_weight: dict[str, float] = {}
        build_reverse: bool = self.cfg.get_bool("REVERSE_STEMMING")
        reverse_map: ReverseStemmer | None = (
            ReverseStemmer(self.pretty) if build_reverse else None
        )
        for keyword, weight in weighted_keywords:
            stem: str = stemmer.stem(keyword)  # type: ignore[reportUnknownMemberType]
            if stem in stem_to_weight:
                stem_to_weight[stem] = max(stem_to_weight[stem], weight)
            else:
                stem_to_weight[stem] = weight
            # ReverseStemMap keeps the original word with the highest weight.
            if reverse_map is not None:
                reverse_map.update(stem, keyword, weight)

        # Convert the dictionary back to a list of tuples.
        merged_stems: list[tuple[str, float]] = list(stem_to_weight.items())
        return merged_stems, reverse_map

    def merge_keyword_weights(
        self,
        extraction_keywords: list[tuple[str, float]],
        closest_words_with_weights: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Merge the KeyBERT (extraction) weights with cosine similarity weights.
        Combines via multiplication (adjustable if needed).

        Parameters:
        extraction_keywords: list of (keyword, extraction_weight)
        closest_words_with_weights: list of (keyword, cosine_similarity)

        Returns:
        List of (keyword, combined_weight) tuples.
        """
        # Build dictionary for quick lookup of extraction weights.
        extraction_weight_dict = {
            keyword: weight for keyword, weight in extraction_keywords
        }
        merged_results: list[tuple[str, float]] = []
        for keyword, cosine_sim in closest_words_with_weights:
            extraction_weight = extraction_weight_dict.get(keyword, 0)
            combined_weight = extraction_weight * cosine_sim
            merged_results.append((keyword, combined_weight))
        merged_results.sort(key=lambda x: x[1], reverse=True)
        return merged_results

    def build_classify_prompt(
        self, prompt_in: Any, formatted_keywords: OrderedDict[str, float]
    ) -> Any:

        prompt = prompt_in.format(
            CLASSIFICATION_WORD_CNT=self.cfg.get("CLASSIFICATION_WORD_CNT"),
            SUMMARY_SENTENCE_CNT=self.cfg.get("SUMMARY_SENTENCE_CNT"),
        )
        json_keywords = json.dumps(
            formatted_keywords, sort_keys=False, ensure_ascii=False
        )
        prompt = prompt + json_keywords
        return prompt
