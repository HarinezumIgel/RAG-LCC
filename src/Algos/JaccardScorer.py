import time
from typing import Any, Dict, List, cast

from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Algos.Synonyms import Synonyms
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class JaccardScorer(ScorerBase):
    """
    Jaccard scorer: no language in ctor. Call score(text, language).
    Caches per-language processed banlist (normalized, tokenized, char-ngrams).
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        self.sharedHelpers: SharedHelpers = SharedHelpers()
        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        from AI.AIHelpers import AIHelpers

        self.aiHelpers: AIHelpers = AIHelpers()
        self.perf_logger: PerfLogger = PerfLogger()

        # static config keys / tunables (language-independent)
        self.algo: str = self.cfg.get_str("_JACCARD")

        # english banlist source (raw strings), expanded with WordNet synonyms
        self.banlist_en: List[str] = Synonyms().expand(
            self.helpers.get_banned_phrases_config_slot()
        )

        # per-language cache:
        # lang -> list of dicts { phrase, toks, char_grams }
        self.banlist_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.prepared_banlist: List[Dict[str, Any]] = []

        self.toks_text: list[str] | None = None
        self.char_grams_text: list[str] | None = None
        self.threshold: float | None = None

    def _prepare_banlist_for_language(
        self, language: str, stage: str
    ) -> List[Dict[str, Any]]:
        lang: str = (language or "en").lower()
        cached: List[Dict[str, Any]] | None = self.banlist_cache.get(lang)
        if cached is not None:
            return cached

        self.perf_logger.log(
            "JaccardScorer._prepare_banlist", f"start cache build lang={lang}"
        )
        _t0 = time.perf_counter()
        # read tunables (these are per-algo keys in config)
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        char_ngram_range: tuple[int, int] = cast(
            tuple[int, int],
            tuple(
                self.cfg.get(
                    f"{compliance_config_slot}.PIPELINE.{self.algo}.CHAR_NGRAM_RANGE"
                )
            ),
        )
        banlist: List[str] = self.sharedHelpers.get_translated_wordlist(
            self.banlist_en, lang, self.algo
        )
        self.helpers.require_set(char_ngram_range=char_ngram_range, banlist=banlist)
        prepared: List[Dict[str, Any]] = []
        for phrase in banlist:
            toks: list[str] = self.sharedHelpers.tokenize(phrase)
            char_grams: list[str] = self.sharedHelpers.char_ngrams(
                phrase, *char_ngram_range
            )
            prepared.append({"phrase": phrase, "toks": toks, "char_grams": char_grams})
        self.banlist_cache[lang] = prepared
        self.perf_logger.log(
            "JaccardScorer._prepare_banlist",
            f"stop  cache build lang={lang} n={len(prepared)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        self.pretty.write(
            "O",
            "Cache build Jaccard Banned",
            f"Built Jaccard n-gram cache with {len(prepared)} entries for language {language}",
        )
        return prepared

    def _verify_impl(
        self, text: str, language: str = "en", stage: str = "PIPELINE_CHECK"
    ) -> List[ComplianceAlgoResult]:
        """
        Score the text against the banlist for the given language.
        Returns list of dicts with keys: algo, phrase, score, threshold, detail
        """
        if not text:
            return []

        # read tunables (these are per-algo keys in config)
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        char_ngram_range: tuple[int, int] = cast(
            tuple[int, int],
            tuple(
                self.cfg.get(
                    f"{compliance_config_slot}.PIPELINE.{self.algo}.CHAR_NGRAM_RANGE"
                )
            ),
        )
        self.threshold = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD"
        )
        self.threshold_min = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD_MIN"
        )
        self.helpers.require_set(
            char_ngram_range=char_ngram_range,
            threshold=self.threshold,
            threshold_min=self.threshold_min,
        )
        lang: str = (language or "en").lower()
        self.prepared_banlist = self._prepare_banlist_for_language(lang, stage)

        text_norm: str = self.sharedHelpers.normalize(text)
        self.toks_text = self.sharedHelpers.tokenize(text_norm)
        self.char_grams_text = self.sharedHelpers.char_ngrams(
            text_norm, *char_ngram_range
        )
        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D",
                "Jaccard Scorer",
                f"Running with: char_ngram_range: {char_ngram_range} "
                f"threshold {self.threshold} threshold min: {self.threshold_min}",
            )
        return self.return_algo_result()

    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        results: List[ComplianceAlgoResult] = []
        assert self.toks_text is not None and self.char_grams_text is not None
        for entry in self.prepared_banlist:
            phrase: str = entry["phrase"]
            toks_phrase: list[str] = entry["toks"]
            jacc_t: float = self.sharedHelpers.jaccard(self.toks_text, toks_phrase)
            contain: float = self.sharedHelpers.containment(self.toks_text, toks_phrase)
            jacc_c: float = self.sharedHelpers.jaccard(
                self.char_grams_text, entry["char_grams"]
            )
            score_val: float = max(jacc_t, jacc_c, contain)
            if score_val >= self.threshold_min:
                results.append(
                    ComplianceAlgoResult(
                        algo=self.algo,
                        phrase=phrase,
                        score=score_val,
                        threshold=self.threshold,
                        detail=f"token={jacc_t:.3f}, char={jacc_c:.3f}, contain={contain:.3f}",
                    )
                )
        return results
