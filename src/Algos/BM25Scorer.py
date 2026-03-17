import math
from collections import Counter
from typing import Any, Dict, List, Optional

from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Commons.SingletonMixin import SingletonMixin
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class BM25Scorer(SingletonMixin, ScorerBase):

    def __init__(
        self,
        cfg: "Config | None" = None,
        *,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()
        self.sharedHelpers: SharedHelpers = SharedHelpers()
        from AI.AIHelpers import AIHelpers

        self.aiHelpers: AIHelpers = AIHelpers()

        # base banlist and algo name
        self.banlist_en: List[str] = self.helpers.get_banned_phrases_config_slot()
        self.algo: str = self.cfg.get_str("_BM25") or self.cfg.get_str("_M25") or "BM25"

        # caches per language
        self.banlist_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.idf_cache: Dict[str, Dict[str, float]] = {}
        self.avg_len_cache: Dict[str, float] = {}

        # runtime
        self.scores_raw: List[tuple[str, float]] = []
        self.scores_norm: List[tuple[str, float, float]] = []

        self.pretty.write(
            "O",
            "BM25Scorer",
            f"Initialized BM25 Scorer (algo={self.algo}) with {len(self.banlist_en)} base phrase(s)",
        )

    # ----------------------------
    # Prepare and cache banlist for language
    # ----------------------------
    def _prepare_banlist_for_language(
        self, language: Optional[str], stage: str
    ) -> List[Dict[str, Any]]:
        lang: str = (language or "en").lower()
        cached: List[Dict[str, Any]] | None = self.banlist_cache.get(lang)
        if cached is not None:
            return cached

        banlist: List[str] = self.sharedHelpers.get_banlist_for_language(
            self.banlist_en, lang, self.algo
        )
        self.helpers.require_set(banlist=banlist)

        prepared: List[Dict[str, Any]] = []
        df_counter: Counter[str] = Counter()
        total_len: int = 0

        for phrase in banlist:
            toks: List[str] = self.sharedHelpers.tokenize(phrase)
            toks = [t.lower() for t in toks if t.strip()]

            tf: Counter[str] = Counter(toks)
            phrase_len: int = len(toks)
            total_len += phrase_len

            for tok in set(toks):
                df_counter[tok] += 1

            prepared.append(
                {
                    "phrase": phrase,
                    "toks": toks,
                    "tf": dict[str, int](tf),
                    "len": phrase_len,
                }
            )
            if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                self.pretty.write(
                    "D",
                    "BM25 Scorer Cache",
                    f"Phrase: {phrase}, toks: {toks}, tf: {tf}, phrase len: {phrase_len}",
                )

        self.banlist_cache[lang] = prepared

        N: int = len(prepared) if prepared else 1
        idf: Dict[str, float] = {}
        for tok, freq in df_counter.items():
            idf[tok] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))
        self.idf_cache[lang] = idf

        avg_len: float = (total_len / N) if N > 0 else 1.0
        self.avg_len_cache[lang] = avg_len

        self.pretty.write(
            "O",
            "BM25 Scorer Cache",
            f"Built BM25 banlist cache with {len(prepared)} entries for language {lang}",
        )
        return prepared

    # ----------------------------
    # Compute IDF and avg_len (cached helpers)
    # ----------------------------
    def _get_idf_for_lang(
        self, lang: str, prepared: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        cached: Dict[str, float] | None = self.idf_cache.get(lang)
        if cached is not None:
            return cached
        return self._compute_idf_from_prepared(prepared)

    def _compute_idf_from_prepared(
        self, prepared: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        N: int = len(prepared) if prepared else 1
        df: Counter[str] = Counter()
        for entry in prepared:
            toks: list[str] = entry.get("toks", []) or []
            for t in set(toks):
                df[t] += 1
        idf: Dict[str, float] = {}
        for t, freq in df.items():
            idf[t] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))
        return idf

    def _get_avg_len_for_lang(self, lang: str, prepared: List[Dict[str, Any]]) -> float:
        cached: float | None = self.avg_len_cache.get(lang)
        if cached is not None:
            return cached
        if not prepared:
            avg_len = 1.0
        else:
            total = sum(len(entry.get("toks", [])) for entry in prepared)
            avg_len = total / len(prepared) if len(prepared) > 0 else 1.0
        self.avg_len_cache[lang] = avg_len
        return avg_len

    # ----------------------------
    # BM25 scoring using cached tf and len
    # ----------------------------
    def _bm25_score_from_entry(
        self,
        query_tokens: List[str],
        entry: Dict[str, Any],
        idf: Dict[str, float],
        avg_len: float,
        k1: float,
        b: float,
    ) -> float:
        tf_map: dict[str, int] = entry.get("tf", {}) or {}
        phrase_len: int = int(entry.get("len", 0) or 0)
        if phrase_len == 0 or not tf_map:
            return 0.0

        score: float = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            freq: float = float(tf_map.get(term, 0.0))
            term_idf: float = float(idf.get(term, 0.0))
            denom: float = freq + k1 * (1.0 - b + b * (phrase_len / (avg_len + 1e-12)))
            score += term_idf * (freq * (k1 + 1.0)) / (denom + 1e-12)
        return float(score)

    # ----------------------------
    # Percentile normalization helper (robust)
    # ----------------------------
    def _normalize_scores_percentile(
        self, scores: List[tuple[str, float]], percentile: float = 95.0
    ) -> List[tuple[str, float, float]]:
        """
        Input: [(phrase, raw_score), ...]
        Output: [(phrase, raw_score, norm_score), ...] with norm_score in [0,1]
        Normalization uses the given percentile as the scale denominator.
        """
        eps: float = 1e-12
        if not scores:
            return []
        phrases: tuple[str, ...]
        raw_vals: tuple[float, ...]
        phrases, raw_vals = zip(*scores)
        vals: list[float] = [float(v) for v in raw_vals]
        # compute percentile value
        n: int = len(vals)
        if n == 1:
            scale: float = max(vals[0], eps)
        else:
            sorted_vals: list[float] = sorted(vals)
            # linear interpolation for percentile index
            rank: float = (percentile / 100.0) * (n - 1)
            lo: int = int(math.floor(rank))
            hi: int = int(math.ceil(rank))
            if lo == hi:
                scale = sorted_vals[lo]
            else:
                frac: float = rank - lo
                scale = sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
            scale = max(scale, eps)
        normalized: list[tuple[str, float, float]] = []
        for p, v in zip(phrases, vals):
            norm: float = float(v) / float(scale)
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            normalized.append((p, float(v), norm))
        return normalized

    # ----------------------------
    # Main verify() entry point
    # ----------------------------
    def verify(
        self, text: str, language: Optional[str], stage: str = ""
    ) -> List[ComplianceAlgoResult]:
        if not text:
            return []
        compliance_slot: str = self.helpers.get_compliance_config_slot(stage)
        # thresholds expected in normalized scale [0,1]
        self.threshold = self.cfg.get_float(
            f"{compliance_slot}.PIPELINE.{self.algo}.THRESHOLD", 0.0
        )
        self.threshold_min = self.cfg.get_float(
            f"{compliance_slot}.PIPELINE.{self.algo}.THRESHOLD_MIN", 0.0
        )

        # BM25 params (k1 and b)
        k1_cfg: Any = self.cfg.get(
            f"{compliance_slot}.PIPELINE.{self.algo}.TERM_FREQ_SATURATION"
        )
        b_cfg: Any = self.cfg.get(
            f"{compliance_slot}.PIPELINE.{self.algo}.LENGTH_NORMALIZATION"
        )
        k1: float = float(k1_cfg) if k1_cfg is not None else 1.2
        b: float = float(b_cfg) if b_cfg is not None else 0.75

        self.helpers.require_set(
            threshold=self.threshold,
            threshold_min=self.threshold_min,
            term_freq_saturation=k1,
            length_normalization=b,
        )

        # prepare language-specific banlist and caches
        prepared: List[Dict[str, Any]] = self._prepare_banlist_for_language(
            language, stage
        )
        lang: str = (language or "en").lower()
        idf: Dict[str, float] = self.idf_cache.get(
            lang
        ) or self._compute_idf_from_prepared(prepared)
        self.idf_cache[lang] = idf
        avg_len: float = self.avg_len_cache.get(lang) or self._get_avg_len_for_lang(
            lang, prepared
        )

        # runtime config overrides (per-call / per-config)
        min_overlap: int = self.cfg.get_int(
            f"{compliance_slot}.PIPELINE.{self.algo}.MIN_OVERLAP"
        )
        min_raw_score: float = self.cfg.get_float(
            f"{compliance_slot}.PIPELINE.{self.algo}.MIN_RAW_SCORE"
        )
        norm_percentile: float = self.cfg.get_float(
            f"{compliance_slot}.PIPELINE.{self.algo}.NORM_PERCENTILE"
        )

        if self.cfg.get_int("DEBUG_LEVEL") >= 4:
            self.pretty.write(
                "D",
                "BM25 Scorer",
                f"min_overlap: {min_overlap} min_raw_score: {min_raw_score} norm_percentile: {norm_percentile}",
            )
        # Normalize text (remove leet speak, confusables) to match banlist normalization
        text_normalized: str = self.sharedHelpers.normalize(text)
        query_tokens = self.sharedHelpers.tokenize(text_normalized)
        if not query_tokens:
            return []
        # compute raw scores using cached tf and len, with overlap gate and raw threshold
        self.scores_raw = []
        qset: set[str] = set(query_tokens)
        for entry in prepared:
            phrase: str = entry.get("phrase", "")
            entry_toks: list[str] = entry.get("toks", []) or []
            overlap: set[str] = qset & set(entry_toks)
            if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                self.pretty.write(
                    "D",
                    "BM25 Scorer",
                    f"overlap: {len(overlap)} min_overlap: {min_overlap}",
                )
            if len(overlap) < min_overlap:
                continue
            raw: float = self._bm25_score_from_entry(
                query_tokens, entry, idf, avg_len, k1, b
            )
            if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                self.pretty.write(
                    "D", "BM25 Scorer", f"raw: {raw} min_raw_score: {min_raw_score}"
                )
            if raw < min_raw_score:
                continue

            self.scores_raw.append((phrase, raw))

        # normalize using percentile normalization (robust to outliers)
        self.scores_norm = self._normalize_scores_percentile(
            self.scores_raw, percentile=norm_percentile
        )

        return self.return_algo_result()

    # ----------------------------
    # Contract: return_algo_result()
    # ----------------------------
    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        results: List[ComplianceAlgoResult] = []

        for phrase, raw_score, norm_score in getattr(self, "scores_norm", []):
            if norm_score < (self.threshold_min or 0.0):
                continue

            results.append(
                ComplianceAlgoResult(
                    algo=self.algo,
                    phrase=phrase,
                    score=norm_score,
                    threshold=self.threshold,
                    detail=f"bm25_raw={raw_score:.6f} bm25_norm={norm_score:.6f}",
                )
            )
        return results
