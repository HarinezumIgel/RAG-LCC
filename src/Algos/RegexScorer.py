import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Algos.Synonyms import Synonyms
# Replace these imports with your real modules if present
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class RegexScorer(ScorerBase):
    """
    Memory-only Regex detector.
    - Keeps compiled patterns in an in-memory cache (lang -> stage -> list of entries).
    - Stores human-readable pattern_text in meta for explainability/serialization.
    - Implements return_algo_result() to satisfy the ScorerBase contract.
    """

    def __init__(self) -> None:
        self.sharedHelpers: SharedHelpers = SharedHelpers()
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.helpers: Helpers = Helpers()
        from AI.AIHelpers import AIHelpers

        self.aiHelpers: AIHelpers = AIHelpers()

        self.algo: str = self.cfg.get_str("_REGEX")

        # in-memory cache: lang -> stage -> List[entry]
        self.compiled_cache: DefaultDict[str, Dict[str, List[Dict[str, Any]]]] = (
            defaultdict(dict)
        )
        self.banlist_en: List[str] = Synonyms().expand(
            self.helpers.get_banned_phrases_config_slot()
        )

        # runtime fields set in verify()
        self.soft_score_hard: float = 0.0
        self.soft_score_fuzzy: float = 0.0
        self.fuzzy_eval_after_hard: bool = True
        self.threshold: float = 0.0
        self.patterns: List[Dict[str, Any]] = []
        self.text_norm: str = ""
        self.window_max_chars: int = 0
        self.pref_suf_len: int = 0
        self.pref_suf_len = 0
        self.sep_class: str = ""  # default separator class (whitespace)

    def _compile_patterns(self, language: str, stage: str) -> List[Dict[str, Any]]:
        lang: str = (language or "en").lower()
        cached: List[Dict[str, Any]] | None = self.compiled_cache[lang].get(stage)
        if cached is not None:
            return cached

        banlist: List[str] = self.sharedHelpers.get_banlist_for_language(
            self.banlist_en, lang, self.algo
        )
        compiled_list: List[Dict[str, Any]] = []

        # local copies for readability
        pref_suf_len: int = self.pref_suf_len
        window_max: int = self.window_max_chars
        sep_class: str = self.sep_class  # expected to be a regex class like r"\s"

        for phrase in banlist:
            toks: list[str] = self.sharedHelpers.tokenize(phrase)
            if not toks:
                continue
            try:
                # Single-token handling
                if len(toks) == 1:
                    token: str = toks[0]
                    # Strict word boundary pattern (escaped)
                    strict_pat_text: str = rf"\b{re.escape(token)}\b"

                    # Fuzzy: if token short, match escaped token anywhere; otherwise use escaped prefix/suffix with separator window
                    if len(token) <= (pref_suf_len * 2):
                        fuzzy_pat_text: str = re.escape(token)
                    else:
                        prefix: str = re.escape(token[:pref_suf_len])
                        suffix: str = re.escape(token[-pref_suf_len:])
                        # sep allows up to window_max arbitrary separator chars (use sep_class as-is)
                        fuzzy_pat_text = (
                            rf"{prefix}{sep_class}{{0,{window_max}}}{suffix}"
                        )

                    cp_fuzzy: re.Pattern[str] | None = None
                    cp_strict: re.Pattern[str] | None = None
                    try:
                        cp_fuzzy = self.sharedHelpers.compile_regex(
                            fuzzy_pat_text, flags=re.IGNORECASE
                        )
                    except re.error as e:
                        if self.cfg.get_int("DEBUG_LEVEL") >= 2:
                            self.pretty.write(
                                "W",
                                "Regex Scorer",
                                f"Fuzzy compile failed for phrase {phrase!r}: {e}",
                            )
                        cp_fuzzy = None

                    try:
                        cp_strict = self.sharedHelpers.compile_regex(
                            strict_pat_text, flags=re.IGNORECASE
                        )
                    except re.error:
                        cp_strict = None

                    compiled_list.append(
                        {
                            "phrase": phrase,
                            "pattern": cp_fuzzy,
                            "strict_compiled": cp_strict,
                            "pattern_text": fuzzy_pat_text,
                            "meta": {
                                "type": "single_token",
                                "strict_pattern_text": strict_pat_text,
                                "fuzzy_pattern_text": fuzzy_pat_text,
                            },
                        }
                    )

                else:
                    pieces: List[str] = []
                    for t in toks:
                        # use escaped prefix for long tokens, full escape for short tokens
                        if len(t) <= pref_suf_len:
                            pieces.append(re.escape(t))
                        else:
                            pieces.append(re.escape(t[:pref_suf_len]))

                    sep: str = rf"{sep_class}{{0,{window_max}}}"
                    fuzzy_seq: str = sep.join(pieces)

                    # strict sequence: join full escaped tokens with a spacer that allows punctuation or whitespace
                    spacer: str = r"(?:\s*[\-_/.,;:]+\s*|\s+)"
                    strict_seq: str = (
                        r"\b" + spacer.join(re.escape(t) for t in toks) + r"\b"
                    )

                    cp_fuzzy = None
                    cp_strict = None
                    try:
                        cp_fuzzy = self.sharedHelpers.compile_regex(
                            fuzzy_seq, flags=re.IGNORECASE
                        )
                    except re.error as e:
                        if self.cfg.get_int("DEBUG_LEVEL") >= 2:
                            self.pretty.write(
                                "W",
                                "Regex Scorer",
                                f"Fuzzy compile failed for phrase {phrase!r}: {e}",
                            )
                        cp_fuzzy = None

                    try:
                        cp_strict = self.sharedHelpers.compile_regex(
                            strict_seq, flags=re.IGNORECASE
                        )
                    except re.error:
                        cp_strict = None

                    compiled_list.append(
                        {
                            "phrase": phrase,
                            "pattern": cp_fuzzy,
                            "strict_compiled": cp_strict,
                            "pattern_text": fuzzy_seq,
                            "meta": {
                                "type": "multi_token",
                                "strict_pattern_text": strict_seq,
                                "fuzzy_pattern_text": fuzzy_seq,
                            },
                        }
                    )

            except re.error:
                self.pretty.write(
                    "W", "Regex Scorer", f"Invalid pattern for phrase: {phrase}"
                )
                continue

        # cache in-memory only
        self.compiled_cache[lang][stage] = compiled_list
        if self.cfg.get_int("DEBUG_LEVEL") >= 1:
            self.pretty.write(
                "O",
                "Cache build Regex Banned",
                f"Built compiled Regex cache with {len(compiled_list)} entries for language {language} stage: {stage}",
            )
        return compiled_list

    def verify(
        self, text: str, language: str, stage: str
    ) -> List[ComplianceAlgoResult]:
        """
        Run detection and return ComplianceAlgoResult list.
        This method configures runtime params, compiles patterns (in-memory), normalizes text, and calls return_algo_result().
        """
        if not text:
            return []

        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        self.soft_score_hard = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.SOFT_SCORE_HARD"
        )
        self.soft_score_fuzzy = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.SOFT_SCORE_FUZZY"
        )
        self.fuzzy_eval_after_hard = self.cfg.get_bool(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.FUZZY_REGEX_EVAL_AFTER_HARD"
        )
        self.threshold = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD"
        )
        self.threshold_min = self.cfg.get_float(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.THRESHOLD_MIN"
        )
        self.helpers.require_set(
            soft_score_hard=self.soft_score_hard, threshold=self.threshold
        )

        # coerce tunables
        try:
            self.window_max_chars = self.cfg.get_int(
                f"{compliance_config_slot}.PIPELINE.{self.algo}.WINDOW_MAX_CHARS",
                self.window_max_chars,
            )
        except (TypeError, ValueError):
            pass
        try:
            self.pref_suf_len = self.cfg.get_int(
                f"{compliance_config_slot}.PIPELINE.{self.algo}.PREFIX_SUFFIX_LEN",
                self.pref_suf_len,
            )
        except (TypeError, ValueError):
            pass
        self.sep_class = self.cfg.get_str(
            f"{compliance_config_slot}.PIPELINE.{self.algo}.SEPARATOR_CLASS", ""
        )

        self.helpers.require_set(
            window_max_chars=self.window_max_chars,
            pref_suf_len=self.pref_suf_len,
            sep_class=self.sep_class,
        )

        lang: str = (language or "en").lower()
        self.patterns = self._compile_patterns(lang, stage) or []
        self.text_norm = self.sharedHelpers.normalize(text)
        if self.cfg.get_int("DEBUG_LEVEL") >= 4:
            self.pretty.write(
                "D",
                "Regex Scorer",
                f"Running with: soft_score_hard: {self.soft_score_hard} soft_score_fuzzy: {self.soft_score_fuzzy} threshold: {self.threshold} fuzzy_eval_after_hard: {self.fuzzy_eval_after_hard}",
            )
        return self.return_algo_result()

    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        results: List[ComplianceAlgoResult] = []
        for entry in self.patterns:
            # Try strict (word-boundary) match first for full confidence
            strict_m: re.Match[str] | None = None
            strict_compiled: re.Pattern[str] | None = entry.get("strict_compiled")
            if strict_compiled is not None:
                try:
                    strict_m = strict_compiled.search(self.text_norm)
                except Exception:
                    strict_m = None

            if strict_m is not None:
                detail = strict_m.group(0)
                score = float(self.soft_score_hard)
                if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                    self.pretty.write(
                        "D",
                        "Regex Scorer",
                        f"Strict match for phrase {entry.get('phrase')!r}: {detail!r}",
                    )
            elif self.fuzzy_eval_after_hard:
                # Fuzzy evaluation only runs when the switch is enabled
                m: re.Match[str] | None = None
                try:
                    m = entry["pattern"].search(self.text_norm)
                except Exception:
                    if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                        self.pretty.write(
                            "W",
                            "Regex Scorer",
                            f"Runtime error applying pattern for phrase: {entry.get('phrase')}",
                        )
                if m is not None:
                    detail = m.group(0)
                    score = float(self.soft_score_fuzzy)
                    if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                        self.pretty.write(
                            "D",
                            "Regex Scorer",
                            f"Fuzzy-only match for phrase {entry.get('phrase')!r}: {detail!r} (fuzzy score)",
                        )
                else:
                    score = 0.0
                    detail = None
            else:
                score = 0.0
                detail = None

            # Build serializable meta (do not include compiled objects)
            entry_meta: dict[str, Any] = entry.get("meta", {}) or {}
            meta: Dict[str, Any] = {
                "type": entry_meta.get("type"),
                "pattern_text": entry.get("pattern_text"),
            }
            if entry_meta.get("strict_pattern_text"):
                meta["strict_pattern_text"] = entry_meta.get("strict_pattern_text")

            results.append(
                ComplianceAlgoResult(
                    algo=self.algo,
                    phrase=entry["phrase"],
                    score=float(score),
                    threshold=float(self.threshold),
                    detail=detail,
                    meta=meta,
                )
            )

        return results
