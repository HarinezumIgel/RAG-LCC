from typing import List, cast

from Algos.ComplianceAlgoResult import ComplianceAlgoResult, ScorerBase
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class LevenshteinScorer(ScorerBase):
    """
    Computes the Levenshtein edit distance between two strings.
    Useful for typo detection, fuzzy compliance checks, and detecting obfuscated
    banned phrases.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        from AI.AIHelpers import AIHelpers

        self.aiHelpers: AIHelpers = AIHelpers()
        self.helpers: Helpers = helpers or Helpers()
        self.perf_logger: PerfLogger = PerfLogger()
        self.algo: str = "Levenshtein"
        self.threshold: float | None = None
        self.compliance_results: List[ComplianceAlgoResult] = []

    def _distance(self, a: str | None, b: str | None) -> int:
        """
        Compute the Levenshtein edit distance (Wagner–Fischer).
        Returns integer edit distance (0 means identical).
        None inputs are treated as empty strings.
        """
        a = a or ""
        b = b or ""
        len_a: int = len(a)
        len_b: int = len(b)
        dp: list[list[int]] = [[0] * (len_b + 1) for _ in range(len_a + 1)]
        for i in range(len_a + 1):
            dp[i][0] = i
        for j in range(len_b + 1):
            dp[0][j] = j
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost: int = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[len_a][len_b]

    def _verify_impl(
        self, compliance_results: List[ComplianceAlgoResult], stage: str
    ) -> List[ComplianceAlgoResult]:
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        self.threshold = cast(
            "float | None",
            self.cfg.get(
                f"{compliance_config_slot}.PIPELINE.Regex.{self.algo}.THRESHOLD"
            ),
        )
        self.helpers.require_set(threshold_default=self.threshold)
        self.compliance_results = compliance_results
        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D", "Levenshtein Scorer", f"Running with: threshold: {self.threshold}"
            )
        return self.return_algo_result()

    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        results: List[ComplianceAlgoResult] = []
        for r in self.compliance_results:
            if r.score == 0.0:
                continue
            match_text: str | None = r.detail
            phrase: str = r.phrase
            if not match_text or not phrase:
                continue
            else:
                dist: int = self._distance(match_text, phrase)
                max_len: int = max(len(match_text), len(phrase))
                score = 1.0 - (dist / max_len)
                results.append(
                    ComplianceAlgoResult(
                        algo=self.algo,
                        phrase=phrase,
                        score=score,
                        threshold=self.threshold,
                        detail=f"levenshtein={score:.3f} threshold: {self.threshold} Regex match: {r.detail}",
                    )
                )
        return results
