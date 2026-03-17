from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import (TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional, Set,
                    Tuple)

import wcwidth  # type: ignore[reportMissingTypeStubs]

from Algos.ComplianceAlgoResult import (InternalResult, PhraseMeta,
                                        ResultsForPrint)
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import BRIGHT_BLUE, BRIGHT_RED, CYAN, GREEN, RESET
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.Helpers import Helpers

if TYPE_CHECKING:
    from AI.AIHelpers import AIHelpers


class Accumulator(SingletonMixin):
    """
    Singleton class to accumulate per-chunk detection outputs for compliance checks.
    Provides methods to add results, show accumulated results, and consolidate phrase-level scores.

    Naming:
        - breadth = number of distinct algos that provided any score for a phrase (score > 0.0).
        - depth   = number of distinct algos that passed their per-algo threshold (score >= threshold).

    Refactor notes:
        - Consolidation stores structured PhraseMeta under algo_map["meta"].
        - Presentation reads meta directly (no fragile string parsing).
    """

    def __init__(
        self,
        *,
        cfg: Config | None = None,
        pretty: PrettyWriter | None = None,
        helpers: Helpers | None = None,
    ):
        # Initialize only once (singleton pattern)
        if self._initialized:
            return
        self._initialized = True

        # Raw rows from all chunks (unfiltered)
        self._raw_results: List[InternalResult] = []

        # Per-chunk boolean decisions (did this chunk require human review)
        self._per_chunk_decisions: List[bool] = []

        # Per-chunk lists of rows that passed per-algo thresholds
        self._per_chunk_filtered: List[List[InternalResult]] = []

        # Per-chunk phrase hits mapping phrase -> set(algo names) (counts algos with any score)
        self._per_chunk_phrase_hits: List[Dict[str, Set[str]]] = []

        self.cfg: Config = cfg or Config()
        from AI.AIHelpers import AIHelpers as _AIHelpers

        self.aiHelpers: AIHelpers = _AIHelpers()
        self.helpers: Helpers = helpers or Helpers()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.default_algos: list[str] = self.cfg.get_list("_DEFAULT_ALGOS")

        self.keybert: str = self.cfg.get_str("_KEYBERT")
        self.jaccard: str = self.cfg.get_str("_JACCARD")
        self.regex: str = self.cfg.get_str("_REGEX")
        self.cosine: str = self.cfg.get_str("_COSINE")
        self.levenshtein: str = self.cfg.get_str("_LEVENSHTEIN")

    # ----------------- adders -----------------
    def add_results(self, results: list[Any], stage: str) -> tuple[bool, list[Any]]:
        """
        Add results for a chunk and evaluate if human review is required.
        Returns (human_review_required, filtered_results).
        """
        if not results:
            self._raw_results.extend([])
            self._per_chunk_decisions.append(False)
            self._per_chunk_filtered.append([])
            self._per_chunk_phrase_hits.append({})
            return False, []

        human_review, filtered, breadth_hits = self._evaluate_human_review(
            results, stage
        )

        # Wrap all results into InternalResult
        tmp = [InternalResult.from_base(r) for r in results]
        self._raw_results.extend(tmp)

        tmp = [InternalResult.from_base(r) for r in filtered]
        self._per_chunk_filtered.append(tmp)
        self._per_chunk_decisions.append(bool(human_review))
        self._per_chunk_phrase_hits.append(breadth_hits)

        return bool(human_review), filtered

    # ----------------- helpers -----------------
    def _evaluate_human_review(
        self, results: list[Any], stage: str
    ) -> tuple[bool, list[Any], DefaultDict[Any, set[Any]]]:
        """
        Evaluate if human review is required for a chunk based on depth and breadth.
        Returns (human_review_required, depth_filtered_results, breadth_hits).
        """
        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        required_depth: int = self.cfg.get_int(
            f"{compliance_config_slot}.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        )
        required_breadth: int = self.cfg.get_int(
            f"{compliance_config_slot}.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE"
        )
        self.helpers.require_set(
            required_depth=required_depth, required_breadth=required_breadth
        )

        # Sort by score/threshold ratio so the strongest evidence appears first
        results_sorted: list[Any] = sorted(
            results,
            key=lambda r: (
                (r.score / r.threshold) if (r.score is not None and r.threshold) else 0
            ),
            reverse=True,
        )

        # Depth: threshold-filtered rows: only rows where score >= threshold
        depth_filtered: list[Any] = [
            r
            for r in results_sorted
            if r.score is not None
            and r.threshold is not None
            and r.score >= r.threshold
        ]

        # Count distinct algos that passed threshold in this chunk (depth)
        algos_passing_threshold: set[str] = {r.algo for r in depth_filtered}
        depth_algo_count: int = len(algos_passing_threshold)

        # Breadth: count ALL algos that detected a phrase with a non-zero score (threshold-agnostic)
        breadth_hits: DefaultDict[Any, set[Any]] = defaultdict(set)
        for r in results_sorted:
            if r.score is not None and r.score > 0.0:
                breadth_hits[r.phrase].add(r.algo)

        # Phrase-level depth consensus check (using breadth_hits counts)
        phrase_depth_consensus: bool = any(
            len(algos) >= required_breadth for algos in breadth_hits.values()
        )

        # Final chunk-level human review decision: depth (document-level) OR phrase-level depth consensus
        human_review: bool = (
            depth_algo_count >= required_depth
        ) or phrase_depth_consensus

        return bool(human_review), depth_filtered, breadth_hits

    # ----------------- DEBUG OUTPUT -----------------
    def _show_raw_results_for_debug(self) -> None:
        """
        Print all raw results for debugging purposes.
        """
        self.pretty.write("D", "KeyWrdChk Debug", "Showing ALL raw hits (debug >= 4)")
        header: str = f"{'Phrase':<24} {'Algo':<18} {'Score':<10} {'Threshold':<10}"
        self.pretty.write("D", "KeyWrdChk Debug", header)
        for r in self._raw_results:
            algo_display: str = self.helpers.get_label_alias(r.algo) if r.algo else ""
            row: str = (
                f"{r.phrase:<24} {algo_display:<18} {r.score:<10.4f} {r.threshold:<10.4f}"
            )
            self.pretty.write("D", "KeyWrdChk Debug", row)

    # ----------------- finalization / view -----------------
    def show_accumulated(self, stage: str) -> tuple[bool, list[Any]]:
        """
        Show accumulated results for all chunks and decide if human review is required.
        Returns (human_review_required, phrase_table).
        """
        if self.cfg.get_int("DEBUG_LEVEL") >= 4:
            self._show_raw_results_for_debug()

        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        required_depth: int = self.cfg.get_int(
            f"{compliance_config_slot}.PIPELINE.REQUIRED_ALGOS_ABOVE_THRESHOLD"
        )
        required_breadth: int = self.cfg.get_int(
            f"{compliance_config_slot}.PIPELINE.REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE"
        )
        self.helpers.require_set(
            required_depth=required_depth, required_breadth=required_breadth
        )
        raw: Any = self.cfg.get(f"{compliance_config_slot}.PIPELINE.ALGOS_TO_PROCESS")
        algos_to_process: OrderedDict[str, bool] = self.helpers.make_ordered_dict(raw)

        # 1) collect and annotate hits across chunks
        threshold_passed_rows: List[InternalResult]
        merged_phrase_hits: Dict[str, Set[str]]
        threshold_passed_rows, merged_phrase_hits = self._prepare_merged_data(
            required_breadth
        )

        # 2) decide which phrases/rows to show and consolidate into phrase_table
        phrase_table: dict[str, dict[str, Any]] = self._evaluate_and_consolidate(
            threshold_passed_rows,
            merged_phrase_hits,
            required_depth,
            required_breadth,
            stage,
        )

        # 3) final messages, build return dict, cleanup
        effective_depth_pass_hits: int = len(
            {r.algo for r in threshold_passed_rows if r.algo}
        )
        depth_trigger: bool = effective_depth_pass_hits >= required_depth

        breadth_trigger: bool = any(
            len(a) >= required_breadth for a in merged_phrase_hits.values()
        )
        effective_breadth_score_hits: int = max(
            (len(a) for a in merged_phrase_hits.values()), default=0
        )

        final_human_review: bool = bool(phrase_table)
        warn_flag: bool = depth_trigger or breadth_trigger
        if final_human_review:
            self._show_results(phrase_table, algos_to_process, warn_flag)

        self._print_ensemble_messages(
            breadth_trigger,
            depth_trigger,
            effective_depth_pass_hits,
            required_depth,
            effective_breadth_score_hits,
            required_breadth,
        )

        y: List[ResultsForPrint] = self._decompose_score_str(phrase_table)

        # Cleanup internal buffers so accumulator can be reused
        self._raw_results.clear()
        self._per_chunk_decisions.clear()
        self._per_chunk_filtered.clear()
        self._per_chunk_phrase_hits.clear()

        return (breadth_trigger or depth_trigger), y

    # ----------------- helper 1 -----------------
    def _prepare_merged_data(
        self, required_breadth: int
    ) -> Tuple[List[InternalResult], Dict[str, Set[str]]]:
        """
        Prepare merged phrase-level data from all chunks for consolidation.
        Returns (threshold_passed_rows, merged_phrase_hits).
        """
        threshold_passed_rows: List[InternalResult] = []
        for chunk in self._per_chunk_filtered:
            threshold_passed_rows.extend(chunk)

        merged_phrase_hits: Dict[str, Set[str]] = defaultdict(set)
        for chunk_hits in self._per_chunk_phrase_hits:
            for phrase, algos in chunk_hits.items():
                merged_phrase_hits[phrase].update(algos)

        for r in threshold_passed_rows:
            if r.algo:
                breadth_count = len(merged_phrase_hits.get(r.phrase, set()))
                r.matched_algos_count = breadth_count

        existing_phrases: set[str] = {r.phrase for r in threshold_passed_rows}
        for phrase, algos in merged_phrase_hits.items():
            breadth_count: int = len(algos)
            if breadth_count >= required_breadth and phrase not in existing_phrases:
                threshold_passed_rows.append(
                    InternalResult(
                        algo=None,
                        phrase=phrase,
                        score=None,
                        threshold=None,
                        detail=None,
                        matched_algos_count=breadth_count,
                    )
                )
        return threshold_passed_rows, merged_phrase_hits

    # ----------------- helper 2 -----------------
    def _evaluate_and_consolidate(
        self,
        threshold_passed_rows: List[InternalResult],
        merged_phrase_hits: Dict[str, Set[str]],
        required_depth: int,
        required_breadth: int,
        stage: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate and consolidate phrase-level results for presentation.
        Returns phrase_table.
        """
        # Determine phrases to show based on breadth consensus
        phrases_with_enough_breadth: set[str] = {
            phrase.strip()
            for phrase, algos in merged_phrase_hits.items()
            if len(algos) >= required_breadth
        }

        threshold_exceeded_to_show: list[InternalResult] = [
            r
            for r in threshold_passed_rows
            #        if r.phrase and r.phrase.strip() in phrases_with_enough_breadth
        ]

        if not threshold_exceeded_to_show and phrases_with_enough_breadth:
            if self.cfg.get_int("DEBUG_LEVEL") >= 4:
                self.pretty.write(
                    "D",
                    "KeyWrdChk Debug",
                    "threshold_exceeded_to_show empty; creating synthetic representative rows for consolidation",
                )
            threshold_exceeded_to_show = []
            for phrase in phrases_with_enough_breadth:
                algos = merged_phrase_hits.get(phrase, set())
                threshold_exceeded_to_show.append(
                    InternalResult(
                        algo=None,
                        phrase=phrase,
                        score=None,
                        threshold=None,
                        detail=None,
                        matched_algos_count=len(algos),
                    )
                )

        phrase_table: dict[str, dict[str, Any]] = self._consolidate_to_one_row(
            threshold_exceeded_to_show, required_depth, required_breadth, stage
        )

        effective_depth_pass_hits: int = len(
            {r.algo for r in threshold_passed_rows if r.algo}
        )
        breadth_trigger: bool = effective_depth_pass_hits >= required_depth
        if not phrase_table and breadth_trigger:
            breadth_phrases: set[str] = {
                r.phrase for r in threshold_passed_rows if r.algo
            }
            if breadth_phrases:
                rows_for_breadth: list[InternalResult] = [
                    r for r in threshold_passed_rows if r.phrase in breadth_phrases
                ]
                phrase_table = self._consolidate_to_one_row(
                    rows_for_breadth, required_depth, required_breadth, stage
                )

        return phrase_table

    def _decompose_score_str(
        self, phrase_table: dict[str, dict[str, Any]]
    ) -> List[ResultsForPrint]:
        """
        Decompose phrase_table into a list of ResultsForPrint for output.
        """
        rows: List[ResultsForPrint] = []
        for phrase, algo_map in phrase_table.items():
            for algo, data in algo_map.items():
                if algo == "meta":
                    continue
                if algo == "algos_matched":
                    continue
                score_str = data.score_str
                if score_str == "-/-":
                    continue
                failing = score_str.startswith("*")
                if failing:
                    score_str = score_str[1:]
                thresh_val: str = "0"
                if score_str == "Disabled":
                    score_val = 0
                    threshold = 0
                else:
                    score_val, thresh_val = score_str.split("/")
                score = float(score_val)
                threshold = float(thresh_val)
                rows.append(
                    ResultsForPrint(
                        algo=algo,
                        phrase=phrase,
                        score=score,
                        score_str=score_str,
                        threshold=threshold,
                        detail=data.detail,
                        matched_algos_count=data.matched_algos_count,
                    )
                )
        return rows

    def _pad_display(self, text: str, width: int, align: str = "left") -> str:
        """
        Pad text for display, accounting for visible width.
        """
        visible: int = sum(wcwidth.wcwidth(c) for c in text)
        pad: int = max(0, width - visible)
        if align == "left":
            return text + " " * pad
        else:
            return " " * pad + text

    # ----------------- presentation helpers -----------------
    def _color_count(self, count: Optional[int], required: Optional[int]) -> str:
        """
        Return colored string for count vs required for display.
        """
        if count is None or required is None:
            return "-"
        return f"{BRIGHT_RED if count >= required else GREEN}{count}{RESET}"

    def _format_colored_meta(self, meta: PhraseMeta, width: int) -> str:
        """
        Format colored meta display for phrase summary.
        """
        # Build plain and colored parts
        plain = meta.plain()
        # Build colored tokens
        depth_colored = (
            f"{self._color_count(meta.depth_count, meta.depth_req)}/{meta.depth_req}"
        )
        breadth_colored = f"{self._color_count(meta.breadth_count, meta.breadth_req)}/{meta.breadth_req}"
        # Split padded_plain into two slots by finding the two tokens
        # We know plain looks like "D/R  B/R" (two spaces between)
        left_token, _, right_token = plain.partition("  ")
        left_slot = self._pad_display(left_token, width // 2, align="left")
        right_slot = self._pad_display(
            right_token, width - len(left_slot), align="left"
        )
        left_colored = left_slot.replace(left_token, depth_colored, 1)
        right_colored = right_slot.replace(right_token, breadth_colored, 1)
        return left_colored + right_colored

    # ----------------- helper 3 (printing) -----------------
    def _show_results(
        self,
        phrase_table: dict[str, dict[str, Any]],
        algos_to_process: OrderedDict[str, bool],
        warn_flag: bool,
    ):
        """
        Print phrase-level results table for compliance checks.
        """
        user_order: list[str] = list(algos_to_process.keys())
        if self.levenshtein in user_order:
            user_order.remove(self.levenshtein)
        remaining: list[str] = [a for a in self.default_algos if a not in user_order]
        all_algos: list[str] = user_order + remaining

        phrase_w: int = 30
        meta_w: int = 14
        col_w: int = 22
        marker_w: int = 1
        value_w: int = col_w - marker_w

        last_algo: str | None = all_algos[-1] if all_algos else None

        header_parts: list[str] = [
            self._pad_display("Phrase", phrase_w, align="left"),
            self._pad_display("dpt / brth", meta_w, align="left"),
        ]
        for algo in user_order:
            algo_alias: str = self.helpers.get_label_alias(algo)
            label: str = (
                algo_alias
                if algo == last_algo
                else f"{algo_alias} {Symbols.sym_arrow()}"
            )
            header_parts.append(self._pad_display(label, col_w, align="left"))
        for algo in remaining:
            algo_alias: str = self.helpers.get_label_alias(algo)
            label: str = (
                algo_alias
                if algo == last_algo
                else f"{algo_alias} {Symbols.sym_fail()} "
            )
            header_parts.append(self._pad_display(label, col_w, align="left"))

        header: str = " ".join(header_parts)
        level: str = "W" if warn_flag else "O"
        self.pretty.write(level, "KeyWrdChk Summary", header, max_line_length=999)

        for phrase in sorted(phrase_table.keys(), key=str.lower):
            algo_map: dict[str, Any] = phrase_table[phrase]

            # Prefer structured meta if present, otherwise try to parse legacy string
            meta_obj: Optional[PhraseMeta] = algo_map.get("meta")
            if meta_obj is None:
                # fallback: try to parse legacy "algos_matched" string
                raw_meta = algo_map.get("algos_matched", "-/-")
                try:
                    tokens = [t for t in raw_meta.split() if t]
                    if len(tokens) >= 2:
                        d_token, b_token = tokens[0], tokens[1]
                    elif len(tokens) == 1:
                        d_token, b_token = tokens[0], "-/-"
                    else:
                        d_token, b_token = "-/-", "-/-"
                    d_count_s, d_req_s = (
                        d_token.split("/") if "/" in d_token else ("-", "-")
                    )
                    b_count_s, b_req_s = (
                        b_token.split("/") if "/" in b_token else ("-", "-")
                    )
                    d_count = int(d_count_s) if d_count_s.isdigit() else 0
                    d_req = int(d_req_s) if d_req_s.isdigit() else 0
                    b_count = int(b_count_s) if b_count_s.isdigit() else 0
                    b_req = int(b_req_s) if b_req_s.isdigit() else 0
                    meta_obj = PhraseMeta(
                        depth_count=d_count,
                        depth_req=d_req,
                        breadth_count=b_count,
                        breadth_req=b_req,
                    )
                except Exception:
                    meta_obj = PhraseMeta(
                        depth_count=0, depth_req=0, breadth_count=0, breadth_req=0
                    )

            meta_display: str = self._format_colored_meta(meta_obj, meta_w)

            parts: list[str] = [
                self._pad_display(phrase, phrase_w, align="left"),
                meta_display,
            ]

            for algo in all_algos:
                raw: ResultsForPrint | str = algo_map.get(algo, "-/-")
                score_str: str = (
                    raw.score_str if isinstance(raw, ResultsForPrint) else raw
                )

                if score_str.startswith("*"):
                    marker = "*"
                    value = score_str[1:]
                    cell_color = BRIGHT_BLUE
                else:
                    marker = " "
                    value = score_str
                    if score_str == "Disabled":
                        cell_color = CYAN
                    else:
                        cell_color = BRIGHT_RED if score_str != "-/-" else GREEN

                cell = f"{cell_color}{marker}{self._pad_display(value, value_w, align='left')}{RESET}"
                parts.append(cell)

            row = " ".join(parts)
            self.pretty.write("W", "KeyWrdChk Summary", row, max_line_length=999)

    # ----------------- helper 4 (messages) -----------------
    def _print_ensemble_messages(
        self,
        breadth_trigger: bool,
        depth_trigger: bool,
        effective_depth_pass_hits: int,
        required_depth: int,
        effective_breadth_score_hits: int,
        required_breadth: int,
    ) -> None:
        """Print summary messages for depth and breadth triggers."""
        depth_icon: str = "W" if depth_trigger else "O"
        depth_color: str = BRIGHT_RED if depth_trigger else GREEN
        depth_msg: str = (
            f"{effective_depth_pass_hits} algos passed threshold vs. required {required_depth}"
        )
        self.pretty.write(depth_icon, "KeyWrdChk Depth", depth_msg, color=depth_color)

        breadth_icon: str = "W" if breadth_trigger else "O"
        breadth_color: str = BRIGHT_RED if breadth_trigger else GREEN
        breadth_msg: str = (
            f"{effective_breadth_score_hits} algos had a score vs. required {required_breadth}"
        )
        self.pretty.write(
            breadth_icon, "KeyWrdChk Breadth", breadth_msg, color=breadth_color
        )

    def _consolidate_to_one_row(
        self,
        rows_to_show: list[InternalResult],
        required_depth: int,
        required_breadth: int,
        stage: str = "",
    ) -> dict[str, dict[str, Any]]:
        """
        Consolidate phrase-level results into a single row for presentation.
        Returns phrase_table.
        """
        # Build score_map from raw InternalResult objects
        score_map: DefaultDict[str, DefaultDict[str, list[InternalResult]]]
        score_map = defaultdict(lambda: defaultdict(list))

        compliance_config_slot: str = self.helpers.get_compliance_config_slot(stage)
        raw: Any = self.cfg.get(f"{compliance_config_slot}.PIPELINE.ALGOS_TO_PROCESS")
        algos_to_process: OrderedDict[str, bool] = self.helpers.make_ordered_dict(raw)
        for r in self._raw_results:
            if r.algo is None:
                continue
            if r.score is not None:
                score_map[r.phrase][r.algo].append(r)

        table: dict[str, dict[str, Any]] = defaultdict(dict)
        if not rows_to_show:
            return table
        phrases: set[str] = {r.phrase for r in rows_to_show}

        for phrase in phrases:
            # Cache real hits per algo so we don't repeat the same filtering twice
            algo_real_hits: Dict[str, List[InternalResult]] = {}
            breadth_count = 0
            depth_pass_count = 0

            for algo in self.default_algos:
                hits: list[InternalResult] = score_map[phrase].get(algo, [])
                real_hits: list[InternalResult] = [
                    h for h in hits if h.score is not None and h.score > 0.0
                ]
                if not real_hits:
                    continue
                algo_real_hits[algo] = real_hits
                breadth_count += 1
                if any(
                    h.threshold is not None
                    and h.score is not None
                    and h.score >= h.threshold
                    for h in real_hits
                ):
                    depth_pass_count += 1

            # Store structured meta
            table[phrase]["meta"] = PhraseMeta(
                depth_count=depth_pass_count,
                depth_req=required_depth,
                breadth_count=breadth_count,
                breadth_req=required_breadth,
            )
            # keep legacy string for compatibility
            table[phrase][
                "algos_matched"
            ] = f"{depth_pass_count}/{required_depth}  {breadth_count}/{required_breadth}"

            # Build per-algo ResultsForPrint using cached hits. Include explicit
            # placeholders for algos that are present in the ordered config and disabled.
            for algo in self.default_algos:
                enabled: bool = bool(algos_to_process.get(algo, True))
                cached_hits: list[InternalResult] | None = algo_real_hits.get(algo)

                if not enabled:
                    table[phrase][algo] = ResultsForPrint(
                        algo=algo,
                        phrase=phrase,
                        score=None,
                        score_str="Disabled",
                        threshold=None,
                        detail=None,
                        matched_algos_count=breadth_count,
                    )
                    continue

                if not cached_hits:
                    # No hits for this enabled algo; leave absent (will display as "-/-").
                    continue

                passing: list[InternalResult] = [
                    h
                    for h in cached_hits
                    if h.threshold is not None
                    and h.score is not None
                    and h.score >= h.threshold
                ]
                failing: list[InternalResult] = [
                    h
                    for h in cached_hits
                    if h.threshold is not None
                    and h.score is not None
                    and h.score < h.threshold
                ]

                if passing:
                    best: InternalResult = max(passing, key=lambda h: h.score or 0.0)
                    score_str: str = f"{best.score:.4f}/{best.threshold:.4f}"
                else:
                    best = max(failing, key=lambda h: h.score or 0.0)
                    score_str = f"*{best.score:.4f}/{best.threshold:.4f}"

                table[phrase][algo] = ResultsForPrint(
                    algo=algo,
                    phrase=phrase,
                    score=best.score,
                    score_str=score_str,
                    threshold=best.threshold,
                    detail=best.detail,
                    matched_algos_count=breadth_count,
                )

        return table
