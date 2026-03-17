from typing import Any, Dict, List, Optional, Tuple

from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Gui.Symbols import Symbols
from Helpers.Helpers import Helpers


class PipelineSettingsSummarizer:

    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.helpers: Helpers = Helpers()
        self.friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        cfg_default: list[Any] = self.cfg.get_list("_DEFAULT_ALGOS")
        self.default_algos: list[str] = (
            [str(x) for x in cfg_default]
            if cfg_default
            else ["Cosine", "Jaccard", "Regex", "Keybert"]
        )

    # -------------------------
    # Public API
    # -------------------------
    def display(self) -> Tuple[bool, List[Any]]:
        """
        Returns (any_disabled_stage, disabled_algos_list).
        Prints a compact summary via self.pretty.write.
        """
        algos_cfg: Any = {}
        pipeline_ok: bool = False
        prompt_ok: bool = False
        slot: str = self.helpers.get_compliance_config_slot("PIPELINE_CHECK")
        algos_cfg = self.cfg.get(f"{slot}.PIPELINE.ALGOS_TO_PROCESS")
        pipeline_row: Dict[str, Dict[str, Optional[bool]]]
        pipeline_row, pipeline_ok = self._evaluate_algo_present(algos_cfg)
        prompt_row: Dict[str, Dict[str, Optional[bool]]]
        prompt_row, prompt_ok = self._build_prompt_row()

        results: list[tuple[str, bool, Dict[str, Dict[str, Optional[bool]]]]] = [
            ("PROMPT_CHECK", not prompt_ok, prompt_row),
            ("PIPELINE_CHECK", not pipeline_ok, pipeline_row),
        ]

        self._print_results(results)

        return (pipeline_ok or prompt_ok), results

    # -------------------------
    # Internal helpers
    # -------------------------

    def _evaluate_algo_present(
        self, algos_cfg: Any
    ) -> Tuple[Dict[str, Dict[str, Optional[bool]]], bool]:
        row: Dict[str, Dict[str, Optional[bool]]] = {}
        if not isinstance(algos_cfg, dict):
            for a in self.default_algos:
                row[a] = {"enabled": False}
        else:
            algos_dict: dict[str, Any] = dict(algos_cfg)  # type: ignore[reportUnknownArgumentType]
            for a in self.default_algos:
                if a in algos_dict:
                    enabled = bool(algos_dict[a])
                    row[a] = {"enabled": enabled}
                else:
                    row[a] = {"enabled": False}
            for a, en in algos_dict.items():
                if a not in row:
                    row[a] = {"enabled": bool(en)}
        ok: bool = self._all_enabled(row)
        return row, ok

    def _set_algos_to_none(self) -> Dict[str, Dict[str, Optional[bool]]]:
        row: Dict[str, Dict[str, Optional[bool]]] = {}
        for a in self.default_algos:
            row[a] = {"enabled": None}
        return row

    def _build_prompt_row(self) -> Tuple[Dict[str, Dict[str, Optional[bool]]], bool]:
        row: Dict[str, Dict[str, Optional[bool]]] = {}
        prompt_algos: Any = None
        if self.friendly_name != "RAGLoad":
            prompt_slot: str = self.helpers.get_compliance_config_slot("PROMPT_CHECK")
            prompt_check: bool = self.cfg.get_bool(f"{prompt_slot}.Check", False)
            prompt_algos = self.cfg.get(f"{prompt_slot}.PIPELINE.ALGOS_TO_PROCESS")
        else:
            prompt_check = False

        if not prompt_check:
            row = self._set_algos_to_none()
        else:
            row, _ = self._evaluate_algo_present(prompt_algos)

        PROMPT_ALGO = "Prompt Check"
        if self.friendly_name == "RAGLoad":
            row[PROMPT_ALGO] = {"enabled": None}
            ok = True
        else:
            row[PROMPT_ALGO] = {"enabled": prompt_check}
            ok = self._all_enabled(row)
        return (row, ok)

    def _all_enabled(self, per_algo: Dict[str, Dict[str, Optional[bool]]]) -> bool:
        non_na: list[Optional[bool]] = [
            v["enabled"] for v in per_algo.values() if v["enabled"] is not None
        ]
        return all(bool(v) for v in non_na) if non_na else True

    def _print_results(
        self, results: list[tuple[str, bool, Dict[str, Dict[str, Optional[bool]]]]]
    ) -> None:
        # gather config from PIPELINE_CHECK and DISPLAY (DISPLAY only for mask resolution)
        masking_slot: str = self.helpers.get_compliance_config_slot("MASKING")
        mask: bool = self.cfg.get_bool(f"{masking_slot}.APPLY_MASKING")
        lines: list[str] = ["*** Banned word check status by stage and algorithm:"]
        # Pad stage names to uniform width so algo columns align across rows
        stage_width: int = max(
            (len(s) for s, _, _ in results),
            default=len("PIPELINE_CHECK"),
        )
        stage_width = max(stage_width, len("MASKING"))
        for stage, _, per_algo in results:
            if per_algo:
                parts: list[str] = []
                for algo, info in per_algo.items():
                    algo_alias: str = self.helpers.get_label_alias(algo)
                    en: Optional[bool] = info.get("enabled")
                    icon: str = (
                        Symbols.sym_ok()
                        if en is True
                        else (
                            Symbols.sym_fail() if en is False else Symbols.sym_neutral()
                        )
                    )
                    parts.append(f"{algo_alias}: {icon}")
                row_str: str = "  ".join(parts)
                lines.append(f"- {stage:<{stage_width}s}: {row_str}")
            else:
                lines.append(
                    f"- {stage:<{stage_width}s}: (no algorithm details available)"
                )

        mask_icon: str = (
            Symbols.sym_ok()
            if mask is True
            else (Symbols.sym_fail() if mask is False else Symbols.sym_neutral())
        )
        suffix: str = ""
        if self.friendly_name in ("RAGLoad", "DocClassify"):
            suffix = " applied on load"
        elif self.friendly_name == "RAGChat":
            suffix = " applied on final result"
        lines.append(f"- {'MASKING':<{stage_width}s}: {mask_icon}{suffix}")

        severity: str = "O" if all(not r[1] for r in results) else "W"

        for l in lines:
            self.pretty.write(severity, "Banned Word Check", l)
        self.pretty.write("N", "", "")
        banned_count: int = self._print_banned()
        masking_count: int = self._print_masking()
        warn_user: bool = severity == "W" and (banned_count > 0 or masking_count > 0)
        sleep: int = 10 if warn_user else 1
        msg: str = (
            f"Sleeping for {sleep} seconds to give you time to review the settings..."
            if warn_user
            else "All checks passed. Starting immediately..."
        )
        self.helpers.pretty_sleep(sleep, message=msg)

    def _print_banned(self) -> int:
        banned_config: list[str] = self.helpers.get_banned_phrases_config_slot()
        banned_words_entries: int = len(banned_config)
        if banned_words_entries == 0:
            self.pretty.write(
                "W", "Banned Words", f"{Symbols.sym_fail()} No banned words configured."
            )
        else:
            self.pretty.write(
                "O",
                "Banned Words",
                f"{Symbols.sym_ok()} Current banned words: {banned_words_entries}",
            )
        return banned_words_entries

    def _print_masking(self) -> int:
        masking_config: dict[str, Any] = self.helpers.get_masking_regexes_config_slot()
        masking_entries: int = len(masking_config)
        if masking_entries == 0:
            self.pretty.write(
                "W",
                "Masking regexes",
                f"{Symbols.sym_fail()} No masking regexes configured.",
            )
        else:
            self.pretty.write(
                "O",
                "Masking regexes",
                f"{Symbols.sym_ok()} Current masking regexes: {masking_entries}",
            )
        return masking_entries
