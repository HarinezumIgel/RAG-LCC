import re
from dataclasses import asdict
from typing import Any

from Config.Config import Config
from Helpers.Helpers import Helpers


class BannedPhraseCollector:
    """
    Collects match dicts and prepares merged rows for CSV output.

    Input for add_match:
      {
        "algo": "cosine similarity" | "jaccard" | "regex" | "Double KeyBert",
        "phrase": "some phrase",
        "score": float,
        "threshold": float,
        "chunk": "text snippet",
        ... (other optional fields)
      }
    """

    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.helpers: Helpers = Helpers()
        self.default_algos: list[str] = self.cfg.get_list("_DEFAULT_ALGOS")
        self.conf_default_metadata_keys: list[str] = self.cfg.get_list(
            "_KEYS_FOR_HUMAN_REVIEW_CSV"
        )

    def _normalize_key(self, name: str) -> str:
        """Make a safe column name: lowercase, spaces -> _, remove non-alnum/_."""
        s = re.sub(r"[^A-Za-z0-9_]", "", name)
        return s

    def prepare_for_csv_print(
        self,
        phrase_rows: list[Any],
        meta: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert a list of per-algo rows into a list of rows matching _KEYS_FOR_HUMAN_REVIEW_CSV.

        Args:
            phrase_rows: list of dicts with keys like:
                {
                    'algo': 'Cosine',
                    'phrase': 'Hedgehog',
                    'score': 0.2175,
                    'score_str': '0.2175/0.5000',
                    'threshold': 0.5,
                    'detail': 'cosine=0.217',
                    'matched_algos_count': 3,
                    'algos_matched': '3/3'
                }
            meta: optional mapping phrase -> metadata dict (FilePath, Language, etc.)

        Returns:
            List[dict] rows ready to pass to write_json2csv.
        """
        results: list[dict[str, Any]] = []
        meta = meta or {}

        # Group incoming rows by phrase for easier aggregation
        by_phrase: dict[str, list[Any]] = {}

        for r in phrase_rows:
            phrase = r.phrase or ""
            by_phrase.setdefault(phrase, []).append(r)

        for phrase, rows in by_phrase.items():
            row: dict[str, Any] = {}
            row["Phrase"] = phrase

            max_score_overall: float | None = None

            # For each configured algorithm, find the best matching entry (highest score)
            for algo in self.default_algos:
                # collect entries for this phrase+algo
                algo_entries: list[Any] = [e for e in rows if e.algo == algo]

                best_entry: dict[str, Any] | None = None
                for e in algo_entries:
                    # prefer entries that have numeric score
                    score: Any = e.score
                    if score is None:
                        continue
                    best_entry = asdict(e)
                    best_entry["score"] = float(score)

                score_key: str = f"Score {algo}"
                thresh_key: str = f"Threshold {algo}"

                if best_entry is not None and best_entry.get("score") is not None:
                    best_score: float = float(best_entry["score"])
                    best_thresh: Any = best_entry.get("threshold")
                    # format values as in original implementation
                    row[score_key] = f"{best_score:.4f}"
                    row[thresh_key] = (
                        f"{float(best_thresh):.4f}" if best_thresh is not None else ""
                    )
                    # preserve detail if desired
                    row[f"{algo} detail"] = best_entry.get("detail", "")
                    # keep matched count / algos_matched if present
                    if "matched_algos_count" in best_entry:
                        row["Matched Algos Count"] = str(
                            best_entry.get("matched_algos_count")
                        )
                    if "algos_matched" in best_entry:
                        row["Algos Matched"] = str(best_entry.get("algos_matched"))

                    # Keep the original PASS/FAIL logic (unchanged)
                    row[algo] = (
                        "FAIL"
                        if (
                            best_thresh is not None and best_score >= float(best_thresh)
                        )
                        else "FAIL_ALGOS_PER_PHRASE"
                    )

                    if max_score_overall is None or best_score > max_score_overall:
                        max_score_overall = best_score
                else:
                    # No data for this algo
                    row[score_key] = ""
                    row[thresh_key] = ""
                    row[f"{algo} detail"] = ""
                    row[algo] = ""

            # Max Score column (formatted)
            row["Max Score"] = (
                f"{max_score_overall:.4f}" if max_score_overall is not None else ""
            )

            # Merge per-phrase metadata if available (only keys present in _KEYS_FOR_HUMAN_REVIEW_CSV)
            self._add_meta_keys(row, meta)
            row = self._normalize_row(row)
            row = self.helpers.replace_keys_with_aliases(row)
            results.append(row)
        return results

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized_row: dict[str, Any] = {}
        for k, v in row.items():
            nk = self._normalize_key(k)
            if nk in normalized_row:
                raise ValueError(f"Duplicate normalized key: {nk!r} from {k!r}")
            normalized_row[nk] = v
        return normalized_row

    def _add_meta_keys(
        self, row: dict[str, Any], meta: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Merge per-phrase metadata into the row dictionary.

        Modifies `row` in-place and returns it for convenience.
        Only adds keys present in _KEYS_FOR_HUMAN_REVIEW_CSV.
        """
        if meta is None:
            return row
        for key in self.conf_default_metadata_keys:
            if key not in row and key in meta and meta[key] is not None:
                row[key] = meta[key]
        return row

    def prepare_print_for_chat(
        self, meta: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        row: dict[str, Any] = {}
        row = self._add_meta_keys(row, meta)
        row = self._normalize_row(row)
        results.append(row)
        return results

    # # --- Example usage ---
    # def run_example(self):

    #     phrase_table = {
    #         "i am here": {
    #             "Keybert": ["0.4321/0.4000", "0.5123/0.5000"],
    #             "Cosine": ["0.3000/0.2500"],
    #         },
    #         "hello world": {
    #             "Cosine": ["0.1000/0.2000", "0.2500/0.2000"],
    #             "Regex": ["0.0000/0.0000"],
    #         },
    #     }

    #     meta = {
    #         "FilePath": "/data/doc1.txt",
    #         "Language": "en",
    #         "Hello World": "Hello",
    #     }

    #     rows = self.prepare_print(phrase_table, meta)
