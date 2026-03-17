import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import xlsxwriter  # type: ignore[reportMissingTypeStubs]

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Globals import Globals
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class CSVWriterError(RuntimeError):
    pass


class CSVWriter(SingletonMixin):

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

        self.globals: Globals = Globals()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        self.cfg: Config = cfg or Config()
        self.csv_delimiter: str = self.cfg.get_str("CSV_DELIMITER", ";")

        # stateMap: status -> { "path": str, "file": IO, "writer": csv.DictWriter, "fieldnames": List[str] }
        self.stateMap: Dict[str, Dict[str, Any]] = {}

        self._define_csv_files()

    def __del__(self):
        self._close_all

    # ---------------------------
    # Helpers
    # ---------------------------

    def _ensure_dir_for_path(self, path: str) -> None:
        directory: str = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _get_desired_keys(self, status: str) -> List[str]:
        if status == "HUMAN_REVIEW":
            return self.cfg.get_list("_KEYS_FOR_HUMAN_REVIEW_CSV", [])
        return self.cfg.get_list("_CLASSIFICATION_KEYS", [])

    def _csv_path_for(self, _FRIENDLY_NAME: str, status: str, log_dir: str) -> str:
        date_str: str = self.globals.get_date()
        csv_name: str = f"{_FRIENDLY_NAME}_{status}"
        return os.path.join(log_dir, f"{csv_name}_{date_str}.csv")

    # ---------------------------
    # Writer lifecycle
    # ---------------------------

    def _open_writer(self, csv_path: str, fieldnames: List[str]) -> Dict[str, Any]:
        self._ensure_dir_for_path(csv_path)

        file_obj = open(csv_path, mode="a+", newline="", encoding="utf-8")
        try:
            file_obj.seek(0, os.SEEK_END)
            is_empty: bool = file_obj.tell() == 0

            writer: csv.DictWriter[str] = csv.DictWriter(
                file_obj,
                fieldnames=fieldnames,
                delimiter=self.csv_delimiter,
                quoting=csv.QUOTE_ALL,
            )
            if is_empty:
                writer.writeheader()
                file_obj.flush()

            return {"file": file_obj, "writer": writer, "fieldnames": fieldnames}
        except Exception:
            try:
                file_obj.close()
            except Exception:
                pass
            raise

    def _ensure_writer(self, status: str) -> None:
        meta: Dict[str, Any] = self.stateMap.setdefault(status, {})
        if meta.get("writer") and meta.get("file") and not meta["file"].closed:
            return

        csv_path: str | None = meta.get("path")
        if not csv_path:
            raise CSVWriterError(f"No CSV path configured for status '{status}'")

        fieldnames: List[str] = self._get_desired_keys(status)
        if not fieldnames:
            raise CSVWriterError(f"No fieldnames configured for status '{status}'")
        try:
            writer_meta: Dict[str, Any] = self._open_writer(csv_path, fieldnames)
            meta.update(writer_meta)
        except Exception as e:
            self.pretty.write("E", "CSVWriter", f"Failed opening CSV {csv_path}: {e}")
            raise CSVWriterError(f"Failed opening CSV {csv_path}: {e}") from e

    def _define_csv_files(self) -> None:
        _FRIENDLY_NAME: str = self.cfg.get_str("_FRIENDLY_NAME")
        log_dir: str = self.cfg.get_str("_LOG_DIRECTORY")

        if not _FRIENDLY_NAME:
            return
        if not log_dir:
            self.pretty.write("E", "CSVWriter", "_LOG_DIRECTORY not configured")
            return

        for status in ["OK", "NOT_OK", "HUMAN_REVIEW"]:
            csv_path = self._csv_path_for(_FRIENDLY_NAME, status, log_dir)
            self.stateMap.setdefault(status, {})["path"] = csv_path

            fieldnames = self._get_desired_keys(status)
            if not fieldnames:
                continue
            try:
                writer_meta = self._open_writer(csv_path, fieldnames)
                self.stateMap[status].update(writer_meta)
            except Exception as e:
                self.pretty.write(
                    "E", "CSVWriter", f"Failed opening CSV {csv_path}: {e}"
                )
                raise CSVWriterError(f"Failed opening CSV {csv_path}: {e}") from e

    # ---------------------------
    # Row handling and writing
    # ---------------------------

    def _normalize_value(self, v: Any) -> str:
        """Format value for CSV: lists as comma-separated, dicts as key: value pairs, others as str."""
        if isinstance(v, list):
            lst: List[Any] = cast(List[Any], v)
            return ", ".join(str(item) for item in lst)
        if isinstance(v, dict):
            dct: Dict[str, Any] = cast(Dict[str, Any], v)
            return ", ".join(f"{k}: {val}" for k, val in dct.items())
        return str(v) if v is not None else ""

    def _row_from_json(self, json_data: dict[str, Any], status: str) -> dict[str, Any]:
        keys = self._get_desired_keys(status)
        return {key: json_data.get(key, "") for key in keys}

    def write_json2csv(
        self,
        json_data: dict[str, Any] | list[dict[str, Any]] | None,
        status: str,
    ) -> None:
        if not json_data:
            return

        self._ensure_writer(status)
        meta: Dict[str, Any] = self.stateMap.get(status, {})

        writer: csv.DictWriter[str] = meta["writer"]
        file_obj = meta["file"]
        fieldnames: list[str] = meta.get("fieldnames", [])

        items: list[dict[str, Any]] = (
            [json_data] if isinstance(json_data, dict) else list(json_data)
        )

        try:
            for item in items:
                row: dict[str, Any] = self._row_from_json(item, status)
                filtered_row: dict[str, Any] = {
                    k: self._normalize_value(row.get(k, "")) for k in fieldnames
                }
                writer.writerow(filtered_row)
            file_obj.flush()
        except Exception as e:
            self.pretty.write(
                "E", "CSVWriter", f"Failed writing CSV {meta.get('path')}: {e}"
            )
            raise CSVWriterError(f"Failed writing CSV {meta.get('path')}: {e}") from e

    # ---------------------------
    # CSV -> XLSX conversion using xlsxwriter
    # ---------------------------

    def _coerce_value(self, val: str) -> str | int | float:
        """
        Try to coerce a CSV string to int/float if it looks numeric.
        Otherwise return the original string.
        """
        if val is None:  # type: ignore[reportUnnecessaryComparison]
            return ""
        if isinstance(val, (int, float)):
            return val
        s: str = str(val).strip()
        if s == "":
            return ""
        # try int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # try float
        try:
            f: float = float(s)
            return f
        except Exception:
            return s

    def _csv_to_xlsx(
        self,
        csv_path: Path,
        xlsx_path: Path,
        *,
        sep: str = ";",
        convert_numbers: bool = True,
    ) -> None:
        """
        Stream CSV rows into an XLSX file using xlsxwriter.
        - convert_numbers: if True, attempt to coerce numeric-looking strings to numbers.
        """
        if xlsxwriter is None:
            raise CSVWriterError(
                "xlsxwriter is required for CSV->XLSX conversion but is not installed"
            )

        # Use constant_memory option for low memory usage
        workbook: xlsxwriter.Workbook = xlsxwriter.Workbook(
            str(xlsx_path), {"constant_memory": True}
        )
        worksheet = workbook.add_worksheet()  # type: ignore[reportUnknownMemberType]

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=sep)
            for r, row in enumerate(reader):
                for c, cell in enumerate(row):
                    if convert_numbers:
                        val = self._coerce_value(cell)
                    else:
                        val = cell
                    # xlsxwriter will accept Python types (int/float/str)
                    worksheet.write(r, c, val)  # type: ignore[reportUnknownMemberType]
        workbook.close()

    def _doConvert(
        self, status: str, *, read_csv_kwargs: Optional[dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Convert the CSV for a given status to XLSX using xlsxwriter.
        Returns the XLSX path on success, None on failure.
        """
        kwargs: dict[str, Any] = dict(read_csv_kwargs or {})
        # default separator from config
        sep: str = kwargs.pop("sep", self.csv_delimiter)
        convert_numbers: bool = kwargs.pop("convert_numbers", True)

        try:
            path: str | None = self.stateMap.get(status, {}).get("path")
            if not path:
                return None
            p: Path = Path(path)
            xlsx_path: Path = p.with_suffix(".xlsx")
            if p.exists() and p.suffix.lower() == ".csv":
                self._csv_to_xlsx(
                    p, xlsx_path, sep=sep, convert_numbers=convert_numbers
                )
                self.pretty.write("I", "Logs", f"Created xlsx file {xlsx_path}")
                return xlsx_path
            return xlsx_path

        except CSVWriterError:
            raise
        except Exception as e:
            self.pretty.write(
                "E",
                "Error",
                f"Failed converting {self.stateMap.get(status, {}).get('path')} to xlsx: {e}",
            )
            return None

    def convert_csv2xlsx(self, status: str) -> Optional[Path]:
        return self._doConvert(status)

    # ---------------------------
    # Shutdown
    # ---------------------------

    def _close_all(self) -> None:
        for status, meta in self.stateMap.items():
            f = meta.get("file")
            if f and not f.closed:
                try:
                    f.flush()
                except Exception:
                    pass
                try:
                    f.close()
                except Exception as e:
                    self.pretty.write(
                        "W", "CSVWriter", f"Failed closing file for {status}: {e}"
                    )
