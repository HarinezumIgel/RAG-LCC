import csv
import os
from typing import Set

from Commons.Exceptions import ClassifyCSVNotFoundError
from Config.Config import Config
from Gui.Colors import ORANGE, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import build_csv_path


class ClassifyCSVReader:
    """Read DocClassify OK / HUMAN_REVIEW CSVs and return a set of file paths.

    CSV paths are built from:  ``{logDir}/DocClassify_{status}_{runStamp}.csv``
    """

    def __init__(
        self,
        runStamp: str,
        *,
        includeHumanReview: bool = False,
        logDir: str | None = None,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.runStamp: str = runStamp
        self.includeHumanReview: bool = includeHumanReview
        self.logDir: str = logDir or self.cfg.get_str("_LOG_DIRECTORY")
        self.delimiter: str = self.cfg.get_str("CSV_DELIMITER", ";")

    def _buildCsvPath(self, status: str) -> str:
        """Build ``DocClassify_{status}_{runStamp}.csv`` inside logDir."""
        return build_csv_path("DocClassify", status, self.runStamp, self.logDir)

    def _readFilePaths(self, csvPath: str) -> Set[str]:
        """Return normalised FilePath values from a CSV file."""
        if not os.path.isfile(csvPath):
            msg = f"CSV file not found: {csvPath}"
            self.pretty.write(
                "E",
                "ClassifyCSVReader",
                msg,
                color=RED,
            )
            raise ClassifyCSVNotFoundError(msg)

        paths: Set[str] = set()
        with open(csvPath, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter, quoting=csv.QUOTE_ALL)
            for row in reader:
                filePath = (row.get("FilePath") or "").strip()
                if filePath:
                    paths.add(os.path.normpath(filePath))

        self.pretty.write(
            "I",
            "ClassifyCSVReader",
            f"Loaded {len(paths)} file paths from {csvPath}",
        )
        return paths

    def readOkFilePaths(self) -> Set[str]:
        """Return normalised FilePath values from the OK CSV.

        When *includeHumanReview* is enabled the HUMAN_REVIEW CSV
        is merged in as well.
        """
        okPath = self._buildCsvPath("OK")
        paths = self._readFilePaths(okPath)

        if self.includeHumanReview:
            hrPath = self._buildCsvPath("HUMAN_REVIEW")
            if os.path.isfile(hrPath):
                hrPaths = self._readFilePaths(hrPath)
                duplicates = paths & hrPaths
                if duplicates:
                    self.pretty.write(
                        "I",
                        "ClassifyCSVReader",
                        f"Dedup: {len(duplicates)} path(s) appear in both "
                        "OK and HUMAN_REVIEW CSVs",
                    )
                paths |= hrPaths
            else:
                self.pretty.write(
                    "W",
                    "ClassifyCSVReader",
                    f"HUMAN_REVIEW CSV not found (expected {hrPath}), "
                    "continuing with OK paths only",
                    color=ORANGE,
                )

        return paths
