import csv
import os
import sqlite3
from typing import Set

from Commons.Exceptions import ClassifyCSVNotFoundError
from Config.Config import Config
from Gui.Colors import CYAN, GREEN, RED
from Gui.PrettyWriter import PrettyWriter


class ClassifyCSVReader:
    """Read a DocClassify CSV and return a set of file paths.

    Accepts a CSV filename (resolved relative to ``_LOG_DIRECTORY``) or an
    absolute path.  An optional SQL WHERE clause filters the rows before
    the allow-set is built.
    """

    def __init__(
        self,
        csvPath: str,
        *,
        query: str = "",
        logDir: str | None = None,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
    ) -> None:
        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.delimiter: str = self.cfg.get_str("CSV_DELIMITER", ";")
        self.query: str = query.strip()

        # Resolve the CSV path
        if os.path.isabs(csvPath):
            self.csvPath: str = csvPath
        else:
            effectiveLogDir = logDir or self.cfg.get_str("_LOG_DIRECTORY")
            self.csvPath = os.path.join(effectiveLogDir, csvPath)

        if self.query:
            self.pretty.write(
                "I",
                "ClassifyCSVReader",
                f"CSV query active: {self.query}",
            )

    def readFilePaths(self) -> Set[str]:
        """Return normalised FilePath values from the CSV.

        When a query is active the CSV is loaded into an in-memory
        SQLite table and only rows that satisfy the SQL WHERE clause
        are included in the returned set.
        """
        if not os.path.isfile(self.csvPath):
            msg = f"CSV file not found: {self.csvPath}"
            self.pretty.write(
                "E",
                "ClassifyCSVReader",
                msg,
                color=RED,
            )
            raise ClassifyCSVNotFoundError(msg)

        ext = os.path.splitext(self.csvPath)[1].lower()
        if ext != ".csv":
            csvAlt = os.path.splitext(self.csvPath)[0] + ".csv"
            hint = (
                f" A matching .csv exists — use that instead: {os.path.basename(csvAlt)}"
                if os.path.isfile(csvAlt)
                else ""
            )
            msg = (
                f"Expected a .csv file but got '{ext}': {self.csvPath}.{hint}"
            )
            self.pretty.write("E", "ClassifyCSVReader", msg, color=RED)
            raise ClassifyCSVNotFoundError(msg)

        rows: list[dict[str, str]] = []
        with open(self.csvPath, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter, quoting=csv.QUOTE_ALL)
            columns: list[str] = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        if not self.query:
            paths: Set[str] = set()
            for row in rows:
                filePath = (row.get("FilePath") or "").strip()
                if filePath:
                    paths.add(os.path.normpath(filePath))
            self.pretty.write(
                "I",
                "ClassifyCSVReader",
                f"Loaded {len(paths)} file paths from {self.csvPath}",
            )
            return paths

        return self._queryWithSqlite(rows, columns)

    def _queryWithSqlite(
        self,
        rows: list[dict[str, str]],
        columns: list[str],
    ) -> Set[str]:
        """Load *rows* into an in-memory SQLite table and run the user query."""
        debugLevel = self.cfg.get_int("DEBUG_LEVEL", 0)

        # Determine which CSV columns are referenced in the WHERE clause
        # so they can be included in debug output.
        queryCols: list[str] = [
            c for c in columns if c != "FilePath" and c in self.query
        ]

        conn = sqlite3.connect(":memory:")
        try:
            cur = conn.cursor()
            quotedCols = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" for _ in columns)
            cur.execute(f"CREATE TABLE csv_data ({quotedCols})")
            for row in rows:
                values = [row.get(c, "") for c in columns]
                cur.execute(
                    f"INSERT INTO csv_data ({quotedCols}) VALUES ({placeholders})",
                    values,
                )

            selectCols = ['"FilePath"'] + [f'"{c}"' for c in queryCols]
            sql = f'SELECT {", ".join(selectCols)} FROM csv_data WHERE {self.query}'
            try:
                cur.execute(sql)
            except sqlite3.OperationalError as exc:
                self.pretty.write(
                    "E",
                    "ClassifyCSVReader",
                    f"Invalid CSV query: {exc}",
                    color=RED,
                )
                raise ValueError(f"Invalid CLASSIFY_CSV_QUERY: {exc}") from exc

            matched = cur.fetchall()
            totalRows = len(rows)
        finally:
            conn.close()

        paths: Set[str] = set()
        infoRows: list[tuple[str, ...]] = []
        for matchedRow in matched:
            fp = (matchedRow[0] or "").strip()
            if fp:
                normed = os.path.normpath(fp)
                paths.add(normed)
                if debugLevel >= 3:
                    infoRows.append((normed,) + matchedRow[1:])

        skipped = totalRows - len(matched)
        self.pretty.write(
            "I",
            "ClassifyCSVReader",
            f"Loaded {len(paths)} file paths from {self.csvPath}",
        )
        if skipped:
            self.pretty.write(
                "I",
                "ClassifyCSVReader",
                f"Query filtered out {skipped} row(s) from {self.csvPath}",
            )
        if len(matched) != 0:
            self.pretty.write(
                "I",
                "ClassifyCSVReader",
                f"Query found {len(matched)} row(s) from {self.csvPath}",
            )
        if infoRows:
            sortedRows = sorted(infoRows)
            colNames = ["FilePath"] + queryCols
            colWidths = [len(c) for c in colNames]
            for dr in sortedRows:
                for i, val in enumerate(dr):
                    colWidths[i] = max(colWidths[i], len(str(val)))
            headerLine = f"  {colNames[0].ljust(colWidths[0])}"
            if queryCols:
                headerExtras = " | ".join(
                    colNames[j + 1].ljust(colWidths[j + 1])
                    for j in range(len(queryCols))
                )
                headerLine += f"  |  {headerExtras}"
            self.pretty.write("I", "ClassifyCSVReader", headerLine, color=GREEN)
            for dr in sortedRows:
                line = f"  {str(dr[0]).ljust(colWidths[0])}"
                if queryCols:
                    extras = " | ".join(
                        str(dr[j + 1]).ljust(colWidths[j + 1])
                        for j in range(len(queryCols))
                    )
                    line += f"  |  {extras}"
                self.pretty.write("I", "ClassifyCSVReader", line, color=CYAN)

        return paths
