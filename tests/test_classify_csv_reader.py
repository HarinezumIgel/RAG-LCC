# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for ClassifyCSVReader — reads a DocClassify CSV and returns a set
of normalised file paths, with optional SQL WHERE filtering.
"""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Commons.Exceptions import ClassifyCSVNotFoundError
from Helpers.ClassifyCSVReader import ClassifyCSVReader

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "CSV_DELIMITER": ";",
        }
        return mapping.get(key, default)

    def get_str(self, key, default=""):
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_int(self, key, default=0):
        val = self.get(key, default)
        return int(val) if val is not None else default


class StubPrettyWriter:
    def __init__(self):
        self.messages: list[tuple[Any, ...]] = []

    def write(self, *a, **kw):
        self.messages.append((a, kw))
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OK_HEADER = '"Status";"Time";"Stage";"FilePath";"Classification"'


def _write_ok_csv(
    logDir: str, rows: list[tuple[str, str]], name: str = "classify.csv"
) -> str:
    """Write a minimal OK CSV and return its path."""
    path = os.path.join(logDir, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(OK_HEADER + "\n")
        for status, fp in rows:
            f.write(f'"{status}";"2026-01-01";"Summary";"{fp}";"Science"\n')
    return path


def _make_reader(
    csvPath: str,
    query: str = "",
) -> tuple[ClassifyCSVReader, StubPrettyWriter]:
    pw = StubPrettyWriter()
    reader = ClassifyCSVReader(
        csvPath,
        query=query,
        cfg=StubConfig(),
        pretty=pw,
    )
    return reader, pw


# ---------------------------------------------------------------------------
# Tests — readFilePaths
# ---------------------------------------------------------------------------


class TestReadFilePaths:
    def test_reads_ok_paths(self, tmp_path):
        csvPath = _write_ok_csv(
            str(tmp_path),
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Dogs.png"),
            ],
        )
        reader, _ = _make_reader(csvPath)
        paths = reader.readFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/project/docs/Cats.md") in paths
        assert os.path.normpath("D:/project/docs/Dogs.png") in paths

    def test_empty_csv_returns_empty_set(self, tmp_path):
        csvPath = os.path.join(str(tmp_path), "empty.csv")
        with open(csvPath, "w", encoding="utf-8") as f:
            f.write(OK_HEADER + "\n")

        reader, _ = _make_reader(csvPath)
        paths = reader.readFilePaths()
        assert paths == set()

    def test_missing_file_raises_exception(self, tmp_path):
        csvPath = os.path.join(str(tmp_path), "missing.csv")
        reader, pw = _make_reader(csvPath)
        with pytest.raises(ClassifyCSVNotFoundError, match="not found"):
            reader.readFilePaths()
        assert any("not found" in str(m) for m in pw.messages)

    def test_deduplicates_within_single_csv(self, tmp_path):
        csvPath = _write_ok_csv(
            str(tmp_path),
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Dogs.png"),
            ],
        )
        reader, _ = _make_reader(csvPath)
        paths = reader.readFilePaths()
        assert len(paths) == 2

    def test_blank_filepath_rows_are_skipped(self, tmp_path):
        csvPath = _write_ok_csv(
            str(tmp_path),
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", ""),
                ("OK", "  "),
            ],
        )
        reader, _ = _make_reader(csvPath)
        paths = reader.readFilePaths()
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# Tests — path resolution (filename vs. absolute)
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_absolute_path_used_as_is(self, tmp_path):
        csvPath = _write_ok_csv(str(tmp_path), [("OK", "D:/docs/A.md")])
        reader, _ = _make_reader(csvPath)
        assert reader.csvPath == csvPath

    def test_relative_name_resolved_via_logdir(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(logDir, [("OK", "D:/docs/A.md")], name="my.csv")
        pw = StubPrettyWriter()
        reader = ClassifyCSVReader(
            "my.csv",
            logDir=logDir,
            cfg=StubConfig(),
            pretty=pw,
        )
        assert reader.csvPath == os.path.join(logDir, "my.csv")
        paths = reader.readFilePaths()
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# Helpers — rich CSV with classification columns for query tests
# ---------------------------------------------------------------------------

RICH_HEADER = (
    '"Status";"Time";"Stage";"FilePath";"Classification";"Language";"Animal";"Mammal"'
)


def _write_rich_ok_csv(
    logDir: str, rows: list[dict[str, str]], name: str = "rich.csv"
) -> str:
    """Write an OK CSV with full classification columns."""
    path = os.path.join(logDir, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(RICH_HEADER + "\n")
        for r in rows:
            f.write(
                f'"{r.get("Status", "OK")}";'
                f'"2026-01-01";'
                f'"Summary";'
                f'"{r["FilePath"]}";'
                f'"{r.get("Classification", "")}";'
                f'"{r.get("Language", "")}";'
                f'"{r.get("Animal", "")}";'
                f'"{r.get("Mammal", "")}"\n'
            )
    return path


# ---------------------------------------------------------------------------
# Tests — CLASSIFY_CSV_QUERY (sqlite3 WHERE clause)
# ---------------------------------------------------------------------------


class TestClassifyCsvQuery:
    def test_like_filters_matching_rows(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/Cats.md", "Animal": "Cat", "Mammal": "Cat: Yes"},
                {
                    "FilePath": "D:/docs/Fish.md",
                    "Animal": "Salmon",
                    "Mammal": "Salmon: No",
                },
                {"FilePath": "D:/docs/Dogs.md", "Animal": "Dog", "Mammal": "Dog: Yes"},
            ],
        )
        reader, _ = _make_reader(csvPath, query="Mammal LIKE '%Yes%'")
        paths = reader.readFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/docs/Cats.md") in paths
        assert os.path.normpath("D:/docs/Dogs.md") in paths
        assert os.path.normpath("D:/docs/Fish.md") not in paths

    def test_not_like_excludes_rows(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/Cats.md", "Mammal": "Cat: Yes"},
                {
                    "FilePath": "D:/docs/Mixed.md",
                    "Mammal": "Cat: Yes, Salmon: Dont know",
                },
                {"FilePath": "D:/docs/Fish.md", "Mammal": "Salmon: No"},
            ],
        )
        reader, _ = _make_reader(csvPath, query="Mammal NOT LIKE '%Dont know%'")
        paths = reader.readFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/docs/Cats.md") in paths
        assert os.path.normpath("D:/docs/Fish.md") in paths

    def test_and_combines_conditions(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {
                    "FilePath": "D:/docs/CatEN.md",
                    "Language": "English",
                    "Mammal": "Cat: Yes",
                },
                {
                    "FilePath": "D:/docs/CatDE.md",
                    "Language": "German",
                    "Mammal": "Cat: Yes",
                },
                {
                    "FilePath": "D:/docs/FishEN.md",
                    "Language": "English",
                    "Mammal": "Salmon: No",
                },
            ],
        )
        reader, _ = _make_reader(
            csvPath, query="Mammal LIKE '%Yes%' AND Language = 'English'"
        )
        paths = reader.readFilePaths()

        assert len(paths) == 1
        assert os.path.normpath("D:/docs/CatEN.md") in paths

    def test_or_matches_either_condition(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/A.md", "Classification": "Science"},
                {"FilePath": "D:/docs/B.md", "Classification": "Finance"},
                {"FilePath": "D:/docs/C.md", "Classification": "Legal"},
            ],
        )
        reader, _ = _make_reader(
            csvPath, query="Classification = 'Science' OR Classification = 'Legal'"
        )
        paths = reader.readFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/docs/A.md") in paths
        assert os.path.normpath("D:/docs/C.md") in paths

    def test_empty_query_returns_all_rows(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/A.md", "Mammal": "Cat: Yes"},
                {"FilePath": "D:/docs/B.md", "Mammal": "Salmon: No"},
            ],
        )
        reader, _ = _make_reader(csvPath, query="")
        paths = reader.readFilePaths()
        assert len(paths) == 2

    def test_invalid_query_raises_valueerror(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/A.md", "Mammal": "Cat: Yes"},
            ],
        )
        reader, _ = _make_reader(csvPath, query="INVALID SYNTAX %%% !!!")
        with pytest.raises(ValueError, match="Invalid CLASSIFY_CSV_QUERY"):
            reader.readFilePaths()

    def test_query_logs_filtered_count(self, tmp_path):
        csvPath = _write_rich_ok_csv(
            str(tmp_path),
            [
                {"FilePath": "D:/docs/A.md", "Mammal": "Cat: Yes"},
                {"FilePath": "D:/docs/B.md", "Mammal": "Salmon: No"},
                {"FilePath": "D:/docs/C.md", "Mammal": "Dog: Yes"},
            ],
        )
        reader, pw = _make_reader(csvPath, query="Mammal LIKE '%Yes%'")
        reader.readFilePaths()
        assert any("filtered out 1 row" in str(m) for m in pw.messages)
