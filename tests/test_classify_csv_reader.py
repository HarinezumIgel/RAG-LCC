# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for ClassifyCSVReader — reads DocClassify OK / HUMAN_REVIEW CSVs
and returns a set of normalised file paths.
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

STAMP = "20260101_000000"


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "CSV_DELIMITER": ";",
        }
        return mapping.get(key, default)

    def get_str(self, key, default=""):
        val = self.get(key, default)
        return str(val) if val is not None else default


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


def _write_ok_csv(logDir: str, rows: list[tuple[str, str]]) -> str:
    """Write a minimal OK CSV and return its path."""
    path = os.path.join(logDir, f"DocClassify_OK_{STAMP}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(OK_HEADER + "\n")
        for status, fp in rows:
            f.write(f'"{status}";"2026-01-01";"Summary";"{fp}";"Science"\n')
    return path


HR_HEADER = '"Status";"Time";"Stage";"FilePath";"Phrase"'


def _write_hr_csv(logDir: str, rows: list[tuple[str, str]]) -> str:
    """Write a minimal HUMAN_REVIEW CSV and return its path."""
    path = os.path.join(logDir, f"DocClassify_HUMAN_REVIEW_{STAMP}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HR_HEADER + "\n")
        for status, fp in rows:
            f.write(f'"{status}";"2026-01-01";"Summary";"{fp}";"genetic"\n')
    return path


def _make_reader(
    logDir: str,
    includeHumanReview: bool = False,
    stamp: str = STAMP,
) -> tuple[ClassifyCSVReader, StubPrettyWriter]:
    pw = StubPrettyWriter()
    reader = ClassifyCSVReader(
        stamp,
        includeHumanReview=includeHumanReview,
        logDir=logDir,
        cfg=StubConfig(),
        pretty=pw,
    )
    return reader, pw


# ---------------------------------------------------------------------------
# Tests — readOkFilePaths (OK CSV only)
# ---------------------------------------------------------------------------


class TestReadOkFilePaths:
    def test_reads_ok_paths(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(
            logDir,
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Dogs.png"),
            ],
        )
        reader, _ = _make_reader(logDir)
        paths = reader.readOkFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/project/docs/Cats.md") in paths
        assert os.path.normpath("D:/project/docs/Dogs.png") in paths

    def test_empty_csv_returns_empty_set(self, tmp_path):
        logDir = str(tmp_path)
        ok_path = os.path.join(logDir, f"DocClassify_OK_{STAMP}.csv")
        with open(ok_path, "w", encoding="utf-8") as f:
            f.write(OK_HEADER + "\n")

        reader, _ = _make_reader(logDir)
        paths = reader.readOkFilePaths()
        assert paths == set()

    def test_missing_file_raises_exception(self, tmp_path):
        reader, pw = _make_reader(str(tmp_path))
        with pytest.raises(ClassifyCSVNotFoundError, match="not found"):
            reader.readOkFilePaths()
        # Should also log an error
        assert any("not found" in str(m) for m in pw.messages)

    def test_deduplicates_within_single_csv(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(
            logDir,
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", "D:/project/docs/Dogs.png"),
            ],
        )
        reader, _ = _make_reader(logDir)
        paths = reader.readOkFilePaths()
        assert len(paths) == 2

    def test_blank_filepath_rows_are_skipped(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(
            logDir,
            [
                ("OK", "D:/project/docs/Cats.md"),
                ("OK", ""),
                ("OK", "  "),
            ],
        )
        reader, _ = _make_reader(logDir)
        paths = reader.readOkFilePaths()
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# Tests — includeHumanReview
# ---------------------------------------------------------------------------


class TestIncludeHumanReview:
    def test_merges_ok_and_human_review(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(logDir, [("OK", "D:/docs/Cats.md")])
        _write_hr_csv(logDir, [("OK", "D:/docs/Hedgehogs.pdf")])

        reader, _ = _make_reader(logDir, includeHumanReview=True)
        paths = reader.readOkFilePaths()

        assert len(paths) == 2
        assert os.path.normpath("D:/docs/Cats.md") in paths
        assert os.path.normpath("D:/docs/Hedgehogs.pdf") in paths

    def test_dedup_across_ok_and_human_review(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(logDir, [("OK", "D:/docs/Cats.md")])
        _write_hr_csv(
            logDir,
            [
                ("OK", "D:/docs/Cats.md"),
                ("OK", "D:/docs/Cats.md"),
            ],
        )

        reader, pw = _make_reader(logDir, includeHumanReview=True)
        paths = reader.readOkFilePaths()

        assert len(paths) == 1
        # Should log dedup message
        assert any("Dedup" in str(m) for m in pw.messages)

    def test_missing_human_review_warns_and_continues(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(logDir, [("OK", "D:/docs/Cats.md")])
        # No HUMAN_REVIEW file created

        reader, pw = _make_reader(logDir, includeHumanReview=True)
        paths = reader.readOkFilePaths()

        assert len(paths) == 1
        assert any("HUMAN_REVIEW CSV not found" in str(m) for m in pw.messages)

    def test_human_review_ignored_when_flag_false(self, tmp_path):
        logDir = str(tmp_path)
        _write_ok_csv(logDir, [("OK", "D:/docs/Cats.md")])
        _write_hr_csv(logDir, [("OK", "D:/docs/Extra.txt")])

        reader, _ = _make_reader(logDir, includeHumanReview=False)
        paths = reader.readOkFilePaths()

        assert len(paths) == 1
        assert os.path.normpath("D:/docs/Extra.txt") not in paths


# ---------------------------------------------------------------------------
# Tests — _buildCsvPath
# ---------------------------------------------------------------------------


class TestBuildCsvPath:
    def test_builds_ok_path(self, tmp_path):
        reader, _ = _make_reader(str(tmp_path))
        result = reader._buildCsvPath("OK")
        expected = os.path.join(str(tmp_path), f"DocClassify_OK_{STAMP}.csv")
        assert result == expected

    def test_builds_human_review_path(self, tmp_path):
        reader, _ = _make_reader(str(tmp_path))
        result = reader._buildCsvPath("HUMAN_REVIEW")
        expected = os.path.join(str(tmp_path), f"DocClassify_HUMAN_REVIEW_{STAMP}.csv")
        assert result == expected
