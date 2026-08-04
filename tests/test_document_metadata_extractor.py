# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for DocumentMetadataExtractor (multi-format + synonym normalization)."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.Chunkers.DocumentMetadataExtractor import DocumentMetadataExtractor

TESTDOCS = os.path.join(ROOT, "TestDocs")


@pytest.fixture
def extractor():
    # Fresh singleton bound to the real project config (ENABLED=True).
    DocumentMetadataExtractor._reset()
    inst = DocumentMetadataExtractor()
    yield inst
    DocumentMetadataExtractor._reset()


class _StubCfg:
    """Minimal Config stub for the disabled-switch test."""

    def get_bool(self, key, default=False, *, silent=False):
        return False  # ENABLED off

    def get_dict(self, key, default=None, *, silent=False):
        return default or {}


# ---------------------------------------------------------------------------
# Format-specific readers (use the real sample docs shipped with the repo)
# ---------------------------------------------------------------------------


class TestFormats:
    def test_pdf(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Hedgehogs.pdf"), "pdf")
        assert "DocCreated" in meta
        assert "DocModified" in meta
        assert meta.get("Creator")  # authoring app (PDF)
        assert "FileSizeBytes" not in meta  # doc-info tier, not generic

    def test_docx(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Apes.docx"), "docx")
        assert meta.get("Author")
        assert "DocCreated" in meta

    def test_pptx(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Lions.pptx"), "pptx")
        assert "DocCreated" in meta
        assert meta.get("LastModifiedBy")

    def test_xlsx_creator_normalised_to_author(self, extractor):
        # openpyxl exposes the author under "creator"; it must surface as Author.
        meta = extractor.extract(os.path.join(TESTDOCS, "LionsAndApes.xlsx"), "xlsx")
        assert meta.get("Author")

    def test_generic_for_markdown(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Cats.md"), "md")
        assert "FileSizeBytes" in meta
        assert "FileModified" in meta
        assert "Author" not in meta  # no doc-info for plain text

    def test_generic_for_image(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Dogs.png"), "png")
        assert "FileSizeBytes" in meta
        assert "FileModified" in meta

    def test_legacy_office_falls_through_to_generic(self, extractor, tmp_path):
        # .doc is not a doc-info type here → generic filesystem fields only.
        legacy = tmp_path / "old.doc"
        legacy.write_bytes(b"not a real doc")
        meta = extractor.extract(str(legacy), "doc")
        assert "FileSizeBytes" in meta
        assert "Author" not in meta


# ---------------------------------------------------------------------------
# Pages field (only attached when > 0)
# ---------------------------------------------------------------------------


class TestPages:
    def test_pdf_has_pages(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Hedgehogs.pdf"), "pdf")
        assert isinstance(meta.get("Pages"), int)
        assert meta["Pages"] > 0

    def test_pptx_slides_counted_as_pages(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Lions.pptx"), "pptx")
        assert meta.get("Pages", 0) > 0

    def test_image_has_no_pages(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Dogs.png"), "png")
        assert "Pages" not in meta

    def test_text_has_no_pages(self, extractor):
        meta = extractor.extract(os.path.join(TESTDOCS, "Cats.md"), "md")
        assert "Pages" not in meta

    def test_page_count_helper_zero_for_unknown(self, extractor):
        assert extractor._page_count("png", os.path.join(TESTDOCS, "Dogs.png")) == 0


# ---------------------------------------------------------------------------
# Synonym normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_creation_date_synonym(self, extractor):
        out = extractor._normalize({"creation_date": "2020-01-01"}, "DOC_INFO_FIELDS")
        assert out.get("DocCreated") == "2020-01-01"

    def test_created_synonym(self, extractor):
        out = extractor._normalize({"created": "2021-02-02"}, "DOC_INFO_FIELDS")
        assert out.get("DocCreated") == "2021-02-02"

    def test_first_non_empty_wins(self, extractor):
        # "creation_date" is listed before "created" for DocCreated.
        out = extractor._normalize(
            {"creation_date": "A", "created": "B"}, "DOC_INFO_FIELDS"
        )
        assert out.get("DocCreated") == "A"

    def test_empty_values_skipped(self, extractor):
        out = extractor._normalize({"author": "   "}, "DOC_INFO_FIELDS")
        assert "Author" not in out

    def test_case_insensitive_keys(self, extractor):
        out = extractor._normalize({"AUTHOR": "Jane"}, "DOC_INFO_FIELDS")
        assert out.get("Author") == "Jane"

    def test_generic_fields_mapping(self, extractor):
        out = extractor._normalize({"size": 123, "modified": "x"}, "GENERIC_FIELDS")
        assert out.get("FileSizeBytes") == "123"
        assert out.get("FileModified") == "x"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_missing_file_returns_empty(self, extractor):
        assert extractor.extract(os.path.join(TESTDOCS, "nope.pdf"), "pdf") == {}

    def test_empty_path_returns_empty(self, extractor):
        assert extractor.extract("", "pdf") == {}

    def test_disabled_returns_empty(self):
        DocumentMetadataExtractor._reset()
        inst = DocumentMetadataExtractor(cfg=_StubCfg())  # type: ignore[arg-type]
        try:
            result = inst.extract(os.path.join(TESTDOCS, "Hedgehogs.pdf"), "pdf")
            assert result == {}
        finally:
            DocumentMetadataExtractor._reset()
