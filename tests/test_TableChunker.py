# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportPrivateUsage=false
"""Real-document table extraction and ingestion contract tests (no model downloads)."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document as WordDocument
from docx.oxml import OxmlElement
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from Algos.Masker import Masker
from Algos.UnicodeNormalizer import UnicodeNormalizer
from Commons.Exceptions import DataProcessingError
from Pipeline.LoadAndClassifyProcessor import LoadAndClassifyProcessor
from Strategies.Chunkers.HeadingChunker import HeadingChunker
from Strategies.Chunkers.PdfPageChunker import PdfPageChunker
from Strategies.Chunkers.RecursiveChunker import RecursiveChunker
from Strategies.Chunkers.SemanticChunker import SemanticChunker
from Strategies.Chunkers.SentenceWindowChunker import SentenceWindowChunker
from Strategies.Chunkers.SlidingWindowChunker import SlidingWindowChunker
from Strategies.Chunkers.TableChunker import DocumentBlock, TableChunker
from Strategies.DocumentIngestionStrategy import DocumentIngestionStrategy
from Strategies.StrategyType import StrategyType


@pytest.fixture
def tokenizer():
    # A real byte-level tokenizer makes dates/underscores cost multiple tokens,
    # exercising the word-vs-token mismatch without downloading any model.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(["[UNK]", "[CLS]", "[SEP]", *alphabet])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=512,
    )


def _chunker(kind, size=240, mode="suffix"):
    cfg = MagicMock()
    cfg.get_int.return_value = size
    cfg.get_str.return_value = mode
    cfg.get_bool.return_value = False
    cfg.get_list.return_value = ["\n\n", "\n", " ", ""]
    helpers = MagicMock()
    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.TEST"
    kwargs = dict(cfg=cfg, helpers=helpers)
    if kind is SemanticChunker:
        kwargs["embedder"] = MagicMock()
    return kind(**kwargs)


def _mark_header(row):
    row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))


def _word_fixture(path: Path):
    doc = WordDocument()
    doc.add_heading("Employees", level=1)
    doc.add_paragraph("Before the first table.")
    doc.add_paragraph("Staff register", style="Caption")
    for heading, name in [("", "Alice"), ("Customers", "Bob")]:
        if heading:
            doc.add_heading(heading, level=1)
        table = doc.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "name"
        table.rows[0].cells[1].text = "birth_date"
        _mark_header(table.rows[0])
        for i in range(10):
            cells = table.add_row().cells
            cells[0].text = f"{name}{i}"
            cells[1].text = f"1980-01-{i + 1:02}"
        doc.add_paragraph("After this table.")
    doc.save(str(path))
    return WordDocument(str(path))


def _pdf_fixture(path: Path, pages=1, merged=False):
    """Write a small genuine PDF with ruled tables using the existing pypdf."""
    writer = PdfWriter()
    for page_index in range(pages):
        page = writer.add_blank_page(width=400, height=800)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        commands = ["BT /F1 11 Tf 40 760 Td (Before tables) Tj ET"]
        for y, name in [(700, "Alice"), (450, "Bob")]:
            rows = [
                ("name", "birth_date"),
                (name, f"198{page_index}-01-01"),
                (f"{name}Junior", "2000-02-02"),
            ]
            if merged:
                rows.insert(0, ("Person", ""))
            for x in [40, 180, 320]:
                start = y - 24 if merged and x == 180 else y
                commands.append(f"{x} {start} m {x} {y - 24 * len(rows)} l S")
            for boundary in range(len(rows) + 1):
                line_y = y - 24 * boundary
                commands.append(f"40 {line_y} m 320 {line_y} l S")
            for i, row in enumerate(rows):
                for x, value in zip([45, 185], row):
                    commands.append(
                        f"BT /F1 11 Tf 1 0 0 1 {x} {y - 16 - 24 * i} Tm ({value}) Tj ET"
                    )
        commands.append("BT /F1 11 Tf 40 300 Td (After tables) Tj ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as stream:
        writer.write(stream)


def test_real_word_order_identity_headers_and_caption(tmp_path, tokenizer):
    blocks = TableChunker.extract_word(_word_fixture(tmp_path / "tables.docx"))
    tables = [block for block in blocks if block.table_index]
    assert [b.table_index for b in tables] == [1, 2]
    assert [b.heading_path for b in tables] == ["Employees", "Customers"]
    assert tables[0].caption == "Staff register"
    assert tables[1].caption == ""
    assert "Before the first table" in blocks[0].text
    assert blocks[1] == tables[0]
    assert "After this table" in blocks[2].text
    chunker = TableChunker(tokenizer, 210, pretty=MagicMock())
    chunks = [
        d
        for table in tables
        for d in chunker.chunk_table(table, {"FilePath": "tables.docx"})
    ]
    for doc in chunks:
        assert "birth_date" in doc.page_content
        assert doc.metadata["TableHeaderSource"] == "explicit"
        assert chunker.fits(doc.page_content)
        assert doc.metadata["TableContext"] in doc.page_content
    assert len(chunks) > 2
    assert {d.metadata["TableIndex"] for d in chunks} == {1, 2}
    assert all(
        f"Alice{i}" in "\n".join(d.page_content for d in chunks) for i in range(10)
    )


def test_word_merged_multirow_and_headerless_tables():
    doc = WordDocument()
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Person"
    table.cell(1, 0).text = "name"
    table.cell(1, 1).text = "birth_date"
    table.cell(2, 0).text = "Alice"
    table.cell(2, 1).text = "1980-01-01"
    _mark_header(table.rows[0])
    _mark_header(table.rows[1])
    block = TableChunker.extract_word(doc)[0]
    assert block.header_rows == 2
    assert TableChunker._columns(block) == ["Person > name", "Person > birth_date"]
    tableless = replace(
        block,
        rows=(("Alice", "London"), ("Bob", "Paris")),
        header_rows=0,
        header_source="positional",
    )
    assert TableChunker._columns(tableless) == ["column_1", "column_2"]
    assert TableChunker._header_count(tableless.rows) == 0


def test_unmarked_word_header_is_inferred_without_dropping_ambiguous_data():
    doc = WordDocument()
    table = doc.add_table(rows=2, cols=2)
    for row, values in zip(
        table.rows, [("name", "birth_date"), ("Alice", "1980-01-01")]
    ):
        for cell, value in zip(row.cells, values):
            cell.text = value
    block = TableChunker.extract_word(doc)[0]
    assert block.header_rows == 1
    assert block.header_source == "inferred"
    table.cell(0, 0).text = "Alice"
    table.cell(0, 1).text = "London"
    table.cell(1, 0).text = "Bob"
    table.cell(1, 1).text = "Paris"
    block = TableChunker.extract_word(doc)[0]
    assert block.header_rows == 0
    assert block.rows[0] == ("Alice", "London")


def test_real_pdf_tables_prose_order_pages_and_no_duplicate_cells(tmp_path, tokenizer):
    path = tmp_path / "tables.pdf"
    _pdf_fixture(path, pages=2)
    blocks = TableChunker.extract_pdf(str(path), MagicMock())
    tables = [b for b in blocks if b.table_index]
    assert [b.table_index for b in tables] == [1, 2, 3, 4]
    assert [b.page_number for b in tables] == [1, 1, 2, 2]
    assert blocks[0].text == "Before tables"
    assert blocks[1] == tables[0]
    prose = "\n".join(b.text for b in blocks if not b.table_index)
    assert "Alice" not in prose and "birth_date" not in prose
    assert "After tables" in prose
    assert all(b.header_rows == 1 and b.header_source == "inferred" for b in tables)
    chunker = TableChunker(tokenizer, 140)
    for block in tables:
        for doc in chunker.chunk_table(block, {}):
            assert f"Table {block.table_index}" in doc.page_content
            assert f"PDF page {block.page_number}" in doc.page_content
            assert "Columns: name | birth_date" in doc.page_content
            assert chunker.fits(doc.page_content)


def test_real_pdf_merged_multirow_header(tmp_path, tokenizer):
    path = tmp_path / "merged.pdf"
    _pdf_fixture(path, merged=True)
    tables = [
        b for b in TableChunker.extract_pdf(str(path), MagicMock()) if b.table_index
    ]
    assert len(tables) == 2
    assert tables[0].header_rows == 2
    assert TableChunker._columns(tables[0]) == ["Person > name", "Person > birth_date"]
    docs = TableChunker(tokenizer, 180).chunk_table(tables[0], {})
    assert all(
        "Columns: Person > name | Person > birth_date" in d.page_content for d in docs
    )


@pytest.mark.parametrize(
    "kind",
    [
        PdfPageChunker,
        HeadingChunker,
        RecursiveChunker,
        SlidingWindowChunker,
        SentenceWindowChunker,
        SemanticChunker,
    ],
)
def test_pdf_labels_reach_table_prose_and_llm_context(tmp_path, tokenizer, kind):
    from pypdf import PdfReader
    from pypdf.constants import PageLabelStyle
    from Helpers.Helpers import Helpers

    path = tmp_path / "labelled.pdf"
    _pdf_fixture(path, pages=2)
    writer = PdfWriter()
    writer.append(PdfReader(path))
    writer.set_page_label(0, 0, style=PageLabelStyle.LOWERCASE_ROMAN, start=3)
    writer.set_page_label(1, 1, style=PageLabelStyle.DECIMAL, start=10)
    writer.write(path)

    blocks = TableChunker.extract_pdf(str(path), MagicMock())
    assert {b.page_number: b.page_label for b in blocks} == {1: "iii", 2: "10"}
    metadata = {"FilePath": str(path), "FileName": path.name, "FileType": "pdf"}
    docs, _ = _chunker(kind, size=240).chunk_document(
        "",
        metadata,
        blocks,
        tokenizer,
    )
    assert any("TableIndex" in doc.metadata for doc in docs)
    assert any("TableIndex" not in doc.metadata for doc in docs)
    helper = object.__new__(Helpers)
    helper.cfg = MagicMock()
    for doc in docs:
        physical = doc.metadata["PageNumber"]
        printed = doc.metadata["PageLabel"]
        reference = f"Page {printed}"
        assert physical in (1, 2)
        if kind is PdfPageChunker or "TableIndex" in doc.metadata:
            assert reference in doc.page_content
        assert reference in helper.format_document(doc)
        assert "PDF page" not in helper.format_document(doc)
        assert "PageNumber:" not in helper.format_document(doc)


@pytest.mark.parametrize(
    "kind",
    [
        RecursiveChunker,
        HeadingChunker,
        PdfPageChunker,
        SemanticChunker,
        SentenceWindowChunker,
        SlidingWindowChunker,
    ],
)
def test_shared_path_covers_configured_prose_chunkers(kind, tokenizer):
    blocks = [
        DocumentBlock(text="Before.", heading_path="People", page_number=3),
        DocumentBlock(
            rows=(("name", "birth_date"), ("Alice", "1980-01-01")),
            table_index=1,
            header_rows=1,
            heading_path="People",
            page_number=3,
        ),
        DocumentBlock(text="After.", heading_path="People", page_number=3),
    ]
    chunks, vectors = _chunker(kind).chunk_document(
        "",
        {"FileType": "docx", "FilePath": "must-not-reopen.docx"},
        blocks,
        tokenizer,
    )
    assert len(chunks) == 3
    assert "birth_date" in chunks[1].page_content
    assert [d.metadata["MyChunk"] for d in chunks] == [0, 1, 2]
    assert vectors == [None, None, None]
    assert chunks[1].metadata["HeadingPath"] == "People"
    assert chunks[1].metadata["PageNumber"] == 3
    assert all("birth_date" not in d.page_content for d in [chunks[0], chunks[2]])


def test_normalization_and_masking_apply_to_cells_and_context(tokenizer):
    masker = object.__new__(Masker)
    masker.apply_masking = True
    masker.cfg = MagicMock()
    masker.cfg.get_int.return_value = 0
    masker.cfg.get_str.return_value = "none"
    masker.pretty = MagicMock()
    import re

    masker.specs = [(re.compile("alice"), "[MASKED]", "name")]
    blocks = TableChunker.normalize_blocks(
        [
            DocumentBlock(
                rows=(("name",), ("ALICE",)),
                table_index=1,
                header_rows=1,
                heading_path="ALICE",
                caption="ALICE",
                text="ALICE",
            ),
        ],
        UnicodeNormalizer(),
        masker,
    )
    text = TableChunker(tokenizer, 200).chunk_table(blocks[0], {})[0].page_content
    assert "ALICE" not in text and "alice" not in text
    assert "[MASKED]" in text
    assert blocks[0].text == "[MASKED]"


def test_oversized_row_continuations_preserve_every_character(tokenizer):
    cell = "1980-01-01 " * 70
    block = DocumentBlock(rows=(("birth_date",), (cell,)), table_index=2, header_rows=1)
    chunker = TableChunker(tokenizer, 150, pretty=MagicMock())
    chunks = chunker.chunk_table(block, {})
    assert len(chunks) > 1
    recovered = []
    for doc in chunks:
        assert "Table 2\nColumns: birth_date\nRow 2 (part " in doc.page_content
        assert "; column 1): " in doc.page_content
        recovered.append(doc.page_content.split("; column 1): ", 1)[1])
        assert chunker.fits(doc.page_content)
    assert "".join(recovered) == cell


def test_impossible_header_budget_is_explicit(tokenizer):
    block = DocumentBlock(
        rows=(("birth_date" * 50,), ("1980-01-01",)), table_index=1, header_rows=1
    )
    with pytest.raises(DataProcessingError, match="complete header exceed"):
        TableChunker(tokenizer, 80).chunk_table(block, {})


def test_no_room_for_row_data_is_explicit(tokenizer):
    block = DocumentBlock(
        rows=(("birth_date",), ("x" * 100,)), table_index=1, header_rows=1
    )
    prefix_size = len(tokenizer.encode("Table 1\nColumns: birth_date"))
    with pytest.raises(DataProcessingError, match="no room for row data"):
        TableChunker(tokenizer, prefix_size, pretty=MagicMock()).chunk_table(block, {})


def test_headerless_rows_and_cell_delimiters_are_preserved(tokenizer):
    block = DocumentBlock(
        rows=(("Alice | Smith", "London\nUK"), ("Bob", "Paris")), table_index=1
    )
    docs = TableChunker(tokenizer, 200).chunk_table(block, {})
    assert "Columns: column_1 | column_2" in docs[0].page_content
    assert r"Row 1: Alice \| Smith | London\nUK" in docs[0].page_content
    assert "Row 2: Bob | Paris" in docs[0].page_content


def test_documents_without_tables_keep_original_chunk_path():
    chunker = _chunker(HeadingChunker)
    expected = ([], None)
    with patch.object(HeadingChunker, "chunk", return_value=expected) as original:
        assert chunker.chunk_document("unchanged", {"FileType": "md"}, []) == expected
    original.assert_called_once_with("unchanged", {"FileType": "md"})


def test_structured_blocks_require_tokenizer():
    with pytest.raises(DataProcessingError, match="requires the embedding tokenizer"):
        _chunker(HeadingChunker).chunk_document("", {}, [DocumentBlock(table_index=1)])


def test_delegation_calls_table_helper_only_for_detected_table(tokenizer):
    from langchain_core.documents import Document

    before = DocumentBlock(text="Before.", heading_path="People")
    table = DocumentBlock(
        rows=(("birth_date",), ("1980-01-01",)),
        table_index=2,
        header_rows=1,
        heading_path="People",
    )
    after = DocumentBlock(text="After.", heading_path="People")
    original_metadata = {"FilePath": "source.docx", "FileType": "docx"}
    prose_before = Document(page_content="Before.", metadata={"HeadingPath": "People"})
    table_doc = Document(page_content="Table 2", metadata={"TableIndex": 2})
    prose_after = Document(page_content="After.", metadata={"HeadingPath": "People"})
    with (
        patch.object(
            HeadingChunker,
            "chunk_prose",
            side_effect=[
                ([prose_before], [[1.0]]),
                ([prose_after], [[2.0]]),
            ],
        ) as prose,
        patch.object(TableChunker, "chunk_table", return_value=[table_doc]) as delegate,
        patch.object(HeadingChunker, "chunk") as source_reread,
    ):
        docs, vectors = _chunker(HeadingChunker).chunk_document(
            "",
            original_metadata,
            [before, table, after],
            tokenizer,
        )
    assert docs == [prose_before, table_doc, prose_after]
    assert vectors == [[1.0], None, [2.0]]
    assert [doc.metadata["MyChunk"] for doc in docs] == [0, 1, 2]
    assert [call.args[0] for call in prose.call_args_list] == ["Before.", "After."]
    delegate.assert_called_once_with(
        table,
        {
            **original_metadata,
            "HeadingPath": "People",
        },
    )
    source_reread.assert_not_called()
    assert original_metadata == {"FilePath": "source.docx", "FileType": "docx"}


def test_no_detection_does_not_instantiate_table_delegate():
    with (
        patch.object(HeadingChunker, "chunk", return_value=([], None)) as prose,
        patch("Strategies.Chunkers.TableChunker.TableChunker") as delegate,
    ):
        _chunker(HeadingChunker).chunk_document("ordinary prose", {}, [])
    prose.assert_called_once_with("ordinary prose", {})
    delegate.assert_not_called()


def test_pdf_failure_warns_and_preserves_fallback(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    pretty = MagicMock()
    assert TableChunker.extract_pdf(str(path), pretty) == []
    pretty.write.assert_called_once()
    assert "Using existing text extraction" in pretty.write.call_args.args[2]


def test_pdf_text_progress_preserves_pdfminer_output(tmp_path):
    from pdfminer.high_level import extract_text

    path = tmp_path / "text-progress.pdf"
    _pdf_fixture(path, pages=2)
    with patch("Pipeline.LoadAndClassifyProcessor.tqdm") as progress:
        actual = LoadAndClassifyProcessor._extract_pdf_text(str(path))
    assert actual == extract_text(str(path))
    updates = progress.return_value.__enter__.return_value.update.call_args_list
    assert [call.args for call in updates] == [(1,), (1,)]
    progress.return_value.__exit__.assert_called_once()


def test_pdf_text_progress_closes_on_extraction_failure(tmp_path):
    path = tmp_path / "failed-text.pdf"
    _pdf_fixture(path)
    with (
        patch("Pipeline.LoadAndClassifyProcessor.tqdm") as progress,
        patch("Pipeline.LoadAndClassifyProcessor.TextConverter") as converter,
        patch(
            "Pipeline.LoadAndClassifyProcessor.PDFPageInterpreter.process_page",
            side_effect=ValueError("page extraction failed"),
        ),
    ):
        with pytest.raises(ValueError, match="page extraction failed"):
            LoadAndClassifyProcessor._extract_pdf_text(str(path))
    progress.return_value.__enter__.return_value.update.assert_not_called()
    progress.return_value.__exit__.assert_called_once()
    converter.return_value.close.assert_called_once()


def test_pdf_ocr_progress_keeps_fallback_and_closes_images():
    obj = object.__new__(LoadAndClassifyProcessor)
    obj.fileName = "scanned.pdf"
    obj.escapedFilePath = "scanned.pdf"
    obj.strategy = MagicMock()
    obj.strategy.strategy_type = StrategyType.CLASSIFY
    obj.valid_extsInstance = MagicMock()
    obj.valid_extsInstance.check.return_value = True
    pages = [MagicMock(), MagicMock()]
    with (
        patch.object(LoadAndClassifyProcessor, "_extract_pdf_text", return_value=""),
        patch(
            "Pipeline.LoadAndClassifyProcessor.convert_from_path", return_value=pages
        ),
        patch(
            "Pipeline.LoadAndClassifyProcessor.pytesseract.image_to_string",
            side_effect=["First page", "Second page"],
        ),
    ):
        content, disabled = obj._extract_content_for_current_file()
    assert content == "First page\nSecond page\n"
    assert disabled == ""
    for page in pages:
        page.close.assert_called_once()


def test_pdf_table_progress_reports_detection_and_assembly(tmp_path, capsys):
    path = tmp_path / "table-progress.pdf"
    _pdf_fixture(path, pages=2)
    blocks = TableChunker.extract_pdf(str(path), MagicMock())
    assert len([block for block in blocks if block.table_index]) == 4
    output = capsys.readouterr().err
    assert "PDF table detection" in output
    assert "PDF block assembly" in output
    assert "PDF table detection: 2 page [" in output
    assert "PDF block assembly: 2 page [" in output
    assert "%" not in output
    assert "|" not in output


def test_table_free_pdf_skips_discarded_text_and_word_assembly(tmp_path):
    import pdfplumber.page

    path = tmp_path / "no-detected-tables.pdf"
    _pdf_fixture(path, pages=2)
    with (
        patch.object(pdfplumber.page.Page, "find_tables", return_value=[]) as detect,
        patch.object(pdfplumber.page.Page, "extract_text") as text,
        patch.object(pdfplumber.page.Page, "extract_words") as words,
    ):
        assert TableChunker.extract_pdf(str(path), MagicMock()) == []
    assert detect.call_count == 2
    text.assert_not_called()
    words.assert_not_called()


def test_table_free_pages_in_mixed_pdf_keep_original_prose(tmp_path):
    import pdfplumber
    import pdfplumber.page

    path = tmp_path / "mixed.pdf"
    _pdf_fixture(path, pages=2)
    with pdfplumber.open(path) as pdf:
        expected = " ".join(
            word["text"]
            for word in sorted(
                pdf.pages[0].extract_words(),
                key=lambda word: (word["top"], word["x0"]),
            )
        )
    original = pdfplumber.page.Page.find_tables

    def find_tables(page, *args, **kwargs):
        return [] if page.page_number == 1 else original(page, *args, **kwargs)

    with patch.object(pdfplumber.page.Page, "find_tables", find_tables):
        blocks = TableChunker.extract_pdf(str(path), MagicMock())
    assert blocks[0].text == expected
    assert blocks[0].page_number == 1
    assert [block.table_index for block in blocks if block.table_index] == [1, 2]
    assert [block.page_number for block in blocks if block.table_index] == [2, 2]


def test_pdf_page_detection_failure_keeps_page_text(tmp_path):
    path = tmp_path / "tables.pdf"
    _pdf_fixture(path, pages=2)
    import pdfplumber.page

    # The second page remains structured even when the first needs fallback.
    with patch.object(
        pdfplumber.page.Page,
        "find_tables",
        side_effect=[ValueError("bad geometry"), []],
    ):
        pretty = MagicMock()
        # With no surviving tables, ingestion deliberately uses the old path.
        assert TableChunker.extract_pdf(str(path), pretty) == []
        pretty.write.assert_called_once()


def test_pdf_failed_page_preserved_alongside_successful_tables(tmp_path):
    path = tmp_path / "tables.pdf"
    _pdf_fixture(path, pages=2)
    import pdfplumber.page

    original = pdfplumber.page.Page.find_tables

    def find_tables(page, *args, **kwargs):
        if page.page_number == 1:
            raise ValueError("bad geometry")
        return original(page, *args, **kwargs)

    with patch.object(pdfplumber.page.Page, "find_tables", find_tables):
        blocks = TableChunker.extract_pdf(str(path), MagicMock())
    assert blocks[0].page_number == 1 and not blocks[0].table_index
    assert "Alice" in blocks[0].text
    assert [b.page_number for b in blocks if b.table_index] == [2, 2]


def _strategy(tokenizer, chunker, blocks):
    obj = object.__new__(DocumentIngestionStrategy)
    obj._document_blocks = blocks
    obj._chunk_map = {"DEFAULT": "RECURSIVE"}
    obj.chunker = chunker
    obj.content = "Content"
    obj.doc = {"meta": {"FileType": "docx", "FilePath": "tables.docx"}}
    obj.fileType = "docx"
    obj.file = "tables.docx"
    obj.escapedFilePath = "tables.docx"
    obj.pretty = MagicMock()
    obj.cfg = MagicMock()
    obj.cfg.get_int.return_value = 512
    obj.perf_logger = MagicMock()
    obj.helpers = MagicMock()
    obj.helpers.get_model_args.return_value = {"hf_model_name": "local-test-tokenizer"}
    obj.models_cache = MagicMock()
    obj.models_cache.get_embedding_tokenizer.return_value = tokenizer
    obj.models_cache.truncate_texts.return_value = []
    obj.chromaDBHelper = MagicMock()
    obj.embedder = MagicMock()
    obj.embedder.embed_documents.return_value = [[0.1, 0.2]]
    obj.collection = MagicMock()
    obj.client = MagicMock()
    obj.client.get_max_batch_size.return_value = 5461
    obj.collection_name = "test"
    obj.bm25_retriever = MagicMock()
    obj.regex_retriever = MagicMock()
    obj.graph_retriever = MagicMock()
    return obj


def test_final_text_reaches_embedding_storage_and_all_secondary_indexes(tokenizer):
    block = DocumentBlock(
        rows=(("name", "birth_date"), ("Alice", "1980-01-01")),
        table_index=1,
        header_rows=1,
    )
    prose_chunker = _chunker(RecursiveChunker)
    obj = _strategy(tokenizer, prose_chunker, [block])
    with patch.object(
        DocumentIngestionStrategy, "_make_chunker", return_value=prose_chunker
    ):
        docs, vectors = obj._chunk_current_document()
    assert isinstance(obj.chromaDBHelper, MagicMock)
    obj.chromaDBHelper.clean_metadata.return_value = docs[0].metadata
    texts, metas, persisted, embeddings = obj._prepare_vectors_for_chunks(docs, vectors)
    assert persisted == texts == [docs[0].page_content]
    assert "\nColumns: name | birth_date\nRow 2: Alice | 1980-01-01" in persisted[0]
    obj.embedder.embed_documents.assert_called_once_with(persisted)
    obj._upsert_chunks(["id"], embeddings, metas, persisted, 1, 0)
    obj._update_secondary_indexes(["id"], persisted, metas)
    assert isinstance(obj.collection, MagicMock)
    assert obj.collection.add.call_args.kwargs["documents"] == persisted
    for retriever in [obj.bm25_retriever, obj.graph_retriever, obj.regex_retriever]:
        assert retriever.ingest_file.call_args.args[4] == persisted


@pytest.mark.parametrize(
    "count, expected_sizes",
    [
        (0, []),
        (1, [1]),
        (5461, [5461]),
        (5695, [5461, 234]),
        (10922, [5461, 5461]),
    ],
)
def test_chroma_writes_respect_client_batch_limit(tokenizer, count, expected_sizes):
    obj = _strategy(tokenizer, _chunker(RecursiveChunker), [])
    ids = [f"id-{i}" for i in range(count)]
    embeddings = [[float(i), 0.0] for i in range(count)]
    metas = [{"FileName": "large.pdf", "MyChunk": i} for i in range(count)]
    texts = [f"chunk {i}" for i in range(count)]

    obj._upsert_chunks(ids, embeddings, metas, texts, count, 0)

    assert isinstance(obj.collection, MagicMock)
    calls = obj.collection.add.call_args_list
    assert [len(call.kwargs["ids"]) for call in calls] == expected_sizes
    for key, expected in [
        ("ids", ids),
        ("embeddings", embeddings),
        ("metadatas", metas),
        ("documents", texts),
    ]:
        assert [value for call in calls for value in call.kwargs[key]] == expected


def test_failed_chroma_batch_propagates_without_success_message(tokenizer):
    obj = _strategy(tokenizer, _chunker(RecursiveChunker), [])
    assert isinstance(obj.collection, MagicMock)
    assert isinstance(obj.pretty, MagicMock)
    obj.client.get_max_batch_size.return_value = 1
    obj.collection.add.side_effect = [None, RuntimeError("write failed")]
    with pytest.raises(RuntimeError, match="write failed"):
        obj._upsert_chunks(["a", "b"], [[0.0], [1.0]], [{}, {}], ["a", "b"], 2, 0)
    obj.pretty.write.assert_not_called()


def test_invalid_chroma_batch_limit_fails_explicitly(tokenizer):
    obj = _strategy(tokenizer, _chunker(RecursiveChunker), [])
    assert isinstance(obj.collection, MagicMock)
    obj.client.get_max_batch_size.return_value = 0
    with pytest.raises(DataProcessingError, match="non-positive"):
        obj._upsert_chunks(["a"], [[0.0]], [{}], ["a"], 1, 0)
    obj.collection.add.assert_not_called()


def test_over_budget_final_table_fails_before_embedding(tokenizer):
    obj = _strategy(tokenizer, _chunker(RecursiveChunker, 20), [])
    from langchain_core.documents import Document

    docs = [Document(page_content="x" * 512, metadata={"TableIndex": 1})]
    with pytest.raises(DataProcessingError, match="final embedding token budget"):
        obj._prepare_vectors_for_chunks(docs, None)
    obj.embedder.embed_documents.assert_not_called()


def test_table_budget_is_independent_of_prose_and_preserved_for_embedding(tokenizer):
    block = DocumentBlock(
        rows=(("name",), *(("x" * 50,) for _ in range(20))),
        table_index=1,
        header_rows=1,
    )
    prose_chunker = _chunker(RecursiveChunker, 20)
    obj = _strategy(tokenizer, prose_chunker, [block])
    docs, vectors = prose_chunker.chunk_document("", {}, [block], tokenizer)

    assert prose_chunker.chunk_size == 20
    assert len(docs) > 1
    assert all(20 < len(tokenizer.encode(doc.page_content)) <= 512 for doc in docs)
    assert sum(doc.page_content.count("x" * 50) for doc in docs) == 20
    texts, _, persisted, _ = obj._prepare_vectors_for_chunks(docs, vectors)
    assert persisted == texts == [doc.page_content for doc in docs]
    obj.embedder.embed_documents.assert_called_once_with(persisted)
    assert isinstance(obj.cfg, MagicMock)
    obj.cfg.get_int.assert_called_once_with("_CHUNKERS.TABLE.MAX_CHUNK_TOKENS")


def test_table_budget_counts_special_tokens_and_respects_model_limit(tokenizer):
    chunker = TableChunker(tokenizer, 512)
    assert chunker.fits("x" * 510)
    assert not chunker.fits("x" * 511)
    tokenizer.model_max_length = 128
    capped = TableChunker(tokenizer, 512)
    assert capped.max_tokens == 128
    assert capped.fits("x" * 126)
    assert not capped.fits("x" * 127)


def test_changed_prose_invalidates_precomputed_embedding(tokenizer):
    from langchain_core.documents import Document

    obj = _strategy(tokenizer, _chunker(RecursiveChunker), [])
    docs = [Document(page_content="Original prose", metadata={})]
    assert isinstance(obj.models_cache, MagicMock)
    obj.models_cache.truncate_texts.return_value = ["Truncated prose"]
    obj._prepare_vectors_for_chunks(docs, [[9.0, 9.0]])
    obj.embedder.embed_documents.assert_called_once_with(["Truncated prose"])


def test_existing_chunks_survive_failed_budget(tokenizer):
    obj = _strategy(
        tokenizer,
        _chunker(RecursiveChunker, 20),
        [
            DocumentBlock(rows=(("birth_date" * 100,),), table_index=1, header_rows=1),
        ],
    )
    with (
        patch.object(DocumentIngestionStrategy, "_open_collection_for_ingestion"),
        patch.object(DocumentIngestionStrategy, "_make_doc"),
        patch.object(
            DocumentIngestionStrategy, "_gate_unsupported_language", return_value=False
        ),
        patch.object(
            DocumentIngestionStrategy, "_make_chunker", return_value=obj.chunker
        ),
    ):
        with pytest.raises(DataProcessingError):
            obj.ingest()
    assert isinstance(obj.collection, MagicMock)
    obj.collection.delete.assert_not_called()


def test_word_loader_carries_converted_document_blocks(tmp_path):
    doc = _word_fixture(tmp_path / "converted.docx")
    obj = object.__new__(LoadAndClassifyProcessor)
    obj.fileName = "original.doc"
    obj.escapedFilePath = "original.doc"
    obj.strategy = MagicMock()
    obj.strategy.strategy_type = StrategyType.DOCUMENT_INGESTION
    obj.valid_extsInstance = MagicMock()
    obj.valid_extsInstance.check.side_effect = [False, False, True]
    obj.consider_as_text_file = []
    obj.office_doc_extraction = {"Word": True}
    obj.office_convInstance = MagicMock()
    obj.office_convInstance.convert_office_file.return_value = ("docx", doc)
    with patch(
        "Helpers.OfficeDocConverter.OfficeDocConverter.is_windows_supported",
        return_value=True,
    ):
        content, disabled = obj._extract_content_for_current_file()
    assert disabled == ""
    assert len([b for b in obj._document_blocks if b.table_index]) == 2
    assert content.index("Before") < content.index("Alice0") < content.index("After")


def test_pdf_page_chunker_preserves_ocr_fallback():
    chunker = _chunker(PdfPageChunker)
    with (
        patch.object(PdfPageChunker, "_parse_pdf", return_value=[]),
        patch(
            "Strategies.Chunkers.PdfPageChunker.os.path.isfile",
            return_value=True,
        ),
    ):
        docs, _ = chunker.chunk(
            "OCR text from scanned page",
            {
                "FileType": "pdf",
                "FilePath": "scanned.pdf",
            },
        )
    assert len(docs) == 1
    assert "OCR text from scanned page" in docs[0].page_content
