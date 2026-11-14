"""Ordered Word/PDF extraction and format-independent table chunking."""

import re
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from docx.document import Document as WordDocument
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from tqdm import tqdm

from Algos.Masker import Masker
from Algos.UnicodeNormalizer import UnicodeNormalizer
from Commons.Exceptions import DataProcessingError
from Gui.PrettyWriter import PrettyWriter
from Strategies.Chunkers.ChunkerStrategy import format_pdf_page_reference
from Strategies.Chunkers.HeadingChunker import docx_heading_level

if TYPE_CHECKING:
    from pdfplumber.table import Table as PdfTable


class TableTokenizer(Protocol):
    """Only token counting is needed; no callable tokenizer interface."""

    @property
    def model_max_length(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...


class _HeaderFlag(Protocol):
    """Typed XML attribute access absent from python-docx's lxml annotations."""

    def get(self, key: str, default: str) -> str: ...


@dataclass(frozen=True)
class DocumentBlock:
    """Transient structure, kept outside Chroma's scalar-only metadata."""

    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    table_index: int = 0
    header_rows: int = 0
    header_source: str = "positional"
    heading_path: str = ""
    page_number: int = 0
    page_label: str = ""
    caption: str = ""

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.heading_path:
            result["HeadingPath"] = self.heading_path
        if self.page_number:
            result["PageNumber"] = self.page_number
            result["PageLabel"] = self.page_label
        return result


class TableChunker:
    """Share extraction, identity and row splitting across all prose chunkers."""

    def __init__(
        self,
        tokenizer: TableTokenizer,
        max_tokens: int,
        *,
        pretty: PrettyWriter | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = min(max_tokens, tokenizer.model_max_length)
        self.pretty = pretty or PrettyWriter()
        if self.max_tokens <= 0:
            raise DataProcessingError("Table chunk token budget must be positive.")

    @staticmethod
    def _header_count(rows: tuple[tuple[str, ...], ...]) -> int:
        # Infer only when a text-only first row contrasts with typed data.
        # Ambiguous all-text tables retain every row as data.
        if len(rows) < 2 or not rows[0] or not all(rows[0]):
            return 0
        first = rows[0]
        if any(re.search(r"\d", cell) for cell in first):
            return 0
        return int(any(re.search(r"\d", cell) for row in rows[1:] for cell in row))

    @classmethod
    def extract_word(cls, doc: WordDocument) -> list[DocumentBlock]:
        """Use the loaded document, including legacy DOC's temporary conversion."""
        blocks: list[DocumentBlock] = []
        trail: dict[int, str] = {}
        body: list[str] = []
        table_index = 0
        caption = ""
        for item in doc.iter_inner_content():
            path = " > ".join(trail[k] for k in sorted(trail))
            if isinstance(item, Paragraph):
                style = item.style.name if item.style else ""
                level = docx_heading_level(style or "")
                if level:
                    if body:
                        blocks.append(
                            DocumentBlock(text="\n".join(body), heading_path=path)
                        )
                        body = []
                    trail = {k: v for k, v in trail.items() if k < level}
                    trail[level] = item.text.strip()
                elif item.text.strip():
                    body.append(item.text.strip())
                caption = item.text.strip() if style == "Caption" else ""
                continue

            if body:
                blocks.append(DocumentBlock(text="\n".join(body), heading_path=path))
                body = []
            table_index += 1
            width = len(item.columns)
            rows: list[tuple[str, ...]] = []
            explicit = 0
            for row in item.rows:
                values = [""] * row.grid_cols_before
                for cell in row.cells:
                    # Preserve nested table content within its containing cell.
                    parts: list[str] = []
                    for child in cell.iter_inner_content():
                        if isinstance(child, Paragraph):
                            parts.append(child.text)
                        else:
                            parts.extend(
                                " | ".join(c.text for c in r.cells) for r in child.rows
                            )
                    values.append("\n".join(parts).strip())
                values.extend([""] * (width - len(values)))
                rows.append(tuple(values))
                # python-docx has no public API for repeating header flags.
                flags = cast(
                    list[_HeaderFlag],
                    row._tr.xpath(  # type: ignore[reportPrivateUsage]
                        "w:trPr/w:tblHeader"
                    ),
                )
                flag = flags[0] if flags else None
                if (
                    len(rows) == explicit + 1
                    and flag is not None
                    and flag.get(qn("w:val"), "true") not in ("0", "false", "off")
                ):
                    explicit += 1
            data = tuple(rows)
            headers = explicit or cls._header_count(data)
            blocks.append(
                DocumentBlock(
                    rows=data,
                    table_index=table_index,
                    header_rows=headers,
                    header_source=(
                        "explicit"
                        if explicit
                        else ("inferred" if headers else "positional")
                    ),
                    heading_path=path,
                    caption=caption,
                )
            )
            caption = ""
        if body:
            blocks.append(
                DocumentBlock(
                    text="\n".join(body),
                    heading_path=" > ".join(trail[k] for k in sorted(trail)),
                )
            )
        return blocks

    @classmethod
    def _pdf_rows(cls, table: "PdfTable") -> tuple[tuple[tuple[str, ...], ...], int]:
        raw = table.extract()
        if not raw:
            return (), 0
        width = max(len(row) for row in raw)
        edges = sorted({x for cell in table.cells for x in (cell[0], cell[2])})
        owners = {
            cell: (raw[r][c] or "").strip()
            for r, row in enumerate(table.rows)
            for c, cell in enumerate(row.cells)
            if cell is not None
        }
        rows: list[tuple[str, ...]] = []
        for r, row in enumerate(raw):
            values: list[str] = []
            for c in range(width):
                value = row[c] if c < len(row) else None
                if value is None and c + 1 < len(edges):
                    # A missing grid position may belong to a spanning cell.
                    # Recover only a geometric owner, never a guessed label.
                    x = (edges[c] + edges[c + 1]) / 2
                    y = (table.rows[r].bbox[1] + table.rows[r].bbox[3]) / 2
                    for (x0, top, x1, bottom), text in owners.items():
                        if x0 <= x < x1 and top <= y < bottom:
                            value = text
                            break
                values.append((value or "").strip())
            rows.append(tuple(values))
        data = tuple(rows)
        headers = cls._header_count(data)
        if headers and any(cell is None for cell in raw[0]):
            # A merged group row plus a complete text-only label row is a
            # multi-row header; stop before typed values or ordinary data.
            for row in raw[1:]:
                if any(re.search(r"\d", cell or "") for cell in row):
                    break
                headers += 1
                if all(cell is not None for cell in row):
                    break
        return data, headers

    @classmethod
    def extract_pdf(
        cls,
        file_path: str,
        pretty: PrettyWriter,
        *,
        detect_printed_labels: bool = True,
    ) -> list[DocumentBlock]:
        """Detect text-layer tables; do not guess cross-page continuations."""
        import pdfplumber
        from pdfminer.pdfparser import PDFSyntaxError
        from pdfplumber.utils.exceptions import (MalformedPDFException,
                                                 PdfminerException)
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        from Strategies.Chunkers.PdfPageChunker import PdfPageChunker

        blocks: list[DocumentBlock] = []
        table_index = 0
        try:
            raw_labels = PdfPageChunker._page_labels(PdfReader(file_path))
            labels: dict[int, str] = {}
            with pdfplumber.open(file_path) as pdf:
                detected: list[
                    list[tuple[tuple[float, float, float, float], DocumentBlock]] | None
                ] = []
                for page_number, page in enumerate(
                    tqdm(
                        pdf.pages,
                        desc="PDF table detection",
                        unit="page",
                        dynamic_ncols=True,
                        bar_format="{desc}: {n_fmt} {unit} [{elapsed}, {rate_fmt}]",
                    ),
                    1,
                ):
                    try:
                        page_tables: list[
                            tuple[tuple[float, float, float, float], DocumentBlock]
                        ] = []
                        for table in page.find_tables():
                            data, headers = cls._pdf_rows(table)
                            if not any(cell for row in data for cell in row):
                                continue
                            page_tables.append(
                                (
                                    table.bbox,
                                    DocumentBlock(
                                        rows=data,
                                        header_rows=headers,
                                        header_source=(
                                            "inferred" if headers else "positional"
                                        ),
                                        page_number=page_number,
                                    ),
                                )
                            )
                        detected.append(page_tables)
                    except (
                        ValueError,
                        TypeError,
                        IndexError,
                        KeyError,
                        PdfminerException,
                        MalformedPDFException,
                    ) as exc:
                        pretty.write(
                            "W",
                            "TableChunker",
                            f"Table extraction failed on page {page_number}: {exc}. Using page text.",
                        )
                        detected.append(None)
                # No structured blocks will be consumed unless a usable table exists.
                if not any(detected):
                    return []
                for page_number, (page, extracted) in enumerate(
                    tqdm(
                        zip(pdf.pages, detected),
                        total=len(pdf.pages),
                        desc="PDF block assembly",
                        unit="page",
                        dynamic_ncols=True,
                        bar_format="{desc}: {n_fmt} {unit} [{elapsed}, {rate_fmt}]",
                    ),
                    1,
                ):
                    printed = (
                        PdfPageChunker._detect_printed_label(page.extract_text() or "")
                        if detect_printed_labels
                        else ""
                    )
                    raw_label = (
                        raw_labels[page_number - 1]
                        if page_number <= len(raw_labels)
                        else ""
                    )
                    labels[page_number] = printed or (
                        raw_label if raw_label != str(page_number) else ""
                    )
                    try:
                        if extracted is None:
                            blocks.append(
                                DocumentBlock(
                                    text=page.extract_text() or "",
                                    page_number=page_number,
                                )
                            )
                            continue
                        # Explicit sorting avoids callback-based extraction interfaces.
                        ordered = sorted(
                            (bbox[1], bbox[0], i)
                            for i, (bbox, _) in enumerate(extracted)
                        )
                        words = page.extract_words()
                        prose: list[list[tuple[float, float, str]]] = [
                            [] for _ in range(len(ordered) + 1)
                        ]
                        for word in words:
                            x = (word["x0"] + word["x1"]) / 2
                            y = (word["top"] + word["bottom"]) / 2
                            if any(
                                x0 <= x <= x1 and y0 <= y <= y1
                                for (x0, y0, x1, y1), _ in extracted
                            ):
                                continue
                            slot = sum(y >= extracted[i][0][1] for _, _, i in ordered)
                            prose[slot].append((word["top"], word["x0"], word["text"]))
                        for slot, bucket in enumerate(prose):
                            text = " ".join(word for _, _, word in sorted(bucket))
                            if text:
                                blocks.append(
                                    DocumentBlock(text=text, page_number=page_number)
                                )
                            if slot < len(ordered):
                                table_index += 1
                                block = extracted[ordered[slot][2]][1]
                                blocks.append(replace(block, table_index=table_index))
                    except (
                        ValueError,
                        TypeError,
                        IndexError,
                        KeyError,
                        PdfminerException,
                        MalformedPDFException,
                    ) as exc:
                        pretty.write(
                            "W",
                            "TableChunker",
                            f"Table extraction failed on page {page_number}: {exc}. Using page text.",
                        )
                        blocks.append(
                            DocumentBlock(
                                text=page.extract_text() or "",
                                page_number=page_number,
                            )
                        )
        except (
            OSError,
            ValueError,
            PDFSyntaxError,
            PdfReadError,
            PdfminerException,
            MalformedPDFException,
        ) as exc:
            pretty.write(
                "W",
                "TableChunker",
                f"PDF table extraction failed: {exc}. Using existing text extraction.",
            )
            return []
        return (
            [
                replace(block, page_label=labels.get(block.page_number, ""))
                for block in blocks
            ]
            if table_index
            else []
        )

    @staticmethod
    def normalize_blocks(
        blocks: list[DocumentBlock],
        normalizer: UnicodeNormalizer,
        masker: Masker,
    ) -> list[DocumentBlock]:
        result: list[DocumentBlock] = []
        for block in blocks:
            # Normalize/mask payload and context before any chunker sees them;
            # original-file re-reads must not restore unmasked values.
            rows = tuple(
                tuple(masker.mask(normalizer.normalize(cell)) for cell in row)
                for row in block.rows
            )
            result.append(
                replace(
                    block,
                    rows=rows,
                    text=masker.mask(
                        normalizer.normalize(block.text, preserve_newlines=True)
                    ),
                    heading_path=masker.mask(normalizer.normalize(block.heading_path)),
                    caption=masker.mask(normalizer.normalize(block.caption)),
                )
            )
        return result

    def fits(self, text: str) -> bool:
        return (
            len(self.tokenizer.encode(text, add_special_tokens=True)) <= self.max_tokens
        )

    @staticmethod
    def _cell_text(text: str) -> str:
        # Escape serialization delimiters without changing column ownership.
        return (
            text.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )

    @staticmethod
    def _columns(block: DocumentBlock) -> list[str]:
        width = max((len(row) for row in block.rows), default=0)
        columns: list[str] = []
        for i in range(width):
            labels: list[str] = []
            for row in block.rows[: block.header_rows]:
                if i < len(row) and row[i] and row[i] not in labels:
                    labels.append(row[i])
            columns.append(" > ".join(labels) if labels else f"column_{i + 1}")
        return columns

    def chunk_table(
        self, block: DocumentBlock, metadata: dict[str, Any]
    ) -> list[Document]:
        columns = self._columns(block)
        identity = f"Table {block.table_index}"
        if block.heading_path:
            identity += f" | Section: {block.heading_path}"
        if block.page_number:
            identity += " | " + format_pdf_page_reference(
                block.page_number, block.page_label
            )
        if block.caption:
            identity += f" | Caption: {block.caption}"
        header = "Columns: " + " | ".join(self._cell_text(column) for column in columns)
        prefix = f"{identity}\n{header}"
        if not self.fits(prefix):
            raise DataProcessingError(
                f"Table {block.table_index}: identity and complete header exceed "
                f"the {self.max_tokens}-token budget. Increase the chunk size."
            )
        meta = dict(metadata)
        meta.update(block.metadata())
        meta.update(
            TableIndex=block.table_index,
            TableHeader=header,
            TableContext=identity,
            TableHeaderSource=block.header_source,
        )
        texts: list[str] = []
        pending: list[str] = []
        for row_number, row in enumerate(
            block.rows[block.header_rows :], block.header_rows + 1
        ):
            line = f"Row {row_number}: " + " | ".join(
                self._cell_text(cell) for cell in row
            )
            candidate = "\n".join([prefix, *pending, line])
            if self.fits(candidate):
                pending.append(line)
                continue
            if pending:
                texts.append("\n".join([prefix, *pending]))
                pending = []
            if self.fits(f"{prefix}\n{line}"):
                pending.append(line)
                continue
            self.pretty.write(
                "W",
                "TableChunker",
                f"Table {block.table_index}, row {row_number} exceeds budget; splitting into column continuations.",
            )
            # Column numbers map each continuation back to the complete header.
            part = 0
            for column, cell in enumerate(row, 1):
                remaining = cell
                while True:
                    part += 1
                    label = (
                        f"{prefix}\nRow {row_number} (part {part}; column {column}): "
                    )
                    low, high = 0, len(remaining)
                    while low < high:
                        mid = (low + high + 1) // 2
                        if self.fits(label + remaining[:mid]):
                            low = mid
                        else:
                            high = mid - 1
                    if not self.fits(label) or (remaining and low == 0):
                        raise DataProcessingError(
                            f"Table {block.table_index}: no room for row data after "
                            "the complete header. Increase the chunk size."
                        )
                    texts.append(label + remaining[:low])
                    remaining = remaining[low:]
                    if not remaining:
                        break
        if pending or not texts:
            texts.append("\n".join([prefix, *pending]))
        return [
            Document(page_content=text, metadata=dict(meta), id=str(uuid.uuid4()))
            for text in texts
        ]
