# Standard library imports
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import pandas as pd  # type: ignore[reportMissingTypeStubs]
import pytesseract  # type: ignore[reportMissingTypeStubs]
from chromadb.api import Collection  # type: ignore[reportPrivateImportUsage]
from langdetect import detect  # type: ignore[reportMissingTypeStubs]
from openpyxl.worksheet.worksheet import Worksheet
from pdf2image import \
    convert_from_path  # type: ignore[reportUnknownVariableType]
from pdfminer.high_level import extract_text as extractPDF
from PIL import Image

from Algos.Masker import Masker
from Algos.UnicodeNormalizer import UnicodeNormalizer
from Commons.Exceptions import DataProcessingError, DocumentsDirError
from Commons.SingletonMixin import SingletonMixin
from Compliance.Exclusions import Exclusions
from Config.Config import Config
from Globals.CounterInstance import (ExclusionsCount, FailedCount,
                                     IgnoredCount, ProcessedCount)
from Globals.Globals import Globals
from Gui.Colors import BRIGHT_BLUE, CYAN, ORANGE, RESET
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.OfficeDocConverter import OfficeDocConverter
from Helpers.PerfLogger import PerfLogger
from Helpers.ValidExtensions import ValidExtensions
from Strategies.ProcessingStrategy import ProcessingStrategy
from Strategies.StrategyType import StrategyType


class LoadAndClassifyProcessor(SingletonMixin):
    """
    Singleton that walks a work directory, extracts text from many file
    formats (PDF, Office, images, etc.), and delegates either to a
    home-brew chunker or to the classification implementation.
    """

    def process(self) -> None:
        self.process_files()

    def __init__(
        self,
        strategy: ProcessingStrategy,
        *,
        allowed_paths: set[str] | None = None,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.strategy: ProcessingStrategy = strategy

        # Optional allow-set loaded from a DocClassify CSV
        self.allowed_paths: set[str] | None = allowed_paths

        # Utilities & counters
        self.failed_countInstance: FailedCount = FailedCount()
        self.processed_countInstance: ProcessedCount = ProcessedCount()
        self.ignored_countInstance: IgnoredCount = IgnoredCount()
        self.exclusions_countInstance: ExclusionsCount = ExclusionsCount()
        self.exclusions: Exclusions = Exclusions()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.pretty_always: PrettyWriter = PrettyWriter(always_on=True)
        self.helpers: Helpers = helpers or Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.csvWriter: CSVWriter = CSVWriter()
        self.masker: Masker = Masker()
        self.unicode_normalizer: UnicodeNormalizer = UnicodeNormalizer()
        self.perf_logger: PerfLogger = PerfLogger()

        self.fileName: str | None = None
        self.fileHash: str = "N/A"

        # Validators, converters & delegates
        self.valid_extsInstance: ValidExtensions = ValidExtensions()
        self.office_convInstance: OfficeDocConverter = OfficeDocConverter()
        self.globalsInstance: Globals = Globals()

        # Configuration
        self.cfg = cfg or Config()
        self.doc_dir: str = self.cfg.get_str("DOC_DIR")
        p = Path(self.doc_dir)
        # readable
        if not os.access(p, os.R_OK):
            self.pretty.write(
                "E",
                "Documents directory",
                f"Path {self.doc_dir} is not readable. Check DOC_DIR in Configuration/Config_Global.py",
            )
            raise DocumentsDirError

        self.use_exclusions: bool = self.cfg.get_bool("USE_EXCLUSIONS")
        # PROCESS_IF_UNCHANGED only applies to RAGLoad (DOCUMENT_INGESTION).
        self.process_unchanged: bool = (
            self.cfg.get_bool("PROCESS_IF_UNCHANGED", True)
            if strategy.strategy_type == StrategyType.DOCUMENT_INGESTION
            else False
        )
        self.friendly_name: str = self.cfg.get_str("_FRIENDLY_NAME")
        self.collection: Collection | None = None
        self.consider_as_text_file: list[str] = self.cfg.get_list(
            "_CONSIDER_AS_TEXT_FILE"
        )

        self.office_doc_extraction: Dict[str, bool] = self.cfg.get_dict(
            "_OFFICE_DOC_EXTRACTION"
        )
        if sys.platform != "win32":
            self.office_doc_extraction = {
                key: False for key in self.office_doc_extraction
            }
            self.pretty.write(
                "W",
                "Office conversion",
                "Legacy Office conversion is Windows-only; disabling _OFFICE_DOC_EXTRACTION on this host.",
                color=ORANGE,
            )

        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        if self.friendly_name == "RAGLoad":
            self.collection_name: str
            self.collection_name, _ = self.chromaDBHelper.change_chroma_collection(
                self.cfg.get_str("COLLECTION"), True  # type: ignore[reportArgumentType]
            )

        self.tesseract_path = self.helpers.configure_tesseract()

    def worksheet_to_dataframe(self, ws: Worksheet) -> pd.DataFrame:
        rows: list[Any] = list(ws.iter_rows(values_only=True))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=rows[0])

    def docChanged(self) -> bool:
        # ————————————————
        # 1) OPEN CHROMA & GET STORED HASH
        # ————————————————
        self.coll_name, self.persist_directory = (
            self.chromaDBHelper.chroma_coll_name_and_mkdir_or_del(
                "create", self.collection_name
            )
        )
        self.client, self.collection = (
            self.chromaDBHelper.get_chroma_client_and_collection(
                self.persist_directory, self.coll_name, stamp=True
            )
        )

        # 2) ask for up to one matching entry’s metadata
        resp: Any = self.collection.get(
            where={"FilePath": self.escapedFilePath},
            limit=1,
            include=["metadatas", "documents"],
        )

        self.fileHash = self.fileUtils.hash_file(self.escapedFilePath)
        # 3) pull out FileHash if it exists
        metas: list[Any] = resp.get("metadatas", [])
        if metas:
            prev_hash: str | None = metas[0].get("FileHash")
        else:
            prev_hash = None

        # ————————————————
        # 2) COMPUTE CURRENT HASH & COMPARE
        # ————————————————
        if prev_hash == self.fileHash:
            if self.process_unchanged is True:
                self.pretty.write(
                    "I", "Vector store", f"No change in {self.escapedFilePath}"
                )
                self.pretty.write(
                    "I",
                    "Vector store",
                    "but PROCESS_IF_UNCHANGED " f"{self.process_unchanged} override",
                )
                return True
            self.pretty.write(
                "A",
                "Vector Store",
                f"No change in {self.escapedFilePath}. Skip",
                color=CYAN,
            )
            self.processed_countInstance.increment()
            self.csvWriter.write_json2csv(
                {"FilePath": self.escapedFilePath, "Status": "NO CHANGE IN FILE"},
                "OK",
            )
            return False
        else:
            return True

    def _chunker_preserves_newlines(self) -> bool:
        """Check whether the chunker mapped to the current file type
        has ``PRESERVE_NEWLINES`` enabled in its ``_CHUNKERS`` config."""
        profile: str = self.cfg.get_str("_ACTIVE_CHUNKER_CONFIG")
        chunk_map: dict[str, str] = self.cfg.get_dict(f"_CHUNK_STRATEGY.{profile}")
        chunker_name: str = chunk_map.get(
            self.ftype, chunk_map.get("DEFAULT", "SEMANTIC")
        )
        return self.cfg.get_bool(f"_CHUNKERS.{chunker_name}.PRESERVE_NEWLINES", False)

    def _make_doc(self) -> Dict[str, Any]:
        self.content, _ = self.helpers.safe_decode_to_unicode(
            self.content, True
        )  # Assuming not already UTF-8.
        detected: str = str(detect(self.content))  # type: ignore[reportUnknownArgumentType]
        language: str = detected
        self.doc: Dict[str, Any] = {
            "meta": {
                "FileName": self.fileName,
                "FilePath": self.escapedFilePath,
                "CreationDate": self.creation_date,
                "FileType": self.ftype,
                "Language": language,
                "WordCount": self.fileUtils.count_words(self.content),
                "FileHash": self.fileHash,
            },
            "content": self.content,
        }

        return self.doc

    def _prepare_current_file(self, root: str, file_name: str) -> None:
        self.fileName = file_name
        self.filePath = os.path.join(root, self.fileName)
        self.escapedFilePath = self.fileUtils.normalize_path(self.filePath)

    def _skip_by_path_filters(self) -> bool:
        # When a classify CSV allow-set is active it is the
        # authoritative source — DocClassify already applied
        # exclusions, so we skip the exclusion check here.
        if self.allowed_paths is not None:
            normalized = os.path.normpath(self.escapedFilePath)
            if normalized not in self.allowed_paths:
                self.pretty.write(
                    "I",
                    "ClassifyCSV",
                    f"Skipped (not in classify CSV): {self.escapedFilePath}",
                )
                self.ignored_countInstance.increment()
                return True
            return False

        if self.use_exclusions and self.exclusions.contains(self.escapedFilePath):
            self.pretty.write(
                "W",
                "EXCLUSIONS",
                f"Excluding: {self.escapedFilePath}",
                color=ORANGE,
            )
            self.exclusions_countInstance.increment()
            return True

        return False

    def _validate_current_extension(self) -> bool:
        file_name: str = self.fileName or ""
        self.ftype = self.valid_extsInstance.getFileType(self.escapedFilePath)
        if self.valid_extsInstance.check(file_name, self.ftype):
            return True

        self.pretty.write(
            "I",
            "Ignored extensions",
            f"Ignored (invalid ext): {self.escapedFilePath} ({self.ftype})",
        )
        self.ignored_countInstance.increment()
        return False

    def _skip_unchanged_ingestion_doc(self) -> bool:
        if self.strategy.strategy_type != StrategyType.DOCUMENT_INGESTION:
            return False
        return self.docChanged() is False

    def _set_creation_timestamp(self) -> None:
        try:
            c_ts: float = os.path.getctime(self.escapedFilePath)
            self.creation_date = time.ctime(c_ts)
        except OSError:
            self.creation_date = ""

    def _extract_content_for_current_file(self) -> tuple[str, str]:
        file_name: str = self.fileName or ""
        content: str = ""
        disabled_office_component: str = ""

        if self.valid_extsInstance.check(file_name, ["pdf"]):
            logging.getLogger("pdfminer").setLevel(logging.ERROR)
            content = extractPDF(self.escapedFilePath).strip()
            if not content:
                for page in convert_from_path(self.escapedFilePath, dpi=300):
                    try:
                        content = (
                            str(pytesseract.image_to_string(page)) + "\n"  # type: ignore[reportUnknownMemberType]
                        )
                    finally:
                        page.close()

        # _CONSIDER_AS_TEXT_FILE: plain-text formats whose
        # content is read as-is (txt, md, py, csv, log, …).
        elif self.valid_extsInstance.check(file_name, self.consider_as_text_file):
            with open(self.escapedFilePath, "r", encoding="utf-8") as f:
                content = f.read()

        elif self.valid_extsInstance.check(file_name, ["doc", "docx"]):
            if (
                self.office_doc_extraction.get("Word")
                and OfficeDocConverter.is_windows_supported()
            ):
                _, doc_obj = self.office_convInstance.convert_office_file(
                    self.escapedFilePath
                )
                content = "\n".join(p.text for p in doc_obj.paragraphs)
                content += "\n".join(
                    cell.text
                    for tbl in doc_obj.tables
                    for row in tbl.rows
                    for cell in row.cells
                )
            else:
                disabled_office_component = "MS Word"

        elif self.valid_extsInstance.check(file_name, ["ppt", "pptx"]):
            if (
                self.office_doc_extraction.get("Power Point")
                and OfficeDocConverter.is_windows_supported()
            ):
                _, pres = self.office_convInstance.convert_office_file(
                    self.escapedFilePath
                )
                for slide in pres.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            content += shape.text + "\n"
            else:
                disabled_office_component = "MS Power Point"

        elif self.valid_extsInstance.check(file_name, ["xls", "xlsx"]):
            if (
                self.office_doc_extraction.get("Excel")
                and OfficeDocConverter.is_windows_supported()
            ):
                _, wb = self.office_convInstance.convert_office_file(
                    self.escapedFilePath
                )
                df = self.worksheet_to_dataframe(wb.active)
                content = str(df.to_string(index=False))  # type: ignore[reportUnknownMemberType]
            else:
                disabled_office_component = "MS Excel"

        elif self.valid_extsInstance.check(
            file_name,
            ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
        ):
            img = Image.open(self.escapedFilePath)
            if file_name.lower().endswith(".webp"):
                self.pretty.write("I", "Conversion", "Converting WebP → RGB for OCR")
                img = img.convert("RGB")
            content = str(pytesseract.image_to_string(img))  # type: ignore[reportUnknownMemberType]

        return content, disabled_office_component

    def _handle_extraction_error(self, error: Exception, is_office: bool) -> None:
        if is_office:
            self.pretty.write(
                "W",
                "Office conversion",
                f"Office conversion failed (is MS Office installed?): {self.escapedFilePath}: {error}",
                color=ORANGE,
            )
        else:
            self.pretty.write(
                "W",
                "Extraction fail",
                f"Extraction failed: {self.escapedFilePath}: {error}",
            )
        self.failed_countInstance.increment()
        self.globalsInstance.add_failed_doc(
            {"FilePath": self.escapedFilePath, "error": str(error)}
        )

    def _handle_empty_content(self, disabled_office_component: str) -> bool:
        if disabled_office_component:
            self.pretty.write(
                "W",
                "Office component",
                f"File {self.escapedFilePath} was not processed because running on Unix or office component {disabled_office_component} is disabled in _OFFICE_DOC_EXTRACTION (Configuration/Config_Global.py)",
                color=ORANGE,
            )
        else:
            self.pretty.write(
                "I",
                "Empty file",
                f"File {self.escapedFilePath} has no content and is ignored",
            )
        return False

    def _normalize_and_route_current_doc(self, content: str) -> None:
        self.content = self.unicode_normalizer.normalize(
            content,
            preserve_newlines=self._chunker_preserves_newlines(),
        )
        self.content = self.masker.mask(self.content)
        self.doc = self._make_doc()
        self.strategy.process(self.doc)
        self.processed_countInstance.increment()

    def _extract_and_process_current_file(self) -> bool:
        self.pretty.write("I", "Extract text", "Extracting text from Document")
        try:
            content, disabled_office_component = (
                self._extract_content_for_current_file()
            )
        except DataProcessingError as error:
            self._handle_extraction_error(error, is_office=True)
            return False
        except Exception as error:
            self._handle_extraction_error(error, is_office=False)
            return False

        if content == "":
            return self._handle_empty_content(disabled_office_component)

        self._normalize_and_route_current_doc(content)
        return True

    def process_files(self) -> None:
        """
        Walk self.doc_dir, extract text from each file, then either
        chunk or classify.
        """
        # Performance logging for batch processing
        self.perf_logger.log(
            "LoadAndClassifyProcessor.process_files",
            "pipeline",
            f"start batch processing dir={self.doc_dir}",
        )
        _t0_batch = time.perf_counter()

        for root, _, files in os.walk(self.doc_dir):
            for file_name in files:
                self._prepare_current_file(root, file_name)

                if self._skip_by_path_filters():
                    continue

                self.pretty_always.write(
                    "I",
                    "START",
                    f"{BRIGHT_BLUE}Processing: {self.escapedFilePath}{RESET}",
                )

                if not self._validate_current_extension():
                    continue

                if self._skip_unchanged_ingestion_doc():
                    continue

                self._set_creation_timestamp()
                if not self._extract_and_process_current_file():
                    continue

        # Log batch completion
        elapsed_batch = time.perf_counter() - _t0_batch
        self.perf_logger.log(
            "LoadAndClassifyProcessor.process_files",
            "pipeline",
            f"stop  batch processing elapsed={elapsed_batch:.3f}s",
        )
