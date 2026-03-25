# Local module imports
# Standard library imports
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from chromadb.api import Collection  # type: ignore[attr-defined]
# langchain and related libraries
from langchain_core.documents.base import Document as langchainDoc
# from langchain.docstore.document import Document  # or use langchain.schema.Document depending on your version
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AI.AIHelpers import AIHelpers
from AI.ModelsCache import ModelsCache
from Commons.SingletonMixin import SingletonMixin
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Compliance.Exclusions import Exclusions
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.CounterInstance import HumanReviewCount, ProcessedCount
from Globals.Globals import Globals
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Strategies.StrategyType import StrategyType

# Third-party imports
# Imaging, file detection & data processing libraries


class ChunksToDBStrategy(SingletonMixin):
    strategy_type = StrategyType.CHUNKS_TO_DB  # enum identifier

    def process(self, doc: Optional[Dict[str, Any]]) -> None:
        self.doc = doc
        self.docChunksToDBStrategy()

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

        # counters & utilities
        self.processedCounter: ProcessedCount = ProcessedCount()
        self.humanReviewCount: HumanReviewCount = HumanReviewCount()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.cfg: Config = cfg or Config()
        self.csvWriter: CSVWriter = CSVWriter()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.globalsInstance: Globals = Globals()
        self.bannedPhraseCollector: BannedPhraseCollector = BannedPhraseCollector()
        self.accumulator: Accumulator = Accumulator()
        self.exclusions: Exclusions = Exclusions()

        _coll_name, _ = self.chromaDBHelper.change_chroma_collection(
            self.cfg.get_str("COLLECTION"), False
        )

        # splitter settings
        self.separators: list[Any] = self.cfg.get_list("_SEPARATORS")
        chroma_slot: str = self.helpers.get_chroma_config_slot()
        self.chunk_size: int = self.cfg.get_int(f"{chroma_slot}.CHUNK_SIZE")
        self.overlap: int = self.cfg.get_int(f"{chroma_slot}.CHUNK_OVERLAP", 0)
        self.use_exclusions: bool = self.cfg.get_bool("USE_EXCLUSIONS")

        # embeddings model
        self.embedder: Any = self.models_cache.get_hf_embeddings()

        self.content: str | None = None
        self.collection: Collection | None = None

        # attributes populated per-document by process() / _reverse_doc()
        self.doc: dict[str, Any] | None = None
        self.file: str | None = None
        self.escapedFilePath: str | None = None
        self.creation_date: str | None = None
        self.fileType: str | None = None
        self.language: str | None = None
        self.wordCount: int = 0
        self.fileHash: str | None = None
        self.collection_name: str = ""
        self.client: Any = None

    def _reverse_doc(self) -> None:
        assert self.doc is not None, "doc must be set before _reverse_doc"  # noqa
        doc: dict[str, Any] = self.doc
        meta: dict[str, Any] = doc.get("meta", {})
        self.file: str | None = meta.get("FileName")
        self.escapedFilePath: str | None = meta.get("FilePath")
        self.creation_date: str | None = meta.get("CreationDate")
        self.fileType: str | None = meta.get("FileType")
        self.language: str | None = meta.get("Language")
        self.wordCount: int = 0
        self.fileHash: str | None = meta.get("FileHash")
        self.content = doc.get("content", "")

    def _make_doc(self) -> dict[str, Any]:
        self._reverse_doc()
        if self.content:
            self.wordCount = self.fileUtils.count_words(self.content)

        self.doc = {
            "meta": {
                "FileName": self.file,
                "FilePath": self.escapedFilePath,
                "CreationDate": self.creation_date,
                "FileType": self.fileType,
                "Language": self.language,
                "WordCount": self.wordCount,
                "FileHash": self.fileHash,
            },
            "content": self.content,
        }
        return self.doc

    def docChunksToDBStrategy(self):
        """
        High-level orchestration: open DB, prepare chunks, screen, upsert, finalize.
        """
        human_review: bool = False

        # 1) Open chroma and collection
        persist_dir: str
        self.collection_name, persist_dir = (
            self.chromaDBHelper.chroma_coll_name_and_mkdir_or_del("create")
        )
        self.client, self.collection = (
            self.chromaDBHelper.get_chroma_client_and_collection(
                persist_dir, self.collection_name, stamp=True
            )
        )

        # 2) Refresh in-memory doc
        self._make_doc()

        # --- unsupported-language gate ---
        lang: str = self.language or "en"
        lang_action: str | None = SharedHelpers().check_language_support(
            lang, self.escapedFilePath or "?"
        )
        if lang_action == "NOT_OK":
            assert self.doc is not None
            meta_ref: dict[str, Any] = self.doc.get("meta", {})
            meta_ref.update(
                {"Status": "NOT_OK", "Stage": "Language", "Time": datetime.now()}
            )
            self.csvWriter.write_json2csv(meta_ref, "NOT_OK")
            return
        if lang_action == "HUMAN_REVIEW":
            human_review = True
        lang_human_review: bool = human_review
        human_review_reason: str = (
            f"Unsupported language '{lang}'" if lang_action == "HUMAN_REVIEW" else ""
        )

        # 3) Prepare chunks
        doc_chunks: list[langchainDoc] = self._prepare_chunks()
        self.pretty.write(
            "I",
            "Chunks ingestion",
            f"Prepared {len(doc_chunks)} new chunk(s) for ingestion.",
        )

        # 4) Clear old chunks for this file
        self._clear_old_chunks()

        # 5) Preprocess texts and embed
        texts: list[str]
        metas: list[dict[str, Any]]
        texts, metas = self._extract_texts_and_metas(doc_chunks)  # type: ignore[reportUnknownMemberType]
        texts_trunc: list[str] = self.models_cache.truncate_texts(
            texts,
            model_name=self.helpers.get_model_args("_EMBED")["MODEL"],
            max_length=self.chunk_size,
            padding=True,
        )
        embeddings: list[list[float]] = self.embedder.embed_documents(texts_trunc)

        # 6) Filter by keyword and similarity checks
        #    Compliance checks run on the full (un-truncated) texts so that
        #    words near the end of a chunk are not missed due to tokenizer
        #    truncation.  Embeddings and upserted texts still use texts_trunc.
        (
            kept_ids,
            kept_embeddings,
            kept_metas,
            kept_texts,
            skipped,
            phrase_table,
            human_review,
        ) = self._filter_chunks(
            [d.id or str(uuid.uuid4()) for d in doc_chunks],
            embeddings,
            metas,
            texts_trunc,
            compliance_texts=texts,
        )
        human_review = human_review or lang_human_review

        # 7) Upsert safe chunks
        self._upsert_chunks(
            kept_ids, kept_embeddings, kept_metas, kept_texts, len(doc_chunks), skipped
        )

        # 8) Finalize and write CSVs
        self._finalize(
            skipped,
            len(kept_ids),
            phrase_table,
            human_review,
            self.wordCount,
            human_review_reason=human_review_reason,
        )

    # --- Helpers ---
    def _prepare_chunks(self) -> list[langchainDoc]:
        assert self.content is not None, "content must be set"
        assert self.doc is not None, "doc must be set"
        stop_words: list[str] = self.fileUtils.get_stopwords(self.content)
        if len(stop_words) > 0:
            cleaned: str | None = self.fileUtils.removeStopwords(
                self.content, set(stop_words)
            )
        else:
            cleaned = self.content
        separators_str: list[str] = [str(s) for s in self.separators]
        splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self.fileUtils.count_words,
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        assert self.doc is not None, "doc must be set"
        doc_ref: dict[str, Any] = self.doc
        base_doc: langchainDoc = langchainDoc(
            page_content=cleaned or "", metadata=doc_ref["meta"]
        )
        slices: list[langchainDoc] = splitter.split_documents([base_doc])

        ids: list[str] = [str(uuid.uuid4()) for _ in slices]
        chunks: list[langchainDoc] = []
        for i, s in enumerate(slices):
            meta = dict(doc_ref["meta"])
            meta["MyChunk"] = i
            chunks.append(
                langchainDoc(page_content=s.page_content, metadata=meta, id=ids[i])
            )
        return chunks

    def _clear_old_chunks(self):
        assert self.collection is not None, "collection must be set"
        self.collection.delete(where={"FilePath": self.escapedFilePath or ""})
        self.pretty.write(
            "I", "Vector store", f"Cleared old chunks for {self.escapedFilePath}"
        )

    def _extract_texts_and_metas(
        self, doc_chunks: list[langchainDoc]
    ) -> tuple[list[str], list[dict[str, str | int | float | bool]]]:
        texts: list[str] = [d.page_content for d in doc_chunks]
        metas: list[dict[str, str | int | float | bool]] = [self.chromaDBHelper.clean_metadata(cast(dict[str, Any], d.metadata)) for d in doc_chunks]  # type: ignore[reportUnknownMemberType]
        return texts, metas

    def _filter_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metas: list[dict[str, Any]],
        texts: list[str],
        compliance_texts: list[str] | None = None,
    ):
        stage: str = "PIPELINE_CHECK"
        kept_ids: list[str] = []
        kept_embeddings: list[list[float]] = []
        kept_metas: list[dict[str, Any]] = []
        kept_texts: list[str] = []
        skipped: int = 0
        done: int = 0
        human_review: bool = False
        check_texts: list[str] = (
            compliance_texts if compliance_texts is not None else texts
        )
        loop_length: int = min(len(ids), len(embeddings), len(metas), len(texts))
        for cid, emb, meta, text, check_text in zip(
            ids, embeddings, metas, texts, check_texts
        ):
            (
                human_review,
                _,
                _,
            ) = self.aiHelpers.run_ensemble_checks(
                check_text,
                self.language or "en",
                stage=stage,
                accumulate=True,
                require_keybert=False,
                embedding=emb,
            )

            if human_review:
                skipped += 1
                continue

            kept_ids.append(cid)
            kept_embeddings.append(emb)
            kept_metas.append(meta)
            kept_texts.append(text)
            done += 1
            self.helpers.show_progress(done, loop_length)
        self.helpers.show_progress(done, loop_length, print_newline=True)
        human_review, phrase_table = self.accumulator.show_accumulated(stage)

        return (
            kept_ids,
            kept_embeddings,
            kept_metas,
            kept_texts,
            skipped,
            phrase_table,
            human_review,
        )

    def _upsert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metas: list[dict[str, str | int | float | bool]],
        texts: list[str],
        total_chunks: int,
        skipped: int,
    ) -> None:
        if ids:
            assert self.collection is not None, "collection must be set"
            self.collection.add(
                ids=ids, embeddings=embeddings, metadatas=metas, documents=texts  # type: ignore[reportArgumentType]
            )
            self.pretty.write(
                "O",
                "Vector store",
                f"Upserted {len(ids)} cleaned chunk(s) to {self.collection_name}. Skipped {skipped} due to banned similarity.",
            )
        else:
            self.pretty.write(
                "W",
                "Vector store",
                f"No chunks inserted  to {self.collection_name}. All {total_chunks} chunk(s) were skipped.",
            )

    def _finalize(
        self,
        skipped: int,
        inserted: int,
        phrase_table: List[dict[str, Any]],
        human_review: bool,
        word_count: int = 0,
        human_review_reason: str = "",
    ):

        if skipped == inserted or inserted == 0:
            skipStatus = "ALL"
        elif skipped == 0:
            skipStatus = "NONE"
        else:
            skipStatus = "PARTIAL"

        assert self.doc is not None, "doc must be set"  # noqa
        doc_ref: dict[str, Any] = self.doc
        meta_ref: dict[str, Any] = doc_ref.get("meta", {})

        meta_ref.update(
            {
                "Status": "OK",
                "Skip Status": skipStatus,
                "Skipped Chunks": skipped,
                "Inserted Chunks": inserted,
                "WordCount": word_count,
                "Stage": "Summary",
                "Time": datetime.now(),
            }
        )
        if skipStatus != "ALL":
            self.csvWriter.write_json2csv(meta_ref, "OK")
        else:
            meta_ref.update(
                {
                    "Status": "NOT_OK",
                }
            )
            self.csvWriter.write_json2csv(meta_ref, "NOT_OK")

        if human_review:
            self.humanReviewCount.increment()
            hr_data: list[dict[str, Any]] | dict[str, Any]
            if phrase_table:
                hr_data = self.bannedPhraseCollector.prepare_for_csv_print(
                    phrase_table, meta_ref
                )
            else:
                hr_data = dict(meta_ref)
            if human_review_reason:
                if isinstance(hr_data, dict):
                    hr_data["Reason"] = human_review_reason
                    hr_data["Stage"] = "Language"
                elif hr_data:
                    hr_data[0]["Reason"] = human_review_reason
                    hr_data[0]["Stage"] = "Language"
            self.csvWriter.write_json2csv(hr_data, "HUMAN_REVIEW")
            if self.use_exclusions:
                self.exclusions.add(self.escapedFilePath or "")
