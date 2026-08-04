# Local module imports
# Standard library imports
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from chromadb.api import Collection  # type: ignore[attr-defined]
# langchain and related libraries
from langchain_core.documents.base import Document as langchainDoc

from AI.AIHelpers import AIHelpers
from AI.ModelsCache import ModelsCache
from Commons.SingletonMixin import SingletonMixin
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Compliance.Exclusions import Exclusions
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.CounterInstance import HumanReviewCount, ProcessedCount
from Globals.Globals import Globals
from Gui.Colors import CYAN, RESET
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.BM25Retriever import BM25Retriever
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy
from Strategies.Chunkers.DocumentMetadataExtractor import \
    DocumentMetadataExtractor
from Strategies.Chunkers.HeadingChunker import HeadingChunker
from Strategies.Chunkers.PdfPageChunker import PdfPageChunker
from Strategies.Chunkers.RecursiveChunker import RecursiveChunker
from Strategies.Chunkers.SemanticChunker import SemanticChunker
from Strategies.Chunkers.SentenceWindowChunker import SentenceWindowChunker
from Strategies.Chunkers.SlideChunker import SlideChunker
from Strategies.Chunkers.SlidingWindowChunker import SlidingWindowChunker
from Strategies.GraphRetriever import GraphRetriever
from Strategies.StrategyType import StrategyType

# Third-party imports
# Imaging, file detection & data processing libraries


class DocumentIngestionStrategy(SingletonMixin):
    strategy_type = StrategyType.DOCUMENT_INGESTION  # enum identifier

    def process(self, doc: Optional[Dict[str, Any]]) -> None:
        self.doc = doc
        self.ingest()

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
        self.metadataExtractor: DocumentMetadataExtractor = DocumentMetadataExtractor()
        self.cfg: Config = cfg or Config()
        self.csvWriter: CSVWriter = CSVWriter()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.globalsInstance: Globals = Globals()
        self.bannedPhraseCollector: BannedPhraseCollector = BannedPhraseCollector()
        self.accumulator: Accumulator = Accumulator()
        self.exclusions: Exclusions = Exclusions()
        self.bm25_retriever: BM25Retriever = BM25Retriever()
        self.graph_retriever: GraphRetriever = GraphRetriever()
        self.perf_logger: PerfLogger = PerfLogger()

        _coll_name, _ = self.chromaDBHelper.change_chroma_collection(
            self.cfg.get_str("COLLECTION"), False
        )

        # chunker — two-level strategy routing via _CHUNK_STRATEGY
        active_profile: str = self.cfg.get_str("_ACTIVE_CHUNKER_CONFIG")
        self._chunk_map: dict[str, str] = self.cfg.get_dict(
            f"_CHUNK_STRATEGY.{active_profile}"
        )
        self.chunker: ChunkerStrategy = self._make_chunker(
            self._chunk_map.get("DEFAULT", "SEMANTIC")
        )
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

    @staticmethod
    def _make_chunker(name: str) -> "ChunkerStrategy":
        """Instantiate a chunker by its config key name."""
        if name == "SEMANTIC":
            return SemanticChunker(chunker_name=name)
        if name == "SENTENCE_WINDOW":
            return SentenceWindowChunker(chunker_name=name)
        if name == "SLIDING_WINDOW":
            return SlidingWindowChunker(chunker_name=name)
        if name == "HEADING":
            return HeadingChunker(chunker_name=name)
        if name == "SLIDE":
            return SlideChunker(chunker_name=name)
        if name == "PDF_PAGE":
            return PdfPageChunker(chunker_name=name)
        return RecursiveChunker(chunker_name=name)

    def _resolve_chunker_for_file(self) -> None:
        """Swap ``self.chunker`` to the strategy mapped to the current
        document's file type in the active ``_CHUNK_STRATEGY`` profile."""
        ft: str = (self.fileType or "").lower()
        chunker_name: str = self._chunk_map.get(
            ft, self._chunk_map.get("DEFAULT", "SEMANTIC")
        )
        self.chunker = self._make_chunker(chunker_name)

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

        meta: dict[str, Any] = {
            "FileName": self.file,
            "FilePath": self.escapedFilePath,
            "CreationDate": self.creation_date,
            "FileType": self.fileType,
            "Language": self.language,
            "WordCount": self.wordCount,
            "FileHash": self.fileHash,
        }
        # Attach configurable extra metadata (e.g. author/dates) to every
        # chunk — the chunker copies this dict into each chunk's metadata.
        extra: dict[str, Any] = self.metadataExtractor.extract(
            self.escapedFilePath or "", self.fileType or ""
        )
        for key, value in extra.items():
            meta.setdefault(key, value)

        if extra and self.cfg.get_bool("SHOW_EXTRACTED_METADATA", False, silent=True):
            details = "  ".join(f"{k}={v}" for k, v in extra.items())
            self.pretty.write("I", "Metadata", f"{self.file}: {details}", color=MAGENTA)

        self.doc = {
            "meta": meta,
            "content": self.content,
        }
        return self.doc

    def ingest(self):
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

        # 3) Pick the chunker for this file type
        self._resolve_chunker_for_file()

        chunker_label = type(self.chunker).__name__
        self.pretty.write(
            "I",
            "Chunker",
            f"{CYAN}{chunker_label} defined as chunker for extension: {self.fileType}{RESET}",
        )

        # 4) Prepare chunks
        #    chunk() returns (docs, pre_embeddings).  SemanticChunker provides
        #    weighted-average sentence embeddings for most chunks (saving a
        #    full second embedding pass).  Other chunkers return None.
        assert self.content is not None and self.doc is not None
        doc_chunks: list[langchainDoc]
        pre_embeddings: list[list[float] | None] | None
        self.perf_logger.log(
            "DocumentIngestionStrategy.ingest",
            "ingestion",
            f"start chunking chunker={chunker_label}",
        )
        _t_chunk = time.perf_counter()
        doc_chunks, pre_embeddings = self.chunker.chunk(
            self.content, self.doc.get("meta", {})
        )
        self.perf_logger.log(
            "DocumentIngestionStrategy.ingest",
            "ingestion",
            f"stop  chunking chunker={chunker_label} n={len(doc_chunks)} elapsed={time.perf_counter() - _t_chunk:.3f}s",
        )
        self.pretty.write(
            "I",
            "Chunks ingestion",
            f"Prepared {len(doc_chunks)} new chunk(s) for ingestion.",
        )

        # 5) Clear old chunks for this file
        self._clear_old_chunks()

        # 6) Preprocess texts and embed
        #    If the chunker already provided embeddings we reuse them and
        #    only call the embedding model for entries marked None (e.g.
        #    oversized splits that couldn't reuse sentence vectors).
        texts: list[str]
        metas: list[dict[str, Any]]
        texts, metas = self._extract_texts_and_metas(doc_chunks)  # type: ignore[reportUnknownMemberType]
        texts_trunc: list[str] = self.models_cache.truncate_texts(
            texts,
            model_name=self.helpers.get_model_args("_ACTIVE_EMBED")["MODEL"],
            max_length=self.chunker.chunk_size,
            padding=True,
        )
        embeddings: list[list[float]] = self._resolve_embeddings(
            texts_trunc, pre_embeddings
        )

        # 7) Filter by keyword and similarity checks
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

        # 8) Upsert safe chunks
        self._upsert_chunks(
            kept_ids, kept_embeddings, kept_metas, kept_texts, len(doc_chunks), skipped
        )

        # 8b) Update BM25 index incrementally (remove old + add new)
        self.bm25_retriever.ingest_file(
            self.escapedFilePath or "",
            self.collection_name,
            self.collection,
            kept_ids,
            kept_texts,
            kept_metas,
        )

        # 8c) Update graph index incrementally (remove old + add new)
        self.graph_retriever.ingest_file(
            self.escapedFilePath or "",
            self.collection_name,
            self.collection,
            kept_ids,
            kept_texts,
            kept_metas,
        )

        # 9) Finalize and write CSVs
        self._finalize(
            skipped,
            len(kept_ids),
            phrase_table,
            human_review,
            self.wordCount,
        )

    # --- Helpers ---
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

    def _resolve_embeddings(
        self,
        texts_trunc: list[str],
        pre_embeddings: list[list[float] | None] | None,
    ) -> list[list[float]]:
        """Build the final embedding list, re-using pre-computed vectors.

        * If *pre_embeddings* is ``None`` (non-semantic chunkers) we embed
          every chunk from scratch — same behaviour as before.
        * If *pre_embeddings* is a list, entries that are not ``None`` are
          kept as-is.  Only the ``None`` entries (e.g. oversized splits)
          are sent to the embedding model, saving a potentially expensive
          second full-batch call.
        """
        self.perf_logger.log(
            "DocumentIngestionStrategy._resolve_embeddings",
            "ingestion",
            f"start embed n={len(texts_trunc)}",
        )
        _t_emb = time.perf_counter()
        if pre_embeddings is None:
            # No pre-computed embeddings — embed everything
            result_emb = self.embedder.embed_documents(texts_trunc)
            self.perf_logger.log(
                "DocumentIngestionStrategy._resolve_embeddings",
                "ingestion",
                f"stop  embed (full) n={len(result_emb)} elapsed={time.perf_counter() - _t_emb:.3f}s",
            )
            return result_emb

        # Identify which indices still need embedding
        need_idx: list[int] = [i for i, emb in enumerate(pre_embeddings) if emb is None]

        if not need_idx:
            # All chunks already have embeddings — skip model entirely
            self.pretty.write(
                "I",
                "Embeddings",
                "Reusing all pre-computed chunk embeddings (0 to embed).",
            )
            self.perf_logger.log(
                "DocumentIngestionStrategy._resolve_embeddings",
                "ingestion",
                f"stop  embed (all cached) n={len(pre_embeddings)} elapsed={time.perf_counter() - _t_emb:.3f}s",
            )
            return [emb for emb in pre_embeddings]  # type: ignore[misc]

        # Embed only the missing chunks
        texts_to_embed: list[str] = [texts_trunc[i] for i in need_idx]
        new_embs: list[list[float]] = self.embedder.embed_documents(texts_to_embed)

        self.pretty.write(
            "I",
            "Embeddings",
            f"Reused {len(texts_trunc) - len(need_idx)} pre-computed embeddings, "
            f"embedded {len(need_idx)} remaining chunk(s).",
        )

        # Merge: fill in the None slots with freshly computed embeddings
        result: list[list[float]] = list(pre_embeddings)  # type: ignore[arg-type]
        for idx, emb in zip(need_idx, new_embs):
            result[idx] = emb

        self.perf_logger.log(
            "DocumentIngestionStrategy._resolve_embeddings",
            "ingestion",
            f"stop  embed (partial) reused={len(texts_trunc) - len(need_idx)} embedded={len(need_idx)} elapsed={time.perf_counter() - _t_emb:.3f}s",
        )
        return result

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
            self.helpers.show_progress(done, loop_length, label="Embeddings")
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
        file_name = metas[0].get("FileName") if metas else None
        file_name = file_name or self.file or "<unknown>"
        if ids:
            assert self.collection is not None, "collection must be set"
            self.collection.add(
                ids=ids, embeddings=embeddings, metadatas=metas, documents=texts  # type: ignore[reportArgumentType]
            )
            self.pretty.write(
                "O",
                "Vector store",
                f"Upserted {len(ids)} cleaned chunk(s) for {file_name} to collection {self.collection_name}. Skipped {skipped} due to banned similarity.",
            )
        else:
            self.pretty.write(
                "W",
                "Vector store",
                f"No chunks inserted for {file_name} to collection {self.collection_name}. All {total_chunks} chunk(s) were skipped.",
            )

    def _finalize(
        self,
        skipped: int,
        inserted: int,
        phrase_table: List[dict[str, Any]],
        human_review: bool,
        word_count: int = 0,
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
            meta_ref["Status"] = "NOT_OK"
            hr_data: list[dict[str, Any]] | dict[str, Any]
            if phrase_table:
                hr_data = self.bannedPhraseCollector.prepare_for_csv_print(
                    phrase_table, meta_ref
                )
            else:
                hr_data = dict(meta_ref)
            self.csvWriter.write_json2csv(hr_data, "HUMAN_REVIEW")
            if self.use_exclusions:
                self.exclusions.add(self.escapedFilePath or "")
