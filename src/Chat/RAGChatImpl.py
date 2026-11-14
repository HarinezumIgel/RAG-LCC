# ── Local Module Imports ──
import ntpath
import os
import threading
# ── Standard Library Imports ──
from typing import Any, Sequence, Tuple

from chromadb.api import Collection  # type: ignore[attr-defined]
# ── LangChain Ecosystem ──
from langchain_chroma import Chroma
from langchain_core.documents import Document

from AI.ModelsCache import ModelsCache
from AI.TokenBudget import TokenBudget
from Chat.ChatContext import ChatContext
from Chat.ConfidenceEvaluator import ConfidenceEvaluator
from Chat.DocumentMarkingService import DocumentMarkingService
from Chat.FileSelector import FileSelector
from Chat.LocalRetrieverExecutor import LocalRetrieverExecutor
from Chat.PromptRewrite import PromptRewrite
from Chat.QueryNormalizer import QueryNormalizer
from Chat.RetrievalContextBuilder import RetrievalContextBuilder
from Chat.RetrievalGate import RetrievalGate
from Chat.RetrievalReranker import RetrievalReranker
from Commons.Exceptions import CollectionNotFoundError
from Commons.SingletonMixin import SingletonMixin
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.ConfidenceLogger import ConfidenceLogger
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import (Helpers, align_retriever_sources_for_print,
                             truncate_for_print)
from Helpers.PerfLogger import PerfLogger
from Orchestrator.Orchestrator import Orchestrator
from Retrievers.BM25Retriever import BM25Retriever
from Retrievers.GraphRetriever import GraphRetriever
from Retrievers.RegexRetriever import RegexRetriever
from Retrievers.RetrieverProtocol import RetrieverProtocol
from Retrievers.WebPreFilter import WebPreFilter
from Retrievers.WebRetriever import WebRetriever
from Retrievers.WebSearchFilter import WebSearchFilter

_CHAT_COMMANDS: tuple[str, ...] = (
    "new topic:",
    "new:",
    "newtopic:",
    "summarize:",
    "summary:",
)


class RAGChatImpl(
    SingletonMixin,
    RetrievalReranker,
    QueryNormalizer,
    LocalRetrieverExecutor,
    DocumentMarkingService,
    RetrievalContextBuilder,
):
    """Retrieval facade owning shared model, store, and turn dependencies."""

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

        # Instantiate helper objects/singletons as instance attributes.
        # Cache for stopwords per language.
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.confidence_logger: ConfidenceLogger = ConfidenceLogger()
        self.confidenceEvaluator: ConfidenceEvaluator = ConfidenceEvaluator(
            confidence_logger=self.confidence_logger,
            pretty=self.pretty,
        )
        self.helperInstance: Helpers = helpers or Helpers()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.chatContext: ChatContext = ChatContext()
        self.promptRewrite: PromptRewrite = PromptRewrite()
        self.retrievalGate: RetrievalGate = RetrievalGate()
        self.models_cache: ModelsCache = ModelsCache()
        # Initialize the embeddings using Ollama.
        self.device: Any
        self.device, _, _, _ = self.models_cache.switch2device()
        self.embed_model_name: str = self.helperInstance.get_model_args(
            "_ACTIVE_EMBED"
        )["MODEL"]
        _cross_args: dict[str, Any] = self.helperInstance.get_model_args(
            "_ACTIVE_CROSS"
        )
        self.cross_encoder_model_name: str = _cross_args["MODEL"]
        # Optional query prefix for rerankers that document one.
        # Empty string means no prefix (bge-reranker-v2-m3 needs none).
        self.cross_encoder_query_instruction: str = _cross_args.get(
            "QUERY_INSTRUCTION", ""
        )
        self.x_encoder: Any = self.models_cache.get_cross_encoder()
        self.embedder: Any = self.models_cache.get_hf_embeddings()
        self.tokenBudget: TokenBudget = TokenBudget()
        self.bm25_retriever: BM25Retriever = BM25Retriever()
        self.graph_retriever: GraphRetriever = GraphRetriever()
        self.regex_retriever: RegexRetriever = RegexRetriever()
        self.web_retriever: WebRetriever = WebRetriever()
        self.perf_logger: PerfLogger = PerfLogger()
        self.web_pre_filter: WebPreFilter = WebPreFilter(
            cfg=self.cfg, embedder=self.embedder, pretty=self.pretty
        )
        # Intent filter singleton — shared with WebRetriever (already built there).
        # get_instance() is a no-op if WebRetriever.__init__ ran first.
        from Configuration.Config_WebSearch import WEB_SEARCH_INTENT_EXTENSIONS

        intent_log = self.cfg.get("_INTENT_FILTER_LOG")
        intent_log_path: str = (
            os.path.join("logs", "RAGChat", "intent_filter.log")
            if intent_log is None
            else str(intent_log)
        )
        self.intent_filter: WebSearchFilter = WebSearchFilter.get_instance(
            extensions_cfg=WEB_SEARCH_INTENT_EXTENSIONS,
            log_path=intent_log_path,
        )
        self._shared: SharedHelpers = SharedHelpers()
        self._fileUtils: FileUtils = FileUtils()
        # Resolve the translation backend for user-query normalisation.
        # Allowed values: "argos" | "off".
        cfg_backend: str = (
            (self.cfg.get_str("_QUERY_REWRITE.TRANSLATION_BACKEND") or "off")
            .strip()
            .lower()
        )
        if cfg_backend not in ("argos", "off"):
            cfg_backend = "off"
        self._translation_backend: str = cfg_backend
        self.persist_directory: str | None = None
        self.vector_store: Chroma | None = None
        self.collection: Collection | None = None
        self._lock = threading.Lock()
        self.orchestrator: Orchestrator = Orchestrator(self)

    def set_vector_store(self, mySession: Session) -> bool:
        """Thread-safe wrapper — acquires ``self._lock`` then delegates."""
        with self._lock:
            return self._set_vector_store(mySession)

    def _set_vector_store(self, mySession: Session) -> bool:
        self.collection_name, self.persist_directory = (
            self.chromaDBHelper.change_chroma_collection(
                mySession.collection_name, True
            )
        )
        if self.collection_name and self.persist_directory:
            self.pretty.write(
                "I",
                "VectorStore",
                f"Set Chroma vector store. Name: {self.collection_name} Path: {self.persist_directory}",
            )

        if not os.path.exists(self.persist_directory):
            msg = (
                f"Collection '{self.collection_name}' not found at {self.persist_directory}. "
                f"Create a new collection running RAGLoad.py --collection MyCollection "
                f"or provide an existing collection running RAGChat.py --collection existingCollection"
            )
            self.pretty.write("E", "Collection", msg, color=RED)
            raise CollectionNotFoundError(msg)

        local_retrievers: list[tuple[str, RetrieverProtocol]] = [
            ("BM25", self.bm25_retriever),
            ("Graph", self.graph_retriever),
            ("Regex", self.regex_retriever),
        ]
        for label, retriever in local_retrievers:
            index_dir = retriever.get_index_dir(self.collection_name)
            index_filename = str(getattr(retriever, "INDEX_FILENAME", "")).strip()
            if not index_filename:
                msg = (
                    f"{label} retriever does not expose INDEX_FILENAME. "
                    f"Cannot validate persisted index for collection '{self.collection_name}'."
                )
                self.pretty.write("E", f"{label} index", msg, color=RED)
                raise CollectionNotFoundError(msg)
            index_path = os.path.join(index_dir, index_filename)
            if not os.path.isfile(index_path):
                msg = (
                    f"{label} index for collection '{self.collection_name}' not found at "
                    f"{index_path}. "
                    f"Re-run RAGLoad with RETRIEVAL_STORES_KEEP = False to rebuild "
                    f"the collection and all its retrieval indexes."
                )
                self.pretty.write("E", f"{label} index", msg, color=RED)
                raise CollectionNotFoundError(msg)

        # Load Chroma client and collection from persisted directory
        self.client, self.collection = (
            self.chromaDBHelper.get_chroma_client_and_collection(
                self.persist_directory, self.collection_name
            )
        )
        # Initialize vector store with cosine similarity metric
        self.vector_store = Chroma(
            embedding_function=self.embedder,
            client=self.client,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return True

    def _print_chroma_debug(self, docs: Sequence[Any]) -> None:
        """
        Prints a table of chroma stats for each doc when DEBUG is enabled.
        docs: a sequence of objects with doc.metadata containing:
        - 'position'
        - 'chroma_score' (float)
        - 'chroma_sim'   (float)
        - optional 'dist' (float); defaults to chroma_sim
        - 'FileName'     (str)
        """
        # if not self.cfg.get("DEBUG_LEVEL"):
        #    return

        # column formats: add position column first
        header = "{:>6}  {:>12}  {:>9}  {:>8}  {:>24}   {}"
        row = "{:>6}  {:>12.4f}  {:>9.4f}  {:>8.4f}  {:>24}   {}"

        # print header + separator once
        self.pretty.write(
            "D",
            "Chroma",
            header.format(
                "Pos", "ChromaScore", "ChromaSim", "Distance", "Retrievers", "File"
            ),
            color=CYAN,
        )
        self.pretty.write("D", "Chroma", "-" * 90, color=CYAN)

        # print one row per doc
        for i, doc in enumerate(docs, start=1):
            md: dict[str, Any] = doc.metadata
            score: Any = md.get("chroma_score", 0.0)
            sim: Any = md.get("chroma_sim", 0.0)
            dist: Any = md.get("dist", sim)
            sources: str = align_retriever_sources_for_print(
                str(md.get("retriever_sources", "")), 24
            )
            fn: Any = md.get("FileName", "<unknown>")

            self.pretty.write(
                "D", "Chroma", row.format(i, score, sim, dist, sources, fn)
            )

    def retrieve(self, mySession: Session) -> Tuple[str, int]:
        """Thread-safe wrapper — acquires ``self._lock`` then delegates."""
        with self._lock:
            return self._retrieve(mySession)

    def _retrieve(self, mySession: Session) -> Tuple[str, int]:
        return self.orchestrator.run(mySession)

    @staticmethod
    def parse_chat_command(query: str) -> tuple[str, str]:
        query = query.strip()
        for prefix in _CHAT_COMMANDS:
            if query.lower().startswith(prefix):
                command = "summarize" if prefix in ("summarize:", "summary:") else "new"
                return command, query[len(prefix) :].strip()
        return "", query

    @staticmethod
    def summary_selection_warning(session: Session, target: str = "") -> str | None:
        if target.strip() or (
            (session.file_name or session.file_path)
            and session.file_name != "Chat Context"
            and session.file_path != "Chat Context"
        ):
            return None
        if getattr(session, "is_service", False):
            return (
                "Summarize: requires an indexed document. Enter summarize: followed "
                "by a filename or full path, or set file or path in OpenWebUI "
                "Advanced Parameters and enter summarize:. Alternatively ask a "
                "question first; summarize: then offers indexed documents cited "
                "in the answer when grounded sources are available."
            )
        return (
            "Summarize: requires a file to be selected. Use path! in session "
            "settings to select the document, then enter summarize:.\n"
            "If no file can be selected, enter a file name contained in the "
            "collection or issue a query which fills the picker"
        )

    def retrieve_summary_documents(
        self, session: Session, target: str = ""
    ) -> list[Document]:
        """Fetch all indexed chunks, restricted only by file identity."""
        warning = self.summary_selection_warning(session, target)
        if warning is not None:
            raise ValueError(warning)
        file_name, file_path = session.file_name, session.file_path
        if target:
            target = target.strip()
            if len(target) >= 2 and target[0] == target[-1] and target[0] in "\"'":
                target = target[1:-1]
            target = target.strip()
            file_name = ntpath.basename(target)
            file_path = target if ntpath.dirname(target) else None
        elif file_path:
            file_name = file_name or ntpath.basename(file_path)

        if not file_name or file_path == "Chat Context":
            raise ValueError(
                "Summarize: provide a valid file name or full document path."
            )
        file_filter: dict[str, Any] = {"FileName": {"$eq": file_name}}
        if file_path:
            file_filter["FilePath"] = {"$eq": file_path}

        self.pretty.write(
            "I",
            "Summarize",
            f"Selecting indexed file by {'FileName + FilePath' if file_path else 'FileName'}: "
            f"{file_path or file_name}",
        )
        normalize_path_match = False
        with self._lock:
            if self.collection is None or (
                session.collection_name
                and session.collection_name != self.collection_name
            ):
                if not self._set_vector_store(session):
                    raise ValueError(
                        "Summarize: could not load the selected indexed collection."
                    )
            if self.collection is None:
                raise ValueError("Summarize: no indexed collection is loaded.")
            # A metadata get without a limit is truly unbounded: no top-k,
            # similarity threshold, language/metadata filter, or reranker.
            result = self.collection.get(
                where=self._chroma_where(file_filter),
                include=["documents", "metadatas"],
            )
            if file_path and not result["documents"]:
                # Chroma equality is literal; ingestion may store POSIX slashes.
                result = self.collection.get(
                    where={"FileName": {"$eq": file_name}},
                    include=["documents", "metadatas"],
                )
                normalize_path_match = True

        texts = result["documents"]
        metadatas = result["metadatas"]
        if not texts:
            raise ValueError(
                f"Summarize: no indexed chunks found for {target or file_path or file_name}."
            )
        if metadatas is None or len(metadatas) != len(texts):
            raise ValueError("Summarize: indexed chunks have missing metadata.")
        documents = [
            Document(page_content=text, metadata=dict(meta or {}))
            for text, meta in zip(texts, metadatas)
        ]
        if normalize_path_match and file_path:
            requested_path = FileSelector.path_identity(file_path)
            documents = [
                doc
                for doc in documents
                if FileSelector.path_identity(str(doc.metadata.get("FilePath", "")))
                == requested_path
            ]
            if not documents:
                raise ValueError(f"Summarize: no indexed chunks found for {file_path}.")
            self.pretty.write(
                "I",
                "Summarize",
                "Matched indexed path after normalizing path separators.",
            )
        paths = {
            FileSelector.path_identity(str(doc.metadata.get("FilePath", "")))
            for doc in documents
        }
        if len(paths) > 1:
            guidance = (
                "Set path in OpenWebUI Advanced Parameters or provide the full "
                "document path after summarize:."
                if getattr(session, "is_service", False)
                else "Use path! or provide the full document path."
            )
            raise ValueError(
                f"Summarize: multiple files named {file_name!r}. " f"{guidance}"
            )
        # Chroma get preserves insertion order; stable page ordering also
        # supports page-based chunkers, including several chunks per page.
        documents.sort(key=lambda doc: int(doc.metadata.get("PageNumber", 0)))
        return documents

    def indexed_file_metadata(
        self, session: Session, file_names: list[str]
    ) -> list[dict[str, Any]]:
        """Return metadata for requested names from the active indexed collection.

        Collection access stays here so collection switching and the shared
        Chroma client remain protected by the RAGChatImpl lock. Selection and
        path de-duplication are handled by FileSelector.
        """
        names = [name for name in dict.fromkeys(file_names) if name]
        if not names:
            return []
        where: dict[str, Any] = (
            {"FileName": {"$eq": names[0]}}
            if len(names) == 1
            else {"FileName": {"$in": names}}
        )
        with self._lock:
            if self.collection is None or (
                session.collection_name
                and session.collection_name != self.collection_name
            ):
                if not self._set_vector_store(session):
                    raise ValueError(
                        "Summarize: could not load the selected indexed collection."
                    )
            if self.collection is None:
                raise ValueError("Summarize: no indexed collection is loaded.")
            result = self.collection.get(where=where, include=["metadatas"])
        return [dict(meta or {}) for meta in result.get("metadatas") or []]

    def _prepare_session(self, mySession: Session) -> str:
        """Reset per-turn flags, handle mode-change and topic-switch resets.

        Returns user_query_original (the query before any translation/rewrite).
        """
        mySession.reset_turn_state()

        self.confidence_logger.reset_turn(mySession)
        self._fileUtils.reset_lang_detection_events()

        if (
            mySession.last_web_search is not None
            and mySession.last_web_search != mySession.web_search
            and mySession.use_chat_context
        ):
            self.chatContext.reset_conversation()
            self.pretty.write(
                "W",
                "TopicSwitch",
                "Web search mode changed — chat history cleared to prevent context contamination.",
            )

        if (
            mySession.last_fetch_page_content is not None
            and mySession.last_fetch_page_content != mySession.fetch_page_content
            and mySession.use_chat_context
        ):
            self.chatContext.reset_conversation()
            self.pretty.write(
                "W",
                "TopicSwitch",
                "fetch_page_content mode changed — chat history cleared to prevent context contamination.",
            )

        input_query = getattr(mySession, "user_query_input", None)
        display_query = (
            input_query if input_query is not None else (mySession.query or "")
        )
        command, argument = self.parse_chat_command(display_query)
        if command == "new":
            if input_query is None:
                mySession.query = argument
            mySession.force_skip_rewrite = True
            mySession.last_topic_referents = None
            self.chatContext.reset_conversation()
            self.pretty.write(
                "W",
                "TopicSwitch",
                "Topic switch detected — chat history cleared, query rewrite disabled for this turn.",
            )

        mySession.last_web_search = mySession.web_search
        mySession.last_fetch_page_content = mySession.fetch_page_content

        user_query_original: str = mySession.query or ""
        mySession.user_query_original = user_query_original
        self.pretty.write(
            "I",
            "UserQuery",
            f"Original user query: {display_query!r}",
            color=CYAN,
        )
        return user_query_original

    def _fetch_web_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Delegate web retrieval orchestration to Orchestrator."""
        return self.orchestrator.run_web_retriever(
            mySession,
            retrieve_mode,
            user_query_original,
        )

    def _print_bm25_debug(self, docs: list[Any]) -> None:
        """Print BM25 retrieval debug table."""
        self._print_single_score_retriever_debug(
            docs,
            channel="BM25",
            score_header="BM25Score",
            score_key="bm25_score",
            file_header="File",
            file_key="FileName",
        )

    def _print_graph_debug(self, docs: list[Any]) -> None:
        """Print graph retrieval debug table (shown at debug_level >= 30)."""
        self._print_single_score_retriever_debug(
            docs,
            channel="Graph",
            score_header="GraphScore",
            score_key="graph_score",
            file_header="File",
            file_key="FileName",
        )

    def _print_single_score_retriever_debug(
        self,
        docs: list[Any],
        *,
        channel: str,
        score_header: str,
        score_key: str,
        file_header: str,
        file_key: str,
    ) -> None:
        """Print a standard single-score retriever debug table."""
        header = "{:>6}  {:>12}  {:>24}   {}"
        row = "{:>6}  {:>12.4f}  {:>24}   {}"
        self.pretty.write(
            "D",
            channel,
            header.format("Pos", score_header, "Retrievers", file_header),
            color=CYAN,
        )
        self.pretty.write("D", channel, "-" * 71, color=CYAN)
        for i, doc in enumerate(docs[:20], start=1):
            score = doc.metadata.get(score_key, 0.0)
            sources = align_retriever_sources_for_print(
                str(doc.metadata.get("retriever_sources", "")), 24
            )
            file_value = doc.metadata.get(file_key, "<unknown>")
            self.pretty.write(
                "D",
                channel,
                row.format(i, score, sources, file_value),
            )

    def _print_regex_debug(self, docs: list[Any]) -> None:
        """Print regex retrieval debug table (shown at debug_level >= 30)."""
        header = "{:>6}  {:>12}  {:>8}  {:>8}  {:>24}   {}"
        row = "{:>6}  {:>12.4f}  {:>8}  {:>8}  {:>24}   {}"
        self.pretty.write(
            "D",
            "Regex",
            header.format(
                "Pos", "RegexScore", "VerbHit", "NounHit", "Retrievers", "File"
            ),
            color=CYAN,
        )
        self.pretty.write("D", "Regex", "-" * 94, color=CYAN)
        for i, doc in enumerate(docs[:20], start=1):
            score = doc.metadata.get("regex_score", 0.0)
            verb_hits = doc.metadata.get("regex_verb_hits", 0)
            noun_hits = doc.metadata.get("regex_noun_hits", 0)
            sources = align_retriever_sources_for_print(
                str(doc.metadata.get("retriever_sources", "")), 24
            )
            fn = doc.metadata.get("FileName", "<unknown>")
            self.pretty.write(
                "D",
                "Regex",
                row.format(i, score, verb_hits, noun_hits, sources, fn),
            )

    def _print_web_prefilter_debug(
        self,
        kept: list[Any],
        dropped: list[Any],
        filter_name: str,
        threshold: float,
    ) -> None:
        """Print a kept/dropped table after a web pre-filter step."""
        header = "{:>6}  {:>8}  {:>10}   {:40}  {}"
        row = "{:>6}  {:>8}  {:>10.4f}   {:40}  {}"
        tag = f"WebPF/{filter_name}"
        self.pretty.write(
            "D",
            tag,
            header.format("Status", "Pos", "Score", "URL", "Snippet"),
            color=CYAN,
        )
        self.pretty.write("D", tag, "-" * 100, color=CYAN)
        for i, doc in enumerate(kept, start=1):
            score = float(doc.metadata.get("chroma_score") or 0.0)  # type: ignore[arg-type]
            url = str(doc.metadata.get("FilePath", "") or "")[:40]
            snip = str(doc.metadata.get("snippet") or doc.page_content or "")[:60]
            self.pretty.write("D", tag, row.format("KEPT", i, score, url, snip))
        for i, doc in enumerate(dropped, start=1):
            score = float(doc.metadata.get("chroma_score") or 0.0)  # type: ignore[arg-type]
            url = str(doc.metadata.get("FilePath", "") or "")[:40]
            snip = str(doc.metadata.get("snippet") or doc.page_content or "")[:60]
            self.pretty.write("D", tag, row.format("DROP", i, score, url, snip))

    def _print_web_debug(self, docs: list[Any]) -> None:
        """Print web retrieval debug table (shown at debug_level >= 10)."""
        self._print_single_score_retriever_debug(
            docs,
            channel="Web",
            score_header="WebScore",
            score_key="chroma_score",
            file_header="URL",
            file_key="FilePath",
        )

    def _print_merged_debug(self, docs: list[Any]) -> None:
        """Print RRF-merged results debug table with retriever origin and filename.

        Column widths mirror the Rerank debug table so Pos / Retrievers / File
        stay visually aligned across both outputs:
          Pos(6)  RRFScore(10)  [blank AdjScore](10)  Retrievers(24)  File(30)
        """
        header = "{:>6}  {:>10}  {:>10}  {:>24}  {:<30}"
        row = "{:>6}  {:>10.4f}  {:>10}  {:>24}  {:<30}"
        self.pretty.write(
            "D",
            "Merge",
            header.format("Pos", "RRFScore", "", "Retrievers", "File"),
            color=CYAN,
        )
        self.pretty.write("D", "Merge", "-" * 82, color=CYAN)
        for i, doc in enumerate(docs, start=1):
            score = doc.metadata.get("rrf_score", 0.0)
            sources = align_retriever_sources_for_print(
                str(doc.metadata.get("retriever_sources", "")), 24
            )
            fn = doc.metadata.get("FileName", "<unknown>")
            self.pretty.write("D", "Merge", row.format(i, score, "", sources, fn))

    # ── Multi-Query Expansion ──────────────────────────────────────────────────

    # ── Chunk Near-Duplicate Removal ───────────────────────────────────────────
