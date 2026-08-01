# ── Local Module Imports ──
import os
import threading
import time
# ── Standard Library Imports ──
from typing import Any, Sequence, Tuple, cast

from chromadb.api import Collection  # type: ignore[attr-defined]
# ── LangChain Ecosystem ──
from langchain_chroma import Chroma
from langdetect import detect  # type: ignore[import-untyped]  # noqa: F401

from AI.ModelsCache import ModelsCache
from AI.TokenBudget import TokenBudget
from Chat.ChatContext import ChatContext
from Chat.PromptRewrite import PromptRewrite
from Chat.RetrievalGate import RetrievalGate
from Commons.Exceptions import CollectionNotFoundError, RerankError
from Commons.SingletonMixin import SingletonMixin
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers, truncate_for_print
from Helpers.PerfLogger import PerfLogger
from Strategies.BM25Retriever import BM25Retriever
from Strategies.GraphRetriever import GraphRetriever
from Strategies.HomeBrewChunkSelector import ChunkSelectionService
from Strategies.WebPreFilter import WebPreFilter
from Strategies.WebRetriever import WebRetriever
from Strategies.WebSearchFilter import WebSearchFilter

# ── Third-Party Libraries ──

# Prefixes the user can type at the start of their query to signal a deliberate
# topic change.  The prefix is stripped before translation/retrieval and the
# rewriter LLM is skipped for that turn only.
_TOPIC_SWITCH_PREFIXES: tuple[str, ...] = ("new topic:", "new:", "newtopic:")


class RAGChatImpl(SingletonMixin):
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
        # Optional instruction prefix for instruction-tuned cross-encoders
        # (e.g. BAAI/bge-reranker-v2-m3).  Empty string means no prefix.
        self.cross_encoder_query_instruction: str = _cross_args.get(
            "QUERY_INSTRUCTION", ""
        )
        self.x_encoder: Any = self.models_cache.get_cross_encoder()
        self.embedder: Any = self.models_cache.get_hf_embeddings()
        self.tokenBudget: TokenBudget = TokenBudget()
        self.bm25_retriever: BM25Retriever = BM25Retriever()
        self.graph_retriever: GraphRetriever = GraphRetriever()
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
        # Allowed values: "argos" | "m2m100" | "off".
        cfg_backend: str = (
            (self.cfg.get_str("_QUERY_REWRITE.TRANSLATION_BACKEND") or "off")
            .strip()
            .lower()
        )
        if cfg_backend not in ("argos", "m2m100", "off"):
            cfg_backend = "off"
        self._translation_backend: str = cfg_backend
        self.persist_directory: str | None = None
        self.vector_store: Chroma | None = None
        self.collection: Collection | None = None
        self._lock = threading.Lock()

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

        # Validate BM25 index exists for this collection
        bm25_dir = self.bm25_retriever.get_bm25_dir(self.collection_name)
        bm25_index_path = os.path.join(bm25_dir, BM25Retriever.INDEX_FILENAME)
        if not os.path.isfile(bm25_index_path):
            msg = (
                f"BM25 index for collection '{self.collection_name}' not found at "
                f"{bm25_index_path}. "
                f"Re-run RAGLoad with RETRIEVAL_STORES_KEEP = False to rebuild "
                f"the collection and all its retrieval indexes."
            )
            self.pretty.write("E", "BM25 index", msg, color=RED)
            raise CollectionNotFoundError(msg)

        # Validate graph index exists for this collection
        graph_dir = self.graph_retriever.get_graph_dir(self.collection_name)
        graph_index_path = os.path.join(graph_dir, GraphRetriever.INDEX_FILENAME)
        if not os.path.isfile(graph_index_path):
            msg = (
                f"Graph index for collection '{self.collection_name}' not found at "
                f"{graph_index_path}. "
                f"Re-run RAGLoad with RETRIEVAL_STORES_KEEP = False to rebuild "
                f"the collection and all its retrieval indexes."
            )
            self.pretty.write("E", "Graph index", msg, color=RED)
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

    def _rerank(self, mySession: Session, all_docs: list[Any]) -> list[Any]:
        self.perf_logger.log(
            "RAGChatImpl._rerank",
            "chat",
            f"start rerank pairs={len(all_docs)} model={self.cross_encoder_model_name!r}",
        )
        _t_rerank = time.perf_counter()
        # Prepare query-document pairs for re-ranking.
        # For web documents, use the original search-engine snippet rather than
        # doc.page_content for the cross-encoder input.  When fetch_page_content
        # is active, page_content is the full fetched page — often thousands of
        # tokens — and the cross-encoder's 512-token truncation window is filled
        # by navigation headers and boilerplate before the relevant content is
        # reached.  The snippet (stored in metadata["snippet"]) is the concise
        # search-engine description that scores reliably for relevance.
        # The full page_content is untouched and still goes to the LLM prompt.
        candidates: list[Any] = all_docs
        pairs: list[tuple[str, str]] = []
        # sentence_transformers CrossEncoder always expects (str_A, str_B) pairs;
        # the tokenizer handles [CLS] query [SEP] passage [SEP] formatting.
        # Instruction-tuned models (e.g. BAAI/bge-reranker-v2-m3) may need a
        # task-prefix on the query — set QUERY_INSTRUCTION in the _CROSS model
        # config entry.  Standard BERT cross-encoders leave it empty.
        query_text = (
            self.cross_encoder_query_instruction + mySession.query
            if self.cross_encoder_query_instruction and mySession.query
            else mySession.query or ""
        )
        for doc in candidates:
            chunk_text = (
                doc.metadata["snippet"]
                if doc.metadata.get("Source") == "Web" and doc.metadata.get("snippet")
                else doc.page_content
            )
            pairs.append((query_text, chunk_text))

        if not pairs:
            self.pretty.write(
                "W",
                "Rerank",
                f"Reranking with {self.cross_encoder_model_name} returned {len(pairs)} chunks",
            )
            return pairs

        # Get re-ranking scores from cross-encoder model
        try:
            raw_rerank = self.x_encoder.predict(pairs, show_progress_bar=False)
        except RuntimeError as e:
            self.pretty.write(
                "E",
                "Rerank",
                f"Reranking failed due to a model input mismatch. "
                f"Did you run Load.py with a different _CHROMA_EMBED_PARAMS.CHUNK_SIZE than RAGChat.py is using now? "
                f"Details: {e}",
                color=RED,
            )
            raise RerankError

        # Credits: Fix according to input from Don Karter (u/donk8r on Reddit).
        # Before fix: We only hire people if they are among the best applicants in today's candidate pool.
        # So an identical candidate may pass on Tuesday and fail on Wednesday.
        # Fix: We hire anyone scoring above 80. Then rank hired candidates afterward.
        #
        # Score normalization — plain min-max over the unified pool (local + web).
        #
        #   rerank_score = (raw − pool_lo) / (pool_hi − pool_lo)
        #
        # rerank_score is used ONLY for intra-query ordering.  It is deliberately
        # query-relative: the best chunk in any pool approaches 1.0, so it must
        # not be used for absolute keep/drop decisions.
        #
        # Threshold decisions (keep/drop) use raw_rerank_score — the raw
        # cross-encoder logit stored below — which is on a query-independent scale.
        # See HomeBrewChunkSelector.filter_threshold.
        #
        # Web docs are multiplied by web_wt for ordering so local knowledge
        # retains a natural edge.  In web-only mode the multiplier is skipped
        # (it would uniformly lower all scores without changing their order).

        # Resolve the web weight from the session override or the global config default.
        web_wt: float = float(
            mySession.web_weight
            if mySession.web_weight is not None
            else (self.cfg.get_float("_WEB_SEARCH.default_web_weight") or 0.5)
        )

        # Store raw cross-encoder logits; compute pool min/max for ordering.
        for i, doc in enumerate(candidates):
            doc.metadata["raw_rerank_score"] = float(raw_rerank[i])

        all_raw_scores = [float(raw_rerank[i]) for i in range(len(candidates))]
        lo = min(all_raw_scores) if all_raw_scores else 0.0
        hi = max(all_raw_scores) if all_raw_scores else 1.0
        full_range = hi - lo if hi != lo else 1.0

        # Assign rerank_score: plain min-max for relative ordering only.
        # Web docs are additionally scaled by web_wt to let local knowledge retain
        # a natural edge when both sources are present.  In web-only mode (all docs
        # are web) the multiplier is skipped — there is no local content to protect
        # and applying it would only uniformly lower all scores without changing order.
        all_web: bool = bool(candidates) and all(
            d.metadata.get("Source") == "Web" for d in candidates
        )
        for i, doc in enumerate(candidates):
            raw = float(raw_rerank[i])
            normalized = (raw - lo) / full_range
            if doc.metadata.get("Source") == "Web" and not all_web:
                doc.metadata["rerank_score"] = web_wt * normalized
            else:
                doc.metadata["rerank_score"] = normalized

        # Sort documents by combined score in descending order
        reranked: list[Any] = sorted(
            candidates, key=lambda d: d.metadata["rerank_score"], reverse=True  # type: ignore[reportUnknownLambdaType, reportUnknownMemberType]
        )

        # debug print
        if DebugHelper.check_session(mySession, 10):
            header = "{:>6}  {:>10}  {:>10}  {:>17}  {:<40}  {}"
            row = "{:>6}  {:>10.4f}  {:>10.4f}  {:>17}  {:<40}  {}"

            # Print header row — RawScore is the cross-encoder output; AdjScore is
            # the pool-normalized value (web docs also scaled by web_weight).
            self.pretty.write(
                "D",
                "Rerank",
                header.format(
                    "Pos", "RawScore", "AdjScore", "Retrievers", "File", "Text"
                ),
                color=CYAN,
            )
            self.pretty.write("D", "Rerank", "-" * 110, color=CYAN)

            # Print each row aligned with header
            for i, d in enumerate(reranked[: mySession.final_chunks_to_llm]):
                file_name: str = truncate_for_print(
                    str(d.metadata.get("FileName", d.metadata.get("source", ""))), 40
                )
                sources: str = str(d.metadata.get("retriever_sources", ""))
                # Show the text actually fed to the cross-encoder: snippet for
                # web docs (prefixed [S]), page_content for all others.
                scoring_text: str = (
                    "[S] " + d.metadata["snippet"]
                    if d.metadata.get("Source") == "Web" and d.metadata.get("snippet")
                    else d.page_content
                )
                self.pretty.write(
                    "D",
                    "Rerank",
                    row.format(
                        i + 1,
                        d.metadata.get("raw_rerank_score", 0.0),
                        d.metadata["rerank_score"],
                        sources,
                        file_name,
                        scoring_text[:40],  # truncate text to fit
                    ),
                )

        self.pretty.write(
            "I",
            "Rerank",
            f"Reranking with {self.cross_encoder_model_name} returned {len(reranked)} chunks",
            color=CYAN,
        )
        self.perf_logger.log(
            "RAGChatImpl._rerank",
            "chat",
            f"stop  rerank n={len(reranked)} elapsed={time.perf_counter() - _t_rerank:.3f}s",
        )
        return reranked

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
        header = "{:>6}  {:>12}  {:>9}  {:>8}  {:>17}   {}"
        row = "{:>6}  {:>12.4f}  {:>9.4f}  {:>8.4f}  {:>17}   {}"

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
            sources: str = str(md.get("retriever_sources", ""))
            fn: Any = md.get("FileName", "<unknown>")

            self.pretty.write(
                "D", "Chroma", row.format(i, score, sim, dist, sources, fn)
            )

    def _get_translator(self, backend: str) -> Any:
        """Return the translator for *backend* (``"argos"``, ``"m2m100"``, ``"off"``).

        Unknown or ``"off"`` values return ``None`` so callers can skip
        translation with a simple ``if translator is not None`` guard.
        """
        b = (backend or "off").lower()
        if b == "argos":
            return self._shared
        if b == "m2m100":
            from Compliance.HfTranslator import HfTranslator

            return HfTranslator()
        return None

    def retrieve(self, mySession: Session) -> Tuple[str, int]:
        """Thread-safe wrapper — acquires ``self._lock`` then delegates."""
        with self._lock:
            return self._retrieve(mySession)

    def _retrieve(self, mySession: Session) -> Tuple[str, int]:
        if not self._set_vector_store(mySession):
            return "", 0
        self.perf_logger.log("RAGChatImpl._retrieve", "chat", "start retrieve")
        user_query_original = self._prepare_session(mySession)
        final_query, alternate_queries = self._normalize_query(
            mySession, user_query_original
        )
        if self._check_gates(mySession, final_query):
            return "", 0
        retrieve_mode: str = (mySession.retrieve_mode or "VECTOR").upper()
        bm25_query: str = mySession.query or ""
        vector_docs, bm25_docs, graph_docs = self._fetch_local_docs(
            mySession, retrieve_mode, bm25_query, alternate_queries
        )
        web_docs = self._fetch_web_docs(mySession, retrieve_mode, user_query_original)
        chosen = self._merge_and_select(
            mySession, vector_docs, bm25_docs, graph_docs, web_docs
        )
        return self._build_context(mySession, chosen)

    def _prepare_session(self, mySession: Session) -> str:
        """Reset per-turn flags, handle mode-change and topic-switch resets.

        Returns user_query_original (the query before any translation/rewrite).
        """
        mySession.force_skip_rewrite = False
        mySession.effective_query = None
        mySession.effective_query_reason = None

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

        raw_query: str = mySession.query or ""
        for pfx in _TOPIC_SWITCH_PREFIXES:
            if raw_query.lower().startswith(pfx):
                mySession.query = raw_query[len(pfx) :].strip()
                mySession.force_skip_rewrite = True
                self.chatContext.reset_conversation()
                self.pretty.write(
                    "W",
                    "TopicSwitch",
                    f"Topic switch detected (prefix {pfx!r}) — chat history cleared, query rewrite disabled for this turn.",
                )
                break

        mySession.last_web_search = mySession.web_search
        mySession.last_fetch_page_content = mySession.fetch_page_content

        user_query_original: str = mySession.query or ""
        self.pretty.write(
            "I",
            "UserQuery",
            f"Original user query: {user_query_original!r}",
            color=CYAN,
        )
        return user_query_original

    def _normalize_query(
        self, mySession: Session, user_query_original: str
    ) -> tuple[str, list[str]]:
        """Translate → rewrite → re-translate; set effective_query; expand alternate queries.

        Returns (final_query, alternate_queries).
        """
        backend: str = (
            getattr(mySession, "translation_backend", None)
            or self._translation_backend
            or "off"
        ).lower()
        translator = self._get_translator(backend)
        was_translated: bool = False

        if translator is not None and mySession.query:
            original_query: str = mySession.query
            detected_lang: str = self._fileUtils.get_user_text_language(
                original_query,
                output="nltk",
                native_lang=None,
            )
            # Tag the session so ChatContext can filter turns by language.
            mySession.current_query_lang = detected_lang

            if detected_lang != "english":
                translated: str = translator.translate_text(
                    original_query,
                    target_lang="en",
                    source_lang=detected_lang,
                )
                if translated and translated != original_query:
                    self.pretty.write(
                        "I",
                        "QueryNorm",
                        f"Normalized user query [{detected_lang}\u2192english, "
                        f"backend={backend}]: "
                        f"{original_query!r} \u2192 {translated!r}",
                    )
                    mySession.query = translated
                    was_translated = True

        pre_rewrite_query: str = mySession.query or ""
        if mySession.use_chat_context:
            mySession.query = self.promptRewrite.rewrite(mySession)
        was_rewritten: bool = (mySession.query or "") != pre_rewrite_query

        # Post-rewrite re-normalisation: rewriter may introduce non-English names.
        if translator is not None and mySession.query:
            rewritten_query: str = mySession.query
            detected_after_rewrite: str = self._fileUtils.get_user_text_language(
                rewritten_query,
                output="nltk",
                native_lang=None,
            )
            if detected_after_rewrite != "english":
                translated_after: str = translator.translate_text(
                    rewritten_query,
                    target_lang="en",
                    source_lang=detected_after_rewrite,
                )
                if translated_after and translated_after != rewritten_query:
                    self.pretty.write(
                        "I",
                        "QueryNorm",
                        f"Normalized rewritten query "
                        f"[{detected_after_rewrite}\u2192english, "
                        f"backend={backend}]: "
                        f"{rewritten_query!r} \u2192 {translated_after!r}",
                    )
                    mySession.query = translated_after
                    was_translated = True

        final_query: str = mySession.query or ""
        if final_query != user_query_original:
            mySession.effective_query = final_query
            if was_translated and was_rewritten:
                mySession.effective_query_reason = "translated+rewritten"
            elif was_translated:
                mySession.effective_query_reason = "translated"
            elif was_rewritten:
                mySession.effective_query_reason = "rewritten"
            else:
                mySession.effective_query_reason = "changed"
            self.pretty.write(
                "I",
                "FinalQuery",
                f"Final query for retrieval: {final_query!r} "
                f"(was: {user_query_original!r})",
                color=CYAN,
            )
        else:
            self.pretty.write(
                "I",
                "FinalQuery",
                f"Final query for retrieval: {final_query!r} (unchanged)",
                color=CYAN,
            )

        alternate_queries: list[str] = self._generate_alternate_queries(
            final_query, mySession
        )
        if alternate_queries and DebugHelper.check_session(mySession, 29):
            self.pretty.write(
                "D",
                "MultiQuery",
                f"Alternate queries ({len(alternate_queries)}): "
                + " | ".join(f"{i+1}: {q!r}" for i, q in enumerate(alternate_queries)),
                color=CYAN,
            )

        return final_query, alternate_queries

    def _check_gates(self, mySession: Session, final_query: str) -> bool:
        """Run the retrieval eligibility and intent classifier gates.

        Returns True if retrieval should be aborted (caller returns "", 0).
        """
        if self.retrievalGate.check(mySession):
            return True

        score, outcome, reasons = self.intent_filter.score_query(
            final_query, path="local"
        )
        if outcome == "REFUSE":
            reason_str = ", ".join(reasons) if reasons else "intent score"
            mySession.clarification_response = (
                f"Your query was blocked by the content policy "
                f"(intent score {score} — {reason_str})."
            )
            if DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "W",
                    "IntentFilter",
                    f"Query blocked \u2014 score={score}, reasons={reasons}",
                )
            return True
        if outcome == "ALLOW_WITH_SAFETY_FRAMING":
            if DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "W",
                    "IntentFilter",
                    f"Dual-use query detected \u2014 score={score}, reasons={reasons}. "
                    "Proceeding with retrieval.",
                )
        return False

    def _fetch_local_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Run Vector (+ alternate-query expansion), BM25, and Graph retrievers.

        Returns (vector_docs, bm25_docs, graph_docs).
        """
        VECTOR_MODES = ("VECTOR", "ALL", "VECTOR_GRAPH", "VECTOR_BM25")
        BM25_MODES = ("BM25", "ALL", "BM25_GRAPH", "VECTOR_BM25")
        GRAPH_MODES = ("GRAPH", "ALL", "VECTOR_GRAPH", "BM25_GRAPH")

        # --- VECTOR retrieval ---
        vector_docs: list[Any] = []
        vector_weight = float(
            mySession.vector_weight if mySession.vector_weight is not None else 1.0
        )
        if retrieve_mode in VECTOR_MODES and vector_weight != 0.0:
            self.pretty.write(
                "I",
                "Chroma",
                f"Querying Chroma DB on vector store {self.persist_directory}",
            )
            assert (
                self.vector_store is not None
            ), "vector_store not initialized; call set_vector_store first"
            self.perf_logger.log(
                "RAGChatImpl._retrieve", "chat", "start vector similarity_search"
            )
            _t_vec = time.perf_counter()
            hits: list[Any] = self.vector_store.similarity_search_with_score(
                mySession.query or "", **(mySession.base_kwargs or {})
            )
            vector_docs = self.chatContext.annotate_chunks(hits)
            self.perf_logger.log(
                "RAGChatImpl._retrieve",
                "chat",
                f"stop  vector similarity_search n={len(vector_docs)} elapsed={time.perf_counter() - _t_vec:.3f}s",
            )
            for d in vector_docs:
                d.metadata["retriever_sources"] = "Vector"

            if alternate_queries:
                existing_ids: set[str] = {
                    str(d.metadata.get("id", d.page_content)) for d in vector_docs
                }
                for qi, aq in enumerate(alternate_queries, start=1):
                    try:
                        aq_hits: list[Any] = (
                            self.vector_store.similarity_search_with_score(
                                aq, **(mySession.base_kwargs or {})
                            )
                        )
                        aq_docs = self.chatContext.annotate_chunks(aq_hits)
                        for d in aq_docs:
                            doc_id = str(d.metadata.get("id", d.page_content))
                            if doc_id not in existing_ids:
                                d.metadata["retriever_sources"] = f"Vector-AQ{qi}"
                                vector_docs.append(d)
                                existing_ids.add(doc_id)
                    except Exception as aq_exc:
                        self.pretty.write(
                            "W",
                            "MultiQuery",
                            f"Alternate query {qi} vector search failed: {aq_exc}",
                        )
            if DebugHelper.check_session(mySession, 10):
                self._print_chroma_debug(vector_docs)
            self.pretty.write(
                "O",
                "Chroma",
                f"Querying Chroma DB query returned {len(vector_docs)} chunks",
            )

        # --- BM25 retrieval ---
        bm25_docs: list[Any] = []
        bm25_weight = float(
            mySession.bm25_weight if mySession.bm25_weight is not None else 1.0
        )
        if retrieve_mode in BM25_MODES and bm25_weight != 0.0:
            self.pretty.write(
                "I",
                "BM25",
                f"Querying BM25 index on collection {self.collection_name}",
            )
            assert self.collection is not None
            assert self.persist_directory is not None
            self.perf_logger.log(
                "RAGChatImpl._retrieve",
                "chat",
                f"start bm25 block collection={self.collection_name}",
            )
            _t_bm25 = time.perf_counter()
            bm25_dir = self.bm25_retriever.get_bm25_dir(self.collection_name)
            self.bm25_retriever.load_or_rebuild(
                bm25_dir,
                self.collection_name,
                self.collection,
            )
            bm25_filter: dict[str, Any] | None = None
            if mySession.base_kwargs and "filter" in mySession.base_kwargs:
                bm25_filter = mySession.base_kwargs["filter"]

            bm25_docs = self.bm25_retriever.query(
                bm25_query,
                k=mySession.retriever_k or 100,
                file_filter=bm25_filter,
            )
            for d in bm25_docs:
                d.metadata["retriever_sources"] = "BM25"
            self.pretty.write(
                "O",
                "BM25",
                f"BM25 retrieval returned {len(bm25_docs)} chunks",
            )
            self.perf_logger.log(
                "RAGChatImpl._retrieve",
                "chat",
                f"stop  bm25 block n={len(bm25_docs)} elapsed={time.perf_counter() - _t_bm25:.3f}s",
            )
            if DebugHelper.check_session(mySession, 10):
                self._print_bm25_debug(bm25_docs)

        # --- Graph retrieval ---
        graph_docs: list[Any] = []
        graph_weight = float(
            mySession.graph_weight if mySession.graph_weight is not None else 1.0
        )
        if retrieve_mode in GRAPH_MODES and graph_weight != 0.0:
            self.pretty.write(
                "I",
                "Graph",
                f"Querying graph index on collection {self.collection_name}",
            )
            assert self.collection is not None
            self.perf_logger.log(
                "RAGChatImpl._retrieve",
                "chat",
                f"start graph block collection={self.collection_name}",
            )
            _t_graph = time.perf_counter()
            graph_dir = self.graph_retriever.get_graph_dir(self.collection_name)
            self.graph_retriever.load_or_rebuild(
                graph_dir,
                self.collection_name,
                self.collection,
            )
            graph_filter: dict[str, Any] | None = None
            if mySession.base_kwargs and "filter" in mySession.base_kwargs:
                graph_filter = mySession.base_kwargs["filter"]

            graph_docs = self.graph_retriever.query(
                bm25_query,
                k=mySession.retriever_k or 100,
                file_filter=graph_filter,
            )
            for d in graph_docs:
                d.metadata["retriever_sources"] = "Graph"
            self.pretty.write(
                "O",
                "Graph",
                f"Graph retrieval returned {len(graph_docs)} chunks",
            )
            self.perf_logger.log(
                "RAGChatImpl._retrieve",
                "chat",
                f"stop  graph block n={len(graph_docs)} elapsed={time.perf_counter() - _t_graph:.3f}s",
            )
            if DebugHelper.check_session(mySession, 30):
                self._print_graph_debug(graph_docs)

        return vector_docs, bm25_docs, graph_docs

    def _fetch_web_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Run web retrieval and apply BM25/cosine pre-filters.

        Returns web_docs (empty list when web search is disabled or blocked).
        """
        web_docs: list[Any] = []
        web_triggered: bool = mySession.web_search or retrieve_mode == "WEB"
        if not web_triggered:
            return web_docs

        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        if web_mode == "0":
            if retrieve_mode == "WEB":
                self.pretty.write(
                    "W",
                    "Web",
                    'retrieve_mode=WEB requested but WEB_SEARCH_MODE="0" '
                    "— no results will be returned.",
                )
            else:
                self.pretty.write(
                    "W",
                    "Web",
                    'Web search blocked by administrator (WEB_SEARCH_MODE = "0") — skipping',
                )
            return web_docs
        if web_mode != "1":
            self.pretty.write(
                "W",
                "Web",
                f"Web search blocked (WEB_SEARCH_MODE = {web_mode!r}) — skipping",
            )
            return web_docs

        label = (
            "retrieve_mode=WEB — querying web only..."
            if retrieve_mode == "WEB"
            else "Querying web search..."
        )
        self.pretty.write("I", "Web", label)
        self.perf_logger.log("RAGChatImpl._retrieve", "chat", "start web block")
        _t_web = time.perf_counter()
        web_docs = self.web_retriever.query(
            mySession.query or "",
            k=self.cfg.get_int("_WEB_SEARCH.max_results") or 5,
            fetch_page_content=bool(mySession.fetch_page_content),
            original_query=user_query_original,
            collection=mySession.collection_name or "",
        )
        self.pretty.write(
            "O",
            "Web",
            f"Web search returned {len(web_docs)} results",
            color=CYAN,
        )
        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  web block n={len(web_docs)} elapsed={time.perf_counter() - _t_web:.3f}s",
        )
        if DebugHelper.check_session(mySession, 10):
            self._print_web_debug(web_docs)

        # --- Web pre-filters (BM25 and/or cosine against query) ---
        # Each filter is a no-op when its threshold is 0.0 (default).
        # BM25 runs first (cheap), cosine second (embedding calls).
        pre_bm25: float = self.cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        pre_cosine: float = self.cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        if web_docs and (pre_bm25 > 0.0 or pre_cosine > 0.0):
            if DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "D",
                    "WebPreFilter",
                    f"Pre-filtering {len(web_docs)} web result(s) — "
                    f"bm25_pre_filter={pre_bm25:.3f}, "
                    f"cosine_pre_filter={pre_cosine:.3f}",
                    color=CYAN,
                )
            before_pre = len(web_docs)
            if pre_bm25 > 0.0:
                docs_before_bm25 = web_docs
                web_docs = self.web_pre_filter.bm25_prefilter(
                    web_docs, mySession.query or ""
                )
                if DebugHelper.check_session(mySession, 30):
                    self.pretty.write(
                        "D",
                        "WebPreFilter",
                        f"After BM25 pre-filter: {len(web_docs)}/{before_pre} kept",
                        color=CYAN,
                    )
                    kept_set = set(id(d) for d in web_docs)
                    dropped_bm25 = [
                        d for d in docs_before_bm25 if id(d) not in kept_set
                    ]
                    self._print_web_prefilter_debug(
                        web_docs, dropped_bm25, "BM25", pre_bm25
                    )
            if pre_cosine > 0.0 and web_docs:
                pre_query_vec: list[float] = self.embedder.embed_query(
                    mySession.query or ""
                )
                docs_before_cosine = web_docs
                web_docs = self.web_pre_filter.cosine_prefilter(
                    web_docs,
                    mySession.query or "",
                    query_vec=pre_query_vec,
                )
                if DebugHelper.check_session(mySession, 30):
                    self.pretty.write(
                        "D",
                        "WebPreFilter",
                        f"After cosine pre-filter: {len(web_docs)}/{before_pre} kept",
                        color=CYAN,
                    )
                    kept_set = set(id(d) for d in web_docs)
                    dropped_cosine = [
                        d for d in docs_before_cosine if id(d) not in kept_set
                    ]
                    self._print_web_prefilter_debug(
                        web_docs, dropped_cosine, "Cosine", pre_cosine
                    )

        return web_docs

    def _merge_and_select(
        self,
        mySession: Session,
        vector_docs: list[Any],
        bm25_docs: list[Any],
        graph_docs: list[Any],
        web_docs: list[Any],
    ) -> list[Any]:
        """RRF-fuse local docs, cap, append web docs, dedup, rerank, and select.

        Returns the final chosen list.
        """
        # Local retrievers (Vector, BM25, Graph) are fused via RRF and then
        # capped to retriever_k.  Web docs are appended AFTER the cap so they
        # always reach the reranker.  Including web in the same RRF pool caused
        # them to be pushed off the list: with weight=0.5 and only 5 results
        # their best RRF score (≈0.008) falls below the 100th local slot (≈0.011).
        local_sources = [
            (vector_docs, "Vector"),
            (bm25_docs, "BM25"),
            (graph_docs, "Graph"),
        ]
        local_active_labeled = [(d, lbl) for d, lbl in local_sources if d]
        local_active = [d for d, _ in local_active_labeled]
        local_active_labels = [lbl for _, lbl in local_active_labeled]
        local_weight_map = {
            "Vector": float(
                mySession.vector_weight if mySession.vector_weight is not None else 1.0
            ),
            "BM25": float(
                mySession.bm25_weight if mySession.bm25_weight is not None else 1.0
            ),
            "Graph": float(
                mySession.graph_weight if mySession.graph_weight is not None else 1.0
            ),
        }
        local_weights = [local_weight_map.get(lbl, 1.0) for lbl in local_active_labels]
        if len(local_active) > 1:
            capped_docs = BM25Retriever.reciprocal_rank_fusion(
                *local_active,
                k=self.bm25_retriever.rrf_k,
                labels=local_active_labels,
                weights=local_weights,
            )
            capped_docs = capped_docs[: mySession.retriever_k]
            self.pretty.write(
                "O",
                "Merge",
                f"Reciprocal Rank Fusion (RRF) produced {len(capped_docs)} local chunks",
            )
            if DebugHelper.check_session(mySession, 10):
                self._print_merged_debug(capped_docs)
        elif local_active:
            capped_docs = local_active[0][: mySession.retriever_k]
            # Stamp retriever_sources so rerank/selection debug prints show the origin.
            src = local_active_labels[0]
            for doc in capped_docs:
                doc.metadata["retriever_sources"] = src
        else:
            capped_docs = []

        if web_docs:
            capped_docs = capped_docs + web_docs
            self.pretty.write(
                "O",
                "Merge",
                f"Appended {len(web_docs)} web result(s) → reranker pool: {len(capped_docs)} chunks",
            )

        dedup_enabled: bool = self.cfg.get_bool("_CHUNK_DEDUP.enabled")
        dedup_threshold: float = self.cfg.get_float("_CHUNK_DEDUP.threshold") or 0.85
        if dedup_enabled and capped_docs:
            before_dedup = len(capped_docs)
            capped_docs = self._remove_similar_chunks(capped_docs, dedup_threshold)
            dropped = before_dedup - len(capped_docs)
            if dropped > 0 and DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "I",
                    "ChunkDedup",
                    f"Removed {dropped} near-duplicate chunk(s) "
                    f"(threshold={dedup_threshold:.2f}, kept {len(capped_docs)})",
                    color=CYAN,
                )

        if mySession.rerank == 1:
            capped_docs = self._rerank(mySession, capped_docs)

        chosen: list[Any] = cast(list[Any], ChunkSelectionService(mySession).select_chunks(capped_docs))  # type: ignore[reportUnknownMemberType]

        if DebugHelper.check_session(mySession, 30):
            self.pretty.write(
                "D",
                "ChunkSelect",
                f"After chunk selection: {len(chosen)}/{len(capped_docs)} kept",
                color=CYAN,
            )

        return chosen

    def _build_context(self, mySession: Session, chosen: list[Any]) -> Tuple[str, int]:
        """Populate session grounding fields and format the LLM context string.

        Returns (context, len(chosen)), or ("", 0) when chosen is empty.
        """
        mySession.chunk_texts_for_grounding = (
            [
                getattr(doc, "page_content", "") or ""
                for doc in chosen
                if str((getattr(doc, "metadata", {}) or {}).get("Source", "")).lower()
                != "web"
            ]
            if chosen
            else []
        )

        mySession.last_chosen_chunks = chosen

        # Pre-compute distinct FileNames/URLs so weak LLMs can't skip sources.
        seen_local: set[str] = set()
        distinct_local: list[str] = []
        seen_web: set[str] = set()
        distinct_web: list[str] = []
        for d in chosen:
            if d.metadata.get("Source") == "Web":
                fp = str(d.metadata.get("FilePath", "")).strip()
                if fp and fp not in seen_web:
                    seen_web.add(fp)
                    distinct_web.append(fp)
            else:
                fn = str(d.metadata.get("FileName", "")).strip()
                if fn and fn not in seen_local:
                    seen_local.add(fn)
                    distinct_local.append(fn)

        header_parts: list[str] = []
        if distinct_local:
            header_parts.append(
                f"LOCAL SOURCE FILES ({len(distinct_local)} files \u2014 you MUST consider every one of them and MUST include any that contain relevant information in your Sources section):\n"
                + "\n".join(f"  - {fn}" for fn in distinct_local)
            )
        if distinct_web:
            header_parts.append(
                f"WEB SOURCES ({len(distinct_web)} URLs \u2014 treat as supplementary internet context and MUST include any that contain relevant information in your Sources section):\n"
                + "\n".join(f"  - {url}" for url in distinct_web)
            )
        header: str = "\n\n".join(header_parts) + "\n\n" if header_parts else ""

        body: str = "\n\n".join(
            self.helperInstance.format_document(doc) for doc in chosen  # type: ignore[reportUnknownMemberType]
        )
        context: str = header + body
        if chosen:
            self.perf_logger.log(
                "RAGChatImpl._retrieve", "chat", f"stop  retrieve n={len(chosen)}"
            )
            return context, len(chosen)
        return "", 0

    def _mark_sources(self, mySession: Session, chosen: list[Any]) -> None:
        """Orchestrate in-memory visual marking of local source documents.

        Yellow highlights = retrieved chunk spans.
        Orange highlights = verbatim chunk fragments that grounded the answer.
        """
        self.pretty.write(
            "W", "VisualMarker", f"_mark_sources called: {len(chosen)} chunk(s)"
        )

        grouped = self._group_chunks_by_file(chosen)
        if not grouped:
            self.pretty.write(
                "W",
                "VisualMarker",
                "No local PDF paths resolved — nothing to highlight (check FilePath metadata above)",
            )
            mySession.marked_documents = []
            return

        highlight_color, answer_mark_color = self._resolve_mark_colors()
        grounded_snippets = self._build_grounded_snippets(
            mySession, chosen, answer_mark_color
        )
        produced = self._produce_marked_bytes(
            grouped, grounded_snippets, highlight_color
        )

        mySession.marked_documents = produced
        if produced:
            self.pretty.write(
                "I",
                "VisualMarker",
                f"Prepared {len(produced)} highlighted document(s) (in memory)",
                color=CYAN,
            )

    # ------------------------------------------------------------------
    # _mark_sources helpers
    # ------------------------------------------------------------------

    def _group_chunks_by_file(self, chosen: list[Any]) -> "dict[str, list[Any]]":
        """Return a {file_path: [ChunkSnippet, …]} dict for all local chunks."""
        from VisualMarkers import ChunkSnippet

        grouped: dict[str, list[Any]] = {}
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() == "web":
                continue
            file_path = str(meta.get("FilePath", "")).strip()
            if not file_path or not os.path.isfile(file_path):
                self.pretty.write(
                    "D",
                    "VisualMarker",
                    f"Skipping chunk: FilePath={file_path!r} "
                    f"isfile={os.path.isfile(file_path) if file_path else 'n/a'}",
                )
                continue
            page_number = meta.get("PageNumber")
            try:
                page_int: int | None = (
                    int(page_number) if page_number is not None else None
                )
            except (TypeError, ValueError):
                page_int = None
            grouped.setdefault(file_path, []).append(
                ChunkSnippet(
                    text=getattr(doc, "page_content", "") or "", page_number=page_int
                )
            )
        return grouped

    def _resolve_mark_colors(self) -> "tuple[str, str]":
        """Return (highlight_color, answer_mark_color) from config with fallbacks."""
        highlight_color: str = ""
        answer_mark_color: str = ""
        get_str = getattr(self.cfg, "get_str", None)
        if callable(get_str):
            highlight_color = str(
                get_str("_MARKED_DOCS_COLORS.highlight") or ""
            )  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            answer_mark_color = str(
                get_str("_MARKED_DOCS_COLORS.answer_mark") or ""
            )  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        else:
            get_fn = getattr(self.cfg, "get", None)
            if callable(get_fn):
                colors = get_fn("_MARKED_DOCS_COLORS", {})
                if isinstance(colors, dict):
                    highlight_color = str(
                        colors.get("highlight", "") or ""
                    )  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                    answer_mark_color = str(
                        colors.get("answer_mark", "") or ""
                    )  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not highlight_color:
                indirect_get = getattr(self.cfg, "indirect_get", None)
                if callable(indirect_get):
                    result: Any = indirect_get(
                        "MARKED_DOCS_HIGHLIGHT_COLOR"
                    )  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    val: Any = result[0]  # pyright: ignore[reportUnknownVariableType]
                    highlight_color = str(val or "")
        return highlight_color or "yellow", answer_mark_color or "orange"

    def _build_grounded_snippets(
        self,
        mySession: Session,
        chosen: list[Any],
        answer_mark_color: str,
    ) -> "dict[str, list[Any]]":
        """Return {file_path: [orange ChunkSnippet, …]} for grounded sentences."""
        from VisualMarkers import ChunkSnippet

        grounded: dict[str, list[Any]] = {}
        chunk_texts: list[str] = list(
            getattr(mySession, "chunk_texts_for_grounding", []) or []
        )
        if DebugHelper.check(self.cfg, 33):
            self.pretty.write(
                "D",
                "Grounding",
                f"chunk_texts_for_grounding={len(chunk_texts)}  "
                f"last_answer_content={bool(getattr(mySession, 'last_answer_content', ''))}  "
                f"answer_mark_color={answer_mark_color!r}",
                color=CYAN,
            )

        if not chunk_texts:
            return grounded

        answer_text = getattr(mySession, "last_answer_content", "")
        if not answer_text:
            return grounded

        from VisualMarkers.AnswerGrounder import AnswerGrounder

        grounder = AnswerGrounder()

        seen_by_file: dict[str, set[str]] = {}
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() == "web":
                continue
            file_path = str(meta.get("FilePath", "")).strip()
            if not file_path or not os.path.isfile(file_path):
                if DebugHelper.check(self.cfg, 33):
                    self.pretty.write(
                        "D",
                        "Grounding",
                        f"skip chunk: FilePath={file_path!r}  "
                        f"exists={os.path.isfile(file_path) if file_path else False}",
                        color=CYAN,
                    )
                continue
            chunk_text = getattr(doc, "page_content", "") or ""
            if not chunk_text.strip():
                continue

            matched = grounder.find_grounded_sentences(answer_text, [chunk_text])
            if DebugHelper.check(self.cfg, 33):
                fn = meta.get("FileName", os.path.basename(file_path))
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"chunk {fn!r}  matched={len(matched)}  "
                    + (repr(matched) if matched else "(none)"),
                    color=CYAN,
                )
            if not matched:
                continue

            # Use verbatim chunk sentences (matched by window overlap) for PDF highlighting.
            pdf_texts = grounder.find_grounding_fragments_in_chunk(matched, chunk_text)
            if DebugHelper.check(self.cfg, 33):
                fn = meta.get("FileName", os.path.basename(file_path))
                fallback = pdf_texts == matched
                status = (
                    "FALLBACK — no overlap match"
                    if fallback
                    else "chunk sentences matched"
                )
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"chunk {fn!r}  pdf_snippets={len(pdf_texts)}  ({status})",
                    color=CYAN,
                )
                for i, sentence in enumerate(matched):
                    hits = grounder.find_grounding_fragments_in_chunk(
                        [sentence], chunk_text
                    )
                    hit_strs = hits if hits != [sentence] else []
                    self.pretty.write(
                        "D",
                        "Grounding",
                        f"  answer[{i}]: {sentence.strip()!r}",
                        color=CYAN,
                    )
                    if hit_strs:
                        for h in hit_strs:
                            span = grounder.find_first_overlap_span(sentence, h)
                            self.pretty.write(
                                "D", "Grounding", f"    → pdf:   {h!r}", color=CYAN
                            )
                            if span:
                                self.pretty.write(
                                    "D",
                                    "Grounding",
                                    f"    → match: >>>{span}<<<",
                                    color=CYAN,
                                )
                    else:
                        self.pretty.write(
                            "D", "Grounding", "    → NO MATCH in chunk", color=CYAN
                        )

            seen = seen_by_file.setdefault(file_path, set())
            for text in pdf_texts:
                key = " ".join(text.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                grounded.setdefault(file_path, []).append(
                    ChunkSnippet(text=text, page_number=None, color=answer_mark_color)
                )
        return grounded

    def _produce_marked_bytes(
        self,
        grouped: "dict[str, list[Any]]",
        grounded_snippets: "dict[str, list[Any]]",
        highlight_color: str,
    ) -> "list[tuple[str, bytes]]":
        """Combine yellow + orange snippets, run the marker, return (path, bytes) pairs."""
        from pathlib import Path

        from VisualMarkers import VisualMarkerFactory

        grounded_paths = set(grounded_snippets.keys())
        only_grounded = bool(grounded_paths)
        if DebugHelper.check(self.cfg, 33):
            total_orange = sum(len(v) for v in grounded_snippets.values())
            self.pretty.write(
                "D",
                "Grounding",
                f"grounded_paths={len(grounded_paths)}  total_orange_snippets={total_orange}  "
                f"only_grounded={only_grounded}  grouped_files={list(grouped.keys())}",
                color=CYAN,
            )

        produced: list[tuple[str, bytes]] = []
        for src_path, snippets in grouped.items():
            if only_grounded and src_path not in grounded_paths:
                if DebugHelper.check(self.cfg, 33):
                    self.pretty.write(
                        "D",
                        "Grounding",
                        f"skipping (no grounded sentences): {src_path!r}",
                        color=CYAN,
                    )
                continue
            marker = VisualMarkerFactory.for_path(src_path)
            if marker is None:
                continue
            all_snippets = list(snippets) + grounded_snippets.get(src_path, [])
            if DebugHelper.check(self.cfg, 33):
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"marking {os.path.basename(src_path)!r}: "
                    f"yellow={len(snippets)}  orange={len(grounded_snippets.get(src_path, []))}  "
                    f"marker={type(marker).__name__}",
                    color=CYAN,
                )
            try:
                pdf_bytes = marker.mark_to_bytes(
                    Path(src_path), all_snippets, highlight_color=highlight_color
                )
            except Exception as exc:
                self.pretty.write(
                    "W", "VisualMarker", f"Failed to mark {src_path}: {exc}"
                )
                continue
            produced.append((src_path, pdf_bytes))
        return produced

    def _print_bm25_debug(self, docs: list[Any]) -> None:
        """Print BM25 retrieval debug table."""
        header = "{:>6}  {:>12}  {:>17}   {}"
        row = "{:>6}  {:>12.4f}  {:>17}   {}"
        self.pretty.write(
            "D",
            "BM25",
            header.format("Pos", "BM25Score", "Retrievers", "File"),
            color=CYAN,
        )
        self.pretty.write("D", "BM25", "-" * 71, color=CYAN)
        for i, doc in enumerate(docs[:20], start=1):
            score = doc.metadata.get("bm25_score", 0.0)
            sources = str(doc.metadata.get("retriever_sources", ""))
            fn = doc.metadata.get("FileName", "<unknown>")
            self.pretty.write("D", "BM25", row.format(i, score, sources, fn))

    def _print_graph_debug(self, docs: list[Any]) -> None:
        """Print graph retrieval debug table (shown at debug_level >= 30)."""
        header = "{:>6}  {:>12}  {:>17}   {}"
        row = "{:>6}  {:>12.4f}  {:>17}   {}"
        self.pretty.write(
            "D",
            "Graph",
            header.format("Pos", "GraphScore", "Retrievers", "File"),
            color=CYAN,
        )
        self.pretty.write("D", "Graph", "-" * 71, color=CYAN)
        for i, doc in enumerate(docs[:20], start=1):
            score = doc.metadata.get("graph_score", 0.0)
            sources = str(doc.metadata.get("retriever_sources", ""))
            fn = doc.metadata.get("FileName", "<unknown>")
            self.pretty.write("D", "Graph", row.format(i, score, sources, fn))

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
        header = "{:>6}  {:>12}  {:>17}   {}"
        row = "{:>6}  {:>12.4f}  {:>17}   {}"
        self.pretty.write(
            "D",
            "Web",
            header.format("Pos", "WebScore", "Retrievers", "URL"),
            color=CYAN,
        )
        self.pretty.write("D", "Web", "-" * 71, color=CYAN)
        for i, doc in enumerate(docs[:20], start=1):
            score = doc.metadata.get("chroma_score", 0.0)
            sources = str(doc.metadata.get("retriever_sources", ""))
            url = doc.metadata.get("FilePath", "<unknown>")
            self.pretty.write("D", "Web", row.format(i, score, sources, url))

    def _print_merged_debug(self, docs: list[Any]) -> None:
        """Print RRF-merged results debug table with retriever origin and filename.

        Column widths mirror the Rerank debug table so Pos / Retrievers / File
        stay visually aligned across both outputs:
          Pos(6)  RRFScore(10)  [blank AdjScore](10)  Retrievers(17)  File(30)
        """
        header = "{:>6}  {:>10}  {:>10}  {:>17}  {:<30}"
        row = "{:>6}  {:>10.4f}  {:>10}  {:>17}  {:<30}"
        self.pretty.write(
            "D",
            "Merge",
            header.format("Pos", "RRFScore", "", "Retrievers", "File"),
            color=CYAN,
        )
        self.pretty.write("D", "Merge", "-" * 82, color=CYAN)
        for i, doc in enumerate(docs, start=1):
            score = doc.metadata.get("rrf_score", 0.0)
            sources = doc.metadata.get("retriever_sources", "")
            fn = doc.metadata.get("FileName", "<unknown>")
            self.pretty.write("D", "Merge", row.format(i, score, "", sources, fn))

    # ── Multi-Query Expansion ──────────────────────────────────────────────────

    def _generate_alternate_queries(
        self, query: str, mySession: "Session"
    ) -> list[str]:
        """Generate alternate phrasings of *query* using the rewrite LLM.

        Returns a list of up to ``_MULTI_QUERY.num_variants`` strings on
        success, or an empty list on any error or when the feature is disabled.
        """
        import json

        enabled: bool = self.cfg.get_bool("_MULTI_QUERY.enabled")
        if not enabled or not query:
            return []

        num_variants: int = self.cfg.get_int("_MULTI_QUERY.num_variants") or 3
        temperature: float = (
            self.cfg.get_float("_MULTI_QUERY.LLM_PARAM.temperature") or 0.5
        )
        top_k: int = self.cfg.get_int("_MULTI_QUERY.LLM_PARAM.top_k") or 40
        top_p: float = self.cfg.get_float("_MULTI_QUERY.LLM_PARAM.top_p") or 0.95
        num_predict: int = self.cfg.get_int("_MULTI_QUERY.LLM_PARAM.num_predict") or 256
        use_gpu: bool = self.cfg.get_bool("_MULTI_QUERY.LLM_PARAM.use_ollama_gpu")

        # Reuse the same model already loaded for topic detection / query rewrite.
        llm_args: dict[str, Any] = self.helperInstance.get_model_args(
            "_ACTIVE_LLM_REWRITE_PROMPT"
        )
        llm_model: str = llm_args["MODEL"]
        expand_prompt_var: str = llm_args.get(
            "PROMPT_QUERY_EXPAND", "_PROMPT_QUERY_EXPAND"
        )
        prompt_template: str
        prompt_name: str | None
        prompt_template, prompt_name = self.cfg.indirect_get(expand_prompt_var)

        formatted: str = prompt_template.format(
            num_variants=num_variants,
            query=query,
        )

        effective_ctx: int = self.tokenBudget.get_effective_context_limit(
            llm_model, mySession
        )

        ollama_options: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": effective_ctx,
        }
        if not use_gpu:
            ollama_options["num_gpu"] = 0

        try:
            from AI.LLMCaller import LLMCaller

            result: dict[str, str] = LLMCaller().call_llm(
                model=llm_model,
                prompt=formatted,
                ollama_options=ollama_options,
                answer_is_json=True,
                template_name=prompt_name,
                streaming=False,
                stage="Multi-query expansion",
            )
        except Exception as exc:
            self.pretty.write(
                "W",
                "MultiQuery",
                f"LLM call failed — skipping expansion: {exc}",
            )
            return []

        raw: str = (result.get("content") or "").strip()
        if not raw:
            return []

        try:
            variants: Any = json.loads(raw)
            if not isinstance(variants, list):
                return []
            cleaned: list[str] = [
                str(v).strip() for v in variants if str(v).strip()  # type: ignore[reportUnknownVariableType]
            ]
            return cleaned[:num_variants]
        except Exception:
            self.pretty.write(
                "W",
                "MultiQuery",
                f"JSON parse failed — skipping expansion. Raw: {raw[:120]}",
            )
            return []

    # ── Chunk Near-Duplicate Removal ───────────────────────────────────────────

    def _remove_similar_chunks(self, docs: list[Any], threshold: float) -> list[Any]:
        """Remove near-duplicate chunks using Jaccard similarity on word tokens.

        Iterates *docs* in ranked order (highest rank first).  For each
        candidate, compares it against all already-kept chunks; if the Jaccard
        similarity of their lowercased word tokens meets *threshold*, the
        candidate is discarded.  The kept list preserves the original ranking.
        """
        kept: list[Any] = []
        kept_tokens: list[list[str]] = []

        for doc in docs:
            tokens: list[str] = doc.page_content.lower().split()
            is_dup = any(
                self._shared.jaccard(tokens, kt) >= threshold for kt in kept_tokens
            )
            if not is_dup:
                kept.append(doc)
                kept_tokens.append(tokens)

        return kept
