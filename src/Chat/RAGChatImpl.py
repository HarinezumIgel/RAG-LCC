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
from Chat.RetrievalOrchestrator import RetrievalOrchestrator
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
from Helpers.Helpers import (Helpers, align_retriever_sources_for_print,
                             truncate_for_print)
from Helpers.PerfLogger import PerfLogger
from Retrievers.BM25Retriever import BM25Retriever
from Retrievers.GraphRetriever import GraphRetriever
from Retrievers.RegexRetriever import RegexRetriever
from Retrievers.RetrieverProtocol import RetrieverProtocol
from Retrievers.WebPreFilter import WebPreFilter
from Retrievers.WebRetriever import WebRetriever
from Retrievers.WebSearchFilter import WebSearchFilter
from Strategies.HomeBrewChunkSelector import ChunkSelectionService

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
        self.retrieval_orchestrator: RetrievalOrchestrator = RetrievalOrchestrator(self)

    def set_vector_store(self, mySession: Session) -> bool:
        """Thread-safe wrapper — acquires ``self._lock`` then delegates."""
        with self._lock:
            return self._set_vector_store(mySession)

    @staticmethod
    def _chroma_where(flt: "dict[str, Any] | None") -> "dict[str, Any] | None":
        """Normalise a flat multi-field filter into ChromaDB ``where`` form.

        ChromaDB requires an explicit ``$and`` when more than one field is
        present. BM25/graph matchers AND flat multi-key dicts directly, so the
        shared ``base_kwargs`` filter stays flat and is converted only here for
        the vector query.
        """
        if not flt:
            return flt
        # Single field or an existing logical operator ($and/$or) — pass through.
        if len(flt) <= 1:
            return flt
        return {"$and": [{k: v} for k, v in flt.items()]}

    def _vector_kwargs(self, mySession: Session) -> "dict[str, Any]":
        """Return base_kwargs with the filter normalised for the Chroma query."""
        kwargs: dict[str, Any] = dict(mySession.base_kwargs or {})
        if "filter" in kwargs:
            kwargs["filter"] = self._chroma_where(kwargs["filter"])
        return kwargs

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
            header = "{:>6}  {:>10}  {:>10}  {:>24}  {:<40}  {}"
            row = "{:>6}  {:>10.4f}  {:>10.4f}  {:>24}  {:<40}  {}"

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
                sources: str = align_retriever_sources_for_print(
                    str(d.metadata.get("retriever_sources", "")), 24
                )
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

    def _get_translator(self, backend: str) -> Any:
        """Return translator for *backend*.

        Supported values:
        - ``"argos"``
        - ``"off"``

        Unknown or ``"off"`` values return ``None`` so callers can skip
        translation with a simple ``if translator is not None`` guard.
        """
        b = (backend or "off").lower()
        if b == "argos":
            return self._shared
        return None

    def retrieve(self, mySession: Session) -> Tuple[str, int]:
        """Thread-safe wrapper — acquires ``self._lock`` then delegates."""
        with self._lock:
            return self._retrieve(mySession)

    def _retrieve(self, mySession: Session) -> Tuple[str, int]:
        return self.retrieval_orchestrator.run(mySession)

    def _prepare_session(self, mySession: Session) -> str:
        """Reset per-turn flags, handle mode-change and topic-switch resets.

        Returns user_query_original (the query before any translation/rewrite).
        """
        mySession.force_skip_rewrite = False
        mySession.effective_query = None
        mySession.effective_query_reason = None
        mySession.user_language = None
        mySession.retrieval_language = "english"
        mySession.orig_translated_query_en = None
        mySession.seed_retrieval_query = None
        mySession.t1_query = None
        mySession.rewritten_query = None
        mySession.rewrite_language = None
        mySession.post_rewrite_query_en = None
        mySession.final_retrieval_query = None
        mySession.t2_query = None
        mySession.retrieval_top_k_orig_query_en = None
        mySession.retrieval_top_k_post_rewrite_query_en = None
        mySession.retrieval_top_k_seed_query = None
        mySession.retrieval_top_k_final_query = None
        mySession.retrieval_top_k_before_t2 = None
        mySession.retrieval_top_k_after_t2 = None

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

    def _resolve_turn_translation_backend(self, mySession: Session) -> tuple[str, Any]:
        """Resolve the per-turn translation backend and translator instance."""
        backend: str = (
            getattr(mySession, "translation_backend", None)
            or self._translation_backend
            or "off"
        ).lower()
        return backend, self._get_translator(backend)

    def _detect_raw_query_language(self, mySession: Session, raw_query: str) -> str:
        """Detect and persist the user query language for this turn."""
        raw_query_language: str = (
            self._fileUtils.get_user_text_language(
                raw_query,
                output="nltk",
                native_lang=mySession.preferred_response_language,
            )
            if raw_query
            else "english"
        )
        # Keep chat-history language filtering on the user language, not the
        # retrieval language, so language-specific contexts remain isolated.
        mySession.user_language = raw_query_language
        mySession.current_query_lang = raw_query_language
        mySession.retrieval_language = "english"
        return raw_query_language

    def _translate_query_to_english(
        self,
        query_text: str,
        source_language: str,
        translator: Any,
        backend: str,
        log_prefix: str,
    ) -> tuple[str, bool]:
        """Translate query_text to English when required; return (text, translated?)."""
        if translator is None or not query_text or source_language == "english":
            return query_text, False

        translated: str = translator.translate_text(
            query_text,
            target_lang="en",
            source_lang=source_language,
        )
        if translated and translated != query_text:
            self.pretty.write(
                "I",
                "QueryNorm",
                f"{log_prefix} [{source_language}→english, "
                f"backend={backend}]: "
                f"{query_text!r} → {translated!r}",
            )
            return translated, True
        return query_text, False

    def _rewrite_query_with_strict_retry(
        self,
        mySession: Session,
    ) -> tuple[str, bool]:
        """Rewrite the query and retry once with strict-English instructions."""
        pre_rewrite_query: str = mySession.query or ""
        if mySession.use_chat_context:
            mySession.query = self.promptRewrite.rewrite(mySession)

        rewritten_query: str = mySession.query or ""
        was_rewritten: bool = rewritten_query != pre_rewrite_query

        rewrite_language: str = (
            self._fileUtils.get_user_text_language(
                rewritten_query,
                output="nltk",
                native_lang=None,
            )
            if rewritten_query
            else "english"
        )
        mySession.rewritten_query = rewritten_query
        mySession.rewrite_language = rewrite_language

        # If rewrite drifted out of English, retry with stricter instructions
        # before falling back to translation.
        if (
            rewrite_language != "english"
            and mySession.use_chat_context
            and not mySession.force_skip_rewrite
        ):
            strict_candidate: str = (
                self.promptRewrite.rewrite(mySession, strict_english=True) or ""
            ).strip()
            if strict_candidate:
                strict_lang: str = self._fileUtils.get_user_text_language(
                    strict_candidate,
                    output="nltk",
                    native_lang=None,
                )
                if strict_lang == "english":
                    mySession.query = strict_candidate
                    rewritten_query = strict_candidate
                    rewrite_language = strict_lang
                    mySession.rewritten_query = rewritten_query
                    mySession.rewrite_language = rewrite_language
                    was_rewritten = rewritten_query != pre_rewrite_query
                    self.pretty.write(
                        "I",
                        "QueryNorm",
                        "Strict English rewrite retry succeeded.",
                    )

        return rewrite_language, was_rewritten

    def _apply_post_rewrite_translation(
        self,
        mySession: Session,
        rewrite_language: str,
        translator: Any,
        backend: str,
    ) -> bool:
        """Enforce English after rewrite; return whether a translation happened."""
        if rewrite_language == "english":
            return False

        if translator is not None and mySession.query:
            translated_after, translated_flag = self._translate_query_to_english(
                mySession.query,
                rewrite_language,
                translator,
                backend,
                "Normalized rewritten query",
            )
            if translated_flag:
                mySession.query = translated_after
            return translated_flag

        self.pretty.write(
            "W",
            "QueryNorm",
            "Rewrite language is non-English but no translator is active; "
            "retrieval query may stay cross-lingual.",
        )
        return False

    def _log_query_language_drift(
        self,
        mySession: Session,
        raw_query_language: str,
        orig_translated_query_en: str,
    ) -> None:
        """Emit standardized language-drift debug lines for retrieval stages."""
        self.pretty.write(
            "I",
            "LangDrift",
            f"raw_query_language={raw_query_language}  "
            f"orig_translated_query_en={orig_translated_query_en!r}",
        )
        self.pretty.write(
            "I",
            "LangDrift",
            f"rewrite_language={mySession.rewrite_language}  "
            f"rewritten_query={mySession.rewritten_query!r}",
        )
        self.pretty.write(
            "I",
            "LangDrift",
            f"post_rewrite_query_en={mySession.post_rewrite_query_en!r}",
        )

    def _set_effective_query_metadata(
        self,
        mySession: Session,
        final_query: str,
        user_query_original: str,
        was_translated: bool,
        was_rewritten: bool,
    ) -> None:
        """Persist effective-query fields and emit the FinalQuery log line."""
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
            return

        self.pretty.write(
            "I",
            "FinalQuery",
            f"Final query for retrieval: {final_query!r} (unchanged)",
            color=CYAN,
        )

    def _normalize_query(
        self, mySession: Session, user_query_original: str
    ) -> tuple[str, list[str]]:
        """Apply the retrieval-language contract and produce alternate queries.

        Pipeline:
          1) Detect raw user language (user_language)
          2) Build orig_translated_query_en in retrieval_language (English)
          3) Rewrite in retrieval language
          4) Enforce retrieval language after rewrite
          5) Generate alternate retrieval queries

        Returns (final_query, alternate_queries).
        """
        backend, translator = self._resolve_turn_translation_backend(mySession)

        raw_query: str = mySession.query or ""
        raw_query_language: str = self._detect_raw_query_language(
            mySession,
            raw_query,
        )

        # Build the original translated retrieval query before rewrite.
        orig_translated_query_en, translated_orig_query = (
            self._translate_query_to_english(
                raw_query,
                raw_query_language,
                translator,
                backend,
                "Normalized user query",
            )
        )
        was_translated: bool = translated_orig_query

        mySession.orig_translated_query_en = orig_translated_query_en
        # Seed retrieval query = the original user query normalized to English
        # before rewrite. Guardrail retrieval can reuse this form when the
        # post-rewrite query drifts and misses relevant chunks.
        mySession.seed_retrieval_query = orig_translated_query_en
        mySession.t1_query = orig_translated_query_en
        mySession.query = orig_translated_query_en

        rewrite_language, was_rewritten = self._rewrite_query_with_strict_retry(
            mySession
        )

        # Build the post-rewrite English retrieval query.
        translated_post_rewrite = self._apply_post_rewrite_translation(
            mySession,
            rewrite_language,
            translator,
            backend,
        )

        was_translated = translated_post_rewrite or was_translated

        final_query: str = mySession.query or ""
        mySession.post_rewrite_query_en = final_query
        mySession.final_retrieval_query = final_query
        mySession.t2_query = final_query

        if DebugHelper.check_session(mySession, 29):
            self._log_query_language_drift(
                mySession,
                raw_query_language,
                orig_translated_query_en,
            )

        self._set_effective_query_metadata(
            mySession,
            final_query,
            user_query_original,
            was_translated,
            was_rewritten,
        )

        alternate_queries: list[str] = self._generate_alternate_queries(
            final_query,
            mySession,
        )
        if alternate_queries and DebugHelper.check_session(mySession, 29):
            query_lines = [f"{i+1}: {q!r}" for i, q in enumerate(alternate_queries)]
            self.pretty.write(
                "D",
                "MultiQuery",
                f"Alternate queries ({len(alternate_queries)})"
                + "\n"
                + "\n".join(query_lines),
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

    @staticmethod
    def _resolve_file_filter(mySession: Session) -> dict[str, Any] | None:
        """Extract the shared metadata file/path filter from base_kwargs."""
        if mySession.base_kwargs and "filter" in mySession.base_kwargs:
            return mySession.base_kwargs["filter"]
        return None

    def _query_language_label(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None = None,
    ) -> str:
        """Resolve a compact language label for retrieval query logs."""
        if isinstance(file_filter, dict) and "Language" in file_filter:
            language_value = file_filter.get("Language")
            if isinstance(language_value, dict):
                language_dict = cast(dict[str, Any], language_value)
                language_value = language_dict.get("$eq")
            normalized = self._normalize_language_bucket(language_value)
            if normalized:
                return normalized

        for candidate in (
            getattr(mySession, "retrieval_language", None),
            getattr(mySession, "current_query_lang", None),
            getattr(mySession, "user_language", None),
        ):
            normalized = self._normalize_language_bucket(candidate)
            if normalized:
                return normalized

        return "default"

    def _normalize_language_bucket(self, language: Any) -> str | None:
        """Normalize language names/codes to lower-case ISO-ish bucket code."""
        text = str(language or "").strip().lower()
        if not text:
            return None

        mapping_obj = getattr(self._shared, "lang_name_to_code", None)
        if isinstance(mapping_obj, dict):
            mapping = cast(dict[str, Any], mapping_obj)
            mapped = mapping.get(text)
            if mapped is not None:
                mapped_text = str(mapped).strip().lower()
                if mapped_text:
                    return mapped_text

        return text

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
    ) -> tuple[str, str]:
        """Shape Graph/Regex stage queries for one language bucket.

        Retrieval queries are normalized to English upstream. For non-English
        stage buckets, translate from English to the bucket language only when
        the configured query-translation backend is Argos.
        """
        _ = mySession
        target_language = self._normalize_language_bucket(language_bucket)
        if not target_language:
            return primary_query, guardrail_query

        if not target_language:
            return primary_query, guardrail_query
        if target_language == "en" or target_language.startswith("en-"):
            return primary_query, guardrail_query

        backend = str(getattr(self, "_translation_backend", "off") or "off")
        if backend.strip().lower() != "argos":
            return primary_query, guardrail_query

        translate_text = getattr(self._shared, "translate_text", None)
        if not callable(translate_text):
            return primary_query, guardrail_query

        def _translate_query(query: str) -> str:
            if not query:
                return query
            try:
                translated_obj = translate_text(query, target_language, "en")
            except Exception:
                return query
            translated = str(translated_obj or "").strip()
            return translated or query

        return _translate_query(primary_query), _translate_query(guardrail_query)

    def _run_indexed_retriever_with_guardrail(
        self,
        *,
        mySession: Session,
        retriever: Any,
        label: str,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[list[Any], int, int]:
        """Run one indexed retriever on final and optional guardrail query.

        The guardrail query is the pre-rewrite English seed retrieval query
        (``orig_translated_query_en``), used as a second retrieval leg when it
        differs from the final rewritten query.

        Returns:
            (docs, top_k_orig_query_en, top_k_post_rewrite_query_en)
        """
        assert self.collection is not None

        label_lower = label.lower()
        query_language = self._query_language_label(
            mySession,
            file_filter=file_filter,
        )
        self.pretty.write(
            "I",
            label,
            f"Querying {label_lower} index on collection {self.collection_name} "
            f"for language {query_language}",
        )
        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"start {label_lower} block collection={self.collection_name}",
        )
        _t_block = time.perf_counter()

        index_dir = retriever.get_index_dir(self.collection_name)
        retriever.load_or_rebuild(
            index_dir,
            self.collection_name,
            self.collection,
        )

        docs_final = retriever.query(
            primary_query,
            k=mySession.retriever_k or 100,
            file_filter=file_filter,
        )
        for doc in docs_final:
            doc.metadata["retriever_sources"] = label

        docs = docs_final
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = len(docs_final)

        if use_guardrail:
            # Guardrail leg: query again with the pre-rewrite English seed
            # retrieval query to recover when rewrite/post-rewrite translation
            # drifts away from the user intent.
            docs_seed = retriever.query(
                guardrail_query,
                k=mySession.retriever_k or 100,
                file_filter=file_filter,
            )
            for doc in docs_seed:
                doc.metadata["retriever_sources"] = label
            top_k_orig_query_en = len(docs_seed)
            if docs_seed:
                docs = BM25Retriever.reciprocal_rank_fusion(
                    docs_final,
                    docs_seed,
                    k=self.bm25_retriever.rrf_k,
                    labels=[label, label],
                    weights=[1.0, 1.0],
                )

        self.pretty.write(
            "O",
            label,
            f"{label} retrieval returned {len(docs)} chunks",
        )
        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  {label_lower} block n={len(docs)} elapsed={time.perf_counter() - _t_block:.3f}s",
        )

        return docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _run_vector_retriever_with_guardrail(
        self,
        *,
        mySession: Session,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> tuple[list[Any], int, int]:
        """Run vector retrieval on final and optional guardrail query.

        The guardrail query is the pre-rewrite English seed retrieval query
        (``orig_translated_query_en``), used as a second retrieval leg when it
        differs from the final rewritten query.

        Also runs alternate-query fanout and de-duplicates added chunks by id.

        Returns:
            (docs, top_k_orig_query_en, top_k_post_rewrite_query_en)
        """
        query_language = self._query_language_label(mySession)
        self.pretty.write(
            "I",
            "Chroma",
            f"Querying Chroma DB on vector store {self.persist_directory} "
            f"for language {query_language}",
        )
        assert (
            self.vector_store is not None
        ), "vector_store not initialized; call set_vector_store first"

        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            "start vector similarity_search",
        )
        _t_vec = time.perf_counter()
        vector_kwargs = self._vector_kwargs(mySession)

        hits_final: list[Any] = self.vector_store.similarity_search_with_score(
            primary_query,
            **vector_kwargs,
        )
        vector_docs_final = self.chatContext.annotate_chunks(hits_final)
        for doc in vector_docs_final:
            doc.metadata["retriever_sources"] = "Vector"

        vector_docs = vector_docs_final
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = len(vector_docs_final)

        if use_guardrail:
            try:
                # Guardrail leg mirrors indexed retrievers: fuse final-query and
                # pre-rewrite English seed-retrieval-query hits so retrieval
                # remains resilient to rewrite drift.
                hits_seed: list[Any] = self.vector_store.similarity_search_with_score(
                    guardrail_query,
                    **vector_kwargs,
                )
                vector_docs_seed = self.chatContext.annotate_chunks(hits_seed)
                for doc in vector_docs_seed:
                    doc.metadata["retriever_sources"] = "Vector"
                top_k_orig_query_en = len(vector_docs_seed)
                if vector_docs_seed:
                    vector_docs = BM25Retriever.reciprocal_rank_fusion(
                        vector_docs_final,
                        vector_docs_seed,
                        k=self.bm25_retriever.rrf_k,
                        labels=["Vector", "Vector"],
                        weights=[1.0, 1.0],
                    )
            except Exception as seed_exc:
                # Vector guardrail failures should not abort the whole turn.
                # We keep the primary-query result set and continue.
                self.pretty.write(
                    "W",
                    "MultiQuery",
                    f"Seed-query vector search failed: {seed_exc}",
                )

        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  vector similarity_search n={len(vector_docs)} elapsed={time.perf_counter() - _t_vec:.3f}s",
        )

        if alternate_queries:
            existing_ids: set[str] = {
                str(doc.metadata.get("id", doc.page_content)) for doc in vector_docs
            }
            for query_index, alt_query in enumerate(alternate_queries, start=1):
                try:
                    alt_hits: list[Any] = (
                        self.vector_store.similarity_search_with_score(
                            alt_query,
                            **vector_kwargs,
                        )
                    )
                    alt_docs = self.chatContext.annotate_chunks(alt_hits)
                    for doc in alt_docs:
                        doc_id = str(doc.metadata.get("id", doc.page_content))
                        if doc_id not in existing_ids:
                            doc.metadata["retriever_sources"] = "Vector"
                            vector_docs.append(doc)
                            existing_ids.add(doc_id)
                except Exception as alt_exc:
                    self.pretty.write(
                        "W",
                        "MultiQuery",
                        f"Alternate query {query_index} vector search failed: {alt_exc}",
                    )

        if DebugHelper.check_session(mySession, 10):
            self._print_chroma_debug(vector_docs)
        self.pretty.write(
            "O",
            "Chroma",
            f"Querying Chroma DB query returned {len(vector_docs)} chunks",
        )

        return vector_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    @staticmethod
    def _mode_flags_for_local_retrievers(
        retrieve_mode: str,
    ) -> tuple[bool, bool, bool, bool]:
        """Return enabled flags for Vector/BM25/Graph/Regex local retrievers."""
        mode_upper = (retrieve_mode or "VECTOR").upper()
        mode_parts = set(mode_upper.split("_"))
        mode_all = mode_upper == "ALL"

        vector_enabled = mode_all or "VECTOR" in mode_parts
        bm25_enabled = mode_all or "BM25" in mode_parts
        graph_enabled = mode_all or "GRAPH" in mode_parts
        regex_enabled = mode_all or "REGEX" in mode_parts
        return vector_enabled, bm25_enabled, graph_enabled, regex_enabled

    @staticmethod
    def _is_guardrail_query_active(primary_query: str, guardrail_query: str) -> bool:
        """Return True when the pre-rewrite English seed query should run.

        The second guardrail leg runs only when this seed retrieval query is
        distinct from the final rewritten retrieval query.
        """
        return bool(
            guardrail_query and guardrail_query.lower() != primary_query.strip().lower()
        )

    def _resolve_guardrail_queries(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> tuple[str, str, bool]:
        """Resolve primary and guardrail retrieval query forms.

        ``primary_query`` is the final rewritten retrieval query.
        ``guardrail_query`` is the pre-rewrite English seed retrieval query.
        The guardrail leg is active only when both forms differ.
        """
        primary_query: str = bm25_query or ""
        guardrail_query: str = (orig_translated_query_en or "").strip()
        use_guardrail: bool = self._is_guardrail_query_active(
            primary_query,
            guardrail_query,
        )
        return primary_query, guardrail_query, use_guardrail

    @staticmethod
    def _accumulate_topk_counts(
        current_orig_query_en: int,
        current_post_rewrite_query_en: int,
        add_orig_query_en: int,
        add_post_rewrite_query_en: int,
    ) -> tuple[int, int]:
        """Return updated guardrail counters after adding one retriever leg."""
        return (
            current_orig_query_en + add_orig_query_en,
            current_post_rewrite_query_en + add_post_rewrite_query_en,
        )

    def _indexed_retriever_specs(
        self,
        mySession: Session,
        *,
        bm25_enabled: bool,
        graph_enabled: bool,
        regex_enabled: bool,
    ) -> list[tuple[str, bool, float, Any, int, Any]]:
        """Build retrieval specs for indexed local retrievers."""
        bm25_weight = float(
            mySession.bm25_weight if mySession.bm25_weight is not None else 1.0
        )
        graph_weight = float(
            mySession.graph_weight if mySession.graph_weight is not None else 1.0
        )
        regex_weight = float(
            mySession.regex_weight if mySession.regex_weight is not None else 1.0
        )
        return [
            (
                "BM25",
                bm25_enabled,
                bm25_weight,
                self.bm25_retriever,
                10,
                self._print_bm25_debug,
            ),
            (
                "Graph",
                graph_enabled,
                graph_weight,
                self.graph_retriever,
                30,
                self._print_graph_debug,
            ),
            (
                "Regex",
                regex_enabled,
                regex_weight,
                self.regex_retriever,
                30,
                self._print_regex_debug,
            ),
        ]

    def _run_indexed_local_retrievers(
        self,
        mySession: Session,
        *,
        indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run BM25/Graph/Regex retrieval specs and collect guardrail counters.

        Guardrail counters compare hits from the pre-rewrite English seed
        retrieval query versus the final rewritten retrieval query.
        """
        indexed_docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = 0

        for (
            label,
            enabled,
            weight,
            retriever,
            debug_level,
            debug_printer,
        ) in indexed_specs:
            if not enabled or weight == 0.0:
                continue
            if label == "BM25":
                assert self.persist_directory is not None

            (
                docs,
                top_k_orig,
                top_k_post,
            ) = self._run_indexed_retriever_with_guardrail(
                mySession=mySession,
                retriever=retriever,
                label=label,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
            indexed_docs[label] = docs
            top_k_orig_query_en += top_k_orig
            top_k_post_rewrite_query_en += top_k_post
            if DebugHelper.check_session(mySession, debug_level):
                debug_printer(docs)

        return indexed_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _store_guardrail_retrieval_topk(
        self,
        mySession: Session,
        *,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None:
        """Store guardrail dual-query retrieval counts on the session.

        ``*_orig_query_en`` fields track the pre-rewrite English seed retrieval
        query. ``*_post_rewrite_query_en`` fields track the final rewritten
        retrieval query.
        """
        # These counters are only meaningful when the guardrail second leg ran.
        mySession.retrieval_top_k_orig_query_en = top_k_orig_query_en
        mySession.retrieval_top_k_post_rewrite_query_en = top_k_post_rewrite_query_en
        mySession.retrieval_top_k_seed_query = top_k_orig_query_en
        mySession.retrieval_top_k_final_query = top_k_post_rewrite_query_en
        mySession.retrieval_top_k_before_t2 = top_k_orig_query_en
        mySession.retrieval_top_k_after_t2 = top_k_post_rewrite_query_en
        self.pretty.write(
            "I",
            "LangDrift",
            f"retrieval_top_k_orig_query_en={top_k_orig_query_en}  "
            f"retrieval_top_k_post_rewrite_query_en={top_k_post_rewrite_query_en}",
        )

    def _fetch_local_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Delegate local retrieval orchestration to RetrievalOrchestrator."""
        return self.retrieval_orchestrator.run_local_retrievers(
            mySession,
            retrieve_mode,
            bm25_query,
            alternate_queries,
            orig_translated_query_en,
        )

    def _fetch_web_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        user_query_original: str,
    ) -> list[Any]:
        """Delegate web retrieval orchestration to RetrievalOrchestrator."""
        return self.retrieval_orchestrator.run_web_retriever(
            mySession,
            retrieve_mode,
            user_query_original,
        )

    def _merge_and_select(
        self,
        mySession: Session,
        vector_docs: list[Any],
        bm25_docs: list[Any],
        graph_docs: list[Any],
        regex_docs: list[Any],
        web_docs: list[Any],
    ) -> list[Any]:
        """RRF-fuse local docs, cap, append web docs, dedup, rerank, and select.

        Returns the final chosen list.
        """
        # Local retrievers (Vector, BM25, Graph, Regex) are fused via RRF and then
        # capped to retriever_k.  Web docs are appended AFTER the cap so they
        # always reach the reranker.  Including web in the same RRF pool caused
        # them to be pushed off the list: with weight=0.5 and only 5 results
        # their best RRF score (≈0.008) falls below the 100th local slot (≈0.011).
        local_sources = [
            (vector_docs, "Vector"),
            (bm25_docs, "BM25"),
            (graph_docs, "Graph"),
            (regex_docs, "Regex"),
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
            "Regex": float(
                mySession.regex_weight if mySession.regex_weight is not None else 1.0
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
            # Confine the answer highlight to the source chunk's physical page so a
            # fragment that also occurs earlier (TOC/front matter) is not marked there.
            try:
                page_value: Any = meta.get("PageNumber")
                page_int: int | None = (
                    int(page_value) if page_value is not None else None
                )
            except (TypeError, ValueError):
                page_int = None
            for text in pdf_texts:
                key = " ".join(text.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                grounded.setdefault(file_path, []).append(
                    ChunkSnippet(
                        text=text, page_number=page_int, color=answer_mark_color
                    )
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

            self.pretty.write(
                "I",
                "LLM Plan",
                f"Alternate queries: generating up to {num_variants} retrieval variants from the normalized query.",
            )

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
