"""Typed shared-state contract for RAGChatImpl responsibility classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from typing import Sequence, Tuple

    from chromadb.api.models.Collection import Collection
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from AI.TokenBudget import TokenBudget
    from Chat.ChatContext import ChatContext
    from Chat.ConfidenceEvaluator import ConfidenceEvaluator
    from Chat.PromptRewrite import PromptRewrite
    from Chat.RetrievalGate import RetrievalGate
    from Compliance.SharedHelpers import SharedHelpers
    from Config.Config import Config
    from Globals.Session import Session
    from Gui.PrettyWriter import PrettyWriter
    from Helpers.ConfidenceLogger import ConfidenceLogger
    from Helpers.FileUtils import FileUtils
    from Helpers.Helpers import Helpers
    from Helpers.PerfLogger import PerfLogger
    from Orchestrator.Orchestrator import Orchestrator
    from Retrievers.BM25Retriever import BM25Retriever
    from Retrievers.GraphRetriever import GraphRetriever
    from Retrievers.RegexRetriever import RegexRetriever
    from Retrievers.WebPreFilter import WebPreFilter
    from Retrievers.WebRetriever import WebRetriever
    from Retrievers.WebSearchFilter import WebSearchFilter


class RagRetrievalContext(Protocol):
    _fileUtils: FileUtils

    _shared: SharedHelpers

    _translation_backend: str

    bm25_retriever: BM25Retriever

    cfg: Config

    chatContext: ChatContext

    collection: Collection | None

    collection_name: str

    confidenceEvaluator: ConfidenceEvaluator

    confidence_logger: ConfidenceLogger

    cross_encoder_model_name: str

    cross_encoder_query_instruction: str

    embedder: Any

    graph_retriever: GraphRetriever

    helperInstance: Helpers

    intent_filter: WebSearchFilter

    orchestrator: Orchestrator

    perf_logger: PerfLogger

    persist_directory: str | None

    pretty: PrettyWriter

    promptRewrite: PromptRewrite

    regex_retriever: RegexRetriever

    retrievalGate: RetrievalGate

    tokenBudget: TokenBudget

    vector_store: Chroma | None

    web_pre_filter: WebPreFilter

    web_retriever: WebRetriever

    x_encoder: Any

    def set_vector_store(self, mySession: Session) -> bool: ...

    @staticmethod
    def _chroma_where(flt: "dict[str, Any] | None") -> "dict[str, Any] | None": ...

    def _vector_kwargs(self, mySession: Session) -> "dict[str, Any]": ...

    def _set_vector_store(self, mySession: Session) -> bool: ...

    def _get_rerank_language_queries(self) -> dict[str, str]: ...

    def _predict_rerank_scores(
        self,
        candidates: list[Any],
        pairs: list[tuple[str, str]],
        queries_by_language: dict[str, str] | None = None,
    ) -> list[float]: ...

    def _select_best_rerank_language(
        self, candidates: list[Any], raw_scores: list[float]
    ) -> tuple[list[Any], list[float], str | None]: ...

    def _predict_keyword_action_scores(
        self, mySession: Session, candidates: list[Any], pairs: list[tuple[str, str]]
    ) -> list[float]: ...

    def _rerank(self, mySession: Session, all_docs: list[Any]) -> list[Any]: ...

    def _print_chroma_debug(self, docs: Sequence[Any]) -> None: ...

    def _get_translator(self, backend: str) -> Any: ...

    def retrieve(self, mySession: Session) -> Tuple[str, int]: ...

    def _retrieve(self, mySession: Session) -> Tuple[str, int]: ...

    @staticmethod
    def parse_chat_command(query: str) -> tuple[str, str]: ...

    @staticmethod
    def summary_selection_warning(session: Session, target: str = "") -> str | None: ...

    def retrieve_summary_documents(
        self, session: Session, target: str = ""
    ) -> list[Document]: ...

    def indexed_file_metadata(
        self, session: Session, file_names: list[str]
    ) -> list[dict[str, Any]]: ...

    def _prepare_session(self, mySession: Session) -> str: ...

    def _drain_lang_detection_confidence_events(self, mySession: Session) -> None: ...

    def _resolve_turn_translation_backend(
        self, mySession: Session
    ) -> tuple[str, Any]: ...

    def _detect_raw_query_language(self, mySession: Session, raw_query: str) -> str: ...

    @staticmethod
    def _is_native_retrieval(mySession: Session) -> bool: ...

    def _translate_query_to_english(
        self,
        query_text: str,
        source_language: str,
        translator: Any,
        backend: str,
        log_prefix: str,
    ) -> tuple[str, bool]: ...

    def _rewrite_query_with_strict_retry(
        self, mySession: Session
    ) -> tuple[str, bool]: ...

    def _apply_post_rewrite_translation(
        self, mySession: Session, rewrite_language: str, translator: Any, backend: str
    ) -> bool: ...

    def _log_query_language_drift(
        self, mySession: Session, raw_query_language: str, orig_translated_query_en: str
    ) -> None: ...

    def _set_effective_query_metadata(
        self,
        mySession: Session,
        final_query: str,
        user_query_original: str,
        was_translated: bool,
        was_rewritten: bool,
    ) -> None: ...

    def _normalize_query(
        self, mySession: Session, user_query_original: str
    ) -> tuple[str, list[str]]: ...

    def _check_gates(self, mySession: Session, final_query: str) -> bool: ...

    @staticmethod
    def _resolve_file_filter(mySession: Session) -> dict[str, Any] | None: ...

    def _query_language_label(
        self, mySession: Session, *, file_filter: dict[str, Any] | None = None
    ) -> str: ...

    def _normalize_language_bucket(self, language: Any) -> str | None: ...

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        enable_translation: bool = True,
        enable_synonym_expansion: bool = True,
    ) -> tuple[str, str]: ...

    def _expand_indexed_query_with_synonyms(self, query: str) -> str: ...

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
    ) -> tuple[list[Any], int, int]: ...

    def _run_vector_retriever_with_guardrail(
        self,
        *,
        mySession: Session,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> tuple[list[Any], int, int]: ...

    @staticmethod
    def _mode_flags_for_local_retrievers(
        retrieve_mode: str,
    ) -> tuple[bool, bool, bool, bool]: ...

    @staticmethod
    def _is_guardrail_query_active(
        primary_query: str, guardrail_query: str
    ) -> bool: ...

    def _resolve_guardrail_queries(
        self, *, bm25_query: str, orig_translated_query_en: str
    ) -> tuple[str, str, bool]: ...

    @staticmethod
    def _accumulate_topk_counts(
        current_orig_query_en: int,
        current_post_rewrite_query_en: int,
        add_orig_query_en: int,
        add_post_rewrite_query_en: int,
    ) -> tuple[int, int]: ...

    def _indexed_retriever_specs(
        self,
        mySession: Session,
        *,
        bm25_enabled: bool,
        graph_enabled: bool,
        regex_enabled: bool,
    ) -> list[tuple[str, bool, float, Any, int, Any]]: ...

    def _run_indexed_local_retrievers(
        self,
        mySession: Session,
        *,
        indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]: ...

    def _store_guardrail_retrieval_topk(
        self,
        mySession: Session,
        *,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None: ...

    def _fetch_local_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]: ...

    def _fetch_web_docs(
        self, mySession: Session, retrieve_mode: str, user_query_original: str
    ) -> list[Any]: ...

    def _merge_local_docs_by_language(
        self,
        local_active: list[list[Any]],
        labels: list[str],
        weights: list[float],
        *,
        rrf_k: int,
        limit: int | None,
    ) -> list[Any] | None: ...

    def _merge_and_select(
        self,
        mySession: Session,
        vector_docs: list[Any],
        bm25_docs: list[Any],
        graph_docs: list[Any],
        regex_docs: list[Any],
        web_docs: list[Any],
    ) -> list[Any]: ...

    def _build_context(
        self, mySession: Session, chosen: list[Any]
    ) -> Tuple[str, int]: ...

    def _mark_sources(
        self, mySession: Session, chosen: list[Any], *, summary: bool = False
    ) -> None: ...

    def _group_chunks_by_file(self, chosen: list[Any]) -> "dict[str, list[Any]]": ...

    def _resolve_mark_colors(self) -> "tuple[str, str]": ...

    def _produce_marked_bytes(
        self,
        grouped: "dict[str, list[Any]]",
        grounded_snippets: "dict[str, list[Any]]",
        highlight_color: str,
    ) -> "list[tuple[str, bytes]]": ...

    def _mark_pdfs_in_parallel(
        self, jobs: "list[tuple[str, Any, list[Any]]]", highlight_color: str
    ) -> "dict[str, bytes]": ...

    def _print_bm25_debug(self, docs: list[Any]) -> None: ...

    def _print_graph_debug(self, docs: list[Any]) -> None: ...

    def _print_single_score_retriever_debug(
        self,
        docs: list[Any],
        *,
        channel: str,
        score_header: str,
        score_key: str,
        file_header: str,
        file_key: str,
    ) -> None: ...

    def _print_regex_debug(self, docs: list[Any]) -> None: ...

    def _print_web_prefilter_debug(
        self, kept: list[Any], dropped: list[Any], filter_name: str, threshold: float
    ) -> None: ...

    def _print_web_debug(self, docs: list[Any]) -> None: ...

    def _print_merged_debug(self, docs: list[Any]) -> None: ...

    def _generate_alternate_queries(
        self, query: str, mySession: "Session"
    ) -> list[str]: ...

    def _remove_similar_chunks(
        self, docs: list[Any], threshold: float
    ) -> list[Any]: ...
