from __future__ import annotations

# Local module imports
from typing import TYPE_CHECKING, Any, Optional

from Config.Config import Config
from Grounding.SourceReferences import SummaryStatement
from Gui.PrettyWriter import PrettyWriter
from Orchestrator import FlowConfig

if TYPE_CHECKING:
    from Chat.ConfidenceEvaluator import AnswerConfidenceResult
    from Chat.KeywordActionSplitter import KeywordActionSplit


class Session:
    """Per-request mutable state for the RAG pipeline.

    This is intentionally *not* a singleton.  The CLI path creates one
    instance and reuses it across the interactive loop; the service path
    creates a fresh instance for every incoming API request so that
    concurrent requests cannot clobber each other's parameters.
    """

    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.is_service: bool = False

        self.file_name: Optional[str | None] = None
        self.file_path: Optional[str | None] = None
        self.file_path_select: Optional[str | None] = (
            None  # Internal flag only, will not be shown to user
        )
        # Interactive metadata filters ({field: value}) applied as an extra
        # ChromaDB `where` condition during retrieval (metadata! / metadata= /
        # metadata- in-chat commands).
        self.metadata_filters: dict[str, str] = {}
        self.query: str | None = None
        # API input before command removal and operator-system-prompt composition.
        self.user_query_input: str | None = None
        self.indexed_query_notices: list[str] = []
        self.indexed_query_error: str | None = None
        self.indexed_query_warning_keys: set[tuple[str, str | None]] = set()
        self.strategy: str | None = None
        default_orchestration_flow = FlowConfig.default_flow_name(self.cfg)
        self.orchestrator_flow: str = default_orchestration_flow
        # Alias used by orchestration pipeline diagnostics.
        self.flow_name: str = default_orchestration_flow
        # None inherits the selected flow; baseline answer confidence is independent.
        self.confidence_evaluation: str | None = None
        self.summary_length: str | None = None
        self.summary_temperature: float | None = None
        self.chunk_overlap: int | None = None
        self.retriever_k: int | None = None
        self.rerank: bool | None = None
        self.final_chunks_to_llm: int | None = None
        self.chroma_threshold: float | None = None
        self.per_file_limit: int | None = None
        self.use_chat_context: bool | None = None
        self.turns: int | None = None
        self.prune_batch: int | None = None
        self.max_history_turns: int | None = None
        self.topic_summary_mode: str | None = None
        self.last_topic_referents: list[str] | None = (
            None  # saved from previous rewrite turn
        )
        self.vector_weight: float | None = None
        self.bm25_weight: float | None = None
        self.graph_weight: float | None = None
        self.regex_weight: float | None = None
        self.web_search: bool = False
        self.web_weight: float | None = None
        self.web_rerank_threshold: float | None = None
        self.fetch_page_content: bool = False
        self.web_fetch_k: int | None = None
        self.chat_name: str = self.cfg.get_str("_DEFAULT_CHAT_NAME")
        self.max_output_tokens: int | None = None
        self.max_output_tokens_override: int | None = None
        self.context_size_override: int | None = None
        self.temperature: float | None = None
        self.top_k: float | None = None
        self.top_p: float | None = None
        self.base_kwargs: dict[str, Any] | None = None
        self.collection_name: str | None = None
        self.retrieve_mode: str | None = "ALL"  # See _ALLOWED_RETRIEVE_MODES
        self.debug_level: int | None = None
        # Comparison mode for debug_level: "ge" = >= level (default), "is" = == level exactly
        self.debug_mode: str = "ge"
        # Extra Ollama options forwarded from API clients (e.g. OpenWebUI advanced params)
        self.extraOllamaOptions: dict[str, Any] | None = None
        # Top-level Ollama payload params forwarded from API clients (think, keep_alive, format)
        self.ollamaTopLevelParams: dict[str, Any] | None = None
        # One-shot flag set when the user prefixes their query with "new:" or
        # "new topic:" to signal a deliberate topic change.  When True,
        # PromptRewrite.rewrite() skips the LLM call for this turn only.
        # RAGChatImpl resets it to False at the start of every _retrieve() call.
        self.force_skip_rewrite: bool = False
        # Per-turn controls populated by Orchestrator flow selection.
        # None means "use global PromptRewrite defaults".
        self.enable_query_rewrite: bool | None = None
        self.enable_keyword_action_splitter: bool | None = None
        self.keyword_action_split: KeywordActionSplit | None = None
        # Query IDs are zero-based positions in split.retrieval_queries.
        # For ["Where do hedgehogs live?", "What do hedgehogs eat?"],
        # query ID 0 is the habitat query; query ID 1 is the diet query.
        # Keys are stable chunk identities from OrchestratorHelper.doc_identity.
        # Example: {"id:chunk-A": [0, 1]} means the chunk was found for both needs.
        self.keyword_action_query_ids: dict[str, list[int]] = {}
        # Raw cross-encoder scores per chunk AND need, rather than one score
        # against the entire compound request: {"id:chunk-A": {0: 4.2, 1: 0.3}}.
        # These associations do not reserve context slots or change selection.
        self.keyword_action_scores: dict[str, dict[int, float]] = {}
        # Each need's indexed query variants by document-language bucket.
        # Example: {0: {"de": "Lebensraum Igel", "en": "hedgehog habitat"}}.
        self.keyword_action_language_queries: dict[int, dict[str, str]] = {}
        self.keyword_action_status: str = "not_run"
        self.keyword_action_local_needs: int = 0
        self.keyword_action_keyword_leg: bool = False
        self.enable_pronoun_substitution: bool | None = None
        self.enable_guardrail_leg: bool | None = None
        self.enable_vector_alternate_queries: bool | None = None
        self.enable_rerank: bool | None = None
        self.enable_low_score_fallback: bool | None = None
        self.enable_low_recall_rescue: bool | None = None
        self.enable_grounding: bool | None = None
        self.enable_indexed_query_shaping: bool | None = None
        self.enable_indexed_query_translation: bool | None = None
        self.enable_indexed_query_wordnet_expansion: bool | None = None
        self.enable_original_language_vector_leg: bool | None = None
        self.enable_vector_retriever: bool | None = None
        self.enable_bm25_retriever: bool | None = None
        self.enable_graph_retriever: bool | None = None
        self.enable_regex_retriever: bool | None = None
        self.force_retrieve_mode: str | None = None
        self.main_query_source: str | None = None
        self.secondary_query_source: str | None = None
        self.rerank_query_source: str | None = None
        # Session override of the flow's retrieval_language (ENGLISH | NATIVE).
        # Separate from retrieval_language_mode, which holds the effective value.
        self.retrieval_language_override: str | None = None
        # Per-step confidence labels emitted by Orchestrator.
        self.orchestration_step_confidence: dict[str, str] = {}
        self.confidence_step_events: list[dict[str, Any]] = []
        self.lang_detection_events: list[dict[str, Any]] = []
        # Final answer confidence derived from selected retrieval evidence.
        self.answer_confidence_result: AnswerConfidenceResult | None = None
        self.rerank_low_confidence_fallback_triggered: bool = False
        self.rerank_partial_recall_rescue_triggered: bool = False
        # Preferred language for LLM responses, set interactively via /settings.
        self.preferred_response_language: str | None = None
        # Tracks the web_search state used in the previous query so RAGChatImpl
        # can auto-reset chat context when the mode is toggled mid-session.
        self.last_web_search: bool | None = None
        # Tracks the fetch_page_content state used in the previous query so
        # RAGChatImpl can auto-reset chat context when the mode is toggled.
        self.last_fetch_page_content: bool | None = None
        # Detected language (NLTK name) of the current user query, set by
        # RAGChatImpl after each language-detection call.  ChatContext uses it
        # to tag stored turns and filter fetched turns by language so that
        # German context never bleeds into an English query and vice-versa.
        self.current_query_lang: str | None = None
        # Language contract fields for the query-normalisation pipeline.
        # user_language: language detected on the raw user turn (response side)
        # retrieval_language: target language used for retrieval queries
        # orig_translated_query_en: raw user turn translated to retrieval language
        # rewritten_query: query text immediately after PromptRewrite
        # rewrite_language: detected language of rewritten_query
        # post_rewrite_query_en: final retrieval query after post-rewrite enforcement
        self.user_language: str | None = None
        self.retrieval_language: str | None = None
        # ENGLISH (legacy) | NATIVE; set per turn from the orchestration flow.
        self.retrieval_language_mode: str = "ENGLISH"
        self.orig_translated_query_en: str | None = None
        self.post_rewrite_query_en: str | None = None

        # Compatibility aliases.
        self.seed_retrieval_query: str | None = None
        self.final_retrieval_query: str | None = None
        # Query the cross-encoder scores against; None => use self.query.
        self.rerank_query: str | None = None
        self.t1_query: str | None = None
        self.rewritten_query: str | None = None
        self.rewrite_language: str | None = None
        self.t2_query: str | None = None

        # Retrieval guardrail counters: documents returned by seed-query
        # versus final-query legs before they are fused.
        self.retrieval_top_k_orig_query_en: int | None = None
        self.retrieval_top_k_post_rewrite_query_en: int | None = None

        # Compatibility aliases.
        self.retrieval_top_k_seed_query: int | None = None
        self.retrieval_top_k_final_query: int | None = None
        self.retrieval_top_k_before_t2: int | None = None
        self.retrieval_top_k_after_t2: int | None = None
        # Raw user query after topic-prefix stripping, captured before
        # translation/rewrite normalization.
        self.user_query_original: str | None = None
        # Original-language retrieval leg diagnostics.
        self.original_query_leg_enabled: bool = False
        self.original_query_leg_query: str | None = None
        self.original_query_leg_language: str | None = None
        self.original_query_leg_reason: str | None = None
        self.original_query_leg_vector_hits: int | None = None
        self.original_query_leg_vector_added: int | None = None
        self.original_query_leg_vector_overlap: int | None = None
        self.original_query_leg_bm25_hits: int | None = None
        self.original_query_leg_bm25_added: int | None = None
        self.original_query_leg_bm25_overlap: int | None = None
        self.original_query_leg_graph_hits: int | None = None
        self.original_query_leg_graph_added: int | None = None
        self.original_query_leg_graph_overlap: int | None = None
        self.original_query_leg_regex_hits: int | None = None
        self.original_query_leg_regex_added: int | None = None
        self.original_query_leg_regex_overlap: int | None = None
        # Set by RAGChatImpl to the effective (translated / rewritten) retrieval
        # query when it differs from the user's original input.  Chatter uses
        # this to show a notice at the top of the answer.
        self.effective_query: str | None = None
        # Why the query changed: "translated", "rewritten", or
        # "translated+rewritten".  Chatter uses this to pick the notice label.
        self.effective_query_reason: str | None = None
        # Set by PromptRewrite when it strips a pronoun from a depends=False /
        # referents=[] query.  Used by RetrievalGate as a secondary trigger.
        self.rewrite_was_underspecified: bool = False
        # Set by RetrievalGate when retrieval should be skipped.  Chatter reads
        # this and prints the clarification message instead of the LLM answer.
        self.clarification_response: str | None = None
        # When True, RAGChatImpl produces a copy of every retrieved local
        # PDF source with the chunk text highlighted, kept in memory only
        # (see `marked_documents`).  Chatter opens the highlighted PDF(s)
        # in the OS default viewer so the user can save via "Save As".
        # RAGChat: always True (saves locally). RAGChatService: True only
        # when SERVE_IN_MEMORY_DOCS_HTTP=1 (set in ChatCompletionHandler).
        self.mark_text: bool = self.cfg.get_str("_FRIENDLY_NAME") == "RAGChat"
        # Output of the visual-marker pass: list of (source_path,
        # highlighted_pdf_bytes) tuples for the current query.  Populated
        # by RAGChatImpl when mark_text is True; consumed by Chatter.
        self.marked_documents: list[tuple[str, bytes]] = []
        # Raw text of every chunk sent to the LLM for this query.  Populated
        # by RAGChatImpl after chunk selection so that Grounder can
        # match answer sentences back to source content.
        self.chunk_texts_for_grounding: list[str] = []
        # Last generated answer content for this query. Stored so that
        # grounded sentences can be extracted and marked in PDFs after
        # answer generation completes.
        self.last_answer_content: str | None = None
        # Translated grounding proxy, set when the answer language differs
        # from the evidence language and the translation was grounded.
        # Maps displayed answer sentences to their translations so CLI and
        # PDF highlights match what the evidence check accepted. The
        # displayed answer is never replaced.
        self.grounding_proxy_text: str = ""
        self.grounding_proxy_sentences: dict[str, str] = {}
        self.summary_grounding: bool = False
        self.summary_statements: list[SummaryStatement] = []
        # Last chosen chunks for this query. Stored so that orange grounding
        # can map grounded sentences back to their source documents.
        self.last_chosen_chunks: list[Any] = []

    def reset_turn_state(self) -> None:
        """Reset per-turn retrieval/rewrite telemetry and language contract fields."""
        self.force_skip_rewrite = False
        self.keyword_action_split = None
        self.keyword_action_query_ids = {}
        self.keyword_action_scores = {}
        self.keyword_action_language_queries = {}
        self.keyword_action_status = "not_run"
        self.keyword_action_local_needs = 0
        self.keyword_action_keyword_leg = False
        self.summary_grounding = False
        self.summary_statements = []
        self.effective_query = None
        self.effective_query_reason = None
        self.indexed_query_notices = []
        self.indexed_query_error = None
        self.indexed_query_warning_keys = set()
        self.orchestration_step_confidence = {}
        self.answer_confidence_result = None
        self.rerank_low_confidence_fallback_triggered = False
        self.user_language = None
        self.retrieval_language = "english"
        self.orig_translated_query_en = None
        self.seed_retrieval_query = None
        self.t1_query = None
        self.rewritten_query = None
        self.rewrite_language = None
        self.post_rewrite_query_en = None
        self.final_retrieval_query = None
        self.rerank_query = None
        self.t2_query = None
        self.retrieval_top_k_orig_query_en = None
        self.retrieval_top_k_post_rewrite_query_en = None
        self.retrieval_top_k_seed_query = None
        self.retrieval_top_k_final_query = None
        self.retrieval_top_k_before_t2 = None
        self.retrieval_top_k_after_t2 = None
        self.user_query_original = None
        self.original_query_leg_enabled = False
        self.original_query_leg_query = None
        self.original_query_leg_language = None
        self.original_query_leg_reason = None
        self.original_query_leg_vector_hits = None
        self.original_query_leg_vector_added = None
        self.original_query_leg_vector_overlap = None
        self.original_query_leg_bm25_hits = None
        self.original_query_leg_bm25_added = None
        self.original_query_leg_bm25_overlap = None
        self.original_query_leg_graph_hits = None
        self.original_query_leg_graph_added = None
        self.original_query_leg_graph_overlap = None
        self.original_query_leg_regex_hits = None
        self.original_query_leg_regex_added = None
        self.original_query_leg_regex_overlap = None
        self.rewrite_was_underspecified = False

    def export_session_state_as_cell(self, max_items_per_line: int = 6) -> str:
        """
        Return all session attributes as a single CSV-safe cell.
        No commas or delimiters are added. Optional line breaks
        keep the cell readable without breaking CSV structure.
        """

        # Collect only the attributes you explicitly defined
        keys = [
            "file_name",
            "summary_length",
            "summary_temperature",
            "chunk_overlap",
            "file_path",
            "view_doc",
            "query",
            "strategy",
            "retriever_k",
            "rerank",
            "final_chunks_to_llm",
            "chroma_threshold",
            "per_file_limit",
            "use_chat_context",
            "turns",
            "prune_batch",
            "max_history_turns",
            "topic_summary_mode",
            "vector_weight",
            "bm25_weight",
            "graph_weight",
            "regex_weight",
            "web_search",
            "web_weight",
            "fetch_page_content",
            "web_fetch_k",
            "chat_name",
            "max_output_tokens",
            "temperature",
            "base_kwargs",
            "collection_name",
            "retrieve_mode",
            "debug",
            "extraOllamaOptions",
            "ollamaTopLevelParams",
        ]

        # Build "key=value" pairs
        items: list[str] = []
        for key in keys:
            value = getattr(self, key, None)
            items.append(f"{key}={value}")

        # Insert line breaks every N items
        lines: list[str] = []
        for i in range(0, len(items), max_items_per_line):
            chunk = " | ".join(items[i : i + max_items_per_line])
            lines.append(chunk)

        # Final output: one single CSV cell
        return "\n".join(lines)
