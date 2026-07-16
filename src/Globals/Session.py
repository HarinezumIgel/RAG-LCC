# Local module imports
from typing import Any, Optional

from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter


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

        self.file_name: Optional[str | None] = None
        self.file_path: Optional[str | None] = None
        self.file_path_select: Optional[str | None] = (
            None  # Internal flag only, will not be shown to user
        )
        self.query: str | None = None
        self.strategy: str | None = None
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
        self.web_search: bool = False
        self.web_weight: float | None = None
        self.web_rerank_threshold: float | None = None
        self.fetch_page_content: bool = False
        self.chat_name: str = self.cfg.get_str("_DEFAULT_CHAT_NAME")
        self.max_output_tokens: int | None = None
        self.max_output_tokens_override: int | None = None
        self.context_size_override: int | None = None
        self.temperature: float | None = None
        self.top_k: float | None = None
        self.top_p: float | None = None
        self.base_kwargs: dict[str, Any] | None = None
        self.collection_name: str | None = None
        self.retrieve_mode: str | None = "HYBRID"  # VECTOR, BM25, HYBRID (default)
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
        # by RAGChatImpl after chunk selection so that AnswerGrounder can
        # match answer sentences back to source content.
        self.chunk_texts_for_grounding: list[str] = []
        # Last generated answer content for this query. Stored so that
        # grounded sentences can be extracted and marked in PDFs after
        # answer generation completes.
        self.last_answer_content: str | None = None
        # Last chosen chunks for this query. Stored so that orange grounding
        # can map grounded sentences back to their source documents.
        self.last_chosen_chunks: list[Any] = []

    def export_session_state_as_cell(self, max_items_per_line: int = 6) -> str:
        """
        Return all session attributes as a single CSV-safe cell.
        No commas or delimiters are added. Optional line breaks
        keep the cell readable without breaking CSV structure.
        """

        # Collect only the attributes you explicitly defined
        keys = [
            "file_name",
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
            "web_search",
            "web_weight",
            "fetch_page_content",
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
