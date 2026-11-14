"""Command and settings metadata shared by the query CLI."""

from typing import Any

from Orchestrator import FlowConfig

COMMAND_SPECS: dict[str, dict[str, Any]] = {
    "summary_length": {
        "section": "Summarize settings",
        "prompt": "Summary detail: short, medium, long, very_long. Applies to every summary stage.",
        "mode": "summary_length",
        "type": "str",
        "attr": "summary_length",
    },
    "summary_temperature": {
        "section": "Summarize settings",
        "prompt": "Shared temperature for all summary stages, from 0 to 1.",
        "mode": "normal",
        "type": "float",
        "attr": "summary_temperature",
    },
    "chunk_overlap": {
        "section": "Summarize settings",
        "prompt": "Non-negative number of whole chunks to overlap between summary batches.",
        "mode": "normal",
        "type": "int",
        "attr": "chunk_overlap",
    },
    "collection": {
        "section": "Talk with",
        "prompt": (
            "Selects the active knowledge base or document collection. "
            "This determines which embeddings, files, and Chat Context are used for retrieval. "
            "Changing the collection resets file selections and context."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "collection_name",
    },
    "chat_name": {
        "section": "Talk with",
        "prompt": (
            "Sets the chat_name used for storing and retrieving Chat Context. "
            "Useful when multiple people share the same environment or when you want isolated histories."
        ),
        "mode": "chat_name",
        "type": "str",
        "attr": "chat_name",
    },
    "strategy": {
        "section": "Retrieval",
        "prompt": (
            "Chooses a predefined retrieval configuration (narrow, balanced_file_cap, wide, default). "
            "Each strategy sets defaults for chunk selection, thresholds, and sampling parameters. "
            "Use this when you want consistent behavior without manually tuning every value."
        ),
        "mode": "strategy",
        "type": "string",
        "attr": "strategy",
    },
    "orchestrator_flow": {
        "section": "Retrieval",
        "prompt": (
            "Chooses a predefined retrieval-orchestration flow profile. "
            "Flow profiles are configured in Config_Orchestrator.py and can enable/disable "
            "query sources, guardrail legs, retriever switches, and shaping behavior. "
            "Local/web stage execution is controlled by web_search/retrieve_mode. "
            "Use orchestrator_flow! to pick one of the predefined profiles."
        ),
        "mode": "orchestrator_flow",
        "type": "string",
        "attr": "orchestrator_flow",
    },
    "confidence_evaluation": {
        "section": "Confidence",
        "prompt": (
            "Retrieval diagnostics / answer-text review: off, heuristic, llm, or both. "
            "Both shows separate sections without averaging. "
            "Unset inherits the orchestration flow. Baseline HIGH/MEDIUM/LOW "
            "confidence remains visible even when evaluation is off."
        ),
        "mode": "confidence_evaluation",
        "type": "str",
        "attr": "confidence_evaluation",
    },
    "force_retrieve_mode": {
        "section": "Retrieval",
        "prompt": (
            "Optional retriever-mode override (VECTOR/BM25/GRAPH/REGEX/VECTOR_BM25/.../ALL/WEB). "
            "Overrides the flow's force_retrieve_mode for this session. A local mode such as "
            "BM25 or VECTOR_BM25 limits the local retrievers to that mode (on top of the "
            "flow's run_* switches); ALL keeps the flow's retrievers; WEB skips local retrieval."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "force_retrieve_mode",
    },
    "main_query_source": {
        "section": "Retrieval",
        "prompt": (
            "Primary retrieval query source selector. "
            "Allowed values: REWRITTEN_QUERY | TRANSLATED_QUERY | ORIGINAL_QUERY."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "main_query_source",
    },
    "secondary_query_source": {
        "section": "Retrieval",
        "prompt": (
            "Secondary retrieval query source selector. "
            "Allowed values: REWRITTEN_QUERY | TRANSLATED_QUERY | ORIGINAL_QUERY."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "secondary_query_source",
    },
    "rerank_query_source": {
        "section": "Retrieval",
        "prompt": (
            "Query version the cross-encoder reranker scores against. "
            "Allowed values: REWRITTEN_QUERY | TRANSLATED_QUERY | ORIGINAL_QUERY "
            "(ORIGINAL_QUERY only with a multilingual reranker)."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "rerank_query_source",
    },
    "retrieval_language": {
        "section": "Retrieval",
        "prompt": (
            "Retrieval language: english (translate the question to English before "
            "rewrite/retrieval) or native (keep the user's language; needs a "
            "multilingual embedder and reranker). Overrides the flow for this session."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "retrieval_language_override",
    },
    "rerank": {
        "section": "Retrieval",
        "prompt": (
            "Enables or disables the reranker (0 = off, 1 = on). "
            "Flow profile run_rerank provides the baseline; this session switch overrides it. "
            "When enabled, retrieved chunks are re\u2011scored using a more accurate cross\u2011encoder. "
            "This improves relevance at the cost of additional computation."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "rerank",
    },
    "vector_weight": {
        "section": "Retrieval",
        "prompt": (
            "Weight applied to vector (embedding) retrieval scores before RRF fusion. "
            "Set to 0.0 to effectively disable vector retrieval in the merged ranking. "
            "Relative to bm25_weight, graph_weight, and regex_weight."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "vector_weight",
    },
    "bm25_weight": {
        "section": "Retrieval",
        "prompt": (
            "Weight applied to BM25 keyword retrieval scores before RRF fusion. "
            "Set to 0.0 to effectively disable BM25 in the merged ranking. "
            "Relative to vector_weight, graph_weight, and regex_weight."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "bm25_weight",
    },
    "graph_weight": {
        "section": "Retrieval",
        "prompt": (
            "Weight applied to graph (entity co-occurrence) retrieval scores before RRF fusion. "
            "Set to 0.0 to effectively disable graph retrieval in the merged ranking. "
            "Relative to vector_weight, bm25_weight, and regex_weight."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "graph_weight",
    },
    "regex_weight": {
        "section": "Retrieval",
        "prompt": (
            "Weight applied to regex (content verb/noun) retrieval scores before RRF fusion. "
            "Set to 0.0 to effectively disable regex retrieval in the merged ranking. "
            "Relative to vector_weight, bm25_weight, and graph_weight."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "regex_weight",
    },
    "web_search": {
        "section": "Retrieval",
        "prompt": (
            "Control web search mode. "
            "local_only = use only local indexes. "
            "local_and_web = query the web via DuckDuckGo and merge results with local retrieval. "
            "web_only = skip all local indexes; use web search only. "
            "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (web_search=local_and_web)."
        ),
        "mode": "web_search",
        "type": "str",
        "attr": "web_search",
    },
    "web_weight": {
        "section": "Retrieval",
        "prompt": (
            "Weight applied to web search results before RRF fusion. "
            "Set to 0.0 to exclude web results from the merged ranking. "
            "Relative to vector_weight, bm25_weight, graph_weight, and regex_weight. "
            "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (e.g. web_weight=0.5)."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "web_weight",
    },
    "fetch_page_content": {
        "section": "Retrieval",
        "prompt": (
            "Controls how web results are used. "
            "'snippets only' = use the search snippet as the document text (fast, default). "
            "'fetch pages' = fetch the full page and extract main text (slower, more content). "
            "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (fetch_page_content=true)."
        ),
        "mode": "fetch_page_content",
        "type": "str",
        "attr": "fetch_page_content",
    },
    "web_fetch_k": {
        "section": "Retrieval",
        "prompt": (
            "Number of web documents fetched per query when web search is enabled. "
            "Overrides _WEB_SEARCH.max_results for this session only. "
            "Can be set permanently for a model via OpenWebUI → Admin → Models → "
            "Advanced Parameters (web_fetch_k=10)."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "web_fetch_k",
    },
    "web_rerank_threshold": {
        "section": "Retrieval",
        "prompt": (
            "Minimum cross-encoder score a web result must reach after reranking. "
            "0.0 = no additional filtering (default \u2014 web results are already "
            "quality-gated by bm25_pre_filter and cosine_pre_filter). "
            "Raise for stricter post-rerank filtering (e.g. web_rerank_threshold=0.05). "
            "Unlike threshold, this applies only to web-sourced chunks."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "web_rerank_threshold",
    },
    "fetch_k": {
        "section": "Retrieval",
        "prompt": (
            "Defines how many chunks each retriever (vector, BM25, graph, regex) fetches before filtering and fusion. "
            "Higher values increase recall but may introduce noise. "
            "Lower values improve precision but risk missing relevant context."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "retriever_k",
    },
    "context_chunks": {
        "section": "Retrieval",
        "prompt": (
            "Sets the maximum number of chunks kept after all filters and reranking. "
            "This controls how much context the model sees. "
            "A larger value helps with broad questions; a smaller one keeps responses focused."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "final_chunks_to_llm",
    },
    "threshold": {
        "section": "Retrieval",
        "prompt": (
            "Minimum similarity score a chunk must meet to be considered relevant. "
            "Raising the threshold increases precision but may drop borderline‑useful chunks. "
            "Lowering it increases recall but risks including irrelevant text."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "chroma_threshold",
    },
    "max_output_tokens": {
        "section": "Retrieval",
        "prompt": (
            "Cap the number of output tokens the model may generate in its reply. "
            "Leave unset to use the fully dynamic budget. "
            "A warning is shown if the value exceeds the computed budget; "
            "Ollama will then receive the dynamic value instead "
            "(e.g. max_output_tokens=512 for short answers)."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "max_output_tokens_override",
    },
    "context_size": {
        "section": "Retrieval",
        "prompt": (
            "KV-cache / context window size sent to Ollama as num_ctx. "
            "Leave unset to use the auto-detected model limit (capped by TOKEN_BUDGET_CONTEXT_CAP). "
            "Setting this higher than the hardware cap is ignored with a warning. "
            "Use a lower value to reduce VRAM usage on constrained hardware "
            "(e.g. context_size=4096)."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "context_size_override",
    },
    "terminal_line_size": {
        "section": "Retrieval",
        "prompt": (
            "Width (in characters) used for wrapping terminal output. "
            "Increase for wide terminals to reduce line wrapping in debug tables; "
            "decrease for narrow windows (e.g. terminal_line_size=120)."
        ),
        "mode": "terminal_line_size",
        "type": "int",
    },
    "temperature": {
        "section": "Retrieval",
        "prompt": (
            "Controls randomness in the model’s generation. "
            "Lower values produce deterministic, factual answers; higher values increase creativity and variation. "
            "Use low temperature for compliance, code, and precise Q&A; use higher values for brainstorming."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "temperature",
    },
    "top_p": {
        "section": "Retrieval",
        "prompt": (
            "Nucleus sampling threshold that limits token selection to the smallest set whose cumulative probability reaches p. "
            "Lower values restrict generation to highly probable tokens; higher values allow more diverse outputs. "
            "This is a softer, more adaptive alternative to top_k."
        ),
        "mode": "normal",
        "type": "float",
        "attr": "top_p",
    },
    "top_k": {
        "section": "Retrieval",
        "prompt": (
            "Limits generation to the top k most likely next tokens. "
            "Smaller values reduce noise and hallucination risk; larger values increase creativity. "
            "Use this together with top_p to shape the model’s output behavior."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "top_k",
    },
    "file": {
        "section": "File Input",
        "prompt": (
            "Selects a specific file from the active collection to query. "
            "Useful when you want to restrict retrieval to a single document. "
            "The file picker helps you avoid typos and path issues."
        ),
        "mode": "file",
        "type": "string",
        "attr": "file_name",
    },
    "path": {
        "section": "File Input",
        "prompt": (
            "Same as 'file', but allows selecting by full path. "
            "Useful when working with external or newly added files."
        ),
        "mode": "path",
        "type": "string",
        "attr": "path_name",
    },
    "metadata": {
        "section": "File Input",
        "prompt": (
            "Filters retrieval to chunks whose metadata matches selected "
            "field=value pairs (author, dates, page label, file size, …). "
            "metadata! opens a picker to select multiple fields/values; "
            "metadata=Field:Value assigns one directly; metadata- clears all."
        ),
        "mode": "metadata",
        "type": "string",
        "attr": "metadata_filters",
    },
    "file_cap": {
        "section": "File Input",
        "prompt": (
            "Sets the maximum number of chunks allowed per file (balanced_file_cap / default strategy). "
            "Prevents large files from dominating retrieval. "
            "Helps maintain balanced context across multiple documents."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "per_file_limit",
    },
    "use_chat_context": {
        "section": "Chat Context",
        "prompt": (
            "Enables or disables retrieval from previous chat messages. "
            "When enabled, the system treats past conversation turns as additional context chunks. "
            "Useful for multi‑turn reasoning or long workflows."
        ),
        "mode": "normal",
        "type": "bool",
        "attr": "use_chat_context",
    },
    "history_keep": {
        "section": "Chat Context",
        "prompt": (
            "Pruning threshold: maximum number of context entries kept per chat. "
            "When the count exceeds this value, the oldest `history_prune` entries "
            "are summarized into one entry and removed."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "turns",
    },
    "history_prune": {
        "section": "Chat Context",
        "prompt": (
            "Pruning granularity: number of oldest entries compressed into a "
            "single summary per pruning pass. Smaller values preserve more "
            "fine-grained history; larger values compress more aggressively."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "prune_batch",
    },
    "rewrite_context": {
        "section": "Chat Context",
        "prompt": (
            "Number of most recent chat turns sent to the query rewriter. "
            "Lower values keep the rewrite focused on the latest exchange; "
            "higher values give the rewriter more conversational context. "
            "0 disables the limit (all fetched turns are used)."
        ),
        "mode": "normal",
        "type": "int",
        "attr": "max_history_turns",
    },
    "topic_summary": {
        "section": "Chat Context",
        "prompt": (
            "Rolling topic summary mode for the query rewriter. "
            "'last' = use only the most recent ASSISTANT response as topic context; "
            "'all' = concatenate all ASSISTANT responses in the history window. "
            "'last' is lower cost and avoids stale context; 'all' gives the rewriter "
            "the full topic arc across multiple turns."
        ),
        "mode": "normal",
        "type": "str",
        "attr": "topic_summary_mode",
    },
    "debug_level": {
        "section": "Chat Context",
        "prompt": (
            "Sets the verbosity of debug output. "
            "Accepts a plain level number (e.g. '30') for >= mode, "
            "'ge 30' for explicit >= mode, 'is 30' for exact-match mode, "
            "or 'le 30' for <= mode (this level and all below). "
            "Higher levels show internal decisions, scoring, and filtering steps. "
            "Useful for diagnosing retrieval behavior or teaching how the pipeline works."
        ),
        "mode": "debug",
        "type": "debug_level",
        "attr": "debug_level",
    },
    "debug_mode": {
        "section": "Chat Context",
        "prompt": (
            "Sets the comparison mode for debug_level. "
            "'ge' activates the selected level and all below it (>=). "
            "'is' activates that exact level only (==). "
            "'le' activates that level and all above it (<=)."
        ),
        "mode": "debug_mode",
        "type": "str",
        "attr": "debug_mode",
    },
    "help": {
        "section": "General",
        "prompt": (
            "Displays detailed help for all available commands. "
            "Use help? to show this information at any time."
        ),
        "mode": "help",
    },
    "mark_text": {
        "section": "Visual",
        "prompt": (
            "When enabled, every retrieved local PDF source is highlighted "
            "in memory and opened from a short-lived temp file (auto-deleted "
            "on exit). The original files are not modified. "
            "Currently supports PDF only; other document types are "
            "skipped silently."
        ),
        "mode": "normal",
        "type": "bool",
        "attr": "mark_text",
    },
}

_FLOW_SWITCH_TO_SESSION_ATTR: dict[str, str] = dict(FlowConfig.FLOW_SWITCH_KEY_TO_FIELD)

_FRIENDLY_TOKEN_MAP: dict[str, str] = {
    "treshold": "threshold",
    "flow_name": "orchestrator_flow",
    "secondary_query": "enable_guardrail_leg",
    "per_language_indexed_query": "enable_indexed_query_shaping",
    "indexed_translate": "enable_indexed_query_translation",
    "indexed_wordnet": "enable_indexed_query_wordnet_expansion",
    "indexed_synonyms": "enable_indexed_query_wordnet_expansion",
    "orig_lang_vector": "enable_original_language_vector_leg",
    "vector_alternates": "enable_vector_alternate_queries",
    "vector": "enable_vector_retriever",
    "bm25": "enable_bm25_retriever",
    "graph": "enable_graph_retriever",
    "regex": "enable_regex_retriever",
    "query_rewrite": "enable_query_rewrite",
    "keyword_action_splitter": "enable_keyword_action_splitter",
    "pronoun_subst": "enable_pronoun_substitution",
    "rerank_stage": "enable_rerank",
    "low_score_fallback": "enable_low_score_fallback",
    "low_recall_rescue": "enable_low_recall_rescue",
    "grounding": "enable_grounding",
}

_PARSER_ONLY_TOKENS: list[str] = ["show"]

EXTRA_TOKENS: list[str] = ["context"]

_ASK_DIRECT_VALUE_TOKENS: set[str] = {
    "summary_temperature",
    "chunk_overlap",
    "confidence_evaluation",
    "fetch_k",
    "window",
    "threshold",
    "max_output_tokens",
    "context_size",
    "temperature",
    "top_p",
    "top_k",
    "file_cap",
    "rerank",
    "vector_weight",
    "bm25_weight",
    "graph_weight",
    "regex_weight",
    "web_weight",
    "web_fetch_k",
    "web_rerank_threshold",
    "use_chat_context",
    "history_keep",
    "history_prune",
    "rewrite_context",
    "force_retrieve_mode",
    "main_query_source",
    "secondary_query_source",
    "rerank_query_source",
    "retrieval_language",
    "mark_text",
}

_SET_DIRECT_VALUE_TOKENS: set[str] = {
    "summary_length",
    "summary_temperature",
    "chunk_overlap",
    "confidence_evaluation",
    "chat_name",
    "fetch_k",
    "context_chunks",
    "threshold",
    "max_output_tokens",
    "context_size",
    "temperature",
    "top_p",
    "top_k",
    "file_cap",
    "rerank",
    "vector_weight",
    "bm25_weight",
    "graph_weight",
    "regex_weight",
    "web_search",
    "web_weight",
    "web_fetch_k",
    "web_rerank_threshold",
    "fetch_page_content",
    "use_chat_context",
    "history_keep",
    "history_prune",
    "rewrite_context",
    "orchestrator_flow",
    "force_retrieve_mode",
    "main_query_source",
    "secondary_query_source",
    "rerank_query_source",
    "retrieval_language",
    "mark_text",
}

_SETTINGS_LAYOUT: dict[str, dict[str, tuple[str, ...]]] = {
    "Chat settings": {
        "Debug": ("debug_level", "debug_mode"),
        "Chat Context": (
            "use_chat_context",
            "history_keep",
            "history_prune",
            "rewrite_context",
            "topic_summary",
        ),
        "Talk with": ("collection", "chat_name"),
        "File Input": ("file", "path", "metadata", "file_cap"),
        "Visual": ("mark_text",),
        "Response": ("preferred_response_language",),
        "Commands": ("help",),
    },
    "Summarize settings": {
        "Summary": ("summary_length", "summary_temperature", "chunk_overlap"),
    },
    "Retrieval settings": {
        "Retrieval strategy": ("strategy",),
        "Strategy overrides": (),
        "LLM": ("temperature", "top_p", "top_k"),
        "Retriever tuning": ("fetch_k", "context_chunks", "threshold"),
        "Retriever weights": (
            "vector_weight",
            "bm25_weight",
            "graph_weight",
            "regex_weight",
            "web_weight",
        ),
        "Context/session": (
            "file_cap",
            "use_chat_context",
            "history_keep",
            "history_prune",
            "rewrite_context",
            "topic_summary",
        ),
        "Output": ("max_output_tokens", "context_size", "terminal_line_size"),
        "API passthrough": ("extra_ollama_options", "ollama_top_level_params"),
        "Web": (
            "web_search",
            "fetch_page_content",
            "web_fetch_k",
            "bm25_pre_filter",
            "cosine_pre_filter",
            "web_rerank_threshold",
        ),
    },
    "Orchestration settings": {
        "Orchestration strategy": ("orchestrator_flow",),
        "Orchestrator overrides": (),
        "Confidence": ("confidence_evaluation",),
        "Flow selectors": (
            "force_retrieve_mode",
            "main_query_source",
            "secondary_query_source",
            "rerank_query_source",
            "retrieval_language",
            "rerank",
        ),
        "Flow stages": (
            "secondary_query",
            "per_language_indexed_query",
            "indexed_translate",
            "indexed_wordnet",
            "orig_lang_vector",
            "vector_alternates",
        ),
        "Flow retrievers": ("vector", "bm25", "graph", "regex"),
        "Flow runtime": (
            "query_rewrite",
            "keyword_action_splitter",
            "pronoun_subst",
            "rerank_stage",
            "low_score_fallback",
            "low_recall_rescue",
            "grounding",
        ),
    },
}
