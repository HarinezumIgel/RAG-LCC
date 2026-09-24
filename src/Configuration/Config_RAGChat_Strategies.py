# -----------------------------------------------------------------------------
# Chunk selection strategy
# -----------------------------------------------------------------------------
_ALLOWED_STRATEGIES = ["ULTRA_WIDE", "WIDE", "BALANCED_FILE_CAP", "NARROW", "DEFAULT"]
_ACTIVE_CHUNK_SELECT_STRATEGY = "DEFAULT"  # Default: pick one of _ALLOWED_STRATEGIES
_ALLOWED_RETRIEVE_MODES = [
    "VECTOR",  # Embedding-based retrieval only
    "BM25",  # Keyword-based retrieval only
    "GRAPH",  # Entity co-occurrence graph only
    "REGEX",  # Verb/noun regex-style retrieval only
    "VECTOR_BM25",  # Vector + BM25 fused via RRF
    "VECTOR_GRAPH",  # Vector + graph fused via RRF
    "BM25_GRAPH",  # BM25 + graph fused via RRF
    "VECTOR_REGEX",  # Vector + regex fused via RRF
    "BM25_REGEX",  # BM25 + regex fused via RRF
    "GRAPH_REGEX",  # Graph + regex fused via RRF
    "ALL",  # Vector + BM25 + graph + regex fused via RRF (all local retrieval algorithms)
    "WEB",  # Web search only; skips all local indexes (requires web_search enabled)
]  # Retrieval mode

_STRATEGIES: dict[str, dict[str, int | float | bool | str]] = {
    "NARROW": {  # Precision-oriented — only strong semantic matches
        "final_chunks_to_llm": 20,  # Max chunks after selection — small for focused answers
        "retriever_k": 80,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.60,  # cross-encoder confidence floor; sigmoid(logit) ≥ 0.60 needs a clearly positive logit (precision). If no chunk clears it, rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 8192,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 20,  # Narrow token sampling — focused word choices
        "top_p": 0.8,  # Tight nucleus sampling — conservative candidate pool
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 0,  # Graph disabled — SingleDocumentSelector discards cross-file results anyway
        "regex_weight": 1,  # RRF weight for regex retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 5,  # Max 5 chunks per file — prevents large files filling all 20 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 5,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode (see _ALLOWED_RETRIEVE_MODES)
    },
    "BALANCED_FILE_CAP": {  # Balanced precision / recall with per-file chunk cap
        "final_chunks_to_llm": 40,  # Moderate selection window
        "retriever_k": 60,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.55,  # cross-encoder confidence floor; sigmoid(logit) ≥ 0.55 (slightly positive logit). If no chunk clears it, rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 40,  # Moderate token sampling — some variety
        "top_p": 0.92,  # Moderate nucleus sampling
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "regex_weight": 1,  # RRF weight for regex retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 10,  # Max 10 chunks per file — enforces actual diversity across files
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode (see _ALLOWED_RETRIEVE_MODES)
    },
    "DEFAULT": {  # General-purpose balanced retrieval
        "final_chunks_to_llm": 50,  # Moderate selection window
        "retriever_k": 100,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.50,  # cross-encoder confidence floor at the neutral logit (sigmoid(0)=0.50). If no chunk clears it the rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 40,  # Moderate token sampling — some variety
        "top_p": 0.92,  # Moderate nucleus sampling
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "regex_weight": 1,  # RRF weight for regex retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 15,  # Max 15 chunks per file — prevents large docs filling all 40 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode (see _ALLOWED_RETRIEVE_MODES)
    },
    "WIDE": {  # Recall-oriented — exploratory search across many chunks
        "final_chunks_to_llm": 60,  # Large selection window — more context for the LLM
        "retriever_k": 160,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.50,  # cross-encoder confidence floor at the neutral logit (sigmoid(0)=0.50); favors recall. If no chunk clears it the rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 100,  # Broad token sampling — 100 candidates per step
        "top_p": 0.97,  # Wide nucleus sampling — most of the probability mass
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "regex_weight": 1,  # RRF weight for regex retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 20,  # Loose cap — breadth allowed but no file takes more than 20 of 60 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode (see _ALLOWED_RETRIEVE_MODES)
    },
    "ULTRA_WIDE": {  # Diagnostic / exploratory — very high recall, high cost
        "final_chunks_to_llm": 1500,  # Very large selection window (high computational cost)
        "retriever_k": 3000,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.45,  # cross-encoder confidence floor below the neutral logit; keeps weak (slightly negative logit) matches and rarely falls back to retrieval order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 100,  # Broad token sampling — 100 candidates per step
        "top_p": 0.97,  # Wide nucleus sampling — most of the probability mass
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "regex_weight": 1,  # RRF weight for regex retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 0,  # No per-file chunk limit
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode (see _ALLOWED_RETRIEVE_MODES)
    },
}

# -----------------------------------------------------------------------------
# Prompt template for chat responses
# -----------------------------------------------------------------------------
