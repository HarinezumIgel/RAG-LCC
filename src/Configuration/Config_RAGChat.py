# -------------------------------------------------------------------------
# - Lookup order: Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#   Config_Banned, Config_Models.py, Config_Globals.py
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# Do not change _FRIENDLY_NAME
_FRIENDLY_NAME = "RAGChat"

# -----------------------------------------------------------------------------
# Chunk selection strategy
# -----------------------------------------------------------------------------
_ALLOWED_STRATEGIES = ["ULTRA_WIDE", "WIDE", "MEDIUM", "NARROW"]
CHUNK_SELECT_STRATEGY = "MEDIUM"  # Default: pick one of _ALLOWED_STRATEGIES

# -----------------------------------------------------------------------------
# Keyword extraction
# -----------------------------------------------------------------------------
_KEY_BERT = {
    "TOP_N_FIRST": 100,  # Keywords from first  KeyBERT pass
    "TOP_N_SECOND": 60,  # Keywords from second KeyBERT pass
}
# -----------------------------------------------------------------------------
# Chat history and user defaults
# -----------------------------------------------------------------------------
_HISTORY_DIRECTORY = r"history"  # Where to store chat histories and metadata

# Keys to extract
_CLASSIFICATION_KEYS = []

_DEFAULT_CHAT_NAME = "MyFirstChat"  # Fallback user identifier

# -----------------------------------------------------------------------------
# Retrieval-Augmented Generation strategy profiles
# -----------------------------------------------------------------------------
_STRATEGIES: dict[str, dict[str, int | float | bool | str]] = {
    "NARROW": {  # Precision-oriented retrieval profile (favoring strong semantic matches)
        "chunks_window": 20,  # Smaller context expansion around each retrieved chunk
        "chroma_k_value": 160,  # Number of vector candidates retrieved prior to reranking
        "threshold": 0.75,  # Reranker score cutoff used to retain chunks
        "max_output_tokens": 8192,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low-variance sampling configuration
        "top_k": 20,  # Limits token sampling breadth
        "top_p": 0.8,  # Nucleus sampling probability threshold
        "rerank": 1,  # Enables cross-encoder reranking
        "chroma_weight": 0.6,  # Relative weighting of vector similarity vs. reranker score
        "filelim": 0,  # No explicit limit on contributing files
        "chat_context_k_value": 5,  # Number of prior chat turns considered (if enabled)
        "use_chat_context": False,  # Excludes prior chat turns from retrieval
        "turns": 5,  # Maximum chat history depth (if enabled)
        "batch_size": 5,  # Number of parallel retrieval batches
        "debug_level": 3,  # Debug logging verbosity
    },
    "MEDIUM": {  # Balanced retrieval profile (precision / recall trade-off)
        "chunks_window": 40,  # Moderate context expansion around each retrieved chunk
        "chroma_k_value": 60,  # Moderate-size candidate pool for reranking
        "threshold": 0.55,  # Reranker score cutoff balancing relevance and context
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low-variance sampling configuration
        "top_k": 40,  # Moderately broad token sampling
        "top_p": 0.92,  # Moderately broad nucleus sampling
        "rerank": 1,  # Enables cross-encoder reranking
        "chroma_weight": 0.6,  # Relative weighting of vector similarity vs. reranker score
        "filelim": 4,  # Limits the number of contributing files
        "chat_context_k_value": 5,  # Number of prior chat turns considered (if enabled)
        "use_chat_context": False,  # Excludes prior chat turns from retrieval
        "turns": 10,  # Maximum chat history depth (if enabled)
        "batch_size": 5,  # Number of parallel retrieval batches
        "debug_level": 3,  # Debug logging verbosity
    },
    "WIDE": {  # Recall-oriented retrieval profile (exploratory search)
        "chunks_window": 100,  # Larger context expansion around each retrieved chunk
        "chroma_k_value": 160,  # Larger candidate pool for reranking
        "threshold": 0.40,  # Lower reranker score cutoff to favor recall
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low-variance sampling configuration
        "top_k": 100,  # Broad token sampling
        "top_p": 0.97,  # Broad nucleus sampling
        "rerank": 1,  # Enables cross-encoder reranking
        "chroma_weight": 0.6,  # Relative weighting of vector similarity vs. reranker score
        "filelim": 0,  # No explicit limit on contributing files
        "chat_context_k_value": 5,  # Number of prior chat turns considered (if enabled)
        "use_chat_context": False,  # Excludes prior chat turns from retrieval
        "turns": 10,  # Maximum chat history depth (if enabled)
        "batch_size": 5,  # Number of parallel retrieval batches
        "debug_level": 3,  # Debug logging verbosity
    },
    "ULTRA_WIDE": {  # Diagnostic / exploratory retrieval profile with very high recall
        "chunks_window": 1500,  # Very large context expansion (high computational cost)
        "chroma_k_value": 3000,  # Very large vector candidate pool
        "threshold": 0.20,  # Permissive reranker score cutoff
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low-variance sampling configuration
        "top_k": 100,  # Broad token sampling
        "top_p": 0.97,  # Broad nucleus sampling
        "rerank": 1,  # Enables cross-encoder reranking
        "chroma_weight": 0.6,  # Relative weighting of vector similarity vs. reranker score
        "filelim": 0,  # No explicit limit on contributing files
        "chat_context_k_value": 5,  # Number of prior chat turns considered (if enabled)
        "use_chat_context": False,  # Excludes prior chat turns from retrieval
        "turns": 10,  # Maximum chat history depth (if enabled)
        "batch_size": 5,  # Number of parallel retrieval batches
        "debug_level": 3,  # Debug logging verbosity
    },
}

# -----------------------------------------------------------------------------
# Prompt template for chat responses
# -----------------------------------------------------------------------------
_PROMPT_CHAT = """
CRITICAL: You must ONLY use information found in the context below.
Do NOT use your training knowledge, do NOT guess, do NOT infer beyond what the context states.
If the context is empty, incomplete, or irrelevant to the query, respond EXACTLY:
  I couldn't find relevant information to answer your query.
Then stop — do not add anything else.

Context:
---------------------
{context}
---------------------

Before answering, verify:
- Is the context above non-empty?
- Does it contain information relevant to the query?
If EITHER check fails, respond EXACTLY as instructed above.

If you do find an answer:
  • Cite direct evidence from the context.
  • At the end, list only the metadata fields you used:
     - FileName
     - FilePath
     - PageNumber (if available)
  • Pretty-format your output using Markdown.

Query:
{input}

Answer:
"""

# Legacy aliases – both point to the unified prompt above
_PROMPT_CHAT_MISTRAL = _PROMPT_CHAT
_PROMPT_CHAT_LLAMA = _PROMPT_CHAT
