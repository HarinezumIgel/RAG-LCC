# -------------------------------------------------------------------------
# RAGChatService is RAGChat served over HTTP (OpenWebUI-compatible REST API).
# All RAGChat configuration is reused as-is — only the API host/port are added.
#
# _FRIENDLY_NAME = "RAGChatService" matches the alias added in Config_Banned.py
# so compliance lookups resolve to the correct RAGChatService pipeline.
# -------------------------------------------------------------------------

# import * silently drops _-prefixed names — re-import them explicitly:
# Re-export the full RAGChat config so the Config loader finds every key
# (_KEY_BERT, _STRATEGIES, _ALLOWED_STRATEGIES, _HISTORY_DIRECTORY, etc.)
from Configuration.Config_RAGChat import *  # noqa: F401, F403
from Configuration.Config_RAGChat import (  # pyright: ignore[reportPrivateUsage, reportUnusedImport]; pyright: ignore[reportPrivateUsage, reportUnusedImport, reportUnknownVariableType]; noqa: F401; pyright: ignore[reportPrivateUsage, reportUnusedImport]
    _ALLOWED_STRATEGIES, _CLASSIFICATION_KEYS, _DEFAULT_CHAT_NAME,
    _HISTORY_DIRECTORY, _KEY_BERT, _STRATEGIES)

_FRIENDLY_NAME = (
    "RAGChatService"  # Matches the alias in _BANNED_DETECT (Config_Banned.py)
)

# -----------------------------------------------------------------------------
# Override prompt: RAGChatService gets OpenWebUI-specific hint
# -----------------------------------------------------------------------------
_PROMPT_CHAT = """
CRITICAL: You must ONLY use information found in the context below.
Do NOT use your training knowledge, do NOT guess, do NOT infer beyond what the context states.
If the context is empty, incomplete, or irrelevant to the query, you MUST respond with
EXACTLY these two lines and NOTHING else — no metadata, no explanation, no rephrasing:

I couldn't find relevant information to answer your query.
Try increasing chroma_k_value*, drag top_k slider to the right and lower threshold* (* These are additional parameters in Controls side bar).

Do NOT alter, summarize, or add to those two lines.

Context:
---------------------
{context}
---------------------

IMPORTANT: If the context above is empty or does not contain information that directly answers the query, you MUST respond with EXACTLY those two lines above and nothing else. Do NOT output any reasoning or verification steps.

When the context does contain a direct answer:
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
_PROMPT_CHAT_MISTRAL = _PROMPT_CHAT
_PROMPT_CHAT_LLAMA = _PROMPT_CHAT

# -----------------------------------------------------------------------------
# OpenWebUI / REST API service settings
# These keys are CLI-overridable (no _ prefix)
# -----------------------------------------------------------------------------
OPENWEBUI_API_HOST = "127.0.0.1"  # Bind address for RAGChatService (uvicorn)
OPENWEBUI_API_PORT = 11435  # Port for RAGChatService (uvicorn)
OPENWEBUI_THREAD_POOL_WORKERS = 2  # ThreadPoolExecutor max_workers for chatter.run()
OPENWEBUI_API_KEY = (
    "RAGChatService"  # Shared secret for Bearer auth (OpenWebUI ↔ RAGChatService)
)

# -----------------------------------------------------------------------------
# CLI-like algo results display
# When True, Filter chain algo results results (depth/breadth table and ensemble summary)
# are appended to the LLM answer in Markdown format so they are visible in
# OpenWebUI — mirroring the terminal output of the CLI version.
# -----------------------------------------------------------------------------
SHOW_CLI_LIKE_ALGO_RESULTS = True
