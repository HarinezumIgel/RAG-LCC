import os
from typing import Any

# import * silently drops _-prefixed names — re-import them explicitly:
# Re-export the full RAGChat config so the Config loader finds every key
# (_KEY_BERT, _STRATEGIES, _ALLOWED_STRATEGIES, _HISTORY_DIRECTORY, etc.)
from Configuration.Config_RAGChat import *  # noqa: F401, F403
from Configuration.Config_RAGChat import (  # pyright: ignore[reportPrivateUsage, reportUnusedImport, reportUnknownVariableType]; pyright: ignore[reportPrivateUsage, reportUnusedImport]; noqa: F401; pyright: ignore[reportPrivateUsage, reportUnusedImport]
    _ACTIVE_CHUNK_SELECT_STRATEGY, _ALLOWED_RETRIEVE_MODES,
    _ALLOWED_STRATEGIES, _CHUNK_DEDUP, _CLASSIFICATION_KEYS,
    _DEFAULT_CHAT_NAME, _HISTORY_DIRECTORY, _KEY_BERT, _MARKED_DOCS_COLORS,
    _MARKED_DOCS_GROUNDING, _MULTI_QUERY, _PROMPT_QUERY_EXPAND,
    _PROMPT_TOPIC_DETECT, _QUERY_REWRITE, _STRATEGIES)
from Configuration.Config_WebSearch import \
    _OPENWEB_UI_WEBSEARCH  # noqa: F401; pyright: ignore[reportPrivateUsage, reportUnusedImport]

# -------------------------------------------------------------------------
# RAGChatService is RAGChat served over HTTP (OpenWebUI-compatible REST API).
# All RAGChat configuration is reused as-is — only the API host/port are added.
#
# _FRIENDLY_NAME = "RAGChatService" matches the alias added in Config_Banned.py
# so compliance lookups resolve to the correct RAGChatService pipeline.
# -------------------------------------------------------------------------


_FRIENDLY_NAME = (
    "RAGChatService"  # Matches the alias in _BANNED_DETECT (Config_Banned.py)
)

_LOG_DIRECTORY = os.path.join(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    ),
    "logs",
    "RAGChatService",
)

# Per-service audit logs (separate from RAGChat logs — each app owns its log dir).
_QUERY_LOG: str = os.path.join(_LOG_DIRECTORY, "queries.log")
_INTENT_FILTER_LOG: str = os.path.join(_LOG_DIRECTORY, "intent_filter.log")

# -----------------------------------------------------------------------------
# Override prompt: RAGChatService gets OpenWebUI-specific hint
# -----------------------------------------------------------------------------
_PROMPT_CHAT = """
CRITICAL: You must ONLY use information found in the context below.
Do NOT use your training knowledge, do NOT guess, do NOT infer beyond what the context states.
If the context is entirely empty or absolutely irrelevant to the query, you MUST respond with
EXACTLY these two lines and NOTHING else — no metadata, no explanation, no rephrasing:

I couldn't find relevant information to answer your query.
Try increasing retriever_k*, drag top_k slider to the right and lower threshold* (* These are additional parameters in Controls side bar).

Do NOT alter, summarize, or add to those two lines.

Context:
---------------------
{context}
---------------------

IMPORTANT: If the context above is entirely empty or absolutely irrelevant to the query, you MUST respond with EXACTLY those two lines above and nothing else. Do NOT output any reasoning or verification steps.

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

# Directory that holds static service assets (favicon.ico, etc.).
# Relative paths are resolved from the project root at startup.
# Change this if you want to serve a custom favicon from a different location.
_SERVICE_RESOURCES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Apps", "resources"
)

# RAGChatService host/port are now configured in Config_Models.py:
# _MODELS.ragchatservice._RAGCHATSERVICE.HOST
# _MODELS.ragchatservice._RAGCHATSERVICE.PORT

OPENWEBUI_THREAD_POOL_WORKERS = 2  # ThreadPoolExecutor max_workers for chatter.run()

# -----------------------------------------------------------------------------
# CLI-like algo results display
# When True, Filter chain algo results results (depth/breadth table and ensemble summary)
# are appended to the LLM answer in Markdown format so they are visible in
# OpenWebUI — mirroring the terminal output of the CLI version.
# -----------------------------------------------------------------------------
SHOW_CLI_LIKE_ALGO_RESULTS = True

# -----------------------------------------------------------------------------
# Document-serving service (in-memory token store)
#
# Server enable/disable gate lives in Configuration/Config_Internet_Env.py:
#   os.environ["SERVE_IN_MEMORY_DOCS_HTTP"] = "0" or "1"
#
# This block configures the runtime behaviour of the in-memory document
# server once it is enabled at the internet-config level.
#
# When the user enables `mark_text=True`, RAG-LCC produces in-memory
# highlighted copies of every retrieved local PDF. In service mode those
# bytes are stored under a short-lived, unguessable token and exposed at
# `GET /marked/{token}.pdf`, so OpenWebUI clients (potentially running on a
# different host) can fetch them directly without any shared filesystem.
#
# Security
#   * Tokens are 256-bit random strings — unforgeable.
#   * Entries auto-expire after `ttl_seconds`.
#   * Total in-memory size is capped (`max_total_bytes`); oldest entries
#     evicted FIFO when full.
#   * `single_use=True` destroys the entry after the first successful
#     fetch (use when you do not want refresh / sharing).
#   * The /marked endpoint is *not* protected by the API_KEY
#     bearer middleware (browsers won't send it). The unguessable token
#     is the access credential.
#
# Reverse-proxy / multi-host setups
#   * Set ``_SERVE_DOCS["public_base_url"]`` to the externally reachable URL
#     of this service (e.g. "https://rag.example.com"). When set it is
#     used as the prefix in the markdown links injected into the LLM
#     answer; otherwise the binding host:port is used.
#   * Set ``_SERVE_DOCS["cors_origins"]`` to a list of allowed origins (or
#     ["*"] to allow all) so cross-origin browsers can fetch the document.
# -----------------------------------------------------------------------------
_SERVE_DOCS: dict[str, Any] = {
    # cors_origins: list of allowed browser origins for the /marked endpoint.
    # Use ["*"] to allow all, or specify exact origins such as
    # ["https://openwebui.example.com"].  Leave empty to disable CORS middleware.
    "cors_origins": [],
    # TTL for highlighted documents stored in the in-memory cache.
    # Each GET /marked/<token> resets the expiry clock.
    # Shorter values reduce attack surface; longer values survive slow clients.
    "ttl_seconds": 1800,  # 30 min — also acts as a security control
    # Maximum total size of all cached documents.  Oldest entries are evicted FIFO
    # when an incoming document would exceed this limit.
    "max_total_mb": 200,  # MB
    # When True the cache entry is destroyed after the first successful GET.
    # Set to True when sharing / refresh of the link is undesirable.
    "single_use": False,
    # Externally-reachable base URL used to build the /marked links injected into
    # LLM answers (e.g. "https://rag.example.com").  Leave empty to derive the URL
    # from the incoming request or fall back to the bound host:port.
    "public_base_url": "",
}
