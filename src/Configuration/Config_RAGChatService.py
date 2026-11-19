import os
from typing import Any

# RAGChat split settings are loaded directly by Config._load_from_constants
# for RAGChatService. This file only defines service-specific overrides.

# -------------------------------------------------------------------------
# RAGChatService is RAGChat served over HTTP (OpenWebUI-compatible REST API).
# Answer prompts and grounding settings are inherited from RAGChat.
#
# _FRIENDLY_NAME = "RAGChatService" matches the alias added in
# Config_Banned_Detection.py
# so compliance lookups resolve to the correct RAGChatService pipeline.
# -------------------------------------------------------------------------

_FRIENDLY_NAME = (
    "RAGChatService"  # Matches the alias in _BANNED_DETECT (Config_Banned_Detection.py)
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

# Default orchestration flow is inherited from
# Config_Orchestrator._DEFAULT_ORCHESTRATION_FLOW (single source of truth for
# CLI and service). Uncomment to use a different default for the service only;
# per-request "orchestrator_flow" still overrides either value.
# _DEFAULT_ORCHESTRATION_FLOW = "ENGLISH_ALL_MULTILEG"

# -----------------------------------------------------------------------------
# Short-lived document and grounded-answer service (in-memory token store)
#
# Server enable/disable gate lives in Configuration/Config_Internet_Env.py:
#   os.environ["SERVE_RAGCHATSERVICE_GROUNDED_DOCS"] = "0" or "1"
#
# This block configures the runtime behaviour of the in-memory document
# server once it is enabled at the internet-config level.
#
# When enabled, RAG-LCC stores highlighted source documents and verified
# grounded-answer HTML at `/marked/<document-token>.<ext>?token=<session-token>`.
# OpenWebUI clients can open the links without a shared filesystem.
#
# Security
#   * Tokens are 256-bit random strings — unforgeable.
#   * A separate 192-bit session capability is required on every GET and rotates
#     at service startup. Both tokens are embedded automatically in each link.
#   * Entries auto-expire after `ttl_seconds`.
#   * Total in-memory size is capped (`max_total_bytes`); oldest entries
#     evicted FIFO when full.
#   * `single_use=True` destroys the entry after the first successful
#     fetch (use when you do not want refresh / sharing).
#   * The /marked endpoint is *not* protected by the API_KEY
#     bearer middleware (browsers won't send it). The two unguessable tokens
#     are the access credentials. Anyone with the full link can fetch the bytes.
#
# Reverse-proxy / multi-host setups
#   * The document routes share the RAGChatService API listener. The default
#     HOST is 127.0.0.1. Remote browsers need a tunnel or an explicitly configured
#     reachable interface, firewall restrictions, and preferably an HTTPS proxy.
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
    # TTL for marked documents and grounded-answer pages in the in-memory cache.
    # Expiry is measured from creation; GET requests do not extend it.
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
