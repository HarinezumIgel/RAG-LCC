"""RAGChat configuration defaults and topic-specific re-exports."""

# pyright: reportUnusedVariable=false, reportUnknownVariableType=false

# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned_Detection.py,
#     Config_Banned_Content.py, Config_Banned_Prompts.py,
#     Config_Models.py, Config_Global.py
# - Entries starting with _ cannot be overwritten using CLI arguments
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

import os

# Do not change _FRIENDLY_NAME
_FRIENDLY_NAME = "RAGChat"

# -----------------------------------------------------------------------------
# Web search - per-app audit logs
# All shared web-search switches (WEB_SEARCH_MODE env var, _WEB_SEARCH dict,
# WEB_SEARCH_INTENT_EXTENSIONS) are in Config_WebSearch.py.
# _QUERY_LOG:  Append-only audit log written for every web-search attempt
#   (including blocked ones).  Written to the RAGChat log directory.
# -----------------------------------------------------------------------------

_LOG_DIRECTORY = os.path.join(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    ),
    "logs",
    "RAGChat",
)

_QUERY_LOG: str = os.path.join(_LOG_DIRECTORY, "queries.log")

# Append-only log for WebSearchFilter intent-classifier decisions (both web
# and local queries).  Set to "" to disable.
_INTENT_FILTER_LOG: str = os.path.join(_LOG_DIRECTORY, "intent_filter.log")

# -----------------------------------------------------------------------------
# Chat history and user defaults
# -----------------------------------------------------------------------------
_HISTORY_DIRECTORY = r"history"  # Where to store chat histories and metadata

# Keys to extract# -----------------------------------------------------------------------------
# Keyword extraction
# -----------------------------------------------------------------------------
_KEY_BERT = {
    "TOP_N_FIRST": 100,  # Keywords from first  KeyBERT pass
    "TOP_N_SECOND": 60,  # Keywords from second KeyBERT pass
}

_CLASSIFICATION_KEYS = []

_DEFAULT_CHAT_NAME = "MyFirstChat"  # Fallback user identifier

# -----------------------------------------------------------------------------
# Declared preferred response language of the user (lowercase NLTK language name,
# e.g. "english", "german", "french"). Must match a value in
# _ARGOS_DEFINITIONS.LANG_CODE_TO_NAME. Used by FileUtils.get_user_text_language()
# -----------------------------------------------------------------------------
# Retrieval-Augmented Generation strategy profiles
# -----------------------------------------------------------------------------
