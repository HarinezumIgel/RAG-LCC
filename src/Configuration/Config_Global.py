"""Global shared configuration defaults and hash-pinned settings."""

# pylint: disable=invalid-name

# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned_Detection.py,
#     Config_Banned_Content.py, Config_Banned_Prompts.py,
#     Config_Load_Retrievers.py, Config_Load_Chunkers.py,
#     Config_Models.py,
#     Config_Languages.py, Config_Global.py
# - CLI overrides are allowed only for Config_Global and the active app config
# - CLI-overridable entries must not start with _ or $
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

import os

from Compliance.path_validation import validate_absolute_path

_VERSION = "v0.5.2.0/1520 2026-09-28"

# -----------------------------------------------------------------------------
# Adjust these hashes when you changed any of these files:
# Config_Models.py, Config_Banned_Detection.py,
# Config_Banned_Content.py, Config_Banned_Prompts.py,
# Config_Load_Retrievers.py, Config_Load_Chunkers.py,
# Config_WebSearch.py, or Config_Internet_Env.py.
# All keys below must be present.
# Run:  python src/Scripts/RecalcConfigHashes.py  to update automatically.
# -----------------------------------------------------------------------------
_CRITICAL_CONFIG_HASHES = {
    "Config_Models": "d79d430b22620b243bd6eaf46e35988060ab7c7740bf51120e0245d3944e6426",
    "Config_Banned_Detection": "7950c8fc408bae518deb255f895fcd8e4b249b360af87bed32350ec5aade5fe1",
    "Config_Banned_Content": "a27ecae368968c806ed220d6794b72e402e9175d502e95eb584bf632eebf0c35",
    "Config_Banned_Prompts": "3b556d4888f65702fcba0f0cb27d36456a7be5ff3a8130ddb262d9449f41a05f",
    "Config_Load_Retrievers": "7cc6146112ca0f50f27671e9f927e04c52a750532ef081abd62eca764201ff3f",
    "Config_Load_Chunkers": "97252ab64e994be70052b82b6f039e02058360e32f2b0caf89a54a12ae1cf353",
    "Config_WebSearch": "b1f8424bc12425e7fac0e870f060380f59be344e87990ea79c56ac0f1068dd9b",
    "Config_Internet_Env": "72bec360e2a260f59749fd39371afd7adf04d93d241c24118955c65b8e00e559",
}

# -----------------------------------------------------------------------------
# Force CPU or GPU usage. Set EMBEDDER_BITS to 32 if USE_CPU = True
# -----------------------------------------------------------------------------
USE_CPU = False
# !! set to 32 if USE_CPU = TRUE.
# If 16 bits are used on GPU, accelerate needs to be installed
EMBEDDER_BITS = 32  # 16, 32 !! set to 32 if USE_CPU = TRUE

# -----------------------------------------------------------------------------
# Base paths and modes
# -----------------------------------------------------------------------------

# Auto-detect project root from this file's location (source/Configuration/)
_ABSOLUTE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)


# Guard at source-of-truth module level as well.
_ABSOLUTE_PATH = validate_absolute_path(  # pyright: ignore[reportConstantRedefinition]
    _ABSOLUTE_PATH,
    "Config_Global._ABSOLUTE_PATH",
    "Config_Global",
)

# -----------------------------------------------------------------------------
# Paths for document loading and processing. Root folder for documents
# Change "Test" to point to your document dir
# -----------------------------------------------------------------------------
DOC_DIR = os.path.join(_ABSOLUTE_PATH, "TestDocs")

# -----------------------------------------------------------------------------
# Handle exclusions
# -----------------------------------------------------------------------------
# The exclusions file can be used to exclude files that contained banned words
_EXCLUSIONS_DIR = os.path.join(_ABSOLUTE_PATH, r"Exclusions")
# Omit excluded files from processing
USE_EXCLUSIONS = False

# -----------------------------------------------------------------------------
# HF pathes
# -----------------------------------------------------------------------------
_HF_HOME = (
    os.path.join(
        os.path.expanduser("~"), ".cache"
    )  # Windows: C:\Users\<current-user>\.cache
    if os.name == "nt"
    else os.path.join("/home", "vscode", ".cache")  # Linux / macOS
)
_HF_HUB_CACHE = os.path.join(_HF_HOME, ".hf-cache")


# -----------------------------------------------------------------------------
# Unsupported-language handling
# -----------------------------------------------------------------------------
# What to do when a document's detected language is not installed in
# Argos Translate (i.e. banlists cannot be translated).
#   "NOT_OK"       – reject the document; write to NOT_OK CSV, skip processing
#   "FALLBACK_EN"  – process silently with English-fallback banlists (legacy default)
UNSUPPORTED_LANGUAGE_ACTION = "NOT_OK"

# -----------------------------------------------------------------------------
# Tesseract for text extraction
# -----------------------------------------------------------------------------
# Tesseract lookup is resolved in Helpers.configure_tesseract().

# -----------------------------------------------------------------------------
# Set to false if you don't have the MS Office component installed
# -----------------------------------------------------------------------------
_OFFICE_DOC_EXTRACTION = {
    "Word": True,
    "Power Point": True,
    "Excel": True,
}

# -----------------------------------------------------------------------------
# Suffixes that are considered as text files
# -----------------------------------------------------------------------------
_CONSIDER_AS_TEXT_FILE = ["txt", "md", "py", "c", "h", "cpp", "csv", "log"]
# -----------------------------------------------------------------------------
# Debug settings.
# Format: 30  (== 30, exact)   ge 30  (>= 30)   is 30  (== 30)   none / ""  (silent)
# Can also be set via --debug-level CLI arg or in-chat with: set debug ge 30
# -----------------------------------------------------------------------------
_ALLOWED_DEBUG_LEVELS = {
    "None": 0,  # Completely silent
    "Basic": 10,  # Pass/fail outcomes across all subsystems
    "Service": 20,  # API request dump, service lifecycle
    "Retrieval Orchestration": 28,  # Stage-by-stage retrieval orchestration flow
    "Query Rewrite": 29,  # Rewrite decisions and topic-detect results
    "Standard": 30,  # Pipeline flow: session state, retrieval decisions
    # Text fed to banned-phrase filter chain + system-message hint.
    "Prompt Check Input": 31,
    # Full retrieved chunk text + metadata (file, chunk_id, hash, scores).
    "Chunk Content": 32,
    # Answer grounding: per-chunk sentence matches, snippet counts, marker type.
    "Grounding": 33,
    "Algos": 40,  # Scorer internals, masker, accumulator, keyword extraction
    "Components": 55,  # Synonyms detail, argostranslate, transformers, URL logging
    "Chat Prompt": 60,  # Full prompt text sent to LLM
    "Extracted Content": 70,  # Raw document content from classification
    "Ollama Response": 80,  # Raw Ollama request/response detail
    "Streaming": 100,  # Per-chunk raw streaming output
}

DEBUG_LEVEL = (
    30  # Examples: none (silent)  30 / ge 30 (>=30)  is 33 (==33 Grounding only)
)

# Enable urllib (http requests) debugging
URL_DEBUG = False

# Enable HF Debug
HF_DEBUG = False

# -----------------------------------------------------------------------------
# Fix broken LLM JSON outputs
# Try to fix broken JSON outputs from LLMs by appending missing closing braces.
# This is a quick and dirty fix if models do not return "strict JSON"
# -----------------------------------------------------------------------------
TRY_FIX_JSON_LLM_REPLY = True

# Log File (Legal Compliance)
LOG_FILE = r"compliance.log"

# Performance event log — set False to suppress all perf_event output.
# When True, start/stop timestamps are written to logs/Performance/perf.log.
PERFORMANCE_LOGGING = True

# Confidence signal for retrieval/answer pipeline tuning.
# - enabled: master switch for confidence step/event logging.
# - emit_pretty: print confidence lines to terminal/service logs.
# - csv_enabled: persist one CSV row per turn for offline analysis.
_CONFIDENCE_LOGGING: dict[str, bool] = {
    "enabled": True,
    "emit_pretty": True,
    "csv_enabled": True,
}

_LOG_DIRECTORY = os.path.join(_ABSOLUTE_PATH, "logs")

CSV_DELIMITER = r";"

# Seconds to wait for Ollama/OpenAI responses
REQUEST_TIMEOUT = 600

# -----------------------------------------------------------------------------
# NLP resources
# -----------------------------------------------------------------------------
# Custom directory for NLTK stopwords data
_CUSTOM_NLTK_DATA_DIRECTORY = (
    _ABSOLUTE_PATH + r"\AppData\Roaming\nltk_data\corpora\stopwords"  # Windows
    if os.name == "nt"
    else "/home/vscode/nltk_data"  # Linux / macOS
)

# =============================================================================
# Language detection tuning (lingua-language-detector)
# Consumed exclusively by FileUtils._detect_lang_iso / _log_lang_detection.
# Independent of Argos Translate — kept separate to avoid confusion.
# =============================================================================
_LANGUAGE_DETECTION: dict[str, int | float] = {
    # -----------------------------------------------------------------
    # Minimum word count required to attempt language detection.
    # Texts with fewer words skip detection and fall back to 'en'.
    # 1–2-word inputs carry too little signal for reliable detection
    # even with full-accuracy lingua.
    # -----------------------------------------------------------------
    "MIN_WORDS": 3,
    # -----------------------------------------------------------------
    # Lingua confidence floor applied to long text
    # (>= CONF_FULL_WORDS words).  For shorter text the effective
    # threshold is scaled up linearly toward 0.90 — see
    # FileUtils._detect_lang_iso.  Calibrated for lingua's realistic
    # confidence values (unlike langdetect's inflated scores).
    # -----------------------------------------------------------------
    "MIN_CONFIDENCE": 0.60,
    # -----------------------------------------------------------------
    # Word count at which MIN_CONFIDENCE is applied without any upward
    # scaling.  Below this the threshold rises linearly.
    # -----------------------------------------------------------------
    "CONF_FULL_WORDS": 10,
}

# Language activation/mapping and Argos pair catalog moved to
# src/Configuration/Config_Languages.py.

# -----------------------------------------------------------------------------
# Terminal line size: characters per output line used for PrettyWriter word-wrap.
# RAGChat overrides this with a {"debug": …, "no_debug": …} dict (Config_RAGChat.py)
# so the width switches automatically when debug_level is toggled in-session.
# -----------------------------------------------------------------------------
TERMINAL_LINE_SIZE = 140

# -----------------------------------------------------------------------------
# LLM answer display
#   bg: name of a Colors.py constant for the background, or None to disable
#   fg: name of a Colors.py constant for the foreground, or None for default
# Adjust the RGB values in Colors.py (ANSWER_BG / ANSWER_FG) to your taste.
# -----------------------------------------------------------------------------
ANSWER_DISPLAY = {
    "bg": "ANSWER_BG",
    "fg": "ANSWER_FG",
}


# -----------------------------------------------------------------------------
# DO NOT EDIT: Shared absolute-path guard wrappers used by other config modules.
# Keep these wrappers stable to preserve import/call contracts.
# -----------------------------------------------------------------------------
def _validate_absolute_path(path_value: str, caller_name: str) -> str:
    """Validate absolute path values for Config_Global wrappers."""
    return validate_absolute_path(
        path_value,
        "Config_Global._ABSOLUTE_PATH",
        caller_name,
    )


def require_absolute_path(caller_name: str) -> str:
    """Return validated `_ABSOLUTE_PATH` for dependent config modules."""
    return _validate_absolute_path(_ABSOLUTE_PATH, caller_name)
