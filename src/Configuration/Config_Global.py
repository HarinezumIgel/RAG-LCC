# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned.py, Config_Models.py,
#     Config_Languages.py, Config_Global.py
# - Entries starting with _ cannot be overwritten using CLI arguments
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

import os
from typing import Any

_VERSION = "v0.5.1.1/1486 2026-09-21"

# -----------------------------------------------------------------------------
# Adjust these hashes when you changed Config_Models.py, Config_Banned.py,
# Config_WebSearch.py, or Config_Internet_Env.py. All four keys must be present.
# Run:  python src/Scripts/RecalcConfigHashes.py  to update automatically.
# -----------------------------------------------------------------------------
_CRITICAL_CONFIG_HASHES = {
    "Config_Models": "",
    "Config_Banned": "",
    "Config_WebSearch": "",
    "Config_Internet_Env": "",
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
# Retrieval stores handling on startup (ChromaDB collection, BM25 index, graph index, regex index)
# -----------------------------------------------------------------------------
#   RAGLoad only, variable is here so it can be passed as CLI argument
#   True  = preserve existing collection, BM25 index, graph index, and regex index
#   False = wipe and recreate on each run
RETRIEVAL_STORES_KEEP = False

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
    "Prompt Check Input": 31,  # Text fed to banned-phrase filter chain + system-msg hint
    "Chunk Content": 32,  # Full retrieved chunk text + metadata (file, chunk_id, hash, scores)
    "Grounding": 33,  # Answer grounding: per-chunk sentence matches, orange snippet counts, marker type
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
# Retrieval stores
#
# The four stores below are managed as a single unit by RETRIEVAL_STORES_KEEP:
#   - False → all four are deleted and rebuilt together on every RAGLoad run.
#   - True  → all four must exist on disk; a missing index aborts startup.
#
# Each path can be set independently, as long as all four stay in sync.
# Paths MUST be inside the project root (_ABSOLUTE_PATH): delete_file_or_dir()
# enforces a jailbreak guard that refuses to delete anything outside it.
# -----------------------------------------------------------------------------

# Directory where ChromaDB stores its document embeddings
_CHROMA_DB_DIR = os.path.join(_ABSOLUTE_PATH, "chromadb", "docs")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The collection load and queries are done with (you can override this setting
# or invoke RAGLoad.py with the --collection parameter and switch the collection
# in RAGChat.py using collection! command in chat
# -----------------------------------------------------------------------------
COLLECTION = "Test"

# =============================================================================
# COLLECTION SCHEMA — all settings that define *how* documents are stored
# =============================================================================
# Everything in this section is baked into the ChromaDB collection at creation
# time.  Changing ANY value here requires dropping and reloading the collection:
#
#     RETRIEVAL_STORES_KEEP = False
#
# This includes HNSW neighbour counts, the active chunker, chunk sizes,
# AUTO_CHUNK routing, and breakpoint thresholds.
# =============================================================================


_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG = "THOROUGH"

_CHROMA_EMBED_AND_RETRIEVE_PARAMS: dict[str, dict[str, float | int]] = {
    # ==========================
    # Variant: THOROUGH
    # ==========================
    # More HNSW neighbours — favors recall and context
    # at the cost of index build time and query latency.
    "THOROUGH": {
        "NEIGHBORS_ON_LOAD": 512,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 512,  # Explore more neighbours at query (chat) time.
    },
    # ==========================
    # Variant: COMPACT
    # ==========================
    # Fewer neighbours — favors precision and speed.
    "COMPACT": {
        "NEIGHBORS_ON_LOAD": 64,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 64,  # Explore more neighbours at query (chat) time.
    },
}

# Active chunker configuration profile — pick one of the keys in _CHUNK_STRATEGY.
# Select DETAILED only if you have a strong CPU or better GPU. SEMANTIC chunking takes time
_ACTIVE_CHUNKER_CONFIG = "DETAILED"

# Two-level chunker routing: each profile maps file extensions to a chunker.
# Keys are lowercase file extensions (from ValidExtensions.getFileType).
# "DEFAULT" is the fallback for any extension not explicitly mapped.
_CHUNK_STRATEGY: dict[str, dict[str, str]] = {
    "DETAILED": {
        "pdf": "PDF_PAGE",
        "doc": "HEADING",
        "docx": "HEADING",
        "pptx": "SLIDE",
        "ppt": "SLIDE",
        "xlsx": "RECURSIVE",
        "xls": "RECURSIVE",
        "csv": "RECURSIVE",
        "txt": "SLIDING_WINDOW",
        "md": "HEADING",
        "py": "RECURSIVE",
        "c": "RECURSIVE",
        "h": "RECURSIVE",
        "cpp": "RECURSIVE",
        "log": "RECURSIVE",
        "png": "RECURSIVE",
        "jpg": "RECURSIVE",
        "jpeg": "RECURSIVE",
        "gif": "RECURSIVE",
        "bmp": "RECURSIVE",
        "tiff": "RECURSIVE",
        "webp": "RECURSIVE",
        "DEFAULT": "SEMANTIC",
    },
    "FAST": {
        "DEFAULT": "RECURSIVE",
    },
}

_CHUNKERS: dict[str, dict[str, float | int | bool | str]] = {
    "RECURSIVE": {
        "CHUNK_SIZE": 256,  # Number of words per chunk
        "CHUNK_OVERLAP": 32,  # Overlap between consecutive chunks (10-30% of CHUNK_SIZE)
        "PRESERVE_NEWLINES": False,
    },
    "SEMANTIC": {
        "MAX_CHUNK_SIZE": 256,  # Safety cap — split semantic segments exceeding this
        # Higher = fewer breaks = larger chunks (diffuse embeddings).
        # Lower  = more breaks  = smaller, tighter chunks (sharper retrieval).
        # 50-70 is a good starting range; 10 was far too conservative.
        "BREAKPOINT_PERCENTILE": 15,  # Bottom N% cosine similarity = chunk boundary
        "EMBED_BATCH_SIZE": 32,  # Sentences per embedding batch (controls GPU memory)
        "MIN_SENTENCE_WORDS": 15,  # Merge consecutive fragments shorter than this before
        # embedding — prevents noisy vectors from PDF table rows
        "PRESERVE_NEWLINES": False,
    },
    "SENTENCE_WINDOW": {
        "MAX_CHUNK_SIZE": 256,  # Pack sentences up to this many words per chunk
        "PRESERVE_NEWLINES": False,
    },
    "SLIDING_WINDOW": {
        "MAX_CHUNK_SIZE": 256,  # Pack sentences up to this many words per chunk
        "OVERLAP_SENTENCES": 3,  # Re-include last N sentences of previous chunk in next
        "PRESERVE_NEWLINES": False,
    },
    "HEADING": {
        "MAX_CHUNK_SIZE": 256,  # Max words per heading-section chunk
        "PRESERVE_NEWLINES": True,  # Newlines needed for heading detection
        # Where to place the heading breadcrumb ("H1 > H2 > H3") inside each chunk.
        #   "prefix"    — prepend to chunk text (legacy; pollutes leading tokens
        #                 when many chunks share the same breadcrumb, which can
        #                 confuse small LLMs and weight all embeddings toward
        #                 the shared prefix).
        #   "suffix"    — append after the body text (recommended default: the
        #                 chunk's leading tokens are the actual content, while
        #                 the breadcrumb is still embedded for section context).
        #   "off"       — omit from chunk text entirely. The breadcrumb is still
        #                 preserved in metadata["HeadingPath"] for filtering /
        #                 display / reranking.
        "BREADCRUMB_MODE": "suffix",
    },
    "SLIDE": {
        "MAX_CHUNK_SIZE": 256,  # Max words per slide chunk
        "PRESERVE_NEWLINES": False,
    },
    "PDF_PAGE": {
        "MAX_CHUNK_SIZE": 200,  # Max words per page chunk; dense pages are split
        "PRESERVE_NEWLINES": False,
        # Best-effort: recover the printed page number (e.g. roman "iii") from
        # each page's footer/header text when the PDF's /PageLabels metadata
        # does not declare it. Heuristic — set False to trust /PageLabels only.
        "DETECT_PRINTED_LABEL": True,
    },
}

# =============================================================================
# Document metadata extraction
# =============================================================================
# Extra metadata harvested from source files at load time and attached to
# EVERY chunk of that file (all chunkers benefit — extraction happens once in
# the ingestion pipeline via DocumentMetadataExtractor). Formats with readable
# document properties (PDF, docx, pptx, xlsx) yield the rich DOC_INFO_FIELDS;
# every other type (images, text, csv, code, legacy Office, …) falls back to
# the generic filesystem GENERIC_FIELDS. Chunk metadata is baked into the
# ChromaDB collection, so changing anything here requires a reload
# (RETRIEVAL_STORES_KEEP = False).
#
# ChromaDB only accepts scalar metadata (str/int/float/bool) — every value is
# coerced to a string; missing/empty fields are skipped.
# =============================================================================
_METADATA_EXTRACTION: dict[str, Any] = {
    # Master switch. False = attach nothing extra (legacy behaviour).
    "ENABLED": True,
    # Canonical chunk-metadata field name → ordered list of raw source-property
    # synonyms searched across formats (case-insensitive; first non-empty wins).
    # Different formats name the same concept differently — e.g. PDFs expose a
    # "creation_date" while Office core-properties use "created". Format quirks
    # that are truly ambiguous are pre-normalised inside DocumentMetadataExtractor
    # (e.g. the xlsx "creator" property, which actually means *author*, is
    # emitted as "author"), so the synonyms below stay simple.
    "DOC_INFO_FIELDS": {
        "Author": ["author"],
        "DocTitle": ["title"],
        "Subject": ["subject"],
        "Creator": ["creator"],  # authoring application (PDF)
        "Producer": ["producer"],  # producing application (PDF)
        "DocCreated": ["creation_date", "created"],
        "DocModified": ["modification_date", "modified"],
        "LastModifiedBy": ["last_modified_by"],
        "Keywords": ["keywords"],
    },
    # Generic fallback fields for every OTHER file type (images, text, csv,
    # code, legacy Office, …) that has no readable document properties. Sourced
    # from the filesystem: "size" (bytes) and "modified" (mtime).
    "GENERIC_FIELDS": {
        "FileSizeBytes": ["size"],
        "FileModified": ["modified"],
    },
    # Printed per-page page label (e.g. "i", "ii", "1", "2") captured by the
    # PDF page chunker and written under this field. Empty string = disabled.
    # The physical 1-based page index always stays in "PageNumber".
    "PDF_PAGE_LABEL_FIELD": "PageLabel",
    # Append a "Document metadata" section (author/dates/pages per source
    # file) to CLI and RAGChatService answers, built from the harvested fields.
    "SHOW_IN_ANSWER": True,
}

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
