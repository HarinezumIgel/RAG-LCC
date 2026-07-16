# -------------------------------------------------------------------------
# RAG-LCC Framework - Internet & Network Configuration
# -------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Argos Translate (Optional Local Translation)
# -----------------------------------------------------------------------------
# If the user installs Argos Translate and its language packages manually,
# RAG‑LCC will use them for local translation.
#
# Sentence boundary detection uses SpaCy (ARGOS_CHUNK_TYPE=SPACY) instead
# of stanza, which is broken offline in argos-translate ≥1.11.
# See argosopentech/argos-translate#385 / #512.
#
# When ARGOS_STANZA_DOWNLOAD is "1", RAG‑LCC will prompt the user to accept
# the Argos license and download language packages at startup if consent
# has not yet been recorded.
#
# -----------------------------------------------------------------------------
# NLTK Stopwords (Text Preprocessing)
# -----------------------------------------------------------------------------
# RAG‑LCC can use the NLTK stopwords corpus during text preprocessing.
#
# If the stopwords corpus is not available:
#   - when NLTK_STOPWORDS_DOWNLOAD is set to "0", an empty stopword list is used
#   - when NLTK_STOPWORDS_DOWNLOAD is set to "1", the system may attempt to
#     retrieve the required NLTK stopwords resource automatically
#
# Stopwords are treated as small, commonly distributed linguistic resources.
# Their use does not introduce additional per‑language consent handling within
# this configuration.

import os

# ============================================================================
# RAG-LCC Framework Configuration - Internet & Network Settings
# The framework works fine offline after downloading required components
# ============================================================================

# (License files for LLMs defined in Config_Models.py, NLTK stopwords)
os.environ["LICENSE_DOWNLOAD"] = "0"

# (NLTK stopwords)
os.environ["NLTK_STOPWORDS_DOWNLOAD"] = "0"

# RAG_LCC_NW_TRACE: Enable network-level socket tracing for debugging
# Set to "1" to see detailed network activity
os.environ["RAG_LCC_NW_TRACE"] = ""

# RAG_LCC_STACK_TRACE: Enable stack trace output for debugging
os.environ["RAG_LCC_STACK_TRACE"] = "0"

# WEB_SEARCH_MODE: Master web-search switch.
# Allowed values: "0" | "1"
# "0"        = web search disabled (default safe mode)
# "1"        = full web search path enabled
os.environ["WEB_SEARCH_MODE"] = "0"

# TESSERACT_PATH: OS-aware Tesseract executable path for OCR.
# Specify paths for both Windows and Linux. The framework automatically
# selects the appropriate path based on the current platform.
#
# Format: "windows_path|linux_path"
# - Paths are separated by the pipe character "|"
# - Windows path comes first, Linux path second
# - Leave a side empty to skip that platform, or use "tesseract" for PATH lookup
#
# Examples:
#   Both platforms with explicit paths:
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe|/usr/bin/tesseract"
#
#   Windows explicit, Linux from PATH:
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe|tesseract"
#
#   Both from PATH:
#     "tesseract|tesseract"
#
# Default uses standard installation locations:
os.environ.setdefault(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe|/usr/bin/tesseract"
)

# ============================================================================
# Hugging Face Hub Offline Mode - MUST be set BEFORE importing transformers
# ============================================================================

# HF_HUB_OFFLINE: Disable Hugging Face Hub access if set to "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# TRANSFORMERS_OFFLINE: Disable Hugging Face transformers library hub access
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# HF_DATASETS_OFFLINE: Disable Hugging Face datasets library hub access
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ARGOS_STANZA_DOWNLOAD: Control Argos Translate consent & package downloads.
# "0" = only use locally installed argostranslate language pairs;
#        non-installed languages fall back to English-normalized patterns.
# "1" = prompt for Argos license consent and download language packages
#        at startup if consent has not yet been recorded.
# Note: stanza is no longer used for Sentence Boundary Detection (see ARGOS_CHUNK_TYPE below).
os.environ["ARGOS_STANZA_DOWNLOAD"] = "0"

# ARGOS_MODEL_PROVIDER: Force Argos to use local package-based translation
# and avoid remote provider paths (LibreTranslate/OpenAI).
os.environ["ARGOS_MODEL_PROVIDER"] = "OPENNMT"

# ARGOS_CHUNK_TYPE: Select the sentence boundary detection (SBD) backend
# used by Argos Translate before translating.
# "SPACY"  = use SpaCy sentencizer (works offline, default used in this repository.
# "STANZA" = use stanza Pipeline (broken offline in argos-translate ≥1.11,
#            see argosopentech/argos-translate#385 / #512).
# "MINISBD" / "ARGOSTRANSLATE" / "DEFAULT" = other options.
os.environ["ARGOS_CHUNK_TYPE"] = "SPACY"

# HF_HUB_DISABLE_PROGRESS_BARS: Suppress Hugging Face Hub progress bar output
# "0" = show progress bars (useful for monitoring downloads)
# "1" = hide progress bars (cleaner terminal output)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# ============================================================================
# Optional: Safe Default Settings
# ============================================================================

# TOKENIZERS_PARALLELISM: Prevent tokenizer warnings about parallelism
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================================
# OpenWebUI Chat Service Endpoint
# ============================================================================

# SERVE_OPENWEBUI_CHAT: Enable the OpenAI-compatible REST API that allows
# OpenWebUI (or any OpenAI-compatible client) to connect to RAG-LCC as a
# custom model backend via POST /v1/chat/completions.
# Chroma collections are exposed as selectable models via GET /v1/models.
#
# "0" = disabled (safe default) — RAGChatService.py will refuse to start
# "1" = enabled — start RAGChatService.py with uvicorn on the configured port
#
# Launch command (from project root):
#   .venv\Scripts\python.exe src/Apps/RAGChatService.py
os.environ["SERVE_OPENWEBUI_CHAT"] = "0"

# SERVE_IN_MEMORY_DOCS_HTTP: Gate for the in-memory document HTTP server used
# by RAGChatService (serves both marked and non-marked document bytes via
# short-lived /marked/<token> links).
#
# "0" = disabled (safe default) — no in-memory docs store; mark_text is forced off
# "1" = enabled — in-memory docs store and /marked links are active
os.environ["SERVE_IN_MEMORY_DOCS_HTTP"] = "0"
