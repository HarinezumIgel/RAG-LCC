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
os.environ["RAG_LCC_NW_TRACE"] = "0"

# RAG_LCC_STACK_TRACE: Enable stack trace output for debugging
os.environ["RAG_LCC_STACK_TRACE"] = "0"

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

# HF_HUB_DISABLE_PROGRESS_BARS: Suppress progress bar output (set to "0")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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
# "0" = disabled (default) — RAGChatService.py will refuse to start
# "1" = enabled — start RAGChatService.py with uvicorn on the configured port
#
# Launch command (from project root):
#   .venv\Scripts\python.exe src/Apps/RAGChatService.py
os.environ["SERVE_OPENWEBUI_CHAT"] = "0"
