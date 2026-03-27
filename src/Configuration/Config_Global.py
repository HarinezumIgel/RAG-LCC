# -------------------------------------------------------------------------
# - Lookup order: Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#   Config_Banned, Config_Models.py, Config_Globals.py
# - Entries starting with _ cannot be overwritten using CLI arguments
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

import os

_VERSION = "v0.1.3/1045 03/25/2025"

# -----------------------------------------------------------------------------
#    Adjust this hash when you changed Config_Models.py
# -----------------------------------------------------------------------------
_MODELS_CONFIG_HASH = ""

# -----------------------------------------------------------------------------
# Adjust this hash when you changed Config_Banned.py
# -----------------------------------------------------------------------------
_BANNED_CONFIG_HASH = ""

# -----------------------------------------------------------------------------
# Force CPU or GPU usage. Set EMBEDDER_BITS to 32 if USE_CPU = True
# -----------------------------------------------------------------------------
USE_CPU = False
# !! set to 32 if USE_CPU = TRUE.
# If 16 bits are used on GPU, accelerate needs to be installed
EMBEDDER_BITS = 32  # 16, 32 !! set to 32 if USE_CPU = TRUE
USE_OLLAMA_GPU = True

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
_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
_HF_HUB_CACHE = os.path.join(_HF_HOME, "hub")

# -----------------------------------------------------------------------------
# Ollama URL and endpoint
# -----------------------------------------------------------------------------
_OLLAMA_BASE_URL = "http://localhost:11434/api/generate"

# -----------------------------------------------------------------------------
# Token budget for dynamic max_output_tokens calculation.
# TOKEN_BUDGET_CONTEXT_CAP acts as a hardware cap: if Ollama reports a larger
# context window than this value, the cap is used instead.  This protects
# weak CPUs / GPUs from being asked to fill a context they cannot hold.
# RESERVED_OUTPUT  – tokens kept unconditionally for the model reply.
# RESERVED_SYSTEM  – tokens reserved for the system / instruction preamble.
# -----------------------------------------------------------------------------
TOKEN_BUDGET_CONTEXT_CAP = 16384
TOKEN_BUDGET_RESERVED_OUTPUT = 2048
TOKEN_BUDGET_RESERVED_SYSTEM = 1024

# -----------------------------------------------------------------------------
# Ollama streaming
# -----------------------------------------------------------------------------
OLLAMA_STREAMING_REQ = False

# -----------------------------------------------------------------------------
# Unsupported-language handling
# -----------------------------------------------------------------------------
# What to do when a document's detected language is not installed in
# Argos Translate (i.e. banlists cannot be translated).
#   "NOT_OK"       – reject the document; write to NOT_OK CSV, skip processing
#   "FALLBACK_EN"  – process silently with English-fallback banlists (legacy default)
UNSUPPORTED_LANGUAGE_ACTION = "NOT_OK"

# -----------------------------------------------------------------------------
# ChromaDB collection handling on startup
# -----------------------------------------------------------------------------
#   RAGLoad only, variable is here so it can be passed as CLI argument
#   True  = preserve existing
#   False = wipe and recreate on each run
CHROMA_COLLECTION_KEEP = False

# -----------------------------------------------------------------------------
# Tesseract for text extraction
# -----------------------------------------------------------------------------
_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------------------------------------------------------
# Set to false if you don't have the MS Office component installed
# -----------------------------------------------------------------------------
_OFFICE_DOC_EXTRACTION = {
    "Word": False,
    "Power Point": False,
    "Excel": False,
}

# -----------------------------------------------------------------------------
# Suffixes that are considered as text files
# -----------------------------------------------------------------------------
_CONSIDER_AS_TEXT_FILE = ["txt", "md", "py", "c", "h", "cpp", "csv", "log"]
# -----------------------------------------------------------------------------
# Debug settings. Levels vary from 0 (none) to 4 (very verbose)
# -----------------------------------------------------------------------------
_ALLOWED_DEBUG_LEVELS = {
    "None": 0,
    "Basic": 1,
    "Standard": 3,
    "Alogs": 4,
    "Components": 50,  # argostranslate, transformers
    "Chat Prompt": 60,
    "Extracted Content": 70,
    "Ollama response": 80,
    "Streaming request output": 100,
}

DEBUG_LEVEL = 3  # See above

# Enable urllib (http requests) debugging
URL_DEBUG = False

# Enable HF Debug
HF_DEBUG = False

# -----------------------------------------------------------------------------
# ChromaDB settings
# -----------------------------------------------------------------------------
# Directory where ChromaDB stores its document embeddings
_CHROMA_DB_DIR = _ABSOLUTE_PATH + r"\chromadb\docs"

# -----------------------------------------------------------------------------
# The collection load and queries are done with (you can override this setting
# or invoke RAGLoad.py with the --collection parameter and switch the collection
# in RAGChat.py using collection! command in chat
# -----------------------------------------------------------------------------
COLLECTION = "Test"

# -----------------------------------------------------------------------------
# These settings should be the *same* for RAGChat.py and RAGLoad.py
# Placing them into the global config file ensures this
# Switching variants requires dropping and reloading the collection
# (CHROMA_COLLECTION_KEEP = False) because HNSW parameters are immutable
# after creation.
# -----------------------------------------------------------------------------


_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG = "THOROUGH"

_CHROMA_EMBED_AND_RETRIEVE_PARAMS: dict[str, dict[str, float | int]] = {
    # ==========================
    # Variant: THOROUGH
    # ==========================
    # Larger chunks and more HNSW neighbours — favors recall and context
    # at the cost of index build time and query latency.
    "THOROUGH": {
        "CHUNK_SIZE": 256,  # Number of tokens per chunk
        "CHUNK_OVERLAP": 32,  # How many units from the end of one chunk are carried over to the next
        # Don't make this too big. Lower overlap 10 %, Higher overlap 20-30% of CHUNK_SIZE
        "NEIGHBORS_ON_LOAD": 512,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 512,  # Explore more neighbours at query (chat) time.
    },
    # ==========================
    # Variant: COMPACT
    # ==========================
    # Smaller chunks and fewer neighbours — favors precision and speed.
    "COMPACT": {
        "CHUNK_SIZE": 128,  # was 256
        "CHUNK_OVERLAP": 16,  # was 32
        # Don't make this too big. Lower overlap 10 %, Higher overlap 20-30% of CHUNK_SIZE
        "NEIGHBORS_ON_LOAD": 64,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 64,  # Explore more neighbours at query (chat) time.
    },
}

# -----------------------------------------------------------------------------
# Fix broken LLM JSON outputs
# Try to fix broken JSON outputs from LLMs by appending missing closing braces.
# This is a quick and dirty fix if models do not return "strict JSON"
# -----------------------------------------------------------------------------
TRY_FIX_JSON_LLM_REPLY = True

# Log File (Legal Compliance)
LOG_FILE = r"compliance.log"

_LOG_DIRECTORY = os.path.join(_ABSOLUTE_PATH, "logs")

CSV_DELIMITER = r";"

# Seconds to wait for Ollama/OpenAI responses
REQUEST_TIMEOUT = 600

# -----------------------------------------------------------------------------
# NLP resources
# -----------------------------------------------------------------------------
# Custom directory for NLTK stopwords data
_CUSTOM_NLTK_DATA_DIRECTORY = (
    _ABSOLUTE_PATH + r"\AppData\Roaming\nltk_data\corpora\stopwords"
)

# =============================================================================
# Argos Translate definitions
# Groups language-code mapping and translation pairs in one slot.
# =============================================================================
_ARGOS_DEFINITIONS: dict[str, float | dict[str, str] | list[tuple[str, str]]] = {
    # -----------------------------------------------------------------
    # Minimum text length (characters) required to attempt language detection.
    # Texts shorter than this skip detection and fall back to 'en'.
    # Single words give langdetect too little signal and produce 100 % confident
    # but wrong results (e.g. "igel" → Danish instead of German).
    # -----------------------------------------------------------------
    "LANG_DETECT_MIN_CHARS": 20,
    # -----------------------------------------------------------------
    # Minimum langdetect confidence to trust the top detected language.
    # If the top result is below this threshold, language falls back to 'en'.
    # Prevents short queries (e.g. "tell me about llama") from being
    # misclassified as a non-English language and triggering translation warnings.
    # -----------------------------------------------------------------
    "LANG_DETECT_MIN_CONFIDENCE": 0.90,
    # -----------------------------------------------------------------
    # Language code ↔ name mapping (single source of truth)
    # ISO-639-1 code → NLTK / human-readable name.
    # Used by FileUtils (code→name) and SharedHelpers (name→code, reversed).
    # Covers the intersection of langdetect output and NLTK stopword names.
    # -----------------------------------------------------------------
    "LANG_CODE_TO_NAME": {
        "ar": "arabic",
        "bn": "bengali",
        "ca": "catalan",
        "da": "danish",
        "de": "german",
        "el": "greek",
        "en": "english",
        "es": "spanish",
        "fi": "finnish",
        "fr": "french",
        "he": "hebrew",
        "hu": "hungarian",
        "id": "indonesian",
        "it": "italian",
        "ja": "japanese",
        "ko": "korean",
        "ne": "nepali",
        "nl": "dutch",
        "no": "norwegian",
        "pt": "portuguese",
        "ro": "romanian",
        "ru": "russian",
        "sl": "slovene",
        "sq": "albanian",
        "sv": "swedish",
        "ta": "tamil",
        "tr": "turkish",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
    },
    # -----------------------------------------------------------------
    # Argos Translate language pairs
    # Used by src/Scripts/ArgosTranslatePackages.py and the startup consent check.
    # Each tuple is (from_code, to_code).
    # -----------------------------------------------------------------
    "ARGOS_LANGUAGES": [
        # ("en", "ar"),  # English → Arabic
        # ("en", "az"),  # English → Azerbaijani
        # ("en", "bg"),  # English → Bulgarian
        # ("en", "bn"),  # English → Bengali
        # ("en", "ca"),  # English → Catalan
        # ("en", "cs"),  # English → Czech
        # ("en", "da"),  # English → Danish
        ("en", "de"),  # English → German
        # ("en", "el"),  # English → Greek
        # ("en", "eo"),  # English → Esperanto
        ("en", "es"),  # English → Spanish
        # ("en", "et"),  # English → Estonian
        # ("en", "eu"),  # English → Basque
        # ("en", "fa"),  # English → Persian
        # ("en", "fi"),  # English → Finnish
        ("en", "fr"),  # English → French
        # ("en", "ga"),  # English → Irish
        # ("en", "gl"),  # English → Galician
        # ("en", "he"),  # English → Hebrew
        # ("en", "hi"),  # English → Hindi
        # ("en", "hu"),  # English → Hungarian
        # ("en", "id"),  # English → Indonesian
        ("en", "it"),  # English → Italian
        # ("en", "ja"),  # English → Japanese
        # ("en", "ko"),  # English → Korean
        # ("en", "ky"),  # English → Kyrgyz
        # ("en", "lt"),  # English → Lithuanian
        # ("en", "lv"),  # English → Latvian
        # ("en", "ms"),  # English → Malay
        # ("en", "nb"),  # English → Norwegian
        # ("en", "nl"),  # English → Dutch
        # ("en", "pb"),  # English → Portuguese (Brazil)
        # ("en", "pl"),  # English → Polish
        # ("en", "pt"),  # English → Portuguese
        # ("en", "ro"),  # English → Romanian
        # ("en", "ru"),  # English → Russian
        # ("en", "sk"),  # English → Slovak
        # ("en", "sl"),  # English → Slovenian
        # ("en", "sq"),  # English → Albanian
        # ("en", "sv"),  # English → Swedish
        # ("en", "th"),  # English → Thai
        # ("en", "tl"),  # English → Tagalog
        # ("en", "tr"),  # English → Turkish
        # ("en", "uk"),  # English → Ukrainian
        # ("en", "ur"),  # English → Urdu
        # ("en", "vi"),  # English → Vietnamese
        # ("en", "zh"),  # English → Chinese
        # ("en", "zt"),  # English → Chinese (traditional)
    ],
}

# -----------------------------------------------------------------------------
# Terminal size
# -----------------------------------------------------------------------------
TERMINAL_LINE_SIZE = 160

_LEET_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
}
_CONFUSABLES = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "в": "b",
    "к": "k",
    "м": "m",
    "н": "h",
    "т": "t",
    "İ": "i",
    "ı": "i",
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
}
