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

# - Keep this file in sync with Config_Global.py's _CRITICAL_CONFIG_HASHES
#   entries for Config_Banned* modules.
# - This structure is intentionally explicit: each section documents intent,
#   and runtime behavior.
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Algorithm name constants (single source of truth for labels used
# throughout the pipeline). Use these constants when referencing algos.
# ---------------------------------------------------------------------
_COSINE = "Cosine"
_JACCARD = "Jaccard"
_REGEX = "Regex"
_KEYBERT = "Keybert"
_LEVENSHTEIN = "Levenshtein"  # Levenshtein results are added to regex
_BM25 = "BM25"

# ---------------------------------------------------------------------
# Human-friendly alias mapping used for CSV headers and CLI summaries.
# This ensures consistent column names when Regex and Levenshtein are
# reported together as a single combined label.
# ---------------------------------------------------------------------
_REGEX_LEVENSHTEIN = _REGEX + "+" + _LEVENSHTEIN

_LABEL_ALIAS = {
    _REGEX: _REGEX_LEVENSHTEIN,
    "Score " + _REGEX: "Score " + _REGEX_LEVENSHTEIN,
    "Threshold " + _REGEX: "Threshold " + _REGEX_LEVENSHTEIN,
    "Detail " + _REGEX: "Details " + _REGEX_LEVENSHTEIN,
}

# ---------------------------------------------------------------------
# Default algorithms to run in the pipeline when no custom selection
# is provided. Order here is not enforcement order; it's a default list.
# ---------------------------------------------------------------------
_DEFAULT_ALGOS = [
    _JACCARD,
    _BM25,
    _REGEX,
    _KEYBERT,
    #   _COSINE,
]

# ---------------------------------------------------------------------
# Keys/columns included in the CSV produced for human review.
# Keep this list stable to avoid breaking downstream analysis scripts.
# ---------------------------------------------------------------------
_KEYS_FOR_HUMAN_REVIEW_CSV = [
    "Status",
    "Time",
    "Stage",
    "Skip Status",
    "Skipped Chunks",
    "Inserted Chunks",
    "Phrase",
    "Max Score",
    "Matched Algos Count",
    "Algos Matched",
    _JACCARD,
    "Score " + _JACCARD,
    "Threshold " + _JACCARD,
    _REGEX_LEVENSHTEIN,
    "Score " + _REGEX_LEVENSHTEIN,
    "Threshold " + _REGEX_LEVENSHTEIN,
    _BM25,
    "Score " + _BM25,
    "Threshold " + _BM25,
    _KEYBERT,
    "Score " + _KEYBERT,
    "Threshold " + _KEYBERT,
    #   _COSINE,
    #    "Score " + _COSINE,
    #    "Threshold " + _COSINE,
    "WordCount",
    "Temperature",
    "Session",
    "FilePath",
    "FileType",
    "Language",
    "CreationDate",
    "Chunk",
    "FileHash",
]

# ---------------------------------------------------------------------
# Detection configuration: top-level container for per-app rules.
# Each app (RAGLoad, RAGChat, DocClassify) has:
#   - MASKING: runtime masking toggles
#   - PROMPT_CHECK: whether to run prompt-level LLM checks and params
#   - PIPELINE_CHECK: the retrieval/matching pipeline configuration
# ---------------------------------------------------------------------
_ACTIVE_DETECTION_CONFIG = "STRICT_DETECT_CONFIG"
_BANNED_DETECT = {
    # Strict detection profile used by multiple apps
    "STRICT_DETECT_CONFIG": {
        # -------------------------
        # RAGLoad: checks applied when loading documents into the RAG store
        # - Masking is applied by default to avoid storing banned content.
        # - Prompt checks are disabled for RAGLoad (we only check content).
        # -------------------------
        "RAGLoad": {
            "MASKING": {
                "APPLY_MASKING": True,  # If True, redact/mask matched spans before storage
            },
            "PROMPT_CHECK": {
                "Check": False,  # No LLM prompt-level check during load
            },
            "PIPELINE_CHECK": {
                # PIPELINE contains per-algo thresholds and tuning parameters
                "PIPELINE": {
                    "Jaccard": {
                        # Character n-gram range used for Jaccard similarity
                        "CHAR_NGRAM_RANGE": (4, 6),
                        "THRESHOLD": 0.75,
                        "THRESHOLD_MIN": 0.5,
                    },
                    # Cosine is intentionally commented out; enable if vectors exist
                    # "Cosine": { "THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2 },
                    "Keybert": {
                        # KeyBERT keyword overlap threshold and top-k extraction size
                        "THRESHOLD": 0.45,
                        "THRESHOLD_MIN": 0.2,
                        "TOP_K": 1000,  # large TOP_K for indexing-time scans
                    },
                    "BM25": {
                        # BM25 hyperparameters and normalization thresholds
                        "THRESHOLD": 0.7,
                        "THRESHOLD_MIN": 0.2,
                        "TERM_FREQ_SATURATION": 1.2,  # k1: term-frequency saturation
                        "LENGTH_NORMALIZATION": 0.75,  # b: length normalization
                        "MIN_OVERLAP": 2,  # minimum overlapping terms to consider a match
                        "MIN_RAW_SCORE": 25,  # raw score floor; tune on dev set
                        "NORM_PERCENTILE": 97,  # percentile used for normalization
                    },
                    "Regex": {
                        # Regex + fuzzy anchors + Levenshtein integration
                        "THRESHOLD": 1.0,
                        "THRESHOLD_MIN": 0.5,
                        "WINDOW_MAX_CHARS": 20,  # max filler chars between anchors
                        "PREFIX_SUFFIX_LEN": 3,  # chars taken from token edges for anchors
                        "SEPARATOR_CLASS": r"[A-Za-z0-9_\-]",  # allowed filler characters
                        "SOFT_SCORE_HARD": 1.0,  # score assigned on strict (exact) regex match
                        "SOFT_SCORE_FUZZY": 0.75,  # score assigned on fuzzy-only regex match
                        "FUZZY_REGEX_EVAL_AFTER_HARD": False,  # run fuzzy evaluation when strict misses
                        "Levenshtein": {
                            "THRESHOLD": 0.5,  # fuzzy edit-distance threshold
                        },
                    },
                    # How many algos must be above their thresholds to trigger a block
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 3,
                    # How many different algos must produce a non-zero score
                    "REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 4,
                    # Which algorithms to actually run in this pipeline
                    "ALGOS_TO_PROCESS": {
                        "Regex": True,
                        "Jaccard": True,
                        "BM25": True,
                        # "Cosine": True,
                        "Keybert": True,
                    },
                },
            },
        },
        # -------------------------
        # RAGChat: checks applied at chat-time (user prompts + retrieved context)
        # - Masking is applied.
        # - Prompt-level LLM check is enabled (Check: True) with LLM params.
        # - Pipeline thresholds are tuned for runtime (smaller TOP_K for Keybert).
        # -------------------------
        "RAGChat": {
            "MASKING": {
                "APPLY_MASKING": True,
            },
            "PROMPT_CHECK": {
                "Check": True,
                "LLM_PARAM": {
                    # Deterministic LLM parameters for prompt-checking LLM runs
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                    "use_ollama_gpu": True,
                },
                "PIPELINE": {
                    "Jaccard": {
                        "CHAR_NGRAM_RANGE": (4, 6),
                        "THRESHOLD": 0.75,
                        "THRESHOLD_MIN": 0.5,
                    },
                    # Cosine is intentionally commented out; enable if vectors exist
                    # "Cosine": { "THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2 },
                    "Keybert": {
                        "THRESHOLD": 0.45,
                        "THRESHOLD_MIN": 0.2,
                        "TOP_K": 20,  # smaller TOP_K for chat-time performance
                    },
                    "BM25": {
                        "THRESHOLD": 0.7,
                        "THRESHOLD_MIN": 0.2,
                        "TERM_FREQ_SATURATION": 1.2,
                        "LENGTH_NORMALIZATION": 0.75,
                        "MIN_OVERLAP": 2,
                        "MIN_RAW_SCORE": 25,
                        "NORM_PERCENTILE": 97,
                    },
                    "Regex": {
                        "THRESHOLD": 1.0,
                        "THRESHOLD_MIN": 0.5,
                        "WINDOW_MAX_CHARS": 20,
                        "PREFIX_SUFFIX_LEN": 3,
                        "SEPARATOR_CLASS": r"[A-Za-z0-9_\-]",
                        "SOFT_SCORE_HARD": 1.0,
                        "SOFT_SCORE_FUZZY": 0.75,
                        "FUZZY_REGEX_EVAL_AFTER_HARD": False,
                        "Levenshtein": {"THRESHOLD": 0.5},
                    },
                    # Chat-time requires fewer algos above threshold to be strict but performant
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
                    "REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 3,
                    "ALGOS_TO_PROCESS": {
                        "Regex": True,
                        "Jaccard": True,
                        "BM25": True,
                        # "Cosine": True,
                        "Keybert": True,
                    },
                },
            },
            # PIPELINE_CHECK: checks applied to retrieved documents (post-retrieval)
            "PIPELINE_CHECK": {
                "PIPELINE": {
                    "Jaccard": {
                        "CHAR_NGRAM_RANGE": (4, 6),
                        "THRESHOLD": 0.75,
                        "THRESHOLD_MIN": 0.5,
                    },
                    # Cosine is intentionally commented out; enable if vectors exist
                    # "Cosine": { "THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2 },
                    "Keybert": {
                        "THRESHOLD": 0.4,
                        "THRESHOLD_MIN": 0.2,
                        "TOP_K": 20,
                    },
                    "BM25": {
                        "THRESHOLD": 0.7,
                        "THRESHOLD_MIN": 0.2,
                        "TERM_FREQ_SATURATION": 1.2,
                        "LENGTH_NORMALIZATION": 0.75,
                        "MIN_OVERLAP": 2,
                        "MIN_RAW_SCORE": 25,
                        "NORM_PERCENTILE": 97,
                    },
                    "Regex": {
                        "THRESHOLD": 1.0,
                        "THRESHOLD_MIN": 0.5,
                        "WINDOW_MAX_CHARS": 20,
                        "PREFIX_SUFFIX_LEN": 3,
                        "SEPARATOR_CLASS": r"[A-Za-z0-9_\-]",
                        "SOFT_SCORE_HARD": 1.0,
                        "SOFT_SCORE_FUZZY": 0.75,
                        "FUZZY_REGEX_EVAL_AFTER_HARD": False,
                        "Levenshtein": {"THRESHOLD": 0.5},
                    },
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 4,
                    "REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 4,
                    "ALGOS_TO_PROCESS": {
                        "Regex": True,
                        "Jaccard": True,
                        "BM25": True,
                        # "Cosine": True,
                        "Keybert": True,
                    },
                },
            },
        },
        # -------------------------
        # DocClassify: checks applied when classifying documents
        # - Prompt checks enabled with slightly different LLM params.
        # - Pipeline thresholds tuned for classification tasks (stricter).
        # -------------------------
        "DocClassify": {
            "MASKING": {"APPLY_MASKING": True},
            "PROMPT_CHECK": {
                "Check": True,
                "LLM_PARAM": {
                    "temperature": 0.1,
                    "top_k": 20,
                    "top_p": 0.8,
                    "use_ollama_gpu": True,
                },
                "PIPELINE": {
                    "Jaccard": {
                        "CHAR_NGRAM_RANGE": (4, 6),
                        "THRESHOLD": 0.75,
                        "THRESHOLD_MIN": 0.5,
                    },
                    # Cosine is intentionally commented out; enable if vectors exist
                    # "Cosine": { "THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2 },
                    "Keybert": {
                        "THRESHOLD": 0.45,
                        "THRESHOLD_MIN": 0.2,
                        "TOP_K": 20,
                    },
                    "BM25": {
                        "THRESHOLD": 0.7,
                        "THRESHOLD_MIN": 0.2,
                        "TERM_FREQ_SATURATION": 1.2,
                        "LENGTH_NORMALIZATION": 0.75,
                        "MIN_OVERLAP": 2,
                        "MIN_RAW_SCORE": 25,
                        "NORM_PERCENTILE": 97,
                    },
                    "Regex": {
                        "THRESHOLD": 1.0,
                        "THRESHOLD_MIN": 0.5,
                        "WINDOW_MAX_CHARS": 20,
                        "PREFIX_SUFFIX_LEN": 3,
                        "SEPARATOR_CLASS": r"[A-Za-z0-9_\-]",
                        "SOFT_SCORE_HARD": 1.0,
                        "SOFT_SCORE_FUZZY": 0.75,
                        "FUZZY_REGEX_EVAL_AFTER_HARD": False,
                        "Levenshtein": {"THRESHOLD": 0.5},
                    },
                    # For DocClassify we require more algos above threshold to be conservative
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 4,
                    "REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 4,
                    "ALGOS_TO_PROCESS": {
                        "Regex": True,
                        "Jaccard": True,
                        "BM25": True,
                        # "Cosine": True,
                        "Keybert": True,
                    },
                },
            },
            "PIPELINE_CHECK": {
                "PIPELINE": {
                    "Jaccard": {
                        "CHAR_NGRAM_RANGE": (4, 6),
                        "THRESHOLD": 0.75,
                        "THRESHOLD_MIN": 0.5,
                    },
                    # Cosine is intentionally commented out; enable if vectors exist
                    # "Cosine": { "THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2 },
                    "Keybert": {
                        "THRESHOLD": 0.4,
                        "THRESHOLD_MIN": 0.2,
                        "TOP_K": 20,
                    },
                    "BM25": {
                        "THRESHOLD": 0.7,
                        "THRESHOLD_MIN": 0.2,
                        "TERM_FREQ_SATURATION": 1.2,
                        "LENGTH_NORMALIZATION": 0.75,
                        "MIN_OVERLAP": 2,
                        "MIN_RAW_SCORE": 25,
                        "NORM_PERCENTILE": 96,  # slightly different percentile for classification
                    },
                    "Regex": {
                        "THRESHOLD": 1.0,
                        "THRESHOLD_MIN": 0.5,
                        "WINDOW_MAX_CHARS": 20,
                        "PREFIX_SUFFIX_LEN": 3,
                        "SEPARATOR_CLASS": r"[A-Za-z0-9_\-]",
                        "SOFT_SCORE_HARD": 1.0,
                        "SOFT_SCORE_FUZZY": 0.75,
                        "FUZZY_REGEX_EVAL_AFTER_HARD": False,
                        "Levenshtein": {"THRESHOLD": 0.5},
                    },
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
                    "REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 3,
                    "ALGOS_TO_PROCESS": {
                        "Regex": True,
                        "Jaccard": True,
                        "BM25": True,
                        # "Cosine": True,
                        "Keybert": True,
                    },
                },
            },
        },
    },  # end STRICT_DETECT_CONFIG
}  # end DETECTION_CONFIG

# RAGChatService is RAGChat served over HTTP — reuse identical compliance pipeline
_BANNED_DETECT["STRICT_DETECT_CONFIG"]["RAGChatService"] = (
    _BANNED_DETECT["STRICT_DETECT_CONFIG"]["RAGChat"]
)

# Banned words definitions
