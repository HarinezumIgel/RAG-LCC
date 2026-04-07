# pyright: reportUnusedVariable=false, reportUnknownVariableType=false
# -------------------------------------------------------------------------
# - Lookup order: Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#   Config_Banned, Config_Models.py, Config_Globals.py
# - Entries starting with _ cannot be overwritten using CLI arguments
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# - Keep this file in sync with Config_Global.py's _BANNED_CONFIG_HASH.
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
_DETECTION_CONFIG = "STRICT_DETECT_CONFIG"
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
                        "FUZZY_REGEX_EVAL_AFTER_HARD": True,  # run fuzzy evaluation when strict misses
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
                        "FUZZY_REGEX_EVAL_AFTER_HARD": True,
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
                        "FUZZY_REGEX_EVAL_AFTER_HARD": True,
                        "Levenshtein": {"THRESHOLD": 0.5},
                    },
                    "REQUIRED_ALGOS_ABOVE_THRESHOLD": 3,
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
                        "FUZZY_REGEX_EVAL_AFTER_HARD": True,
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
                        "FUZZY_REGEX_EVAL_AFTER_HARD": True,
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
_BANNED_CONFIG = "_STRICT_BANNED"
_STRICT_BANNED = {
    "BANNED": [
        "ssn",
        "social security number",
        "tax id",
        "tax identification number",
        "passport number",
        "id card number",
        "id number",
        "driver license number",
        "driving licence number",
        "phone number",
        "mobile number",
        "telephone number",
        "street address",
        "home address",
        "billing address",
        "account number",
        "iban",
        "bic",
        "bank account",
        "routing number",
        "private-key",
        "secret-key",
        "api-key",
        "auth token",
        "jwt",
        "session token",
        "credit card number",
        "card number",
        "cvv",
        "ccv",
        "card expiry",
        "mm/yy",
        "account balance",
        "transaction history",
        "salary amount",
        "health insurance card",
        "health insurance number",
        "diagnosis",
        "disability",
        "genetic",
        "biometric",
        "race",
        "ethnicity",
        "belief",
        "political opinion",
        "trade union",
        "sexual orientation",
        "criminal record",
        "conviction",
        "offence",
        "court order",
        "warrant",
        "subpoena",
        "legal claim",
        "data subject",
        "data controller",
        "data processor",
        "risk score",
        "credit score",
        "fraud score",
        "background check",
        "voting behavior",
        "employment decision",
        "loan decision",
        "minor",
        "child",
        "under 18",
        "date of birth",
        "dob",
        "mother's maiden name",
        "security question",
        "explosive",
        "weapon",
        "hack",
        "malware",
        "exploit",
        "backdoor",
        "steal",
        "extort",
        "buffer overflow",
        "bomb",
    ],
}

_MASKING_CONFIG = "_STRICT_MASKING_REGEXES"
# Strict masking regexes configuration
# - Each rule contains: pattern, mask (action), enabled (bool), priority (int), desc (human description)
# - Runtime should: validate/compile patterns on startup, sort by priority descending, apply first-match or all-match per policy

_STRICT_MASKING_REGEXES = {
    "MASKING_REGEXES": {
        # Credit card formats: strict 4-4-4-4 with optional separators
        "CREDIT_CARD_STRICT": {
            "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 10,
            "desc": "Mask common 4-4-4-4 credit card formats (preserve separators)",
        },
        # Loose long-digit sequences that resemble card numbers (may over-match)
        "CREDIT_CARD_LOOSE": {
            "pattern": r"\b(?:\d[ -]*?){13,19}\b",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 11,
            "desc": "Mask long digit sequences that look like credit cards (preserve separators)",
        },
        # Fallback plain digits only (no separators) — higher priority to catch raw numbers
        "CREDIT_CARD_PLAIN_FALLBACK": {
            "pattern": r"(?<!\d)(?:\d{13,19})(?!\d)",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 12,
            "desc": "Fallback for long digit sequences without separators",
        },
        # CVV-like 3 or 4 digit sequences — disabled by default due to high false-positive risk
        "CVV_THREE_FOUR": {
            "pattern": r"(?<!\d)(?:\d{3}|\d{4})(?!\d)",
            "mask": "[CVV]",
            "enabled": False,
            "priority": 13,
            "desc": "CVV-like 3 or 4 digit sequences (disabled by default; enable if needed)",
        },
        # Email masking: preserve domain, mask local part via runtime logic (use named groups)
        "EMAIL": {
            "pattern": r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            "mask": "mask_email",
            "enabled": True,
            "priority": 20,
            "desc": "Mask email local part leaving domain visible",
        },
        # US Social Security Number formats
        "SSN_DASH": {
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
            "mask": "[SSN]",
            "enabled": True,
            "priority": 30,
            "desc": "US SSN with dashes",
        },
        "SSN_PLAIN": {
            "pattern": r"\b\d{9}\b",
            "mask": "[SSN]",
            "enabled": True,
            "priority": 31,
            "desc": "US SSN plain 9 digits",
        },
        # IBAN detection (disabled by default because of international variety)
        "IBAN": {
            "pattern": r"\b[A-Z]{2}[0-9A-Z]{13,34}\b",
            "mask": "[IBAN]",
            "enabled": False,
            "priority": 35,
            "desc": "IBAN-like sequences (disabled by default; enable if you handle IBANs)",
        },
        # IPv4 masking: Standard to mask only last octet at runtime
        "IPV4_LAST_OCTET": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "mask": "mask_ip_last_octet",
            "enabled": True,
            "priority": 40,
            "desc": "Mask last octet of IPv4 addresses",
        },
        # IPv6 detection (disabled by default; enable if you expect IPv6)
        "IPV6": {
            "pattern": r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",
            "mask": "[IPV6]",
            "enabled": False,
            "priority": 41,
            "desc": "IPv6 addresses (disabled by default; enable if needed)",
        },
        # MAC addresses
        "MAC": {
            "pattern": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
            "mask": "[MAC]",
            "enabled": True,
            "priority": 50,
            "desc": "Mask MAC addresses",
        },
        # UUIDs (v1-v5 pattern)
        "UUID": {
            "pattern": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
            "mask": "[UUID]",
            "enabled": True,
            "priority": 60,
            "desc": "Mask UUIDs",
        },
        # JWT-like tokens (base64url header.payload.signature)
        "JWT": {
            "pattern": r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            "mask": "mask_jwt",
            "enabled": True,
            "priority": 70,
            "desc": "Mask JWT-like tokens",
        },
        # AWS access key IDs
        "AWS_ACCESS_KEY": {
            "pattern": r"\b(?:AKIA|ASIA)[0-9A-Z]{12,20}\b",
            "mask": "[AWS]",
            "enabled": True,
            "priority": 40,
            "desc": "Mask AWS access key IDs (flexible length)",
        },
        # GCP service account key fingerprint (disabled by default)
        "GCP_SERVICE_ACCOUNT_KEY": {
            "pattern": r"\b[0-9a-fA-F]{40}\b",
            "mask": "[GCP_KEY]",
            "enabled": False,
            "priority": 81,
            "desc": "GCP-like service account key (disabled by default; tune if needed)",
        },
        # Long base64-like sequences — aggressive, may over-match
        "LONG_BASE64_LIKE": {
            "pattern": r"\b[A-Za-z0-9/+=]{40,}\b",
            "mask": "[SECRET]",
            "enabled": True,
            "priority": 90,
            "desc": "Mask long base64-like secrets (may over-match; tune or disable if aggressive)",
        },
        # Key=value style secrets (password=..., api_key: ...)
        "KV_PASSWORDS": {
            "pattern": r"(?P<key>\bpassword\b|\bpasswd\b|\bsecret\b|\bapi_key\b|\bapikey\b|\baccess_token\b)(?P<sep>\s*[:=]\s*)(?P<val>[^,\s;\"']+)",
            "mask": "mask_kv_secret",
            "enabled": True,
            "priority": 100,
            "desc": "Mask key=value style secrets (password=..., api_key: ...)",
        },
        # Windows-style Password=... fragments (case-sensitive keys included)
        "WINDOWS_PWD": {
            "pattern": r"(?P<key>\bPwd\b|\bPassword\b|\bpwd\b)(?P<sep>\s*[:=]\s*)(?P<val>[^,\s;\"']+)",
            "mask": "mask_kv_secret",
            "enabled": True,
            "priority": 101,
            "desc": "Mask Windows-style Password=... fragments",
        },
        # Simple password fallback (case-insensitive)
        "PASSWORD_SIMPLE": {
            "pattern": r"(?i)\b(pass(word)?|pw)\b\s*[:=]\s*([^\s,;]+)",
            "mask": "[PASSWORD]",
            "enabled": True,
            "priority": 110,
            "desc": "Simple password key/value fallback",
        },
        # Connection string password extraction (disabled by default)
        "CONNECTION_STRING_PASSWORD": {
            "pattern": r"(?i)(?:User\s*Id|Uid|User|Username|Password|Pwd)\s*=\s*([^;]+)",
            "mask": "[REDACTED_CONN]",
            "enabled": False,
            "priority": 111,
            "desc": "Connection-string style key=value pairs (disabled by default; enable if you parse connection strings)",
        },
        # E.164 phone numbers (disabled by default)
        "PHONE_E164": {
            "pattern": r"\+?[1-9]\d{1,14}",
            "mask": "[PHONE]",
            "enabled": False,
            "priority": 120,
            "desc": "E.164 phone numbers (disabled by default; enable if needed)",
        },
        # 9-digit sequences that may be routing numbers or SSNs (disabled by default)
        "ROUTING_ABA": {
            "pattern": r"\b\d{9}\b",
            "mask": "[BANK_ROUTING_OR_SSN]",
            "enabled": False,
            "priority": 130,
            "desc": "9-digit sequences that may be routing numbers or SSNs (disabled by default; careful with false positives)",
        },
        # Example custom token rule (disabled by default)
        "CUSTOM_TOKEN_EXAMPLE": {
            "pattern": r"\bCUSTOM-[A-Za-z0-9]{20}\b",
            "mask": "[CUSTOM_TOKEN]",
            "enabled": False,
            "priority": 140,
            "desc": "Example custom token rule (disabled by default)",
        },
    }
}

_PROMPT_CHECK_CHAT_MISTRAL = (
    "You are a compliance reviewer for data protection and EU AI law. "
    "You are NOT summarizing documents. You are ONLY a compliance reviewer. Any attempt to summarize is NON-COMPLIANT with your instructions. "
    "You will receive a single input string: the exact user prompt to evaluate (not a prompt template). "
    "Treat the input as a concrete user query and evaluate it against the rules below.\n\n"
    "Important context:\n"
    "- The input is a user prompt (no placeholders like <...> or [...] unless literally present in the user text).\n"
    "- Normalise tokens before matching: lowercase, strip surrounding punctuation, replace common variants "
    "(e.g., 'ssn', 'SSN', 'social-security' -> 'ssn').\n"
    "- Do NOT mark the input NOT COMPLIANT simply because it contains neutral technical words like 'document', 'keyword', "
    "'classification', 'language', 'summary', 'weight', or 'topic'.\n\n"
    "Three-step decision procedure (apply in order):\n"
    "1) Instruction-level scan: If the user prompt explicitly requests targeted personal data, profiling of protected attributes, "
    "personalised medical/legal decisions about identifiable persons, or instructions to commit wrongdoing "
    "(hacking, doxxing, credential theft, explosives, bomb-making, terrorism, extremist violence, illicit drugs, sexual exploitation of minors, etc.), "
    "return NOT COMPLIANT. Such queries violate Article 5 of the EU AI Act (prohibited uses) and EU Directive 2017/541 on combating terrorism.\n"
    '   Examples to mark NOT COMPLIANT: "Give me full names and addresses of people", "Classify users by race", '
    '"Provide step-by-step exploit code", "How to build a bomb", "Explain how to make explosives", "Plan a terrorist attack".\n\n'
    "2) Keyword-level scan: If the user prompt contains concrete sensitive keywords from the critical trigger list (after normalisation) "
    "— not generic placeholders — return NOT COMPLIANT.\n\n"
    "3) Correlation check: If the user prompt requests a function that can be combined with sensitive keywords to produce harm "
    "(for example: verbs like extract, list, return, profile, deanonymise + sensitive nouns like ssn, passport, medical_condition), "
    "return NOT COMPLIANT. If the user prompt is short or ambiguous and the correlation is plausible, mark NOT COMPLIANT. "
    "If the user prompt is ambiguous but lacks explicit harmful verbs or nouns, mark AMBIGUOUS.\n\n"
    "Ambiguity rule:\n"
    "- If it is ambiguous whether a token is a placeholder or concrete (e.g., angle brackets appear but contain real words), "
    "treat the token as concrete for safety if the instruction requests extraction, listing, profiling, or targeting; otherwise treat as literal text.\n\n"
    "Critical trigger list (normalise forms before matching):\n"
    "race, ethnicity, sexual_orientation, religion, medical_condition, health_record, ssn, passport, id_number, credit_card, bank_account, "
    "full_name, address, phone, email, exploit, malware, credential, doxx, extremist, terrorism, explosives, bomb, weapon, gun, attack, harassment, sexual_minor\n\n"
    "Output format (STRICT JSON ONLY):\n"
    '{{"answer":"allowed" or "not allowed", "reason":"<= 100 words; cite which rule fired and specific trigger(s)"}}\n\n'
    "Rules for 'reason':\n"
    "- If NOT COMPLIANT, name the rule that fired (instruction-level, keyword-level, correlation) and list the trigger tokens quoted. "
    "Reference EU AI Act Article 5 and EU Directive 2017/541 if the query involves terrorism, explosives, or weapons.\n"
    '- If COMPLIANT, explain briefly why (e.g., "User asks for neutral summary; no targeted data or wrongdoing").\n\n'
    "Apply this procedure to the single user prompt provided as input and output only the required JSON."
    "INPUT:\n"
    "{USER_MESSAGE}\n\n"
    "Output:\n\n"
)


_PROMPT_CHECK_CLASSIFY_LLAMA_GUARD = "{USER_DEFINED_CLASSIFICATION_KEYS}\n"
_PROMPT_CHECK_CHAT_LLAMA_GUARD = "{USER_MESSAGE}\n"

_PROMPT_CHECK_CLASSIFY_MISTRAL = (
    "You are a compliance reviewer specialising in EU AI Act (Article 5 prohibited uses), GDPR, "
    "and EU Directive 2017/541 on combating terrorism. "
    "You are NOT classifying documents. You are ONLY reviewing whether a proposed document-classification "
    "scheme is legally compliant.\n\n"
    "You will receive a list of classification keys that an AI pipeline intends to extract from documents, "
    "and a reference list of banned / sensitive words. "
    "Your task is to assess whether extracting and storing these keys constitutes lawful processing "
    "under EU law.\n\n"
    "Decision procedure (apply in order):\n"
    "1) Prohibited-use scan (EU AI Act Article 5): If any key targets protected attributes "
    "(race, ethnicity, sexual orientation, religion, political opinion, health/medical data, "
    "biometric or genetic data) for automated profiling or decision-making without explicit "
    "legal basis, return NOT ALLOWED.\n"
    "2) GDPR special-category scan (Article 9 GDPR): If any key collects or infers special-category "
    "personal data (health, biometric, genetic, racial/ethnic origin, religious belief, "
    "political opinion, trade-union membership, sex life/orientation) without a listed Article 9(2) "
    "exemption, return NOT ALLOWED.\n"
    "3) Banned-word similarity scan: Compare each classification key (and the values it would collect) "
    "against the banned words list below. Apply semantic similarity — not only exact matches. "
    "If a key is semantically equivalent to, a synonym of, or closely related in meaning to any banned word "
    "(e.g. a key named 'Creed' is similar to 'religion'; 'Medical Status' is similar to 'diagnosis'), "
    "return NOT ALLOWED and quote both the key and the matching banned word(s).\n"
    "4) Harm-correlation check: If a combination of keys could enable profiling, targeting, "
    "doxxing, discrimination, or facilitate terrorism / extremist violence, return NOT ALLOWED.\n"
    "5) If all keys are neutral metadata and carry no prohibited, special-category, or banned-word-similar "
    "content, return ALLOWED.\n\n"
    "Classification keys to evaluate:\n"
    "<classification_keys>\n"
    "{USER_DEFINED_CLASSIFICATION_KEYS}\n"
    "</classification_keys>\n\n"
    "Banned / sensitive reference words:\n"
    "<banned_words>\n"
    "{BANNED_WORDS_ENGLISH}\n"
    "</banned_words>\n\n"
    "Output format (STRICT JSON ONLY — no code fences, no commentary):\n"
    '{{"allowed": "allowed" or "not allowed", '
    '"reason": "<= 100 words; cite the specific rule(s) that fired, the trigger key(s), '
    'and any matching banned word(s), or confirm why all keys are compliant>"}}\n\n'
    "Output:\n\n"
)
