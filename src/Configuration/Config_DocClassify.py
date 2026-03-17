# -------------------------------------------------------------------------
# Configuration Lookup Rules
# -------------------------------------------------------------------------
# - Lookup order:
#     Config_<RAGChat.py | Config_RAGLoad.py | Config_DocClassify.py>,
#     Config_Banned, Config_Models.py, Config_Globals.py
# - Entries starting with '$' are resolved via indirect lookup
# - Top-level configuration keys are expected to be uppercase
# -------------------------------------------------------------------------

"""
Configuration for RAGChatAndDocClassify

This file defines constant configuration values consumed by the pipeline.
Most settings can be overridden via CLI arguments (see ChunkAndExtract.py -h).
"""

# ---------------------------------------------------------------------------
# Friendly configuration identifier
# ---------------------------------------------------------------------------
_FRIENDLY_NAME = "DocClassify"

# ---------------------------------------------------------------------------
# Extraction / Classification LLM parameter variants
# ---------------------------------------------------------------------------
# Purpose:
# - Control sampling behavior for extraction-oriented LLM usage
# - Favor predictable token selection and constrained output where applicable
#
# Notes:
# - Lower temperature reduces sampling variability
# - Smaller top_k / top_p values constrain the candidate token space
# - Backend implementations may differ in how strictly these parameters
#   are applied

_ACTIVE_EXTRACTION_CONFIG = "STRICT"  # Options: "STRICT", "BALANCED", "RECALL"

_EXTRACTION_MODEL_PARAMS: dict[str, dict[str, float | int]] = {
    # ==========================
    # Variant: STRICT
    # ==========================
    # Intended for highly constrained extraction scenarios where the model
    # is expected to abstain when confidence is low.
    "STRICT": {
        "TEMPERATURE_EXT": 0.0,  # Minimal sampling variability (backend-dependent)
        "TOP_K_EXT": 1,  # Very small candidate set
        "TOP_P_EXT": 1.0,  # Neutral nucleus value to avoid interaction effects
    },
    # ==========================
    # Variant: BALANCED
    # ==========================
    # Intended for cases where extremely tight constraints lead to unstable
    # formatting or incomplete outputs.
    "BALANCED": {
        "TEMPERATURE_EXT": 0.0,
        "TOP_K_EXT": 10,  # Small but non-minimal candidate pool
        "TOP_P_EXT": 0.85,  # Restricts low-probability tokens
    },
    # ==========================
    # Variant: RECALL
    # ==========================
    # Intended for scenarios requiring broader token exploration while still
    # remaining within a constrained sampling regime.
    "RECALL": {
        "TEMPERATURE_EXT": 0.1,  # Slightly increased variability
        "TOP_K_EXT": 40,
        "TOP_P_EXT": 0.92,
    },
}

# ---------------------------------------------------------------------------
# KeyBERT extraction parameters (double-pass)
# ---------------------------------------------------------------------------
# Purpose:
# - PASS 1: extract multi-word phrases carrying relational or descriptive meaning
# - PASS 2: normalize to unigrams for downstream processing
#
# Notes:
# - Lower TOP_N values reduce candidate volume
# - Smaller n-gram ranges limit long or weakly bound phrases
# - PASS 2 typically uses fewer candidates than PASS 1

_ACTIVE_KEYBERT_CONFIG = "STRICT"  # Options: "STRICT", "BALANCED", "RECALL"

_KEY_BERT: dict[str, dict[str, int | tuple[int, int]]] = {
    # ==========================
    # Variant: STRICT
    # ==========================
    # Intended for low-noise extraction pipelines where candidate volume
    # should be tightly controlled.
    "STRICT": {
        "TOP_N_FIRST": 60,  # Limited phrase candidate set
        "TOP_N_SECOND": 30,  # Restricted unigram list
        "NGRAM_PASS1": (1, 4),  # Constrains phrase length
        "NGRAM_PASS2": (1, 1),  # Unigrams only
    },
    # ==========================
    # Variant: BALANCED
    # ==========================
    # Intended for moderate expansion of candidate space while retaining
    # some control over noise.
    "BALANCED": {
        "TOP_N_FIRST": 80,
        "TOP_N_SECOND": 50,
        "NGRAM_PASS1": (1, 5),
        "NGRAM_PASS2": (1, 1),
    },
    # ==========================
    # Variant: RECALL
    # ==========================
    # Intended for broader candidate generation, with the expectation that
    # downstream filtering or scoring will be applied.
    "RECALL": {
        "TOP_N_FIRST": 150,
        "TOP_N_SECOND": 100,
        "NGRAM_PASS1": (1, 6),
        "NGRAM_PASS2": (1, 1),
    },
}

# ---------------------------------------------------------------------------
# Reverse stemming: when True, values for _YOUR_CLASSIFICATION_KEYS in the
# classification output are post-processed to substitute stems back to their
# best original surface word before any CSV is written.
# ---------------------------------------------------------------------------
REVERSE_STEMMING = True


# ---------------------------------------------------------------------------
# Character replacements
# ---------------------------------------------------------------------------
# Map of characters to replace in extracted text to normalize tokens.
_UNWANTED_CHAR_MAP = {
    "ß": "ss",
}

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  CUSTOMISE HERE — classification keys                             ║
# ║  Add, remove or reorder the keys the LLM must return.             ║
# ║  If you change these, update the prompt templates below to match. ║
# ╚═══════════════════════════════════════════════════════════════════╝

_YOUR_CLASSIFICATION_KEYS = [
    "Classification",
    "Purpose",
    "Topic",
    "Animal",
    "Mammal",
    "Language",
]  # For user‑defined keys beyond the core set
# ╚═══════════════════════════════════════════════════════════════════╝

_CLASSIFICATION_KEYS = [
    "Status",
    "Time",
    "Stage",
    "FilePath",
    *_YOUR_CLASSIFICATION_KEYS,  # User‑defined keys can be added here
    "Temperature",
    "WordCount",
    "FileType",
    "CreationDate",
    "FileHash",
]
# ---------------------------------------------------------------------------
# Limits for generated labels and summaries
# ---------------------------------------------------------------------------
# CLASSIFICATION_WORD_CNT is derived from the number of user-defined keys.
# Update _USER_DEFINED_CLASSIFICATION_KEYS to change this automatically.
CLASSIFICATION_WORD_CNT = len(_YOUR_CLASSIFICATION_KEYS)
SUMMARY_SENTENCE_CNT = 10

# ---------------------------------------------------------------------------
# Prompt templates for different LLMs
# ---------------------------------------------------------------------------
# These prompts expect a dictionary of weighted keywords and instruct the LLM
# to return a single JSON object with the specified keys only.
# If you change keys here, update _CLASSIFICATION_KEYS accordingly.


_PROMPT_CLASSIFY_MISTRAL = (
    # System prompt to instruct the LLM on how to perform classification based on weighted keywords.
    "You are given a dictionary of keywords extracted from a document, "
    "each key is a keyword and its value is a relevance weight. Analyze these "
    "keywords and determine:\n"
    # Task prompt
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  CUSTOMISE HERE — classification fields                           ║
    # ║  Add, remove or modify the numbered fields below.                 ║
    # ║  Names MUST match entries in _CLASSIFICATION_KEYS.                ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    "1. Classification: category labels (up to {CLASSIFICATION_WORD_CNT} words).\n"  # The general knowledge of the LLM is wanted. The system prompt allows the LLM to use its general knowledge
    "2. Purpose: brief summary (up to {SUMMARY_SENTENCE_CNT} sentences).\n"
    "3. Language: detected document language.\n"
    "4. Topic: short topic phrase.\n"
    "5. Animal: What animals are discussed.\n"
    "6. Mammal: For *each* animal from field 5, decide whether it is a mammal\n"  # Here the knowledge from the context is wanted.
    "using ONLY the information present in the provided keywords.\n"
    "DO NOT use your own biological knowledge. If the keywords do not\n"
    "explicitly say an animal is or is not a mammal, you MUST answer Dont know.\n"
    "Answer strictly one of: Yes, No, or Dont know.\n"
    "Output format: animal: answer, e.g. AnimalA: yes, AnimalB: dont know\n"
    "CRITICAL: The Mammal field MUST contain exactly one answer for EVERY animal listed in Animal.\n"
    "Never collapse or merge answers. If there are 3 animals, there must be 3 separate answers.\n"
    # ╚═══════════════════════════════════════════════════════════════════╝
    "\n"
    "IMPORTANT: Return ONLY ONE valid JSON object. No explanation, commentary, or code fences.\n"
    "The JSON object must contain exactly these keys: "
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  CUSTOMISE HERE — JSON keys returned by the LLM                   ║
    # ║  Must mirror the numbered fields above.                           ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    '"Classification", "Purpose", "Language", "Topic", "Animal", "Mammal".\n'
    # ╚═══════════════════════════════════════════════════════════════════╝
    "Every value MUST be a plain string. Never use nested objects, arrays, or sets as values.\n"
    "Only mention animals that actually appear in the keywords. Do not invent or add animals from the example.\n"
    "The response must start with {{{{ and end with }}}}.\n"
    "Example output:\n"
    '{{{{"Classification": "Science", "Purpose": "A document summary", '
    '"Language": "English", "Topic": "Research", '
    '"Animal": "AnimalA, AnimalB", "Mammal": "AnimalA: yes, AnimalB: dont know"}}}}\n\n'
    "Weighted Keywords: "
)

_PROMPT_CLASSIFY_LLAMA = (
    # System prompt to instruct the LLM on how to perform classification based on weighted keywords.
    "You are given a dictionary of keywords extracted from a document, "
    "each key is a keyword and its value is a relevance weight. Analyze these "
    "keywords and determine:\n"
    # Task prompt
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  CUSTOMISE HERE — classification fields                           ║
    # ║  Add, remove or modify the numbered fields below.                 ║
    # ║  Names MUST match entries in _CLASSIFICATION_KEYS.                ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    "1. Classification: category labels (up to {CLASSIFICATION_WORD_CNT} words).\n"  # The general knowledge of the LLM is wanted. The system prompt allows the LLM to use its general knowledge
    "2. Purpose: brief summary (up to {SUMMARY_SENTENCE_CNT} sentences).\n"
    "3. Language: detected document language.\n"
    "4. Topic: short topic phrase.\n"
    "5. Animal: What animals are discussed.\n"
    "6. Mammal: For *each* animal from field 5, decide whether it is a mammal\n"  # Here is the knowledge from the context is wanted.
    "using ONLY the information present in the provided keywords.\n"
    "DO NOT use your own biological knowledge. If the keywords do not\n"
    "explicitly say an animal is or is not a mammal, you MUST answer Dont know.\n"
    "Answer strictly one of: Yes, No, or Dont know.\n"
    "Output format: animal: answer, e.g. AnimalA: yes, AnimalB: dont know\n"
    "CRITICAL: The Mammal field MUST contain exactly one answer for EVERY animal listed in Animal.\n"
    "Never collapse or merge answers. If there are 3 animals, there must be 3 separate answers.\n"
    # ╚═══════════════════════════════════════════════════════════════════╝
    "\n"
    "IMPORTANT: Return ONLY ONE valid JSON object. No explanation, commentary, or code fences.\n"
    "The JSON object must contain exactly these keys: "
    # ╔═══════════════════════════════════════════════════════════════════╗
    # ║  CUSTOMISE HERE — JSON keys returned by the LLM                   ║
    # ║  Must mirror the numbered fields above.                           ║
    # ╚═══════════════════════════════════════════════════════════════════╝
    '"Classification", "Purpose", "Language", "Topic", "Animal", "Mammal".\n'
    # ╚═══════════════════════════════════════════════════════════════════╝
    "Every value MUST be a plain string. Never use nested objects, arrays, or sets as values.\n"
    "Only mention animals that actually appear in the keywords. Do not invent or add animals from the example.\n"
    "The response must start with {{ and end with }}.\n"
    "Example output:\n"
    '{{"Classification": "Science", "Purpose": "A document summary", '
    '"Language": "English", "Topic": "Research", '
    '"Animal": "AnimalA, AnimalB", "Mammal": "AnimalA: yes, AnimalB: dont know"}}\n\n'
    "Weighted Keywords: "
)

# ---------------------------------------------------------------------------
# End of configuration
# ---------------------------------------------------------------------------
# Notes:
# - Preserve the duplicate "Stage" entry if other parts of the code rely on it.
# - If you change any prompt keys, update _CLASSIFICATION_KEYS.
