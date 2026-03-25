# -------------------------------------------------------------------------
# - Lookup order: Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#   Config_Banned, Config_Models.py, Config_Globals.py
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# Do not change _FRIENDLY_NAME
_FRIENDLY_NAME = "RAGLoad"

# -----------------------------------------------------------------------------
# Load only files that were classified as OK by a previous DocClassify run.
# Pass via --load-from-classify-csv on the command line.
LOAD_FROM_CLASSIFY_CSV = False

# Also include files from the corresponding HUMAN_REVIEW CSV.
# Pass via --load-from-human-review-csv on the command line.
LOAD_FROM_HUMAN_REVIEW_CSV = False

# The run-stamp (YYYYMMDD_HHMMSS) that identifies the DocClassify log set.
# Required when --load-from-classify-csv is True.
# Pass via --classify-run-stamp <stamp> on the command line.
CLASSIFY_RUN_STAMP = ""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Process all files even if unchanged. Determined by file hash comparison
_PROCESS_IF_UNCHANGED = True
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Text splitting
# -----------------------------------------------------------------------------
_SEPARATORS = ["\n", " ", "."]  # Order matters: first match is used

# -----------------------------------------------------------------------------
# Banned word settings (apply to loaded chunks)
# -----------------------------------------------------------------------------

# Keys to extract
_CLASSIFICATION_KEYS = [
    "Status",
    "Time",
    "Stage",
    "Skip Status",
    "Skipped Chunks",
    "Inserted Chunks",
    "WordCount",
    "Temperature",
    "FilePath",
]

# -----------------------------------------------------------------------------
# Keyword extraction
# -----------------------------------------------------------------------------
_KEY_BERT = {
    "TOP_N_FIRST": 100,  # Keywords from first  KeyBERT pass
    "TOP_N_SECOND": 60,  # Keywords from second KeyBERT pass
}
