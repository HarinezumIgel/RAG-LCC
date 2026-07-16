import os

# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned.py, Config_Models.py, Config_Global.py
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# Do not change _FRIENDLY_NAME
_FRIENDLY_NAME = "RAGLoad"

_LOG_DIRECTORY = os.path.join(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    ),
    "logs",
    "RAGLoad",
)

# -----------------------------------------------------------------------------
# Path to a DocClassify CSV to use as an ingestion filter.
# When non-empty, only file paths listed in this CSV are loaded.
# Accepts a filename (resolved relative to _LOG_DIRECTORY) or an absolute path.
# Example: "DocClassify_OK_20260325_141005.csv"
# Pass via --load-from-classify-csv on the command line.
LOAD_FROM_CLASSIFY_CSV = ""

# Optional SQL WHERE clause applied when loading the classify CSV.
# The CSV is loaded into an in-memory SQLite table; only rows that
# satisfy the WHERE expression are included.  Standard SQLite syntax
# is supported: LIKE, AND, OR, NOT, =, !=, IN, GLOB, etc.
# Column names with spaces or special characters must be quoted: [Col Name].
# Example: "Mammal LIKE '%Yes%'"
#          "Mammal LIKE '%Yes%' AND Language = 'English'"
#          "Classification LIKE '%Science%' AND Mammal NOT LIKE '%Dont know%'"
# Pass via --classify-csv-query on the command line.
CLASSIFY_CSV_QUERY = ""
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
