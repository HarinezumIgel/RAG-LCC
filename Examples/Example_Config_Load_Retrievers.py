"""Load/retriever collection and store settings."""

# pylint: disable=invalid-name

import os

from Configuration import Config_Global

# -----------------------------------------------------------------------------
# Retrieval stores handling on startup (ChromaDB collection, BM25 index,
# graph index, regex index)
# -----------------------------------------------------------------------------
#   RAGLoad only, variable is here so it can be passed as CLI argument
#   True  = preserve existing collection, BM25 index, graph index, and regex index
#   False = wipe and recreate on each run
RETRIEVAL_STORES_KEEP = False

# -----------------------------------------------------------------------------
# Retrieval stores
#
# The four stores below are managed as a single unit by RETRIEVAL_STORES_KEEP:
#   - False -> all four are deleted and rebuilt together on every RAGLoad run.
#   - True  -> all four must exist on disk; a missing index aborts startup.
#
# Each path can be set independently, as long as all four stay in sync.
# Paths MUST be inside the project root validated by
# Config_Global.require_absolute_path(): delete_file_or_dir()
# enforces a jailbreak guard that refuses to delete anything outside it.
# -----------------------------------------------------------------------------

# Directory where ChromaDB stores its document embeddings
_CHROMA_DB_DIR = os.path.join(
    Config_Global.require_absolute_path("Config_Load_Retrievers"),
    "chromadb",
    "docs",
)

# -----------------------------------------------------------------------------
# The collection load and queries are done with (you can override this setting
# or invoke RAGLoad.py with the --collection parameter and switch the collection
# in RAGChat.py using collection! command in chat
# -----------------------------------------------------------------------------
COLLECTION = "Test"

# =============================================================================
# Collection retrieval/index tuning
# =============================================================================
# These settings are part of collection schema behavior in practice.
# Changing any retrieval/indexing shape should be followed by a reload run with:
#
#     RETRIEVAL_STORES_KEEP = False
# =============================================================================

_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG = "THOROUGH"

_CHROMA_EMBED_AND_RETRIEVE_PARAMS: dict[str, dict[str, float | int]] = {
    # ==========================
    # Variant: THOROUGH
    # ==========================
    # More HNSW neighbours -- favors recall and context
    # at the cost of index build time and query latency.
    "THOROUGH": {
        "NEIGHBORS_ON_LOAD": 512,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 512,  # Explore more neighbours at query (chat) time.
    },
    # ==========================
    # Variant: COMPACT
    # ==========================
    # Fewer neighbours -- favors precision and speed.
    "COMPACT": {
        "NEIGHBORS_ON_LOAD": 64,  # Explore more neighbours at load time. Affects Load.py
        "NEIGHBORS_RETRIEVE": 64,  # Explore more neighbours at query (chat) time.
    },
}
