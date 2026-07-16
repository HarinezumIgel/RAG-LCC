<!-- markdownlint-disable MD033 -->
# 📚 RAG‑LCC — Configuration Reference

← Back to [README](README.md) · See also: [INSTALL.md](INSTALL.md) · [EXAMPLES.md](EXAMPLES.md) · [ARCHITECTURE.md](ARCHITECTURE.md)

All settings live under [`src/Configuration/`](src/Configuration/).
Configuration files are split by concern; this reference walks through every
public knob the apps read at startup.

## 📑 Lookup order

RAG-LCC uses **six** configuration files, all located under `src/Configuration/`. They are loaded in a fixed precedence order (highest wins):

> **CLI args** are checked before any file and override everything (except `_`-prefixed keys).

1. **App-specific** — `Config_RAGChat.py`, `Config_RAGLoad.py`, or `Config_DocClassify.py` (highest file priority)
2. **Config_WebSearch.py** — web search master switch, backend, compliance gates
3. **Config_Banned.py** — detection algorithms, thresholds, banned words, masking rules
4. **Config_Models.py** — embedding, cross-encoder, and LLM model definitions
5. **Config_Global.py** — shared defaults (paths, hardware, ChromaDB, token budget, debug)
6. **Config_Internet_Env.py** — internet access, network tracing, and offline toggles (environment variables only)

**Notes:**

- Files 1-5 are loaded as Python modules and therefore require valid Python syntax.
- `Config_Internet_Env.py` contains **only environment variables** (no regular config keys).
- Keys starting with `_` are internal and **cannot** be overridden via CLI arguments.
- Keys starting with `$` are indirect lookups (the value names another config key).
- Top-level settings must be **UPPERCASE**.
- CLI overrides apply **only** to `Config_Global.py` and the **app-specific** config (`Config_RAGChat.py`, `Config_RAGLoad.py`, or `Config_DocClassify.py`). Keys in `Config_Models.py`, `Config_Banned.py`, `Config_WebSearch.py`, and `Config_Internet_Env.py` are **not** exposed as CLI arguments.

## 🌐 1. Config_Global.py — Shared Defaults

### 💻 Hardware and Device

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `USE_CPU` | `False` | Force CPU-only mode. Set `EMBEDDER_BITS = 32` when enabled. |
| `EMBEDDER_BITS` | `32` | Quantisation for embeddings. Use `32` on CPU; `16` on GPU (requires `accelerate`). |

### 📁 Paths

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `DOC_DIR` | `<project>/TestDocs` | Root folder for documents to load or classify. Searched recursively (subdirectories are included). |
| `_EXCLUSIONS_DIR` | `<project>/Exclusions` | Directory for per-collection exclusion CSVs. |
| `_CHROMA_DB_DIR` | `<project>/chromadb/docs` | ChromaDB persistent storage. |
| `TESSERACT_PATH` (env var) | `C:\Program Files\Tesseract-OCR\tesseract.exe\|/usr/bin/tesseract` | OS-aware Tesseract OCR path. Format: `"windows_path\|linux_path"`. The framework automatically selects the appropriate path based on the current platform. Set in `Config_Internet_Env.py`. |

### 🦙 Ollama

The Ollama endpoint, streaming mode, and GPU flag have moved into `_MODELS["ollama"]["_OLLAMA"]`
in `Config_Models.py` (see the "Inference Endpoint Provider Metadata" section below).
The remaining Ollama-related setting in `Config_Global.py` is:

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `REQUEST_TIMEOUT` | `600` | Seconds to wait for an Ollama response before timing out. |

### 🎫 Token Budget

Token budget settings are now configured **per model role** in `Config_Models.py`.
Each model entry (`_LLM`, `_LLM_CHK`, `_LLM_REWRITE_PROMPT`) declares its own
`TOKEN_BUDGET_CONTEXT_CAP`, `TOKEN_BUDGET_RESERVED_OUTPUT`, and
`TOKEN_BUDGET_RESERVED_SYSTEM` values.

`TokenBudget` reads these values from the active model role at runtime. When
`model_role` is provided (e.g., `"_ACTIVE_LLM_CHK"` for compliance checks), the
corresponding model's specific budget values are used.

**Typical values:**

| Model Role | Context Cap | Reserved Output | Reserved System | Purpose |
| --- | --- | --- | --- | --- |
| Main LLM (`_LLM`) | `32768` | `2048` | `1024` | Standard chat/classify responses |
| Rewrite (`_LLM_REWRITE_PROMPT`) | `32768` | `2048` | `1024` | Query expansion/topic detection |
| Check (`_LLM_CHK`) | `32768` | `2048` | `1024` | General compliance checks |
| Guard (`llama_guard._LLM_CHK`) | `32768` | `64` | `64` | Short safe/unsafe verdicts |

- **Context Cap**: Hardware limit. If the backend reports a larger context window,
  this cap is enforced instead.
- **Reserved Output**: Maximum tokens allocated for model reply.
- **Reserved System**: Tokens reserved for system/instruction preamble.

See `Config_Models.py` for model-specific values.

### 🗄️ ChromaDB and Chunking

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `RETRIEVAL_STORES_KEEP` | `False` | `True` = preserve existing collection, BM25 index, and graph index on startup. `False` = wipe and recreate. Applies to `RAGLoad.py` **only** |
| `COLLECTION` | `"Test"` | Active ChromaDB collection name. Override with `--collection` on the CLI. |

All collection-defining settings — HNSW neighbour counts, chunker selection,
chunk sizes, and AUTO_CHUNK routing — are grouped under the **COLLECTION SCHEMA**
section in `Config_Global.py`. They live here (not in `Config_RAGLoad.py`)
so that RAGLoad and RAGChat both refer to the same values
([Lookup order](#-lookup-order)).
Switching any value requires dropping and reloading the collection
(`RETRIEVAL_STORES_KEEP = False`).

```python
_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG = "THOROUGH"   # selector: "THOROUGH" or "COMPACT"

_CHROMA_EMBED_AND_RETRIEVE_PARAMS = {
    "THOROUGH": {
        "NEIGHBORS_ON_LOAD": 512,     # HNSW neighbours explored at index time (RAGLoad)
        "NEIGHBORS_RETRIEVE": 512,    # HNSW neighbours explored at query time (RAGChat)
    },
    "COMPACT": {
        "NEIGHBORS_ON_LOAD": 64,
        "NEIGHBORS_RETRIEVE": 64,
    },
}
```

Chunking strategy, per-file-type routing, and chunker parameters are configured
via `_ACTIVE_CHUNKER_CONFIG`, `_CHUNK_STRATEGY`, and `_CHUNKERS` — also in the COLLECTION SCHEMA
section.  `_CHUNKERS.SEMANTIC.MIN_SENTENCE_WORDS` (default **15**) merges
consecutive short fragments before embedding, which improves retrieval for
PDF tables and spec sheets. For the full reference and chunker descriptions, see
[Chunking Architecture in ARCHITECTURE.md](ARCHITECTURE.md#-chunking-architecture).

> **Note:** The following chunk sizes were observed during experimentation:

- technical papers: 256-512
- short articles: 128-256
- long documents: 512-1024.

The `PDF_PAGE` chunker (used when `_CHUNK_STRATEGY` maps `"pdf"` to `"PDF_PAGE"`) exposes two tunable parameters in `_CHUNKERS.PDF_PAGE`:

| Key | Default | Purpose |
| --- | --- | --- |
| `MAX_CHUNK_SIZE` | `200` | Maximum words per page chunk. Dense pages are split at sentence boundaries until every sub-chunk is at or below this limit. Reducing this value improves retrieval precision for documents where multiple unrelated topics are packed onto a single page (e.g. safety notices followed immediately by environmental specifications). |
| `PRESERVE_NEWLINES` | `False` | When `True`, newlines in the extracted PDF text are kept, which can help tables and bulleted lists stay legible in the chunk. |

### 🔗 Retrieval Stores & Search Modes

RAG‑LCC maintains three retrieval stores per collection:

| Store | Persisted at | Purpose |
| --- | --- | --- |
| **ChromaDB** (vector) | `chromadb/<collection>/` | Dense embedding nearest-neighbour search |
| **BM25 index** | `chromadb/bm25/<collection>/bm25_index.pkl.gz` | Keyword Okapi BM25 scoring |
| **Graph index** | `chromadb/graph/<collection>/graph_index.pkl.gz` | Entity co-occurrence graph traversal |

Both the BM25 and graph indexes are built automatically during `RAGLoad` and loaded on demand during `RAGChat`. When `RETRIEVAL_STORES_KEEP = False` all three stores are deleted together before a reload.

**Graph index and spaCy**
The graph retriever uses [spaCy](https://spacy.io/) (`en_core_web_sm`, **MIT license**, © Explosion AI) for named-entity recognition (NER) and noun-phrase extraction. The `en_core_web_sm` model is downloaded as a separate step (`python -m spacy download en_core_web_sm`); it is not bundled with this project. Entity types and extraction behaviour are configured in `_GRAPH_INDEX` in `Config_Global.py`.

The `_BM25_INDEX` slot in `Config_Global.py` controls BM25 Okapi scoring and
Reciprocal Rank Fusion (RRF) parameters:

```python
_BM25_INDEX = {
    "BM25_INDEX_DIR": "<project_root>/chromadb/bm25",
    "k1": 1.2,      # BM25 term-frequency saturation
    "b":  0.75,     # BM25 document-length normalisation
    "rrf_k": 60,    # RRF fusion constant (ALL / *_GRAPH modes)
}
```

| Key | Default | Purpose |
| --- | --- | --- |
| `BM25_INDEX_DIR` | `<project_root>/chromadb/bm25` | Root directory where per-collection BM25 index subdirectories are created. |
| `k1` | `1.2` | Controls how fast term-frequency gains saturate. Higher values give more weight to repeated terms. |
| `b` | `0.75` | Document-length normalisation factor (0 = no normalisation, 1 = full normalisation). |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant. A large value (e.g. 60) favours agreement between lists over individual rank position; a small value (e.g. 1) favours top-ranked items. 60 is the standard value from the original RRF paper. |

The `retrieve_mode` parameter (set per strategy in `Config_RAGChat.py` or at
query time via `retrieve_mode=ALL`) selects the retrieval mode:

| Mode | Stores | Description |
| --- | --- | --- |
| `VECTOR` | ChromaDB | Embedding-based retrieval only. |
| `BM25` | BM25 | Keyword BM25 Okapi retrieval only. |
| `GRAPH` | Graph | Entity co-occurrence graph traversal only. |
| `VECTOR_BM25` | ChromaDB + BM25 | Both; merged via RRF. |
| `VECTOR_GRAPH` | ChromaDB + Graph | Both; merged via RRF. |
| `BM25_GRAPH` | BM25 + Graph | Both; merged via RRF. |
| `ALL` | ChromaDB + BM25 + Graph | All three stores merged via RRF (default). |
| `WEB` | Internet (DuckDuckGo) | Web-only retrieval; local stores are skipped. Requires `WEB_SEARCH_MODE = "1"` (from `Config_Internet_Env.py`). |

> **When to use GRAPH / graph-combined modes:** The graph retriever seeds traversal from entities and noun phrases extracted from the query. It excels at pulling in thematically connected chunks (e.g. "hedgehog diet" → co-occurring anatomy or habitat chunks). For generic summary queries ("what animals are described?") the vector or BM25 legs carry the load; the graph leg contributes 0 results but does not harm the merge. `ALL` is therefore a safe default.

You can inspect the persisted indexes with the included utilities:

```bash
python src/Scripts/BM25IndexInspector.py   -path chromadb/bm25/Test   -chunks 5
python src/Scripts/GraphIndexInspector.py  -path chromadb/graph/Test  -chunks 5 -edges 10
```

### 📎 Office Document Extraction

```python
_OFFICE_DOC_EXTRACTION = {
    "Word": True,
    "Power Point": True,
    "Excel": True,
}
```

Set a value to `False` if the corresponding Microsoft Office component is not installed.

Microsoft Office is a separately licensed product and is **not** provided by RAG‑LCC.

### ⛔ Exclusion

These settings control whether excluded files are skipped during processing. For the full design, see [Exclusion + Incremental Hash Check in ARCHITECTURE.md](ARCHITECTURE.md#exclusion--incremental-hash-check-skip-unchanged-files).

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `USE_EXCLUSIONS` | `False` | Skip files listed in the per-collection exclusion CSV. Excluded files are those flagged for human review. `RAGLoad` and `DocClassify`. |

### 🐛 Debug Levels

```python
DEBUG_LEVEL = "30"    # Default used in this repository (Standard, >= 30).
                      # Format: 30  (>=30)   ge 30  (>=30)   is 30  (==30)   none  (silent)
```

`DEBUG_LEVEL` is a **single combined key** — the comparison mode is encoded in the
string alongside the numeric level.  There is no separate `DEBUG_MODE` config key;
mode is embedded as an optional prefix: `"ge 30"` means ≥ 30, `"is 30"` means == 30.

Levels are spaced by 10 to allow future insertions without renumbering.
`ge` (the default) activates the selected level **and all levels above it**.
`is` activates **only the exact level specified** — useful for isolating
one subsystem without the noise from higher levels.
`le` activates the selected level **and all levels below it** — useful for
capping output to a maximum verbosity.

| Level | Label | What it shows |
| --- | --- | --- |
| 0 | None | Silent — no diagnostic output |
| 10 | Basic | High-level pass/fail outcomes across all subsystems |
| 20 | Service | API request dump, service lifecycle events |
| 29 | Query Rewrite | Rewrite decisions and topic-detect results |
| 30 | Standard | **Default.** Pipeline flow: session state, retrieval decisions, compliance outcomes |
| 31 | Prompt Check Input | Text fed to the banned-phrase filter chain + system-message hint (skipped, not checked) |
| 32 | Chunk Content | Full text and metadata of every chunk selected for the LLM context window (file, chunk_id, FileHash, all scores, retriever sources, raw `page_content`) |
| 40 | Algos | Scorer internals, masker rules, accumulator, keyword extraction |
| 55 | Components | Synonym detail, argostranslate, transformers, URL logging |
| 60 | Chat Prompt | Full prompt text sent to the LLM |
| 70 | Extracted Content | Raw document content from classification |
| 80 | Ollama Response | Raw Ollama request/response bodies |
| 100 | Streaming | Per-chunk raw streaming output |

You can also use the string form when setting the debug level at the interactive prompt
or via the API `debug_level` field:

| String form | Equivalent | Meaning |
| --- | --- | --- |
| `"30"` | `debug_level=30, mode=ge` | Standard (≥ 30), default mode |
| `"ge 30"` or `">= 30"` | `debug_level=30, mode=ge` | Explicit greater-equal |
| `"is 30"` or `"== 30"` | `debug_level=30, mode=is` | Exact level only |
| `"le 30"` or `"<= 30"` | `debug_level=30, mode=le` | This level and all below (≤ 30) |
| `"none"` | `debug_level=0, mode=is` | Alias for `is 0` — completely silent |

The `DebugHelper` class (`src/Helpers/DebugHelper.py`) provides a thin wrapper
around these checks: `dbg.on(level)`, `dbg.only(level)`, `dbg.active(level)`, and
the static `DebugHelper.parse(raw)` parser.  Two mode-aware static helpers are the
primary consumer API — `DebugHelper.check(cfg, level)` (replaces `level(cfg) >= N`
guards, respects `debug_mode`) and `DebugHelper.check_session(session, level)`
(reads `session.debug_level` and `session.debug_mode`).  All consumer code uses
`DebugHelper` instead of calling `cfg.get_int("DEBUG_LEVEL")` directly, so the
combined string format is parsed in one place.

`debug_level` and `debug_mode` are also **live-settable** at the `🛠️` chat prompt:
`debug_level=30`, `debug_level!` (interactive named-preset picker), `debug_level?`
(show current), `debug_mode=ge` / `debug_mode=is` / `debug_mode=le`, `debug_mode!`.
The effective values appear on the `▶ Debug:` line of the `show?` status block.

- `URL_DEBUG` (`False`) enables `urllib` HTTP debug output.
- `HF_DEBUG` (`False`) enables Hugging Face debug logging.

For individual configuration switches such as `TRY_FIX_JSON_LLM_REPLY` (automatic JSON repair for LLM responses), see [JSON Repair for LLM Replies in ARCHITECTURE.md](ARCHITECTURE.md#json-repair-for-llm-replies).

### 📟 Terminal

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `TERMINAL_LINE_SIZE` | `{"debug": 180, "no_debug": 100}` in `Config_RAGChat.py` | Line width used when wrapping terminal output. Defined only in `Config_RAGChat.py` as a dict resolved at runtime by debug level (the debug branch widens tables; the no-debug branch is narrower). There is no default in `Config_Global.py`. Also settable live via `terminal_line_size=N`, `terminal_line_size!`, and `terminal_line_size?` in the chat prompt. |

### 🔤 Unicode Normalisation & Leet-Speak Detection

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_LEET_MAP` | See config file | Character mapping for leet-speak normalization (e.g. `0`→`o`, `1`→`i`, `3`→`e`). Used to detect obfuscated banned words. |
| `_CONFUSABLES` | See config file | Unicode confusable character mapping (e.g. Cyrillic `а`→`a`, `е`→`e`). Used to detect visually similar character substitutions. |
| `CSV_DELIMITER` | `;` | Delimiter used in CSV output files. |
| `LOG_FILE` | `compliance.log` | Path to the compliance log file. |
| `_CONSIDER_AS_TEXT_FILE` | `["txt", "md", "py", "c", "h", "cpp", "csv", "log"]` | File extensions considered as plain text for processing. |

### 🌐 NLP Resources

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_CUSTOM_NLTK_DATA_DIRECTORY` | Windows: `<project>\AppData\Roaming\nltk_data\corpora\stopwords`<br>Linux/macOS: `/home/vscode/nltk_data` | Custom directory for NLTK stopwords data. Override if NLTK data is installed in a non-standard location. |

### 📚 WordNet Synonym Expansion (Optional)

When enabled (`_WORDNET.ENABLED = True`), the banned-word list is expanded with English synonyms from [NLTK WordNet](https://wordnet.princeton.edu/) before translation and detection. See [WordNet Synonym Expansion in ARCHITECTURE.md](ARCHITECTURE.md#-wordnet-synonym-expansion-optional--bannedword-expansion) for full details.

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_WORDNET.ENABLED` | `True` | Master switch. Set to `False` to skip synonym expansion entirely. |
| `_WORDNET.DEPTH` | `1` | Synonym hop depth. `1` = direct synonyms only (recommended). `2` adds synonyms-of-synonyms. |
| `_WORDNET.MAX_SYNONYMS_PER_PHRASE` | `1` | Maximum synonyms added per original banned phrase. Prevents list explosion. |
| `_WORDNET.POS_FILTER` | `["n", "v"]` | Restrict to these WordNet parts of speech: `n`(oun), `v`(erb), `a`(dj), `r`(adv), `s`(at-adj). Empty list = accept all. |
| `_WORDNET.STOPLIST` | `["word", "number", "figure", ...]` | Words excluded from expansion even if they appear as WordNet synonyms. Case-insensitive. |

### 🌍 Argos Translate Definitions

Groups language-code mapping and translation pairs. Used for banned-word translation (EN→X) and language detection. See [Argos Translate in INSTALL.md](INSTALL.md#-install-argos-translate) for installation instructions.

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME` | See config file | ISO-639-1 language codes mapped to NLTK human-readable names (e.g. `"de"`→`"german"`). |
| `_ARGOS_DEFINITIONS.ARGOS_LANGUAGES` | `[("en", "de"), ("en", "es"), ("en", "fr"), ("en", "it")]` | List of (from_code, to_code) tuples for Argos Translate language pairs. Only EN→X pairs need to be installed. |

## 🛡️ 2b. Config_Banned.py — Detection & Compliance

This file defines the detection algorithms, thresholds, banned word lists, and masking rules used by RAGLoad, RAGChat, and DocClassify. After editing it, update `_BANNED_CONFIG_HASH` in `Config_Global.py`.

### 🔤 Algorithm Constants (Single Source of Truth)

These constants are used throughout the pipeline for consistent algorithm labeling:

| Constant | Value | Purpose |
| --- | --- | --- |
| `_COSINE` | `"Cosine"` | Cosine similarity algorithm (currently disabled by default) |
| `_JACCARD` | `"Jaccard"` | Jaccard character n-gram similarity |
| `_REGEX` | `"Regex"` | Regex pattern matching (includes Levenshtein fuzzy fallback) |
| `_KEYBERT` | `"Keybert"` | KeyBERT keyword extraction via embeddings |
| `_LEVENSHTEIN` | `"Levenshtein"` | Edit distance for fuzzy matching |
| `_BM25` | `"BM25"` | Okapi BM25 probabilistic scoring |

### 📋 Label Aliases

Used for CSV headers and CLI summaries to combine related algorithms:

| Original Label | Alias | Purpose |
| --- | --- | --- |
| `"Regex"` | `"Regex+Levenshtein"` | Regex and Levenshtein are reported together as a single combined label |
| `"Score Regex"` | `"Score Regex+Levenshtein"` | Score column alias |
| `"Threshold Regex"` | `"Threshold Regex+Levenshtein"` | Threshold column alias |
| `"Detail Regex"` | `"Details Regex+Levenshtein"` | Detail column alias |

### 🎯 Default Algorithms

The default algorithms run when no custom selection is provided:

```python
_DEFAULT_ALGOS = [
    _JACCARD,
    _BM25,
    _REGEX,
    _KEYBERT,
    # _COSINE,  # intentionally disabled
]
```

### 📊 CSV Columns for Human Review

The following keys/columns are included in the CSV produced for human review. Keep this list stable to avoid breaking downstream analysis scripts:

```python
_KEYS_FOR_HUMAN_REVIEW_CSV = [
    "Status", "Time", "Stage", "Skip Status", "Skipped Chunks", "Inserted Chunks",
    "Phrase", "Max Score", "Matched Algos Count", "Algos Matched",
    "Jaccard", "Score Jaccard", "Threshold Jaccard",
    "Regex+Levenshtein", "Score Regex+Levenshtein", "Threshold Regex+Levenshtein",
    "BM25", "Score BM25", "Threshold BM25",
    "Keybert", "Score Keybert", "Threshold Keybert",
    "WordCount", "Temperature", "Session", "FilePath", "FileType",
    "Language", "CreationDate", "Chunk", "FileHash",
]
```

### 🧩 Detection Configuration

The top-level container for per-app rules is `_ACTIVE_DETECTION_CONFIG`, which selects the active detection profile from `_BANNED_DETECT`.

#### Active Detection Profile Selector

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_ACTIVE_DETECTION_CONFIG` | `"STRICT_DETECT_CONFIG"` | Selector for the active detection profile. Options: `"STRICT_DETECT_CONFIG"` |

#### Per-App Detection Profiles

Each app (RAGLoad, RAGChat, DocClassify) has its own detection profile with:

- **MASKING**: Runtime masking toggles
- **PROMPT_CHECK**: Whether to run prompt-level LLM checks and params
- **PIPELINE_CHECK**: The retrieval/matching pipeline configuration

##### RAGLoad Profile

| Key | Value | Purpose |
| --- | --- | --- |
| `MASKING.APPLY_MASKING` | `True` | If `True`, redact/mask matched spans before storage |
| `PROMPT_CHECK.Check` | `False` | No LLM prompt-level check during load (content-only) |

##### RAGChat Profile

| Key | Value | Purpose |
| --- | --- | --- |
| `MASKING.APPLY_MASKING` | `True` | If `True`, redact/mask matched spans |
| `PROMPT_CHECK.Check` | `True` | Enable LLM prompt-level check |
| `PROMPT_CHECK.LLM_PARAM.temperature` | `0` | Deterministic LLM parameters for prompt checking |
| `PROMPT_CHECK.LLM_PARAM.top_k` | `1` | |
| `PROMPT_CHECK.LLM_PARAM.top_p` | `1` | |

##### DocClassify Profile

| Key | Value | Purpose |
| --- | --- | --- |
| `MASKING.APPLY_MASKING` | `True` | If `True`, redact/mask matched spans |
| `PROMPT_CHECK.Check` | `True` | Enable LLM prompt-level check |
| `PROMPT_CHECK.LLM_PARAM.temperature` | `0.1` | Slightly increased variability for classification |
| `PROMPT_CHECK.LLM_PARAM.top_k` | `20` | |
| `PROMPT_CHECK.LLM_PARAM.top_p` | `0.8` | |

### 🔍 Pipeline Configuration (Per-App)

Each app has its own pipeline configuration with algorithm-specific thresholds and consensus rules.

#### Consensus Rules

A chunk is flagged for human review when **either** condition is met:

```python
(depth_algo_count >= REQUIRED_ALGOS_ABOVE_THRESHOLD)
    OR
(any_phrase_breadth_count >= REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE)
```

| Key | RAGLoad (pipeline) | RAGChat (prompt check) | RAGChat (pipeline) | DocClassify (prompt check) | Purpose |
| --- | --- | --- | --- | --- | --- |
| `REQUIRED_ALGOS_ABOVE_THRESHOLD` | `3` | `2` | `4` | `4` | How many algos must be above their thresholds to trigger a block |
| `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE` | `4` | `3` | `4` | `4` | How many different algos must produce a non-zero score |

**Rationale for different thresholds:**

- **RAGLoad pipeline (3/4)**: Strictest — prevents loading questionable content during ingestion
- **RAGChat prompt check (2/3)**: Balanced — catches issues at prompt time while maintaining performance
- **RAGChat pipeline (4/4)**: Strictest — ensures retrieved content is highly reliable
- **DocClassify prompt check (4/4)**: Strictest — ensures classification prompts are highly reliable

#### Algorithm Thresholds

Each algorithm has individually configurable thresholds. See the config file for the full table of per-algo settings (Jaccard, BM25, Regex, KeyBERT, Cosine).

### 🎭 Masking Configuration

The masking configuration (`_ACTIVE_MASKING_CONFIG`) controls regex-based pattern masking (credit cards, SSNs, etc.). See the config file for the full `_STRICT_MASKING_REGEXES` pattern list.

### 📚 Banned Word Lists

The banned word lists (`_ACTIVE_BANNED_CONFIG`) are organized by language and application. See the config file for the full `_STRICT_BANNED` list structure.

## 🤖 2. Config_Models.py — Model Definitions

This file defines every model used by RAG-LCC. After editing it, update `_MODELS_CONFIG_HASH` in `Config_Global.py` (the new hash is printed at startup). For details on how model implementations are selected, see [Model Implementation Selectors in ARCHITECTURE.md](ARCHITECTURE.md#model-implementation-selectors) and the [Strategy Selection Pattern](ARCHITECTURE.md#strategy-selection-pattern).

`LICENSE_URL` entries should point to canonical upstream license texts (for example Apache, MIT, or upstream repository LICENSE files), consistent with the setup scripts that fetch native package licenses directly from upstream sources.

### 🔩 Implementation Selectors

`Config_Models.py` uses a two-level dictionary `_MODELS[<impl>][<role>]` to resolve model configurations. Eight top-level selector variables choose which implementation (impl key) to use for each model role:

```python
_ACTIVE_LLM_CHK            = "llama_guard"  # impl for _LLM_CHK role. llama_guard, llama, mistral
_ACTIVE_LLM                = "mistral"      # impl for _LLM role. mistral, llama
_ACTIVE_LLM_REWRITE_PROMPT = "mistral"      # impl for _LLM_REWRITE_PROMPT role. mistral default; llama also available
_ACTIVE_EMBED              = "snowflake"    # impl for _EMBED role
_ACTIVE_CROSS              = "mmarco"       # impl for _CROSS role
_ACTIVE_ENDPOINT           = "ollama"       # active inference endpoint: ollama or vllm
_ACTIVE_OPENWEBUI          = "openwebui"    # impl for _OPENWEBUI role
_ACTIVE_TRANSLATION        = "m2m100"       # impl for _TRANSLATION role. m2m100 (facebook/m2m100_1.2B)
```

| Selector | Default used in this repository | Resolves to | Allowed values |
| --- | --- | --- | --- |
| `_ACTIVE_EMBED` | `"snowflake"` | `_MODELS["snowflake"]["_EMBED"]` | `snowflake` |
| `_ACTIVE_CROSS` | `"mmarco"` | `_MODELS["mmarco"]["_CROSS"]` | `mmarco` |
| `_ACTIVE_LLM` | `"mistral"` | `_MODELS["mistral"]["_LLM"]` | `mistral`, `llama` |
| `_ACTIVE_LLM_CHK` | `"llama_guard"` | `_MODELS["llama_guard"]["_LLM_CHK"]` | `llama_guard`, `llama`, `mistral` |
| `_ACTIVE_LLM_REWRITE_PROMPT` | `"mistral"` | `_MODELS["mistral"]["_LLM_REWRITE_PROMPT"]` | `mistral`, `llama` |
| `_ACTIVE_ENDPOINT` | `"vllm"` | `_MODELS["ollama"]["_OLLAMA"]` or `_MODELS["vllm"]["_VLLM"]` | `ollama`, `vllm` |
| `_ACTIVE_OPENWEBUI` | `"openwebui"` | `_MODELS["openwebui"]["_OPENWEBUI"]` | `openwebui` |
| `_ACTIVE_TRANSLATION` | `"m2m100"` | `_MODELS["m2m100"]["_TRANSLATION"]` | `m2m100` |

To switch models, change the selector value to another key that carries a matching role entry in `_MODELS`.

### 🧲 Embedding Model (`_MODELS["snowflake"]["_EMBED"]`)

Creates vector representations for semantic search. Used during RAGLoad (once per document), DocClassify (once per document), and RAGChat (every query). The impl key is selected by `_ACTIVE_EMBED = "snowflake"`.

```python
"snowflake": {
    "_EMBED": {
        "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
        "LICENSE": "Apache-2.0",
        ...
    },
},
```

### 🔄 _LLM/_LLM_CHK Model Combinations

`Config_Models.py` includes configuration entries for the following LLMs. The `_ACTIVE_LLM` and `_ACTIVE_LLM_CHK` variables select which model implementation is used for general LLM queries and compliance checking, respectively. The table below lists all supported combinations:

| `_ACTIVE_LLM` | `_ACTIVE_LLM_CHK` | LLM Model | Compliance Model | Notes |
| --- | --- | --- | --- | --- |
| `mistral` | `llama_guard` | Mistral 7B | Llama Guard 3 | Default configuration used in this repository. General-purpose LLM + dedicated guard model. |
| `mistral` | `llama` | Mistral 7B | Llama 3.1 8B | General-purpose LLM + Llama as compliance checker. |
| `mistral` | `mistral` | Mistral 7B | Mistral 7B | Same model for both roles. |
| `llama` | `llama_guard` | Llama 3.1 8B | Llama Guard 3 | Llama for generation + dedicated guard model. |
| `llama` | `llama` | Llama 3.1 8B | Llama 3.1 8B | Same model for both roles. |
| `llama` | `mistral` | Llama 3.1 8B | Mistral 7B | Llama for generation + Mistral as compliance checker. |

### ✏️ _LLM_REWRITE_PROMPT Model Options

The `_ACTIVE_LLM_REWRITE_PROMPT` selector chooses which model rewrites follow-up queries for coreference resolution (e.g. resolving "they" or "it" using conversation history). The rewrite model is independent of the generation and compliance models.

| `_ACTIVE_LLM_REWRITE_PROMPT` | Rewrite Model | Notes |
| --- | --- | --- |
| `mistral` | Mistral 7B | Default configuration used in this repository. Apache-2.0. |
| `llama` | Llama 3.1 8B | Available alternative. Follows multi-rule instructions reliably. |

The rewrite LLM can be freely combined with any `_ACTIVE_LLM` / `_ACTIVE_LLM_CHK` combination from the table above.

> **Note:** `llama_guard` is only valid for `_ACTIVE_LLM_CHK` — it is a dedicated safety model and cannot serve the `_ACTIVE_LLM` (generation) role. For details on the model selector mechanism, see [Model Implementation Selectors in ARCHITECTURE.md](ARCHITECTURE.md#model-implementation-selectors).
>
> **Attribution:** Llama 3.1 and Llama Guard 3 — Built with Meta Llama 3. Licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved. By downloading and using these models, operators are bound by the model license terms.
>
> **Operator responsibility:** Each model has its own license. The operator is responsible for reviewing, accepting, and complying with the license terms of every model used. RAG-LCC does not warrant that any model is suitable for a particular purpose. See [License Consent](INSTALL.md#-license-consent) and [Model permission requirement](INSTALL.md#-model-permission-requirement).

---

### 🔀 Cross-Encoder Model (`_MODELS["mmarco"]["_CROSS"]`)

Re-ranks search results retrieved from ChromaDB to improve relevance ordering. The impl key is selected by `_ACTIVE_CROSS = "mmarco"`. Used by `RAGChat.py`.

```python
"mmarco": {
    "_CROSS": {
        "MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "QUERY_INSTRUCTION": "",   # optional prefix prepended to the query
        "LICENSE": "Apache-2.0",
        ...
    },
},
```

| Key | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | HuggingFace model ID for the cross-encoder. |
| `QUERY_INSTRUCTION` | `""` | Optional string prepended to the **query** (not the chunk) before scoring. Useful for instruction-tuned rerankers that expect a task prefix (e.g. `"Represent this sentence for searching relevant passages: "`). Leave empty for standard cross-encoders like mmarco. |

### 🧠 Inference LLM (`_MODELS["mistral"]["_LLM"]`)

Generates responses (RAGChat) or classification labels (DocClassify). Runs locally via Ollama or vLLM (depending on `_ACTIVE_ENDPOINT`). The impl key is selected by `_ACTIVE_LLM = "mistral"` (default used in this repository) or `_ACTIVE_LLM = "llama"`.

```python
"mistral": {
    "_LLM": {
        "MODEL_OLLAMA": "mistral:7b",
        "MODEL_VLLM": "mistral_7b",
        "PROMPT_CHAT": "_PROMPT_CHAT",
        "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_MISTRAL",
        ...
    },
},
```

To switch models, change the `_ACTIVE_LLM` selector variable in `Config_Models.py` (e.g. `_ACTIVE_LLM = "llama"` resolves to `_MODELS["llama"]["_LLM"]` for Llama 3.1 8B). See the [_LLM/_LLM_CHK Model Combinations](#-_llm_llm_chk-model-combinations) table above for all supported values.

### ✏️ Query Rewrite LLM (`_MODELS["llama"]["_LLM_REWRITE_PROMPT"]`)

A dedicated LLM used to rewrite follow-up queries for coreference resolution in multi-turn chat. When chat context is enabled, ambiguous references (e.g. "are they mammals?") are rewritten into self-contained questions (e.g. "which of cats, hedgehogs, and dogs are mammals") before retrieval. Each conversation turn is tagged with the active file filter (or `[No file filter]`), so the rewriter can detect context switches — when the user changes to a different file, prior entities are not carried over. The impl key is selected by `_ACTIVE_LLM_REWRITE_PROMPT = "mistral"` (default used in this repository) or `"llama"`. Rewrite parameters (`temperature`, `top_k`, `top_p`, `num_predict`, `streaming`, `topic_confidence_threshold`, `TOPIC_SUMMARY_MODE`, `TRANSLATION_BACKEND`) are configured separately in `_QUERY_REWRITE` in `Config_RAGChat.py`.

```python
"mistral": {
    "_LLM_REWRITE_PROMPT": {
        "MODEL_OLLAMA": "mistral:7b",
        "MODEL_VLLM": "mistral_7b",
        "PROMPT_REWRITE": "_PROMPT_REWRITE",
        ...
    },
},
```

### 🛡️ Compliance-Check LLM (`_MODELS["llama_guard"]["_LLM_CHK"]`)

A separate LLM used to validate prompts and outputs against compliance rules. Defaults to Llama Guard 3. The impl key is selected by `_ACTIVE_LLM_CHK = "llama_guard"` (default used in this repository), `_ACTIVE_LLM_CHK = "llama"`, or `_ACTIVE_LLM_CHK = "mistral"`.

```python
"llama_guard": {
    "_LLM_CHK": {
        "MODEL_OLLAMA": "llama-guard3:8b",
        "MODEL_VLLM": "llama_guard3_8b",
        "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_LLAMA_GUARD",
        "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_LLAMA_GUARD",
        ...
    },
},
```

### ℹ️ Inference Endpoint Provider Metadata

Records runtime provider details. Not a model itself. Active provider is selected by `_ACTIVE_ENDPOINT`.

**⚠️ Specifying the `<host>` for BASE_URL:**

When configuring `BASE_URL` for Ollama, vLLM Open WebUI replace `<host>` with the actual hostname or IP address where the backend is running.

RAG-LCC automatically tries up to **6 fallback candidates** when the configured endpoint is unavailable (configured host, localhost, 127.0.0.1, host.docker.internal, plus localhost and host.docker.internal with the default port if it differs from the configured port). See [ARCHITECTURE.md § Endpoint Fallback Mechanism](ARCHITECTURE.md#endpoint-fallback-mechanism) for the complete probe sequence and behavior.

Common scenarios:

| Scenario | Use as `<host>` |
| --- | --- |
| **Same machine as RAG-LCC** | `localhost` or `127.0.0.1` |
| **RAG-LCC in Docker, backend on host (Docker Desktop for Windows/Mac)** | `host.docker.internal` |
| **RAG-LCC in Docker, backend on host (Linux)** | `172.17.0.1` (bridge gateway) or host's network IP |
| **Backend on another server** | Server's IP address (e.g., `192.168.1.100`) |

**Container → Host connectivity:** TCP access from Docker container to host is **not blocked by default**—it works natively. The issue is hostname resolution: `localhost` inside a container refers to the container itself, not the host. Use `host.docker.internal` (Docker Desktop) or `172.17.0.1` (Linux bridge gateway) to reach the host.

**Test connectivity before configuring:**

```bash
# Ollama (Docker Desktop):
curl -v http://host.docker.internal:11434/api/tags

# Ollama (Linux):
curl -v http://172.17.0.1:11434/api/tags

# vLLM (Docker Desktop):
curl -v http://host.docker.internal:4000/v1/models

# vLLM (Linux):
curl -v http://172.17.0.1:4000/v1/models
```

If curl returns JSON, connectivity is working—use that hostname/IP as `<host>` in your BASE_URL.

#### Ollama (`_MODELS["ollama"]["_OLLAMA"]`)

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `BASE_URL` | `http://<host>:11434/api/generate` | Ollama endpoint. Replace `<host>` with actual hostname/IP (see above). |
| `STREAMING_REQ` | `False` | Enable streamed responses from Ollama. |
| `TRY_FALLBACK_URLS` | `True` | `True` — try up to 6 fallback candidates on failure. `False` — probe `BASE_URL` once and exit immediately on failure (use for fixed remote IPs). |
| `USE_GPU` | `True` | Let Ollama use the GPU for inference. |

#### vLLM (`_MODELS["vllm"]["_VLLM"]`)

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `BASE_URL` | `http://<host>:4000/v1/chat/completions` | OpenAI-compatible vLLM endpoint. Replace `<host>` with actual hostname/IP (see above). |
| `STREAMING_REQ` | `False` | Enable streamed responses from vLLM. |
| `TRY_FALLBACK_URLS` | `True` | `True` — try up to 6 fallback candidates on failure. `False` — probe `BASE_URL` once and exit immediately on failure (use for fixed remote IPs). |
| `USE_GPU` | `True` | Informational runtime flag for provider metadata. |

Set `_ACTIVE_ENDPOINT = "vllm"` to switch to vLLM and make sure the selected models (`MODEL_VLLM`) are available on that endpoint.

If you want to run models locally behind a [LiteLLM](https://github.com/BerriAI/litellm) proxy (which exposes an OpenAI-compatible endpoint on port 4000), see [HarinezumIgel/harinezumigel-llm-stack](https://github.com/HarinezumIgel/harinezumigel-llm-stack) — a unified LiteLLM + vLLM launcher for managing LLM deployments in Docker with configuration-driven orchestration.

> **Important:** RAG-LCC does **not** download Ollama LLM models. You must install Ollama and pull models yourself, eg. `ollama pull mistral:7b`.

### 🌐 OpenWebUI Provider Metadata (`_MODELS["openwebui"]["_OPENWEBUI"]`)

Records the OpenWebUI provider details (license, BASE_URL). Used by `RAGChatService`. The impl key is selected by `_ACTIVE_OPENWEBUI = "openwebui"`.

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `BASE_URL` | `"http://<openwebui host>:8080"` | URL where OpenWebUI is running. Used by Informer to verify OpenWebUI is reachable at RAGChatService startup. Must be accessible from RAGChatService (use `host.docker.internal` or `172.17.0.1` when RAGChatService runs in Docker and OpenWebUI is on the host). |

#### 📡 `ragchatservice` — RAGChatService HTTP Listener Configuration

Records the RAGChatService listener configuration (host, port, API key). The impl key is selected by `_ACTIVE_RAGCHATSERVICE = "ragchatservice"`.

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `HOST` | `"127.0.0.1"` | Bind address for the HTTP listener. `0.0.0.0` binds all interfaces (required for Docker port forwarding). Change to `127.0.0.1` to restrict to loopback only. |
| `PORT` | `11435` | Port for the HTTP listener. |
| `BASE_URL` | `"http://localhost:11435"` | Full service URL (derived from HOST and PORT). |
| `API_KEY` | `""` | Bearer token for authenticating incoming requests from OpenWebUI. Must match the API key configured in OpenWebUI. |

> **Note:** RAGChatService configuration is centralized in `Config_Models.py` under the `ragchatservice` section. Both HOST and PORT are configurable during `Setup.py` to accommodate different deployment scenarios and port conflicts. See [Config_RAGChatService.py — HTTP Service Configuration](#-4-config_ragchatservicepy--http-service-configuration) for additional service settings.
>
> **URL configuration:** These settings are independent:
>
> - **RAGChatService → OpenWebUI connectivity:** `_MODELS.openwebui._OPENWEBUI.BASE_URL` (Informer checks if OpenWebUI is reachable)
> - **OpenWebUI → RAGChatService connectivity:** `_MODELS.ragchatservice._RAGCHATSERVICE.HOST` and `_MODELS.ragchatservice._RAGCHATSERVICE.PORT` (where RAGChatService binds its listener)
> - **Browser → `/marked` endpoint:** Browsers fetch from RAGChatService's `/marked` endpoint using the same host:port as the RAGChatService API

## 🌐 3b. Config_WebSearch.py — Web Search Configuration

This file contains all web-search-specific settings shared across RAGChat and RAGChatService. After editing it, update `_WEB_SEARCH_CONFIG_HASH` in `Config_Global.py`.

### 🔑 Web-Search Switch Reference

| Switch | Where | Effect |
| --- | --- | --- |
| `WEB_SEARCH_MODE` (environment variable) | `Config_Internet_Env.py` | **MASTER SWITCH.** Overrides all other web-search settings. Values: `"0"` — web search disabled; `"1"` — internet search enabled. |
| `_OPENWEB_UI_WEBSEARCH` | `Config_WebSearch.py` | Default web-search state for new OpenWebUI sessions. Has **no effect** when `WEB_SEARCH_MODE` is not `"1"`. |
| `web_search` (session) | per-request | Caller opt-in. Ignored when `WEB_SEARCH_MODE` is `"0"`. |

### 📊 Web Search Backend Configuration (`_WEB_SEARCH`)

Controls the optional internet retrieval leg added to the RRF pipeline at query time when `WEB_SEARCH_MODE = "1"`.

| Key | Default | Purpose |
| --- | --- | --- |
| `backend` | `"duckduckgo"` | Web search backend. Only `"duckduckgo"` is implemented. `"brave"`, `"tavily"`, `"bing"` are recognised but raise `NotImplementedError`. |
| `api_key` | `""` | Reserved for future brave/tavily/bing backends. Leave empty for duckduckgo. |
| `max_results` | `10` | Maximum web results fetched per query (used when `fetch_k` not set). |
| `max_query_length` | `500` | Queries longer than this are truncated before sending. |
| `block_on_injection` | `True` | Block queries that match prompt-injection / attack patterns. |
| `default_web_weight` | `0.5` | Default RRF weight for web results relative to local retrievers (Vector/BM25/Graph = 1.0). `0.5` means every local result naturally outranks any web result; raise to `1.0+` for equal or higher web influence. Overridable per-session with `web_weight=<value>`. |
| `bm25_pre_filter` | `0.10` | Minimum BM25 score (against the query) a web result must reach to survive before entering the rerank pool. `0.0` = disabled (all results pass). Typical useful range: `0.05–0.30`. Only active when `retrieve_mode` includes web results (`web_search=True` or `retrieve_mode=WEB`). |
| `cosine_pre_filter` | `0.30` | Minimum cosine similarity (query embedding vs. snippet embedding) a web result must reach to survive. `0.0` = disabled. Runs after `bm25_pre_filter` when both are set. Typical useful range: `0.20–0.50`. Requires the embedding model to be loaded (always true in RAGChat). |
| `rerank_threshold` | `0.0` | After reranking, web chunks whose cross-encoder score falls below this value are dropped. Defaults to `0.0` (no additional filtering) because `bm25_pre_filter` and `cosine_pre_filter` already gate quality; cross-encoder scores on short web snippets are structurally lower than on local full-text chunks and must not be compared against the local `chroma_threshold`. Raise if you want stricter post-rerank filtering for web results (e.g. `0.05`). |

### 🎯 Intent-Filter Extensions (`WEB_SEARCH_INTENT_EXTENSIONS`)

Additive-only extensions to the baseline WebSearchFilter entity catalogue and thresholds.

| Key | Type | Purpose |
| --- | --- | --- |
| `entity_extensions` | `dict[str, list[str]]` | Add entity terms to existing baseline categories (e.g. `"illicit_substances": ["new_scheduled_compound"]`) |
| `entity_categories_extra` | `dict[str, dict]` | Add entirely new entity categories (may NOT shadow baseline category names). Each category has `weight` (int) and `entities` (list[str]). |
| `threshold_overrides` | `dict[str, int]` | May only LOWER (tighten) the baseline refuse/warn thresholds; higher values are silently ignored. |

**Example:**

```python
WEB_SEARCH_INTENT_EXTENSIONS = {
    "entity_extensions": {
        # "illicit_substances": ["new_scheduled_compound"],
    },
    "entity_categories_extra": {
        # "demo_watchlist": {"weight": 70, "entities": ["zeta-9"]},
    },
    "threshold_overrides": {
        # "refuse": 45,  # stricter than baseline 60
        # "warn":   20,  # stricter than baseline 30
    },
}
```

**Notes:**

- Uncomment and populate entries as needed for your deployment.
- Remove before production deployment (demo categories are fictional).
- See `src/Compliance/WebSearchFilter.py` for baseline entity categories and thresholds.

## 💬 3. Config_RAGChat.py — Retrieval and Response

### 🎯 Chunk Selection Strategy

```python
_ACTIVE_CHUNK_SELECT_STRATEGY = "DEFAULT"   # One of: "NARROW", "BALANCED_FILE_CAP", "WIDE", "ULTRA_WIDE", "DEFAULT"
```

Each strategy is a complete parameter set defined in `_STRATEGIES` (see [Strategy Selection Pattern in ARCHITECTURE.md](ARCHITECTURE.md#strategy-selection-pattern)). The key differences:

| Strategy | `final_chunks_to_llm` | `retriever_k` | `threshold` | `max_output_tokens` | `filelim` | Use case |
| --- | --- | --- | --- | --- | --- | --- |
| DEFAULT | 50 | 100 | 0.35 | 14 366 | 15 | General-purpose, score-ranked (default) |
| NARROW | 20 | 80 | 0.75 | 8 192 | 5 | Precise, answer-focused |
| BALANCED_FILE_CAP | 40 | 60 | 0.55 | 14 366 | 10 | Balanced with per-file chunk cap |
| WIDE | 60 | 160 | 0.35 | 14 366 | 20 | Exploratory, high recall |
| ULTRA_WIDE | 1 500 | 3 000 | 0.20 | 14 366 | 0 | Exhaustive / debugging |

> `DEFAULT` uses `ScoreRankedSelector` — chunks are selected strictly by descending reranker score.
> `BALANCED_FILE_CAP` uses `PerFileCapSelector` — caps each source file to at most `filelim` chunks, preventing any single file from dominating the context window.
> `NARROW` uses `SingleDocumentSelector` — a tighter threshold and smaller context window focus retrieval on the single best-matching document.  Graph retrieval is disabled (`graph_weight=0`) because the selector discards all cross-file results anyway.
> See [Retrieval Chunk Selection in ARCHITECTURE.md](ARCHITECTURE.md#-retrieval-chunk-selection) for the full selection algorithm and flow diagram.

| Key | Default | Purpose |
| --- | --- | --- |
| `SINGLE_CHUNK_SCORE_BOOST` | `1.25` | Multiplier applied to the reranker score of any chunk whose source file is the only file contributing to the result set. Boosts single-document answers to avoid them being unfairly penalised relative to multi-document pools where the same score distributes across more chunks. |

Strategy parameters explained:

| Parameter | Purpose |
| --- | --- |
| `final_chunks_to_llm` | Maximum number of chunks passed to the LLM after reranking and threshold filtering. |
| `retriever_k` | Number of candidate chunks each retriever fetches before filtering and reranking. |
| `threshold` | Minimum reranker score to keep a chunk. Scores are min-max normalized within the candidate pool (worst candidate → 0.0, best → 1.0), so `0.75` means "top 25 % of this query's pool". Web doc scores use a separate sigmoid scale and are not affected by this normalization. **Relative-band fallback:** when the cross-encoder gives all-negative raw logits (common with mmarco-style models on technical content), the pool's absolute best score falls below the threshold and would cause every chunk to be rejected. In that case the selector automatically falls back to a relative band — keeping chunks whose score is ≥ 75 % of the pool's best local score — so the top-ranked chunk is always surfaced. A diagnostic message is logged at debug level ≥ 10 when the fallback fires. |
| `max_output_tokens` | Output token ceiling for this strategy (may be further reduced by the token budget). |
| `temperature` | LLM sampling temperature. Lower = more deterministic. |
| `top_k` | Limit sampling to the top-k most likely next tokens. |
| `top_p` | Nucleus sampling probability threshold. |
| `rerank` | `1` = enable cross-encoder reranking (used by default). |
| `retrieve_mode` | Retrieval mode: `VECTOR`, `BM25`, `GRAPH`, `VECTOR_BM25`, `VECTOR_GRAPH`, `BM25_GRAPH`, or `ALL`. See [Retrieval Stores & Search Modes](#-retrieval-stores--search-modes) for details. |
| `filelim` | Max chunks per contributing file. `0` = unlimited. Exposed as `per_file_limit` in the API / OpenWebUI Controls. |
| `vector_weight` | RRF weight for the vector (Chroma) retriever. `0` disables vector retrieval entirely. |
| `bm25_weight` | RRF weight for the BM25 retriever. `0` disables BM25 retrieval entirely. |
| `graph_weight` | RRF weight for the entity-graph retriever. `0` disables graph retrieval entirely. NARROW defaults to `0` because `SingleDocumentSelector` discards cross-file graph results anyway. |
| ⚠️ `web_weight` | Default RRF weight applied to web search results when this strategy is loaded (e.g. `0.5`). Overridable per-session. Only takes effect when `web_search` is enabled. |
| `collection` | ChromaDB collection. `"$COLLECTION"` resolves the global `COLLECTION` key. |
| `use_chat_context` | Include previous conversation turns in the prompt. |
| `turns` | Pruning threshold — maximum context entries kept per chat. Exceeding this triggers pruning. |
| `prune_batch` | Pruning granularity — number of oldest entries summarized into one entry per pruning pass. |
| `max_history_turns` | Max recent turns sent to the query rewriter. `0` = unlimited. |
| `TOPIC_SUMMARY_MODE` | Which assistant turns are used to build the rolling topic summary fed to the query rewriter. `"last"` = only the most recent assistant turn (default, lower cost). `"all"` = all assistant turns in the history window (richer context). |

> See [Chat Context](ARCHITECTURE.md#-chat-context) in the architecture guide
> for details on how multi-turn memory, retrieval, and incremental pruning work.

### 🔀 Multi-Query Expansion (`_MULTI_QUERY` in `Config_RAGChat.py`)

Controls the optional alternate-query recall-broadening pass that runs during retrieval.

```python
_MULTI_QUERY: dict[str, Any] = {
    "enabled": True,
    "num_variants": 3,
    "LLM_PARAM": {
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.95,
        "num_predict": 256,
        "use_ollama_gpu": True,
        "streaming": False,
    },
}
```

| Key | Default | Purpose |
| --- | --- | --- |
| `enabled` | `True` | Enable multi-query expansion. When `False`, retrieval runs only on the (optionally rewritten) user query. |
| `num_variants` | `3` | Number of alternate phrasings to generate. Each variant triggers one additional VECTOR search. BM25 and Graph legs are not affected. |
| `LLM_PARAM.temperature` | `0.5` | Sampling temperature for the expansion LLM call. Higher values produce more varied phrasings. |
| `LLM_PARAM.top_k` | `40` | Top-k sampling limit for the expansion call. |
| `LLM_PARAM.top_p` | `0.95` | Nucleus sampling threshold for the expansion call. |
| `LLM_PARAM.num_predict` | `256` | Max output tokens for the expansion call (the response is a compact JSON array). |
| `LLM_PARAM.use_ollama_gpu` | `True` | Route the expansion LLM call through the GPU. |
| `LLM_PARAM.streaming` | `False` | Must be `False` — the JSON response must be parsed in full. |

The prompt template for the expansion call is `PROMPT_QUERY_EXPAND`, configured in `Config_Models.py` under both `mistral._LLM_REWRITE_PROMPT` and `llama._LLM_REWRITE_PROMPT`:

```python
"PROMPT_QUERY_EXPAND": "_PROMPT_QUERY_EXPAND",
```

The actual template string is `_PROMPT_QUERY_EXPAND` in `Config_RAGChat.py`.  It instructs the LLM to return a JSON array of `num_variants` strings — anything else is silently ignored and retrieval falls back to the original query only.

Alternate-query hits are folded into the main RRF candidate pool before the merge step.  Duplicate chunks (same `doc_id`) are discarded; only the first occurrence (highest-ranked from any query variant) is kept.  The generated variants are logged at debug level 29.

See [Multi-query expansion in ARCHITECTURE.md](ARCHITECTURE.md#selection-flow-all-strategies) for the full pipeline description.

### 🧹 Chunk Near-Duplicate Removal (`_CHUNK_DEDUP` in `Config_RAGChat.py`)

Controls the optional post-RRF deduplication pass that collapses near-identical chunks before cross-encoder reranking.

```python
_CHUNK_DEDUP: dict[str, Any] = {"enabled": True, "threshold": 0.85}
```

| Key | Default | Purpose |
| --- | --- | --- |
| `enabled` | `True` | Enable chunk deduplication. When `False`, the full candidate pool is passed to the cross-encoder unchanged. |
| `threshold` | `0.85` | Jaccard similarity threshold (token-level, on lowercased word tokens). Chunks with similarity ≥ `threshold` to an already-kept chunk are discarded. `1.0` removes only exact duplicates; `0.7` removes near-paraphrases. |

Deduplication runs **after** RRF fusion and **before** cross-encoder reranking.  The highest-ranked chunk (lowest RRF rank, i.e. best retrieval position) is always retained; later candidates that are near-duplicates of it are dropped.  This reduces the number of chunks the cross-encoder must score and prevents the LLM from receiving the same passage multiple times.

When at least one chunk is removed, a log entry is written at level `"I"` with the `ChunkDedup` label showing the count removed, the threshold, and the number of chunks kept.

See [Near-duplicate chunk removal in ARCHITECTURE.md](ARCHITECTURE.md#selection-flow-all-strategies) for the full pipeline description.

### ⚠️ Web Search — Admin Knobs

| Key | Location | Default | Purpose |
| --- | --- | --- | --- |
| `WEB_SEARCH_MODE` (environment variable) | `Config_Internet_Env.py` | `"0"` | **Master switch.** Controls whether web search is active. Overrides all other web-search settings, including `_OPENWEB_UI_WEBSEARCH`. Values: `"0"` — web search disabled (safe default); `"1"` — internet search enabled (user queries may be sent to DuckDuckGo). Operators enabling `"1"` must review `LEGAL.md § Web Search` and `SECURITY.md`. |
| `_QUERY_LOG` | `Config_RAGChat.py` / `Config_RAGChatService.py` | `logs/RAGChat/queries.log` | Append-only audit log written for every web search attempt, including blocked ones. Each app writes to its own log subdirectory. Set to `""` to disable logging. |
| `_INTENT_FILTER_LOG` | `Config_RAGChat.py` / `Config_RAGChatService.py` | `logs/RAGChat/intent_filter.log` | Append-only log for `WebSearchFilter` intent-classifier decisions (ALLOW, BLOCK, ESCALATE). Written for both web and local queries. Each app writes to its own log subdirectory. Set to `""` to disable. |

The per-session knobs (`web_search`, `web_weight`, `fetch_page_content`) and the `_WEB_SEARCH` backend dict (`Config_WebSearch.py`) are documented in [Internet Retrieval (Optional) in README.md](README.md#-internet-retrieval-optional).

### ⚠️ OpenWebUI Service Knob (`Config_WebSearch.py`)

| Key | Default | Purpose |
| --- | --- | --- |
| `_OPENWEB_UI_WEBSEARCH` | `False` | When `True` (and `WEB_SEARCH_MODE = "1"`), every OpenWebUI request that does not carry an explicit `web_search` parameter automatically gets web search enabled — users never need to add an OpenWebUI Advanced Parameter manually. Has **no effect** when `WEB_SEARCH_MODE` is `"0"`. A startup warning is printed at **RAGChatService** startup when this is `True` but the master switch is not `"1"`. |

### �️ Visual Markers (`mark_text`)

When `mark_text` is enabled in a chat session, RAG-LCC produces in-memory highlighted copies of every local source document that contributed chunks to the answer, and offers them to the user at the CLI prompt or (in service mode) as clickable links in the OpenWebUI reply.

#### Per-session toggle

```text
🛠️  > mark_text=true     # enable for this session
🛠️  > mark_text=false    # disable
🛠️  > mark_text?         # show current value
```

#### Highlight colours

Colours are configured in one grouped slot in `Config_RAGChat.py`:

| Key | Default | Purpose |
| --- | --- | --- |
| `_MARKED_DOCS_COLORS.highlight` | `"yellow"` | Highlight inside source PDF / DOCX / PPTX (relevant source chunks). CSS-style colour name or hex string. |
| `_MARKED_DOCS_COLORS.answer_mark` | `""` | Reserved for future use. Currently unused — API grounding is shown only in marked documents, not in chat text. |
| `_MARKED_DOCS_COLORS.answer_ansi` | `"48;5;214"` | Grounded/effective answer sentences in CLI terminal (ANSI SGR parameter string without `\033[` and `m`). Pass `""` to disable CLI answer grounding. |

```python
_MARKED_DOCS_COLORS = {
    "highlight": "yellow",      # source-chunk highlight in marked docs
    "answer_mark": "",          # grounded sentences (HTML/MD) — currently unused
    "answer_ansi": "48;5;214",  # grounded sentences (CLI terminal)
}
```

#### Answer grounding sensitivity

Grounding behaviour is configured in one grouped slot:

| Key | Default | Purpose |
| --- | --- | --- |
| `_MARKED_DOCS_GROUNDING.min_sentence_tokens` | `5` | Minimum token count a sentence must have to be considered for grounding. Lower values increase matches. |
| `_MARKED_DOCS_GROUNDING.min_fragment_len` | `12` | Minimum fragment length extracted from chunks for overlap matching. Lower values increase matches. |
| `_MARKED_DOCS_GROUNDING.min_overlap_window` | `5` | Contiguous token window size for paraphrase grounding. Lower values increase matches for paraphrased text. |

Grounding is active whenever `mark_text` is enabled for a session. There is no separate `enabled` switch — set `answer_ansi: ""` to disable CLI terminal highlights, and `mark_text=false` to disable document marking entirely.

```python
_MARKED_DOCS_GROUNDING = {
    "min_sentence_tokens": 5,
    "min_fragment_len": 12,
    "min_overlap_window": 5,
}
```

#### Streaming downgrade

When `mark_text` is enabled and a client requests `stream=True`, the service automatically downgrades to a buffered (non-streaming) reply because grounding requires the complete answer text before sentence highlighting can be applied. The log emits:

``` text
🔵 Call LLM streaming:  False — document grounding active (mark_text=True, streaming downgraded)
```

Streaming is restored automatically as soon as `mark_text` is disabled.

#### Supported document types

| Extension | Highlighter | Mechanism |
| --- | --- | --- |
| `.pdf` | `PdfVisualMarker` | `/Highlight` annotations via pdfplumber + pypdf |
| `.docx` | `DocxVisualMarker` | `<w:highlight>` XML injected into matching paragraph runs |
| `.pptx` / `.ppt` | `PptxVisualMarker` | `<a:highlight>` XML injected via python-pptx + lxml |
| `.md` / `.txt` | `PlainTextVisualMarker` | `<mark>…</mark>` HTML wrapper on matching lines |

#### CLI delivery

In CLI mode the highlighted bytes stay in memory. When the user selects a document from the numbered picker, **only that file** is written to a short-lived temp directory; other documents remain in RAM. If the user skips the prompt entirely, no file is written at all. Temp files are deleted when the process exits.

Terminals that support OSC 8 hyperlinks (Windows Terminal, VS Code, iTerm2, Kitty, WezTerm, …) display clickable links instead of a picker; all files are written up-front in that case so the links are ready to click.

#### Service delivery (`RAGChatService`)

In service mode, highlighted bytes are stored in the `MarkedDocsStore` in-memory cache and served as short-lived URLs (`GET /marked/<token>.<ext>`). The link block is appended to the LLM answer. See the `_MARKED_DOCS` block in `Config_RAGChatService.py`.

### 🎨 Answer Display Style (`ANSWER_DISPLAY`)

The CLI answer block can be styled with a background and foreground colour. Configure in `Config_RAGChat.py`:

```python
ANSWER_DISPLAY = {
    "bg": "ANSWER_BG",   # name of a constant in Gui/Colors.py
    "fg": "ANSWER_FG",   # name of a constant in Gui/Colors.py
}
```

`ANSWER_BG` and `ANSWER_FG` are resolved from `Gui/Colors.py` at import time. The module auto-detects truecolor support:

- **Truecolor** (`COLORTERM=truecolor` / `COLORTERM=24bit` / `WT_SESSION` set): 24-bit RGB sequences — dark gray background, near-white text.
- **Fallback**: 256-color equivalents (`48;5;238` / `38;5;252`) used on terminals without truecolor support (plain PowerShell, cmd, older Linux terminals).

Set either key to `""` to disable that colour component.

### �💾 Chat and Settings History

Chat and settings history are stored in the `history/` directory. The fallback chat identifier is `_DEFAULT_CHAT_NAME = "MyFirstChat"`. Two history files are created:

- `<collection>_<chat name>_Query.txt`
- `<collection>_<chat name>_Settings.txt`

## 📥 5. Config_RAGLoad.py — Document Ingestion

This is the simplest app-specific config:

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_FRIENDLY_NAME` | `"RAGLoad"` | Internal identifier (do not change). |
| `_SEPARATORS` | `["\n", " ", "."]` | Text splitter separators in priority order. |
| `_CLASSIFICATION_KEYS` | `["Status", "Time", "Stage", ...]` | Columns written to the compliance CSV during ingestion. |
| `_KEY_BERT.TOP_N_FIRST` | `100` | Keywords from the first KeyBERT pass. |
| `_KEY_BERT.TOP_N_SECOND` | `60` | Keywords from the second KeyBERT pass. |

### 🔍 Classify‑then‑Load

When a classify CSV path is provided, `RAGLoad` reads the classification CSV produced by a prior `DocClassify` run and limits ingestion to the file paths listed therein. All other files in `DOC_DIR` are skipped.

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `LOAD_FROM_CLASSIFY_CSV` | `""` | Path to a `DocClassify` CSV. Accepts a filename (resolved relative to the `logs/` directory) or an absolute path. When non-empty, only documents listed in the CSV are ingested. |
| `CLASSIFY_CSV_QUERY` | `""` | Optional SQL WHERE clause applied to the loaded CSV rows. The CSV is loaded into an in-memory SQLite table; only rows satisfying the expression are included. Supports `LIKE`, `AND`, `OR`, `NOT LIKE`, `=`, `!=`, `IN`, etc. Example: `"Mammal LIKE '%Yes%' AND Language = 'English'"`. |

CLI example:

```bash
python src/Apps/RAGLoad.py --load-from-classify-csv DocClassify_OK_20260317_111105.csv
```

With a query filter:

```bash
python src/Apps/RAGLoad.py --load-from-classify-csv DocClassify_OK_20260317_111105.csv --classify-csv-query "Mammal LIKE '%Yes%'"
```

When the classify‑then‑load filter is active, exclusion checks (`USE_EXCLUSIONS`) are bypassed because `DocClassify` already evaluated exclusions during its run.

Classification results are heuristic and probabilistic — false positives and false negatives will occur. The classify‑then‑load filter does not add, verify, or guarantee any legal, regulatory, or compliance status of the ingested documents.

### 🔄 Incremental Hash Check

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_PROCESS_IF_UNCHANGED` | `True` | Re-process files even when their hash has not changed. Set to `False` to skip unchanged files. File hash is stored in Chroma DB. |

## 🏷️ 6. Config_DocClassify.py — Document Classification

### 🧩 Extraction Model Parameters (LLM)

Controlled by `_ACTIVE_EXTRACTION_CONFIG` (default used in this repository `"STRICT"`). Each variant is a nested dict inside `_EXTRACTION_MODEL_PARAMS`:

```python
_ACTIVE_EXTRACTION_CONFIG = "STRICT"   # Options: "STRICT", "BALANCED", "RECALL"
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

```

### 🔑 KeyBERT Keyword Extraction

Controlled by `_ACTIVE_KEYBERT_CONFIG` (default used in this repository `"STRICT"`). Each variant is a nested dict inside `_KEY_BERT`:

```python
_ACTIVE_KEYBERT_CONFIG = "STRICT"      # Options: "STRICT", "BALANCED", "RECALL"

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
```

The two selectors are independent — you can mix variants (e.g. `STRICT` extraction with `BALANCED` KeyBERT). `Config_RAGChat.py` and `Config_RAGLoad.py` use a **flat** `_KEY_BERT` dict (no variant nesting); `Helpers._get_keybert_config()` detects the layout automatically.

### 🏷️ Classification Keys

`_YOUR_CLASSIFICATION_KEYS` defines which fields the LLM must return in its JSON response. The default set used in this repository is:

```python
_YOUR_CLASSIFICATION_KEYS = [
    "Classification",   # Category labels
    "Purpose",          # Brief summary
    "Topic",            # Short topic phrase
    "Animal",           # What animals are discussed
    "Mammal",           # Is the animal a mammal
    "Language",         # Detected language
]
```

> **Important:** If you change these keys, you must also update the classification prompt templates (`_PROMPT_CLASSIFY_MISTRAL`, `_PROMPT_CLASSIFY_LLAMA`) so the LLM is instructed to return the matching key names in its JSON response.

`_CLASSIFICATION_KEYS` is the full column list written to CSV and includes both system columns (`Status`, `Time`, `FilePath`, etc.) and the user-defined keys above.

`CLASSIFICATION_WORD_CNT` is automatically derived from `len(_YOUR_CLASSIFICATION_KEYS)`. `SUMMARY_SENTENCE_CNT` default used in this repositoy is `10`.

`REVERSE_STEMMING = True` post-processes classification values to replace stems with their most matching original surface word.

### � Character Replacement Mapping

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `_UNWANTED_CHAR_MAP` | `{"ß": "ss"}` | Map of characters to replace in extracted text to normalize tokens. Used before KeyBERT extraction and classification. |

### �📝 Prompt Templates

Two prompt templates are included — `_PROMPT_CLASSIFY_MISTRAL` and `_PROMPT_CLASSIFY_LLAMA`. Both instruct the LLM to return a single JSON object with the keys listed in `_YOUR_CLASSIFICATION_KEYS`. Edit the prompt text and key list together to add or remove classification fields.

For a hands-on example, see [Change provided example prompt in HANDS_ON_TOUR.md](HANDS_ON_TOUR.md#change-provided-example-prompt).

## 💬 4. Config_RAGChatService.py — HTTP Service Configuration

`Config_RAGChatService.py` re-exports all settings from `Config_RAGChat.py` and adds service-specific configuration for the OpenAI-compatible REST API.

### 🔧 Service-Specific Settings

> **Note:** The listener configuration (HOST, PORT, API_KEY) is defined in `Config_Models.py` under `_MODELS.ragchatservice._RAGCHATSERVICE`. See [ragchatservice — RAGChatService HTTP Listener Configuration](#-ragchatservice--ragchatservice-http-listener-configuration) for details.

| Key | Default | Purpose |
| --- | --- | --- |
| `_QUERY_LOG` | `<project>/logs/RAGChatService/queries.log` | Audit log path for web-search attempts (separate from RAGChat logs) |
| `_INTENT_FILTER_LOG` | `<project>/logs/RAGChatService/intent_filter.log` | Intent classifier decision log |
| `OPENWEBUI_THREAD_POOL_WORKERS` | `2` | Maximum workers in the ThreadPoolExecutor |
| `SHOW_CLI_LIKE_ALGO_RESULTS` | `True` | Append filter chain results to LLM answers in Markdown format |
| `_SERVE_DOCS.cors_origins` | `[]` | Allowed browser origins for the `/marked` endpoint |
| `_SERVE_DOCS.ttl_seconds` | `1800` | TTL for highlighted documents in cache (30 min) |
| `_SERVE_DOCS.max_total_mb` | `200` | Maximum total size of cached documents in MB |
| `_SERVE_DOCS.single_use` | `False` | Destroy cache entry after first fetch |
| `_SERVE_DOCS.public_base_url` | `""` | Externally-reachable base URL for `/marked` links |

### 📋 Re-Exported Settings from Config_RAGChat.py

All settings from `Config_RAGChat.py` are available in `Config_RAGChatService.py`:

- `_STRATEGIES` (NARROW, WIDE, BALANCED_FILE_CAP, ULTRA_WIDE, DEFAULT)
- `_ALLOWED_RETRIEVE_MODES`
- `_KEY_BERT` (flat structure)
- `_QUERY_REWRITE` parameters
- `_MARKED_DOCS_COLORS` and `_MARKED_DOCS_GROUNDING`
- All other RAGChat configuration

### 🔄 Relationship to Config_RAGChat.py

`Config_RAGChatService.py` uses `from Configuration.Config_RAGChat import *` to inherit all settings, then adds its own service-specific overrides and additions. This ensures consistent behavior between the CLI and service versions while allowing endpoint-specific configuration.

## 🚧 7. Config_Banned.py — Detection, Thresholds, and Masking

After editing `Config_Banned.py` update `_BANNED_CONFIG_HASH` in `Config_Global.py`.

### 🧮 Detection Algorithms

Five algorithms are available:

| Constant | Algorithm | Description |
| --- | --- | --- |
| `_JACCARD` | Jaccard | Character n-gram overlap similarity. |
| `_BM25` | BM25 | Term-frequency / inverse-document-frequency scoring. |
| `_REGEX` | Regex+Levenshtein | Two-step pattern matching: strict word-boundary match first, then optional fuzzy anchored match with Levenshtein edit-distance scoring. |
| `_KEYBERT` | KeyBERT | Keyword-based semantic detection. |
| `_COSINE` | Cosine | Embedding-based semantic similarity (disabled because Cosine and Keybert scorers produce similar values. Having both in the pipeline would put too much emphasis on cosine similarity). |

`_DEFAULT_ALGOS` selects which algorithms are active by default used in this repository:

```python
_DEFAULT_ALGOS = [_JACCARD, _BM25, _REGEX, _KEYBERT]
```

### 📋 Per-App Detection Profiles

Detection is configured per application inside `_BANNED_DETECT[_ACTIVE_DETECTION_CONFIG]`. Each app (RAGLoad, RAGChat, DocClassify) has three sections:

- **MASKING** — whether to redact matched spans before processing (`APPLY_MASKING`).
- **PROMPT_CHECK** — whether to run an LLM-based compliance check on the prompt (`Check`), and with which LLM parameters.
- **PIPELINE_CHECK** — the retrieval/content pipeline with per-algorithm thresholds.

Example: RAGLoad disables prompt checking (`"Check": False`) but enables masking and pipeline checks. RAGChat enables all three.

### 📊 Algorithm Thresholds

Each algorithm entry in the `PIPELINE` dict has:

| Key | Purpose |
| --- | --- |
| `THRESHOLD` | Primary trigger threshold. |
| `THRESHOLD_MIN` | Noise floor — only scores above this value are kept; anything below is discarded as noise. |

Algorithm-specific parameters:

- **Jaccard**: `CHAR_NGRAM_RANGE` — character n-gram range for similarity (default used in this repository `(4, 6)`).
- **BM25**: `TERM_FREQ_SATURATION` (k1), `LENGTH_NORMALIZATION` (b), `MIN_OVERLAP`, `MIN_RAW_SCORE`, `NORM_PERCENTILE`.
- **Regex**: Two-step matching controlled by three scoring keys. `SOFT_SCORE_HARD` is the score assigned on a strict (exact word-boundary) match. `SOFT_SCORE_FUZZY` is the lower score assigned when only the fuzzy anchored pattern matches. `FUZZY_REGEX_EVAL_AFTER_HARD` (`True`/`False`) controls whether the fuzzy step runs at all — when `False`, only strict matching is used. Additional parameters: `WINDOW_MAX_CHARS`, `PREFIX_SUFFIX_LEN`, `SEPARATOR_CLASS`, and a nested `Levenshtein.THRESHOLD`.
- **KeyBERT**: `TOP_K` — number of keywords extracted per check (larger at load time, smaller at chat time).
- **Cosine**: No additional parameters beyond `THRESHOLD` and `THRESHOLD_MIN`. Disabled by default; enable if embedding vectors are available (commented out in `Config_Banned.py`).

### 🤝 Consensus Rules

Two parameters control how algorithms vote together (see [Consensus Scoring & Experimentation in ARCHITECTURE.md](ARCHITECTURE.md#consensus-scoring--experimentation) for details):

| Key | RAGLoad (pipeline) | RAGChat (prompt check) | RAGChat (pipeline) | DocClassify (prompt check) |
| --- | --- | --- | --- | --- |
| `REQUIRED_ALGOS_ABOVE_THRESHOLD` | 3 | 2 | 4 | 4 |
| `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE` | 4 | 3 | 4 | 4 |

- **`REQUIRED_ALGOS_ABOVE_THRESHOLD`** (Depth) — how many algorithms must score a phrase **above their individual `THRESHOLD`**. Measures strength of signal: a phrase is flagged only when enough algorithms independently consider it a strong match.
- **`REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE`** (Breadth) — how many distinct algorithms must produce a score **above `THRESHOLD_MIN`** for a phrase, regardless of whether that score exceeds the algorithm's primary `THRESHOLD`. Each algorithm already discards scores below its `THRESHOLD_MIN` (noise floor), so only meaningful signals reach the breadth count. Use case: catch variations by requiring multiple algorithms to detect something, even if each individual signal is below the primary threshold.

Note: RAGChat has **two separate compliance pipelines**:

- **PROMPT_CHECK** validates user prompts (2/3 thresholds for responsiveness)
- **PIPELINE_CHECK** validates retrieved content (4/4 thresholds for strictness)

Raising these values makes detection stricter (fewer false positives, more false negatives). Lowering them catches more violations but increases false positives.

### 🚫 Banned Words

`_ACTIVE_BANNED_CONFIG` points to the active banned-word list. The default used in this repository `_STRICT_BANNED` contains terms covering personal identifiers (SSN, passport, IBAN), credentials (API keys, JWTs, passwords), health and biometric data, protected attributes, and security-related terms. Add or remove entries to match your use case.

### 🎭 Masking Regexes

`_ACTIVE_MASKING_CONFIG` points to the active masking configuration. Each rule in `_STRICT_MASKING_REGEXES["MASKING_REGEXES"]` has:

| Field | Purpose |
| --- | --- |
| `pattern` | Python regex pattern to match. |
| `mask` | Replacement action — a literal string like `"[SSN]"` or a named handler like `"mask_credit_card"`. |
| `enabled` | `True` / `False` toggle. |
| `priority` | Higher priority rules are evaluated first. |
| `desc` | Human-readable description. |

Built-in rules cover credit cards, emails, SSNs, IBANs, IP addresses, MAC addresses, UUIDs, JWTs, AWS keys, passwords, and more. Rules marked `enabled: False` (e.g. CVV, IBAN, phone numbers) can be turned on when needed.

Masking is applied on document ingestion and on RAGChat query output.

To define a custom masking profile, create a new dictionary (e.g. `_MY_MASKING_REGEXES`) and point `_ACTIVE_MASKING_CONFIG` to it.

### 🌐 Web-Search Intent Filter Extensions

`WEB_SEARCH_INTENT_EXTENSIONS` (at the end of `Config_Banned.py`) lets operators
extend the baseline web-search intent classifier defined in `Config_WebSearch.py`
without editing that file. Three keys are supported:

| Key | Type | Purpose |
| --- | --- | --- |
| `entity_extensions` | `dict[str, list[str]]` | Add terms to the entity lists of existing baseline categories. Key = category name (must match a name in the baseline), value = list of additional terms. |
| `entity_categories_extra` | `dict[str, dict]` | Define entirely new intent categories not present in the baseline. Each value must follow the same schema as a baseline category entry. |
| `threshold_overrides` | `dict[str, int]` | Override the score threshold for individual baseline categories. Key = category name, value = new integer threshold. |

All three default to empty dicts — no behavioral change until populated.

> **After editing `WEB_SEARCH_INTENT_EXTENSIONS`**, update `_BANNED_CONFIG_HASH`
> in `Config_Global.py`. Run `python src/Scripts/RecalcConfigHashes.py` to
> calculate and apply the new hash automatically.

## 🌐 8. Config_Internet_Env.py — Internet Access and Network Tracing

This file controls all internet connectivity and diagnostic toggles. It is described in detail in the [Internet Access](INSTALL.md#-internet-access) section. The key environment variables are:

| Environment Variable | default used in this repository | Purpose |
| --- | --- | --- |
| `LICENSE_DOWNLOAD` | `"0"` | Online fetch license on every run. Defined in `Config_Models.py`. When `"0"`, the Compliance module prompts for per-fetch consent. |
| `NLTK_STOPWORDS_DOWNLOAD` | `"0"` | Allow download of missing NLTK stopwords corpus. When `"0"`, the system falls back to an empty stopword list. |
| `RAG_LCC_NW_TRACE` | `"0"` | Socket-level network tracing (debug). |
| `RAG_LCC_STACK_TRACE` | `"0"` | Stack traces on errors. |
| `WEB_SEARCH_MODE` | `"0"` | **Master web-search switch.** `"0"` = disabled (safe default); `"1"` = internet search enabled (user queries may be sent to DuckDuckGo). Operators enabling `"1"` must review `LEGAL.md § Web Search` and `SECURITY.md`. |
| `TESSERACT_PATH` | `r"C:\Program Files\Tesseract-OCR\tesseract.exe\|/usr/bin/tesseract"` | OS-aware Tesseract OCR path. Format: `"windows_path\|linux_path"`. The framework selects the appropriate half at runtime. Set via `os.environ.setdefault` (only applied when not already set in the environment). |
| `HF_HUB_OFFLINE` | `"1"` | Disable Hugging Face Hub downloads when `"1"` (safe default). Set to `"0"` to allow model downloads. |
| `TRANSFORMERS_OFFLINE` | `"1"` | Disable transformers library hub access when `"1"`. |
| `HF_DATASETS_OFFLINE` | `"1"` | Disable HF datasets hub access when `"1"`. |
| `ARGOS_STANZA_DOWNLOAD` | `"0"` | Control Argos Translate consent and package downloads. `"0"` = only use pre-installed language pairs. `"1"` = prompt for Argos license consent and download language packages at startup if consent has not yet been recorded. |
| `ARGOS_CHUNK_TYPE` | `"SPACY"` | Sentence boundary detection backend for Argos Translate. `"SPACY"` = SpaCy sentencizer (offline, default). `"STANZA"` = stanza Pipeline (broken offline in argos-translate ≥ 1.11). |
| `ARGOS_MODEL_PROVIDER` | `"OPENNMT"` | Force Argos Translate to use local packages only. |
| `SERVE_OPENWEBUI_CHAT` | `"0"` | When `"1"`, RAGChatService accepts connections from OpenWebUI (inbound only). Printed as an info banner at startup. |
| `SERVE_IN_MEMORY_DOCS_HTTP` | `"0"` | Enable the in-memory document HTTP server used by RAGChatService for `/marked/<token>` links. `"0"` = disabled (no in-memory docs store). `"1"` = enabled. See `_SERVE_DOCS` in `Config_RAGChatService.py` for TTL and size limits. |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `"0"` | Show Hugging Face Hub progress bars during model downloads. Set to `"1"` to suppress progress bar output for cleaner log output. |
| `TOKENIZERS_PARALLELISM` | `"false"` | Prevent tokenizer parallelism warnings from HuggingFace tokenizers. Set via `os.environ.setdefault` (only applied when not already set). |

## 💻 CLI Parameter Override

You can override any uppercase, non-underscore-prefixed key from `Config_Global.py` or the app-specific config file (`Config_RAGChat.py`, `Config_RAGLoad.py`, `Config_DocClassify.py`, `Config_RAGChatService.py`) via the command line. Keys in `Config_Models.py` and `Config_Banned.py` are not available as CLI arguments — edit those files directly.

```bash
python ./src/Apps/RAGLoad.py --collection mytest --doc_dir MyDocs --debug_level 6
python ./src/Apps/RAGChat.py --collection mytest
python ./src/Apps/RAGChatService.py --collection mytest
python ./src/Apps/DocClassify.py --collection mytest --debug_level 4
```

Run with `--help` to see all overridable parameters.

## 📝 Language detection configuration

Language detection tuning lives in a dedicated `_LANGUAGE_DETECTION` slot in
`Config_Global.py`, separate from Argos Translate settings.
The top-level `UNSUPPORTED_LANGUAGE_ACTION` key (also in `Config_Global.py`) controls what happens when a document language is not supported by the installed Argos translation packages:

| Value | Behaviour |
| --- | --- |
| `"NOT_OK"` | Reject the document — compliance check fails for unsupported languages (default, safe). |
| `"FALLBACK_EN"` | Treat the document as English and continue processing. Use only when non-English content is not a concern. |

```python
UNSUPPORTED_LANGUAGE_ACTION = "NOT_OK"   # "NOT_OK" | "FALLBACK_EN"
```

| Key | Type | Purpose |
| --- | --- | --- |
| `MIN_WORDS` | `int` | Minimum number of words a text must contain before language detection is attempted. Texts shorter than this threshold skip detection and fall back to `"en"`. Prevents single words or very short strings from being misclassified (e.g. `"igel"` detected as Danish instead of German). Default: `3`. |
| `MIN_CONFIDENCE` | `float` | Confidence floor (0–1) applied when a text is at least `CONF_FULL_WORDS` words long. For shorter texts the effective threshold is scaled up linearly toward `0.90`, so ambiguous short queries naturally fall back to `"en"`. Default: `0.60`. |
| `CONF_FULL_WORDS` | `int` | Word count at which `MIN_CONFIDENCE` is used without any upward scaling. Below this count the threshold rises linearly toward `0.90`. Default: `10`. |

## 🌍 Translation configuration (Argos)

Argos Translate settings live in a single `_ARGOS_DEFINITIONS` slot inside
`Config_Global.py`. It contains two keys:

| Key | Type | Purpose |
| --- | --- | --- |
| `LANG_CODE_TO_NAME` | `dict` | Maps ISO-639-1 codes (e.g. `"de"`) to NLTK / human-readable names (e.g. `"german"`). Used for language detection, stopword lookup, and the reverse mapping (name → code) in `SharedHelpers`. |
| `ARGOS_LANGUAGES` | `list[tuple]` | Translation pairs `(from_code, to_code)` that the install script and startup consent check use to download and verify Argos Translate packages. Argos is used **only** by the Compliance pipeline to translate the English banlist into the document language, so only `(en, X)` pairs are needed. Only uncommented pairs are active. |

```python
_ARGOS_DEFINITIONS = {
    "LANG_CODE_TO_NAME": {
        "ar": "arabic",
        "de": "german",
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "it": "italian",
        # … full list in Config_Global.py
    },
    "ARGOS_LANGUAGES": [
        # Uncomment the pairs you need — each pair downloads ~100 MB.
        # ("en", "ar"),  # English → Arabic
        ("en", "de"),    # English → German
        ("en", "es"),    # English → Spanish
        ("en", "fr"),    # English → French
        ("en", "it"),    # English → Italian
        # ("en", "ja"),  # English → Japanese
        # … 48 pairs available, see Config_Global.py for the full list
    ],
}
```

After changing `ARGOS_LANGUAGES`, run the install script to download the
newly enabled packages:

```bash
python src/Scripts/ArgosTranslatePackages.py install
```

To remove all installed packages, stanza models, and consent metadata:

```bash
python src/Scripts/ArgosTranslatePackages.py remove
```

## 🔧 Troubleshooting

| Issue | Cause | Solution |
| --- | --- | --- |
| `Detected modification of Configuration.Config_Models` / `Update Expected hash … _MODELS_CONFIG_HASH to match expected hash` | Edited `Config_Models.py` without updating hash | Copy the new hash from the startup message into `_MODELS_CONFIG_HASH` in `Config_Global.py`. See [Update the hashes](INSTALL.md#-update-the-hashes). |
| `Detected modification of Configuration.Config_Banned` / `Update Expected hash … _BANNED_CONFIG_HASH to match expected hash` | Edited `Config_Banned.py` without updating hash | Copy the new hash from the startup message into `_BANNED_CONFIG_HASH` in `Config_Global.py`. See [Update the hashes](INSTALL.md#-update-the-hashes). |
| 'ModuleNotFoundError: No module named 'Configuration.Config_Banned' | You forgot to copy Config_Banned.py | See [Review the example config files](INSTALL.md#-review-the-example-config-files) and [Copy example configs into place](INSTALL.md#-if-ok-copy-example-configs-into-place) |
| 'ModuleNotFoundError: No module named 'Configuration.Config_Models' | You forgot to copy Config_Models.py | See [Review the example config files](INSTALL.md#-review-the-example-config-files) and [Copy example configs into place](INSTALL.md#-if-ok-copy-example-configs-into-place) |
| `Execution stopped due to compliance check` | Config hash not updated after editing `Config_Models.py` or `Config_Banned.py` | Update both `_MODELS_CONFIG_HASH` and `_BANNED_CONFIG_HASH` in `Config_Global.py` and restart. See [Update the hashes](INSTALL.md#-update-the-hashes). |
| Embeddings seem wrong | Changed embedding model without re-embedding | Set `RETRIEVAL_STORES_KEEP = False` and re-run RAGLoad or delete the collection manually (`./chromadb/docs`) |
| RAGChat is slow | Too many `NEIGHBORS_RETRIEVE` or large `CHUNK_SIZE` | Reduce both in `Config_Global.py` |
| Detection not working | Phrases not in banned list | Add to `Config_Banned.py` and [update the hash](INSTALL.md#-update-the-hashes) |
| Low retrieval quality | Bad chunk settings | Test `CHUNK_SIZE`: 128, 256, 512. Regenerate (load) the collection |
| `RequestsDependencyWarning: urllib3 … or chardet … doesn't match a supported version!` | `chardet` ≥ 6 installed but `requests` requires `chardet < 6` | Run `pip install "chardet<6,>=3.0.2"` to downgrade to a compatible version (e.g. 5.2.0) |
| `Language en package default expects mwt, which has been added` | Stanza (used by Argos Translate) auto-adds the Multi-Word Token processor for the English model | Harmless informational warning — no action required |
| Cursor displaced / misaligned in RAGChat terminal prompts (VS Code) | `pyreadline3` uses low-level Win32 console APIs (`windll.kernel32`) that conflict with VS Code's xterm.js terminal emulator. Known upstream issues: [#40](https://github.com/pyreadline3/pyreadline3/issues/40), [#43](https://github.com/pyreadline3/pyreadline3/issues/43). No upstream fix exists. | RAG‑LCC replaced `pyreadline3` with `prompt_toolkit` for interactive input history. If you still experience cursor issues, run `python test_cursor.py prompt_toolkit` vs `python test_cursor.py pyreadline3` in a fresh VS Code terminal to confirm the cause. Ensure `pyreadline3` is not imported anywhere in your environment. See also `tests/test_cursor.py`. |

## ⚡ Performance Tuning Checklist

- [ ] Set `_PROCESS_IF_UNCHANGED = False` to skip re-processing unchanged files
- [ ] Set `RETRIEVAL_STORES_KEEP = True` to preserve embeddings between runs
- [ ] Use `EMBEDDER_BITS = 16` on GPU for speed (slight quality loss)
- [ ] Reduce `CHUNK_SIZE` to reduce document processing time
- [ ] Reduce `REQUEST_TIMEOUT` for faster failure on slow Ollama responses
- [ ] Use a smaller LLM model

## ℹ️ Further Information

- Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md)
- Legal and governance: [LEGAL.md](LEGAL.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Example usages and CLI examples: [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md)
- Configuration deep-dive: see [Configuration Reference](CONFIGURATION.md) above
- Troubleshooting: [Troubleshooting](#-troubleshooting)

## 📌 Constraints

- RAG-LCC is an experimental lab tool and may contain errors.
- RAG-LCC does not send an `HF_TOKEN` to Hugging Face.
- RAG-LCC does **not** provide access controls.
- RAG-LCC is intended for single-operator usage. No thread safety.

---
