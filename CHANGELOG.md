<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to RAG-LCC are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v0.2.0/1073] — 2026-04-12

### ➕ Added

#### 🧠 WordNet Synonym Expansion for Banned-Word Scoring

- **`src/Algos/Synonyms.py`** — New `Synonyms` singleton that expands banned-word
  lists with WordNet synonyms before they are evaluated by **BM25Scorer**,
  **JaccardScorer**, and **RegexScorer**.  KeyBertScorer is intentionally excluded
  (embeddings already capture semantic neighbours).
  - Configurable via `_WORDNET` config block: `ENABLED`, `DEPTH` (hop count),
    `MAX_SYNONYMS_PER_PHRASE`, `POS_FILTER` (noun/verb/adj/adv), and `STOPLIST`
    to suppress overly generic expansions.
  - Lazy WordNet import — avoids startup crash when NLTK / WordNet corpus is not
    installed; falls back gracefully to the original banned-word list with a warning.
  - Result caching per input list for repeated lookups within the same session.
  - Multi-word phrase support: looks up underscore-joined WordNet lemma first, then
    falls back to individual token lookup.
  - Breadth-first synonym-of-synonym traversal for `DEPTH ≥ 2`.

### 🐛 Fixed

#### 🖥️ CLI Cursor Positioning

- **`src/Gui/HistoryManager.py`** — Replaced `pyreadline3` with `prompt_toolkit`
  for CLI input handling.  `pyreadline3` miscalculated visible prompt width when
  ANSI escape codes or emoji were present, causing the cursor to be displaced from
  the actual typing position in VS Code's integrated terminal.  `prompt_toolkit`'s
  `ANSI()` formatted-text wrapper computes visible width correctly, keeping the
  cursor aligned.
- **`tests/test_cursor.py`** — Regression tests added: verifies `prompt_toolkit`
  import, absence of `pyreadline3` references in `HistoryManager.py`.

---

## [v0.1.0/1067] — 2026-04-07

### 🔧 Improved

#### 🎨 RAGChatService — Prettified Filter-Chain Result Display

- **`src/Api/ChatCompletionHandler.py`** — Filter-chain compliance results returned
  to OpenWebUI are now formatted as styled Markdown (emoji colour indicators, aligned
  tables, depth/breadth summary) instead of raw plain-text output.  Prompt-check and
  answer-check stages both emit a `### Filter chain algo results` block that renders
  cleanly inside the OpenWebUI chat interface.
- **`src/Helpers/Accumulator.py`** — New `format_results_as_md()` method produces an
  OpenWebUI-safe Markdown representation of the last ensemble check. Includes a
  per-phrase table with emoji-coloured score cells (🔴 above threshold, 🟢 clean,
  🔵 below threshold, ⚪ disabled) and a depth/breadth summary row mirroring the
  CLI ANSI output.  Snapshot fields `last_phrase_table_for_md` and
  `last_ensemble_data` added so the service layer can retrieve results after
  `show_accumulated()`.
- **`src/Api/ChatCompletionHandler.py`** — New `_format_validation_details()` helper
  groups `ResultsForPrint` rows by phrase and renders them as a Markdown
  `**Validation details:**` block with per-algo score lines.

#### 🔒 Thread Safety & Singleton Cleanup

- **`src/Globals/Session.py`** — `Session` no longer inherits `SingletonMixin`.
  The CLI path creates one instance and reuses it across the interactive loop; the
  service path creates a **fresh instance per incoming API request** so that
  concurrent requests cannot clobber each other's state.  Circular import of
  `AIHelpers` removed from `__init__`.
- **`src/Config/Config.py`** — `Config.get()` / `Config.set()` wrapped with
  `threading.RLock()`.  Public methods delegate to private `_get()` / `_set()` under
  the lock, making all configuration reads and writes safe under concurrent API
  requests.
- **`src/Chat/RAGChatImpl.py`** — `set_vector_store()` and `retrieve()` wrapped with
  `threading.Lock()`.  Public wrappers acquire the lock and delegate to private
  `_set_vector_store()` / `_retrieve()`, preventing concurrent ChromaDB collection
  switches from interleaving.
- **`src/Strategies/HomeBrewChunkSelector.py`** — `ChunkSelector`, `ChunkSelectionService`,
  and all concrete selectors (`WideUltraWideSelector`, `MediumSelector`,
  `NarrowSelector`) now accept `session: Session` as a constructor parameter instead
  of calling `Session()` singleton internally.  `RAGChatImpl._retrieve()` passes the
  per-request session when constructing `ChunkSelectionService`.
- **`src/Chat/Chatter.py`** — `run()` now accepts `session: Session` as a positional
  parameter (no longer stored as `self.session`), removing the last implicit
  singleton dependency from the hot path.

### 📖 Documentation

- **Mermaid diagrams** added to project documentation.  Architecture and flow
  visualisations are now rendered with Mermaid for maintainability and inline
  preview support.

---

## [Unreleased] — 2026-04-01

### ➕ Added

#### 🌐 RAGChatService — OpenWebUI / OpenAI-Compatible REST API

- **`src/Apps/RAGChatService.py`** — New FastAPI application served via `uvicorn`
  that exposes RAGChat as an OpenAI-compatible HTTP service.  ChromaDB collections
  are surfaced as selectable models via `GET /v1/models`; chat inference is handled
  via `POST /v1/chat/completions`.  Requires `SERVE_OPENWEBUI_CHAT="1"` in
  `Config_Internet_Env.py`; exits with an explanatory message if the variable is
  absent or disabled.
- **`src/Api/ChatCompletionHandler.py`** (new `src/Api/` module) — FastAPI/Pydantic
  handler that validates incoming `ChatCompletionRequest` payloads, maps OpenAI-style
  fields to RAG-LCC `Session` parameters, and dispatches to `Chatter.run()`.
  Features:
  - Streaming (Server-Sent Events) and non-streaming JSON response paths.
  - Per-request Ollama option passthrough (`seed`, `mirostat`, `repeat_penalty`,
    `num_gpu`, etc.) and top-level payload param forwarding (`think`, `keep_alive`,
    `format`).
  - Automatic detection and short-circuit of OpenWebUI housekeeping prompts
    (follow-up suggestions, tag generation, title generation) — these bypass the
    RAG pipeline entirely.
  - RAG-LCC-specific advanced parameters (`strategy`, `chroma_k_value`, `rerank`,
    `chroma_threshold`, `chunks_window`, `chroma_weight`, `per_file_limit`) that can
    be set via OpenWebUI model **Advanced Parameters**.
  - `chat_id` field accepted (OpenWebUI per-conversation identifier); injected into
    session for optional chat-context keying.
- **`src/Configuration/Config_RAGChatService.py`** — New configuration module that
  re-exports the full `Config_RAGChat` namespace and adds service-specific keys:
  `OPENWEBUI_API_HOST` (default `127.0.0.1`), `OPENWEBUI_API_PORT` (default `11435`),
  `OPENWEBUI_THREAD_POOL_WORKERS` (default `1`).  Overrides `_FRIENDLY_NAME` to
  `"RAGChatService"` and customises `_PROMPT_CHAT` / `_PROMPT_CHAT_MISTRAL` /
  `_PROMPT_CHAT_LLAMA` with OpenWebUI controls-sidebar hints in the no-context
  fallback message.
- **`SERVE_OPENWEBUI_CHAT`** environment variable added to `Config_Internet_Env.py`
  (default `"1"`).  Documented launch command:
  `.venv\Scripts\python.exe src/Apps/RAGChatService.py`.

#### 🏗️ _MODELS Hierarchy — Centralised Model Registry

- **`src/Configuration/Config_Models.py`** promoted from `Examples/Example_Config_Models.py`
  and substantially expanded.  The file now serves as the single authoritative model
  registry:
  - Full governance/documentation header covering license consent, model retrieval
    scope, configuration hash purpose, and implementation selectors.
  - `_MODELS` dictionary with nested `impl → role → config` lookup.
  - Implementation selectors: `_LLM`, `_EMBED`, `_CROSS`, `_OLLAMA`, `_OPENWEBUI`,
    `_LLM_CHK`.
  - New impls: `openwebui` (`_OPENWEBUI` role — OpenWebUI connection metadata) and
    `llama_guard` (`_LLM_CHK` role — Llama Guard 3 8B safety model).
  - All existing model entries (`snowflake`, `mmarco`, `mistral`, `llama`, `ollama`)
    updated to include `"RAGChatService"` in their `USED_BY` list.
- **`src/Configuration/Config_Banned.py`** promoted from `Examples/Example_Config_Banned.py`.
  Added `"RAGChatService"` entry in `_BANNED_DETECT["STRICT_DETECT_CONFIG"]` pointing
  to the same compliance pipeline as `"RAGChat"`.
- Flat config keys **removed** from `Config_Global.py`:
  `USE_OLLAMA_GPU`, `_OLLAMA_BASE_URL`, `OLLAMA_STREAMING_REQ`.
  All consumers now resolve these values from `_MODELS.ollama._OLLAMA.*` via
  `Helpers.get_model_args("_OLLAMA")`.
- Config hash validation **enabled**: `_MODELS_CONFIG_HASH` and `_BANNED_CONFIG_HASH`
  in `Config_Global.py` are now populated (previously left blank).

### 🔧 Improved

#### 🔒 Thread Safety for Concurrent API Requests

- **`Config.get()` / `Config.set()`** — wrapped with `threading.RLock()`;
  public methods delegate to private `_get()` / `_set()` under the lock, making
  all configuration access safe under concurrent API requests.
- **`RAGChatImpl.set_vector_store()` / `RAGChatImpl.retrieve()`** — wrapped with
  `threading.Lock()`; public wrappers acquire the lock and delegate to private
  `_set_vector_store()` / `_retrieve()`.

#### 📦 Session — Singleton Removed, Per-Request Design

- **`Session`** no longer inherits `SingletonMixin`.  Doc comment makes the intent
  explicit: the CLI path creates one instance and reuses it across the interactive
  loop; the service path creates a **fresh instance per incoming API request** so
  that concurrent requests cannot clobber each other's state.
- Added fields for API option forwarding:
  - `extraOllamaOptions: dict[str, Any] | None` — Ollama `options` sub-dict entries
    forwarded from API clients (e.g. `mirostat`, `seed`, `repeat_penalty`).
  - `ollamaTopLevelParams: dict[str, Any] | None` — top-level Ollama payload params
    (`think`, `keep_alive`, `format`).

#### 🤖 Chatter — Streaming Callback & Service Integration

- **`Chatter.run()`** refactored: now accepts `session: Session` as a positional
  parameter (previously stored as `self.session`), an optional
  `apiChunkHandler: Callable[[str], None]` callback for streaming token forwarding,
  and `is_streaming: bool | None` to override the config default per-call.
- When `apiChunkHandler` is provided, the base terminal/CSV chunk handler is wrapped
  so both receive each streamed token.
- "No results" message branched by `_FRIENDLY_NAME`: RAGChatService gets Markdown
  with OpenWebUI Controls sidebar hints; RAGChat gets plain text CLI hints.
- `run()` return signature changed: `bool` → `tuple[bool, str | None]`
  (success flag + final answer text, used by the API path).
- Reads `is_streaming` and `USE_GPU` from `_MODELS.ollama._OLLAMA` dict instead
  of flat config keys.

#### 🔑 QueryParts — Public Session API

- `reset_things()` made public (was `_reset_things()`); callers updated.
- `applyStrategyDefaults(strategy, session=None)` added as an explicit public entry
  point for the API path.
- `_base_defaults()` now accepts an optional `session` keyword parameter; falls back
  to `self.session` when not provided.

#### 🛰️ Informer — OpenWebUI Startup Check

- Added `OpenWebUINotRunning` exception.
- `_check_openwebui_is_running()` — pings the configured OpenWebUI `BASE_URL`
  (`/v1/models`) on startup when `friendly_name == "RAGChatService"` and raises
  `OpenWebUINotRunning` with a clear message if unreachable.
- Reads `is_streaming` / `BASE_URL` from `_MODELS` hierarchy.

#### 🚀 LLMCaller — Option Passthrough

- `call()` gains `extra_options: dict[str, Any] | None` and
  `top_level_params: dict[str, Any] | None` parameters; values are merged into the
  Ollama request `options` dict and top-level payload respectively.
- Ollama `BASE_URL` resolved from `_MODELS.ollama._OLLAMA` instead of flat key.
- Removed `stream_url` parameter (obsolete after config restructure).

#### 🧩 Strategy Selector Injection — Session Parameter

- `HomeBrewChunkSelector`, `WideUltraWideSelector`, `MediumSelector`,
  `NarrowSelector` — all now accept `session: Session` in `__init__()` and store it
  as an instance variable instead of calling `Session()` (singleton).
- `RAGChatImpl._retrieve()` passes `session` when constructing `ChunkSelectionService`.

#### ⚙️ StartupCommons

- Added environment check gate for `SERVE_OPENWEBUI_CHAT` when
  `_FRIENDLY_NAME == "RAGChatService"`.
- Startup informational message updated: Ollama `BASE_URL` and `STREAMING_REQ` now
  read from `_MODELS.ollama._OLLAMA.*`.

#### 🔢 Token Budget

- `TOKEN_BUDGET_CONTEXT_CAP` raised from `16384` to `16384 × 1.5` (≈ 24 576 tokens)
  to support larger context windows.

#### 💬 Prompt Updates

- `Config_RAGChat._PROMPT_CHAT` tightened — removes model self-verification steps;
  instructs model to output the exact two-line fallback without reasoning or
  rephrasing.
- `Config_RAGChatService._PROMPT_CHAT` adds OpenWebUI Controls sidebar hints
  (`chroma_k_value`, `chroma_threshold`, `strategy`) as Markdown in the no-context
  fallback message.

### 🐛 Fixed

- **`Config_Global.py`** — `_DEBUG_LEVELS` key typo `"Alogs"` corrected to
  `"Algos"`.
- **`Config_Internet_Env.py`** — `HF_HUB_OFFLINE` default changed from `"1"` to
  `"0"` (Hugging Face hub is now accessible by default; was offline-first).
- **`Config_Internet_Env.py`** — `RAG_LCC_STACK_TRACE` default changed from `"0"`
  to `"1"` (stack traces enabled by default for diagnostics).
- **`Apps/RAGChat.py`** — per-session state (`session`) is now explicitly shared
  across `QueryParts`, `CommandProcessor`, and `Chatter` so all components operate
  on the same object.

---

## [v0.1.3/1045] — 2026-03-27

### ➕ Added

#### 🌐 Language Detection Robustness

- **`LANG_DETECT_MIN_CHARS`** (`_ARGOS_DEFINITIONS` in `Config_Global.py`) — minimum
  number of characters a text must contain before language detection is attempted.
  Texts shorter than this threshold skip `langdetect` entirely and fall back to `"en"`.
  Prevents single-word inputs (e.g. `"igel"`) from producing high-confidence but
  incorrect language guesses (e.g. Danish instead of German). Default: `20`.
- **`LANG_DETECT_MIN_CONFIDENCE`** (`_ARGOS_DEFINITIONS` in `Config_Global.py`) —
  minimum `langdetect` probability required to accept the top-ranked language.
  If the top result falls below this threshold the detected language is discarded and
  `"en"` is used as fallback. Prevents short, ambiguous queries (e.g.
  `"tell me about llama"`) from being misclassified as a non-English language and
  incorrectly triggering translation warnings. Default: `0.90`.
- Both thresholds are consumed by `FileUtils.detect_language()` via
  `cfg.get_float` / `cfg.get_int` lookups and apply uniformly across all three
  entry points (RAGLoad, RAGChat, DocClassify).
- COLLECTION_KEEP default is now False
- `HUMAN_REVIEW` log file output fixed. Shows now `DEPTH`and `BREADTH` trigger indication.

### 🐛 Fixed

- `COLLECTION_KEEP` default is now `False`

---

## [v0.1.1/1035] — 2026-03-25

### 🐛 Fixed

- **Empty "Algos Matched" CSV column** — `ResultsForPrint` dataclass in
  `ComplianceAlgoResult.py` was missing an `algos_matched` field. The phrase-level
  algorithm-match string was discarded in `Accumulator._decompose_score_str()` and
  never reached `prepare_for_csv_print()`. Added `algos_matched: Optional[str] = None`
  to `ResultsForPrint` and populated it from the phrase-level map.
- **HUMAN_REVIEW CSV Status was "OK" instead of "NOT_OK"** —
  `ChunksToDBStrategy` and `ClassifyStrategy` set `Status` to `"OK"` before the
  human-review branch and never updated it. Added explicit
  `Status = "NOT_OK"` assignment before writing the HUMAN_REVIEW CSV in both
  strategies.

### 🔧 Improved

- **Column-aligned ClassifyCSVReader debug output** — the `Selected paths` debug
  lines now left-justify each column (file path and query-referenced CSV columns)
  to the widest value, so the `|` separators align across rows.

### 📖 Documentation

- `HANDS_ON_TOUR.md` — added **📂 Classify‑then‑Load** walkthrough section with
  Step 1 (DocClassify) and Step 2 (RAGLoad with `--classify-csv-query` examples).
- `README.md` — added **📂 Classify‑then‑Load** subsection under Examples with
  workflow diagram; updated the Classify‑then‑Load overview to mention SQLite query
  filtering; updated the High-Level Features bullet to surface SQL WHERE capability.

### ➕ Added

#### 🌐 Unsupported-Language Handling

- **`UNSUPPORTED_LANGUAGE_ACTION`** config key (`Config_Global.py`) — controls
  what happens when a document’s detected language is not installed in Argos Translate.
  Valid values: `FALLBACK_EN` (default — legacy behaviour), `NOT_OK` (reject and
  write to NOT_OK CSV).
- **`SharedHelpers.is_language_supported(lang)`** — returns `True`
  for English or any installed Argos Translate language.
- **`SharedHelpers.check_language_support(lang, file_path)`** — shared gate
  that reads the config, logs a warning, and returns the action (`None`
  or `"NOT_OK"`). Used by all three entry points.
- **`UnsupportedLanguageError`** exception added to `Commons/Exceptions.py`.
- Gate added to all three entry points:
  - `ClassifyStrategy._process_extract()` — DocClassify
  - `ChunksToDBStrategy.docChunksToDBStrategy()` — RAGLoad
  - `AIHelpers.check_user_prompt_with_filter_chain()` — RAGChat
- 12 new tests in `test_shared_helpers.py` for `is_language_supported()` and
  `check_language_support()`.

#### 📂 Classify‑then‑Load Workflow

- **ClassifyCSVReader** (`Helpers/ClassifyCSVReader.py`) — Reads a `DocClassify` CSV
  and returns a set of allowed file paths for selective ingestion. Accepts a CSV
  filename (resolved relative to `logs/`) or an absolute path.
- **RAGLoad CSV integration** — `RAGLoad.py` accepts two config keys
  (`LOAD_FROM_CLASSIFY_CSV`, `CLASSIFY_CSV_QUERY`) to filter ingestion to documents
  classified by a prior `DocClassify` run.
  CLI flags: `--load-from-classify-csv <path>`, `--classify-csv-query <where>`.
- **`CLASSIFY_CSV_QUERY`** — optional SQL WHERE clause applied to the CSV rows via an
  in-memory `sqlite3` table. Supports standard SQLite syntax (`LIKE`, `AND`, `OR`,
  `NOT LIKE`, `=`, `!=`, `IN`, etc.).
  Example: `--classify-csv-query "Mammal LIKE '%Yes%' AND Language = 'English'"`.
- When the classify‑then‑load filter is active, exclusion checks (`USE_EXCLUSIONS`) are
  bypassed because `DocClassify` already evaluated exclusions during its run.

#### 🗂️ Variant Selector for ChromaDB Parameters

- **`_ACTIVE_CHROMA_EMBED_AND_RETRIEVE_PARAMS_CONFIG`** selector in `Config_Global.py`
  with two preset variants: `THOROUGH` (larger chunks, more HNSW neighbours) and
  `COMPACT` (smaller chunks, fewer neighbours).
- **`Helpers.get_chroma_config_slot()`** encapsulates the selector lookup; all consumers
  (`RAGChat`, `ChunksToDBStrategy`, `ModelsCache`, `ChromaDBHelper`) resolve parameters
  through dot-notation using the active variant.

#### 📖 Documentation

- Classify‑then‑Load workflow documented in `README.md` and `ARCHITECTURE.md`.
- **Selector Pattern Overview** section added to `ARCHITECTURE.md` cataloguing all
  selector + variant dictionary patterns across `Config_Models.py`, `Config_Global.py`,
  `Config_DocClassify.py`, `Config_Banned.py`, and `Config_RAGChat.py`.
- `HANDS_ON_TOUR.md` updated for the new variant structure.

#### 🔧 Refactoring — Instance Variable Naming Convention

- Removed `_` prefix from instance variables across the entire `src/` codebase to
  align with the project's naming convention (`self.cfg`, not `self._cfg`).
  Private **methods** retain their `_` prefix; `_initialized` / `_reset` (SingletonMixin
  internals) are also unchanged.
- **19 source files** updated: `ArgosDownloader`, `Accumulator`, `Exclusions`,
  `Compliance`, `SharedHelpers`, `ModelOutputAdapter`, `QueryParts`, `ChatContext`,
  `HFDownloader`, `Globals`, `ReverseStemmer`, `TokenBudget`, `RAGChatImpl`, `Masker`,
  `CosineScorer`, `KeyBertScorer`, `ValidExtensions`, `RegexScorer`,
  `CollectionPicker`.
- **5 test files** updated with corresponding reference changes
  (`test_argos_downloader`, `test_accumulator`, `test_accumulate_parity`,
  `test_hfdownloader`, `test_masker`).
- 25 additional files audited and confirmed as having no instance variables to rename.

#### 🧪 Testing

- 19 tests for `ClassifyCSVReader` (`tests/test_classify_csv_reader.py`), including
  8 tests for the `CLASSIFY_CSV_QUERY` sqlite3 filtering.
- `test_models_cache.py` updated for nested chroma config structure.

---

## [v0.1.0/1025] — 2026-03-17

### 🎉 Initial Experimental Release

RAG-LCC is a configuration-driven, offline-first **experimental lab and research framework**
for Retrieval-Augmented Generation with an integrated, configurable detection pipeline
(referred to in this project as "compliance"; see [LEGAL.md](LEGAL.md#definition--compliance-rag-lcc)
for the formal definition).

This release is intended **solely for laboratory evaluation and experimentation**.
It is **not intended for production use**.

See [LEGAL.md](LEGAL.md) for governance, liability, and responsibility boundaries.

> **Important Notices**
>
> - This software is provided **"as-is"**, without warranty of any kind. No guarantees
>   are made regarding correctness, fitness for purpose, availability, security, or
>   regulatory suitability. Operators assume all risk arising from use.
> - Detection and scoring are **probabilistic**. False positives and false negatives will
>   occur. RAG-LCC performs automated detection and scoring only; it does not make legal,
>   regulatory, or contractual determinations.
> - **Human review is required** for any blocking, redaction, disclosure, or enforcement
>   decision.
> - Operators are solely responsible for determining whether personal data may be
>   processed lawfully and whether a Data Protection Impact Assessment (DPIA) or other
>   regulatory assessment is required under applicable law.
> - Third-party packages, models, and tools are **not bundled**. Operators install all
>   dependencies from upstream sources and enter into direct licensing relationships with
>   their respective authors. See [3rdPartyLicenses/Licenses.md](3rdPartyLicenses/Licenses.md) for attribution information.
> - Model licenses must be obtained and accepted through the model owner’s official
>   distribution channel prior to download or use.

---

### ➕ Added

#### 📱 Applications

- **RAGLoad** — Document ingestion pipeline that extracts text, chunks content, generates
  embeddings, and upserts data into ChromaDB. Configurable detection pipelines may be
  applied during ingestion. Hash-based identification of unchanged documents can be used
  to skip reprocessing. See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow details.

- **RAGChat** — Interactive retrieval-augmented chat application supporting multiple
  retrieval strategies (NARROW, MEDIUM, WIDE, ULTRA_WIDE), optional cross-encoder
  re-ranking, streaming LLM responses, and detection checks applied to prompts and
  generated outputs. See [ARCHITECTURE.md](ARCHITECTURE.md) for retrieval and validation flow.

- **DocClassify** — Batch document classification using keyword extraction, stemming, and
  optional LLM-assisted label generation. Outputs are written to CSV/XLSX files. Supports
  STRICT, BALANCED, and RECALL extraction presets. See [ARCHITECTURE.md](ARCHITECTURE.md) and
  [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md) for examples.

#### 🧮 Detection Algorithms

- **Jaccard** — Character n-gram similarity with configurable ranges and thresholds.
- **BM25** — Okapi BM25 scoring with configurable term-frequency saturation, length
  normalization, and percentile-based normalization.
- **Regex** — Pattern-based detection with strict and optional fuzzy anchored matching.
- **Levenshtein** — Edit-distance matching for typo and variation tolerance.
- **KeyBERT** — SBERT-based semantic keyword extraction with phrase relevance scoring.
- **Cosine** — Embedding-based cosine similarity detection.
- **Consensus scoring** — Configurable depth and breadth rules combining algorithm
  outputs. See [ARCHITECTURE.md](ARCHITECTURE.md#consensus-scoring--experimentation).

#### 🛡️ Compliance & Governance (Technical)

- Per-application detection pipelines (RAGLoad, RAGChat, DocClassify) with configurable
  PIPELINE_CHECK and PROMPT_CHECK stages. See [ARCHITECTURE.md](ARCHITECTURE.md#compliance-chain).
- Model license consent tracking with metadata recording. See [LEGAL.md](LEGAL.md).
- Hugging Face model download flow with local-first resolution and explicit consent when
  downloads are required. See [ARCHITECTURE.md](ARCHITECTURE.md#hf-model-downloading--caching).
- Argos Translate license consent tracking and language package management.
- Configuration hash acknowledgement: changes to Config_Banned.py or Config_Models.py
  require operator acknowledgement via hash updates in Config_Global.py.
- CSV/XLSX outputs for items flagged for human review.

#### 📄 Text Processing

- Character-level masking using configurable, priority-ordered regex rules.
- Unicode normalization (e.g. NFKC, case-folding, whitespace normalization).
- Reverse stemming for restoring original surface forms in classification output.
- Language detection using langdetect with fallback handling.
- Stopword filtering using NLTK.

#### 📥 Document Extraction

- PDF text extraction.
- Microsoft Office formats (.doc/.docx, .ppt/.pptx, .xls/.xlsx) via locally installed
  Office COM interfaces.
- Image-based OCR via Tesseract.
- Plain-text formats with configurable extension lists.
- Legacy Office format conversion to modern equivalents.

#### 🌐 Translation

- Local, offline translation of banned-word lists using Argos Translate.
- Stanza tokenizer model support for language processing.
- Per-language banlist compilation for detection algorithms.

#### 🤖 LLM Integration

- Ollama-compatible endpoint communication with optional streaming responses.
- Token budget management with configurable context caps and reserved token ranges.
- Heuristic max_output_tokens calculation based on available context window.
- JSON extraction and repair strategies for malformed LLM output.
- Configurable model roles: generation LLM, compliance-check LLM, embedder,
  cross-encoder, and provider definitions. See [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
- Example model configurations for Mistral 7B, Llama 3.1, Llama Guard 3, Snowflake Arctic
  Embed L v2.0, and MMARCO MiniLM v2.

> Attribution: Llama 3.1 and Llama Guard 3 — Built with Meta Llama 3.
> Licensed under the Llama 3.1 Community License Agreement.

#### 🗄️ Vector Storage

- ChromaDB persistent storage with HNSW indexing.
- Configurable chunk size, overlap, and neighbor exploration parameters.
- Collection lifecycle management (create, preserve, wipe-and-recreate).

#### 💬 Chat Features

- Multi-turn conversations with per-collection chat context.
- Interactive command-based terminal UI for runtime configuration changes.
- Session history persistence and recall.

#### ⚙️ Configuration System

- Configuration-driven architecture using Python config files.
- CLI overrides for selected configuration keys.
- Environment-variable-based internet access control.

#### 📟 Terminal UI

- ANSI-colored output with width-aware formatting.
- Severity indicators with emoji and ASCII fallbacks.
- Startup banners and configuration summaries.
- End-of-run statistics and interactive collection selection.

#### 🔐 Networking & Security (Operational Characteristics)

- Offline-first execution model once dependencies are installed.
- Optional Python-level socket activity tracing. See [SECURITY.md](SECURITY.md).
- HF_HUB_OFFLINE and related environment flags.

#### 🧪 Testing

- Pytest-based test suite covering algorithms, detection logic, model caching, and
  configuration handling.
- Network-observation test runner with socket tracing.

#### 📦 Deployment & Tooling

- File signature verification script.
- Third-party license inventory and reporting tools.

#### 📖 Documentation

- [README.md](README.md) — Overview and installation guidance.
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and data flow documentation.
- [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md) — Example walkthroughs.
- [LEGAL.md](LEGAL.md) — Legal, privacy, and governance notes.
- [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) — Third-party attribution.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Community conduct guidelines.
- Class and overview diagrams in Documentation/ClassGraphs/.

---

**License:** MIT — Copyright (c) 2026 @HarinezumIgel
