<!-- markdownlint-disable MD024 MD060 -->
# Changelog

All notable changes to RAG-LCC are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Released] — 2026-08-05

### 🛡️ Startup safeguards — drive-root and working-directory checks

- **`DriveRootExecutionError`** (new, `src/Commons/Exceptions.py`) — raised when
  the project root resolves to a drive or filesystem root (`C:\` / `/`).
- **`Helpers.is_in_drive_root()`** (new) — detects drive/filesystem-root installs
  via `os.path.splitdrive` (`len(tail) <= 2`); emits an **UPPERCASE message in
  `BRIGHT_RED`** and raises `DriveRootExecutionError` when `required=True`.
- **`StartupCommons.common_start`** now wires `Helpers().is_in_drive_root(required=True)`
  directly after the existing venv check, blocking all apps on drive-root installs.
- **`StartupCommons._ensure_started_from_project_root`** (new) — verifies
  `cwd == _ABSOLUTE_PATH` at startup; exits with a red `Startup` error message
  if the app is launched from a different directory.

### 🔒 Defense-in-depth — guarded deletions

- **`_cleanup_dir`** (`src/Chat/MarkedDocsViewer.py`) — explicit `len(tail) <= 2`
  drive-root guard applied to **both** `abs_root` and `abs_path`, closing the
  edge-case where `C:` (no trailing slash) could slip past `abs_path == abs_root`.
- **`ArgosDownloader.remove_stanza_models`** — guard tightened to `len(tail) <= 2`
  parity with `FileUtils`; now names the target directory and requires `y`
  confirmation before deletion.
- **`ArgosDownloader.remove_all`** — lists all packages to be uninstalled and
  requires `y` confirmation before proceeding.
- **Tests** — `TestCleanupDir` extended with mocked drive-root cases (`C:\` and
  `/`); `TestRemoveStanzaModels` extended with removal confirmation and cancel cases.

---

## [Released] — 2026-08-04

### 🖍️ PDF source marking — correct page, tables, and speed

- **Source-page confinement** — answer-grounding (orange) snippets now carry the
  source chunk's physical `PageNumber` (`RAGChatImpl._build_grounded_snippets`).
  Previously they were created with `page_number=None`, so the marker scanned the
  whole document and highlighted the **first** page where a fragment appeared —
  marking front-matter / table-of-contents pages instead of the real source page.
- **Token-window fallback** — `PdfVisualMarker._find_rects` gained a third matching
  tier (`_match_token_windows`): consecutive 4-token slices are matched when the
  full sequence and line/sentence fragments do not. This finally highlights
  multi-column tables and numbered legends (no newlines, no sentence punctuation,
  page word order ≠ stored order) that were previously left unmarked.
- **Performance** — `mark_to_bytes` now caches each touched page's
  `extract_words()`, flattened token list, and a **first-token position index**,
  computed once per page and reused across all snippets, fragments, and windows.
  `_match_word_sequence` uses the index to jump to plausible match starts instead
  of scanning every offset. Marking touches only the selected source pages, so a
  realistic multi-chunk answer on a large document no longer re-extracts pages.
- **Tests** — `test_visual_markers.py` extended with token-window fallback cases.

### 🏷️ Document metadata extraction, display & filtering

- **`DocumentMetadataExtractor`** (new singleton, `src/Strategies/Chunkers/`) —
  harvests document-info fields (author, title, subject, creator, producer,
  created/modified dates, keywords) from PDF (pypdf) and modern Office formats
  (docx/pptx/xlsx), with per-concept **synonym** resolution across formats;
  every other file type falls back to generic filesystem fields. A `Pages`
  count is attached where meaningful. Controlled by `_METADATA_EXTRACTION` in
  `Config_Global.py` (`ENABLED`, `DOC_INFO_FIELDS`, `GENERIC_FIELDS`,
  `PDF_PAGE_LABEL_FIELD`, `SHOW_IN_ANSWER`).
- **Answer display** — `Helpers.build_document_metadata_md` appends a
  **Document metadata** section (per file, empty fields skipped) to CLI and
  HTTP answers; source citations prefer the printed page label.
- **Load-time visibility** — `Config_RAGLoad.py` → `SHOW_EXTRACTED_METADATA`
  (default `True`) prints harvested fields per ingested file.
- **Interactive filtering** — new `metadata!` picker (`Gui/MetadataPicker.py`,
  lists only fields that actually have values), `metadata=Field:Value`, and
  `metadata-` commands set a `{field: value}` filter applied as a flat ChromaDB
  `where` condition and honoured by the BM25 and graph retrievers. Active file /
  metadata filters are printed in turquoise before each query.

### 🖨️ PDF printed page-label detection

- **`PdfPageChunker`** now recovers the **printed** page number from each page's
  footer/header text (numeric or Roman) when the PDF's `/PageLabels` metadata
  omits it, storing it in `PageLabel` while the physical index stays in
  `PageNumber`. Detection is language-aware — the "page"/"of" keywords are
  translated for non-English documents via the cached translator (no banlist
  side effects). New knob `_CHUNKERS.PDF_PAGE.DETECT_PRINTED_LABEL` (default
  `True`).

### 🧹 Chunker refactor

- The `@staticmethod` helpers across the chunkers (`HeadingChunker`,
  `SemanticChunker`, `SentenceSplitter`, `SentenceWindowChunker`, `SlideChunker`,
  `SlidingWindowChunker`, `PdfPageChunker`/`PageBasedChunker`) were converted to
  instance methods so they participate in the singleton pattern and can read
  per-document state (e.g. detected language). Tests updated accordingly.

---

### � Changed — Reranking skipped when the whole pool is unconfident

`HomeBrewChunkSelector.filter_threshold` now detects when **no** local chunk's
`sigmoid(raw_rerank_score)` reaches the configured threshold. In that case the
cross-encoder is unconfident about the entire pool (common with mmarco-style
rerankers on technical/tabular content, where the correct chunk can be demoted
to last), so reranking is **skipped**: every chunk is kept and the selector
orders them by retrieval (RRF) score instead of the logit.

- **Rerank-skip fallback** replaces the previous relative-band fallback. When it
  fires an orange `Rerank skipped` message is emitted.
- **`ChunkSelector._get_retrieval_score` / `_rank_key`** (new): retrieval-score
  ordering used by all three selectors (`ScoreRankedSelector`,
  `PerFileCapSelector`, `SingleDocumentSelector`) while rerank is skipped.
- **`_RELATIVE_LOGIT_MARGIN`** and the relative-band branch removed. The
  score-evaluation math (`sigmoid(raw_logit) ≥ threshold`) is unchanged.
- **`Config_RAGChat.py`** — five strategy thresholds recalibrated around the
  neutral logit (sigmoid(0) = 0.50): NARROW `0.70→0.60`,
  BALANCED_FILE_CAP `0.65→0.55`, DEFAULT/WIDE `0.60→0.50`,
  ULTRA_WIDE `0.55→0.45`.
- **Docs** — `CONFIGURATION_REFERENCE.md` threshold description updated.
- **Tests** — `test_chunk_selector.py` and `test_web_retriever.py` updated for
  the rerank-skip behaviour.

## [Released] — 2026-08-03

### �🖥️ Cross-platform license pager & pip bootstrap in setup scripts

- **`Setup.py` / `NLTK_Stopwords_WordNet.py`** — `_page_text` now supports a
  native Windows pager: on `nt` it pipes the license text through the built-in
  `more` command and drains any leftover keystrokes from the console buffer via
  `msvcrt`, falling back to `less` (Unix) and then to the built-in terminal
  pager. The `less` branch no longer requires `os.name != "nt"`.
- **Fallback pager** — page height clamped to `max(5, min(lines − 2, 40))` so
  very tall or very short terminals still paginate sensibly, and the
  `[Enter]/[q]` prompt now catches `EOFError` to exit cleanly on closed stdin.
- **`Setup.py`** — cryptography preamble now bootstraps pip via
  `python -m ensurepip --upgrade` (then upgrades pip) when pip is missing,
  before installing the `cryptography` module used for signature verification;
  aborts with a clear message if bootstrap fails. Execution plan updated with a
  matching preamble line; error output uses `stderr.strip()`.
- **`Setup.py`** — Black formatting reflow of the connectivity `curl` examples
  and the interactive-configuration prompt.

## [Released] — 2026-08-02

### 📝 Documentation overhaul

- **README.md** — full rewrite: new narrative introduction grouped by the four apps
  (DocClassify → RAGLoad → RAGChat → RAGChatService), updated quick mental model
  diagram now includes RAGChatService and shows all pipeline stages including
  compliance pre/post-check; added `## 🔑 Feature Highlights` section organised by
  app; removed legacy verbose sections now covered by CONFIGURATION_REFERENCE.md.
- **CONFIGURATION_REFERENCE.md** (new) — merged `CONFIGURATION.md` into a single
  reference document; adds topic-organised quick-reference tables (what to configure
  and where) as a navigation aid above the per-file deep reference.  `CONFIGURATION.md`
  deleted; all inbound links updated across `ARCHITECTURE.md`, `INSTALL.md`,
  `EXAMPLES.md`, `CHANGELOG.md`.
- **ARCHITECTURE.md** — Component Hierarchy rewritten as a full `src/` directory tree
  with per-file descriptions; seven previously undocumented modules added (`Api/`,
  `VisualMarkers/`, `Commons/`, `Gui/`, `Config/`, `Scripts/`, `AI/`).
- **Broken-link audit** — fixed ~13 broken anchors across six files including
  wrong leading dashes on emoji-prefixed headings, stale section names in
  `ARCHITECTURE.md` and `INSTALL.md`, and links to README sections removed during
  the rewrite.

---

## [Released] — 2026-08-01

### 🔄 Changed — Chunk threshold uses sigmoid-probability space

`HomeBrewChunkSelector.filter_threshold` now compares `sigmoid(raw_rerank_score) >= threshold`
instead of the raw logit directly.  Threshold values are now true relevance
probabilities in `[0, 1]` — `threshold = 0.60` means "60 % confidence minimum".

- **`HomeBrewChunkSelector._sigmoid`** (new): shared sigmoid helper used by
  `filter_threshold` and `_print_final_score`.
- **Relative-band fallback** updated: guard raised from `<= 0.35` (logit) to
  `<= 0.60` (probability); fallback threshold converted via
  `sigmoid(pool_max − margin)`.
- **`Config_RAGChat.py`** — five strategy thresholds converted to probability
  space: NARROW `0.75→0.70`, BALANCED_FILE_CAP `0.55→0.65`,
  DEFAULT/WIDE `0.35→0.60`, ULTRA_WIDE `0.20→0.55`.
- **`Config_WebSearch.py`** — `rerank_threshold` `0.0→0.50`
  (sigmoid(0) = 50 % = neutral, equivalent to the old default).
- **`ChatCompletionHandler`** threshold clamp `[0.0, 1.0]` is now semantically
  correct for probability values.
- **Tests** — `test_chunk_selector.py` and `test_web_retriever.py` updated.

### 🔄 Changed — `RAGChatImpl._retrieve` split into focused helper methods

`_retrieve` was a ~700-line method. It has been refactored into an orchestrator
(~17 lines) that delegates to seven single-responsibility helpers:

| New method | Responsibility |
|---|---|
| `_prepare_session` | Reset per-turn flags; handle mode-change and topic-switch resets |
| `_normalize_query` | Translate → rewrite → re-translate; set `effective_query`; expand alternate queries |
| `_check_gates` | Retrieval eligibility gate + intent classifier gate |
| `_fetch_local_docs` | Vector (+ alternate-query expansion), BM25, and Graph retrieval |
| `_fetch_web_docs` | Web retrieval + BM25/cosine pre-filters |
| `_merge_and_select` | RRF fusion → cap → append web → dedup → rerank → chunk select |
| `_build_context` | Populate grounding fields; format LLM context string |

No behaviour change. All 79 tests pass.

**Affected file:** `src/Chat/RAGChatImpl.py`

### ✨ Added — Rerank debug table shows raw logit + sigmoid side by side

The post-rerank threshold table (`Rerank select`) now displays both the raw
cross-encoder logit and its sigmoid probability in every row:

```text
   Logit [Sigmoid]      Thr     ΔProb         Retrievers  File
✅   3.7515 [0.977]  (0.6000)  +0.377  ...
❌  -2.1767+[0.102]  (0.6000)  -0.498  ...  ← + = boosted logit (single-chunk)
```

- `[0.977]` = `sigmoid(raw_rerank_score)` — the value actually compared against the threshold.
- `ΔProb` = `sigmoid(eff) − threshold` (probability delta; positive = passed).
- Boosted rows show `logit+[sig]` (effective/boosted logit) instead of the
  overflowing `raw→boosted` format.  A note is printed above the table when
  any boosted chunk is present.
- The **Selected** table also gains a `[Sigmoid]` column alongside the
  normalized `rerank_score`.
- Column headers are aligned to account for emoji display width.

**Affected file:** `src/Strategies/HomeBrewChunkSelector.py`

---

## [Unreleased] — 2026-07-31

### ✨ Added — Web source links displayed in CLI after each answer

Web URLs retrieved by the web search path are now printed to the terminal after
the answer, as clickable links (🌐), alongside local document links (📄).
Previously, only the LLM-generated "Sources" section mentioned web URLs — as
plain partial paths that were not terminal-clickable.

- **`Chatter._show_web_sources()`** (new): collects distinct `FilePath` URLs
  from web chunks on `last_chosen_chunks` and prints them in rank order.
- **`Chatter._show_original_sources()`**: no longer returns early when no local
  files are present; always delegates to `_show_web_sources` at the end.
- **`Chatter.run()`**: `chosen` is now resolved before the `mark_text` branch so
  web links are shown in both the `mark_text=True` and `mark_text=False` paths.

### 🐛 Fixed — Reranker score normalisation uses query-relative pool min-max

*Credits: fix suggested by Don Karter (u/donk8r on Reddit).*

Previously, `rerank_score` was computed as `sigmoid(pool_max_raw) × min-max(raw)`.
The `sigmoid` factor tied the absolute value of `rerank_score` to the best raw
logit in the current query's pool — the same chunk scored differently on each
query. Using `rerank_score` for keep/drop threshold decisions therefore produced
non-deterministic results: a document could pass on one run and fail on another
with an identical query, depending on what other candidates were in the pool.

**Fix:** The sigmoid multiplier is removed. `rerank_score` is now pure
min-max over the pool (`(raw − lo) / (hi − lo)`) and used **only** for
intra-query ordering (best chunk → 1.0, always). Keep/drop threshold
decisions use `raw_rerank_score` — the raw cross-encoder logit stored on each
chunk — which is on a query-independent scale. Web chunks are additionally
scaled by `web_weight` for ordering when local results are also present.

**Affected file:** `src/Chat/RAGChatImpl.py` (`_rerank`)

---

## [Released] — 2026-07-29

This is a major release including last 2 months work. Among bug fixes, small improvements
there are:

- Web Search mode (combined with local RAG retrieval or Web Search only)
- Text grounding (marking) in documents
- vllm  supported as LLM provider
- Support for devcontainers
- Install script Setup.py that guides through the installation (Windows + Unix)

For a quick start read [INSTALL.md](INSTALL.md).

## [Unreleased] — 2026-07-29

### 🐛 Fixed — Progress bar output in non-interactive streams

Fixed garbled progress bar output when running in non-TTY environments (log files,
CI pipelines, captured streams). The `Embeddings` progress bar and subsequent
`KeyWrdChk Summary` output would collide into a single corrupted line because
carriage-return (`\r`) in-place updates don't work correctly in non-interactive
streams.

**Changes:**

- **`Helpers.show_progress()`** now detects TTY status using `sys.stdout.isatty()`
  - Interactive terminals: preserve the existing in-place progress bar behavior using `\r`
  - Non-interactive streams: emit one full line per update with `\n` to avoid concatenation
- Added bounds checking for `total` and `processed` parameters to prevent divide-by-zero
- Updated terminating newline to be conditional on TTY status

**Affected file:** `src/Helpers/Helpers.py`

---

## [Unreleased] — 2026-07-18

### ✨ Added — HF Transfer (xet protocol) support

- Added `xet` to managed packages in deployment script (`deploy.py`)
- Enabled `HF_XET_HIGH_PERFORMANCE` environment variable for faster HuggingFace downloads (replacing deprecated `HF_HUB_ENABLE_HF_TRANSFER`)
- Added documentation to `Config_Internet_Env.py` and `INSTALL.md`

### 🔄 Changed — `PerfLogger` refactored: singleton instance pattern, no threading

`PerfLogger` was refactored to eliminate a race condition where the log
filename was derived from `sys.argv[0]` instead of `_FRIENDLY_NAME` because
the first `PerfLogger().log()` call arrived before `Config` had finished
loading.

**Root cause:** `_build_log_filename()` was called lazily inside a
double-checked lock on first write. A background thread calling `log()` before
the main thread finished setting `_FRIENDLY_NAME` in `Config` caused the
fallback to `sys.argv[0]`, and the wrong filename was then locked in for the
process lifetime.

**Fix:** Callers now own the singleton lifetime. Each class that uses the
logger adds `self.perf_logger: PerfLogger = PerfLogger()` to its `__init__`
**after** `Config` (and therefore `_FRIENDLY_NAME`) is fully initialised.
All subsequent calls use `self.perf_logger.log(...)`. Because construction
is now single-threaded by design, all locking (`_file_lock`, `_times_lock`,
double-checked locking) was removed.

- **`PerfLogger`:** removed `threading` import and both `Lock` instances;
  simplified `_get_file_logger()` and `log()`.
- **All 13 call-site files** updated to the `self.perf_logger` pattern:
  `LLMCaller`, `ModelsCache`, `BM25Scorer`, `CosineScorer`, `JaccardScorer`,
  `KeyBertScorer`, `RegexScorer`, `LevenshteinScorer`, `BM25Retriever`,
  `DocumentIngestionStrategy`, `GraphRetriever`, `WebRetriever`, `RAGChatImpl`.
- **`ScorerBase`** (`ComplianceAlgoResult.py`) gained an `__init__` that sets
  `self.perf_logger` so the shared `verify()` timing wrapper works for any
  subclass.
- **Test factory functions** in `test_bm25_retriever.py`,
  `test_models_cache.py`, `test_graph_retriever.py`, and `test_llm_caller.py`
  updated to set `perf_logger = MagicMock()` after bypassing `__init__`.

### ✨ Added — `PerfLogger` elapsed time as a dedicated log column

The `Δ=...s` elapsed value is now written as its own fourth pipe-separated
column instead of being appended to the detail string. A new `_DETAIL_WIDTH`
constant (65 chars) pads the detail column so the elapsed column aligns
across all log lines.

Previous format:

```text
2026-07-13T14:22:05.145Z | BM25Retriever.query              | stop  bm25 query n=42 elapsed=0.145s  Δ=0.145s
```

New format:

```text
2026-07-13T14:22:05.123Z | BM25Retriever.query              | start bm25 query q='cats'                         |
2026-07-13T14:22:05.145Z | BM25Retriever.query              | stop  bm25 query n=42 elapsed=0.145s              | Δ=0.145s
```

- **Affected file:** `src/Helpers/PerfLogger.py`

---

## [Unreleased] — 2026-07-15

### ✨ Added — Early startup connectivity probe for Ollama / vLLM

`StartupCommons.common_start()` now probes the configured LLM endpoint immediately
after printing the endpoint info banner, before any model or pipeline is loaded.
If the endpoint is unreachable the application prints a red error and raises
`LocalLLMEndpointNotAvailable` (a new subclass of `BackendUnavailableError`), then
exits cleanly.

Probe behaviour is controlled by `TRY_FALLBACK_URLS` in each endpoint's config block
(`Config_Models.py`):

| Value | Behaviour |
| --- | --- |
| `True` *(default)* | Uses the existing `Helpers.find_provider_url()` fallback logic — tries up to 6 candidate URLs (configured host, localhost, 127.0.0.1, host.docker.internal, plus default-port variants). |
| `False` | Probes only `BASE_URL` once. On failure the error is shown immediately and the app exits — no fallback attempts. Use this when `BASE_URL` is a fixed remote IP and fallback probing is undesirable. |

Each failed probe in the fallback sequence now also emits an orange *"Trying next: \<url\>"*
warning so the operator can see which candidates are being tried.

- **New exception:** `LocalLLMEndpointNotAvailable` in `src/Commons/Exceptions.py`
- **New config key:** `TRY_FALLBACK_URLS` in `_MODELS["ollama"]["_OLLAMA"]` and `_MODELS["vllm"]["_VLLM"]` in `src/Configuration/Config_Models.py`
- **Affected files:** `src/Commons/StartupCommons.py`, `src/Commons/Exceptions.py`, `src/Helpers/Helpers.py`, `src/Configuration/Config_Models.py`

---

## [Unreleased] — 2026-07-12

### 🐛 Fixed — `.vscode/settings.json` wrong source path and missing site-packages

`python.analysis.extraPaths` referenced `"${workspaceFolder}/source"` (non-existent)
instead of `"${workspaceFolder}/src"`, meaning Pylance never indexed project
modules. The same typo existed in `python.autoComplete.extraPaths`.

Additionally, Pylance could not resolve installed packages (`fastapi`, `uvicorn`,
`starlette`) despite them being present in `.venv`. Added
`"${workspaceFolder}/.venv/lib/python3.13/site-packages"` to
`python.analysis.extraPaths` to force resolution.

- **Affected file:** `.vscode/settings.json`

### 🐛 Fixed — `RAGChatService.py` duplicate `import sys` and Pylance false positives

- Removed duplicate `import sys` (appeared on lines 3 and 8).
- Moved `# pyright: ignore[reportUnusedFunction]` from the function body line
  to the `def` line for `_verifyBearerToken`, `_validationErrorHandler`,
  `chatCompletions`, and `getMarkedDocument` (FastAPI decorator-registered
  handlers that Pylance incorrectly reports as unused).
- Added `reportPrivateUsage` suppression on the import line for
  `_complianceResponse` and `_format_session_for_error`.

- **Affected file:** `src/Apps/RAGChatService.py`

---

## [Unreleased] — 2026-07-12

### 🐛 Fixed — RAGChatService startup banner printed twice

`RAGChatService.py` called `StartupCommons.common_start()` twice: once at
module level (line 38) and again inside `RAGChatService.__init__()`. This
caused the banner, environment-variable summary, and all startup checks to
be printed twice every time the service was started.

`__init__` now reuses the module-level `ctx` instance instead of creating a
second one, so the banner appears exactly once.

- **Affected file:** `src/Apps/RAGChatService.py`

### 🐛 Fixed — `RAGChatService.run()` fallback host defaulted to loopback

The `run()` method had a hard-coded fallback `"127.0.0.1"` for
`RAG_CHAT_SERVICE_LISTENER`. Under Docker this fallback would have prevented
port-forwarding from reaching the service. Changed to `"0.0.0.0"` to match
the value set in `Config_RAGChatService.py`.

- **Affected file:** `src/Apps/RAGChatService.py`

### ➕ Added — `ddgs` package to requirements and deploy module fixups

The DuckDuckGo search backend (`src/Strategies/WebRetriever.py`) already
supported both `ddgs` (preferred) and the legacy `duckduckgo_search` package
via a try/except import chain, but neither package was listed in
`requirements_final.txt` or in the deploy module fixup map.

- Added `ddgs>=0.1.0` to `requirements/requirements_final.txt`.
- Added `"ddgs": "ddgs"` and `"duckduckgo_search": "ddgs"` mappings to
  `MODULE_FIXUPS` in `deploy/scripts/install_required.py` so the deploy
  scanner correctly translates both import names to the `ddgs` install target.
- Updated `requirements/requirements_final.txt.sha256`.

- **Affected files:** `requirements/requirements_final.txt`,
  `requirements/requirements_final.txt.sha256`,
  `deploy/scripts/install_required.py`

### 🔄 Changed — `HF_HUB_DISABLE_PROGRESS_BARS` default changed to `"0"`

Progress bars are now shown by default during Hugging Face model downloads.
Set to `"1"` in `Config_Internet_Env.py` to suppress them.

- **Affected file:** `src/Configuration/Config_Internet_Env.py`

### 🐛 Fixed — `FakePooling` mock missing `pooling_mode` parameter

`ModelsCache.load_quantized_model()` calls `models.Pooling(pooling_mode="mean", …)`
but `FakePooling.__init__()` in the test did not declare a `pooling_mode`
parameter, causing `TypeError` on every `TestLoadQuantizedModel` test.
Added the missing parameter.

- **Affected file:** `tests/test_models_cache.py`

### 📝 Documentation — CONFIGURATION_REFERENCE.md audit against config files

- **`RAG_CHAT_SERVICE_LISTENER` default**: Corrected from `127.0.0.1` to `0.0.0.0`
  to match `Config_RAGChatService.py`. Added note that `0.0.0.0` is required
  for Docker port forwarding.
- **`HF_HUB_DISABLE_PROGRESS_BARS` default**: Corrected from `"1"` to `"0"`
  to match updated `Config_Internet_Env.py`.

### 📝 Documentation — CONFIGURATION_REFERENCE.md full config-slot audit (2026-07-12)

Second-pass two-way audit comparing every key in all `Configuration/Config_*.py`
files against `CONFIGURATION_REFERENCE.md` section 8.

**Corrected defaults (safe/shipping defaults restored in `Config_Internet_Env.py`):**

- **`NLTK_STOPWORDS_DOWNLOAD`**: corrected to `"0"` — download disabled by default.
- **`HF_HUB_OFFLINE`**: corrected to `"1"` — Hub access offline by default (safe).
- **`ARGOS_STANZA_DOWNLOAD`**: corrected to `"0"` — package download disabled by default.
- **`WEB_SEARCH_MODE`** (both the admin-knobs table and the section 8 table): corrected to `"0"` — web search disabled by default.
- **`SERVE_OPENWEBUI_CHAT`**: corrected to `"0"` — service disabled by default.
- **`SERVE_IN_MEMORY_DOCS_HTTP`**: corrected to `"0"` — in-memory docs store disabled by default.

**Added missing entries to the Config_Internet_Env table:**

- `WEB_SEARCH_MODE` — master web-search switch, default `"1"`.
- `TESSERACT_PATH` — OS-aware Tesseract path (was only in a top-level quick-ref
  table; now also in section 8).
- `SERVE_IN_MEMORY_DOCS_HTTP` — gate for the in-memory document HTTP server,
  default `"1"`.
- `TOKENIZERS_PARALLELISM` — prevents HuggingFace tokenizer parallelism
  warnings, set via `setdefault` to `"false"`.

**Added missing key to Config_RAGChat section:**

- `SINGLE_CHUNK_SCORE_BOOST` (`1.25`) — reranker score multiplier applied
  when only one source file contributes to the result set.

**File structure updated:**

- `src/VisualMarkers/` added to project structure in `EXAMPLES.md` and to the
  Source Tree in `ARCHITECTURE.md` (was omitted).
- `src/Api/` expanded with `MarkedDocsService.py` and `MarkedDocsStore.py`.
- `src/Scripts/` expanded from 1 entry to all 10 scripts.

---

## [Unreleased] — 2026-06-30

### 📝 Documentation — Configuration reference corrections

Comprehensive documentation audit identified and corrected multiple discrepancies
between the markdown documentation and actual configuration file values.

**CONFIGURATION_REFERENCE.md corrections:**

- **`_ACTIVE_ENDPOINT` default**: Corrected from `"ollama"` to `"vllm"` to match
  `Config_Models.py` actual default (`_ACTIVE_ENDPOINT = "vllm"`).

- **Ollama `BASE_URL`**: Updated from `http://127.0.0.1:11434/api/generate` to
  `http://localhost:11434/api/generate` to match `Config_Models.py` definition.

- **vLLM `BASE_URL` port**: Corrected from port `8000` to port `4000`
  (`http://192.168.100.50:4000/v1/chat/completions`) to match `Config_Models.py`.

- **`HF_HUB_OFFLINE` default**: Updated from `"1"` to `"0"` to match
  `Config_Internet_Env.py` actual default (`os.environ["HF_HUB_OFFLINE"] = "0"`).
  Added clarification that `"0"` allows model downloads.

- **RAGChat consensus rules**: Expanded table to distinguish between RAGChat's
  two separate compliance pipelines:
  - **PROMPT_CHECK** (validates user prompts): `2/3` thresholds for responsiveness
  - **PIPELINE_CHECK** (validates retrieved content): `4/4` thresholds for strictness

  Previous documentation conflated these two pipelines into a single `2/3` entry.
  Section 2b consensus table and section 6 consensus rules both updated for clarity.

- **`TERMINAL_LINE_SIZE` default**: Clarified that `Config_Global.py` sets a simple
  integer default of `120`, which `Config_RAGChat.py` overrides with a dict
  `{"debug": 180, "no_debug": 100}` resolved at runtime by debug level.

**ARCHITECTURE.md corrections:**

- **`_ACTIVE_ENDPOINT` example code**: Updated from `"ollama"` to `"vllm"` in
  model selector code block to match repository default.

- **`terminal_line_size` description**: Corrected `Config_Global.py` default from
  `100` to `120`.

**INSTALL.md corrections:**

- **`HF_HUB_OFFLINE` default**: Updated from `"1"` to `"0"` with clarified
  description matching `Config_Internet_Env.py` and `CONFIGURATION_REFERENCE.md`.

All corrections ensure documentation accurately reflects actual configuration
file defaults and behavior as of this date.

---

## [Unreleased] — 2026-06-25

### 🔄 Changed — Per-model token budget configuration

Token budget parameters (`TOKEN_BUDGET_CONTEXT_CAP`, `TOKEN_BUDGET_RESERVED_OUTPUT`,
`TOKEN_BUDGET_RESERVED_SYSTEM`) are now specified individually for each model role
in `Config_Models.py`.

Previously, these values were read only from `_ACTIVE_LLM` during `TokenBudget`
initialization and applied uniformly to all models including compliance-check
models (`_LLM_CHK`). This caused llama-guard3 to receive the default 2048/1024
token reserves instead of its configured 64/64 values.

**Updated components:**

- `TokenBudget.compute_dynamic_max_tokens` now accepts an optional `model_role`
  parameter. When provided, the method reads `TOKEN_BUDGET_RESERVED_OUTPUT` and
  `TOKEN_BUDGET_RESERVED_SYSTEM` from that specific role's configuration.
- `LLMCaller.call_llm` and `LLMCaller._resolve_token_budget` now accept and
  thread `model_role` through the call chain.
- `AIHelpers.check_prompt_with_llm_guard` and `ClassifyStrategy` compliance
  checks now pass `model_role="_ACTIVE_LLM_CHK"` to ensure guard models use
  their own budget values.

**Configuration changes:**

All model roles in `Config_Models.py` now explicitly declare token budget
settings:

- Main LLMs (`mistral._LLM`, `llama._LLM`): 32768 / 2048 / 1024
- Rewrite-prompt models (`mistral._LLM_REWRITE_PROMPT`, `llama._LLM_REWRITE_PROMPT`):
  32768 / 2048 / 1024
- Check models (`mistral._LLM_CHK`, `llama._LLM_CHK`): 32768 / 2048 / 1024
- Guard model (`llama_guard._LLM_CHK`): 32768 / 64 / 64 (optimized for short
  guard responses)

This ensures each model operates within its intended budget regardless of which
role is currently active.

---

## [Unreleased] — 2026-06-19

### ➕ Added — vLLM endpoint support

A second LLM backend adapter (`VllmBackendAdapter`) is now available alongside
the existing Ollama adapter.  Set `_ACTIVE_ENDPOINT = "vllm"` in
`Config_Models.py` to route all inference calls through an OpenAI-compatible
vLLM (or LiteLLM proxy) endpoint.

Each model entry in `Config_Models.py` carries a `MODEL_VLLM` key (short
alias, e.g. `"mistral_7b"`) in addition to the existing `MODEL_OLLAMA` key.
`Helpers.get_model_args` selects the correct name at runtime based on the
active endpoint.

### 🔄 Changed — Install guide adds Docker-first path and host venv fallback

`INSTALL.md` now introduces deployment choices immediately after `git clone`:

- **Docker deployment (supported)** with explicit `docker build` / `docker run`
  commands.
- **Host virtual-environment deployment** as a documented fallback when GPU
  passthrough is not available in containers.
- Both paths now explicitly point to either running `src/Scripts/Setup.py`
  or following the manual step-by-step installation flow.

### 🔄 Changed — Setup disclaimer added at script header

`src/Scripts/Setup.py` now starts with an explicit package-manager and licensing
disclaimer clarifying that package licensing obligations remain with the original
distributors and the user environment.

### 🔄 Changed — Consent metadata records now include legal disclaimer text

Consent JSON writers were updated to include a standard disclaimer field in
recorded metadata:

- `src/Compliance/Compliance.py`
- `src/Compliance/HFDownloader.py`
- `src/Compliance/ArgosDownloader.py`

### 🐛 Fixed — Legacy HF consent metadata could retain null acceptance identity

`src/Compliance/HFDownloader.py` now backfills missing identity fields during the
"already cached / matching metadata" fast path, so legacy records with
`accepted_by: null` are automatically repaired on next touch.

### 🐛 Fixed — Identity capture hardened against empty values

Identity-capture helpers now normalize empty/whitespace values to non-empty
fallbacks (`unknown-user`, `os`, `unknown-host`) so consent records do not
persist blank identity fields:

- `src/Compliance/SharedHelpers.py`
- `src/Scripts/Setup.py`
- `src/Scripts/NLTK_Stopwords_WordNet.py`

### 🔄 Changed — Configuration docs synchronized with config files (two-way audit)

Markdown configuration references were re-audited against `src/Configuration/*.py`
with config files treated as the source of truth.

- Corrected web-search defaults in docs to match `Config_WebSearch.py`:
  `_WEB_SEARCH_MODE = "on"` and `_WEB_SEARCH.max_results = 10`.
- Corrected selector names in docs to canonical active selectors:
  `_ACTIVE_CROSS`, `_ACTIVE_LLM`, `_ACTIVE_OPENWEBUI`.
- Updated endpoint defaults in docs to match `Config_Models.py`:
  Ollama `BASE_URL = "http://127.0.0.1:11434/api/generate"`,
  vLLM `BASE_URL = "http://192.168.100.50:8000/v1/chat/completions"`.

This change updates documentation only; runtime behavior is unchanged by this
entry.

### ➕ Added — Document markup: relevant sources in yellow, effective matches in orange

Retrieved chunks that contribute to an answer are highlighted in **yellow**,
while effective/grounded matches are highlighted in **orange**. This makes it
immediately visible which parts of the source material were used and which
matches were triggered.

### ➕ Added — Dev container configuration

The repository now ships a `.devcontainer/` configuration.  Opening the project
in VS Code (or GitHub Codespaces) automatically builds a container with all
Python dependencies, the correct runtime, and the recommended extensions
pre-installed — no manual environment setup required.

### 🐛 Fixed — vLLM context-length metadata lookup falling back to config cap

`VllmBackendAdapter.get_context_limit` previously did an exact `id` comparison
against `/v1/models`, which failed whenever:

- the vLLM server reported the model under its full HuggingFace path (e.g.
  `mistralai/Mistral-7B-v0.1`) instead of the short config alias (`mistral_7b`);
- a **LiteLLM proxy** was in use — its `/v1/models` listing omits
  `max_model_len` entirely.

The lookup now proceeds through four passes before falling back to the config
cap:

1. `GET /v1/models/{model_name}` → `max_model_len` (plain vLLM, exact name)
2. `GET /v1/models` list — exact `id` match → `max_model_len`
3. `GET /v1/models` list — **normalised match**: strips HuggingFace org prefix
   and common version/variant suffixes (`-v0.1`, `-Instruct`, `-Chat`, …) then
   collapses punctuation, so `mistralai/Mistral-7B-v0.1` → `mistral7b` matches
   config alias `mistral_7b`
4. **LiteLLM proxy** `GET /model/info` → `data[*].model_info.context_length`,
   matched by `model_name` (exact or normalised)
5. Single-model safety net: when only one model is registered and no name match
   was found, that model's `max_model_len` is used.

A log message is emitted before the `/model/info` retry so the fallback chain
is visible in the output.

### 🐛 Fixed — Deploy-module tests not collected

`tests/conftest.py` only added `src/` to `sys.path`.  Tests that import
`deploy.scripts.*` raised `ModuleNotFoundError` at collection time, preventing
the entire suite from running.  The repository root is now also inserted into
`sys.path`.

### 🐛 Fixed — `test_token_budget_message_uses_active_backend_name` assertion mismatch

The test checked for the substring `"context limit"` (with a space) but the
message emitted by `TokenBudget._load_context_limit` has always used
`"context-length"` (hyphenated).  The assertion was corrected to match the
actual message text.

---

## [Unreleased] — 2026-06-04

### 🐛 Fixed — Cross-encoder reranker receiving malformed input pairs

`RAGChatImpl._rerank` was building cross-encoder input as
`[(combined_text, ""), ...]` — a single concatenated string in the first
element and an empty string in the second.  The mmarco MiniLM cross-encoder
expects two separate segments `(query, chunk)` fed as a proper pair.  Scores
were therefore meaningless.

The call now passes `(query_text, chunk_text)` where `query_text` optionally
prepends a per-model `QUERY_INSTRUCTION` prefix (new config key, see below).

### 🔄 Changed — `QUERY_INSTRUCTION` config key added to cross-encoder model entries

A new optional `QUERY_INSTRUCTION` key is supported in `_CROSS` model config
blocks.  When non-empty, its value is prepended to the query string before the
cross-encoder sees it — useful for instruction-tuned rerankers that expect a
task prefix.  The mmarco entry ships with an empty string (no prefix needed).

### 🐛 Fixed — Operating-environment chunk never retrieved for ThinkStation P620

`PDF_PAGE` chunker `MAX_CHUNK_SIZE` was 400 words.  Dense pages caused the
chunker to merge the static-electricity safety section (~300 words) with the
operating-environment data (temperatures, altitude) into a single 400-word
blob.  The chunk embedding was dominated by static-electricity vocabulary, so
queries about operating temperatures never retrieved it.

`MAX_CHUNK_SIZE` reduced to 200 words, splitting the blob into focused chunks.
Existing collections must be reloaded (`RAGLoad.py --collection …`) to benefit.

### 🐛 Fixed — All-negative cross-encoder pool rejecting every chunk

`HomeBrewChunkSelector.filter_threshold` used an absolute rerank threshold
(default 0.35) computed via `score = min_max_normalised × sigmoid(pool_best)`.
When the cross-encoder gives all-negative raw logits (common with the mmarco
MiniLM model on technical content), `sigmoid(pool_best)` is below 0.35 and
every chunk is rejected — even the correct answer ranked first.

A **relative-band fallback** is now applied when `pool_max_local < threshold`:
accept chunks whose score is within 75 % of the pool's best local score
(`effective_threshold = pool_max × 0.75`).  Web chunks are excluded from the
pool-max calculation.  A diagnostic message is emitted at debug level ≥ 10
when the fallback activates.

### 🔄 Changed — Multi-query expansion prompt revised for domain vocabulary

`_PROMPT_QUERY_EXPAND` in `Config_RAGChat.py` was rewritten to encourage
retrieval diversity through synonyms and domain-specific terminology rather than
simple paraphrasing.  The prompt now explicitly instructs the model to prefer
terms a technical document would contain (e.g. "thermal conditions" → also try
"operating temperature", "temperature range", "environmental specifications").

### 🔄 Changed — Full chunk content logged at debug level 32

`HomeBrewChunkSelector._print_final_score` now emits the complete
`page_content` of every chunk (hit and miss) when `DEBUG_LEVEL ≥ 32`.

---

## [Released] — 2026-06-03

### 🐛 Fixed — LLM answer block invisible / grounded spans rendering as bright cyan on non-truecolor terminals

Two related display problems in the CLI answer block:

#### 1. Gray background not shown on plain PowerShell / cmd / older terminals

`ANSWER_BG` and `ANSWER_FG` used 24-bit truecolor ANSI sequences
(`\033[48;2;…m`).  Terminals that don't advertise `COLORTERM=truecolor` or
`COLORTERM=24bit` silently discard those codes, so the entire gray block never
appeared.

`src/Gui/Colors.py` now detects truecolor support at import time:

```python
if _supports_truecolor():          # COLORTERM=truecolor/24bit or WT_SESSION
    ANSWER_BG = "\033[48;2;45;45;45m"
    ANSWER_FG = "\033[38;2;220;220;220m"
else:
    ANSWER_BG = "\033[48;5;238m"   # 256-color dark-gray fallback
    ANSWER_FG = "\033[38;5;252m"   # 256-color light-gray fallback
```

#### 2. Grounded-sentence spans cancelling the gray background mid-line

`ground_answer_cli` closes each highlighted span with a bare `\033[0m`.  When
`print_llm_answer` printed a wrapped line that contained such a span, the reset
wiped `ANSWER_BG`/`ANSWER_FG` for the rest of that line, causing the remaining
text to appear unstyled (or cyan, from the grounding highlight leaking).

`print_llm_answer` in `src/Chat/Chatter.py` now replaces every inner `\033[0m`
with `\033[0m{style}` before padding, so the outer background is restored
immediately after each span closes.

### 🔄 Changed — Marked-sources temp files written lazily (picker path)

Previously all highlighted documents were written to a temp directory before
the user was shown the picker.  Now only the file the user actually selects is
written to disk; the rest remain in RAM.  If the user presses Enter to skip, no
temp directory is created at all.

The OSC 8 hyperlink path (Windows Terminal, VS Code, iTerm2, …) is unaffected —
all files are still written up-front there because the links must be valid before
the user clicks them.

### 🔄 Changed — Removed misleading "saved as .txt.md" message

The informational note that announced `.txt` sources being "saved as .txt.md for
Markdown highlighting" has been removed.  The `.txt.md` rename is an internal
detail of the temp-file mechanism (so the OS opens the file in a Markdown
viewer); it does not represent a permanent save, and the message implied
otherwise.

---

## [Released] — 2026-05-28

### 🐛 Fixed — `RAGLoad.py` classify-CSV path resolution when CSV lives in `logs/DocClassify/`

When a DocClassify CSV produced by the classify-then-load workflow was passed to
`RAGLoad` via `--classify-csv`, the path was constructed by joining the RAGLoad
log directory (`logs/RAGLoad/`) with the bare filename, causing a *file not found*
error for CSVs that now reside in `logs/DocClassify/`.

The path is now resolved correctly: if the argument is already an absolute path it
is used as-is; relative paths are resolved from the project root, not from the
RAGLoad log directory.

### 📖 Docs — animated demo GIF added to README

`Documentation/Pics/RAG-LCC-Screenshots.gif` is now embedded in README.md in a
dedicated `## 🎬 Demo` section, directly visible on the GitHub landing page.

---

## [Released] — 2026-05-27

### 🔄 Changed — `web_search` session parameter renamed `"off"` → `"local_only"`

The per-session web-search tri-state now uses the name `"local_only"` instead
of the old `"off"` for the value that skips web retrieval.  All three valid
values are now:

| Value | Behaviour |
| --- | --- |
| `local_only` | Web retrieval disabled for this session (was `"off"`) |
| `local_and_web` | Both local ChromaDB and web retrieval |
| `web_only` | Web retrieval only (no local docs) |

The boolean shorthand (`True` → `"local_and_web"`, `False` → `"local_only"`)
still works for backwards compatibility.

- **`src/Chat/QueryParts.py`** — prompt text, picker choice labels, `print_values()`,
  `_show()`, `_read_and_apply_value()` validation tuple and error message (8 locations).
- **`src/Api/ChatCompletionHandler.py`** — bool-normalisation comment, two response
  formatting branches (4 locations).

> **Not changed:** the admin-level `_WEB_SEARCH_MODE = "off"/"dry_run"/"on"` in
> `Config_WebSearch.py` retains its original values and is a separate concept.

### 📖 Docs — README and INSTALL updated for renamed `web_search` values

- **`README.md`** — status-line example, per-session switches table, and inline
  code blocks updated to `local_only` / `local_and_web` / `web_only`.
  Added *Lessons Learned Building an Experimental RAG Lab* Reddit write-up to the
  community write-ups section.
- **`INSTALL.md`** — step 5 quick-start example updated to use new value names;
  `web_only` mode mentioned; bool shorthand noted.

### ✨ Added — `tests/test_web_prefilter.py` (40 tests)

New test file covering `src/Strategies/WebPreFilter.py`:

- **Pure helpers** — `_idf()`, `_bm25_score()`, `_cosine()` edge-case coverage
  (empty corpus, zero-length vectors, perfect match).
- **`WebPreFilter.bm25_prefilter()`** — 10 tests: passthrough when disabled,
  threshold filtering, all-pass, all-fail, tie-breaking, configurable `k1`/`b`.
- **`WebPreFilter.cosine_prefilter()`** — 11 tests: passthrough when disabled,
  threshold filtering, empty input, low-similarity rejection.
- Uses `StubConfig` (with `_WEB_SEARCH.bm25_pre_filter`,
  `_WEB_SEARCH.cosine_pre_filter`, `_BM25_INDEX.k1/b`) and `StubPretty`
  (captures messages).

### 🐛 Fixed — `tests/test_web_retriever.py` updated for current production behaviour (8 tests)

Eight tests that had drifted from production code were updated:

**Dry-run path removed from `WebRetriever`** (4 tests):
`_WEB_SEARCH_MODE="dry_run"` is now enforced upstream (by `QueryParts` and
`RAGChatImpl`) before `WebRetriever` is ever called.  The retriever itself no
longer checks this flag and no longer returns a synthetic single "DRY RUN"
notice document.

- `TestQueryDryRun` — replaced `test_dry_run_returns_single_notice_document`
  and `test_dry_run_does_not_call_ddgs` with
  `test_dry_run_mode_setting_has_no_effect_on_retriever`, which confirms the
  retriever executes normally regardless of `_WEB_SEARCH_MODE`.
- `TestAuditLog` — renamed `test_dry_run_pass_writes_log` →
  `test_dry_run_mode_logs_executed_status` (asserts `EXECUTED`); renamed
  `test_dry_run_blocked_writes_dry_run_blocked_status` →
  `test_blocked_query_in_dry_run_mode_logs_blocked` (asserts `BLOCKED`, not
  `DRY_RUN_BLOCKED`).

**Separate `web_rerank_threshold` for web docs** (4 tests):
`HomeBrewChunkSelector.filter_threshold()` applies `web_rerank_threshold`
(default `0.0`) to web-sourced chunks and `threshold` to local chunks.  Tests
that assumed web docs shared the local threshold were corrected.

- `TestFilterThreshold` — `test_web_below_threshold_filtered` →
  `test_web_doc_bypasses_local_threshold_by_default` (web doc with score 0.05
  passes against default `web_rerank_threshold=0.0`);
  `test_web_zero_score_filtered` → `test_web_zero_score_passes_with_default_web_threshold`;
  `test_mixed_pool_correct_filtering` updated (low-score web doc now passes);
  `test_all_below_threshold_nothing_survives` →
  `test_local_below_threshold_filtered_web_passes`.
  New test `test_web_doc_filtered_by_explicit_web_threshold` verifies
  filtering works when `web_rerank_threshold` is explicitly set on the session.

- **Affected files:** `tests/test_web_retriever.py`, `tests/test_web_prefilter.py` (new),
  `src/Chat/QueryParts.py`, `src/Api/ChatCompletionHandler.py`,
  `README.md`, `INSTALL.md`.

---

## [Released] — 2026-05-26

### ✨ Added — Multi-query expansion and chunk near-duplicate removal

Two precision/recall improvements added to `RAGChatImpl._retrieve()`.

#### Multi-query expansion (`_generate_alternate_queries`)

Before the retrieval loop, the final (resolved, translated) query is
expanded into up to `_MULTI_QUERY.num_variants` (default 3) alternate
phrasings by the rewrite LLM:

- **Config** — `_MULTI_QUERY` dict in `Config_RAGChat.py`: `enabled`,
  `num_variants`, and a dedicated `LLM_PARAM` block (`temperature`,
  `top_k`, `top_p`, `num_predict`, `use_ollama_gpu`).
- **Prompt** — `_PROMPT_QUERY_EXPAND` template added to `Config_RAGChat.py`;
  registered as `PROMPT_QUERY_EXPAND` in both `mistral` and `llama` model
  entries in `Config_Models.py`.
- **Retrieval** — each alternate query runs an additional
  `similarity_search_with_score()` call; new chunks (not already in the
  base hit set, identified by chunk `id` metadata) are appended with
  `retriever_sources = "Vector-AQ{n}"` so they remain traceable in debug
  tables.  The existing RRF merge and rerank steps are unaffected.
- **Debug** — alternate queries logged at `debug_level >= 45` under the
  `MultiQuery` tag in cyan.
- **Errors** — LLM failures and JSON parse errors are logged as warnings
  and return an empty list; retrieval proceeds normally.

#### Chunk near-duplicate removal (`_remove_similar_chunks`)

After RRF fusion and before reranking, near-duplicate chunks are removed
using Jaccard similarity on lowercased word tokens:

- **Config** — `_CHUNK_DEDUP` dict in `Config_RAGChat.py`: `enabled`,
  `threshold` (default `0.85`).
- **Algorithm** — iterates chunks in ranked order (highest rank first);
  a candidate is dropped if its Jaccard similarity against any already-kept
  chunk meets `threshold`.  Uses the existing `SharedHelpers.jaccard()`
  static method.
- **Debug** — number of dropped chunks logged at `debug_level >= 45` under
  the `ChunkDedup` tag in cyan.

#### Tests

`tests/test_multi_query_and_dedup.py` — 23 tests (9 for
`_remove_similar_chunks`, 14 for `_generate_alternate_queries`) using the
source-extraction + `exec()` pattern to avoid heavy transitive imports.

- **Affected files:** `src/Chat/RAGChatImpl.py`,
  `src/Configuration/Config_RAGChat.py`,
  `src/Configuration/Config_Models.py`,
  `tests/test_multi_query_and_dedup.py`.

---

## [Released] — 2026-05-21

### ✨ Added — `PageBasedChunker` abstract base + `PdfPageChunker`

Introduced a shared intermediate abstract class `PageBasedChunker` that
captures the page/slide chunking pattern common to `SlideChunker` (PPTX)
and the new `PdfPageChunker` (PDF):

- **`PageBasedChunker`** (`Strategies/Chunkers/PageBasedChunker.py`) —
  abstract base extending `ChunkerStrategy`. Provides the full `chunk()`
  template method, `_pages_to_texts()`, `_split_oversized()`, and
  `_to_docs()`.  Concrete subclasses only need to implement `_parse_pages()`
  (and optionally `_format_prefix()` / `_extra_meta_for_page()`).
  Exports the public type alias `PageData = tuple[int, str, str]`
  (page_number, title, body).

- **`PdfPageChunker`** (`Strategies/Chunkers/PdfPageChunker.py`, strategy
  key `PDF_PAGE`) — re-reads PDF files via *pypdf* to recover hard page
  boundaries lost during flat text extraction.  Each PDF page becomes one
  chunk prefixed `"Page N"`.  Pages whose word count exceeds `MAX_CHUNK_SIZE`
  are split with `RecursiveCharacterTextSplitter`.  An integer `PageNumber`
  field is added to every chunk's metadata.  Falls back to treating the
  pre-extracted text as a single chunk for non-PDF types or inaccessible
  files.

- **`SlideChunker` refactored** — now extends `PageBasedChunker` instead of
  `ChunkerStrategy`.  Reduced from 207 to ~100 lines; retains only
  `_parse_pages()`, `_format_prefix()`, and `_parse_pptx()`.

- **`DocumentIngestionStrategy`** — imports `PdfPageChunker`; `_make_chunker()`
  handles the `"PDF_PAGE"` key.

- **`Config_Global.py`** — `DETAILED` profile maps `"pdf"` to `"PDF_PAGE"`;
  `_CHUNKERS` entry `"PDF_PAGE": {"MAX_CHUNK_SIZE": 400, "PRESERVE_NEWLINES": False}`
  added.

- **Tests** — `tests/test_SlideChunker.py` updated for the new API;
  `tests/test_PageBasedChunker.py` (27 tests) and
  `tests/test_PdfPageChunker.py` (33 tests, including a real-PDF integration
  test against `TestDocs/Hedgehogs.pdf`) added.

### 🐛 Fixed — BM25 and Graph index always rebuilt on every query

`BM25Retriever.load_or_rebuild()` and `GraphRetriever.load_or_rebuild()` were
called on every RAGChat query.  When the persisted index was stale (e.g.
after reloading with a different document set), both methods rebuilt the
index in memory but never wrote it back to disk.  On the next query the old
file was loaded again, detected as stale, and rebuilt again — an endless
per-query rebuild loop.

Two fixes applied to both retrievers:

1. **Persist after rebuild** — `_persist(idx_path)` is now called immediately
   after `_rebuild_from_collection()` in `load_or_rebuild()`, so the
   refreshed index is saved and subsequent loads find a current file.
2. **In-memory short-circuit** — if `_data` already holds the index for the
   requested collection with the correct `doc_count_at_build`, `load_or_rebuild()`
   returns immediately without touching disk.  This eliminates redundant
   deserialization on every query within the same process.

- **Affected files:** `src/Strategies/BM25Retriever.py`,
  `src/Strategies/GraphRetriever.py`.

### 🐛 Fixed — Credit-card masking regex false-positive on hex hashes

`CREDIT_CARD_PLAIN_FALLBACK` used `(?<!\d)…(?!\d)` as its boundary guards.
Hex letters (`a`–`f`) are not digits, so a run of 13–19 decimal digits
embedded inside a SHA hash (e.g. the `FileHash` metadata field) satisfied
the guard and was replaced with `mask_credit_card` in both the ingestion
pipeline and the RAGChat output.

`CREDIT_CARD_LOOSE` and `CREDIT_CARD_PLAIN_FALLBACK` now use
`(?<![0-9a-fA-F])` / `(?![0-9a-fA-F])` — any adjacent hex character
(digit or letter `a`–`f`) prevents a match.  Real credit card numbers in
prose (surrounded by spaces or punctuation) are unaffected.

- **Affected file:** `src/Configuration/Config_Banned.py`.

---

### 🔧 Fixed — Rerank score normalization reverted to min-max for local docs

A previous refactoring had replaced min-max normalization with sigmoid
normalization for cross-encoder rerank scores.  Under sigmoid all raw logits
are mapped on an absolute scale: scores for technical-document queries
consistently landed in the `0.0001–0.15` range, far below every configured
threshold (`0.20`–`0.75`), resulting in zero chunks surviving the threshold
filter in all but the most lenient strategies.

- **Local docs** — rerank score is now min-max normalized across the candidate
  pool: worst candidate → `0.0`, best → `1.0`.  Thresholds retain their
  intended meaning (`0.75` = top 25 % of this pool, etc.) regardless of the
  cross-encoder model's absolute logit range.
- **Web docs** — rerank score remains `sigmoid(raw) × web_wt`.  The absolute
  scale correctly discounts web results relative to local results; an
  irrelevant local pool no longer inflates web scores by comparison.
- **Affected file:** `src/Chat/RAGChatImpl.py` (`_rerank_candidates()`).

### 🔧 Fixed — Retrieval fully skipped when a retriever weight is `0`

`graph_weight`, `bm25_weight`, and `vector_weight` were previously only used
during the RRF merge step.  Even at weight `0` the retriever still ran,
rebuilt its index on every query, and occupied slots in the candidate pool.

Each retrieval block now checks its weight before proceeding:

```python
if retrieve_mode in _GRAPH_MODES and _graph_weight != 0.0:   # graph
if retrieve_mode in _BM25_MODES  and _bm25_weight != 0.0:    # BM25
if retrieve_mode in _VECTOR_MODES and _vector_weight != 0.0: # vector
```

Setting a weight to `0` skips the retriever entirely — no index rebuild, no
query, no wasted slots in the reranker pool.

- **Affected file:** `src/Chat/RAGChatImpl.py`.

### 🐛 Fixed — Strategy weight loading treated `0.0` as "not set"

`QueryParts._base_defaults()` applied strategy weights using `if gw else 1.0`
(and equivalent for `vw`, `bw`).  Because `0.0` is falsy in Python, a
configured weight of `0` was silently replaced with `1.0`, making it
impossible to disable a retriever via strategy config.

Changed to `if gw is not None else 1.0` for all three weight fields.

- **Affected file:** `src/Chat/QueryParts.py`.

### 🔧 Changed — NARROW strategy: graph retrieval disabled (`graph_weight = 0`)

Graph retrieval is recall-oriented and cross-file.  `SingleDocumentSelector`
(used by NARROW) discards every chunk except those belonging to the single
best-matching file, so graph results from other files are always thrown away
after retrieval.  Running the graph index rebuild on every NARROW query was
pure overhead.

`graph_weight` in the NARROW strategy config is now `0`.  Combined with the
retrieval-skip guard above, the graph index is not queried at all when NARROW
is active.

- **Affected file:** `src/Configuration/Config_RAGChat.py`.

### 🔧 Fixed — `debug_mode` selector `ge` / `le` labels swapped in interactive picker

Both the `debug_level!` branch and the `debug_mode!` branch in `QueryParts`
showed the `le` choice first and described it as "activates this level and all
below" — the opposite of what `le` does.  `ge` was in second position and was
described as "activates this level and all above".

- `ge` is now listed first (it is the default mode).
- Labels corrected: `ge` → *"activates this level and all below"*;
  `le` → *"activates this level and all above"*.
- The `default=` index in both `inquirer.select()` calls updated accordingly.
- `DebugHelper.active()` docstring corrected to match.

- **Affected files:** `src/Chat/QueryParts.py`, `src/Helpers/DebugHelper.py`.

---

## [Released] — 2026-05-20

### 🔧 Fixed — `debug_mode` now honored in all debug guards

All debug-level guards throughout the codebase previously used a hardcoded `>=`
comparison, effectively ignoring the `debug_mode` setting (`ge`/`is`/`le`).

- Added `DebugHelper.check(cfg, level)` — mode-aware static method that replaces
  `DebugHelper.level(cfg) >= level` guards and respects `is` / `le` / `ge` modes.
- Added `DebugHelper.check_session(session, level)` — session-based equivalent,
  reads `session.debug_level` and `session.debug_mode`.
- Replaced 50 cfg-based guards (`DebugHelper.level(X) >= N`) with
  `DebugHelper.check(X, N)` across 16 source files.
- Replaced 19 session-based guards (`(session.debug_level or 0) >= N`) with
  `DebugHelper.check_session(session, N)` in `RAGChatImpl`, `ChatContext`,
  `PromptRewrite`, and `HomeBrewChunkSelector`.

### 🔧 Changed — Web results now subject to rerank threshold

Previously, web documents (`Source == "Web"`) bypassed the cross-encoder rerank
threshold in `HomeBrewChunkSelector.filter_threshold()` unconditionally, meaning
low-scoring web snippets were always forwarded to the LLM alongside high-scoring
local documents. This caused the LLM to select low-relevance web content over
high-ranking local documents.

Web results are now filtered by the same `chroma_threshold` as local documents.
A high-quality local document will no longer be overshadowed by a flood of
below-threshold web snippets.

### ✨ Added — `debug_mode=le` (less-equal) comparison mode

A third comparison mode `"le"` (less-equal, `<=`) is now accepted everywhere
`debug_mode` is parsed or set.

- `DebugHelper.parse()` accepts `"le 30"` / `"<= 30"` → `(30, "le")`.
- `DebugHelper.active(level)` returns `configured <= level` when mode is `"le"`.
- The `debug_level!` and `debug_mode!` interactive pickers in `QueryParts` show
  a third choice: `<= level  (le — activates this level and all below)`.
- `debug_mode=le` is now accepted by `QueryParts._set()` and the `ChatCompletionHandler`
  alias field.
- `_ALLOWED_DEBUG_LEVELS` label descriptions and `CONFIGURATION_REFERENCE.md` updated accordingly.

### ✨ Added — Debug level 31 "Chunk Content" — full chunk text + metadata

A new named debug level `32` (`"Chunk Content"`) has been added between
*Prompt Check Input* (31) and *Algos* (40).

When active (`debug_level >= 31` with `ge` mode, or `debug_level=31` with `is`),
each chunk selected for the LLM context window is printed in full, showing:

| Field | Source |
| --- | --- |
| file name, file type, language | `metadata["FileName"]`, `FileType`, `Language` |
| full file path | `metadata["FilePath"]` |
| chunk_id | `metadata["chunk_id"]` (falls back to `doc.id`) |
| FileHash | `metadata["FileHash"]` |
| all scores | `chroma_score`, `rerank_score`, `rrf_score` (whichever are present) |
| retriever sources | `metadata["retriever_sources"]` |
| raw text | `page_content` |

Implemented as `ChunkSelector._print_chunk_content()` in
`src/Strategies/HomeBrewChunkSelector.py`. The method lives on the abstract
base class so all three concrete selectors (`ScoreRankedSelector`,
`PerFileCapSelector`, `SingleDocumentSelector`) call it consistently.

### 🔧 Changed — `_ALLOW_WEB_SEARCH` + `_WEB_SEARCH_DRY_RUN` replaced by `_WEB_SEARCH_MODE`

The two boolean web-search switches have been collapsed into a single
three-value string in `Config_WebSearch.py`:

| Value | Behaviour |
| --- | --- |
| `"off"` | No internet leg, no compliance check — all web queries blocked (replaces `_ALLOW_WEB_SEARCH = False`). |
| `"dry_run"` | Compliance gates run as normal; passing queries are **not** sent to the network (replaces `_ALLOW_WEB_SEARCH = True` + `_WEB_SEARCH_DRY_RUN = True`). |
| `"on"` | Full production path — queries pass compliance then go live (replaces `_ALLOW_WEB_SEARCH = True` + `_WEB_SEARCH_DRY_RUN = False`). |

`_WEB_SEARCH_MODE` is the **master switch** and overrides all other web-search
settings, including `_OPENWEB_UI_WEBSEARCH`. All call sites updated
(`ChatCompletionHandler`, `QueryParts`, `RAGChatImpl`, `Chatter`,
`CommandProcessor`, `WebRetriever`, `StartupCommons`). `CONFIGURATION_REFERENCE.md` updated.

### ✨ Added — Startup warning when `_OPENWEB_UI_WEBSEARCH=True` conflicts with `_WEB_SEARCH_MODE`

If `_OPENWEB_UI_WEBSEARCH = True` in `Config_WebSearch.py` but
`_WEB_SEARCH_MODE` is not `"on"`, a `BRIGHT_ORANGE` warning is now printed
once at service startup explaining that the default has no effect and how to
resolve the conflict.

### 🔧 Changed — Startup banner: `"Internet connection"` → `"Outbound downloads"`

The generic `"Internet connection"` startup warning has been renamed to
`"Outbound downloads"` with a more precise message:
*One or more settings allow outbound model/data downloads (HuggingFace hub,
NLTK, Argos, etc.).*

`SERVE_OPENWEBUI_CHAT` has been removed from the `warn_print` triggers — the
service accepting inbound connections from OpenWebUI is normal operation, not
an outbound data risk. It is now printed as a plain info line with the note
*inbound only, no outbound data risk*.

### 🔧 Changed — Web search startup messages scoped to RAGChat / RAGChatService

The web search mode status block in `StartupCommons.common_start()` — covering
`_WEB_SEARCH_MODE = "off"` / `"dry_run"` / `"on"` — is now only printed when
`_FRIENDLY_NAME` is `"RAGChat"` or `"RAGChatService"`. RAGLoad and DocClassify
no longer display these messages at startup; web search is not available in
those apps.

### 🔧 Changed — `_OPENWEB_UI_WEBSEARCH` mismatch warning scoped to RAGChatService

The warning that `_OPENWEB_UI_WEBSEARCH = True` has no effect when
`_WEB_SEARCH_MODE` is not `"on"` is now only printed for `RAGChatService`.
The setting is an OpenWebUI API default and is only meaningful for the service;
the interactive RAGChat CLI does not read it.

### 🔧 Fixed — `pretty_sleep()` final newline dropped at debug level 0

`Helpers.pretty_sleep()` ended with `self.pretty.write("N", "", "")` to emit a
newline after the last progress dot. At debug level 0 with `always_on=False`,
`PrettyWriter.write()` silently returns early for non-error/warning severities,
so no newline was emitted and the next startup line ran directly onto the dot
line.

Fixed by replacing `self.pretty.write("N", "", "")` with a plain `print()` —
unconditional, not subject to PrettyWriter's debug guard.

---

## [v0.2.12] — 2026-05-19

### 🔧 Fixed — `PromptRewrite` honors `context_size_override` in sub-calls

`PromptRewrite.rewrite()` now passes `min(session.context_size_override,
context_limit)` as `num_ctx` to the rewrite LLM when a per-request context
override is active. Previously the rewrite sub-call always used the full
`TokenBudget.get_context_limit()` value, ignoring any
`session.context_size_override` set by the caller. The main chat path and the
rewrite sub-call now behave consistently.

### ✨ Added — `WebSearchFilter.get_instance(log_verbose=True)` logs ALLOW decisions

`WebSearchFilter.get_instance()` and `WebSearchFilter.__init__()` now accept
`log_verbose: bool = False`. When `True`, ALLOW decisions are appended to the
decision log alongside BLOCK and ESCALATE outcomes. The default (`False`)
preserves the existing behavior — only BLOCK and ESCALATE decisions are
written.

### ✨ Added — `Config_Banned.WEB_SEARCH_INTENT_EXTENSIONS`

A new `WEB_SEARCH_INTENT_EXTENSIONS` dict at the end of `Config_Banned.py`
lets operators extend the baseline web-search intent filter (defined in
`Config_WebSearch.py`) without editing that file. Three keys are supported:

| Key | Purpose |
| --- | --- |
| `entity_extensions` | Extend the entity lists of existing baseline categories (e.g. add more finance terms to `"finance"`). |
| `entity_categories_extra` | Add entirely new intent categories not present in the baseline. |
| `threshold_overrides` | Override the score threshold for individual baseline categories. |

All three default to empty dicts — no behavioral change until populated.
Updating this dict requires recalculating `_BANNED_CONFIG_HASH` in
`Config_Global.py` (see `src/Scripts/RecalcConfigHashes.py`).

### 🐛 Fixed — `Session.preferred_response_language` field restored

The `/settings preferred_response_language` interactive command in
`QueryParts.py` referenced a `Session` attribute that was absent after the
v0.2.11 reply-language removal, causing a Pylance error. The
`preferred_response_language: str | None = None` field is restored to
`Session.__init__()`. The LLM reply-language instruction (`{language_instruction}`
placeholder, `lang_hint` assembly, `Chatter.py` integration) remains removed.

### 🐛 Fixed — `QueryParts._base_defaults` web_weight guard

Changed `if ww is not None:` to `if ww:` when applying a strategy's
`web_weight` value to the session. `Config.get_float()` always returns
`float` (never `None`), so the previous guard was vacuously true and would
overwrite a pre-set per-session `web_weight` even when the strategy config
had no `web_weight` entry (which `get_float` silently defaults to `0.0`).
The truthiness guard correctly skips the assignment when the key is absent.

---

## [v0.2.11] — 2026-05-18

### 🗑️ Removed — reply-language instruction (`reply_lang` / `preferred_response_language`)

The mechanism that instructed the chat LLM to reply in the user's native language
has been removed. It consisted of three pieces:

- `{language_instruction}` placeholder in the `_PROMPT_CHAT` template
  (`Config_RAGChat.py`).
- `lang_hint` assembly and the `language_instruction=` kwarg in `Chatter.py`.
- `preferred_response_language` / `response_language` session fields
  (`Session.py`) and the `PREFERRED_RESPONSE_LANG` config key
  (`Config_RAGChat.py`).
- `reply_lang` field in `ChatCompletionRequest` and its `_applyOverrides()`
  handler (`ChatCompletionHandler.py`).
- `reply_lang` interactive command in `QueryParts.py`.

The **query-translation pipeline** (`HfTranslator` / `facebook/m2m100_1.2B`) is
**kept active**: non-English queries are still normalised to English before
retrieval so that vector, BM25, and HYBRID fusion all see a consistent query.
The LLM responds in whatever language it finds most natural given its context.

### 🧹 Cleaned — `test_prompt_rewrite.py`

Removed `get_translated_wordlist` stub and `preferred_response_language`
from `StubSession`. All 37 prompt-rewrite tests pass.

---

## [v0.2.10] — 2026-05-16

### 🔧 Changed — Rerank: sigmoid normalization replaces min-max

- `_rerank()` in `RAGChatImpl.py` now normalises raw cross-encoder logits
  with the sigmoid function (`σ(x) = 1 / (1 + e^−x)`) instead of pool-based
  min-max normalization.
- **Why**: sigmoid preserves absolute relevance scale — a document with a
  poor cross-encoder score stays near 0 regardless of what else is in the
  pool.  Min-max was inflating irrelevant documents whenever all candidates
  were weakly relevant.
- Web-sourced chunks: `rerank_score = sigmoid(raw) × web_wt`
  (default `web_wt = 0.5` from `_WEB_SEARCH.default_web_weight`).
- Local chunks: `rerank_score = sigmoid(raw)`.
- `import math` added to `RAGChatImpl.py`.

### 🔧 Changed — Rerank debug table shows RawScore and AdjScore

- Rerank debug output (debug_level ≥ 10) now prints two score columns:
  - **RawScore** — raw cross-encoder logit (e.g. `−1.42`).
  - **AdjScore** — sigmoid-normalised value after web weight is applied.
- Both columns are absent from the pool-specific tables (Chroma, BM25, Graph,
  Web, Merge) because those stages do not produce cross-encoder scores.

### 🔧 Changed — Merge debug table columns aligned with Rerank table

- `_print_merged_debug()` column widths updated to match the Rerank table:
  `Pos(6)  RRFScore(10)  [blank](10)  Retrievers(17)  File(30)`.
- The blank column occupies the AdjScore slot so Retrievers and File land at
  the same character positions in both tables when reading the terminal log.

### 🔧 Changed — Web docs bypass rerank threshold

- `HomeBrewChunkSelector.filter_threshold()` now passes web-sourced chunks
  through unconditionally, regardless of their rerank score.
- **Rationale**: the cross-encoder looks for direct lexical overlap between
  query and text.  Web snippets often answer indirectly (e.g. *"whales eat
  krill"* for *"do whales eat insects?"*) and therefore score below threshold
  even though they are the most relevant context available.  Their relevance
  was already validated by the search engine before retrieval.
- Only applies to documents whose `metadata["Source"] == "Web"`.

### 🔧 Changed — `_PROMPT_CHAT`: one-step inference carve-out

- Added an **ONE EXCEPTION — DIRECT LOGICAL INFERENCE FROM CONTEXT** clause
  to `_PROMPT_CHAT` in `Config_RAGChat.py`.
- When the context provides an explicit, reasonably complete description of a
  characteristic (e.g. diet, habitat, function) and the query asks whether
  something absent from that description belongs to it, the LLM may now answer
  using **one direct inference step**, citing the context statement.
- Example: context states *"whales eat krill, fish, and squid"*; query asks
  *"do whales eat insects?"* → the LLM may now answer
  *"No — according to the sources, insects are not part of their described diet."*
- The clause is deliberately narrow: single-step only, no classification,
  no outside knowledge, must cite the context statement.
- Previously the strict *"do NOT infer beyond what the context states"*
  instruction caused the LLM to respond *"The context is silent"* even when
  all necessary information was present.

---

## [v0.2.9] — 2026-05-15

### 📚 Docs — README split into focused documents

- `README.md` reduced from ~2 650 lines to ~625 lines and now serves as a
  landing page only. Detailed material moved into three new top-level
  documents linked from the README's *Documentation Map*:
  - `INSTALL.md` — prerequisites, cloning, dependencies, Ollama / Open WebUI /
    Argos / NLTK / Tesseract / spaCy / GPU setup, running the test suite,
    first-run walkthrough.
  - `CONFIGURATION_REFERENCE.md` — per-file reference for every `Config_*.py`
    (Global, Models, RAGChat, RAGLoad, DocClassify, Banned, Internet),
    CLI overrides, translation config, troubleshooting, performance tuning.
  - `EXAMPLES.md` — end-to-end terminal sessions for `RAGLoad`, `RAGChat`,
    `DocClassify`, `RAGChatService`, class diagrams and project structure.
- All cross-document anchors fixed; image-path casing normalised
  (`pics` → `Pics`) across 15 references; trailing-blank-line MD012 lint
  violations eliminated.
- README → `ARCHITECTURE.md` content move under § Query Rewrite:
  - Worked first-turn / follow-up examples (with screenshots) relocated
    next to the rewrite-flow diagram.
  - m2m100 translation-quality caveat and NLLB workaround relocated into
    the *User-Query Translation* subsection.
  - README keeps a concise summary plus a single pointer to ARCHITECTURE.

### 📈 Docs — Query-rewrite flow diagram synced with current code

- `Documentation/FlowCharts/rag_query_rewrite_correct_flow.md` rewritten
  to reflect the actual `PromptRewrite` / `RetrievalGate` pipeline. The
  legacy *"compact history (prune_batch summarized)"* node was replaced
  with the current behaviour:
  - `new:` / `new topic:` topic-switch prefix → history reset +
    `force_skip_rewrite=True`.
  - Topic-summary source selection: prefers `session.last_topic_referents`
    (LLM-distilled, persisted across turns) over the legacy ASSISTANT-block
    fallback (`TOPIC_SUMMARY_MODE = last|all`).
  - Rewriter LLM JSON output contract:
    `{depends_on_previous_turn, confidence, reasoning, contextual_rewrite,
    standalone_rewrite, salient_referents}`.
  - Confidence-thresholded decision between `contextual_rewrite` and
    `standalone_rewrite`.
  - Post-hoc grounding check + pronoun-stripping sanitiser that sets
    `session.rewrite_was_underspecified`.
  - `RetrievalGate` three-signal block (meta-descriptor, unanchored
    pronoun, `rewrite_was_underspecified`) with the `❔` clarification
    short-circuit.
  - `last_topic_referents` persistence loop back to the next turn.
- PNG (`rag_query_rewrite_correct_flow.png`) re-rendered via
  `scripts_posh/private/render_diagrams.py`.

### 🐛 Fixed — query-rewrite topic-detect prompt missed `RAM` ↔ `ram`

- `_PROMPT_TOPIC_DETECT` in `Config_RAGChat.py` gained **RULE 8** instructing
  the topic-detection LLM to treat acronyms case-insensitively
  (`RAM` ≡ `ram`, `GPU` ≡ `gpu`, etc.). Previously the rewriter could
  fail to recognise that a follow-up question about *"ram"* referred to
  the *"RAM"* discussed in the previous turn, leading to spurious
  topic-switch decisions.

### 🔧 Changed — `FixedSizeChunker` renamed to `RecursiveChunker`

- `src/Strategies/Chunkers/FixedSizeChunker.py` renamed to
  `src/Strategies/Chunkers/RecursiveChunker.py`; class `FixedSizeChunker`
  renamed to `RecursiveChunker` throughout.
- The `FIXED_SIZE` config key in `_CHUNK_STRATEGY` / `_CHUNKERS`
  (`Config_Global.py`) renamed to `RECURSIVE`.
  `DocumentIngestionStrategy._make_chunker()` now imports and
  instantiates `RecursiveChunker` for that key.
- `tests/test_chunkers.py` test class renamed `TestFixedSizeChunker` →
  `TestRecursiveChunker`; `tests/test_auto_chunk.py` import updated.
  The rename reflects that the splitter uses
  `RecursiveCharacterTextSplitter` rather than a simple fixed-size split.

### ✅ Tests

- Full collection still green (751 tests collected).

---

## [v0.2.7] — 2026-04-28

### ✨ New — `DebugHelper` — centralised debug-level evaluation

- New `src/Helpers/DebugHelper.py` — thin, config-aware helper that
  centralises all `DEBUG_LEVEL` guard evaluations:
  - `DebugHelper(cfg).on(level)` — `DEBUG_LEVEL >= level` (mode-independent)
  - `DebugHelper(cfg).only(level)` — `DEBUG_LEVEL == level`
  - `DebugHelper(cfg).active(level)` — respects `DEBUG_MODE` (`ge` ≥ or `is` ==)
  - `DebugHelper.level(cfg)` — static; returns the numeric level by parsing
    the combined `"ge 30"` / `"is 45"` / `"none"` string format stored in
    `DEBUG_LEVEL`.
  - `DebugHelper.parse(raw)` — static parser; turns any accepted format into
    `(level, mode)`.
- `DEBUG_LEVEL` can now carry a combined mode+level string (`"ge 30"`,
  `">= 30"`, `"is 45"`, `"== 45"`, `"none"`) as a single config value instead
  of requiring two separate keys.  All consumer code was migrated from
  `cfg.get_int("DEBUG_LEVEL") >= N` guards to `DebugHelper` calls.
- `_ALLOWED_DEBUG_LEVELS` dict added to `Config_Global.py` — named presets
  (`None=0`, `Basic=10`, `Service=20`, `Standard=30`, `Algos=40`,
  `Query Rewrite=45`, `Components=55`, `Chat Prompt=60`,
  `Extracted Content=70`, `Ollama Response=80`, `Streaming=100`)
  drive the interactive picker and validation.

### ✨ New — `debug_level` and `debug_mode` as live interactive settings

- `debug_level` and `debug_mode` added to `QueryParts` `COMMAND_SPECS` so
  the active debug verbosity can be changed at any point during a chat session:
  - `debug_level=30` / `debug_level=ge 30` — assign directly
  - `debug_level!` — interactive picker showing the named presets from
    `_ALLOWED_DEBUG_LEVELS`; also prompts for comparison mode (`ge`/`is`)
  - `debug_level?` — show the current level
  - `debug_mode=ge` / `debug_mode=is` — change comparison mode independently
  - `debug_mode!` — interactive picker for `ge` / `is`
- Both commands write the combined string back to `Config.DEBUG_LEVEL` via
  `cfg.set(..., force=True)` so all downstream `DebugHelper` calls see the
  change immediately. `session.debug_level` and `session.debug_mode` are kept
  in sync for display.
- Current values appear in the `show?` status block on the `▶ Debug:` line.

### ✨ New — `terminal_line_size` live interactive setting

- `terminal_line_size` added to `QueryParts` `COMMAND_SPECS` so terminal
  output width can now be adjusted at chat time without restarting:
  - `terminal_line_size=120` — assign directly from the query prompt
  - `terminal_line_size!` — interactive picker with validation (min 40)
  - `terminal_line_size?` — show current effective value
  - The setting appears in the `show?` status block under **Output**.
- `TERMINAL_LINE_SIZE` is configured as a `{"debug": 160, "no_debug": 80}`
  dict in `Config_Global.py`; `QueryParts._resolve_terminal_line_size()`
  resolves the correct key based on the active debug level, matching the
  logic already used by `PrettyWriter`.

### 🐛 Fixed — live setting changes had no effect on output width

- `PrettyWriter.terminal_line_size` and `Chatter.terminal_line_size` were
  read once at `__init__` time and cached as instance variables.  Any
  `terminal_line_size=` command issued during a session was silently ignored
  because existing instances kept the stale value.
- Both are now `@property` accessors that read `TERMINAL_LINE_SIZE` from
  `Config` on every call, so changes take effect immediately on the next
  output line.

### 🐛 Fixed — `Config.set(force=True)` did not update CLI-overridden keys

- `cfg.set(key, value, force=True)` wrote the new value to `self.cfg` but
  `cfg.get()` checks `self.args` first; when a key was also set via CLI
  (e.g. `--debug_level none`), `self.args` still held the original CLI
  value and `get()` continued to return it.
- `_set` now also updates `self.args[key]` when `force=True` and the key
  is present there, so all downstream reads reflect the new value.

### 🎨 Changed — clarification message displayed in CYAN

- The `❔` clarification prompt returned by `RetrievalGate` is now printed
  in CYAN (via the `color` parameter of `print_llm_answer`) so it is
  visually distinct from normal LLM answers.

---

## [v0.2.6] — 2026-04-27

### ✨ New — RetrievalGate: clarification prompt for underspecified queries

- Added `src/Chat/RetrievalGate.py` — intercepts queries that are too vague
  to retrieve meaningfully and returns a clarification request instead of
  calling the LLM.
- Two detection signals, both powered by spaCy morphology (no word lists):
  - **Attribute signal** — wh-question + meta-descriptor noun phrase (e.g.
    *"what are the specifications?"*) with no named-entity anchor.
  - **Pronoun signal** — 3rd-person pronoun (`Person=3` morph) with no
    named-entity anchor (e.g. *"does it have spines?"*).
- Clarification response prefixed with `❔` and returned to both the terminal
  and the OpenWebUI/API stream.
- Gate is bypassed when chat context is not active.

### ✨ New — spaCy morphology replaces pronoun word lists

- `_FIRST_SECOND_PERSON` frozenset removed from `PromptRewrite.py` and
  `RetrievalGate.py`; pronoun detection now uses `tok.morph.get("Person", [])`.
- `anaphoric_pronouns` list removed from `Config_RAGChat.py` (`_QUERY_REWRITE`).
- `meta_descriptors` list added to `Config_RAGChat.py` (`_QUERY_REWRITE`) —
  drives the attribute signal in `RetrievalGate` (e.g. *specifications*, *features*,
  *capabilities*, *parameters*, …).

### 🔧 Session — new fields

- `Session.rewrite_was_underspecified: bool` — set when the rewritten query
  contains a 3rd-person pronoun; preserves prior referents for the next turn.
- `Session.clarification_response: str | None` — carries the clarification
  message from `RetrievalGate` to `Chatter`.

---

## [v0.2.5] — 2026-04-27

### 🐛 Fixed — `vector_weight`, `bm25_weight`, `graph_weight` settings had no effect

- All three weight tokens were recognised by the settings regex and displayed
  correctly in the `🛠️` prompt, but were absent from `COMMAND_SPECS` in
  `QueryParts.py`.
  Because `_get_attr_name()` resolves the session attribute exclusively via
  that dict, it returned `None` for all three; `_read_and_apply_value` then
  raised `ValueError("Invalid command: …")` silently — the value was never
  written to the `Session` object.
- Fix: added `"vector_weight"`, `"bm25_weight"`, and `"graph_weight"` entries
  to `COMMAND_SPECS` with `"type": "float"` and `"attr"` matching the
  `Session` field names. Assignment (`=`), interactive picker (`!`), and
  query (`?`) now all work correctly.

### ✨ New — `VECTOR_BM25` retrieval mode

- Added `VECTOR_BM25` retrieve mode (ChromaDB + BM25 fused via RRF) — the
  previously missing two-way combination alongside `VECTOR_GRAPH` and `BM25_GRAPH`.
- Added to `_ALLOWED_RETRIEVE_MODES` in `Config_RAGChat.py` and to the
  `_VECTOR_MODES` / `BM25_MODES` membership tuples in `RAGChatImpl.py`.

### 🔧 Rename — `search_mode` → `retrieve_mode`

- Config key, `Session` field, `QueryParts` command token, API request field,
  and all documentation updated from `search_mode` to `retrieve_mode`.
- `_ALLOWED_SEARCH_MODES` in `Config_RAGChat.py` renamed to `_ALLOWED_RETRIEVE_MODES`.
- Fully backwards-incompatible: any OpenWebUI Advanced Parameter or API client
  passing `search_mode` must be updated to `retrieve_mode`.

- **`GraphRetriever.ingest_file()`** — new public method encapsulates
  per-file incremental graph index updates (load-or-rebuild, remove old
  chunks, add new chunks, persist). Replaces the removed private method
  `ChunksToDBStrategy._update_graph_index`.
- **`BM25Retriever.ingest_file()`** — new public method encapsulates
  per-file incremental BM25 index updates (load-or-rebuild, remove old
  chunks, add new chunks, persist). Replaces the removed private method
  `ChunksToDBStrategy._update_bm25_index`.

### 🔧 Refactoring — DocumentIngestionStrategy (renamed from ChunksToDBStrategy)

- **`src/Strategies/ChunksToDBStrategy.py`** renamed to
  `DocumentIngestionStrategy.py`; class renamed `DocumentIngestionStrategy`,
  public entry method renamed `ingest()` (was `docChunksToDBStrategy()`).
- **`src/Strategies/StrategyType.py`** — enum member renamed
  `DOCUMENT_INGESTION` (was `CHUNKS_TO_DB`); string value updated to
  `"DocumentIngestionStrategy"`.
- `DocumentIngestionStrategy` holds `self.bm25_retriever` and
  `self.graph_retriever` as named instance attributes (singleton pattern)
  instead of transient local variables, consistent with all other
  dependencies in `__init__`.
- Updated: `RAGLoad.py`, `LoadAndClassifyProcessor.py`,
  `Chunkers/SemanticChunker.py` (docstring), `tests/test_auto_chunk.py`.

---

## [v0.2.4] — 2026-04-23

### Added

#### HF-Native User-Query Translation Backend (M2M-100)

- New optional translation backend for the user-query normalisation path:
  Facebook/Meta **M2M-100 1.2B** (MIT, ~5 GB on disk), loaded lazily via
  Hugging Face Transformers and routed through the existing
  [Compliance.HFDownloader](src/Compliance/HFDownloader.py) consent +
  audit flow exactly the same way the embedder is.
- Selectable per call via the new `translation_backend` interactive
  command (`argos`, `m2m100`, `off`) and configured globally through
  `_QUERY_REWRITE.TRANSLATION_BACKEND` in
  [Config_RAGChat.py](src/Configuration/Config_RAGChat.py).
- Implementation lives in
  [HfTranslator](src/Compliance/HfTranslator.py) — singleton, lazy load,
  CPU-first defaults, `_TRANSLATION` model role bound through
  [Config_Models.py](src/Configuration/Config_Models.py)
  (`_ACTIVE_TRANSLATION = "m2m100"`).
- Loader silences three benign `tie_word_embeddings` warnings the M2M-100
  HF checkpoint emits, and clears the shipped `max_length=200` /
  `early_stopping=True` defaults so generate() runs cleanly with our
  per-call `max_new_tokens`.

#### Post-Rewrite Re-Translation Pass

- The `PromptRewrite` step can re-introduce non-English entities pulled
  from chat history (e.g. an English query getting rewritten as
  *"Are Igel, Katzen ... mammals?"*). RAG-LCC now detects the
  language of the rewritten query and runs the configured translator a
  second time when needed, so retrieval (vector + BM25) always sees a
  clean, single-language English query.
- The originally detected source language is preserved in
  `Session.response_language` so the final LLM still answers in the
  user's language.

### Added

#### HeadingChunker — Configurable Breadcrumb Placement

- New `_CHUNKERS.HEADING.BREADCRUMB_MODE` option controls where the
  heading trail (`H1 > H2 > H3`) is embedded in each chunk:
  - `prefix` — legacy behaviour: prepended to the body. Pollutes the
    leading tokens of every chunk that shares a section path, which can
    bias small LLMs and make many chunks embed too similarly.
  - `suffix` — **new default**: appended after the body as
    `[section: H1 > H2 > H3]`. Leading tokens of each chunk are now the
    actual paragraph text while the section context is still visible to
    the embedding model.
  - `off` — omitted from chunk text entirely.
- The breadcrumb is now always preserved in `metadata["HeadingPath"]`
  regardless of mode, so reranking, filtering, and citations can use it
  even when it is absent from the chunk text.
- `HeadingChunker._sections_to_texts` now returns `(chunk_text, breadcrumb)`
  pairs and `_split_oversized` re-applies the chosen mode per sub-chunk.
- Re-ingestion of any collection containing `.md` / `.docx` files is
  required for the new mode to take effect on existing data.

#### Generalised Chat Prompts (`_PROMPT_CHAT`, `_PROMPT_REWRITE`)

- Removed corpus-specific examples (cats, hedgehogs, Pferde, etc.) from
  both prompts; replaced with domain-neutral placeholders
  (entity A/B, topic T, property P).
- `_PROMPT_CHAT` STEP 2 now infers entities from chunk **content**
  rather than from filenames; filenames are explicitly downgraded to
  metadata used only for grouping and citations.
- Added an explicit ban on outside-knowledge hedges such as
  `"X is also Y"` / `"X is generally Y"` when the context is silent on
  the attribute.

#### Hybrid Retrieval Strategy (BM25 + Vector)

- Implemented a new hybrid search mode combining semantic vector embeddings with keyword-based BM25 lexical retrieval.
- Retrieval results are merged and re-ranked using Reciprocal Rank Fusion (RRF).
- Added `search_mode` parameter to select the retrieval method dynamically: `VECTOR`, `BM25`, or `HYBRID` (default).

#### BM25 Index Directory Separation

- BM25 index files now live in a dedicated `_BM25_INDEX_DIR` (`chromadb/bm25/<collection>`)
  instead of inside the ChromaDB collection directory. Prevents ChromaDB upgrades from
  wiping the index. Deletion uses the same jailbreak-safe guard as ChromaDB collection delete.

#### Query Rewrite — Topic-Change Detection Gate

- Added a lightweight Jaccard-overlap pre-check before calling the rewrite LLM.
  The new query is tokenized and compared against the last history question;
  if overlap falls below `_QUERY_REWRITE.topic_change_threshold` (default `0.10`)
  the rewrite is skipped entirely. This prevents small models (8b) from
  hallucinating rewrites on unrelated follow-up questions (e.g. switching from
  "PCIe slots on Workstation XY" to "what animals are in the collection").

#### Query Rewrite — Diagnostic Messages

- Every exit path in `PromptRewrite.rewrite()` now logs an informational
  message with the reason the query was or was not rewritten (disabled,
  no history, topic change, LLM error, empty response, unchanged, or
  rewritten with old/new values).

---

## [v0.2.3] — 2026-04-22

### ➕ Added

#### LLM-Based Query Rewriting

- New `PromptRewrite` singleton rewrites follow-up queries using conversation
  history so pronouns and vague references are resolved before retrieval.
- Gated by `use_chat_context` + `_QUERY_REWRITE.enabled`; falls back to the
  original query on error or when there is no history.
- Uses a dedicated `_LLM_REWRITE_PROMPT` model role
  (`_ACTIVE_LLM_REWRITE_PROMPT` in `Config_Models.py`).
- `max_history_turns` session parameter (default `3`) caps how many prior
  turns are fed to the rewriter, preventing stale-context pollution.
  Set to `0` to disable the limit.
- `_PROMPT_REWRITE` includes a RULE 0 language lock (`{query_language}`)
  to prevent the rewriter from outputting the wrong language when chat
  history contains foreign-language tokens.

#### Multilingual Retrieval — BM25 Query Translation

- New `_QUERY_REWRITE.TRANSLATE_RETRIEVAL_QUERY` option (`"off"` / `"argos"`).
  When `"argos"`, the query sent to BM25 is translated to English via Argos
  Translate when `search_mode` is `BM25` or `HYBRID` and a non-English query
  is detected. Vector search is unaffected; the session query stays in the
  original language so the final LLM answers in the user's language.
- `pronouns_by_lang` in `_QUERY_REWRITE` extended to cover German, French,
  Spanish, and Italian pronoun/demonstrative lists so the rewriter correctly
  detects anaphora across languages.

### 🔧 Changed

- `merge_with_chunks` replaced by `PromptRewrite.rewrite()` in
  `RAGChatImpl._retrieve()` — chat history is no longer injected as fake
  document chunks.
- Removed `chat_context_k_value` (stored and displayed but never drove
  retrieval logic) from `Session`, `Config_RAGChat`, `QueryParts`, and
  `ChatCompletionHandler`.

---

## [v0.2.2] — 2026-04-19

### � Fixed

#### API Compliance-Output Heading & Missing Strategy Config Key

- **`src/Api/ChatCompletionHandler.py`** — Fixed duplicate word in the
  compliance-results heading: "Filter chain algo results Results" →
  "Filter chain algo results" (all 6 occurrences).
- **`src/Configuration/Config_RAGChatService.py`** — Added
  `_ACTIVE_CHUNK_SELECT_STRATEGY` to the explicit re-import list.
  Python's `import *` silently drops `_`-prefixed names, so the key
  was missing from the RAGChatService config, causing a "non-existent
  path" warning and an invalid-strategy error at startup.

### �🔧 Changed

#### Generalised Strategy Selection Pattern

- **`src/Configuration/Config_Global.py`** — Replaced `_CHUNKER` (single string)
  and `_AUTO_CHUNK` (flat file-type map) with a two-level
  `_ACTIVE_CHUNKER_CONFIG` / `_CHUNK_STRATEGY` structure.  `_CHUNK_STRATEGY`
  holds named profiles (`DETAILED`, `FAST`); `_ACTIVE_CHUNKER_CONFIG` selects
  which profile is active.  Per-file-type routing is now always enabled —
  the former `AUTO` mode is the only mode.
- **`src/Configuration/Config_Models.py`** — Renamed model-role selectors to
  `_ACTIVE_LLM`, `_ACTIVE_LLM_CHK`, `_ACTIVE_EMBED`, `_ACTIVE_CROSS`,
  `_ACTIVE_OLLAMA`, `_ACTIVE_OPENWEBUI`.  The inner `_MODELS` dict keys
  (role names) are unchanged.
- **`src/Configuration/Config_RAGChat.py`** — Renamed `CHUNK_SELECT_STRATEGY`
  to `_ACTIVE_CHUNK_SELECT_STRATEGY`.
- **`src/Helpers/Helpers.py`** — `get_model_args(role)` now derives the
  selector key by prepending `_ACTIVE` to the role name automatically;
  callers continue to pass the role (e.g. `"_EMBED"`).
- **`src/Compliance/HFDownloader.py`** — Same `_ACTIVE` prefix applied to its
  direct `cfg.get_str()` call for model-type resolution.
- **`src/Strategies/ChunksToDBStrategy.py`** — Replaced the `_CHUNKER`
  if/elif chain and `_auto_mode` flag with a single `_chunk_map` lookup
  resolved from `_CHUNK_STRATEGY.<active_profile>`.
  `_resolve_chunker_for_file()` now always routes via the map.
- **`src/Chat/QueryParts.py`**, **`src/Apps/RAGChat.py`**,
  **`src/Api/ChatCompletionHandler.py`** — Updated config key from
  `CHUNK_SELECT_STRATEGY` to `_ACTIVE_CHUNK_SELECT_STRATEGY`.
- **`tests/test_auto_chunk.py`** — Updated stubs (`_chunk_map` instead of
  `_auto_mode` + `_auto_chunk_map`); removed `test_noop_when_not_auto`.
- **`tests/test_ensemble.py`**, **`tests/test_hfdownloader.py`** — Selector
  keys in stub config mappings updated to `_ACTIVE_*` form.
- **`ARCHITECTURE.md`** — Added "Strategy Selection Pattern" subsection
  documenting the `_ACTIVE` selector → profile → parameters convention.
  Updated chunker-selection and model-selector sections.
- **`README.md`** — Updated all selector references, code snippets, and
  model-combination table.  Added links to the new architecture section.
- **`CHANGELOG.md`** — Historical references updated.

#### Chat Context — Pruning Fix & Rename

- **`src/Chat/ChatContext.py`** — Fixed `_prune_chat_context`: the delete
  call previously wiped **all** entries for the conversation instead of
  only the entries that were summarized.  Now deletes exactly the
  `prune_batch` oldest IDs that were compressed into the summary.
- **`src/Configuration/Config_RAGChat.py`**, **`src/Globals/Session.py`**,
  **`src/Chat/QueryParts.py`**, **`src/Api/ChatCompletionHandler.py`** —
  Renamed `batch_size` → `prune_batch` across all strategy profiles,
  session attribute, help text, regex, display, and strategy loader to
  avoid confusion with ML batch sizes.
- **`ARCHITECTURE.md`** — New "Chat Context" section documenting
  multi-turn memory storage, retrieval, incremental pruning algorithm,
  and the `turns` / `prune_batch` knobs.
- **`README.md`** — Updated parameter table and added cross-reference
  to the new architecture section.

---

## [v0.2.1] — 2026-04-13

### ➕ Added

#### Document-Specific Chunkers & AUTO Routing

- **`src/Strategies/Chunkers/HeadingChunker.py`** — New chunker that splits `.doc`,
  `.docx`, and `.md` files on heading boundaries.
- **`src/Strategies/Chunkers/SlideChunker.py`** — New chunker for `.pptx` / `.ppt`
  files, splitting on slide boundaries.
- **`src/Strategies/Chunkers/SlidingWindowChunker.py`** — New chunker with
  configurable sentence overlap (`OVERLAP_SENTENCES`) between consecutive chunks.
- **`src/Strategies/Chunkers/SentenceWindowChunker.py`** — New chunker that packs
  sentences up to `MAX_CHUNK_SIZE` words per chunk.
- **`src/Configuration/Config_Global.py`** — Added `_CHUNK_STRATEGY` routing table:
  when `_ACTIVE_CHUNKER_CONFIG = "DETAILED"`, each file extension is routed to the most appropriate
  chunker (e.g. `.pdf` → SEMANTIC, `.docx` → HEADING, `.pptx` → SLIDE,
  `.txt` → SLIDING_WINDOW).  Added `_CHUNKERS` entries for all new chunker types.
- **`src/Configuration/Config_Global.py`** — Added
  `_CHROMA_EMBED_AND_RETRIEVE_PARAMS` with `THOROUGH` (512 HNSW neighbours) and
  `COMPACT` (64 HNSW neighbours) variants for tuning recall vs. speed trade-off.

### 🔧 Changed

#### Strategy Rename — MEDIUM → BALANCED_FILE_CAP

- **`src/Configuration/Config_RAGChat.py`** — Strategy `MEDIUM` renamed to
  `BALANCED_FILE_CAP` in `_ALLOWED_STRATEGIES`, `_STRATEGIES` dict key, and
  strategy comment.  Default `_ACTIVE_CHUNK_SELECT_STRATEGY` changed to `WIDE`.
  Fixed `filelim` comment: value is a per-file chunk cap, not a file count limit.
- **`src/Strategies/HomeBrewChunkSelector.py`** — `MediumSelector` class renamed to
  `PerFileCapSelector`; all debug labels and `get_selector()` routing updated.
- **`src/Chat/QueryParts.py`** — Help text, examples, and quick-defaults updated
  from `medium` to `balanced_file_cap`.
- **`src/Chat/CommandProcessor.py`** — Fallback strategy list updated.
- **`src/Chat/Chatter.py`** — Allowed-strategy hint in no-results messages updated.
- **`src/Api/ChatCompletionHandler.py`** — `_ALLOWED_STRATEGIES` frozenset and
  fallback default updated.
- **`src/Apps/RAGChat.py`** — Fallback default updated.
- **`tests/test_chat_completion_handler.py`** — Strategy loop updated.
- **`README.md`** — Strategy code example, strategy table (including corrected
  `filelim` value 4 → 40), and `filelim` parameter description fixed.
- **`ARCHITECTURE.md`** — Selector table updated.
- **`HANDS_ON_TOUR.md`** — All interactive examples updated.
- **`CHANGELOG.md`** — Historical class-name and strategy-list references updated.

### 🐛 Fixed

#### WordNet "18" → leet-decode "ib" False Positive

- **`src/Configuration/Config_Global.py`** — Added `"18"` to `_WORDNET.STOPLIST`.
  `"under 18"` (banned phrase) produced WordNet synonym `"18"`, which
  leet-decode normalised to `"ib"` (`1→i`, `8→b`), triggering false positives.

#### SemanticChunker — Oversized Single-Sentence Segments

- **`src/Strategies/Chunkers/SemanticChunker.py`** — Single-sentence early-return
  path now calls `_split_oversized()` when the sentence exceeds `_max_chunk_size`,
  preventing oversized chunks from passing through unsplit.

#### MIN_SENTENCE_WORDS Documentation

- **`ARCHITECTURE.md`** — Updated SemanticChunker row, added short-fragment
  consolidation subsection documenting `_CHUNKERS.SEMANTIC.MIN_SENTENCE_WORDS`.
- **`README.md`** — Added `MIN_SENTENCE_WORDS` mention in chunking paragraph.

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
  and all concrete selectors (`ScoreRankedSelector`, `PerFileCapSelector`,
  `SingleDocumentSelector`) now accept `session: Session` as a constructor parameter instead
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

## [Released] — 2026-04-01

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
  `RAG_CHAT_SERVICE_LISTENER` (default `127.0.0.1` and `0.0.0.0` in docker), `RAG_CHAT_SERVICE_LISTENER_PORT` (default `11435`),
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
  - Implementation selectors: `_ACTIVE_LLM`, `_ACTIVE_EMBED`, `_ACTIVE_CROSS`, `_ACTIVE_OLLAMA`, `_ACTIVE_OPENWEBUI`,
    `_ACTIVE_LLM_CHK`.
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
  `Helpers.get_model_args("_OLLAMA")` (resolved via `_ACTIVE_OLLAMA`).
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

- `HomeBrewChunkSelector`, `ScoreRankedSelector`, `PerFileCapSelector`,
  `SingleDocumentSelector` — all now accept `session: Session` in `__init__()` and store it
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
  *(Replaced by `_LANGUAGE_DETECTION.MIN_WORDS` in a later release.)*
- **`LANG_DETECT_MIN_CONFIDENCE`** (`_ARGOS_DEFINITIONS` in `Config_Global.py`) —
  minimum `langdetect` probability required to accept the top-ranked language.
  If the top result falls below this threshold the detected language is discarded and
  `"en"` is used as fallback. Prevents short, ambiguous queries (e.g.
  `"tell me about llama"`) from being misclassified as a non-English language and
  incorrectly triggering translation warnings. Default: `0.90`.
  *(Replaced by `_LANGUAGE_DETECTION.MIN_CONFIDENCE` in a later release.)*
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
  retrieval strategies (NARROW, BALANCED_FILE_CAP, WIDE, ULTRA_WIDE), optional cross-encoder
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
  outputs. See [ARCHITECTURE.md](ARCHITECTURE.md#-consensus-scoring--experimentation).

#### 🛡️ Compliance & Governance (Technical)

- Per-application detection pipelines (RAGLoad, RAGChat, DocClassify) with configurable
  PIPELINE_CHECK and PROMPT_CHECK stages. See [ARCHITECTURE.md](ARCHITECTURE.md#compliance-chain).
- Model license consent tracking with metadata recording. See [LEGAL.md](LEGAL.md).
- Hugging Face model download flow with local-first resolution and explicit consent when
  downloads are required. See [ARCHITECTURE.md](ARCHITECTURE.md#-hf-model-downloading--caching).
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
