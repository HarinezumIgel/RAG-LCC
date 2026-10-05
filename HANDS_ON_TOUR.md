# Hands-On Tour

> ⚠️ Experimental Walkthrough
>
> This tour demonstrates commands and observed behavior in a lab setup.
> Output can vary by model, config, data, and hardware.
> Nothing here guarantees correctness or safety.

This guide is a practical travel route through the most important RAG-LCC operations:

1. Start with defaults
2. Learn the core in-chat commands
3. Pin retrieval to one file/path/metadata
4. Compare retrieval strategies
5. Compare orchestration flows
6. Add web retrieval controls
7. Load and use a second collection

---

## 0) Before You Start

### Check CLI arguments

```Windows
python ./src/Apps/RAGLoad.py -h
python ./src/Apps/RAGChat.py -h
python ./src/Apps/DocClassify.py -h
```

### Know the default baseline

- Default collection name comes from `COLLECTION` in `src/Configuration/Config_Load_Retrievers.py`.
- Default retrieval strategy comes from `_ACTIVE_CHUNK_SELECT_STRATEGY` in `src/Configuration/Config_RAGChat_Strategies.py`.
- Default orchestration flow comes from `_DEFAULT_ORCHESTRATION_FLOW` in `src/Configuration/Config_Orchestrator.py`.
- Web retrieval is globally gated by `WEB_SEARCH_MODE` in `src/Configuration/Config_Internet_Env.py`.

### Load a starter corpus

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs
```

---

## 1) Start with Defaults

Launch chat:

```Windows
python ./src/Apps/RAGChat.py
```

Inside chat:

```text
🛠️  > show?
```

This prints the active session state, including strategy, flow, filters, and web mode.

Run a baseline query:

```text
💬 Your actual query> where do hedgehogs live?
```

Run one follow-up query:

```text
💬 Your actual query> what do they eat?
```

With default flow/strategy, rewrite and context handling are active. Keep this baseline in mind for later comparisons.

### Check confidence level and confidence log

After each answer, RAGChat can append an `Answer confidence` block with a level
(`HIGH`, `MEDIUM`, `LOW`) and a final score (`C_final` in `[0,1]`).

To keep this visible and persisted, check `src/Configuration/Config_Global.py`:

- `_CONFIDENCE_LOGGING["enabled"] = True`
- `_CONFIDENCE_LOGGING["emit_pretty"] = True`
- `_CONFIDENCE_LOGGING["csv_enabled"] = True`

With defaults, one CSV file per run is written under:

- `logs/RAGChat/RAGChat_CONFIDENCE_YYYYMMDD_HHMMSS.csv`

Quick check (PowerShell):

```Windows
Get-ChildItem .\logs\RAGChat\*CONFIDENCE*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 3
```

---

## 2) Core Command Pattern (Cheat Sheet)

RAGChat command grammar:

- `key=value` sets a value
- `key!` opens a picker
- `key?` shows current value
- `key-` unsets/clears (where supported)
- `show?` prints all current session values
- `help?` prints command help

Most-used keys in this tour:

- `collection`, `chat_name`
- `strategy`, `orchestrator_flow`, `force_retrieve_mode`
- `file`, `path`, `metadata`
- `web_search`, `web_weight`, `fetch_page_content`
- `mark_text`, `debug`

---

## 3) File Pinning and Scoped Retrieval

This section shows how to narrow retrieval intentionally.

### Pin to a single file

```text
🛠️  > file!
```

Pick a file (example: `Dogs.png`), then query:

```text
💬 Your actual query> summarize only this source
```

You can also set directly:

```text
🛠️  > file=Dogs.png
```

Clear file pin:

```text
🛠️  > file-
```

### Pin to a path

```text
🛠️  > path!
```

Choose a folder/file path from history or set directly.

Clear path pin:

```text
🛠️  > path-
```

### Apply metadata filter

Interactive picker:

```text
🛠️  > metadata!
```

Direct assignment example:

```text
🛠️  > metadata=Language:English
```

Clear metadata filters:

```text
🛠️  > metadata-
```

---

## 4) Compare Retrieval Strategies

Use the same question repeatedly while changing strategy.

### DEFAULT

```text
🛠️  > strategy=DEFAULT
💬 Your actual query> Which animals are discussed in the collection?
```

### NARROW

```text
🛠️  > strategy=NARROW
💬 Your actual query> Which animals are discussed in the collection?
```

### ULTRA_WIDE

```text
🛠️  > strategy=ULTRA_WIDE
💬 Your actual query> Which animals are discussed in the collection?
```

What to watch:

- amount of retrieved context
- precision vs recall in final answer
- speed and verbosity in diagnostics

Tip: `strategy!` opens a picker. `strategy*DEFAULT` applies preset defaults quickly.

---

## 5) Compare Orchestration Flows

Flow profiles change how retrieval stages are orchestrated.

### Inspect current flow

```text
🛠️  > orchestrator_flow?
```

### Pick another flow

```text
🛠️  > orchestrator_flow!
```

Try at least these:

- `THOROUGH_QUERY_REWRITE`
- `TRANSLATION_FOCUSED`
- `ORIGINAL_LANGUAGE_VECTOR_ONLY`
- `ORIGINAL_LANGUAGE_ALL_RETRIEVERS`

Then run the same query for each flow:

```text
💬 Your actual query> Was fressen Igel?
```

Optional advanced override:

```text
🛠️  > force_retrieve_mode=VECTOR
```

Clear override back to flow behavior:

```text
🛠️  > force_retrieve_mode-
```

Tip: `orchestrator_flow*THOROUGH_QUERY_REWRITE` resets to baseline flow defaults.

---

## 6) Web Retrieval Controls (Global + Session)

Web behavior is controlled at two levels:

1. Global master switch: `WEB_SEARCH_MODE` in `src/Configuration/Config_Internet_Env.py`
2. Session knobs in chat

If web is globally enabled, test session modes:

```text
🛠️  > web_search=local_only
🛠️  > web_search=local_and_web
🛠️  > web_search=web_only
```

Adjust web influence in fusion:

```text
🛠️  > web_weight=0.5
```

Choose snippet-only or full-page mode:

```text
🛠️  > fetch_page_content!
```

Then query:

```text
💬 Your actual query> latest hedgehog habitat reports
```

---

## 7) Work with a Second Collection

Now create a second knowledge base and switch between both.

### Step A: Set second collection name

Edit `src/Configuration/Config_Load_Retrievers.py` and set:

```python
COLLECTION = "Test_Second"
```

### Step B: Load documents into second collection

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs
```

(Use another document folder if you want the two collections to differ.)

### Step C: Switch collections in chat

```text
🛠️  > collection!
```

Pick between your original collection (for example `Test`) and `Test_Second`.

Run the same query after each switch:

```text
💬 Your actual query> What do hedgehogs eat?
```

What to watch:

- collection switch resets scoped filters (file/path/metadata)
- answer differences between corpora
- chat context isolation by collection + chat name

---

## 8) Chat Names and Context Isolation

Create a separate chat lane:

```text
🛠️  > chat_name=TourFlowA
```

Switch to another:

```text
🛠️  > chat_name=TourFlowB
```

Use `chat_name!` to pick existing sessions.

This is useful when testing different strategies/flows without cross-contaminating context.

---

## 9) Visual Grounding Quick Pass

Enable source marking:

```text
🛠️  > mark_text=true
```

Query:

```text
💬 Your actual query> where do hedgehogs live?
```

Open a marked source from the prompt. Then disable:

```text
🛠️  > mark_text=false
```

---

## 10) Return to Baseline Defaults

Use this reset sequence at the end of experiments:

```text
🛠️  > strategy*DEFAULT
🛠️  > orchestrator_flow*THOROUGH_QUERY_REWRITE
🛠️  > force_retrieve_mode-
🛠️  > file-
🛠️  > path-
🛠️  > metadata-
🛠️  > web_search=local_only
🛠️  > mark_text=false
🛠️  > show?
```

Important:

- `strategy*DEFAULT` resets strategy slots (weights, thresholds, limits).
- `orchestrator_flow*THOROUGH_QUERY_REWRITE` resets flow profile slots (query source, guardrail/rewrite/rerank/retriever switches).
- These defaults do **not** reset session web knobs: `web_search` and `fetch_page_content`.
- Keep `web_search=local_only` in the reset path for a local-only baseline.
- If you previously used `web_search=web_only`, setting `web_search=local_only` also clears the implicit `retrieve_mode=WEB`.

You are now back at a clean default-like session state.

---

## 11) Document Classification -> Filtered Load

This is the classify-then-load route from the original workflow: classify first,
then ingest only the subset you want.

### Step A: Run document classification

```Windows
python ./src/Apps/DocClassify.py --doc-dir TestDocs
```

At the end of the run, note the generated OK CSV path (example format):

- `logs/DocClassify_OK_YYYYMMDD_HHMMSS.csv`

### Step B: Choose a target collection for filtered ingestion

Edit `src/Configuration/Config_Load_Retrievers.py` and set a dedicated target
collection, for example:

```python
COLLECTION = "Test_Classified_Filtered"
```

### Step B2: Ingest everything vs only changed files

RAGLoad can either reprocess everything or skip unchanged files based on file hash.

Control slot:

- `src/Configuration/Config_RAGLoad.py`
- `PROCESS_IF_UNCHANGED`

Set behavior:

```python
PROCESS_IF_UNCHANGED = True   # process all files, even when hash is unchanged
PROCESS_IF_UNCHANGED = False  # process only changed/new files (incremental mode)
```

Notes:

- RAGLoad CLI flag: `--process-if-unchanged true|false`
- This flag is exposed for RAGLoad only (from `Config_RAGLoad.py`).

### Step C: Load only selected rows from the classification CSV

Use `--load-from-classify-csv` with `--classify-csv-query` (SQLite WHERE syntax).

Animal-focused example:

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs --load-from-classify-csv logs/DocClassify_OK_YYYYMMDD_HHMMSS.csv --classify-csv-query "Animal LIKE '%hedgehog%' OR Animal LIKE '%cat%'"
```

Language-focused example:

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs --load-from-classify-csv logs/DocClassify_OK_YYYYMMDD_HHMMSS.csv --classify-csv-query "Language = 'English'"
```

Combined filter example:

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs --load-from-classify-csv logs/DocClassify_OK_YYYYMMDD_HHMMSS.csv --classify-csv-query "Mammal LIKE '%Yes%' AND Language = 'English'"
```

### Step D: Validate in chat

```Windows
python ./src/Apps/RAGChat.py
```

```text
🛠️  > collection=Test_Classified_Filtered
💬 Your actual query> Which animals are available in this collection?
💬 Your actual query> Answer only from English sources.
```

If the query returns fewer or different documents than the unfiltered collection,
the classify filter is working as intended.

### Step E: Change classification prompts (where to look)

If you want to customize what DocClassify extracts, start here:

Main prompt templates (classification output prompt):

- `src/Configuration/Config_DocClassify.py`
- `_PROMPT_CLASSIFY_MISTRAL`
- `_PROMPT_CLASSIFY_LLAMA`

Output schema keys (must stay aligned with prompt fields):

- `src/Configuration/Config_DocClassify.py`
- `_YOUR_CLASSIFICATION_KEYS`
- `_CLASSIFICATION_KEYS`
- `CLASSIFICATION_WORD_CNT`
- `SUMMARY_SENTENCE_CNT`

Which model/prompt mapping is used for classification:

- `src/Configuration/Config_Models.py`
- `_ACTIVE_LLM`
- `_MODELS[...]["_LLM"]["PROMPT_CLASSIFY"]`

Prompt-check templates for compliance validation (classification stage):

- `src/Configuration/Config_Banned_Prompts.py`
- `_PROMPT_CHECK_CLASSIFY_MISTRAL`
- `_PROMPT_CHECK_CLASSIFY_LLAMA_GUARD`
- mapped via `_MODELS[...]["_LLM_CHK"]["PROMPT_CLASSIFY"]` in `src/Configuration/Config_Models.py`

Minimal rule of thumb:

- When you add/remove classification fields, update both prompt text and `_YOUR_CLASSIFICATION_KEYS` so JSON keys and parser expectations stay in sync.

### Step F: Add banned words and tune detector strictness

Simple banned-word example:

1. Open `src/Configuration/Config_Banned_Content.py`.
2. Add your terms to `_STRICT_BANNED["BANNED"]`.

```python
_STRICT_BANNED = {
    "BANNED": [
        # existing entries...
        "customer internal token",
        "project pegasus secret",
    ],
}
```

Tune "how many algorithms must agree" in `src/Configuration/Config_Banned_Detection.py`.
For RAGChat prompt checks, use the `RAGChat -> PROMPT_CHECK -> PIPELINE` block:

- `REQUIRED_ALGOS_ABOVE_THRESHOLD`: how many algorithms must exceed their threshold.
- `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE`: how many different algorithms must fire (non-zero score).

Typical (current) chat-time values are:

```python
"REQUIRED_ALGOS_ABOVE_THRESHOLD": 2,
"REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 3,
```

Extreme scenario (fires often, for demonstration):

```python
"REQUIRED_ALGOS_ABOVE_THRESHOLD": 1,
"REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 1,
"ALGOS_TO_PROCESS": {
    "Regex": True,
    "Jaccard": True,
    "BM25": True,
    "Keybert": True,
},
```

Impact of this extreme setup:

- detector triggers much more often (high recall, low precision)
- many benign passages may be masked/blocked (false positives increase)
- chat and load runs can feel noisier because more content is flagged

Use this only as a temporary experiment, then restore stricter values.

---

## 12) Example Matrix (High-Insight Tests)

Use these short experiments when you want to understand *why* behavior changed,
not only *that* it changed.

1. Strategy A/B with fixed questions
Change: switch `strategy` across `DEFAULT`, `NARROW`, `ULTRA_WIDE`.
Run: ask the same 5 questions each time.
Observe: trade-off between precision and recall, plus answer confidence changes.

2. Flow comparison on multilingual prompts
Change: switch `orchestrator_flow` between `THOROUGH_QUERY_REWRITE`, `TRANSLATION_FOCUSED`, and `ORIGINAL_LANGUAGE_VECTOR_ONLY`.
Run: use mixed-language prompts like `Was fressen Igel?` and an English follow-up.
Observe: rewrite behavior, retrieval source changes, and grounding differences.

3. Local vs web retrieval modes
Change: set `web_search` to `local_only`, `local_and_web`, and `web_only`; vary `web_weight`.
Run: one query with strong local coverage and one with weak local coverage.
Observe: fusion impact, ranking shifts, and when web context dominates.

4. Incremental ingestion impact
Change: toggle `PROCESS_IF_UNCHANGED` between `True` and `False`.
Run: ingest once, modify one file, ingest again.
Observe: processing time and number of reprocessed documents.

5. Classify-then-load curation impact
Change: load from the same DocClassify CSV with different `--classify-csv-query` filters.
Run: compare `Language = 'English'` vs a domain-specific filter.
Observe: how collection scope changes answer content and confidence.

6. Confidence calibration mini-check
Change: keep config fixed, but choose 3 easy, 3 ambiguous, and 3 out-of-domain questions.
Run: inspect the answer confidence block and the confidence CSV rows.
Observe: whether `HIGH/MEDIUM/LOW` aligns with actual answer quality.

7. Banned detector sensitivity sweep
Change: compare current thresholds with an extreme setup (`REQUIRED_ALGOS_ABOVE_THRESHOLD=1`, `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE=1`).
Run: test the same benign and borderline prompts in both modes.
Observe: false-positive growth, masking frequency, and usability impact.

8. Pinning vs unpinned retrieval
Change: run once with no pins, then with `file=...` and `metadata=...`.
Run: ask the same question all three times.
Observe: source narrowing effects and confidence stability.

---

## Optional Next Step

For more end-to-end examples and slot-level details, see:

- [EXAMPLES.md](EXAMPLES.md)
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)
