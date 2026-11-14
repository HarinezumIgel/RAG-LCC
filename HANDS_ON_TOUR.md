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
RAGChat repeats your original query in yellow immediately before each answer,
regardless of the debug level, even when retrieval rewrites or translates it.

If no relevant information or supporting evidence is found, the hint suggests:

> Try starting a new topic with `new:`, switching to a different collection, or adjusting your session parameters.

For example, start a fresh topic without carrying over the previous conversation:

```text
💬 Your actual query> new: where do hedgehogs live?
```

To try another collection or change session parameters, return to settings with
`b:` and use `collection!` or `help?`. In OpenWebUI, select another collection
from the model selector and review session parameters in **Controls**.

### Check confidence level and confidence log

After each answer, RAGChat can append an `Answer confidence` block with a level
(`HIGH`, `MEDIUM`, `LOW`) and a final score (`C_final` in `[0,1]`).

To configure logging, check
[Config_Confidence.py](src/Configuration/Config_Confidence.py):

- `_CONFIDENCE_LOGGING["enabled"] = True`
- `_CONFIDENCE_LOGGING["emit_pretty"] = True`
- `_CONFIDENCE_LOGGING["csv_enabled"] = True`

With defaults, one CSV file per run is written under:

- `logs/RAGChat/RAGChat_CONFIDENCE_FLAT_YYYYMMDD_HHMMSS.csv`

The answer block has two sections:

- `Heuristic Confidence:` contains retrieval evidence metrics and `C_final`.
- `Quality Dimensions (judge model: <model>):` contains aligned accuracy,
  completeness, reasoning, and provenance scores. The optional judge uses the
  answer-generation model and is enabled in the current defaults.

Set `_QUALITY_CONFIDENCE["llm_judge"]["enabled"] = False` to avoid the extra
LLM call while retaining heuristics, or set
`_QUALITY_CONFIDENCE["emit_in_answer_block"] = False` to hide the quality section.
Set `_QUALITY_CONFIDENCE["enabled"] = False` to disable quality evaluation.
Quality scores do not change `C_final`; neither score is a correctness guarantee.

The CSV stores one turn per row, with individual metric/result columns rather
than JSON blobs. Filter `step.<name>.<occurrence>.status` to find retries and
fallbacks, compare `quality_<dimension>_heuristic`, `quality_<dimension>_llm`, and
`quality_<dimension>_avg`, and inspect `quality_mode`, `quality_error`, and
`quality_judge_model`. In `heuristic_only` mode the `llm` field mirrors the
heuristic value; it does not represent a judge verdict.
Older `CONFIDENCE` files are retained but not converted: truncated JSON and list
placeholders in those files cannot recover their missing data.

Quick check (PowerShell):

```Windows
Get-ChildItem .\logs\RAGChat\*CONFIDENCE_FLAT*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 3
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
- `web_search`, `web_weight`, `web_fetch_k`, `fetch_page_content`
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

### Summarize an indexed document

`summary:` is an alias for `summarize:` in both CLI RAGChat and OpenWebUI.
For example:

```text
summary: document.pdf
summary: "D:\documents\document.pdf"
summary:
summary: cancel
```

Both command names follow the same selection and cancellation rules below.

In CLI RAGChat, use `summarize:` to summarize the pinned file. If no file is
pinned, RAGChat opens the same path picker as `path!` directly. Selecting a
document pins it and continues the summary. Choose **Return - keep current
setting** to leave the selection unchanged and skip the summary.
All settings pickers offer this return option; returning never clears or
resets a setting. For typed settings, submit an empty value to keep the current
value. Use `key-` explicitly when you want to unset a setting.
Alternatively, use `summarize: document.pdf` or
`summarize: "D:\documents\document.pdf"` for a one-turn selection.
The argument is detected automatically: a bare filename filters `FileName`;
a document path filters both `FileName` and `FilePath`. Spaces are accepted
with or without surrounding quotes. Only one matching pair of outer double
quotes (`"..."`) or single quotes (`'...'`) is removed; unmatched quotes are
preserved. An explicit argument overrides the pinned
selection for this turn without changing it.
In OpenWebUI, set `file` (indexed filename) or `path` (indexed document path)
in Advanced Parameters to pin a document. These parameters restrict ordinary
chat retrieval; they do **not** trigger summarization. Enter `summarize:` without
an argument to summarize the pin, or `summarize: filename/path` to override it
for one summary only. With no pin, RAGChatService uses indexed documents cited
in the previous answer when grounded sources are available. If there are several,
it presents a numbered list with a 👉 pick icon beside each file/path; reply with a number to select, or `cancel`/`0` to
skip summarization. Any other message continues the chat without a queued
summary.
Set only one of `file` and `path`; an empty string clears the pin. OpenWebUI
must resend these parameters with each request because service sessions are
request-local. Terminal settings navigation (`b` / `b:`) and native pickers are
unavailable in OpenWebUI.
RAGChat shows the pinned document before the query prompt. Both RAGChat and
RAGChatService confirm the resolved document at the start of each summary response;
the API cannot add a pre-submission notice to the OpenWebUI query editor.
When chat context is enabled, successful summaries are stored in conversation
history with the `summarize:` command and file name/path (including the resolved
path when using a pinned selection), together with the summary response.
If literal path lookup fails, summarization checks same-name indexed files using
normalized path separators (and case for Windows drive/UNC paths). It never
falls back to a different directory; stored metadata remains unchanged.
Summarization reads **all indexed chunks** matching the filename and, when
specified, full path: no retrieval top-k, score threshold, reranking, web
search, or additional metadata/language filter is applied. Ambiguous filenames
require a full path. Only indexed content is available; reload the document if
its index is incomplete or outdated.
The command skips user-prompt checks. Generated summaries still undergo answer
compliance checks, grounding against the selected document, and the configured
confidence/quality evaluation. Unsupported summaries are replaced with a
no-evidence message. The confidence block explicitly notes that unranked
document retrieval has no relevance scores; its heuristic score is not a
probability of factual accuracy.

[Config_ContextSummarizer.py](src/Configuration/Config_ContextSummarizer.py) configures
overlap in **whole indexed chunks** (default 5), prompts, and one shared
`SUMMARY_TEMPERATURE` for chunk, reduction, and final summaries (default 0.3).
The `SUMMARY_WORD_LIMITS` map in
[Config_Grounder.py](src/Configuration/Config_Grounder.py) resolves each preset to
separate maximum requested word counts for intermediate (chunk-batch and recursive
reduction) summaries and the final document summary:

| Preset | Intermediate words per call | Final words |
| --- | ---: | ---: |
| `short` | 150 | 400 |
| `medium` | 300 | 1000 |
| `long` | 600 | 2000 |
| `very_long` | 1000 | 4000 |

Both budgets must be positive integers. The summarizer substitutes the applicable
value for `{max_words}` in the prompt templates; custom templates without that
placeholder receive an appended word-limit instruction. These are LLM instructions,
not enforced word counts or token ceilings; existing model/context token limits
still apply, and output is not truncated or retried to enforce a word count.

Summary source-document highlighting defaults to matching support passages only
(orange), rather than highlighting every input chunk. In
[Config_Grounder.py](src/Configuration/Config_Grounder.py), set
`_GROUNDING_MARKING["summary_support_only"] = False` to restore yellow highlights
for all input chunks alongside orange support passages. With the default `True`,
no marked document is produced when no matching local support passages are found.
This setting does not change normal question-answer highlighting or grounding checks.

The **Summarize settings** section displays effective values. Set
`summary_length=short|medium|long|very_long` (default `medium`),
`summary_temperature=0.3` (0 to 1), or `chunk_overlap=5` (non-negative integer).
Use `summary_length!` to pick a detail level, or the temperature/overlap `!`
commands to enter values. Clear any override with `key-` to inherit configuration.
Document summaries automatically use the collection selected in settings (or the
configured default), loading it if necessary; no extra `collection!` selection or
preliminary query is needed.
The same names are supported as API parameters. Detail instructions apply to
intermediate and final summaries; lengths are not guaranteed and remain bounded
by the active model's output/context budget.
Batches retain as much overlap as fits while always advancing. TokenBudget
supplies the active `_ACTIVE_LLM` context limit and system/output reserves;
oversized chunks are split and intermediate summaries are recursively reduced
before the final summary. Every map/reduce/final request counts the full prompt
(instructions and separators included), reserves system and output tokens, and
honors the smaller of backend context metadata, the configured model cap, and
any session override. Summarization uses UTF-8 byte length as a conservative
token bound, rather than relying only on words times 1.3, which undercounts
code, hex dumps, and dense PDF text. This can produce smaller batches than an
exact tokenizer. Text without whitespace is split too; source text is not
truncated. Prompt templates that cannot fit fail before calling the model.
The terminal shows one introductory message followed by progress bars with
completed/total batches for summarization, each reduction pass, and the final
merge. Counts advance after successful model calls; reduction totals are
determined per pass rather than estimated for the entire operation.
The first model call shows its diagnostics before the bar starts. Subsequent
summary calls suppress informational diagnostics so the bars remain readable;
warnings and errors remain visible.

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
New to flows? [ORCHESTRATION_FLOWS.md](ORCHESTRATION_FLOWS.md) explains query
versions, legs and every shipped flow in plain language, and shows which flows
to compare for which question.

### Inspect current flow

```text
🛠️  > orchestrator_flow?
```

### Pick another flow

```text
🛠️  > orchestrator_flow!
```

Try at least these (each one is explained step by step in `src/Configuration/Config_Orchestrator.py`):

- `NATIVE_ALL` (default: stays in the user's language)
- `ENGLISH_ALL` (baseline: translates to English first)
- `ENGLISH_ALL_MULTILEG` (baseline + extra retrieval legs)
- `ENGLISH_ALL_NO_RERANK` (baseline without the reranker)
- `ORIGINAL_VECTOR` (vector search on the untranslated question)

Old names such as `THOROUGH_QUERY_REWRITE` are still accepted and mapped to the new name.

Then run the same query for each flow:

```text
💬 Your actual query> Was fressen Igel?
```

Optional advanced override:

```text
🛠️  > force_retrieve_mode=VECTOR
```

`force_retrieve_mode=VECTOR` runs only the vector retriever for the next turns
(the other local retrievers are switched off). Clear override back to flow behavior:

```text
🛠️  > force_retrieve_mode-
```

Two more flow overrides are available the same way:

```text
🛠️  > retrieval_language=native
🛠️  > rerank_query_source=ORIGINAL_QUERY
```

Tip: `orchestrator_flow*ENGLISH_ALL` resets to baseline flow defaults.

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

Override how many web documents are fetched:

```text
🛠️  > web_fetch_k=10
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

`COLLECTION` is a base name: RAG-LCC appends the active embedder
(`_ACTIVE_EMBED` in `Config_Models.py`), so with `snowflake` the data lands in
`Test_Second_snowflake`. A collection only works with the embedder that built
it, so the suffix keeps collections of different embedders apart.

### Step B: Load documents into second collection

```Windows
python ./src/Apps/RAGLoad.py --doc-dir TestDocs
```

(Use another document folder if you want the two collections to differ.)

### Step C: Switch collections in chat

```text
🛠️  > collection!
```

Pick between your original collection (for example `Test_embedder_snowflake`) and
`Test_Second_snowflake`. The picker shows only collections created with the
active embedder; switch `_ACTIVE_EMBED` to see the others.

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
🛠️  > orchestrator_flow*NATIVE_ALL
🛠️  > force_retrieve_mode-
🛠️  > file-
🛠️  > path-
🛠️  > metadata-
🛠️  > web_search=local_only
🛠️  > web_fetch_k-
🛠️  > mark_text=false
🛠️  > show?
```

Important:

- `strategy*DEFAULT` resets strategy slots (weights, thresholds, limits).
- `orchestrator_flow*NATIVE_ALL` resets flow profile slots (query sources, retrieval language, secondary leg/rewrite/rerank/retriever switches) to the default flow.
- These defaults do **not** reset session web knobs: `web_search`, `web_fetch_k`, and `fetch_page_content`.
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

Important startup behavior:

- Both `PROMPT_CHECK` and `PIPELINE_CHECK` have a `Check` gate.
- When a stage has `Check=False`, startup skips consensus-threshold validation for that stage.
- When a stage has `Check=True`, startup enforces: `REQUIRED_ALGOS_ABOVE_THRESHOLD >= 1`, `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE >= 1`, and enabled algorithms in `ALGOS_TO_PROCESS` must not be fewer than either required count.

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
Change: switch `orchestrator_flow` between `NATIVE_ALL`, `ENGLISH_ALL`, `ENGLISH_ALL_MULTILEG`, and `ORIGINAL_VECTOR`.
Run: use mixed-language prompts like `Was fressen Igel?` and an English follow-up.
Observe: rewrite behavior, retrieval source changes, and grounding differences.
More flow pairs and what each comparison measures: [How to experiment](ORCHESTRATION_FLOWS.md#how-to-experiment).

3. Local vs web retrieval modes
Change: set `web_search` to `local_only`, `local_and_web`, and `web_only`; vary `web_weight` and `web_fetch_k`.
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
