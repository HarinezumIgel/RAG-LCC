# Architecture Overview

> **Lab / Research Use Only.** See [Lab Disclaimer](#-lab-disclaimer) below and [LEGAL.md](LEGAL.md) for governance guidance.

## 🔬 Lab Disclaimer

This software is a **research and experimental laboratory tool**. It is not a certified compliance product, a legal instrument, or a substitute for qualified legal, security, or regulatory advice.

- Detection algorithms, thresholds, and consensus rules are configurable and must be tuned by the operator for their specific context. Default values are illustrative starting points only.
- No automated system can guarantee detection of all policy violations, data leaks, or compliance issues. False negatives (missed violations) and false positives (incorrect flags) will occur.
- The "[compliance](#-definition-compliance-rag-lcc)" terminology used throughout this codebase refers to a configurable keyword/phrase detection pipeline, not to any legal standard or certification.
- The operator is solely responsible for validating that the tool behaves correctly for their use case, for reviewing all outputs, and for any decisions made based on those outputs.
- This tool does not provide legal advice, does not constitute a Data Protection Impact Assessment (DPIA), and does not satisfy any regulatory obligation on its own.
- **This tool is not intended for production use.** It is a lab and research environment only. There is no hardening, no security audit, no SLA, and no support commitment.
- Use is entirely at the operator's own risk.

## 📖 Definition: Compliance (RAG-LCC)

Compliance in the context of RAG‑LCC means the configurable, algorithmic process used to detect, flag, and optionally block or redact content according to operator‑supplied keyword lists, thresholds, and consensus rules. Compliance checks are a technical detection pipeline only and do not by themselves establish legal, regulatory, or contractual compliance. Final decisions about enforcement, disclosure, or remediation must be made by a qualified human operator.

Key points

Detection only: RAG‑LCC performs automated detection and scoring; it does not make legal determinations.

Operator control: Keyword lists, thresholds, and consensus rules are supplied and tunable by the operator.

Human review required: Any blocking, redaction, or external disclosure requires human review and recorded sign‑off.

Scope: Applies to document ingestion, prompt validation, and LLM output validation as configured in Configuration/Config_*.py.

Limitations: Detection is probabilistic; false positives and false negatives will occur.

## 🏗️ System Design

RAG-LCC follows a modular, configuration-driven architecture intended for laboratory and research use. The system provides a tunable pipeline for experimenting with LLM parameters, detection algorithm combinations, and keyword-based filter chains.

Three applications share the same core infrastructure:

- **RAGLoad** — ingests documents into a ChromaDB vector store; applies text masking and optional compliance checks during ingestion.
- **RAGChat** — retrieves relevant chunks from the store and answers queries via an LLM; applies compliance checks to both the prompt and the LLM response.
- **DocClassify** — classifies documents in a directory using keyword extraction, stemming, and optional LLM-assisted label generation; writes results to CSV.

## 🎯 Core Design Principles

1. **Configuration-Driven**: All behavior parameterized via Python config files (`Config_*.py`) with CLI override support
2. **Offline-First**: Designed to operate locally; actual behavior depends on configuration and third‑party components
3. **User Responsibility**: Framework enables; users verify compliance and appropriateness
4. **Explicit Over Hidden**: No automatic assumptions about behavior
5. **Singleton Patterns**: Stateful components use singletons to maintain consistency
6. **Modular Detection**: Multiple algorithms with consensus scoring for compliance checks

## 🧩 Component Hierarchy

```text
├── Applications (Entry Points)
│   ├── RAGLoad       - Document ingestion pipeline
│   ├── RAGChat       - Interactive retrieval + conversation
│   └── DocClassify   - Batch document classification
│
├── Core Systems
│   ├── Configuration  - Parameter loading and validation
│   ├── Compliance     - License management and verification
│   ├── Globals        - Shared state (logging, counters)
│   └── Session        - User session context
│
├── Processing Pipelines
│   ├── Pipeline       - Orchestration (LoadAndClassifyProcessor)
│   ├── Strategies     - Processing strategies + classification/chunking helpers
│   ├── Load           - (reserved for future document-loading extensions)
│   └── Chat           - Conversation and query handling
│
├── Detection & Algorithms
│   ├── Algos          - Detection algorithms (regex, Jaccard, cosine, KeyBERT, Levenshtein, BM25)
│   │                    + ReverseStemmer — stem → original word lookup for classification output
│   ├── Compliance     - Compliance rule application
│   └── Shared Helpers - Common detection utilities
│
├── Storage & Models
│   ├── ChromaDBHelper - Vector DB interface
│   ├── AIHelpers      - Model loading and inference
│   └── Model License  - License tracking
│
└── Utilities
    ├── Helpers        - General utilities
    ├── PrettyWriter   - Output formatting
    ├── Informer       - System status reporting
    └── Exceptions     - Error types
```

## 🔀 Data Flow

The following diagrams illustrate the intended data flow for experimentation; actual execution paths may vary depending on configuration, runtime conditions, and errors.

### 📥 RAGLoad Pipeline

```text
Document Input
    ↓
[Format Detection & Conversion]
    ↓
[Text Extraction]
    ↓
[Unicode Normalization]
    ↓
[Text Masking] ← Character replacement for sensitive patterns
    ↓
[Compliance Checks] ← Algos first, then Prompt Check if Algos don't fire
    ↓
[Chunking with Overlap]
    ↓
[Metadata Extraction]
    ↓
[Embedding Generation]
    ↓
[ChromaDB Storage]
```

### 💬 RAGChat Pipeline

```text
User Query (Prompt)
    ↓
[Compliance Checks on Prompt] ← Algos first, then Prompt Check if Algos don't fire
    ↓
[Query Parsing & Preprocessing]
    ↓
[Embedding Generation]
    ↓
[Vector Similarity Search on Masked Documents] ← Works with pre-masked content
    ↓
[Document Retrieval (top-k)]
    ↓
[Cross-Encoder Re-ranking]
    ↓
[LLM Prompt Construction]
    ↓
[Streaming LLM Response Generation]
    ↓
[Compliance Checks on Response] ← Algos applied to LLM output for redaction
    ↓
[Conversation Memory Storage]
    ↓
[User Output]
```

### 🏷️ DocClassify Pipeline

```text
Batch of Documents
    ↓
[Format Detection & Text Extraction]
    ↓
[Unicode Normalization]
    ↓
[Text Masking] ← All downstream processing works on masked content
    ↓
[Algos processing per Document (on Masked Content)]
    ↓
[Threshold Application per Algorithm]
    ↓
[Consensus Scoring]
    ↓
[Final Classification Decision]
    ↓
[Result Aggregation with Algorithm Details]
    ↓
[Reverse Stemming] ← Optional: restore stemmed keywords to original forms (REVERSE_STEMMING=True)
    ↓
[CSV Output]
```

## ⚙️ Configuration System

All default values referenced in this document reflect the state of this repository and are not recommendations or guarantees of suitability.

Configuration is hierarchically resolved:

1. **Defaults** - Default used in this repository
2. **Configuration Files** - `Configuration/Config_*.py` files
3. **Environment Variables** - Override via ENV
4. **CLI Flags** - Command-line argument override (applies to `Config_Global.py` and the app-specific config only; `Config_Models.py` and `Config_Banned.py` are not exposed as CLI flags)

Each application loads from:

- `Config_Global.py` - Common settings
- `Config_Models.py` - Model selections
- Application-specific `Config_*.py`

Parameters are accessible via `Config().get(key_path)` using dot notation.

### 🔑 Key Configuration Areas

| Area | File | Purpose |
| --- | --- | --- |
| Global | `Config_Global.py` | Paths, device, debugging |
| Models | `Config_Models.py` | Embedding, cross-encoder, LLM |
| RAGLoad | `Config_RAGLoad.py` | Chunking, batch sizes |
| RAGChat | `Config_RAGChat.py` | Retrieval thresholds, re-ranking |
| DocClassify | `Config_DocClassify.py` | Classification settings, `REVERSE_STEMMING` |
| Compliance | `Config_Banned.py` | Keyword/phrase lists, detection thresholds |
| Network | `Config_Internet_Env.py` | Network connection, network trace |

### 🔩 Model Implementation Selectors

`Config_Models.py` uses a two-level lookup to resolve model configurations. Five top-level variables select which *implementation* (impl) to use for each model *role*:

```python
_LLM_CHK = "llama_guard"   # llama_guard, llama, mistral
_LLM     = "mistral"       # mistral, llama
_EMBED   = "snowflake"     # snowflake
_CROSS   = "mmarco"        # mmarco
_OLLAMA  = "ollama"        # ollama
```

At runtime the framework resolves a role via `_MODELS[<impl>][<role>]`. For example, with `_LLM = "mistral"` the LLM configuration is read from `_MODELS["mistral"]["_LLM"]`.

To switch models, change the impl value to another key that carries a matching role entry in `_MODELS` (the allowed values are listed in the inline comments above).

### 🧩 Extraction & KeyBERT Variant Configuration

`Config_DocClassify.py` organises the LLM extraction parameters and the
KeyBERT keyword-extraction parameters into **named variants** (`STRICT`,
`BALANCED`, `RECALL`).  Two independent selector keys control which variant
is active at runtime:

| Selector Key | Controls | Default used in this repository |
| --- | --- | --- |
| `_ACTIVE_EXTRACTION_CONFIG` | `_EXTRACTION_MODEL_PARAMS` (temperature, top-k, top-p sent to the extraction LLM) | `"STRICT"` |
| `_ACTIVE_KEYBERT_CONFIG` | `_KEY_BERT` (TOP\_N, n-gram ranges for the two-pass KeyBERT extraction) | `"STRICT"` |

Because the selectors are independent you can mix and match, e.g. use a
`STRICT` extraction LLM with a `BALANCED` KeyBERT pass to widen keyword
coverage while keeping the LLM deterministic.

#### Variant summary

| Variant | Extraction LLM | KeyBERT |
| --- | --- | --- |
| `STRICT` | `temperature=0.0`, `top_k=1`, `top_p=1.0` — near-greedy, STRICT — near-greedy, maximum determinism | `TOP_N_FIRST=60`, `TOP_N_SECOND=30`, `NGRAM_PASS1=(1,4)` — minimal noise |
| `BALANCED` | `temperature=0.0`, `top_k=10`, `top_p=0.85` — small candidate pool, less brittle | `TOP_N_FIRST=80`, `TOP_N_SECOND=50`, `NGRAM_PASS1=(1,5)` — moderate coverage |
| `RECALL` | `temperature=0.1`, `top_k=40`, `top_p=0.92` — wider sampling, still conservative | `TOP_N_FIRST=150`, `TOP_N_SECOND=100`, `NGRAM_PASS1=(1,6)` — maximum recall |

#### Consumer code

At startup each consumer reads the active variant name once and resolves
parameters via dot-notation, e.g.
`Config().get("_EXTRACTION_MODEL_PARAMS.STRICT.TEMPERATURE_EXT")`.

| Consumer | Uses extraction config | Uses KeyBERT config |
| --- | --- | --- |
| `ClassifyStrategy.py` | Yes (`TEMPERATURE_EXT`, `TOP_K_EXT`, `TOP_P_EXT`) | Yes (`TOP_N_FIRST`, `TOP_N_SECOND`) |
| `AIHelpers.py` | No | Yes (`TOP_N_FIRST`, `TOP_N_SECOND`) — falls back to flat layout when no variant is defined (e.g. RAGChat) |
| `ClassifyHelper.py` | No | Yes (`NGRAM_PASS1`, `NGRAM_PASS2`) |

> **Note:** `Config_RAGChat.py` keeps a flat `_KEY_BERT` dict (no variant
> nesting) and does not define either selector key.  `AIHelpers.py`
> detects this automatically and falls back to the flat layout.

## 🛡️ Compliance Chain

### 📦 Processing Layers

RAG-LCC applies checks in three sequential stages:

**Layer 1: Text Masking** (Applied after extraction)

- Character-level masking of configured patterns
- Runs for all document types in all applications
- Downstream processing works on the masked copy of the text
- Does not guarantee removal of all obfuscation techniques

**Layer 2: Algorithm Checks** (Configurable keyword/phrase detection)

- Multiple detection algorithms run according to configuration
- Each has individually configurable thresholds
- Consensus rules determine the combined result
- Results depend entirely on keyword lists and threshold settings supplied by the operator

**Layer 3: Prompt Check** (Optional LLM-based check)

- Sends text to a configured LLM for additional screening
- Only executes when algorithm checks produce no flag (configurable)
- Subject to all limitations of the underlying LLM
- Higher latency; output depends on model behaviour

### 🔄 Execution Sequence by Application

**RAGLoad & DocClassify**:

```text
Extract → Normalize → MASK (masked copy used downstream)
                          ↓
                    Algorithm Checks (fast)
                          ↓
                  Algos detect issue? → YES → Block/Flag
                          ↓ NO
                  Prompt Check (if enabled)
                          ↓
                  Prompt detects issue? → YES → Block/Flag
                          ↓ NO
                  Continue to embedding/classification
```

**RAGChat**:

```text
PROMPT VALIDATION PHASE:
  User Query → Algorithm Checks → Algos Fire? → YES: Block
                                         ↓ NO
                              Prompt Check → Fires? → YES: Block
                                         ↓ NO
                              Proceed with retrieval

RETRIEVAL PHASE:
  Search masked documents → Retrieve → Re-rank

GENERATION & RESPONSE VALIDATION PHASE:
  LLM generates response
                     ↓
          Algorithm Checks on Response ← Output validation layer
                     ↓
          Algos detect issue? → YES: Redact/mask response
                     ↓ NO
              Return response to user
```

### 📝 Design Notes

1. **Normalization then masking**: Extracted text is normalized first, then masked; the masked copy is used downstream.
2. **Algorithms are deterministic**: Results depend on the configured keyword lists and thresholds.
3. **Prompt check is a supplementary step**: It runs only when algorithm checks produce no flag, and is subject to the limitations of the chosen LLM.
4. **Response checking is optional**: Applies algorithm checks to LLM output before returning it to the caller.
5. **No guarantee of completeness**: None of these layers guarantee that all violations will be detected.

### ⚙️ Configuring the Chain

- **Control algorithm strictness**: Adjust individual thresholds in `Config_Banned.py`
- **Set consensus rules**: Configure which algorithms must agree for blocking
- **Enable/disable prompt check**: Balance between latency and coverage
- **Chain is OR-based**: If ANY check detects issue, content is flagged

## 🧮 Detection Algorithm Architecture

Configured algorithms work for compliance checking:

Config slot template used by scorers:

- `f"{compliance_slot}.PIPELINE.<Algo>.<Key>"` where `compliance_slot` is stage/app dependent
- Typical resolved slots include pipeline checks (e.g., `...RAGLoad.PIPELINE_CHECK`, `...RAGChat.PIPELINE_CHECK`) and chat prompt checks (`...RAGChat.PROMPT_CHECK`)

### 🔍 Regex Detector (`RegexScorer.py`)

- Two-step matching: strict word-boundary match is attempted first; if it misses and `FUZZY_REGEX_EVAL_AFTER_HARD` is `True`, a fuzzy anchored pattern match is tried as fallback
- Strict match assigns `SOFT_SCORE_HARD` (default used in this repository 1.0); fuzzy-only match assigns `SOFT_SCORE_FUZZY` (default used in this repository 0.75) — this lets the consensus layer distinguish high-confidence exact hits from approximate ones
- Setting `FUZZY_REGEX_EVAL_AFTER_HARD` to `False` disables the fuzzy step entirely, so only exact word-boundary matches produce a score
- Fast and memory-efficient; suitable for known patterns
- RegexScorer internally calls LevenshteinScorer; their scores and thresholds are merged into a single combined result so that two similar algorithms do not carry disproportionate weight in the consensus vote
- Config slot hints: `Regex.THRESHOLD`, `Regex.THRESHOLD_MIN`, `Regex.SOFT_SCORE_HARD`, `Regex.SOFT_SCORE_FUZZY`, `Regex.FUZZY_REGEX_EVAL_AFTER_HARD`, `Regex.WINDOW_MAX_CHARS`, `Regex.PREFIX_SUFFIX_LEN`, `Regex.SEPARATOR_CLASS`

### 🧩 Jaccard Scorer (`JaccardScorer.py`)

- Token-based similarity (Jaccard index)
- Containment scoring
- Character n-gram similarity
- Good for word-level variations
- Config slot hints: `Jaccard.THRESHOLD`, `Jaccard.THRESHOLD_MIN`, `Jaccard.CHAR_NGRAM_RANGE`

### 📏 Cosine Similarity Detector (`CosineKeyWordDetect.py`)

- SBERT embeddings (Snowflake Arctic-embed default used in this repository)
- Semantic similarity matching
- Captures meaning beyond exact surface forms
- Configurable similarity threshold for experimentation
- CosineScorer and KeyBertScorer produce very similar results. This is why CosineScorer is not activated in the provided example configuration
- Config slot hints: `Cosine.THRESHOLD`, `Cosine.THRESHOLD_MIN`

### 🔑 KeyBERT Detector (`KeyBertWordDetect.py`)

- Keyword extraction using SBERT embeddings
- Deterministic phrase ordering for stable results
- OrderedDict caching for consistent index mapping
- Frozen phrase matrix for batch matching
- Two-pass keyword extraction: broad first pass, refined second pass
- Configurable n-gram ranges
- Config slot hints: `Keybert.THRESHOLD`, `Keybert.THRESHOLD_MIN`, `Keybert.TOP_K`

### 📏 Levenshtein Distance (`LevenshteinScorer.py`)

- Edit distance calculation
- Typo and obfuscation detection
- Captures character-level variations
- Config slot hints: `Regex.Levenshtein.THRESHOLD` (nested under Regex pipeline settings)

### 📊 BM25 Scorer (`BM25Scorer.py`)

- Probabilistic relevance scoring between input text and banned phrases using BM25
- Language-aware banlist preparation with cached token frequencies, IDF, and average phrase length
- Configurable BM25 parameters (`TERM_FREQ_SATURATION`, `LENGTH_NORMALIZATION`)
- Overlap and raw-score gating (`MIN_OVERLAP`, `MIN_RAW_SCORE`) before normalization
- Percentile-based score normalization (`NORM_PERCENTILE`) to produce stable `[0,1]` scores for thresholding
- Config slot hints: `BM25.THRESHOLD`, `BM25.THRESHOLD_MIN`, `BM25.TERM_FREQ_SATURATION`, `BM25.LENGTH_NORMALIZATION`, `BM25.MIN_OVERLAP`, `BM25.MIN_RAW_SCORE`, `BM25.NORM_PERCENTILE`

### 🎭 Masking (`Masker.py`)

- Masking is done at document extraction and, although RAGChat works with the already masked chunks ingested into ChromaDB, also on prompt replies
- The masker is regex-based

### 🔤 Unicode normalisation (`UnicodeNormalizer.py`)

- Leet-speak/confusable normalization is applied on input extraction for a canonical text form.

### 🤝 Consensus Scoring & Experimentation

All configured algorithms run for a given phrase; results are combined using **depth** and **breadth** consensus metrics:

#### Depth: Strength of Individual Detection

**Depth** measures how many distinct algorithms **passed their individual thresholds** for a chunk.

- Each algorithm has a per-algorithm threshold (e.g., `THRESHOLD_REGEX`, `THRESHOLD_JACCARD`)
- A result **counts toward depth** only if `score >= algorithm_threshold`
- Depth is the count of unique algorithms meeting this criterion
- **Use case**: Require higher confidence by mandating multiple independent algorithms agree at strong confidence levels

**Config parameter**: `REQUIRED_ALGOS_ABOVE_THRESHOLD` (e.g., `2` = at least 2 algos must pass their thresholds)

#### Breadth: Coverage of Detection

**Breadth** measures how many distinct algorithms **produced a score above `THRESHOLD_MIN`** for a phrase, regardless of whether that score exceeds the algorithm's primary `THRESHOLD`.

- Each algorithm already discards scores below its `THRESHOLD_MIN` (noise floor) before results reach the consensus stage
- A result **counts toward breadth** if it survived the `THRESHOLD_MIN` gate (i.e. `score >= THRESHOLD_MIN`)
- Breadth is the count of unique algorithms reporting a hit for a given phrase
- **Use case**: Catch variations by requiring multiple algos to detect something, even if each individual signal is below the primary threshold

**Config parameter**: `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE` (e.g., `2` = at least 2 algos must report scores above their noise floor)

#### Consensus Rule

A chunk is flagged for human review when the following conditions are met according to the configured rules:

```Python
(depth_algo_count >= REQUIRED_ALGOS_ABOVE_THRESHOLD)
    OR
(any_phrase_breadth_count >= REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE)
```

This means:

- **Depth-driven**: Multiple algorithms all confident → flag
- **Breadth-driven**: Multiple algorithms all detecting something (weak or strong) → flag
- **OR logic**: Either condition triggers

#### Examples

| Config | Behavior | Use Case |
| -------- | ---------- | ---------- |
| `REQUIRED_ALGOS_ABOVE_THRESHOLD=2` `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE=3` | Flag if (≥2 algos passing threshold) OR (≥3 algos with any score) | Balance sensitivity with false-positive control |
| `REQUIRED_ALGOS_ABOVE_THRESHOLD=3` `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE=5` | Strict: require strong depth (3 algos confident) or extremely broad signals (5+ algorithms detecting) | Minimizes false positive; increases false negatives |
| `REQUIRED_ALGOS_ABOVE_THRESHOLD=1` `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE=2` | Flag if just 1 algo passes threshold OR 2 algos detect anything | Minimizes false negatives; increases false positives |

#### Tuning for your Context

Results combined via:

1. **Individual threshold application** - Adjust per-algorithm thresholds in `Config_Banned.py`
2. **Depth consensus** - Raise `REQUIRED_ALGOS_ABOVE_THRESHOLD` for stricter consensus
3. **Breadth consensus** - Raise `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE` to require more algos detecting weakly
4. **Algorithm mix** - Enable/disable algorithms in `ALGOS_TO_PROCESS` to change available voters

**Lab note**: Depth and breadth are adjustable via `Config_Banned.py` to explore different detection behaviours. Results will vary significantly with different configurations, and you are responsible for validating the settings match your operational requirements.

## 🏷️ Classification Output Quality

### 🔄 Reverse Stemming

KeyBERT keyword extraction uses a SnowballStemmer to normalise candidate terms before matching.
This increases recall but produces stemmed tokens in classification output (e.g. `'compli'` instead of `'compliance'`).

`ReverseStemmer` solves this transparently:

```text
KeyBERT extraction
    ↓
stem_keywords_with_weights()
    ├─ returns stemmed keyword list   ─────────────────────────────→ detection/scoring
    └─ returns ReverseStemmer map  (stem → most matching original word)
                    ↓
         ClassifyStrategy._process_extract()
                    ↓
         (if REVERSE_STEMMING=True)
         reverse_stem_map.apply_to_meta(doc["meta"], user_classification_keys)
                    ↓
         Stemmed tokens in classification keys replaced with original words
                    ↓
         CSV output with human-readable keywords
```

Key properties:

- Single mutation point: metadata is mutated once before all CSV write paths
- No duplication: `prepare_for_csv_print` needs no stemming logic
- Weight-winner: when one stem maps to multiple originals, the highest-weight source word is kept

## 🤖 Model Integration

### 🧲 Embedding Models

- HuggingFace local models with optional quantization
- Batch processing for efficiency
- Caching of phrase embeddings for detection

### 🧠 LLM Integration

- Ollama-compatible endpoint (local or remote)
- Streaming response handling
- Temperature, top-k, top-p sampling control
- Optional CPU-only execution mode

### 🔧 JSON Repair for LLM Replies

LLMs occasionally return malformed JSON — missing closing braces/brackets, or set-like literals (`{"a", "b"}`) instead of proper strings. The **`TRY_FIX_JSON_LLM_REPLY`** switch in `Config_Global.py` controls whether `ModelOutputAdapter` attempts to auto-repair these issues.

| Value | Behaviour |
| --- | --- |
| `True` (default used in this repository) | Automatically append missing `}` / `]` closers and flatten set-like literals to comma-separated strings. A warning is logged for every repair. |
| `False` | No repairs are applied. If a fixable issue is detected, a warning is logged suggesting the operator enable `TRY_FIX_JSON_LLM_REPLY`. |

Repairs handled:

- **Missing closing braces / brackets** — appended to truncated JSON output.
- **Set-like values** — `{"val1", "val2"}` flattened to `"val1, val2"`.

Every repair (or skipped repair) is logged via `PrettyWriter` under the `JSON Repair` tag so the operator can audit what was changed.

### 🔀 Cross-Encoder Models

- Re-ranking document results
- Paired input scoring
- Weighted blending with ChromaDB scores

### 🤗 HF Model Downloading & Caching

All HuggingFace model downloads are routed through the **`HFDownloader`** class (`Compliance/HFDownloader.py`). No model loading code contacts HuggingFace Hub directly; every download goes through a consent-based flow that records an auditable metadata trail.

#### Offline-first principle

If `HF_HUB_OFFLINE` is set to `"1"` in `Config_Internet_Env.py`, every model load uses `local_files_only=True`. If the model is not already present in the local HF cache, the load raises an exception that is caught by the caller (`get_hf_embeddings`, `load_quantized_model`, or `get_cross_encoder` in `AIHelpers`) and forwarded to `HFDownloader.download()`.

#### Lazy initialization

`CosineScorer` and `KeyBertScorer` (in `Algos/`) defer all heavy work — model loading, device selection, and cache building — until the first call to `verify()`. This ensures that importing or instantiating these singletons never triggers a network request or GPU allocation.

#### Download decision flow

```text
Model load (local_files_only=True)
        │
        ├── Success → use cached model, no network
        │
        └── Failure → HFDownloader.download(key)
                │
                ├─ 1. Metadata short-circuit
                │     Existing download_meta.json has matching
                │     config_hash + valid local_path?
                │     YES → return immediately (no cache scan, no download)
                │
                ├─ 2. HF cache scan (_find_cached_snapshot)
                │     Walk _HF_HUB_CACHE / models--<id> / snapshots /
                │     Found? → write metadata, return (no download)
                │
                ├─ 3. Internet check
                │     HF_HUB_OFFLINE == "1"?
                │     YES → raise InternetConnectionDisabledError
                │           (message includes model name, revision,
                │            and cache path searched)
                │
                └─ 4. User consent prompt
                      Display model name, revision, source, license.
                      User types "y" → snapshot_download()
                      User types anything else → raise HFDownloaderError
```

#### Revision resolution

The config key `_MODELS.<type>.REVISION` can be:

| Value | Meaning |
| ----- | ------- |
| 40-char hex hash | Pinned to that exact commit / snapshot |
| `""` (empty) | "Latest available" — no specific version pinned |

When `REVISION` is empty, `_find_cached_snapshot` enumerates the `snapshots/` directory and picks the **most-recently-modified** sub-directory. Once found, the actual 40-character snapshot hash is:

1. **Written to `download_meta.json`** (the `"revision"` field) so that subsequent runs can short-circuit without re-scanning the cache directory.
2. **Propagated into runtime config** via `cfg.set("<key>.REVISION", resolved_hash)` so that downstream code (`get_model_args`) returns the real hash. This makes cache keys in `get_hf_embeddings` and `load_quantized_model` deterministic (e.g. `model_ac6544c8…_cuda:0_float16` instead of `model_none_cuda:0_float16`).

The same resolution happens after a fresh `snapshot_download()`: the hash is extracted from the returned path (`…/snapshots/<hash>`) and persisted identically.

#### When does a new download happen?

A download is triggered **only** when all of these are true:

1. The model cannot be loaded locally (`local_files_only=True` fails).
2. No valid `download_meta.json` points to an existing local path with a matching config hash.
3. No matching snapshot exists in the HF hub cache directory.
4. Internet access is enabled (`HF_HUB_OFFLINE != "1"`).
5. The user explicitly consents at the interactive prompt.

Importantly, under typical conditions and unchanged cache state, re-downloads are not triggered — even across restarts, config reloads, or version upgrades of RAG-LCC. The resolved hash is persisted in metadata and compared on subsequent runs.

#### Metadata & audit trail

Every download (or cache-hit) writes a `download_meta.json` file under `ModelGovernance/consents/<config_key>/`:

```json
{
  "model_id": "snowflake/snowflake-arctic-embed-l-v2.0",
  "license_url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
  "license_hash": "283ea6cc2997a1a70da0049e09adf9317bb60ca1b51279b65196b83a69e1996b",
  "tls_cert_fingerprint": "7317c0207cddc1f60f5bab13a886180a5f4e3ab733efe7a167cd1273ee6c02ea",
  "accepted_by": "your user",
  "accepted_by_source": "os",
  "accepted_by_verified": false,
  "accepted_at": "2026-03-01T19:14:43.444321Z",
  "host": "your host",
  "pid": 20492,
  "config_hash": "6cff5e0085cb2e66068ed5dfe2b394a3abdf70499f49874f129cfe209f195b24",
  "downloaded_at": "2026-03-01T19:14:43.444321Z",
  "source": "fetched",
  "consent": true,
  "config": {
    "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
    "FRIENDLY_NAME": "Snowflake Arctic Embed L v2.0",
    "REVISION": "",
    "SOURCE": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
    "LICENSE": "Apache-2.0",
    "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "COMPLIANCE_MSG": "Embedder: Snowflake arctic-embed-l-v2.0 is the newest addition to the suite of embedding models Snowflake has released optimizing for retrieval performance and inference efficiency",
    "MODEL_CARD": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0"
  }
}
```

If the operator changes the model configuration (different model name, different revision, etc.) the config hash will differ, existing metadata will not match, and HFDownloader will prompt for re-consent before downloading the new model.

## 🌐 Internet Access

Internet access is configured in `Config_Internet_Env.py`.

| Environment Variable | default used in this repository | Purpose |
| --- | --- | --- |
| `LICENSE_DOWNLOAD` | `"0"` | Allow online fetch of model license files defined in `Config_Models.py`. When `"0"`, the Compliance module prompts for per-fetch consent. |
| `NLTK_STOPWORDS_DOWNLOAD` | `"0"` | Allow download of missing NLTK stopwords corpus. When `"0"`, the system falls back to an empty stopword list. |
| `RAG_LCC_NW_TRACE` | `"0"` | Socket-level network tracing (debug). |
| `RAG_LCC_STACK_TRACE` | `"0"` | Stack traces on errors. |
| `HF_HUB_OFFLINE` | `"1"` | Disable Hugging Face Hub downloads when `"1"`. |
| `TRANSFORMERS_OFFLINE` | `"1"` | Disable transformers library hub access when `"1"`. |
| `HF_DATASETS_OFFLINE` | `"1"` | Disable HF datasets hub access when `"1"`. |
| `ARGOS_MODEL_PROVIDER` | `"OPENNMT"` | Force Argos Translate to use local packages only. |
| `ARGOS_CHUNK_TYPE` | "SPACY" | ARGOS_CHUNK_TYPE: Select the sentence boundary detection (SBD) backend |
| `ARGOS_STANZA_DOWNLOAD` | `"0"` | Control stanza network access for Argos Translate. When `"0"`, only pre-installed language packages are used. When `"1"`, stanza may download missing tokenizer models at runtime. |

## 🌍 Argos Translate

RAG‑LCC uses [Argos Translate](https://github.com/argosopentech/argos-translate) to translate banned-word lists into the detected document language so that compliance checks work across languages.

- **Environment variables** — `ARGOS_MODEL_PROVIDER` and `ARGOS_STANZA_DOWNLOAD` (see table above) control provider selection and network access.
- **Language pairs & code mapping** — configured via the `_ARGOS_DEFINITIONS` slot in `Config_Global.py`. See [Translation configuration (Argos)](README.md#translation-configuration-argos) in the README for the full reference, available pairs, and install/remove commands.
- **Package management** — `python src/Scripts/ArgosTranslatePackages.py install | remove | status`

Each Argos package (~100 MB) bundles an OpenNMT translation model and the required stanza tokenizer, so no additional network downloads are needed at runtime when `ARGOS_STANZA_DOWNLOAD="0"`.

## 🎫 Token Budget

Token budget calculations are heuristic and may not reflect actual tokenizer behavior or model‑specific limits in all cases.

Every LLM call needs a `max_output_tokens` limit — too large and the model may exceed the hardware context window; too small and the reply is truncated. RAG‑LCC resolves this dynamically via the **`TokenBudget`** singleton (`AI/TokenBudget.py`).

### ⚙️ Configuration (Config_Global.py)

| Key | Default used in this repository | Purpose |
| --- | ------- | ------- |
| `TOKEN_BUDGET_CONTEXT_CAP` | 16 384 | Hardware cap — if Ollama reports a larger context window, this value is used instead. Protects weak CPUs / GPUs from being asked to fill a context they cannot hold. |
| `TOKEN_BUDGET_RESERVED_OUTPUT` | 2 048 | Maximum tokens reserved unconditionally for the model reply (upper clamp). |
| `TOKEN_BUDGET_RESERVED_SYSTEM` | 1 024 | Tokens reserved for the system / instruction preamble that wraps every prompt. |

### 🔍 Per-Model Context Detection

On first access for a given model name, `TokenBudget` queries Ollama `/api/show` and caches the reported `num_ctx`. If Ollama is unreachable the config cap is used as a safe fallback. Because the main inference model and the compliance-check model may differ, each gets its own cached limit.

```text
Ollama /api/show ─► detected num_ctx
                        │
            ┌───────────┴───────────┐
            │ detected > cap?       │
            │   yes → use cap       │
            │   no  → use detected  │
            └───────────────────────┘
                        │
                    cached per model
```

### 🧮 Dynamic Budget Formula

Before every LLM call, `compute_dynamic_max_tokens(prompt, model)` is invoked:

```text
prompt_tokens  ≈ word_count(prompt) × 1.3      (no tokeniser dependency)
available      = context_limit - RESERVED_SYSTEM - prompt_tokens
max_output_tokens = clamp(available, 1, RESERVED_OUTPUT)
```

The resulting `max_output_tokens` is passed to Ollama alongside `num_ctx` (the cached context limit) so Ollama allocates the correct KV-cache rather than its default of 2 048.

### 🎨 User Overrides (RAGChat only)

Operators can override `max_output_tokens` and `context_size` at chat‑time via the `max_output_tokens!` and `context_size!` commands in the interactive session. A warning is emitted when the override exceeds the computed budget. See `Chatter._resolve_token_params()` for the resolution logic.

## 🗄️ Storage Layer

### 🗃️ ChromaDB Integration

- HNSW indexing for efficient similarity search
- Cosine similarity metric
- Configurable neighbor exploration parameters
- Collection-based namespacing

### 💾 Metadata Handling

- Type filtering (Chroma compatibility)
- None value removal
- Complex type stringification
- Searchable metadata fields

## 📦 Processing Strategies

Strategy pattern allows pluggable processing pipelines:

```text
ProcessingStrategy (Abstract)          ← Strategies/ProcessingStrategy.py
├── ChunksToDBStrategy                 ← Strategies/ChunksToDBStrategy.py
├── ClassifyStrategy                   ← Strategies/ClassifyStrategy.py
└── [Custom strategies can be added]

Supporting modules in Strategies/:
├── ClassifyHelper          - Classification workflow helpers
├── HomeBrewChunkSelector   - Custom chunking logic
└── StrategyType            - Strategy enum / type constants

Orchestration:
└── Pipeline/LoadAndClassifyProcessor  - Drives the load + classify pipeline
```

Each strategy implements:

- Validation before processing
- Error handling and recovery
- Progress tracking
- Result aggregation

## ⛔ Exclusion + Incremental Hash Check (Skip Unchanged Files)

### 🎯 Goal

Prevent reprocessing of unchanged files and explicitly exclude non-compliant files.

### 🧩 Components

- Document hash in Chroma DB metadata will be compared on document extraction
  If same hash, don't process
   `_PROCESS_IF_UNCHANGED` = False
  This setting applies to `RAGLoad`only

- Exclusions files in ./Exclusions directory contain paths of excluded files
  If a file is in Exclusion list it will not be processed on document extraction
  `USE_EXCLUSIONS` = True
  Files are added to the exclusion list if they are flagged for HUMAN_REVIEW

### 🔗 See Also

For class-level design and relationships, see:

- `Documentation/ClassGraphs/`

## 🌳 Source Tree

```text
src/
├── AI/                             AI model interaction
│   ├── AIHelpers.py
│   ├── LLMCaller.py
│   ├── ModelOutputAdapter.py
│   ├── ModelsCache.py
│   ├── TensorHelpers.py
│   └── TokenBudget.py
│
├── Algos/                          Detection algorithms
│   ├── BM25Scorer.py
│   ├── ComplianceAlgoResult.py
│   ├── CosineScorer.py
│   ├── JaccardScorer.py
│   ├── KeyBertScorer.py
│   ├── LevenshteinScorer.py
│   ├── Masker.py
│   ├── RegexScorer.py
│   ├── ReverseStemmer.py
│   └── UnicodeNormalizer.py
│
├── Apps/                           Application entry points
│   ├── DocClassify.py
│   ├── RAGChat.py
│   └── RAGLoad.py
│
├── Chat/                           Conversation and query handling
│   ├── ChatContext.py
│   ├── Chatter.py
│   ├── CommandProcessor.py
│   ├── QueryParts.py
│   └── RAGChatImpl.py
│
├── Commons/                        Shared infrastructure
│   ├── Exceptions.py
│   ├── NetworkTracer.py
│   ├── SingletonMixin.py
│   └── StartupCommons.py
│
├── Compliance/                     License management and verification
│   ├── ArgosDownloader.py
│   ├── BannedPhraseCollector.py
│   ├── Compliance.py
│   ├── Exclusions.py
│   ├── HFDownloader.py
│   └── SharedHelpers.py
│
├── Config/                         Runtime configuration singleton
│   ├── AddConstantsFromConfigFile.py
│   └── Config.py
│
├── Configuration/                  Static parameter definitions
│   ├── Config_Banned.py
│   ├── Config_DocClassify.py
│   ├── Config_Global.py
│   ├── Config_Internet_Env.py
│   ├── Config_Models.py
│   ├── Config_RAGChat.py
│   └── Config_RAGLoad.py
│
├── Globals/                        Shared state
│   ├── CounterInstance.py
│   ├── Globals.py
│   └── Session.py
│
├── Gui/                            Terminal UI helpers
│   ├── Banner.py
│   ├── CollectionPicker.py
│   ├── Colors.py
│   ├── FileList.py
│   ├── HistoryManager.py
│   ├── Informer.py
│   ├── PrettyWriter.py
│   └── Symbols.py
│
├── Helpers/                        General utilities
│   ├── Accumulator.py
│   ├── ChromaDBHelper.py
│   ├── CSVWriter.py
│   ├── FileUtils.py
│   ├── Helpers.py
│   ├── OfficeDocConverter.py
│   ├── PipelineSettingsSummarizer.py
│   └── ValidExtensions.py
│
├── Pipeline/                       Orchestration
│   └── LoadAndClassifyProcessor.py
│
├── Scripts/                        Standalone maintenance scripts
│   └── ArgosTranslatePackages.py
│
└── Strategies/                     Processing strategies + helpers
    ├── ChunksToDBStrategy.py
    ├── ClassifyHelper.py
    ├── ClassifyStrategy.py
    ├── HomeBrewChunkSelector.py
    ├── ProcessingStrategy.py
    └── StrategyType.py
```

## ⚠️ Error Handling

Critical exceptions are raised on compliance violations and infrastructure failures; see [src/Commons/Exceptions.py](src/Commons/Exceptions.py) for full reference. All custom exceptions inherit from `RAGLCCException`.

**Compliance-related** (fail-fast):

- `ComplianceViolationError` – Base compliance failure (stops execution)
- `PromptComplianceError` – User prompt or LLM response fails compliance check
- `LLMComplianceCheckError` – LLM-based validation returned non-compliant result

**Infrastructure / Configuration** (fail-fast):

- `DeviceConfigurationError` – Invalid device settings (e.g., CPU + 16-bit incompatible)
- `ConfigurationError` – Missing or invalid config values
- `ConfigPathError` – Configuration key path cannot be resolved
- `ModelLoadError` – LLM or embedding model fails to load
- `CollectionNotFoundError` – ChromaDB collection does not exist at expected path
- `InvalidCollectionName` – Collection name contains path separators or invalid characters
- `ChromaInstallCurrentEmbeddingsMismatch` – ChromaDB refuses metadata update (e.g. distance function mismatch); delete the collection and re-run RAGLoad
- `EmbedModelMismatch` – Embedding model or quantization bits differ from what is stored in the collection; re-run RAGLoad with `CHROMA_COLLECTION_KEEP = False`
- `OllamaNotRunning` – Ollama server not reachable
- `PersistDirError` – Persistence directory missing or inaccessible
- `NoVirtualEnvError` – Python virtual environment not detected

**Data / Processing**:

- `DataProcessingError` – Document extraction or critical processing fails
- `LLMResultError` – LLM result is invalid or cannot be parsed
- `RerankError` – Cross-encoder reranking fails or is not possible
- `ExclusionsError` – Exclusions file or exclusion logic fails
- `DocumentsDirError` – Documents directory missing or inaccessible

**Network / Downloads**:

- `InternetConnectionDisabledError` – Internet access required but not available
- `HFDownloaderError` – Hugging Face model download fails
- `HfHubHTTPError` – Hugging Face Hub returns an HTTP error during model download
- `UserNoDownLoadAccept` – User declined model download consent

**Strategy**:

- Fail-fast on compliance violations; downstream finally-blocks execute for cleanup
- Explicit error messages intended to assist debugging
- Debug-level detailed context in logs
- Graceful degradation where safe

## 📝 Logging Architecture

Three-tier logging:

1. **User-Facing** - `PrettyWriter` with colored output
2. **Operational** - Standard Python logging module
3. **Debug** - Conditional DEBUG_LEVEL output

```Python
_ALLOWED_DEBUG_LEVELS = {
    "None": 0,
    "Basic": 1,
    "Standard": 3,
    "Alogs": 4,
    "Components": 50,  # argostranslate, transformers
    "Chat Prompt": 60,
    "Extracted Content": 70,
    "Ollama response": 80,
    "Streaming request output": 100,
}

DEBUG_LEVEL = 3  # See above
```

RAG-LCC does not log raw user queries or LLM responses **unless** DEBUG_LEVEL is set to show the network traffic (100).

## 🔄 Session Management

`Session` object maintains per-conversation state:

- User context
- Collection selection (collections can be switched on the fly)
- Query parameters  (tokens, use chat context, thresholds, temperature etc.)
- Chat name (chats can be switched on the fly)
- Conversation history reference

## ⚡ Performance Considerations

Optimize memory and throughput via [batch processing](#-batch-processing), [caching](#-caching), and [quantization](#-quantization).

### 📦 Batch Processing

- Documents processed in batches for memory efficiency
- Embedding batched to minimize model overhead
- Configurable batch sizes via `BATCH_SIZE` parameters

### 💾 Caching

Caching is used throughout RAG-LCC to improve performance and reduce redundant computation:

- **Phrase Embeddings**: Embeddings for phrases and banned keywords are cached in memory to avoid repeated model inference
- **Algorithm Artifacts**: Cosine, KeyBERT, BM25, and Regex algorithms cache intermediate results (such as token frequencies, IDF values, and extracted keyword matrices)
- **Model and Transformer Caching**: Loaded embedding models and sentence transformers are cached locally
- **Configuration**: Configuration files are parsed and cached once at startup

These caching strategies are intended to reduce redundant computation and may improve performance depending on workload and environment.

### 📊 Quantization

- Optional INT8 or FP16 model quantization
- Reduces memory and improves inference speed
- Configurable via `EMBEDDER_BITS` and `USE_CPU`

## 🔌 Extension Points

**New Detection Algorithms**: Take one of the existing algos as reference
You must implement `return_algo_results` and return a `ComplianceAlgoResult`

```Python
@dataclass
class ComplianceAlgoResult:
    algo: Optional[str]
    phrase: str
    score: Optional[float]
    threshold: Optional[float]
    detail: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class ScorerBase(ABC):
    @abstractmethod
    def return_algo_result(self) -> List[ComplianceAlgoResult]:
        """Return detection results as a list of ComplianceAlgoResult."""
        raise NotImplementedError
```

- Add new algorithm to `AIHelpers` `_run_ensemble_checks()`
- Add new algorithm to `Config_Banned.py`

**New Configuration**: Add to appropriate `Config_*.py`
**Custom Helpers**: Add utilities to `Helpers/`
**New Document Formats**: Extend `ValidExtensions.py`, `Pipeline/LoadAndClassifyProcessor.py`

## 🧵 Thread Safety

RAG-LCC is intended to run by a **single user**. There is no locking implemented.

- Database access is Chroma-handled
Chroma is thread‑safe, but not process‑safe [Reference](https://cookbook.chromadb.dev/core/system_constraints/)
- Singleton patterns guard against concurrent instantiation

### 💻 Local Development

- All models local
- File-based persistence
- DEBUG_LEVEL >= 3 for development

---

For implementation details, see code documentation in individual modules.
