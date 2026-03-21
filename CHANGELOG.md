# Changelog

All notable changes to RAG-LCC are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
