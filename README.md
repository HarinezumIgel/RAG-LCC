<!-- markdownlint-disable MD033 MD060 -->
# 🧪 RAG‑LCC — Experimental RAG Under Constraints

<p align="center">
  <img src="Documentation/Pics/AI_Igel.png" alt="RAG-LCC Logo" width="50%" />
</p>

## 🎯 Who this is for

- 🔬 Researchers and practitioners exploring *why* RAG pipelines succeed or fail
- 🧠 Engineers working with **large, multilingual, or conflicting document sets**
- 💬 Anyone debugging **multi‑turn chat‑context failures** in RAG systems
- 💻 Users running RAG on **constrained or commodity hardware**
- 🧪 People who want to experiment beyond "embed + cosine + top‑k" — see [Query Output Example](QUERY_OUTPUT_EXAMPLE.md) for what a full pipeline run looks like

---

**RAG‑LCC is an experimental Retrieval‑Augmented Generation (RAG) lab focused on understanding and controlling retrieval and context assembly under real‑world constraints**: limited context windows, modest GPUs, large documents, and multi‑turn chat.

- DocClassify       Document classification - results may be used as input filter for RAGLoad
- RAGLoad           Text extraction (document formats, pictures, MS Office) and Vector DB ingestion
- RAGChat           (CLI GUI) and RAGChatService (Open WebUI integration)

<p align="center">
<img src="Documentation/Pics/GroundedDoc.jpg" alt="RAG-LCC Document grounding" />
<p>

Instead of pushing ever‑larger context sizes, RAG‑LCC treats **classification, chunking, retrieval strategies, and staged loading** as first‑class architectural tools.

---

## 🎬 Demo

![Demo](Documentation/Pics/RAG-LCC-Screenshots.gif)

---

## 🧠 What it does — and why it exists

Standard RAG is deceptively simple: embed documents, embed query, retrieve by cosine similarity, prompt the LLM. In practice this produces systems that are brittle in exactly the ways that matter most — they hallucinate when the corpus has conflicting information, they drift in multi‑turn chat as pronouns accumulate, they fail silently on minority‑language documents, and they have no principled way to prevent prohibited content from being stored or returned.

**RAG‑LCC** (Retrieval‑Augmented Generation — Local Corpus & Classification) is an experimental lab for studying and addressing these failure modes. Instead of pushing ever‑larger context windows, it treats **classification, chunking, retrieval strategy, and content filtering** as first‑class architectural decisions. Documents are analysed, compressed, filtered, and assembled *before* reaching the LLM — so the model reasons over coherent, non‑contradictory context rather than an arbitrary pile of chunks.

The system is built around four applications that form a deliberate pipeline:

---

### 🏷️ DocClassify — Know your corpus before you index it

Before anything enters the retrieval indexes, `DocClassify` runs LLM‑powered keyword extraction and batch classification over your document collection. Every file gets structured metadata — topic, category, language, and any custom fields you define — written to a CSV.

This is not just labelling. It is **semantic compression**: large documents are reduced to meaning‑dense keyword signals early, before expensive embedding and retrieval. You can then filter that CSV with a plain SQL WHERE clause to decide exactly which documents proceed to indexing:

```python
# Index only English mammal-related documents classified as Science
CLASSIFY_CSV_QUERY = "Mammal LIKE '%Yes%' AND Language = 'English'"
```

Documents that fail your criteria never get indexed — reducing token waste, context noise, and compliance surface area.

---

### 📥 RAGLoad — Index with intent, filter at the gate

`RAGLoad` ingests the documents you selected and builds three parallel indexes simultaneously:

- **ChromaDB** — dense embedding vectors (Snowflake Arctic Embed L v2.0) for semantic search
- **BM25** — Okapi BM25 keyword index for term‑frequency scoring and lexical recall; complements vector search on precise terminology and rare terms
- **Entity co‑occurrence graph** — spaCy NER entities and noun phrases extracted from every chunk and linked by document co‑occurrence; enables graph traversal to pull in thematically connected chunks that neither vector nor BM25 search would surface

Before any chunk is stored, it passes through a **multi‑algorithm compliance filter chain** — Regex+Levenshtein, Jaccard, BM25, KeyBERT — that detects and optionally masks prohibited content. Leet‑speak decoding and Unicode confusable normalization run first, so obfuscated phrases are caught before embedding.

Seven chunking strategies handle different document types: semantic boundary detection for free text, heading‑aware chunking for structured documents, per‑page for PDFs, per‑slide for presentations. Chunk boundaries match the document’s natural discourse structure rather than arbitrary token counts.

Files unchanged since the last run are skipped. Files flagged by prior compliance runs can be automatically excluded.

---

### 💬 RAGChat — Retrieval that fights context failures

`RAGChat` is a multi‑turn chat interface backed by the indexes built by RAGLoad. It is designed around the observation that **most RAG failures are not retrieval failures** — they are context assembly failures: semantically similar chunks that reinforce each other’s errors, pronoun references that resolved to the wrong entity two turns ago, or factually contradictory passages delivered side‑by‑side without scoping. See [Query Output Example](QUERY_OUTPUT_EXAMPLE.md) for an annotated full-pipeline run.

Each query runs through a staged pipeline:

1. **[Compliance](LEGAL.md#-definition--compliance-rag-lcc) pre‑check** — the multi‑algorithm filter chain (Regex+Levenshtein, Jaccard, BM25, KeyBERT) runs on the raw query; matched phrases are masked or the request is blocked before anything else happens
2. **Translation** — non‑English queries normalised to English via M2M100 (100 languages, MIT)
3. **Query rewriting** — pronouns and referents from prior turns resolved by a dedicated rewrite LLM; prefix with `new:` to hard‑switch topics without clearing history
4. **Multi‑query expansion** — the LLM generates N alternate phrasings to broaden vocabulary coverage across the retrieval pool
5. **Hybrid retrieval** — Vector + BM25 + Graph fused via weighted Reciprocal Rank Fusion; optional live DuckDuckGo web leg
6. **Near‑duplicate removal** — chunks sharing ≥ 85% token overlap collapsed before reranking
7. **Cross‑encoder reranking** — neural relevance scoring on top‑k candidates
8. **Strategy‑gated context assembly** — five profiles from `NARROW` (20 chunks, high precision) to `ULTRA_WIDE` (1500 chunks, exhaustive), with per‑file diversity caps
9. **LLM reasoning** — context assembled above is passed to the generation model
10. **[Compliance](LEGAL.md#-definition--compliance-rag-lcc) post‑check** — the same filter chain re‑runs on the generated answer; matched spans are masked before the response reaches the user

Answers are **grounded** — every sentence is checked for overlap with retrieved source text and marked visually, in CLI and API output alike. You see exactly which parts of the answer are evidence‑backed and which are not.

---

### 🌐 RAGChatService — OpenAI‑compatible RAG as a service

`RAGChatService` wraps the complete RAGChat pipeline in an **OpenAI‑compatible REST API** (`POST /v1/chat/completions`). Point [OpenWebUI](https://github.com/open-webui/open-webui) at it — or any OpenAI client — and your local RAG pipeline becomes a selectable model with no prompt engineering required on the client side.

ChromaDB collections appear as models in the OpenWebUI dropdown. RAG‑LCC knobs (`strategy`, `retriever_k`, `threshold`, `web_search`, `web_weight`) are exposed as OpenWebUI Advanced Parameters so non‑technical users can tune retrieval without editing config files.

Supports Bearer‑token authentication, optional streaming, configurable host/port, and fully offline operation after initial setup.

---

## 🧭 Quick mental model

```text
Raw documents
        │
        ▼
┌───────────────────────────────────────────┐
│  DocClassify  (optional first pass)       │
│  ─────────────────────────────────────    │
│  KeyBERT keyword extraction               │
│  LLM classification  →  CSV metadata      │
│  compliance filter chain (load-time)      │
│  purpose: semantic compression +          │
│           domain-scoped ingestion         │
└───────────────────────────────────────────┘
        │  optional: filter CSV with
        │  plain SQL WHERE clause, e.g.
        │  "Mammal LIKE '%Yes%'"
        ▼
┌───────────────────────────────────────────┐
│  RAGLoad  (indexes the corpus)            │
│  ─────────────────────────────────────    │
│  leet-speak + Unicode normalisation       │
│  compliance filter chain  →  masking      │
│  7 chunking strategies (per file type)    │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ ChromaDB │ │  BM25    │ │   Graph   │  │
│  │ vectors  │ │ keyword  │ │  entity   │  │
│  │ (HNSW)   │ │  index   │ │  co-occur │  │
│  └──────────┘ └──────────┘ └───────────┘  │
│  skips unchanged files (hash check)       │
└───────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RAGChat  /  RAGChatService                                         │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  RAGChat   — interactive CLI                                        │
│  RAGChatService — OpenAI-compatible REST API  ──► OpenWebUI         │
│              (same pipeline, same config)                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  per-query pipeline                                         │    │
│  │                                                             │    │
│  │  user query                                                 │    │
│  │    │  ① compliance pre-check  (banned-phrase filter chain)  │    │
│  │    │  ② M2M100 translation  →  English (if non-English)     │    │
│  │    │  ③ query rewrite  (coreference resolution via LLM)     │    │
│  │    ▼                                                        │    │
│  │  multi-query expansion  (LLM → N alternate phrasings)       │    │
│  │    │  each variant runs an additional Vector search         │    │
│  │    ▼                                                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐    │    │
│  │  │  Vector  │  │   BM25   │  │  Graph   │  │    Web    │    │    │
│  │  │ (Chroma) │  │ keyword  │  │ entity   │  │ DuckDuckGo│    │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘    │    │
│  │       └─────────────┴─────────────┴───────────────┘         │    │
│  │                  weighted RRF fusion                        │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          near-duplicate removal  (Jaccard)                  │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          threshold filter  (sigmoid score ≥ T)              │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          cross-encoder reranker  (mmarco MiniLM)            │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          chunk selection strategy                           │    │
│  │          NARROW · BALANCED_FILE_CAP · DEFAULT · WIDE        │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          context assembly  →  LLM reasoning                 │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          ④ compliance post-check  (answer validation)       │    │
│  │                      │                                      │    │
│  │                      ▼                                      │    │
│  │          answer grounding  (sentence-level overlap marks)   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Web leg active only when WEB_SEARCH_MODE="1" and web_search=on     │
└─────────────────────────────────────────────────────────────────────┘
```

The goal is **not** to feed the model *more* text — but to feed it **better, safer context**.

---

## 📊 Presentation

A slide deck is available as [`RAG-LCC_Presentation.pptx`](Documentation/Presentations/RAG-LCC_Presentation.pptx).
It provides a quick visual overview of the architecture, the four applications, the retrieval pipeline, and the key design decisions — useful as a starting point before diving into the detailed documentation.

---

## 🔑 Feature Highlights

Key capabilities organized by application. Full configuration details, defaults, and code examples in [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md).

### 🏷️ DocClassify

- **Semantic compression** — KeyBERT keyword extraction + LLM classification produces meaning-dense CSV metadata (topic, category, language, and any custom fields you define)
- **Classify-then-load** — filter the output CSV with a plain SQL WHERE clause before indexing: `"Mammal LIKE '%Yes%' AND Language = 'English'"`. Documents that fail the filter are never embedded
- **Compliance filter chain** runs at classification time — detected phrases are masked before any embedding
- **Customisable extraction keys** — add or remove fields by editing `_YOUR_CLASSIFICATION_KEYS` and the matching prompt template; no code changes needed
- **Reverse stemming** — classification output values are back-projected to original surface forms before CSV export
- **`STRICT` / `BALANCED` / `RECALL`** extraction profiles control the LLM's sampling parameters

### 📥 RAGLoad

- **7 chunking strategies** with per-format routing: PDF→PDF_PAGE, DOCX/MD→heading, PPTX→slide, plain text→sliding window, sentences→sentence window, code/CSV→recursive, default→semantic boundary detection
- **Three parallel indexes built simultaneously**: ChromaDB HNSW dense vectors, Okapi BM25 keyword index, spaCy entity co-occurrence graph
- **Compliance filter chain + masking** — Regex+Levenshtein, Jaccard, BM25, and KeyBERT all run before any chunk is stored; matched spans are redacted in place
- **Obfuscation hardening** — leet-speak decoding (`1→i`, `3→e`, …) and Unicode confusable normalisation (Cyrillic lookalikes, `ß→ss`, …) run before detection
- **Incremental processing** — SHA-256 hash check skips unchanged files; exclusion CSVs automatically drop previously-flagged documents
- **Text extraction** — PDF (pdfplumber + pdfminer), MS Office via COM (Word, PowerPoint, Excel), images via Tesseract OCR, plain text and Markdown
- **Classify-then-load filter** — `LOAD_FROM_CLASSIFY_CSV` + `CLASSIFY_CSV_QUERY` (SQLite WHERE) narrows ingestion to documents that passed DocClassify criteria

### 💬 RAGChat

- **8 retrieval modes** — `VECTOR`, `BM25`, `GRAPH`, or any pair/triple fused via weighted Reciprocal Rank Fusion; optional DuckDuckGo web leg as a fourth RRF arm
- **5 retrieval strategies** from `NARROW` (20 chunks, threshold 0.70, high precision) to `ULTRA_WIDE` (1 500 chunks, exhaustive); `BALANCED_FILE_CAP` enforces per-file diversity caps
- **Multi-query expansion** — a dedicated LLM generates N alternate phrasings of the query; each variant runs an additional Vector search merged into the main pool before fusion
- **Query rewriting / coreference resolution** — a second dedicated LLM resolves pronouns and referents from conversation history (`"are they mammals?"` → `"are hedgehogs mammals?"`); prefix with `new:` to hard-switch topics without clearing history
- **Near-duplicate chunk removal** — Jaccard token-level deduplication of the retrieval pool runs after RRF fusion and before reranking
- **Cross-encoder reranking** — mmarco MiniLM rescores every candidate; per-strategy sigmoid threshold drops weak matches; relative-band fallback ensures the top chunk always surfaces
- **Answer grounding** — every answer sentence is checked for overlap with retrieved source chunks and marked visually; CLI uses ANSI highlights, API returns marked source documents as `/marked/<token>` links
- **Compliance filter chain** runs on queries before retrieval **and** on generated responses before delivery
- **Multi-turn conversational memory** — rolling topic summary, configurable turn window, batch pruning; `new:` prefix isolates topics without discarding history
- **Translation** — M2M100 (100 languages, MIT) normalises non-English queries to English before retrieval and rewriting; Argos Translate expands banlists to the document language
- **Per-session web knobs** — `web_search` (`local_only` / `local_and_web` / `web_only`), `web_weight`, `fetch_page_content` (`snippets only` / `fetch pages`); see [CONFIGURATION_REFERENCE.md § Web Search Admin Knobs](CONFIGURATION_REFERENCE.md#-web-search--admin-knobs)

### 🌐 RAGChatService

- **OpenAI-compatible REST API** (`POST /v1/chat/completions`) — any OpenAI client, LiteLLM proxy, or custom integration works without modification
- **OpenWebUI integration** — ChromaDB collections appear as selectable models in the dropdown; retrieval knobs (`strategy`, `retriever_k`, `threshold`, `web_search`, `web_weight`) surface as Advanced Parameters
- **In-memory document cache** — highlighted source documents served as short-lived `GET /marked/<token>` links; configurable TTL, total size cap, and CORS origins
- **Bearer-token authentication**, configurable host/port, optional streaming, automatic streaming downgrade when document grounding is active
- **`_OPENWEB_UI_WEBSEARCH`** — when `True` (and `WEB_SEARCH_MODE="1"`), the web leg is auto-enabled for every incoming OpenWebUI request that doesn't supply an explicit parameter

### 🔧 Cross-App

- **Compliance pipeline is identical across all apps** — same algorithms (Regex+Levenshtein, Jaccard, BM25, KeyBERT), same banlist, per-app consensus thresholds
- **12 named debug levels** (0–100) — `Standard 30` shows pipeline flow; `Chunk Content 32` dumps full retrieved text; `Chat Prompt 60` shows the LLM input; all changeable live in-chat with `set debug ge 30`
- **Config hash verification** — startup rejects runs where `Config_Models.py` or `Config_Banned.py` was edited without updating the stored hash (`python src/Scripts/RecalcConfigHashes.py` to update)
- **Fully offline after initial setup** — `HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`, `WEB_SEARCH_MODE="0"` in `Config_Internet_Env.py`
- **License consent workflows** — RAG‑LCC does not bundle any model; consent is recorded per-model in `ModelGovernance/licenses/` before first use

---

## 🎛️ Configuration at a Glance

RAG‑LCC exposes every significant architectural decision as a configuration slot. Nothing is hardwired — chunking boundaries, retrieval algorithm mix, scoring thresholds, model roles, compliance rules, and answer grounding sensitivity are all independently tunable.

> **If you have a document corpus and want to optimize retrieval** — start with chunking strategies, retrieval mode and strategy profiles, and BM25/HNSW parameters.
> **If you're studying RAG failure modes** — every stage from query rewriting to answer grounding can be inspected at named debug levels, disabled, or replaced independently.
> **If you need to integrate or deploy it** — Ollama or vLLM backend, OpenAI-compatible REST service (`RAGChatService`), OpenWebUI drop-in, fully offline after initial setup.

| Area | What you configure | Why you'd tune it |
|------|--------------------|-------------------|
| **Chunking** | 7 strategies (Semantic, Heading, PDF/Page, Sliding Window, Recursive…); per-format routing; chunk size and overlap | Chunking quality determines retrieval precision — wrong boundaries produce noisy embeddings, referential ambiguity, and incoherent context |
| **Retrieval mode** | `VECTOR`, `BM25`, `GRAPH`, `ALL`, `WEB` — any combination with per-retriever RRF weights | Switch between lexical precision, semantic recall, and entity-graph traversal; tune each store's influence independently |
| **Retrieval strategy** | 5 profiles (`NARROW` → `ULTRA_WIDE`): chunk count to LLM, score threshold, per-file limits, retriever-k | Dial precision vs recall: 20 chunks for focused Q&A, 1500 for exhaustive exploratory search |
| **Reranking** | Cross-encoder on/off per strategy; sigmoid score threshold | Neural relevance pass after retrieval — switch off for speed, tune threshold for precision |
| **Query processing** | Multi-query expansion (N alternate phrasings); context-dependent rewriting; pronoun/referent resolution; meta-descriptor guard | Boost recall via vocabulary diversity; prevent stale chat history from poisoning retrieval |
| **Chat session** | Turns to keep, history window size, topic summary mode, preferred response language | Control conversational memory budget; isolate topics with `new:` to prevent referential drift |
| **Models** | Any Ollama or vLLM model; separate roles for generation, query rewriting, and safety checking | Swap models per role — use a large model for generation and a small one for rewriting |
| **Prompts** | Fully customisable per model and task: chat, classification, safety check, query rewrite, topic detect | Adapt RAG‑LCC to any domain by editing prompts; no code changes needed |
| **Compliance** | 5-algorithm detection pipeline (Regex+Levenshtein, Jaccard, BM25, KeyBERT); per-app thresholds; masking; consensus count | Fine-tune false-positive/negative tradeoff independently for indexing vs chat |
| **Content hardening** | Leet-speak and Unicode confusable normalization; WordNet synonym expansion; LLM guard model | Defense-in-depth: obfuscation is neutralized before embedding, LLM gates responses before delivery |
| **Classification** | Customisable extraction keys; `STRICT`/`BALANCED`/`RECALL` profiles; SQLite filter for selective indexing | Classify first, then load only the documents that match your query's domain |
| **Language** | 28-language detection (Lingua); M2M100 query translation (100 languages); Argos banlist translation | Retrieve and filter correctly even in multilingual document corpora |
| **Web search** | DuckDuckGo integration; 3-stage pre-filter (BM25 + cosine + rerank); intent blocking; per-session weight | Augment local retrieval with live web results; configure filtering aggressively enough to suppress noise |
| **Answer grounding** | Sentence-level overlap detection; configurable match strictness; color markers per output mode | Distinguish grounded sentences from hallucinations at the sentence level, in CLI and API |
| **Deployment** | Ollama or vLLM backend; `RAGChatService` (OpenAI-compatible REST); OpenWebUI drop-in | Same config and pipeline whether you run CLI, a service, or behind OpenWebUI |
| **Observability** | 12 named debug levels (`Standard 30` → `Streaming 100`); in-chat toggle; performance event log | Trace every step: retrieval scores, merged chunk pool, prompt text, grounding markers, raw token stream |

Full slot-level details: [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) · per-file reference: [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)

---

## ❓ Documentation

| Document | What's inside |
| --- | --- |
| 📘 [README.md](README.md) | Project overview · feature summary · quick-start |
| 🚀 [INSTALL.md](INSTALL.md) | Prerequisites · cloning · dependencies · Ollama / OpenWebUI / Argos / NLTK / Tesseract / spaCy / GPU setup · first-run walkthrough |
| 📚 [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) | Per-file reference for every `Config_*.py` · CLI overrides · translation config · troubleshooting |
| 📸 [EXAMPLES.md](EXAMPLES.md) | End-to-end terminal sessions for `RAGLoad`, `RAGChat`, `DocClassify`, `RAGChatService` |
| 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline internals · compliance chain · chunking · query rewrite · graph index |
| 🧭 [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md) | Curated hands-on session and suggested experiments |
| 🔐 [SECURITY.md](SECURITY.md) | Security policy · threat model · limitations · web search risks |
| ⚖️ [LEGAL.md](LEGAL.md) | This document — definitions, governance, disclaimers |
| 📋 [CHANGELOG.md](CHANGELOG.md) | Version history and release notes |
| 🙏 [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) | Third-party libraries, models, and attribution |

## 📚 Background & related write‑ups

Some design decisions in RAG‑LCC are motivated by concrete failure analyses:

- **Experimenting with RAG‑LCC on constrained hardware**
  DEV.to article on classification as semantic compression and context reduction
  [https://dev.to/harinezumigel/experimenting-with-rag-lcc-on-constrained-hardware-3dlg](https://dev.to/harinezumigel/experimenting-with-rag-lcc-on-constrained-hardware-3dlg)

- **When the pronoun “they” breaks your RAG**
  Reddit write‑up on chat‑context and referential ambiguity failures
  [https://www.reddit.com/r/Rag/comments/1spro5f/when_the_pronoun_they_breaks_your_rag_fixing/](https://www.reddit.com/r/Rag/comments/1spro5f/when_the_pronoun_they_breaks_your_rag_fixing/)

- **When Your RAG System Confidently Asks About Hedgehog RAM**
  Reddit write‑up on chat history poisoning and the `new:` topic‑switch fix
  [https://www.reddit.com/r/Rag/comments/1swbmdr/when_your_rag_system_confidently_asks_about/](https://www.reddit.com/r/Rag/comments/1swbmdr/when_your_rag_system_confidently_asks_about/)

- **Filtering the Noise: A Practical Multi-Layer Banlist Pipeline for RAG Systems**
  Reddit wirte-up on content filtering
  [https://www.reddit.com/r/Rag/comments/1ta1svk/filtering_the_noise_a_practical_multilayer/](https://www.reddit.com/r/Rag/comments/1ta1svk/filtering_the_noise_a_practical_multilayer/)

- **Speaking the Corpus’s Language: How Multilingual RAG Stays Coherent Across Turns**
  DEV.to article on two‑pass query translation and multilingual coherence in multi‑turn RAG
  [https://dev.to/harinezumigel/speaking-the-corpuss-language-how-multilingual-rag-stays-coherent-across-turns-4pf5](https://dev.to/harinezumigel/speaking-the-corpuss-language-how-multilingual-rag-stays-coherent-across-turns-4pf5)
- **Lessons Learned Building an Experimental RAG Lab**
  Reddit write‑up on failure modes that only surface with end‑to‑end visibility: retrieval pool size, context poisoning, multilingual gaps, scoring assumptions, and why old workarounds become bugs
  [https://www.reddit.com/r/Rag/comments/1to784v/lessons_learned_building_an_experimental_rag_lab/](https://www.reddit.com/r/Rag/comments/1to784v/lessons_learned_building_an_experimental_rag_lab/)
- **Adding Web Search to Our RAG Pipeline: What Broke and Why**
  DEV.to article on integrating internet retrieval into an experimental RAG pipeline — query routing, compliance gating, threshold failures, and the edge cases that only appear in production-like conditions
  [https://dev.to/harinezumigel/adding-web-search-to-our-rag-pipeline-what-broke-and-why-4ge5](https://dev.to/harinezumigel/adding-web-search-to-our-rag-pipeline-what-broke-and-why-4ge5)
- **15 Months Building a RAG System in Retirement: Lessons Learned and What Actually Worked**
  Reddit write‑up on lessons learned building RAG‑LCC from the ground up — architectural decisions, what worked, what didn't, and practical insights from extended experimentation
  [https://www.reddit.com/r/Rag/comments/1valvk6/15_months_building_a_rag_system_in_retirement/](https://www.reddit.com/r/Rag/comments/1valvk6/15_months_building_a_rag_system_in_retirement/)
These are not tutorials — they document *observed failure modes* that this lab explores programmatically.

---

## ⚠️ Project status

🧪 **Experimental / lab software**

RAG‑LCC is intended for:

- architectural exploration
- controlled experimentation
- learning and research

It is **not** a plug‑and‑play production framework.

---

## ⭐ Citation & visibility

If this project helps you reason about **retrieval, chunking, and context assembly failures** in RAG systems, a ⭐ helps other practitioners find it.

A `CITATION.cff` file is included for academic or technical reference.

---

### TL;DR — try it locally

Read [INSTALL.md](INSTALL.md) before running anything. You get information what will be done during setup.

```bash
git clone <this-repo>; cd RAG-LCC
python -m venv .venv; .\.venv\Scripts\Activate.ps1   # or source .venv/bin/activate
# Guided setup, recommended
python src/Scripts/Setup.py                           # guided first-run setup (copies configs, downloads models)
# Note: License acceptance is required and recorded on startup
accepted on first start.
python ./src/Apps/RAGLoad.py  --doc-dir TestDocs
python ./src/Apps/RAGChat.py  --doc-dir TestDocs
```

Read [INSTALL.md](INSTALL.md) before running anything — model licenses must be
accepted on first start.

## RAG‑LCC — Disclaimer

### ⚠️ Experimental Research Framework

RAG‑LCC is an **experimental research framework** intended solely for **laboratory use, evaluation, and learning**.
It is **not** production software and must **not** be used in operational, regulated, safety‑critical, or compliance‑critical environments.

### 🚫 No Support, No Warranty, No SLA

This project is provided **as‑is** with **no**:

- support or assistance
- issue response or troubleshooting
- bug fixes, patches, or security updates
- maintenance or compatibility commitments
- service‑level objectives or availability guarantees

No warranty—express or implied—is provided regarding correctness, completeness, security, reliability, or fitness for any purpose.

### 🔐 Legal, Regulatory, and Security Responsibility

All **legal**, **regulatory**, **operational**, and **security** risks arising from the use of this software are assumed entirely by the **operator**.

This project is **not** a legal, security, governance, or compliance solution.
Nothing in the source code, documentation, examples, or logs should be interpreted as legal or security advice.

For definitions, constraints, and further detail, review:

- [LEGAL.md](LEGAL.md)
- [SECURITY.md](SECURITY.md)

### 🎯 Intended Use

RAG‑LCC is intended for:

- local experimentation with RAG pipelines
- research into filter chains and scoring
- teaching and learning RAG architectures
- development and testing of custom detection algorithms

It is **not** intended for end users, enterprises, or regulated operational deployment.

### 📉 Limitations

Detection and validation mechanisms in this framework are **probabilistic**.
False positives and false negatives **will** occur.

Scope includes:
*document ingestion, prompt validation, document classification, and LLM output validation as defined in* `./src/Configuration/Config_*.py`.

### ⚠️ Final Notice

Use of RAG‑LCC is entirely **at the operator’s own risk**.
Nothing in this repository guarantees correctness, safety, regulatory conformity, or suitability for any specific environment or risk profile.
