# Banlist Filtering Pipeline

In most scenarios people worry about not missing relevant context (recall). So are there used cases where extracted context from documents should not be ingested into vector databases?

If you work in a IT security company, you expect to retrieve information about malware or exploits. But do you so if your company produces e.g. ice cream? The first takeaway is:

Unwanted information is domain specific and needs to be configured accordingly.

Reading about content filtering shows that an often followed strategy is to avoid loading unwanted content into the vector database. RAG-LCC follows the startegy not to load chunks that are "marked" containing unwanted content. How this is done will be explained later. For now, it is worth looking what happens if a chunk is omitted. Not all information in the chunk may be unwanted. Ignoring the chunk may cause information loss. Depending on the chunking strategy the information loss may be more or less. If the chunker can interpret the document structure, chances are higher that the information loss is narrowed compared to a simple 512 byte structure unaware chunker. Takeway number two:

Context filtering may lead to information loss. Especially, when the chunking is unaware of the document structure.

Masking is great when secrets or credit card numbers occur in a text. By replacing the secret with a meaningless pattern, the core information is still available.

Having a closer look at how the 3 RAG-LCC apps (classify, load, query) handle unwanted content detection shows a configurable multi-layer approach:

Not lost in translation

A core decision was to maintain the "banlist" in English. The banwords are expanded with synonyms and then cached. If a new language is encountered, a on the fly translation of the banned words and caching is triggered. Thrid takeaway: Using one language to define undesired content makes life easier for users than maintaining n versions depending on the supproted langugages.

Multi layer checks

RAG-LCC uses BM25, Jaccard, KeyBert, Regex+Levenshtein and optional Cosine similarity to detect unwanted content in the chunks. Which algorithms are used and their respective parameters are configurable. Checks include depth and bredth:
Depth requires n of the defined alogrithms to provide a result above the defined threshold
Bredth requires n of the defined algorithms to provide a result at all

If breadth or depth check triggers, the chunk is not loaded. During tests it showed that the bredth check needs a minimum threshold. Otherwise, noise causes the bredth check to trigger on every chunk.

Documents having chunks that are not loaded are flagged in a .csv file and can be excluded from further processing. For this purpose, an optional exclution list is maintained.

---

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                          BANLIST FILTERING — SYSTEM OVERVIEW                              ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

  Config_Banned.py
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  BANNED = ["password", "credit card", "iban", ...]   (English)           │
  │  MASKING_REGEXES = { credit_card: r"\d{4}[- ]\d{4}...", ssn: r"...", }   │
  │  Per-app thresholds: RAGLoad / RAGChat / DocClassify                     │
  └──────────────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
  ┌─────────────┐           ┌──────────────────┐
  │  Synonyms   │           │    Masker        │
  │ (WordNet)   │           │ (regex redact)   │
  │             │           │                  │
  │  "password" │           │ 4111 1111 1111   │
  │  → watchword│           │ → [CREDIT_CARD]  │
  │  → passcode │           │                  │
  │  → ...      │           │ 123-45-6789      │
  │             │           │ → [SSN]          │
  │  NOTE: NOT  │           │                  │
  │  used by    │           │ applied BEFORE   │
  │  Cosine /   │           │ storage (Load)   │
  │  KeyBERT    │           │ and AFTER LLM    │
  │  (embeddings│           │ answer (Chat)    │
  │  handle it) │           └────────┬─────────┘
  └──────┬──────┘                    │
         │  Expanded banlist         │ Redacted text
         ▼                           │
  ┌──────────────────────────┐       │
  │   Argos Translate        │       │
  │   (banlist translation)  │       │
  │                          │       │
  │  EN → DE, FR, ES, ...    │       │
  │  "password" → "Passwort" │       │
  │                          │       │
  │  Caches:                 │       │
  │  • translation_cache     │       │
  │  • translated_list_cache │       │
  └──────────┬───────────────┘       │
             │ Native-language       │
             │ banlist               │
             ▼                       │
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         ENSEMBLE CHECKS                                  │
  │                   (run_ensemble_checks)                                  │
  │                                                                          │
  │   Text ──────────────────────────────────────────────────────────────►   │
  │   Embedding ─────────────────────────────────────────────────────────►   │
  │                                                                          │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  ① Regex         exact/fuzzy pattern anchors on each banned phrase │  │
  │  │        │                                                           │  │
  │  │        └──► ② Levenshtein  edit-distance on regex hits             │  │
  │  │                            (catches typos & l33t-speak)            │  │
  │  │                                                                    │  │
  │  │  ③ Jaccard       char n-gram overlap (n=4–6) vs banlist            │  │
  │  │                  cache: per-language tokenized banlist             │  │
  │  │                                                                    │  │
  │  │  ④ BM25          TF-IDF term match, k1/b tunable                   │  │
  │  │                  cache: banlist_cache, idf_cache, avg_len_cache    │  │
  │  │                                                                    │  │
  │  │  ⑤ KeyBERT       double-pass keyword extraction → embedding        │  │
  │  │                  compare keyword vectors to banned phrase vectors  │  │
  │  │                                                                    │  │
  │  │  ⑥ Cosine        document embedding vs banned phrase embeddings    │  │
  │  │                  cache: pharase_embedding_cache_tensor             │  │
  │  │                  (optional, disabled by default)                   │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │   Each algo produces a score.  Scores go to the Accumulator.             │
  │                                                                          │
  │   ┌────────────────────────────────────────────────────────┐             │
  │   │  Accumulator                                           │             │
  │   │                                                        │             │
  │   │  Depth:  REQUIRED_ALGOS_ABOVE_THRESHOLD = N            │             │
  │   │  Bredth: REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE = M     │             │
  │   │                                                        │             │
  │   │  pass: all scores below threshold                      │             │
  │   │  flag: ≥ N algos exceed their threshold                │             │
  │   └────────────────┬───────────────────────────────────────┘             │
  └────────────────────┼─────────────────────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
         PASS                  FLAGGED
            │                     │
            │               HUMAN_REVIEW CSV
            │               (phrase, algo, score,
            │                threshold, chunk)
            │                     │
            │               USE_EXCLUSIONS=True?
            │                     │
            │               Exclusions file
            │               (skip on next run)
            ▼
       continue pipeline
```

The 3 apps (classify, load, query) use similar approaches re-using the RAG-LCC framework components. Here is the flow for document ingestion:

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                        RAGLoad  —  DOCUMENT INGESTION PATH                                ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

  Document file (PDF / DOCX / PPTX / image / ...)
      │
      ▼
  [Text Extraction]  (pdfminer, python-docx, tesseract OCR, ...)
      │
      ▼
  [Unicode Normalizer]
      │
      ▼
  [Masker]  ◄── MASKING_REGEXES from Config_Banned.py
      │           redacts PII before it ever reaches the store
      │           e.g.  "CC: 4111 1111 1111 1111"
      │               → "CC: [CREDIT_CARD]"
      ▼
  [Language Detection]  (langdetect)
      │
      ├── unsupported language ──► reject / FALLBACK_EN
      │
      ▼
  [Chunker]  (SpaCy / Stanza / Sliding Window / ...)
      │
      ▼  (per chunk)
  ┌─────────────────────────────────────────────────────────┐
  │  ENSEMBLE CHECKS  (PIPELINE_CHECK, accumulate=True)     │
  │  Regex + Levenshtein + Jaccard + BM25 + KeyBERT         │
  │  Banlist translated to document language via Argos      │
  └──────────────────────┬──────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
          PASS                    FLAGGED
            │                         │
            ▼                    HUMAN_REVIEW CSV
  [Embed + store in ChromaDB]    + Exclusions file
```

The chat adds a query verfificatin step using the mentioned algorithms and a LLM for prompt validation:

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                        RAGChat  —  QUERY & ANSWER PATH                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

  User query  (any language)
      │
      ▼
  [Language Detection]
      │
      ├── English ──────────────────────────────────────────────────────────┐
      │                                                                     │
      └── non-English                                                       │
              │                                                             │
              ▼                                                             │
       [HfTranslator]  (M2M-100 / Argos Translate)                          │
       query → English                                                      │
       session.response_language = detected_lang                            │
              │                                                             │
              ▼  (rewriter may mix languages again)                         │
       [Language Detection — 2nd pass]                                      │
              │ still non-English?                                          │
              └──► [HfTranslator — 2nd pass] ───────────────────────────────┤
                                                                            │
                                                                            ▼
                                                                    English query
                                                                            │
                                                                            ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PROMPT CHECK  (filter chain)                                           │
  │                                                                         │
  │  ① Ensemble Checks on query text (PROMPT_CHECK stage)                   │
  │     Regex + Levenshtein + Jaccard + BM25 + KeyBERT                      │
  │     (smaller TOP_K for performance)                                     │
  │                                                                         │
  │  ② LLM Guard  (check_prompt_with_llm_guard)                             │
  │     dedicated safety LLM (Llama-Guard / Mistral-based)                  │
  │     prompt: banlist + user classification keys injected                 │
  └────────────────────────────┬────────────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
           PASS                              REJECTED
              │                              (block / log)
              ▼
  [PromptRewrite]  (coreference resolution via spaCy + LLM)
              │
              ▼
  [Vector Retrieval + BM25 Retrieval + RRF fusion]
              │
              ▼
  [LLM generation]  (Ollama)
              │
              ▼
  ┌───────────────────────────────────────────────────┐
  │  ANSWER COMPLIANCE CHECK  (PIPELINE_CHECK)        │
  │  Ensemble Checks on LLM answer text               │
  │  Regex + Levenshtein + Jaccard + BM25 + KeyBERT   │
  └───────────────────┬───────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
       PASS                    FLAGGED
         │                    answer suppressed
         ▼                    HUMAN_REVIEW CSV
  [Masker]
  redact PII from answer
  (credit cards, SSN, IBAN, ...)
         │
         ▼
  Answer shown to user
  (in session.response_language)
```

The classification step validates the classification prompt and then does the classification. Classification serves 2 purposes:

Classify documents according to user defined criteria
Write a .csv document that can be queried with SQLite to create "targted" collections. E.g. collections that contain mammals only.

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                        DocClassify  —  CLASSIFICATION PATH                                ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

  [STARTUP — once per process]
  ┌──────────────────────────────────────────────────────────────────┐
  │  Prompt Compliance Check  (_ensure_compliance_checked)           │
  │                                                                  │
  │  User-supplied classification prompt fed to LLM guard            │
  │  + filter chain (Ensemble Checks on prompt text)                 │
  │                                                                  │
  │  FAIL → PromptComplianceError (abort)                            │
  │  PASS → continue                                                 │
  └──────────────────────────────────────────────────────────────────┘
      │
      │  per document:
      ▼
  Document
      │
      ▼
  [Text Cleaning]  (punctuation, unwanted chars)
      │
      ▼
  [Language Detection]
      ├── unsupported → reject (NOT_OK CSV)
      ▼
  [Embedding]  (HuggingFace SBERT, cached via ModelsCache)
      │
      ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ENSEMBLE CHECKS  (PIPELINE_CHECK, accumulate=False)    │
  │  Regex + Levenshtein + Jaccard + BM25 + KeyBERT         │
  └──────────────────────┬──────────────────────────────────┘
                         │
      ┌──────────────────┴──────────────────┐
      │   result stored; pipeline continues │
      ▼                                     ▼
  [KeyBERT double-pass]           (flag stored for later)
  Pass 1: extract top-N phrases
  Pass 2: refine to top-M n-grams
      │
      ▼
  [Cosine similarity]
  keyword embeddings vs document vector
      │
      ▼
  [Merge weights]  (KeyBERT × Cosine)
      │
      ▼
  [Snowball Stemmer]  (language-aware)
  + ReverseStemmer (restores surface forms after LLM)
      │
      ▼
  [LLM Classification prompt]
  formatted keyword/weight JSON → Ollama LLM
      │
      ▼
  [ModelOutputAdapter]  (parse JSON answer)
      │
      ▼
  [ReverseStemmer.apply_to_meta]  (restore best surface form)
      │
      ▼
  OK CSV  (classification result)
      │
      └── ensemble flagged earlier?
               │
               ▼
          HUMAN_REVIEW CSV
          + Exclusions file  (if USE_EXCLUSIONS=True)
```

---

``

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                         ALGORITHM DECISION MATRIX                                         ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

                        RAGLoad     RAGChat         RAGChat         DocClassify
                        PIPELINE    PROMPT_CHECK    PIPELINE_CHECK  PIPELINE_CHECK
  ─────────────────────────────────────────────────────────────────────────────────
  Regex + Levenshtein     ✓            ✓               ✓               ✓
  Jaccard                 ✓            ✓               ✓               ✓
  BM25                    ✓            ✓               ✓               ✓
  KeyBERT                 ✓            ✓               ✓               ✓
  Cosine                  –            –               –                –   (opt-in)
  LLM Guard               –            ✓               –                ✓
  ─────────────────────────────────────────────────────────────────────────────────
  Masker                  ✓            –               ✓ (answer)      –
  (PII redaction)       before                       after LLM
                        storage
  ─────────────────────────────────────────────────────────────────────────────────
  Translation             ✓            ✓               ✓               ✓
  of banlist           (Argos)      (Argos)          (Argos)         (Argos)
  ─────────────────────────────────────────────────────────────────────────────────
  Synonym expantion       ✓            ✓               ✓               ✓
  of banlist           (WordNet)    (WordNet)        (WordNet)       (WordNet)
  ─────────────────────────────────────────────────────────────────────────────────
  Query translation       –           ✓ (HF/Argos)    –               –
  (user input → EN)
```
