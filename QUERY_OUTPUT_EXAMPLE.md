# Query Output Example

``` Text
(.venv) PS D:\RAG-LCC> python .\src\Apps\RAGChat.py
🔵 Config path                    Checking configured filesystem path slots (root guard + presence check).
🔵 Config path                    For file-like slots, present means the containing directory exists.
🔵 Config path                    DOC_DIR -> D:\RAG-LCC\TestDocs present: YES
🔵 Config path                    LOG_FILE -> D:\RAG-LCC\compliance.log present (containing dir): YES
🔵 Config path                    _ABSOLUTE_PATH -> D:\RAG-LCC present: YES
🔵 Config path                    _CUSTOM_NLTK_DATA_DIRECTORY -> D:\RAG-LCC\AppData\Roaming\nltk_data\corpora\stopwords present: YES
🔵 Config path                    _EXCLUSIONS_DIR -> D:\RAG-LCC\Exclusions present: YES
🔵 Config path                    _HF_HOME -> C:\Users\pfm\.cache present: YES
🔵 Config path                    _HF_HUB_CACHE -> C:\Users\pfm\.cache\.hf-cache present: YES
🔵 Config path                    _LOG_DIRECTORY -> D:\RAG-LCC\logs\RAGChat present (containing dir): YES
🔵 Config path                    _CHROMA_DB_DIR -> D:\RAG-LCC\chromadb\docs present: YES
🔵 Config path                    _BM25_INDEX.BM25_INDEX_DIR -> D:\RAG-LCC\chromadb\bm25 present: YES
🔵 Config path                    _GRAPH_INDEX.GRAPH_INDEX_DIR -> D:\RAG-LCC\chromadb\graph present: YES
🔵 Config path                    _REGEX_INDEX.REGEX_INDEX_DIR -> D:\RAG-LCC\chromadb\regex present: YES
🔵 Config path                    _INTENT_FILTER_LOG -> D:\RAG-LCC\logs\RAGChat\intent_filter.log present (containing dir): YES
🔵 Config path                    _QUERY_LOG -> D:\RAG-LCC\logs\RAGChat\queries.log present (containing dir): YES


🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔
🦔                                                                      🦔
🦔                 HarinezumIgel RAGChat                                🦔
🦔          An **experimental** lab environment                         🦔
🦔           to test settings and filter chains                         🦔
🦔            Version v0.5.2.0/1533 2026-10-05                          🦔
🦔                                                                      🦔
🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔🦔


🔵 Environment variable           HF_HUB_OFFLINE=0
🔵 Environment variable           HF_DATASETS_OFFLINE=1
🔵 Environment variable           TRANSFORMERS_OFFLINE=1
🔵 Environment variable           LICENSE_DOWNLOAD=0
🔵 Environment variable           RAG_LCC_NW_TRACE=0
🔵 Environment variable           RAG_LCC_STACK_TRACE=1
🔵 Environment variable           ARGOS_MODEL_PROVIDER=OPENNMT
🔵 Environment variable           HF_HUB_DISABLE_PROGRESS_BARS=0
🔵 Environment variable           WEB_SEARCH_MODE=1

🟢 Outbound downloads             One or more settings allow outbound model/data downloads Set the relevant offline flags if you want to prevent network access.
🟡 Web search                     Web search is ENABLED (WEB_SEARCH_MODE="1"). User queries may be sent to the internet. Review LEGAL.md § Web Search
↳                                 and SECURITY.md before deploying.


🔵 OLLAMA                         Access to Ollama Local LLM Provider is enabled. http://localhost:11434/api/generate streaming: False
🔵 OLLAMA                         Probing http://localhost:11434/api/tags → OK
🟢 HF cache                       HF home: C:\Users\pfm\.cache HF hub cache: C:\Users\pfm\.cache\.hf-cache

                                  Running RAGChat. Good luck!
↳
   ⚪ Argos Translate                Lang: Installed English                  (en   )
   ⚪ Argos Translate                Lang: Installed Dutch                    (nl   )
   ⚪ Argos Translate                Lang: Installed French                   (fr   )
   ⚪ Argos Translate                Lang: Installed German                   (de   )
   ⚪ Argos Translate                Lang: Installed Italian                  (it   )
   ⚪ Argos Translate                Lang: Installed Spanish                  (es   )
   ⚪ spaCy                          Pkg:  Installed de_core_news_sm          (de   ) German
   ⚪ spaCy                          Pkg:  Installed en_core_web_sm           (en   ) English
   ⚪ spaCy                          Pkg:  Installed es_core_news_sm          (es   ) Spanish
   ⚪ spaCy                          Pkg:  Installed fr_core_news_sm          (fr   ) French
   ⚪ spaCy                          Pkg:  Installed it_core_news_sm          (it   ) Italian

🟢 Configuration.Config_Models           No change detected in config file D:\RAG-LCC\src\Configuration\Config_Models.py
🟢 Configuration.Config_Banned_Detection No change detected in config file D:\RAG-LCC\src\Configuration\Config_Banned_Detection.py
🟢 Configuration.Config_Banned_Content   No change detected in config file D:\RAG-LCC\src\Configuration\Config_Banned_Content.py
🟢 Configuration.Config_Banned_Prompts   No change detected in config file D:\RAG-LCC\src\Configuration\Config_Banned_Prompts.py
🟢 Configuration.Config_Load_Retrievers  No change detected in config file D:\RAG-LCC\src\Configuration\Config_Load_Retrievers.py
🟢 Configuration.Config_Load_Chunkers    No change detected in config file D:\RAG-LCC\src\Configuration\Config_Load_Chunkers.py
🟢 Configuration.Config_WebSearch        No change detected in config file D:\RAG-LCC\src\Configuration\Config_WebSearch.py
🟢 Configuration.Config_Internet_Env     No change detected in config file D:\RAG-LCC\src\Configuration\Config_Internet_Env.py

🟢 License                        All LICENSE.txt files for active models in Config_Models.py (see _MODELS) are consented
🟢 Argos License                  Consent valid for 8 configured language pair(s)
🔵 Tokenizer Resources            Default tokenizer resources directory: C:\Users\pfm\.local\share\resources (directory not present yet)
🟢 spaCy License                  Consent valid for 5 configured model(s)
🟢 HF Download                    _MODELS._EMBED: model already downloaded with matching revision and config hash. Skipping download.
🟢 HF Download                    _MODELS._CROSS: model already downloaded with matching revision and config hash. Skipping download.
   ⚪ WordNet Synonyms               Expanded 81 phrases → 139 (+60 synonyms, depth=1, max/phrase=1)
🟢 BM25Scorer                     Initialized BM25 Scorer (algo=BM25) with 139 base phrase(s)
🔵 HF                             Try to load snowflake/snowflake-arctic-embed-l-v2.0 revision 'None' key: snowflake/snowflake-arctic-embed-l-v2.0_none_cuda:0_torch.float32 from cache.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 5927.97it/s]
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🔵 TokenBudget                    Detected context_length=32768 for mistral:7b via ollama adapter
🔵 TokenBudget                    Model 'mistral:7b': context_limit=32768 (below cap 32768 — using detected value)
🔵 HF                             Try to load cross-encoder 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' revision 'None' device=cuda:0 from cache.
🟢 HF Download                    _MODELS._CROSS: model already downloaded with matching revision and config hash. Skipping download.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 6865.83it/s]
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🟢 Regex                          Aux-lemma cache built for 5 active language(s)
→ Loaded defaults for 'DEFAULT':
  ▶ Chat settings
    ▶ Debug                 : debug_level=30  debug_mode='ge'
    ▶ Chat Context          : use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Talk with             : collection='Test'  chat_name='MyFirstChat'
    ▶ File Input            : file=None  path=None  metadata=None  file_cap=15
    ▶ Visual                : mark_text=True

  ▶ Retrieval settings
    ▶ Retrieval strategy    : retrieval strategy='DEFAULT'
    ▶ Strategy overrides    :
    ▶ LLM                   : temperature=0.1  top_p=0.92  top_k=40
    ▶ Retriever tuning      : fetch_k=100  context_chunks=50  threshold=0.5
    ▶ Retriever weights     : vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0  web_weight=0.5
    ▶ Context/session       : file_cap=15  use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Output                : max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
    ▶ API passthrough       : extraOllamaOptions=None  ollamaTopLevelParams=None
    ▶ Web                   : web_search='local_only'  fetch_page_content='snippets only'  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5

  ▶ Orchestration settings
    ▶ Orchestration strategy: orchestrator_flow='THOROUGH_QUERY_REWRITE'
    ▶ Orchestrator overrides:
    ▶ Flow selectors        : force_retrieve_mode=None  main_query_source=None  secondary_query_source=None  rerank=None
    ▶ Flow stages           : secondary_query=None  per_language_indexed_query=None  indexed_translate=None  indexed_synonyms=None  orig_lang_vector=None  vector_alternates=None
    ▶ Flow retrievers       : vector=None  bm25=None  graph=None  regex=None
    ▶ Flow runtime          : query_rewrite=None  pronoun_subst=None  rerank_stage=None  low_score_fallback=None  low_recall_rescue=None  grounding=None

🔵 Masker                         Loaded _MASKING_REGEXES from configuration
🟢 Banned Word Check              *** Banned word check status by stage and algorithm:
🟢 Banned Word Check              - PROMPT_CHECK  : Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅  Prompt Check: ✅
🟢 Banned Word Check              - PIPELINE_CHECK: Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅  Pipeline Check: ✅
🟢 Banned Word Check              - MASKING       : ✅ applied on final result

🟢 Banned Words                   ✅ Current banned words: 81
🟢 Masking regexes                ✅ Current masking regexes: 23
Sleeping for 1 seconds .
🔵 OLLAMA                         Probing http://localhost:11434/api/tags → OK
🟢 OLLAMA                         OLLAMA is reachable on: http://localhost:11434/api/tags streaming: False
🟢 GPU                            GPU is available
🔵 GPU                            Number of GPUs: 1
🔵 GPU                            GPU Name: Tesla T4
   -                              ----------------------
🔵 LLM for prompt compliance      llama-guard3:8b (Built with Meta Llama 3.)
🔵 LLM for prompt compliance      Llama Guard 3 is licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved.
🔵 LLM for user chat              LLM: Mistral 7B is a 7.3B parameter model
🔵 LLM for user chat              mistral:7b (LLM: Mistral 7B is a 7.3B parameter model)
Chat with your documents

  ▶ Chat settings
    ▶ Debug                 : debug_level=30  debug_mode='ge'
    ▶ Chat Context          : use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Talk with             : collection='Test'  chat_name='MyFirstChat'
    ▶ File Input            : file=None  path=None  metadata=None  file_cap=15
    ▶ Visual                : mark_text=True

  ▶ Retrieval settings
    ▶ Retrieval strategy    : retrieval strategy='DEFAULT'
    ▶ Strategy overrides    :
    ▶ LLM                   : temperature=0.1  top_p=0.92  top_k=40
    ▶ Retriever tuning      : fetch_k=100  context_chunks=50  threshold=0.5
    ▶ Retriever weights     : vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0  web_weight=0.5
    ▶ Context/session       : file_cap=15  use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Output                : max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
    ▶ API passthrough       : extraOllamaOptions=None  ollamaTopLevelParams=None
    ▶ Web                   : web_search='local_only'  fetch_page_content='snippets only'  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5

  ▶ Orchestration settings
    ▶ Orchestration strategy: orchestrator_flow='THOROUGH_QUERY_REWRITE'
    ▶ Orchestrator overrides:
    ▶ Flow selectors        : force_retrieve_mode=None  main_query_source='FINAL_QUERY'  secondary_query_source='TRANSLATED_QUERY'  rerank=True
    ▶ Flow stages           : secondary_query=True  per_language_indexed_query=True  indexed_translate=True  indexed_synonyms=True  orig_lang_vector=True  vector_alternates=True
    ▶ Flow retrievers       : vector=True  bm25=True  graph=True  regex=True
    ▶ Flow runtime          : query_rewrite=True  pronoun_subst=True  rerank_stage=True  low_score_fallback=True  low_recall_rescue=True  grounding=True

help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat  · type new: your question to start a new topic
💬 Your actual query>  what do hedgehogs eat and where do they live?

🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (default): LOW (confidence: 42%, threshold: 64%)
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PROMPT_CHECK
🟢 Cache build Jaccard Banned     Built Jaccard n-gram cache with 139 entries for language english
🟢 BM25 Scorer Cache              Built BM25 banlist cache with 139 entries for language english
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 1533.63it/s]
🟢 Cache build Keybert Scorer     Built KeyBert embeddings cache with 81 entries
🟢 KeyWrdChk Depth                0 algos passed threshold vs. required 2
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 3
🔵 TokenBudget                    Detected context_length=131072 for llama-guard3:8b via ollama adapter
🔵 TokenBudget                    Model 'llama-guard3:8b' reports 131072 tokens; capped to 32768 (TOKEN_BUDGET_CONTEXT_CAP)
🔵 TokenBudget                    [llama-guard3:8b] context=32768 reserved_sys=64 prompt≈11 → max_output_tokens=64
🔵 LLM Plan                       Prompt validation: running the guard model to classify the user prompt as allow/block.
🔵 TokenBudget                    [llama-guard3:8b] num_ctx=32768 prompt≈11 → num_predict=64
🔵 Call LLM                       Model: llama-guard3:8b prompt template: _PROMPT_CHECK_CHAT_LLAMA_GUARD stage: Check provided prompt
🔵 Call LLM                       options: {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': 64, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: llama-guard3:8b took 00:05
🟢 CheckPrompt                    Provided prompt is considered compliant by: llama-guard3:8b. Reason: Prompt classified as safe
🔵                                Chatter RAG Query LMM: mistral:7b
🔵 Retrieval Orchestration        Chosen orchestration flow: THOROUGH_QUERY_REWRITE
🔵 Retrieval Orchestration        start retrieval turn
🟢 Chroma Collection              Using Chroma DB collection Test
🔵 VectorStore                    Set Chroma vector store. Name: Test Path: D:\RAG-LCC\chromadb\docs\Test
🔵 Retrieval Orchestration        prepare session context
🔵 Retrieval Orchestration        normalize user query
🔵 UserQuery                      Original user query: 'what do hedgehogs eat and where do they live?'
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (raw_user_query): LOW (confidence: 42%, threshold: 64%)
🔵 QueryRewrite                   No conversation history — skipping rewrite
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (rewrite_query): LOW (confidence: 42%, threshold: 64%)
🔵 LangDrift                      raw_query_language=english  orig_translated_query_en='what do hedgehogs eat and where do they live?'
🔵 LangDrift                      rewrite_language=english  rewritten_query='what do hedgehogs eat and where do they live?'
🔵 LangDrift                      post_rewrite_query_en='what do hedgehogs eat and where do they live?'
🔵 FinalQuery                     Final query for retrieval: 'what do hedgehogs eat and where do they live?' (unchanged)
🔵 LLM Plan                       Alternate queries: generating up to 3 retrieval variants from the normalized query.
🔵 TokenBudget                    Authoritative budget=2048 vs caller estimate=256; using authoritative value
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈219 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_QUERY_EXPAND stage: Multi-query expansion
🔵 Call LLM                       options: {'temperature': 0.5, 'top_k': 40, 'top_p': 0.95, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:08
   ⚪ MultiQuery                     Alternate queries (3)
   ↳                                 1: 'Dietary habits of hedgehogs'
   ↳                                 2: 'Habitat and food sources of hedgehogs'
   ↳                                 3: 'Where do hedgehogs reside and what do they consume?'
🔵 Retrieval Orchestration        query rewrite flow:
↳                                 seed_en='what do hedgehogs eat and where do they live?'
↳                                 rewritten='what do hedgehogs eat and where do they live?'
↳                                 final_en='what do hedgehogs eat and where do they live?'
🔵 Retrieval Orchestration        language: en      normalized query final='what do hedgehogs eat and where do they live?' alternate_count=3
🔵 Retrieval Orchestration        language: en      apply retrieval gates
🔵 Retrieval Orchestration        Query rewrite stage                Activated                              knob=use_query_rewrite            value=on
🔵 Retrieval Orchestration        Pronoun substitution stage         Activated                              knob=use_pronoun_substitution     value=on
🔵 Retrieval Orchestration        Secondary query stage              Activated                              knob=use_secondary_query          value=on
🔵 Retrieval Orchestration        Original-language vector stage     Activated                              knob=use_original_language_vector value=on
🔵 Retrieval Orchestration        Indexed query shaping stage        Activated                              knob=shape_indexed_queries        value=on
🔵 Retrieval Orchestration        Indexed query translation stage    Activated                              knob=translate_indexed_queries    value=on
🔵 Retrieval Orchestration        Indexed synonym expansion stage    Activated                              knob=expand_indexed_queries       value=on
🔵 Retrieval Orchestration        Vector alternates stage            Activated                              knob=use_vector_alternates        value=on
🔵 Retrieval Orchestration        Grounding stage                    Activated                              knob=run_grounding                value=on
🔵 Retrieval Orchestration        Rerank stage                       Activated                              knob=run_rerank                   value=on
🔵 Retrieval Orchestration        Low-score fallback action          Activated                              knob=run_low_score_fallback       value=on
🔵 Retrieval Orchestration        Low-recall rescue action           Activated                              knob=run_low_recall_rescue        value=on
🔵 Retrieval Orchestration        Local stage                        Activated                              knob=web_search                   value=local only
🔵 Retrieval Orchestration        Web stage                          Not activated                          knob=web_search                   value=local only
🔵 Retrieval Orchestration        Vector retriever stage             Activated                              knob=run_vector                   value=on
🔵 Retrieval Orchestration        BM25 retriever stage               Activated                              knob=run_bm25                     value=on
🔵 Retrieval Orchestration        Graph retriever stage              Activated                              knob=run_graph                    value=on
🔵 Retrieval Orchestration        Regex retriever stage              Activated                              knob=run_regex                    value=on
🔵 Retrieval Orchestration        original-language vector leg: off lang=en query='what do hedgehogs eat and where do they live?' reason=user query already in English
🔵 Retrieval Orchestration        language: en      stage plan flow=THOROUGH_QUERY_REWRITE mode=ALL guardrail=off original_leg=off rewrite=on pronouns=on rerank=on low_score_fallback=on
↳                                 low_recall_rescue=on grounding=on indexed_shape=on indexed_translate=on indexed_synonyms=on alt_queries=3 retrievers=v=on,b=on,g=on,r=on
🔵 Retrieval Orchestration        effective stage gates: flow=THOROUGH_QUERY_REWRITE mode=ALL web_mode=local_only local_stage=on web_stage=off
🔵 Retrieval Orchestration        language: en      retrieve local candidates mode=ALL
🔵 Retrieval Orchestration        query language flow: user=en current=en rewrite=en retrieval=en
🔵 Retrieval Orchestration        retriever: Vector     query: 'what do hedgehogs eat and where do they live?' dispatch query
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test for language en
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance                Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6164     0.3836    0.3836  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              2        0.5597     0.4403    0.4403  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              3        0.4366     0.5634    0.5634  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              4        0.4149     0.5851    0.5851  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              5        0.3867     0.6133    0.6133  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              6        0.3511     0.6489    0.6489  Vector                     Cats.md
   ⚪ Chroma                              7        0.3006     0.6994    0.6994  Vector                     Cats.md
   ⚪ Chroma                              8        0.2996     0.7004    0.7004  Vector                     Apes.docx
   ⚪ Chroma                              9        0.2988     0.7012    0.7012  Vector                     Fish.txt
   ⚪ Chroma                             10        0.2840     0.7160    0.7160  Vector                     Kamele.txt
   ⚪ Chroma                             11        0.2748     0.7252    0.7252  Vector                     Lions.pptx
   ⚪ Chroma                             12        0.2684     0.7316    0.7316  Vector                     Cats.md
   ⚪ Chroma                             13        0.2615     0.7385    0.7385  Vector                     Cats.md
   ⚪ Chroma                             14        0.2542     0.7458    0.7458  Vector                     Cats.md
   ⚪ Chroma                             15        0.2541     0.7459    0.7459  Vector                     Cats.md
   ⚪ Chroma                             16        0.2530     0.7470    0.7470  Vector                     Lions.pptx
   ⚪ Chroma                             17        0.2512     0.7488    0.7488  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             18        0.2476     0.7524    0.7524  Vector                     Cats.md
   ⚪ Chroma                             19        0.2471     0.7529    0.7529  Vector                     Cats.md
   ⚪ Chroma                             20        0.2439     0.7561    0.7561  Vector                     Cats.md
   ⚪ Chroma                             21        0.2388     0.7612    0.7612  Vector                     Cats.md
   ⚪ Chroma                             22        0.2380     0.7620    0.7620  Vector                     Cats.md
   ⚪ Chroma                             23        0.2359     0.7641    0.7641  Vector                     Cats.md
   ⚪ Chroma                             24        0.2356     0.7644    0.7644  Vector                     Cats.md
   ⚪ Chroma                             25        0.2318     0.7682    0.7682  Vector                     Cats.md
   ⚪ Chroma                             26        0.2264     0.7736    0.7736  Vector                     Cats.md
   ⚪ Chroma                             27        0.2261     0.7739    0.7739  Vector                     Cats.md
   ⚪ Chroma                             28        0.2249     0.7751    0.7751  Vector                     Cats.md
   ⚪ Chroma                             29        0.2229     0.7771    0.7771  Vector                     Elephants.jpg
   ⚪ Chroma                             30        0.2225     0.7775    0.7775  Vector                     Cats.md
   ⚪ Chroma                             31        0.2218     0.7782    0.7782  Vector                     Cats.md
   ⚪ Chroma                             32        0.2160     0.7840    0.7840  Vector                     Cats.md
   ⚪ Chroma                             33        0.2158     0.7842    0.7842  Vector                     Cats.md
   ⚪ Chroma                             34        0.2150     0.7850    0.7850  Vector                     Cats.md
   ⚪ Chroma                             35        0.2141     0.7859    0.7859  Vector                     Fish.txt
   ⚪ Chroma                             36        0.2093     0.7907    0.7907  Vector                     Apes.docx
   ⚪ Chroma                             37        0.2076     0.7924    0.7924  Vector                     Kamele.txt
   ⚪ Chroma                             38        0.2028     0.7972    0.7972  Vector                     Cats.md
   ⚪ Chroma                             39        0.1965     0.8035    0.8035  Vector                     Cats.md
   ⚪ Chroma                             40        0.1912     0.8088    0.8088  Vector                     Dogs.png
   ⚪ Chroma                             41        0.1911     0.8089    0.8089  Vector                     Dogs.png
   ⚪ Chroma                             42        0.1825     0.8175    0.8175  Vector                     Cats.md
   ⚪ Chroma                             43        0.1768     0.8232    0.8232  Vector                     Dogs.png
   ⚪ Chroma                             44        0.1765     0.8235    0.8235  Vector                     Cats.md
   ⚪ Chroma                             45        0.1704     0.8296    0.8296  Vector                     Cats.md
   ⚪ Chroma                             46        0.1666     0.8334    0.8334  Vector                     Kamele.txt
   ⚪ Chroma                             47        0.1552     0.8448    0.8448  Vector                     Fish.txt
   ⚪ Chroma                             48        0.1521     0.8479    0.8479  Vector                     Apes.docx
   ⚪ Chroma                             49        0.1500     0.8500    0.8500  Vector                     Cats.md
   ⚪ Chroma                             50        0.1483     0.8517    0.8517  Vector                     Kamele.txt
   ⚪ Chroma                             51        0.1434     0.8566    0.8566  Vector                     Cats.md
   ⚪ Chroma                             52        0.1426     0.8574    0.8574  Vector                     Cats.md
   ⚪ Chroma                             53        0.1423     0.8577    0.8577  Vector                     Apes.docx
   ⚪ Chroma                             54        0.1361     0.8639    0.8639  Vector                     Fish.txt
   ⚪ Chroma                             55        0.1344     0.8656    0.8656  Vector                     Cats.md
   ⚪ Chroma                             56        0.1301     0.8699    0.8699  Vector                     Cats.md
   ⚪ Chroma                             57        0.1267     0.8733    0.8733  Vector                     Kamele.txt
   ⚪ Chroma                             58        0.1262     0.8738    0.8738  Vector                     Pferde.pdf
   ⚪ Chroma                             59        0.1138     0.8862    0.8862  Vector                     Fish.txt
   ⚪ Chroma                             60        0.1136     0.8864    0.8864  Vector                     Pferde.pdf
   ⚪ Chroma                             61        0.1112     0.8888    0.8888  Vector                     Apes.docx
   ⚪ Chroma                             62        0.1112     0.8888    0.8888  Vector                     Cats.md
   ⚪ Chroma                             63        0.0991     0.9009    0.9009  Vector                     Pferde.pdf
   ⚪ Chroma                             64        0.0980     0.9020    0.9020  Vector                     Lions.pptx
   ⚪ Chroma                             65        0.0953     0.9047    0.9047  Vector                     Cats.md
   ⚪ Chroma                             66        0.0951     0.9049    0.9049  Vector                     Kamele.txt
   ⚪ Chroma                             67        0.0900     0.9100    0.9100  Vector                     Lions.pptx
   ⚪ Chroma                             68        0.0815     0.9185    0.9185  Vector                     Cats.md
   ⚪ Chroma                             69        0.0663     0.9337    0.9337  Vector                     Apes.docx
   ⚪ Chroma                             70        0.0638     0.9362    0.9362  Vector                     Fish.txt
   ⚪ Chroma                             71        0.0499     0.9501    0.9501  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             72        0.0441     0.9559    0.9559  Vector                     Pferde.pdf
   ⚪ Chroma                             73        0.0436     0.9564    0.9564  Vector                     LionsAndApes.xlsx
   ⚪ Chroma                             74        0.0217     0.9783    0.9783  Vector                     Lions.pptx
   ⚪ Chroma                             75        0.0058     0.9942    0.9942  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             76        0.0012     0.9988    0.9988  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             77        0.0009     0.9991    0.9991  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             78       -0.0188     1.0188    1.0188  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             79       -0.0221     1.0221    1.0221  Vector                     BlazingFast_Workstation.md
🟢 Chroma                         Querying Chroma DB query returned 79 chunks
🔵 MultiQuery                     3 Alternate queries for vector retrieval:
↳                                 1: 'Dietary habits of hedgehogs'
↳                                 2: 'Habitat and food sources of hedgehogs'
↳                                 3: 'Where do hedgehogs reside and what do they consume?'
🔵 Retrieval Orchestration        discovered corpus language buckets: [de, en] active config buckets: [de, en, es, fr, it]
   ⚪ WordNet Synonyms               Expanded 9 phrases → 13 (+5 synonyms, depth=1, max/phrase=1)
🔵 Retrieval Orchestration        idx_stage l='en' r='BM25/Graph/Regex' tr=1
↳                                 primary_query='what do hedgehogs eat and where do they live?'
↳                                 stage_primary_query='what do hedgehogs eat and where do they live? act erinaceus europaeus consume answer be'
↳                                 guardrail_query='what do hedgehogs eat and where do they live?'
↳                                 stage_guardrail_query='what do hedgehogs eat and where do they live? act erinaceus europaeus consume answer be'
🔵 Retrieval Orchestration        idx_stage l='de' r='BM25/Graph/Regex' tr=1
↳                                 primary_query='what do hedgehogs eat and where do they live?'
↳                                 stage_primary_query='Was essen Igel und wo leben sie? handeln erinaceus europaeus konsumieren Antwort sein'
↳                                 guardrail_query='what do hedgehogs eat and where do they live?'
↳                                 stage_guardrail_query='Was essen Igel und wo leben sie? handeln erinaceus europaeus konsumieren Antwort sein'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: en      query: 'what do hedgehogs eat and where do they live? act erinaceus europaeus consume answer be'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: de      query: 'Was essen Igel und wo leben sie? handeln erinaceus europaeus konsumieren Antwort sein'
🔵 BM25                           Querying bm25 index on collection Test for language en
🟢 BM25                           Loaded persisted BM25 index (79 chunks, 1907 terms)
🟢 BM25                           BM25 retrieval returned 59 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1       13.0137         BM25                Lions.pptx
   ⚪ BM25                                2       10.8805         BM25                Fish.txt
   ⚪ BM25                                3        8.6701         BM25                Hedgehogs.pdf
   ⚪ BM25                                4        8.3204         BM25                Hedgehogs.pdf
   ⚪ BM25                                5        6.4838         BM25                Fish.txt
   ⚪ BM25                                6        6.4837         BM25                Cats.md
   ⚪ BM25                                7        5.7188         BM25                Fish.txt
   ⚪ BM25                                8        5.6459         BM25                Apes.docx
   ⚪ BM25                                9        4.8889         BM25                Hedgehogs.pdf
   ⚪ BM25                               10        4.7334         BM25                Apes.docx
   ⚪ BM25                               11        4.4736         BM25                Cats.md
   ⚪ BM25                               12        4.1869         BM25                Hedgehogs.pdf
   ⚪ BM25                               13        4.1669         BM25                Cats.md
   ⚪ BM25                               14        4.1464         BM25                Elephants.jpg
   ⚪ BM25                               15        3.9108         BM25                Apes.docx
   ⚪ BM25                               16        3.7140         BM25                Cats.md
   ⚪ BM25                               17        3.6804         BM25                Lions.pptx
   ⚪ BM25                               18        3.6416         BM25                Cats.md
   ⚪ BM25                               19        3.5759         BM25                Hedgehogs.pdf
   ⚪ BM25                               20        3.3728         BM25                Cats.md
🔵 BM25                           Querying bm25 index on collection Test for language de
🟢 BM25                           BM25 retrieval returned 10 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        9.8323         BM25                Kamele.txt
   ⚪ BM25                                2        9.8321         BM25                Kamele.txt
   ⚪ BM25                                3        8.9796         BM25                Kamele.txt
   ⚪ BM25                                4        7.7276         BM25                Pferde.pdf
   ⚪ BM25                                5        7.1057         BM25                Kamele.txt
   ⚪ BM25                                6        6.1597         BM25                Kamele.txt
   ⚪ BM25                                7        5.0208         BM25                Pferde.pdf
   ⚪ BM25                                8        4.9228         BM25                Pferde.pdf
   ⚪ BM25                                9        3.0296         BM25                Pferde.pdf
   ⚪ BM25                               10        2.7825         BM25                Kamele.txt
🔵 Graph                          Querying graph index on collection Test for language en
🟢 Graph                          Loaded persisted graph index (79 chunks, 1132 entities)
🟢 Graph                          Graph retrieval returned 50 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1     4043.0000              Graph          Fish.txt
   ⚪ Graph                               2     3227.0000              Graph          Fish.txt
   ⚪ Graph                               3     3122.0000              Graph          Fish.txt
   ⚪ Graph                               4     2779.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               5     2768.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               6     2107.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               7     1914.0000              Graph          Fish.txt
   ⚪ Graph                               8     1661.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               9     1569.0000              Graph          Fish.txt
   ⚪ Graph                              10     1339.0000              Graph          Cats.md
   ⚪ Graph                              11     1016.0000              Graph          Cats.md
   ⚪ Graph                              12      987.0000              Graph          Apes.docx
   ⚪ Graph                              13      968.0000              Graph          Apes.docx
   ⚪ Graph                              14      941.0000              Graph          Cats.md
   ⚪ Graph                              15      884.0000              Graph          Apes.docx
   ⚪ Graph                              16      796.0000              Graph          Lions.pptx
   ⚪ Graph                              17      776.0000              Graph          Cats.md
   ⚪ Graph                              18      665.0000              Graph          Lions.pptx
   ⚪ Graph                              19      648.0000              Graph          Cats.md
   ⚪ Graph                              20      528.0000              Graph          Fish.txt
🔵 Regex                          Querying regex index on collection Test for language en
🟢 Regex                          Loaded persisted regex index (79 chunks, 296 verbs, 935 nouns)
🟢 Regex                          Regex retrieval returned 2 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ Regex                               1        1.1167         1         1                    Regex    Hedgehogs.pdf
   ⚪ Regex                               2        1.1167         1         1                    Regex    Hedgehogs.pdf
🔵 Graph                          Querying graph index on collection Test for language de
🟢 Graph                          Graph retrieval returned 10 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1     1885.0000              Graph          Kamele.txt
   ⚪ Graph                               2     1851.0000              Graph          Pferde.pdf
   ⚪ Graph                               3     1158.0000              Graph          Kamele.txt
   ⚪ Graph                               4      753.0000              Graph          Kamele.txt
   ⚪ Graph                               5      716.0000              Graph          Kamele.txt
   ⚪ Graph                               6      300.0000              Graph          Kamele.txt
   ⚪ Graph                               7      254.0000              Graph          Pferde.pdf
   ⚪ Graph                               8      230.0000              Graph          Pferde.pdf
   ⚪ Graph                               9      135.0000              Graph          Kamele.txt
   ⚪ Graph                              10      110.0000              Graph          Pferde.pdf
🔵 Regex                          Querying regex index on collection Test for language de
🟢 Regex                          Regex retrieval returned 0 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
🔵 Retrieval Orchestration        language: en      local docs fetched vector=79 bm25=69 graph=60 regex=2
🔵 Retrieval Orchestration        language: en      web stage skipped (web_mode=local_only)
🔵 Retrieval Orchestration        language: en      merge candidates and build context
🟢 Merge                          Reciprocal Rank Fusion (RRF) produced 79 local chunks
   ⚪ Merge                             Pos    RRFScore                            Retrievers  File
   ⚪ Merge                          ----------------------------------------------------------------------------------
   ⚪ Merge                               1      0.0640              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               2      0.0635              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               3      0.0458              Vector BM25 Graph         Fish.txt
   ⚪ Merge                               4      0.0453              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                               5      0.0436              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                               6      0.0431              Vector BM25 Graph         Cats.md
   ⚪ Merge                               7      0.0429              Vector BM25 Graph         Apes.docx
   ⚪ Merge                               8      0.0427              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                               9      0.0418              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              10      0.0417              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              11      0.0416              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              12      0.0415              Vector BM25 Graph         Cats.md
   ⚪ Merge                              13      0.0388              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              14      0.0377              Vector BM25 Graph         Cats.md
   ⚪ Merge                              15      0.0376              Vector BM25 Graph         Cats.md
   ⚪ Merge                              16      0.0363              Vector BM25 Graph         Cats.md
   ⚪ Merge                              17      0.0362              Vector BM25 Graph         Cats.md
   ⚪ Merge                              18      0.0361              Vector BM25 Graph         Elephants.jpg
   ⚪ Merge                              19      0.0348              Vector BM25 Graph         Cats.md
   ⚪ Merge                              20      0.0346              Vector BM25 Graph         Cats.md
   ⚪ Merge                              21      0.0345              Vector BM25 Graph         Cats.md
   ⚪ Merge                              22      0.0338              Vector BM25 Graph         Cats.md
   ⚪ Merge                              23      0.0336              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              24      0.0335              Vector BM25 Graph         Cats.md
   ⚪ Merge                              25      0.0333              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              26      0.0330              Vector BM25 Graph         Cats.md
   ⚪ Merge                              27      0.0329              Vector BM25 Graph         Cats.md
   ⚪ Merge                              28      0.0327              Vector BM25 Graph         Cats.md
   ⚪ Merge                              29      0.0326              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              30      0.0325              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              31      0.0319              Vector BM25 Graph         Cats.md
   ⚪ Merge                              32      0.0318              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              33      0.0317              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              34      0.0317              Vector BM25 Graph         Cats.md
   ⚪ Merge                              35      0.0316              Vector BM25 Graph         Cats.md
   ⚪ Merge                              36      0.0315              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              37      0.0314              Vector BM25 Graph         Cats.md
   ⚪ Merge                              38      0.0305              Vector BM25 Graph         Cats.md
   ⚪ Merge                              39      0.0294              Vector BM25 Graph         Cats.md
   ⚪ Merge                              40      0.0288              Vector BM25 Graph         Cats.md
   ⚪ Merge                              41      0.0284              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              42      0.0280              Vector BM25 Graph         Cats.md
   ⚪ Merge                              43      0.0279              Vector BM25 Graph         Cats.md
   ⚪ Merge                              44      0.0277              Vector BM25 Graph         Cats.md
   ⚪ Merge                              45      0.0274              Vector BM25 Graph         Cats.md
   ⚪ Merge                              46      0.0273              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              47      0.0264              Vector BM25 Graph         Cats.md
   ⚪ Merge                              48      0.0263              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              49      0.0261              Vector BM25               Lions.pptx
   ⚪ Merge                              50      0.0255              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              51      0.0250              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              52      0.0249              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              53      0.0249              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              54      0.0248              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              55      0.0246              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              56      0.0243              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              57      0.0229              Vector BM25               Cats.md
   ⚪ Merge                              58      0.0226              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              59      0.0216              Vector BM25               Cats.md
   ⚪ Merge                              60      0.0212              Vector      Graph         Cats.md
   ⚪ Merge                              61      0.0211              Vector      Graph         Dogs.png
   ⚪ Merge                              62      0.0207              Vector      Graph         Dogs.png
   ⚪ Merge                              63      0.0205              Vector      Graph         Dogs.png
   ⚪ Merge                              64      0.0202              Vector      Graph         Fish.txt
   ⚪ Merge                              65      0.0194              Vector BM25               Lions.pptx
   ⚪ Merge                              66      0.0193              Vector BM25               Cats.md
   ⚪ Merge                              67      0.0190              Vector BM25               Cats.md
   ⚪ Merge                              68      0.0182              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              69      0.0180              Vector BM25               Apes.docx
   ⚪ Merge                              70      0.0179              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              71      0.0168              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              72      0.0166              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              73      0.0165              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              74      0.0159              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              75      0.0133              Vector                    Cats.md
   ⚪ Merge                              76      0.0128              Vector                    Cats.md
   ⚪ Merge                              77      0.0120              Vector                    Cats.md
   ⚪ Merge                              78      0.0119              Vector                    Cats.md
   ⚪ Merge                              79      0.0075              Vector                    LionsAndApes.xlsx
🔵 ChunkDedup                     Removed 1 near-duplicate chunk(s) (threshold=0.85, include_web=on, kept 78)
🔵 Rerank                         Pre-rerank pool local=78 web=0 total=78 original_leg=off lang=en added[V/B/G/R]=0/0/0/0
   ⚪ Rerank                            Pos    RawScore    AdjScore                Retrievers  File                                      Text
   ⚪ Rerank                         --------------------------------------------------------------------------------------------------------------
   ⚪ Rerank                              1      2.6867      1.0000  Vector BM25 Graph Regex   Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
   ⚪ Rerank                              2      2.2072      0.9616  Vector BM25 Graph Regex   Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
   ⚪ Rerank                              3      0.9809      0.8633  Vector BM25 Graph         Kamele.txt                                ernährung kamele sind pflanzenfresser (h
   ⚪ Rerank                              4     -0.6821      0.7300  Vector BM25 Graph         Kamele.txt                                sie liefern milch, fleisch, wolle und le
   ⚪ Rerank                              5     -0.7112      0.7277  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
   ⚪ Rerank                              6     -1.0439      0.7011  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
   ⚪ Rerank                              7     -1.1815      0.6900  Vector BM25 Graph         Kamele.txt                                weitere körperliche anpassungen: - dicht
   ⚪ Rerank                              8     -1.2745      0.6826  Vector BM25 Graph         Apes.docx                                 Gorillas are the largest living primates
   ⚪ Rerank                              9     -2.0536      0.6202  Vector BM25 Graph         Kamele.txt                                kamele – überlebenskünstler der wüste ka
   ⚪ Rerank                             10     -2.0723      0.6187  Vector BM25 Graph         Elephants.jpg                             elephants large, long-lived mammals fami
   ⚪ Rerank                             11     -2.7214      0.5666  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
   ⚪ Rerank                             12     -3.0999      0.5363  Vector BM25 Graph         Apes.docx                                 Orangutans are the only great apes found
   ⚪ Rerank                             13     -3.3741      0.5143  Vector BM25 Graph         Apes.docx                                 Chimpanzees (Pan troglodytes) are found
   ⚪ Rerank                             14     -3.4587      0.5076  Vector BM25 Graph         Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
   ⚪ Rerank                             15     -3.5276      0.5020  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
   ⚪ Rerank                             16     -4.4383      0.4291  Vector BM25               Apes.docx                                 All great apes demonstrate remarkable co
   ⚪ Rerank                             17     -4.4580      0.4275  Vector BM25 Graph         Kamele.txt                                in einigen kulturen spielen kamele auch
   ⚪ Rerank                             18     -4.4847      0.4253  Vector BM25 Graph         Cats.md                                   cats have lived alongside humans for tho
   ⚪ Rerank                             19     -4.4933      0.4247  Vector      Graph         Dogs.png                                  tasks. understanding breed-specific ende
   ⚪ Rerank                             20     -4.6624      0.4111  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
   ⚪ Rerank                             21     -4.7071      0.4075  Vector BM25 Graph         Cats.md                                   cats are crepuscular hunters with: - a h
   ⚪ Rerank                             22     -4.7236      0.4062  Vector BM25 Graph         Fish.txt                                   what fish are fish are a diverse group
   ⚪ Rerank                             23     -4.7514      0.4040  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
   ⚪ Rerank                             24     -4.9828      0.3854  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
   ⚪ Rerank                             25     -5.0429      0.3806  Vector      Graph         Dogs.png                                  , comprehensive text dogs dogs accompani
   ⚪ Rerank                             26     -5.0521      0.3799  Vector BM25 Graph         Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
   ⚪ Rerank                             27     -5.0812      0.3775  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
   ⚪ Rerank                             28     -5.4115      0.3511  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
   ⚪ Rerank                             29     -5.4966      0.3443  Vector BM25 Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
   ⚪ Rerank                             30     -5.6439      0.3324  Vector BM25 Graph         Cats.md                                   cats can detect frequencies up to ~64 kh
   ⚪ Rerank                             31     -5.7690      0.3224  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
   ⚪ Rerank                             32     -5.9925      0.3045  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
   ⚪ Rerank                             33     -6.0600      0.2991  Vector BM25 Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
   ⚪ Rerank                             34     -6.1389      0.2928  Vector BM25 Graph         Cats.md                                   cats communicate through: - vocalization
   ⚪ Rerank                             35     -6.1555      0.2915  Vector BM25 Graph         Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
   ⚪ Rerank                             36     -6.3237      0.2780  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
   ⚪ Rerank                             37     -6.4097      0.2711  Vector BM25 Graph         Fish.txt                                  diversity: fish include jawless fishes (
   ⚪ Rerank                             38     -6.5755      0.2578  Vector BM25 Graph         Cats.md                                   cats suffered during periods of supersti
   ⚪ Rerank                             39     -6.6354      0.2530  Vector BM25 Graph         Cats.md                                   cats were not bred for specific tasks ea
   ⚪ Rerank                             40     -6.6393      0.2527  Vector BM25 Graph         Cats.md                                   modern domestic cats descend from the ne
   ⚪ Rerank                             41     -6.7141      0.2467  Vector BM25 Graph         Cats.md                                   kittens engage in: - stalking. - pouncin
   ⚪ Rerank                             42     -6.7945      0.2402  Vector BM25 Graph         Apes.docx                                 Great apes (Hominidae) are the closest l
   ⚪ Rerank                             43     -6.8205      0.2382  Vector BM25 Graph         Cats.md                                   cats were revered, often associated with
   ⚪ Rerank                             44     -7.1052      0.2154  Vector BM25 Graph         Cats.md                                   free-roaming cats are estimated to kill
   ⚪ Rerank                             45     -7.2450      0.2042  Vector BM25 Graph         Cats.md                                   cats are complex, adaptable, and endless
   ⚪ Rerank                             46     -7.2801      0.2013  Vector BM25 Graph         Cats.md                                   some breeds have known predispositions:
   ⚪ Rerank                             47     -7.3145      0.1986  Vector BM25 Graph         Cats.md                                   cats are often described as solitary, bu
   ⚪ Rerank                             48     -7.3363      0.1968  Vector BM25 Graph         Cats.md                                   these breeds emerged without heavy human
   ⚪ Rerank                             49     -7.3889      0.1926  Vector BM25 Graph         Cats.md                                   even well-fed cats hunt. the sequence: 1
   ⚪ Rerank                             50     -7.4232      0.1899  Vector BM25               Cats.md                                   spaying/neutering is essential to reduce
🔵 Rerank                         Reranking with cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 returned 78 chunks
🔵 Chunk selection                Strategy 'DEFAULT' → ScoreRankedSelector
🔵 Score select                   logit+ = effective logit after single-chunk boost (raw in raw_rerank_score)
🟣 Score select                         Logit [Sigmoid]       Thr      ΔProb  Retrievers                File                                      Text
🟣 Score select                   --------------------------------------------------------------------------------------------------------------------
🟣 Score select                    ❌   -9.7927 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
🟣 Score select                    ❌   -9.7750 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
🟣 Score select                    ❌   -9.7328 [0.000]  (0.5000)    -0.4999  Vector      Graph         Fish.txt                                  poikilotherm: a term describing organism
🟣 Score select                    ❌   -9.6343 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Pferde.pdf                                Page 2 KI-Modelle, die semantische Feinh
🟣 Score select                    ❌   -9.4530 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
🟣 Score select                    ❌   -9.3623 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Fish.txt                                  terminology explained cold-blooded: a co
🟣 Score select                    ❌   -9.2503 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                the blazingfast workstation is an enterp
🟣 Score select                    ❌   -9.1838 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Cats.md                                   indoor cats often live 12–18 years; some
🟣 Score select                    ❌   -9.1820 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Cats.md                                   outdoor access: - pros: exercise, stimul
🟣 Score select                    ❌   -9.0268 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **processors**: dual enterprise platin
🟣 Score select                    ❌   -8.7668 [0.000]  (0.5000)    -0.4998  Vector                    Cats.md                                   a deep, structured exploration of domest
🟣 Score select                    ❌   -8.7263 [0.000]  (0.5000)    -0.4998  Vector BM25               Cats.md                                   - dental disease - kidney disease - hype
🟣 Score select                    ❌   -8.6091+[0.000]  (0.5000)    -0.4998  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
🟣 Score select                    ❌   -8.5323 [0.000]  (0.5000)    -0.4998  Vector BM25 Graph         Cats.md                                   - free feeding can lead to obesity. - pu
🟣 Score select                    ❌   -8.4253 [0.000]  (0.5000)    -0.4998  Vector BM25               BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
🟣 Score select                    ❌   -8.1204 [0.000]  (0.5000)    -0.4997  Vector                    Cats.md                                   - puzzle toys - training sessions - nove
🟣 Score select                    ❌   -8.0141 [0.000]  (0.5000)    -0.4997  Vector                    Cats.md                                   - human interaction - multi-cat househol
🟣 Score select                    ❌   -7.9860 [0.000]  (0.5000)    -0.4997  Vector BM25 Graph         Cats.md                                   the mechanism of purring is still debate
🟣 Score select                    ❌   -7.8690 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Cats.md                                   cats form strong bonds but express affec
🟣 Score select                    ❌   -7.7939 [0.000]  (0.5000)    -0.4996  Vector BM25               Cats.md                                   selective breeding intensified in the 19
🟣 Score select                    ❌   -7.7532 [0.000]  (0.5000)    -0.4996  Vector BM25               Hedgehogs.pdf                             Page 2 behavioral ecology, hibernation p
🟣 Score select                    ❌   -7.7205 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Fish.txt                                  ectotherm: the preferred scientific term
🟣 Score select                    ❌   -7.7091 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Cats.md                                   - kingdom: animalia - phylum: chordata -
🟣 Score select                    ❌   -7.6973 [0.000]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   most adult cats are lactose intolerant.
🟣 Score select                    ❌   -7.6487 [0.000]  (0.5000)    -0.4995  Vector      Graph         Cats.md                                   cats dominate digital culture: - memes (
🟣 Score select                    ❌   -7.6145 [0.000]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   they have a righting reflex, but falls f
🟣 Score select                    ❌   -7.5107 [0.001]  (0.5000)    -0.4995  Vector                    Cats.md                                   - climbing structures - scratching posts
🟣 Score select                    ❌   -7.4619 [0.001]  (0.5000)    -0.4994  Vector BM25               Cats.md                                   - regular veterinary checkups - vaccinat
🟣 Score select                    ❌   -7.4232 [0.001]  (0.5000)    -0.4994  Vector BM25               Cats.md                                   spaying/neutering is essential to reduce
🟣 Score select                    ❌   -7.3889 [0.001]  (0.5000)    -0.4994  Vector BM25 Graph         Cats.md                                   even well-fed cats hunt. the sequence: 1
🟣 Score select                    ❌   -7.3363 [0.001]  (0.5000)    -0.4993  Vector BM25 Graph         Cats.md                                   these breeds emerged without heavy human
🟣 Score select                    ❌   -7.3145 [0.001]  (0.5000)    -0.4993  Vector BM25 Graph         Cats.md                                   cats are often described as solitary, bu
🟣 Score select                    ❌   -7.2801 [0.001]  (0.5000)    -0.4993  Vector BM25 Graph         Cats.md                                   some breeds have known predispositions:
🟣 Score select                    ❌   -7.2450 [0.001]  (0.5000)    -0.4993  Vector BM25 Graph         Cats.md                                   cats are complex, adaptable, and endless
🟣 Score select                    ❌   -7.1052 [0.001]  (0.5000)    -0.4992  Vector BM25 Graph         Cats.md                                   free-roaming cats are estimated to kill
🟣 Score select                    ❌   -6.8205 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph         Cats.md                                   cats were revered, often associated with
🟣 Score select                    ❌   -6.7945 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph         Apes.docx                                 Great apes (Hominidae) are the closest l
🟣 Score select                    ❌   -6.7141 [0.001]  (0.5000)    -0.4988  Vector BM25 Graph         Cats.md                                   kittens engage in: - stalking. - pouncin
🟣 Score select                    ❌   -6.6393 [0.001]  (0.5000)    -0.4987  Vector BM25 Graph         Cats.md                                   modern domestic cats descend from the ne
🟣 Score select                    ❌   -6.6354 [0.001]  (0.5000)    -0.4987  Vector BM25 Graph         Cats.md                                   cats were not bred for specific tasks ea
🟣 Score select                    ❌   -6.5755 [0.001]  (0.5000)    -0.4986  Vector BM25 Graph         Cats.md                                   cats suffered during periods of supersti
🟣 Score select                    ❌   -6.4097 [0.002]  (0.5000)    -0.4984  Vector BM25 Graph         Fish.txt                                  diversity: fish include jawless fishes (
🟣 Score select                    ❌   -6.3237 [0.002]  (0.5000)    -0.4982  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
🟣 Score select                    ❌   -6.1555 [0.002]  (0.5000)    -0.4979  Vector BM25 Graph         Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
🟣 Score select                    ❌   -6.1389 [0.002]  (0.5000)    -0.4978  Vector BM25 Graph         Cats.md                                   cats communicate through: - vocalization
🟣 Score select                    ❌   -6.0600 [0.002]  (0.5000)    -0.4977  Vector BM25 Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
🟣 Score select                    ❌   -5.9925 [0.002]  (0.5000)    -0.4975  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
🟣 Score select                    ❌   -5.7690 [0.003]  (0.5000)    -0.4969  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
🟣 Score select                    ❌   -5.6439 [0.004]  (0.5000)    -0.4965  Vector BM25 Graph         Cats.md                                   cats can detect frequencies up to ~64 kh
🟣 Score select                    ❌   -5.4966 [0.004]  (0.5000)    -0.4959  Vector BM25 Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
🟣 Score select                    ❌   -5.4115 [0.004]  (0.5000)    -0.4956  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
🟣 Score select                    ❌   -5.0812 [0.006]  (0.5000)    -0.4938  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
🟣 Score select                    ❌   -5.0521 [0.006]  (0.5000)    -0.4936  Vector BM25 Graph         Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
🟣 Score select                    ❌   -5.0429 [0.006]  (0.5000)    -0.4936  Vector      Graph         Dogs.png                                  , comprehensive text dogs dogs accompani
🟣 Score select                    ❌   -4.9828 [0.007]  (0.5000)    -0.4932  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
🟣 Score select                    ❌   -4.7514 [0.009]  (0.5000)    -0.4914  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
🟣 Score select                    ❌   -4.7236 [0.009]  (0.5000)    -0.4912  Vector BM25 Graph         Fish.txt                                   what fish are fish are a diverse group
🟣 Score select                    ❌   -4.7071 [0.009]  (0.5000)    -0.4910  Vector BM25 Graph         Cats.md                                   cats are crepuscular hunters with: - a h
🟣 Score select                    ❌   -4.6624 [0.009]  (0.5000)    -0.4906  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
🟣 Score select                    ❌   -4.4933 [0.011]  (0.5000)    -0.4889  Vector      Graph         Dogs.png                                  tasks. understanding breed-specific ende
🟣 Score select                    ❌   -4.4847 [0.011]  (0.5000)    -0.4888  Vector BM25 Graph         Cats.md                                   cats have lived alongside humans for tho
🟣 Score select                    ❌   -4.4580 [0.011]  (0.5000)    -0.4885  Vector BM25 Graph         Kamele.txt                                in einigen kulturen spielen kamele auch
🟣 Score select                    ❌   -4.4383 [0.012]  (0.5000)    -0.4883  Vector BM25               Apes.docx                                 All great apes demonstrate remarkable co
🟣 Score select                    ❌   -3.5276 [0.029]  (0.5000)    -0.4715  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
🟣 Score select                    ❌   -3.4587 [0.031]  (0.5000)    -0.4695  Vector BM25 Graph         Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
🟣 Score select                    ❌   -3.3741 [0.033]  (0.5000)    -0.4669  Vector BM25 Graph         Apes.docx                                 Chimpanzees (Pan troglodytes) are found
🟣 Score select                    ❌   -3.0999 [0.043]  (0.5000)    -0.4569  Vector BM25 Graph         Apes.docx                                 Orangutans are the only great apes found
🟣 Score select                    ❌   -2.7214 [0.062]  (0.5000)    -0.4383  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
🟣 Score select                    ❌   -2.0536 [0.114]  (0.5000)    -0.3863  Vector BM25 Graph         Kamele.txt                                kamele – überlebenskünstler der wüste ka
🟣 Score select                    ❌   -1.8491+[0.136]  (0.5000)    -0.3640  Vector BM25 Graph         Elephants.jpg                             elephants large, long-lived mammals fami
🟣 Score select                    ❌   -1.2745 [0.218]  (0.5000)    -0.2815  Vector BM25 Graph         Apes.docx                                 Gorillas are the largest living primates
🟣 Score select                    ❌   -1.1815 [0.235]  (0.5000)    -0.2652  Vector BM25 Graph         Kamele.txt                                weitere körperliche anpassungen: - dicht
🟣 Score select                    ❌   -1.0439 [0.260]  (0.5000)    -0.2396  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
🟣 Score select                    ❌   -0.7112 [0.329]  (0.5000)    -0.1707  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
🟣 Score select                    ❌   -0.6821 [0.336]  (0.5000)    -0.1642  Vector BM25 Graph         Kamele.txt                                sie liefern milch, fleisch, wolle und le
🟣 Score select                    ✅    2.6867 [0.936]  (0.5000)    +0.4362  Vector BM25 Graph Regex   Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
🟣 Score select                    ✅    2.2072 [0.901]  (0.5000)    +0.4009  Vector BM25 Graph Regex   Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
🟣 Score select                    ✅    0.9809 [0.727]  (0.5000)    +0.2273  Vector BM25 Graph         Kamele.txt                                ernährung kamele sind pflanzenfresser (h
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (default): LOW (confidence: 42%, threshold: 64%)
🔵 Rerank fallback                Low-recall rescue action: Added 3 query-overlap local chunk(s) from retrieval order (input_fetch_k=100 input_context_chunks=50 fixed_k=100 min_local_pool=40
↳                                 target_min_local_hits=24 max_additional=24; local hits 3->6, pool=78).
🔵 Strategy: default              6 chunks selected after thresholding  0.5000  single-chunk boost ×1.25  low-recall rescue +3
🔵 Chunk selector: Score          Ranked     Selected 6 chunks.
🔵 Selected                       6 chunks selected
   ⚪ Selected                        #   Score [Sigmoid]  File                                      Text
   ⚪ Selected                       --------------------------------------------------------------------------------------------------------------------------------------------
   ⚪ Selected                        1    1.0000 [0.936]  Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are small, nocturnal mammals known
   ⚪ Selected                        2    0.9616 [0.901]  Hedgehogs.pdf                             Page 1 suburban gardens and agricultural edges. They are native to muc
   ⚪ Selected                        3    0.8633 [0.727]  Kamele.txt                                ernährung kamele sind pflanzenfresser (herbivoren) und fressen auch tr
   ⚪ Selected                        4    0.7277 [0.329]  Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive bite to subdue it. Seasonal s
   ⚪ Selected                        5    0.7011 [0.260]  Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations include a keen sense of smell and
   ⚪ Selected                        6    0.5020 [0.029]  Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as netting, open drains, and bon
   ⚪ ChunkSelect                    After chunk selection: 6/78 kept
🔵 Selection Confidence           🟢 HIGH C_final=0.79 evidence_chunks=6 rerank_fallback=off
🔵 Retrieval Orchestration        language: en      retrieval completed chosen=6 context_chars=7491
🔵 TokenBudget                    [mistral:7b] context=32768 reserved_sys=1024 prompt≈2661 → max_output_tokens=2048
🔵 Resolved token params          max_output_tokens(api: max_tokens)=2048  num_ctx=32768  (override: max_tokens=None  num_ctx=None)
🔵 LLM Plan                       Answer generation: producing the final grounded response from retrieved context chunks.
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈2661 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_CHAT stage: Run user prompt
🔵 Call LLM                       options: {'temperature': 0.1, 'top_k': 40, 'top_p': 0.92, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:09
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_compliance_language): HIGH (confidence: 100%, threshold: 60%)
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PIPELINE_CHECK
🟢 KeyWrdChk Depth                0 algos passed threshold vs. required 4
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 4
🔵 Deleted chat context           Deleted chat context for MyFirstChat in Test_ChatContext
🔵 Add chat context               Upserted turn 1 for chat_name=MyFirstChat file_tag='' lang='english' to Test_ChatContext
   ⚪ Retrieval Orchestration        Stored query history forms orig='what do hedgehogs eat and where do they live?' t1='what do hedgehogs eat and where do they live?' t2='what do hedgehogs eat and
   ↳                                 where do they live?' effective_reason=''
🔵 Masker                         0 of 15 rules produced matches and were replaced
💬 >   Hedgehogs eat invertebrates, such as beetles, caterpillars, earthworms, snails, and slugs, as well as amphibians, small rodents, eggs, carrion, and fallen fruit when available. They live in
💬 >  various environments, including suburban gardens, agricultural edges, hedgerows, compost heaps, and mixed vegetation. They are native to much of Europe, parts of Asia, and Africa.
💬 >
💬 >  ### Sources
💬 >  - FileName: Hedgehogs.pdf
💬 >    - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf ([open](file:///D:/RAG-LCC/TestDocs/Hedgehogs.pdf))
💬 >    - Page: 1, 2
💬 >
💬 >  ---
💬 >
💬 >  **Answer confidence:**
💬 >  🟢 HIGH
💬 >  - C_final=0.79            : final confidence in [0,1] after penalties.
💬 >  - C_top=0.47              : weighted best-chunk signal (0.50 x top_score).
💬 >  - C_mean_top3=0.26        : weighted top-3 stability (0.30 x mean_top3).
💬 >  - C_coverage=0.02         : weighted evidence coverage (0.15 x coverage).
💬 >  - C_local=0.05            : weighted local-source share (0.05 x local_share).
💬 >  - C_fallback_penalty=0.00 : subtraction when low-confidence rerank fallback was used.
💬 >  - top_score=0.94          : strongest selected chunk confidence.
💬 >  - mean_top3=0.85          : average confidence of the top 3 selected chunks.
💬 >  - coverage=0.12           : selected_chunks / final_chunks_to_llm, capped at 1.00.
💬 >  - local_share=1.00        : fraction of selected chunks from local files.
💬 >  - evidence_chunks=6       : number of selected evidence chunks.
💬 >  - rerank_fallback=off     : whether fallback rerank logic was triggered.
🟡 VisualMarker                   _mark_sources called: 6 chunk(s)
🔵 VisualMarker                   Prepared 1 highlighted document(s) (in memory)


---
### Document metadata

- **Hedgehogs.pdf**
    - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf
    - Creator: Microsoft® Word for Microsoft 365
    - Producer: Microsoft® Word for Microsoft 365
    - DocCreated: 2026-02-17 12:32:04+01:00
    - DocModified: 2026-02-17 12:32:04+01:00
    - Pages: 1, 2
- **Kamele.txt**
    - FilePath: D:/RAG-LCC/TestDocs/Kamele.txt
    - FileSizeBytes: 3487
    - FileModified: 2026-06-23 14:17:15.094329
🔵 Marked sources                 1 highlighted document(s) (click a link to open; use 'Save As' in the viewer to keep a copy):
   📎 Hedgehogs.pdf (highlighted): file:///D:/RAG-LCC/tmp/rag_marked_ws8xf_yq/Hedgehogs_marked.pdf
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
 ```
