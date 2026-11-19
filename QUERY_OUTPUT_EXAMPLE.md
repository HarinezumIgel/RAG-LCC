# Query Output Example

## Current confidence display

Please note that the actual output includes more color; copying and pasting from the VS Code terminal may reduce it.

```text
(.venv) PS D:\RAG-LCC> python .\src\Apps\RAGChat.py --collection Test
🔵 Config path                    Checking configured filesystem path slots (root guard + presence check).
🔵 Config path                    For file-like slots, present means the containing directory exists.
🔵 Config path                    DOC_DIR -> D:\RAG-LCC\TestDocs present: YES
🔵 Config path                    LOG_FILE -> D:\RAG-LCC\compliance.log present (containing dir): YES
🔵 Config path                    _ABSOLUTE_PATH -> D:\RAG-LCC present: YES
🔵 Config path                    _CACHE_PATH_ABSOLUTE -> D:\RAG-LCC\.caches present: YES
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
🦔            Version v0.5.2.1/1660 2026-11-19                          🦔
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
🔵 Environment variable           SERVE_RAGCHAT_GROUNDED_DOCS=1

🟢 Outbound downloads             One or more settings allow outbound model/data downloads Set the relevant offline flags if you want to prevent network access.
🟡 Web search                     Web search is ENABLED (WEB_SEARCH_MODE="1"). User queries may be sent to the internet. Review LEGAL.md § Web Search
↳                                 and SECURITY.md before deploying.


🔵 VLLM                           Access to vLLM OpenAI-compatible Provider is enabled. http://192.168.100.50:4000/v1/chat/completions streaming: False
🔵 VLLM                           Probing http://192.168.100.50:4000/v1/models → OK
🟢 HF cache                       HF home: C:\Users\pfm\.cache HF hub cache: C:\Users\pfm\.cache\.hf-cache

                                  Running RAGChat. Good luck!
↳
   ⚪ Argos Translate                Lang: Installed English                  (en   )
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

🔵 License                        Checking license consent for active models: llama_guard._LLM_CHK, qwen2_5_7b_instruct._LLM, qwen2_5_7b_instruct._LLM_REWRITE_PROMPT, qwen2_5_14b_instruct._LLM_JUDGE,
↳                                 bge_m3._EMBED, bge_reranker._CROSS, vllm._VLLM
🟢 License                        All LICENSE.txt files for active models in Config_Models.py (see _MODELS) are consented
🟢 Argos License                  Consent valid for 8 configured language pair(s)
🟢 spaCy License                  Consent valid for 5 configured model(s)
🔵 HF Download                    _MODELS._EMBED: using remote vLLM model; skipping local weight download. Model licence verification remains required by Compliance.
🔵 HF Download                    _MODELS._CROSS: using remote vLLM model; skipping local weight download. Model licence verification remains required by Compliance.
🟢 BM25Scorer                     Initialized BM25 Scorer (algo=BM25) with 80 base phrase(s)
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🔵 TokenBudget                    /v1/models did not yield max_model_len for 'openai/qwen2_5_7b_instruct'; retrying via LiteLLM /model/info endpoint
🔵 TokenBudget                    Detected context_length=32768 for openai/qwen2_5_7b_instruct via vllm adapter
🔵 TokenBudget                    Model 'openai/qwen2_5_7b_instruct': context_limit=32768 (below cap 32768 — using detected value)
🔵 vLLM                           Using vLLM cross-encoder 'bge-reranker' at http://192.168.100.50:4000/v1
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🟢 Regex                          Aux-lemma cache built for 5 active language(s)
→ Loaded defaults for 'DEFAULT':

  ▶ Chat settings
    ▶ Debug                 : debug_level=30  debug_mode='ge'
    ▶ Chat Context          : use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Talk with             : collection='Test_embedder_bge_m3'  chat_name='MyFirstChat'
    ▶ File Input            : file=None  path=None  metadata=None  file_cap=15
    ▶ Visual                : mark_text=True
    ▶ Response              : preferred_response_language=None

  ▶ Summarize settings
    ▶ Summary               : summary_length='medium'  summary_temperature=0.3  chunk_overlap=5

  ▶ Retrieval settings
    ▶ Retrieval strategy    : retrieval strategy='DEFAULT'
    ▶ Strategy overrides    :
    ▶ LLM                   : temperature=0.1  top_p=0.92  top_k=40
    ▶ Retriever tuning      : fetch_k=100  context_chunks=50  threshold=0.5
    ▶ Retriever weights     : vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0  web_weight=0.5
    ▶ Context/session       : file_cap=15  use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Output                : max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
    ▶ API passthrough       : extraOllamaOptions=None  ollamaTopLevelParams=None
    ▶ Web                   : web_search='local_only'  fetch_page_content='snippets only'  web_fetch_k=10  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5

  ▶ Orchestration settings
    ▶ Orchestration strategy: orchestrator_flow='NATIVE_ALL'
    ▶ Orchestrator overrides:
    ▶ Confidence            : confidence_evaluation='both'
    ▶ Flow selectors        : force_retrieve_mode=None  main_query_source=None  secondary_query_source=None  rerank_query_source=None  retrieval_language=None  rerank=None
    ▶ Flow stages           : secondary_query=None  per_language_indexed_query=None  indexed_translate=None  indexed_wordnet=None  orig_lang_vector=None  vector_alternates=None
    ▶ Flow retrievers       : vector=None  bm25=None  graph=None  regex=None
    ▶ Flow runtime          : query_rewrite=None  keyword_action_splitter=None  pronoun_subst=None  rerank_stage=None  low_score_fallback=None  low_recall_rescue=None  grounding=None

🔵 Masker                         Loaded _MASKING_REGEXES from configuration
🟢 Banned Word Check              *** Banned word check status by stage and algorithm:
🟢 Banned Word Check              - PROMPT_CHECK  : Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅  Prompt Check: ✅
🟢 Banned Word Check              - PIPELINE_CHECK: Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅  Pipeline Check: ✅
🟢 Banned Word Check              - MASKING       : ✅ applied on final result

🟢 Banned Words                   ✅ Current banned words: 80
🟢 Masking regexes                ✅ Current masking regexes: 23
Sleeping for 1 seconds .
🔵 VLLM                           Probing http://192.168.100.50:4000/v1/models → OK
🟢 VLLM                           VLLM is reachable on: http://192.168.100.50:4000/v1/models streaming: False
🟢 GPU                            GPU is available
🔵 GPU                            Number of GPUs: 1
🔵 GPU                            GPU Name: Tesla T4
   -                              ----------------------
🔵 LLM for prompt compliance      llama_guard3_8b (Built with Meta Llama 3.)
🔵 LLM for prompt compliance      Llama Guard 3 is licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved.
🔵 LLM for user chat              LLM: Qwen2.5-7B-Instruct is licensed under the Apache License, Version 2.0.
🔵 LLM for user chat              openai/qwen2_5_7b_instruct (LLM: Qwen2.5 7B Instruct)
Chat with your documents


  ▶ Chat settings
    ▶ Debug                 : debug_level=30  debug_mode='ge'
    ▶ Chat Context          : use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Talk with             : collection='Test_embedder_bge_m3'  chat_name='MyFirstChat'
    ▶ File Input            : file=None  path=None  metadata=None  file_cap=15
    ▶ Visual                : mark_text=True
    ▶ Response              : preferred_response_language=None

  ▶ Summarize settings
    ▶ Summary               : summary_length='medium'  summary_temperature=0.3  chunk_overlap=5

  ▶ Retrieval settings
    ▶ Retrieval strategy    : retrieval strategy='DEFAULT'
    ▶ Strategy overrides    :
    ▶ LLM                   : temperature=0.1  top_p=0.92  top_k=40
    ▶ Retriever tuning      : fetch_k=100  context_chunks=50  threshold=0.5
    ▶ Retriever weights     : vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0  web_weight=0.5
    ▶ Context/session       : file_cap=15  use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
    ▶ Output                : max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
    ▶ API passthrough       : extraOllamaOptions=None  ollamaTopLevelParams=None
    ▶ Web                   : web_search='local_only'  fetch_page_content='snippets only'  web_fetch_k=10  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5

  ▶ Orchestration settings
    ▶ Orchestration strategy: orchestrator_flow='NATIVE_ALL'
    ▶ Orchestrator overrides:
    ▶ Confidence            : confidence_evaluation='both'
    ▶ Flow selectors        : force_retrieve_mode=None  main_query_source='REWRITTEN_QUERY'  secondary_query_source='TRANSLATED_QUERY'  rerank_query_source='REWRITTEN_QUERY'  retrieval_language='NATIVE'  rerank=True
    ▶ Flow stages           : secondary_query=False  per_language_indexed_query=True  indexed_translate=True  indexed_wordnet=False  orig_lang_vector=False  vector_alternates=True
    ▶ Flow retrievers       : vector=True  bm25=True  graph=True  regex=True
    ▶ Flow runtime          : query_rewrite=True  keyword_action_splitter=True  pronoun_subst=True  rerank_stage=True  low_score_fallback=True  low_recall_rescue=True  grounding=True

help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
type new: your question to start a new topic  · type summarize: [file name or path] (alias: summary:) to summarize a file (no argument: pinned document; no pin: open the path picker)  · b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat
💬 Your actual query>  what do hedgehogs eat and where do they live? Create a table.

🔵 LangDetect                     Detected language: English (en) — confidence: 59% below threshold 60% — falling back to English
🔵 LangDetect                     Confidence level (default): LOW (confidence: 59%, threshold: 60%)
🔵 Call LLM                       Stage: Encode
🔵 Call LLM                       Model: bge-m3-embedder prompt template: None purpose: Embedding 1 text(s) via http://192.168.100.50:4000/v1/embeddings
🟢 Cache build Regex Banned       Built compiled Regex cache with 80 entries for language english stage: PROMPT_CHECK
🟢 Cache build Jaccard Banned     Built Jaccard n-gram cache with 80 entries for language english
🟢 BM25 Scorer Cache              Built BM25 banlist cache with 80 entries for language english
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🟢 Cache build Keybert Scorer     Built KeyBert embeddings cache with 80 entries
🟢 KeyWrdChk Depth                0 algos passed threshold vs. required 2
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 3
🔵 TokenBudget                    /v1/models did not yield max_model_len for 'llama_guard3_8b'; retrying via LiteLLM /model/info endpoint
🔵 TokenBudget                    Detected context_length=1024 for llama_guard3_8b via vllm adapter
🔵 TokenBudget                    Model 'llama_guard3_8b': context_limit=1024 (below cap 32768 — using detected value)
🔵 TokenBudget                    [llama_guard3_8b] context=1024 reserved_sys=64 prompt≈15 → max_output_tokens=64
🔵 LLM Plan                       Prompt validation: running the guard model to classify the user prompt as allow/block.
🔵 TokenBudget                    [llama_guard3_8b] num_ctx=1024 prompt≈15 → num_predict=64
🔵 Call LLM                       Stage: Check provided prompt
🔵 Call LLM                       Model: llama_guard3_8b prompt template: _PROMPT_CHECK_CHAT_LLAMA_GUARD purpose: Prompt check
🔵 Call LLM                       options: {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': 64, 'num_ctx': 1024} streaming: False
🔵 Call LLM                       Elapsed time calling: llama_guard3_8b took 00:07
🟢 CheckPrompt                    Provided prompt is considered compliant by: llama_guard3_8b. Reason: Detected safe keywords
🔵                                Chatter RAG Query LMM: openai/qwen2_5_7b_instruct
🔵 Retrieval Orchestration        Chosen orchestration flow: NATIVE_ALL
🔵 Retrieval Orchestration        start retrieval turn
🟢 Chroma Collection              Using Chroma DB collection Test_embedder_bge_m3
🔵 VectorStore                    Set Chroma vector store. Name: Test_embedder_bge_m3 Path: D:\RAG-LCC\chromadb\docs\Test_embedder_bge_m3
🔵 Retrieval Orchestration        prepare session context
🔵 Retrieval Orchestration        normalize user query
🔵 UserQuery                      Original user query: 'what do hedgehogs eat and where do they live? Create a table.'
🔵 LangDetect                     Detected language: English (en) — confidence: 59% below threshold 60% — falling back to English
🔵 LangDetect                     Confidence level (raw_user_query): LOW (confidence: 59%, threshold: 60%)
   ⚪ QueryPlan                      Step 1: detected query language=english; query='what do hedgehogs eat and where do they live? Create a table.'
🔵 QueryRewrite                   No conversation history — skipping rewrite
🔵 LangDetect                     Detected language: English (en) — confidence: 59% below threshold 60% — falling back to English
🔵 LangDetect                     Confidence level (rewrite_query): LOW (confidence: 59%, threshold: 60%)
   ⚪ QueryPlan                      Step 2: rewrite completed; language=english; query='what do hedgehogs eat and where do they live? Create a table.'
🔵 LangDrift                      raw_query_language=english  orig_translated_query_en='what do hedgehogs eat and where do they live? Create a table.'
🔵 LangDrift                      rewrite_language=english  rewritten_query='what do hedgehogs eat and where do they live? Create a table.'
🔵 LangDrift                      post_rewrite_query_en='what do hedgehogs eat and where do they live? Create a table.'
🔵 FinalQuery                     Final query for retrieval: 'what do hedgehogs eat and where do they live? Create a table.' (unchanged)
   ⚪ QueryPlan                      Step 3: generated 0 alternatives after rewrite: []
🔵 Retrieval Orchestration        query rewrite flow:
↳                                 seed_en='what do hedgehogs eat and where do they live? Create a table.'
↳                                 rewritten='what do hedgehogs eat and where do they live? Create a table.'
↳                                 final_en='what do hedgehogs eat and where do they live? Create a table.'
🔵 Retrieval Orchestration        language: en      normalized query final='what do hedgehogs eat and where do they live? Create a table.' alternate_count=0
🔵 Retrieval Orchestration        language: en      apply retrieval gates
🟡 IntentFilter                   Dual-use query detected — score=40, reasons=['synthesis_manufacture']. Proceeding with retrieval.
🔵 KeywordActionSplitter          model='openai/qwen2_5_7b_instruct' language='english' inputs={'original_request': 'what do hedgehogs eat and where do they live? Create a table.',
↳                                 'translated_request': 'what do hedgehogs eat and where do they live? Create a table.', 'accepted_rewrite': 'what do hedgehogs eat and where do they live? Create a
↳                                 table.'}
🔵 TokenBudget                    Authoritative budget=2048 vs caller estimate=1024; using authoritative value
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] num_ctx=32768 prompt≈1397 → num_predict=2048
🔵 Call LLM                       Stage: Keyword/action splitting
🔵 Call LLM                       Model: openai/qwen2_5_7b_instruct prompt template: _PROMPT_KEYWORD_ACTION_SPLITTER purpose: Local retrieval decomposition
🔵 Call LLM                       options: {'temperature': 0.05, 'top_k': 10, 'top_p': 0.9, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_7b_instruct took 00:11
🔵 KeywordActionSplitter          LLM call + validation: 11.556 s
🟢 KeywordActionSplitter          Validated decomposition:
↳                                 Retrieval queries (2):
↳                                 [query_id=0] What do hedgehogs eat?
↳                                 [query_id=1] Where do hedgehogs live?
↳                                 Action: Create a table.
↳                                 Answer instruction: Using only the retrieved context, describe what hedgehogs eat and where they live in a table.
↳                                 Entities: hedgehogs
↳                                 Keywords: hedgehog, diet, habitat, food, nutrition, distribution
🔵 Retrieval Orchestration        Query rewrite stage                Activated                              knob=use_query_rewrite            value=on
🔵 Retrieval Orchestration        Pronoun substitution stage         Activated                              knob=use_pronoun_substitution     value=on
🔵 Retrieval Orchestration        Secondary query stage              Not activated                          knob=use_secondary_query          value=off
🔵 Retrieval Orchestration        Original-language vector stage     Not activated                          knob=use_original_language_vector value=off
🔵 Retrieval Orchestration        Indexed query shaping stage        Activated                              knob=shape_indexed_queries        value=on
🔵 Retrieval Orchestration        Indexed query translation stage    Activated                              knob=translate_indexed_queries    value=on
🔵 Retrieval Orchestration        Indexed WordNet expansion stage    Not activated                          knob=expand_indexed_queries_with_wordnet value=off
🔵 Retrieval Orchestration        Vector alternates stage            Activated                              knob=use_vector_alternates        value=on
🔵 Retrieval Orchestration        Grounding stage                    Activated                              knob=run_grounding                value=on
🔵 Retrieval Orchestration        Rerank stage                       Activated                              knob=run_rerank                   value=on
🔵 Retrieval Orchestration        Low-score fallback action          Activated                              knob=run_low_score_fallback       value=on
🔵 Retrieval Orchestration        Low-recall rescue action           Activated                              knob=run_low_recall_rescue        value=on
🔵 Retrieval Orchestration        Confidence evaluation              Activated                              knob=confidence_evaluation        value=both
🔵 Retrieval Orchestration        Local stage                        Activated                              knob=web_search                   value=local only
🔵 Retrieval Orchestration        Web stage                          Not activated                          knob=web_search                   value=local only
🔵 Retrieval Orchestration        Vector retriever stage             Activated                              knob=run_vector                   value=on
🔵 Retrieval Orchestration        BM25 retriever stage               Activated                              knob=run_bm25                     value=on
🔵 Retrieval Orchestration        Graph retriever stage              Activated                              knob=run_graph                    value=on
🔵 Retrieval Orchestration        Regex retriever stage              Activated                              knob=run_regex                    value=on
🔵 Retrieval Orchestration        original-language vector leg: off lang=en query='what do hedgehogs eat and where do they live? Create a table.' reason=disabled by orchestration flow
🔵 Retrieval Orchestration        language: en      stage plan flow=NATIVE_ALL mode=ALL guardrail=off original_leg=off rewrite=on pronouns=on rerank=on low_score_fallback=on low_recall_rescue=on
↳                                 grounding=on indexed_shape=on indexed_translate=on indexed_wordnet=off alt_queries=0 retrievers=v=on,b=on,g=on,r=on
🔵 Retrieval Orchestration        effective stage gates: flow=NATIVE_ALL mode=ALL web_mode=local_only local_stage=on web_stage=off
🔵 Retrieval Orchestration        language: en      retrieve local candidates mode=ALL
   ⚪ KeywordActionSplitter          Local query_id=0: 'What do hedgehogs eat?'
🔵 Retrieval Orchestration        discovered corpus language buckets: [de, en] active config buckets: [de, en, es, fr, it]
   ⚪ QueryPlan                      Step 4: document languages=['de', 'en']; active/present=['de', 'en']; metadata limit=whole collection
🔵 Retrieval Orchestration        query language flow: user=en current=en rewrite=en retrieval=en
🔵 Retrieval Orchestration        retriever: Vector     query: 'What do hedgehogs eat?' dispatch query
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test_embedder_bge_m3 for language en
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance                Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6676     0.3324    0.3324  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              2        0.6659     0.3341    0.3341  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              3        0.6041     0.3959    0.3959  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              4        0.5945     0.4055    0.4055  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              5        0.5135     0.4865    0.4865  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              6        0.5128     0.4872    0.4872  Vector                     Lions.pptx
   ⚪ Chroma                              7        0.4957     0.5043    0.5043  Vector                     Kamele.txt
   ⚪ Chroma                              8        0.4720     0.5280    0.5280  Vector                     Apes.docx
   ⚪ Chroma                              9        0.4575     0.5425    0.5425  Vector                     Cats.md
   ⚪ Chroma                             10        0.4531     0.5469    0.5469  Vector                     Elephants.jpg
   ⚪ Chroma                             11        0.4411     0.5589    0.5589  Vector                     Fish.txt
   ⚪ Chroma                             12        0.4370     0.5630    0.5630  Vector                     Cats.md
   ⚪ Chroma                             13        0.4339     0.5661    0.5661  Vector                     Cats.md
   ⚪ Chroma                             14        0.4256     0.5744    0.5744  Vector                     Fish.txt
   ⚪ Chroma                             15        0.4239     0.5761    0.5761  Vector                     Cats.md
   ⚪ Chroma                             16        0.4213     0.5787    0.5787  Vector                     Kamele.txt
   ⚪ Chroma                             17        0.4164     0.5836    0.5836  Vector                     Fish.txt
   ⚪ Chroma                             18        0.4158     0.5842    0.5842  Vector                     Kamele.txt
   ⚪ Chroma                             19        0.4121     0.5879    0.5879  Vector                     Cats.md
   ⚪ Chroma                             20        0.4090     0.5910    0.5910  Vector                     Cats.md
   ⚪ Chroma                             21        0.4089     0.5911    0.5911  Vector                     Dogs.png
   ⚪ Chroma                             22        0.4068     0.5932    0.5932  Vector                     Lions.pptx
   ⚪ Chroma                             23        0.4036     0.5964    0.5964  Vector                     Apes.docx
   ⚪ Chroma                             24        0.4026     0.5974    0.5974  Vector                     Cats.md
   ⚪ Chroma                             25        0.4019     0.5981    0.5981  Vector                     Kamele.txt
   ⚪ Chroma                             26        0.3964     0.6036    0.6036  Vector                     Cats.md
   ⚪ Chroma                             27        0.3940     0.6060    0.6060  Vector                     Apes.docx
   ⚪ Chroma                             28        0.3938     0.6062    0.6062  Vector                     Apes.docx
   ⚪ Chroma                             29        0.3930     0.6070    0.6070  Vector                     LionsAndApes.xlsx
   ⚪ Chroma                             30        0.3911     0.6089    0.6089  Vector                     Kamele.txt
   ⚪ Chroma                             31        0.3894     0.6106    0.6106  Vector                     Apes.docx
   ⚪ Chroma                             32        0.3851     0.6149    0.6149  Vector                     Cats.md
   ⚪ Chroma                             33        0.3833     0.6167    0.6167  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             34        0.3702     0.6298    0.6298  Vector                     Cats.md
   ⚪ Chroma                             35        0.3679     0.6321    0.6321  Vector                     Dogs.png
   ⚪ Chroma                             36        0.3677     0.6323    0.6323  Vector                     Cats.md
   ⚪ Chroma                             37        0.3620     0.6380    0.6380  Vector                     Cats.md
   ⚪ Chroma                             38        0.3619     0.6381    0.6381  Vector                     Lions.pptx
   ⚪ Chroma                             39        0.3558     0.6442    0.6442  Vector                     Cats.md
   ⚪ Chroma                             40        0.3540     0.6460    0.6460  Vector                     Cats.md
   ⚪ Chroma                             41        0.3532     0.6468    0.6468  Vector                     Fish.txt
   ⚪ Chroma                             42        0.3528     0.6472    0.6472  Vector                     Dogs.png
   ⚪ Chroma                             43        0.3506     0.6494    0.6494  Vector                     Fish.txt
   ⚪ Chroma                             44        0.3503     0.6497    0.6497  Vector                     Cats.md
   ⚪ Chroma                             45        0.3500     0.6500    0.6500  Vector                     Cats.md
   ⚪ Chroma                             46        0.3485     0.6515    0.6515  Vector                     Apes.docx
   ⚪ Chroma                             47        0.3478     0.6522    0.6522  Vector                     Cats.md
   ⚪ Chroma                             48        0.3429     0.6571    0.6571  Vector                     Cats.md
   ⚪ Chroma                             49        0.3425     0.6575    0.6575  Vector                     Cats.md
   ⚪ Chroma                             50        0.3424     0.6576    0.6576  Vector                     Cats.md
   ⚪ Chroma                             51        0.3398     0.6602    0.6602  Vector                     Cats.md
   ⚪ Chroma                             52        0.3398     0.6602    0.6602  Vector                     Cats.md
   ⚪ Chroma                             53        0.3397     0.6603    0.6603  Vector                     Cats.md
   ⚪ Chroma                             54        0.3357     0.6643    0.6643  Vector                     Cats.md
   ⚪ Chroma                             55        0.3349     0.6651    0.6651  Vector                     Lions.pptx
   ⚪ Chroma                             56        0.3327     0.6673    0.6673  Vector                     Lions.pptx
   ⚪ Chroma                             57        0.3309     0.6691    0.6691  Vector                     Cats.md
   ⚪ Chroma                             58        0.3278     0.6722    0.6722  Vector                     Cats.md
   ⚪ Chroma                             59        0.3275     0.6725    0.6725  Vector                     Pferde.pdf
   ⚪ Chroma                             60        0.3275     0.6725    0.6725  Vector                     Cats.md
   ⚪ Chroma                             61        0.3215     0.6785    0.6785  Vector                     Cats.md
   ⚪ Chroma                             62        0.3207     0.6793    0.6793  Vector                     Cats.md
   ⚪ Chroma                             63        0.3183     0.6817    0.6817  Vector                     Cats.md
   ⚪ Chroma                             64        0.3115     0.6885    0.6885  Vector                     Pferde.pdf
   ⚪ Chroma                             65        0.2993     0.7007    0.7007  Vector                     Pferde.pdf
   ⚪ Chroma                             66        0.2977     0.7023    0.7023  Vector                     Kamele.txt
   ⚪ Chroma                             67        0.2931     0.7069    0.7069  Vector                     Cats.md
   ⚪ Chroma                             68        0.2905     0.7095    0.7095  Vector                     Cats.md
   ⚪ Chroma                             69        0.2865     0.7135    0.7135  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             70        0.2759     0.7241    0.7241  Vector                     Fish.txt
   ⚪ Chroma                             71        0.2668     0.7332    0.7332  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             72        0.2662     0.7338    0.7338  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             73        0.2659     0.7341    0.7341  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             74        0.2611     0.7389    0.7389  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             75        0.2354     0.7646    0.7646  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             76        0.2184     0.7816    0.7816  Vector                     Pferde.pdf
🟢 Chroma                         Querying Chroma DB query returned 76 chunks
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=en retrievers=BM25/Graph/Regex shaping=on changed=0
↳                                 primary_query source=REWRITTEN_QUERY lang=en->en
↳                                   original='What do hedgehogs eat?'
↳                                   staged='What do hedgehogs eat?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->en
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->en count=0
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=de retrievers=BM25/Graph/Regex shaping=on changed=1
↳                                 primary_query source=REWRITTEN_QUERY lang=en->de
↳                                   original='What do hedgehogs eat?'
↳                                   staged='Was essen Igel?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->de
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->de count=0
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: en      query: 'What do hedgehogs eat?'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: de      query: 'Was essen Igel?'
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language en
🟢 BM25                           Loaded persisted BM25 index (76 chunks, 1876 terms)
🟢 BM25                           BM25 retrieval returned 8 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        3.6145         BM25                Lions.pptx
   ⚪ BM25                                2        2.9876         BM25                Hedgehogs.pdf
   ⚪ BM25                                3        2.8919         BM25                Hedgehogs.pdf
   ⚪ BM25                                4        2.8816         BM25                Hedgehogs.pdf
   ⚪ BM25                                5        2.4446         BM25                Fish.txt
   ⚪ BM25                                6        2.1021         BM25                Fish.txt
   ⚪ BM25                                7        1.9255         BM25                Hedgehogs.pdf
   ⚪ BM25                                8        1.9164         BM25                Hedgehogs.pdf
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=en; query='What do hedgehogs eat?'; hits={'BM25': 8, 'Graph': 0, 'Regex': 0}
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language de
🟢 BM25                           BM25 retrieval returned 0 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=de; query='Was essen Igel?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
🔵 Graph                          Querying graph index on collection Test_embedder_bge_m3 for language en
🟢 Graph                          Loaded persisted graph index (76 chunks, 1115 entities)
🟢 Graph                          Graph retrieval returned 50 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1     2427.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               2     2334.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               3     1761.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               4     1544.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               5      744.0000              Graph          Fish.txt
   ⚪ Graph                               6      650.0000              Graph          Fish.txt
   ⚪ Graph                               7      544.0000              Graph          Fish.txt
   ⚪ Graph                               8      465.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               9      366.0000              Graph          Apes.docx
   ⚪ Graph                              10      310.0000              Graph          Fish.txt
   ⚪ Graph                              11      280.0000              Graph          Fish.txt
   ⚪ Graph                              12      272.0000              Graph          Cats.md
   ⚪ Graph                              13      260.0000              Graph          Apes.docx
   ⚪ Graph                              14      258.0000              Graph          Lions.pptx
   ⚪ Graph                              15      243.0000              Graph          Elephants.jpg
   ⚪ Graph                              16      228.0000              Graph          Cats.md
   ⚪ Graph                              17      226.0000              Graph          Apes.docx
   ⚪ Graph                              18      209.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                              19      201.0000              Graph          Cats.md
   ⚪ Graph                              20      198.0000              Graph          Cats.md
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=en; query='What do hedgehogs eat?'; hits={'BM25': 0, 'Graph': 50, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language en
🟢 Regex                          Loaded persisted regex index (76 chunks, 293 verbs, 925 nouns)
🟢 Regex                          Regex retrieval returned 0 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=en; query='What do hedgehogs eat?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
🔵 Graph                          Querying graph index on collection Test_embedder_bge_m3 for language de
🟢 Graph                          Graph retrieval returned 10 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1     1792.0000              Graph          Pferde.pdf
   ⚪ Graph                               2     1771.0000              Graph          Kamele.txt
   ⚪ Graph                               3      326.0000              Graph          Kamele.txt
   ⚪ Graph                               4      311.0000              Graph          Pferde.pdf
   ⚪ Graph                               5      261.0000              Graph          Pferde.pdf
   ⚪ Graph                               6      207.0000              Graph          Kamele.txt
   ⚪ Graph                               7      202.0000              Graph          Kamele.txt
   ⚪ Graph                               8      140.0000              Graph          Pferde.pdf
   ⚪ Graph                               9       89.0000              Graph          Kamele.txt
   ⚪ Graph                              10       76.0000              Graph          Kamele.txt
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=de; query='Was essen Igel?'; hits={'BM25': 0, 'Graph': 10, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language de
🟢 Regex                          Regex retrieval returned 0 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=de; query='Was essen Igel?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
   ⚪ KeywordActionSplitter          query_id=0 hits: {'vector': 76, 'bm25': 8, 'graph': 60, 'regex': 0}
   ⚪ KeywordActionSplitter          Local query_id=1: 'Where do hedgehogs live?'
   ⚪ QueryPlan                      Step 4: document languages=['de', 'en']; active/present=['de', 'en']; metadata limit=whole collection
🔵 Retrieval Orchestration        query language flow: user=en current=en rewrite=en retrieval=en
🔵 Retrieval Orchestration        retriever: Vector     query: 'Where do hedgehogs live?' dispatch query
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test_embedder_bge_m3 for language en
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance                Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6719     0.3281    0.3281  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              2        0.6263     0.3737    0.3737  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              3        0.6193     0.3807    0.3807  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              4        0.5725     0.4275    0.4275  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              5        0.5474     0.4526    0.4526  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              6        0.4994     0.5006    0.5006  Vector                     Apes.docx
   ⚪ Chroma                              7        0.4972     0.5028    0.5028  Vector                     Apes.docx
   ⚪ Chroma                              8        0.4891     0.5109    0.5109  Vector                     Lions.pptx
   ⚪ Chroma                              9        0.4665     0.5335    0.5335  Vector                     Apes.docx
   ⚪ Chroma                             10        0.4499     0.5501    0.5501  Vector                     Apes.docx
   ⚪ Chroma                             11        0.4335     0.5665    0.5665  Vector                     Elephants.jpg
   ⚪ Chroma                             12        0.4323     0.5677    0.5677  Vector                     Fish.txt
   ⚪ Chroma                             13        0.4292     0.5708    0.5708  Vector                     Kamele.txt
   ⚪ Chroma                             14        0.4237     0.5763    0.5763  Vector                     Fish.txt
   ⚪ Chroma                             15        0.4223     0.5777    0.5777  Vector                     Apes.docx
   ⚪ Chroma                             16        0.4151     0.5849    0.5849  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             17        0.4041     0.5959    0.5959  Vector                     Lions.pptx
   ⚪ Chroma                             18        0.4038     0.5962    0.5962  Vector                     Lions.pptx
   ⚪ Chroma                             19        0.4037     0.5963    0.5963  Vector                     Fish.txt
   ⚪ Chroma                             20        0.4007     0.5993    0.5993  Vector                     Cats.md
   ⚪ Chroma                             21        0.3982     0.6018    0.6018  Vector                     Cats.md
   ⚪ Chroma                             22        0.3980     0.6020    0.6020  Vector                     Cats.md
   ⚪ Chroma                             23        0.3963     0.6037    0.6037  Vector                     LionsAndApes.xlsx
   ⚪ Chroma                             24        0.3927     0.6073    0.6073  Vector                     Cats.md
   ⚪ Chroma                             25        0.3923     0.6077    0.6077  Vector                     Cats.md
   ⚪ Chroma                             26        0.3899     0.6101    0.6101  Vector                     Dogs.png
   ⚪ Chroma                             27        0.3797     0.6203    0.6203  Vector                     Cats.md
   ⚪ Chroma                             28        0.3785     0.6215    0.6215  Vector                     Pferde.pdf
   ⚪ Chroma                             29        0.3739     0.6261    0.6261  Vector                     Cats.md
   ⚪ Chroma                             30        0.3738     0.6262    0.6262  Vector                     Dogs.png
   ⚪ Chroma                             31        0.3692     0.6308    0.6308  Vector                     Cats.md
   ⚪ Chroma                             32        0.3659     0.6341    0.6341  Vector                     Cats.md
   ⚪ Chroma                             33        0.3656     0.6344    0.6344  Vector                     Apes.docx
   ⚪ Chroma                             34        0.3647     0.6353    0.6353  Vector                     Cats.md
   ⚪ Chroma                             35        0.3566     0.6434    0.6434  Vector                     Cats.md
   ⚪ Chroma                             36        0.3532     0.6468    0.6468  Vector                     Cats.md
   ⚪ Chroma                             37        0.3497     0.6503    0.6503  Vector                     Fish.txt
   ⚪ Chroma                             38        0.3489     0.6511    0.6511  Vector                     Cats.md
   ⚪ Chroma                             39        0.3455     0.6545    0.6545  Vector                     Fish.txt
   ⚪ Chroma                             40        0.3448     0.6552    0.6552  Vector                     Kamele.txt
   ⚪ Chroma                             41        0.3395     0.6605    0.6605  Vector                     Lions.pptx
   ⚪ Chroma                             42        0.3387     0.6613    0.6613  Vector                     Cats.md
   ⚪ Chroma                             43        0.3373     0.6627    0.6627  Vector                     Kamele.txt
   ⚪ Chroma                             44        0.3354     0.6646    0.6646  Vector                     Dogs.png
   ⚪ Chroma                             45        0.3319     0.6681    0.6681  Vector                     Cats.md
   ⚪ Chroma                             46        0.3314     0.6686    0.6686  Vector                     Cats.md
   ⚪ Chroma                             47        0.3313     0.6687    0.6687  Vector                     Kamele.txt
   ⚪ Chroma                             48        0.3310     0.6690    0.6690  Vector                     Cats.md
   ⚪ Chroma                             49        0.3286     0.6714    0.6714  Vector                     Cats.md
   ⚪ Chroma                             50        0.3279     0.6721    0.6721  Vector                     Cats.md
   ⚪ Chroma                             51        0.3277     0.6723    0.6723  Vector                     Lions.pptx
   ⚪ Chroma                             52        0.3245     0.6755    0.6755  Vector                     Cats.md
   ⚪ Chroma                             53        0.3234     0.6766    0.6766  Vector                     Cats.md
   ⚪ Chroma                             54        0.3212     0.6788    0.6788  Vector                     Cats.md
   ⚪ Chroma                             55        0.3199     0.6801    0.6801  Vector                     Kamele.txt
   ⚪ Chroma                             56        0.3180     0.6820    0.6820  Vector                     Cats.md
   ⚪ Chroma                             57        0.3180     0.6820    0.6820  Vector                     Kamele.txt
   ⚪ Chroma                             58        0.3176     0.6824    0.6824  Vector                     Cats.md
   ⚪ Chroma                             59        0.3158     0.6842    0.6842  Vector                     Pferde.pdf
   ⚪ Chroma                             60        0.3153     0.6847    0.6847  Vector                     Cats.md
   ⚪ Chroma                             61        0.3139     0.6861    0.6861  Vector                     Cats.md
   ⚪ Chroma                             62        0.3112     0.6888    0.6888  Vector                     Cats.md
   ⚪ Chroma                             63        0.3013     0.6987    0.6987  Vector                     Cats.md
   ⚪ Chroma                             64        0.2993     0.7007    0.7007  Vector                     Cats.md
   ⚪ Chroma                             65        0.2990     0.7010    0.7010  Vector                     Cats.md
   ⚪ Chroma                             66        0.2981     0.7019    0.7019  Vector                     Cats.md
   ⚪ Chroma                             67        0.2793     0.7207    0.7207  Vector                     Cats.md
   ⚪ Chroma                             68        0.2788     0.7212    0.7212  Vector                     Pferde.pdf
   ⚪ Chroma                             69        0.2753     0.7247    0.7247  Vector                     Fish.txt
   ⚪ Chroma                             70        0.2555     0.7445    0.7445  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             71        0.2507     0.7493    0.7493  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             72        0.2352     0.7648    0.7648  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             73        0.2164     0.7836    0.7836  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             74        0.2124     0.7876    0.7876  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             75        0.2044     0.7956    0.7956  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             76        0.1825     0.8175    0.8175  Vector                     Pferde.pdf
🟢 Chroma                         Querying Chroma DB query returned 76 chunks
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=en retrievers=BM25/Graph/Regex shaping=on changed=0
↳                                 primary_query source=REWRITTEN_QUERY lang=en->en
↳                                   original='Where do hedgehogs live?'
↳                                   staged='Where do hedgehogs live?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->en
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->en count=0
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=de retrievers=BM25/Graph/Regex shaping=on changed=1
↳                                 primary_query source=REWRITTEN_QUERY lang=en->de
↳                                   original='Where do hedgehogs live?'
↳                                   staged='Wo leben Igel?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->de
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->de count=0
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: en      query: 'Where do hedgehogs live?'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: de      query: 'Wo leben Igel?'
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language en
🟢 BM25                           BM25 retrieval returned 18 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        5.6325         BM25                Fish.txt
   ⚪ BM25                                2        3.6145         BM25                Lions.pptx
   ⚪ BM25                                3        2.9876         BM25                Hedgehogs.pdf
   ⚪ BM25                                4        2.8919         BM25                Hedgehogs.pdf
   ⚪ BM25                                5        2.8816         BM25                Hedgehogs.pdf
   ⚪ BM25                                6        2.5221         BM25                Cats.md
   ⚪ BM25                                7        2.3007         BM25                Elephants.jpg
   ⚪ BM25                                8        2.2576         BM25                Cats.md
   ⚪ BM25                                9        2.0735         BM25                Apes.docx
   ⚪ BM25                               10        2.0284         BM25                Cats.md
   ⚪ BM25                               11        2.0138         BM25                Lions.pptx
   ⚪ BM25                               12        1.9255         BM25                Hedgehogs.pdf
   ⚪ BM25                               13        1.9164         BM25                Hedgehogs.pdf
   ⚪ BM25                               14        1.8415         BM25                Apes.docx
   ⚪ BM25                               15        1.7604         BM25                Fish.txt
   ⚪ BM25                               16        1.7386         BM25                Apes.docx
   ⚪ BM25                               17        1.5994         BM25                Cats.md
   ⚪ BM25                               18        1.2780         BM25                Cats.md
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=en; query='Where do hedgehogs live?'; hits={'BM25': 18, 'Graph': 0, 'Regex': 0}
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language de
🟢 BM25                           BM25 retrieval returned 0 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=de; query='Wo leben Igel?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
🔵 Graph                          Querying graph index on collection Test_embedder_bge_m3 for language en
🟢 Graph                          Graph retrieval returned 50 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1     2427.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               2     2334.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               3     1761.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               4     1544.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               5      744.0000              Graph          Fish.txt
   ⚪ Graph                               6      650.0000              Graph          Fish.txt
   ⚪ Graph                               7      544.0000              Graph          Fish.txt
   ⚪ Graph                               8      465.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                               9      366.0000              Graph          Apes.docx
   ⚪ Graph                              10      310.0000              Graph          Fish.txt
   ⚪ Graph                              11      280.0000              Graph          Fish.txt
   ⚪ Graph                              12      272.0000              Graph          Cats.md
   ⚪ Graph                              13      260.0000              Graph          Apes.docx
   ⚪ Graph                              14      258.0000              Graph          Lions.pptx
   ⚪ Graph                              15      243.0000              Graph          Elephants.jpg
   ⚪ Graph                              16      228.0000              Graph          Cats.md
   ⚪ Graph                              17      226.0000              Graph          Apes.docx
   ⚪ Graph                              18      209.0000              Graph          Hedgehogs.pdf
   ⚪ Graph                              19      201.0000              Graph          Cats.md
   ⚪ Graph                              20      198.0000              Graph          Cats.md
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=en; query='Where do hedgehogs live?'; hits={'BM25': 0, 'Graph': 50, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language en
🟢 Regex                          Regex retrieval returned 5 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ Regex                               1        1.0000         0         1                    Regex    Hedgehogs.pdf
   ⚪ Regex                               2        1.0000         0         1                    Regex    Hedgehogs.pdf
   ⚪ Regex                               3        1.0000         0         1                    Regex    Hedgehogs.pdf
   ⚪ Regex                               4        1.0000         0         1                    Regex    Hedgehogs.pdf
   ⚪ Regex                               5        1.0000         0         1                    Regex    Hedgehogs.pdf
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=en; query='Where do hedgehogs live?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 5}
🔵 Graph                          Querying graph index on collection Test_embedder_bge_m3 for language de
🟢 Graph                          Graph retrieval returned 0 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=de; query='Wo leben Igel?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language de
🟢 Regex                          Regex retrieval returned 0 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=de; query='Wo leben Igel?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
   ⚪ KeywordActionSplitter          query_id=1 hits: {'vector': 76, 'bm25': 18, 'graph': 50, 'regex': 5}
   ⚪ KeywordActionSplitter          BM25 keyword query: 'hedgehog diet habitat food nutrition distribution'
   ⚪ QueryPlan                      Step 4: document languages=['de', 'en']; active/present=['de', 'en']; metadata limit=whole collection
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=en retrievers=BM25/Graph/Regex shaping=on changed=0
↳                                 primary_query source=REWRITTEN_QUERY lang=en->en
↳                                   original='hedgehog diet habitat food nutrition distribution'
↳                                   staged='hedgehog diet habitat food nutrition distribution'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->en
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->en count=0
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=de retrievers=BM25/Graph/Regex shaping=on changed=1
↳                                 primary_query source=REWRITTEN_QUERY lang=en->de
↳                                   original='hedgehog diet habitat food nutrition distribution'
↳                                   staged='Hedgehog-Diät Lebensraum Nahrung Ernährung Verteilung'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->de
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->de count=0
🔵 Retrieval Orchestration        retrievers='BM25' language: en      query: 'hedgehog diet habitat food nutrition distribution'
🔵 Retrieval Orchestration        retrievers='BM25' language: de      query: 'Hedgehog-Diät Lebensraum Nahrung Ernährung Verteilung'
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language en
🟢 BM25                           BM25 retrieval returned 15 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        8.2136         BM25                Hedgehogs.pdf
   ⚪ BM25                                2        6.6074         BM25                Dogs.png
   ⚪ BM25                                3        5.2595         BM25                Fish.txt
   ⚪ BM25                                4        4.9437         BM25                Hedgehogs.pdf
   ⚪ BM25                                5        4.8173         BM25                Hedgehogs.pdf
   ⚪ BM25                                6        4.8173         BM25                Hedgehogs.pdf
   ⚪ BM25                                7        3.8023         BM25                Cats.md
   ⚪ BM25                                8        3.7800         BM25                Apes.docx
   ⚪ BM25                                9        3.5549         BM25                Cats.md
   ⚪ BM25                               10        3.4159         BM25                Cats.md
   ⚪ BM25                               11        2.9946         BM25                Lions.pptx
   ⚪ BM25                               12        2.8414         BM25                Lions.pptx
   ⚪ BM25                               13        2.3217         BM25                Cats.md
   ⚪ BM25                               14        1.9164         BM25                Hedgehogs.pdf
   ⚪ BM25                               15        1.6381         BM25                Fish.txt
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=en; query='hedgehog diet habitat food nutrition distribution'; hits={'BM25': 15, 'Graph': 0, 'Regex': 0}
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language de
🟢 BM25                           BM25 retrieval returned 1 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        3.1253         BM25                Kamele.txt
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=de; query='Hedgehog-Diät Lebensraum Nahrung Ernährung Verteilung'; hits={'BM25': 1, 'Graph': 0, 'Regex': 0}
🔵 Retrieval Orchestration        language: en      local docs fetched vector=76 bm25=26 graph=60 regex=5
🔵 Retrieval Orchestration        language: en      web stage skipped (web_mode=local_only)
🔵 Retrieval Orchestration        language: en      merge candidates and build context
🔵 Merge                          Language-scoped RRF candidate pool languages=2 per_language={'en': 66, 'de': 10} interleaved=76/100
   ⚪ Merge                             Pos    RRFScore                            Retrievers  File
   ⚪ Merge                          ----------------------------------------------------------------------------------
   ⚪ Merge                               1      0.0643              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               2      0.0487              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                               3      0.0641              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               4      0.0313              Vector      Graph         Kamele.txt
   ⚪ Merge                               5      0.0616              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               6      0.0311              Vector      Graph         Pferde.pdf
   ⚪ Merge                               7      0.0613              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                               8      0.0310              Vector      Graph         Kamele.txt
   ⚪ Merge                               9      0.0609              Vector BM25 Graph Regex   Hedgehogs.pdf
   ⚪ Merge                              10      0.0310              Vector      Graph         Pferde.pdf
   ⚪ Merge                              11      0.0451              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              12      0.0306              Vector      Graph         Kamele.txt
   ⚪ Merge                              13      0.0448              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              14      0.0306              Vector      Graph         Kamele.txt
   ⚪ Merge                              15      0.0433              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              16      0.0299              Vector      Graph         Pferde.pdf
   ⚪ Merge                              17      0.0425              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              18      0.0292              Vector      Graph         Kamele.txt
   ⚪ Merge                              19      0.0413              Vector BM25 Graph         Elephants.jpg
   ⚪ Merge                              20      0.0290              Vector      Graph         Pferde.pdf
   ⚪ Merge                              21      0.0409              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              22      0.0400              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              23      0.0396              Vector BM25 Graph         Dogs.png
   ⚪ Merge                              24      0.0379              Vector BM25 Graph         Cats.md
   ⚪ Merge                              25      0.0377              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              26      0.0375              Vector BM25 Graph         Cats.md
   ⚪ Merge                              27      0.0366              Vector BM25 Graph         Cats.md
   ⚪ Merge                              28      0.0365              Vector BM25 Graph         Cats.md
   ⚪ Merge                              29      0.0365              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              30      0.0362              Vector BM25 Graph         Cats.md
   ⚪ Merge                              31      0.0357              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              32      0.0352              Vector BM25 Graph         Cats.md
   ⚪ Merge                              33      0.0342              Vector BM25 Graph         Cats.md
   ⚪ Merge                              34      0.0284              Vector      Graph         Apes.docx
   ⚪ Merge                              35      0.0268              Vector BM25               Lions.pptx
   ⚪ Merge                              36      0.0252              Vector      Graph         Hedgehogs.pdf
   ⚪ Merge                              37      0.0239              Vector BM25               Cats.md
   ⚪ Merge                              38      0.0237              Vector      Graph         Fish.txt
   ⚪ Merge                              39      0.0236              Vector      Graph         Fish.txt
   ⚪ Merge                              40      0.0230              Vector      Graph         Cats.md
   ⚪ Merge                              41      0.0228              Vector      Graph         Cats.md
   ⚪ Merge                              42      0.0222              Vector      Graph         Cats.md
   ⚪ Merge                              43      0.0221              Vector      Graph         Cats.md
   ⚪ Merge                              44      0.0219              Vector      Graph         Cats.md
   ⚪ Merge                              45      0.0218              Vector      Graph         Cats.md
   ⚪ Merge                              46      0.0218              Vector      Graph         Dogs.png
   ⚪ Merge                              47      0.0212              Vector      Graph         Lions.pptx
   ⚪ Merge                              48      0.0210              Vector      Graph         Cats.md
   ⚪ Merge                              49      0.0207              Vector      Graph         Cats.md
   ⚪ Merge                              50      0.0206              Vector      Graph         Cats.md
   ⚪ Merge                              51      0.0205              Vector      Graph         Cats.md
   ⚪ Merge                              52      0.0202              Vector      Graph         Cats.md
   ⚪ Merge                              53      0.0202              Vector      Graph         Cats.md
   ⚪ Merge                              54      0.0200              Vector      Graph         Cats.md
   ⚪ Merge                              55      0.0196              Vector      Graph         Cats.md
   ⚪ Merge                              56      0.0195              Vector      Graph         Cats.md
   ⚪ Merge                              57      0.0192              Vector      Graph         Cats.md
   ⚪ Merge                              58      0.0191              Vector      Graph         Lions.pptx
   ⚪ Merge                              59      0.0187              Vector      Graph         Cats.md
   ⚪ Merge                              60      0.0185              Vector      Graph         Cats.md
   ⚪ Merge                              61      0.0185              Vector      Graph         Fish.txt
   ⚪ Merge                              62      0.0184              Vector      Graph         Cats.md
   ⚪ Merge                              63      0.0114              Vector                    LionsAndApes.xlsx
   ⚪ Merge                              64      0.0108              Vector                    Dogs.png
   ⚪ Merge                              65      0.0102              Vector                    Apes.docx
   ⚪ Merge                              66      0.0101              Vector                    Cats.md
   ⚪ Merge                              67      0.0100              Vector                    Cats.md
   ⚪ Merge                              68      0.0099              Vector                    Cats.md
   ⚪ Merge                              69      0.0092              Vector                    Cats.md
   ⚪ Merge                              70      0.0084              Vector                    Cats.md
   ⚪ Merge                              71      0.0083              Vector                    BlazingFast_Workstation.md
   ⚪ Merge                              72      0.0082              Vector                    BlazingFast_Workstation.md
   ⚪ Merge                              73      0.0081              Vector                    BlazingFast_Workstation.md
   ⚪ Merge                              74      0.0081              Vector                    BlazingFast_Workstation.md
   ⚪ Merge                              75      0.0080              Vector                    BlazingFast_Workstation.md
   ⚪ Merge                              76      0.0079              Vector                    BlazingFast_Workstation.md
🔵 ChunkDedup                     Removed 1 near-duplicate chunk(s) (threshold=0.85, include_web=on, kept 75)
🔵 Rerank                         Pre-rerank pool local=75 web=0 total=75 original_leg=off lang=en added[V/B/G/R]=0/0/0/0
🔵 Rerank                         Rerank query: 'what do hedgehogs eat and where do they live? Create a table.'
🔵 Rerank                         Passing language=en chunks=66 to reranker; query='What do hedgehogs eat?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 66 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=en top5=[1:Hedgehogs.pdf (raw=0.9866), 2:Hedgehogs.pdf (raw=0.9539), 3:Hedgehogs.pdf (raw=0.5214), 4:Hedgehogs.pdf (raw=0.3531), 5:Hedgehogs.pdf
↳                                 (raw=0.0407)]
🔵 Rerank                         Passing language=de chunks=9 to reranker; query='Was essen Igel?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 9 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=de top5=[1:Kamele.txt (raw=0.0043), 2:Kamele.txt (raw=0.0014), 3:Kamele.txt (raw=0.0005), 4:Kamele.txt (raw=0.0002), 5:Pferde.pdf (raw=0.0000)]
🔵 Rerank summary                 Language           Chunks    Top raw  Mean top3   Mean raw
🔵 Rerank summary                 en                     66     0.9866     0.8206     0.0438
🔵 Rerank summary                 de                      9     0.0043     0.0021     0.0007
🔵 Rerank                         Passing language=en chunks=66 to reranker; query='Where do hedgehogs live?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 66 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=en top5=[1:Hedgehogs.pdf (raw=0.9287), 2:Hedgehogs.pdf (raw=0.9226), 3:Hedgehogs.pdf (raw=0.3475), 4:Hedgehogs.pdf (raw=0.1105), 5:Hedgehogs.pdf
↳                                 (raw=0.0579)]
🔵 Rerank                         Passing language=de chunks=9 to reranker; query='Wo leben Igel?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 9 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=de top5=[1:Pferde.pdf (raw=0.0005), 2:Kamele.txt (raw=0.0004), 3:Kamele.txt (raw=0.0002), 4:Kamele.txt (raw=0.0002), 5:Kamele.txt (raw=0.0000)]
🔵 Rerank summary                 Language           Chunks    Top raw  Mean top3   Mean raw
🔵 Rerank summary                 en                     66     0.9287     0.7329     0.0364
🔵 Rerank summary                 de                      9     0.0005     0.0004     0.0002
🔵 Rerank                         Selected language=en by highest top raw reranker score=0.9866; top chunk='Hedgehogs.pdf'; passing 66 matching-language local candidates + 0 web candidates to chunk
↳                                 selection.
   ⚪ Rerank                            Pos    RawScore    AdjScore                Retrievers  File                                      Text
   ⚪ Rerank                         --------------------------------------------------------------------------------------------------------------
   ⚪ Rerank                              1      0.9866      1.0000  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 suburban gardens and agricult
   ⚪ Rerank                              2      0.9539      0.9669  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs a
   ⚪ Rerank                              3      0.5214      0.5285  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations
   ⚪ Rerank                              4      0.3531      0.3579  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, deci
   ⚪ Rerank                              5      0.1105      0.1120  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—suc
   ⚪ Rerank                              6      0.0112      0.0113  Vector      Graph         Apes.docx                                 Orangutans are the only great apes found
   ⚪ Rerank                              7      0.0093      0.0095  Vector BM25 Graph         Elephants.jpg                             elephants large, long-lived mammals fami
   ⚪ Rerank                              8      0.0075      0.0076  Vector BM25 Graph         Apes.docx                                 Gorillas are the largest living primates
   ⚪ Rerank                              9      0.0067      0.0068  Vector BM25 Graph         Apes.docx                                 Chimpanzees (Pan troglodytes) are found
   ⚪ Rerank                             10      0.0051      0.0052  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
   ⚪ Rerank                             11      0.0029      0.0029  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
   ⚪ Rerank                             12      0.0021      0.0021  Vector BM25 Graph         Fish.txt                                   what fish are fish are a diverse group
   ⚪ Rerank                             13      0.0017      0.0017  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
   ⚪ Rerank                             14      0.0014      0.0014  Vector BM25 Graph         Fish.txt                                  diversity: fish include jawless fishes (
   ⚪ Rerank                             15      0.0013      0.0013  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
   ⚪ Rerank                             16      0.0012      0.0012  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
   ⚪ Rerank                             17      0.0010      0.0010  Vector BM25 Graph         Cats.md                                   - free feeding can lead to obesity. - pu
   ⚪ Rerank                             18      0.0008      0.0008  Vector BM25 Graph         Cats.md                                   cats have lived alongside humans for tho
   ⚪ Rerank                             19      0.0008      0.0008  Vector BM25 Graph         Apes.docx                                 Great apes (Hominidae) are the closest l
   ⚪ Rerank                             20      0.0007      0.0007  Vector BM25 Graph         Dogs.png                                  tasks. understanding breed-specific ende
   ⚪ Rerank                             21      0.0005      0.0005  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
   ⚪ Rerank                             22      0.0005      0.0005  Vector      Graph         Cats.md                                   cats suffered during periods of supersti
   ⚪ Rerank                             23      0.0004      0.0004  Vector      Graph         Fish.txt                                  terminology explained cold-blooded: a co
   ⚪ Rerank                             24      0.0004      0.0004  Vector                    Dogs.png                                  , comprehensive text dogs dogs accompani
   ⚪ Rerank                             25      0.0004      0.0003  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
   ⚪ Rerank                             26      0.0003      0.0003  Vector      Graph         Cats.md                                   - kingdom: animalia - phylum: chordata -
   ⚪ Rerank                             27      0.0003      0.0003  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
   ⚪ Rerank                             28      0.0003      0.0003  Vector      Graph         Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernati
   ⚪ Rerank                             29      0.0002      0.0002  Vector      Graph         Fish.txt                                  ectotherm: the preferred scientific term
   ⚪ Rerank                             30      0.0002      0.0002  Vector      Graph         Cats.md                                   these breeds emerged without heavy human
   ⚪ Rerank                             31      0.0002      0.0002  Vector      Graph         Cats.md                                   most adult cats are lactose intolerant.
   ⚪ Rerank                             32      0.0002      0.0002  Vector BM25 Graph         Cats.md                                   even well-fed cats hunt. the sequence: 1
   ⚪ Rerank                             33      0.0002      0.0001  Vector      Graph         Cats.md                                   cats were revered, often associated with
   ⚪ Rerank                             34      0.0002      0.0001  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
   ⚪ Rerank                             35      0.0001      0.0001  Vector                    Apes.docx                                 All great apes demonstrate remarkable co
   ⚪ Rerank                             36      0.0001      0.0001  Vector                    Cats.md                                   - climbing structures - scratching posts
   ⚪ Rerank                             37      0.0001      0.0001  Vector BM25 Graph         Cats.md                                   indoor cats often live 12–18 years; some
   ⚪ Rerank                             38      0.0001      0.0001  Vector                    Cats.md                                   a deep, structured exploration of domest
   ⚪ Rerank                             39      0.0001      0.0001  Vector BM25               Cats.md                                   outdoor access: - pros: exercise, stimul
   ⚪ Rerank                             40      0.0001      0.0001  Vector BM25 Graph         Cats.md                                   cats are often described as solitary, bu
   ⚪ Rerank                             41      0.0001      0.0001  Vector      Graph         Cats.md                                   they have a righting reflex, but falls f
   ⚪ Rerank                             42      0.0001      0.0000  Vector      Graph         Cats.md                                   free-roaming cats are estimated to kill
   ⚪ Rerank                             43      0.0001      0.0000  Vector      Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
   ⚪ Rerank                             44      0.0001      0.0000  Vector                    Cats.md                                   - puzzle toys - training sessions - nove
   ⚪ Rerank                             45      0.0001      0.0000  Vector                    Cats.md                                   spaying/neutering is essential to reduce
   ⚪ Rerank                             46      0.0000      0.0000  Vector      Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
   ⚪ Rerank                             47      0.0000      0.0000  Vector      Graph         Lions.pptx                                Slide 2: Physical Characteristics Male l
   ⚪ Rerank                             48      0.0000      0.0000  Vector      Graph         Cats.md                                   cats dominate digital culture: - memes (
   ⚪ Rerank                             49      0.0000      0.0000  Vector      Graph         Cats.md                                   cats are crepuscular hunters with: - a h
   ⚪ Rerank                             50      0.0000      0.0000  Vector      Graph         Cats.md                                   cats are complex, adaptable, and endless
🔵 Rerank                         Reranking with bge-reranker returned 66 chunks
🔵 Chunk selection                Strategy 'DEFAULT' → ScoreRankedSelector
🔵 Score select                   logit+ = effective logit after single-chunk boost (raw in raw_rerank_score)
🟣 Score select                         Logit [Sigmoid]       Thr      ΔProb  Retrievers                File                                      Text
🟣 Score select                   --------------------------------------------------------------------------------------------------------------------
🟣 Score select                    ✅    0.9866 [0.728]  (0.5000)    +0.2284  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 suburban gardens and agricult
🟣 Score select                    ✅    0.9539 [0.722]  (0.5000)    +0.2219  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs a
🟣 Score select                    ✅    0.5214 [0.627]  (0.5000)    +0.1275  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations
🟣 Score select                    ✅    0.3531 [0.587]  (0.5000)    +0.0874  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, deci
🟣 Score select                   ⬆✅    0.2325+[0.558]  (0.5000)    +0.0579  Vector BM25 Graph         Elephants.jpg                             elephants large, long-lived mammals fami
🟣 Score select                   ⬆✅    0.2243+[0.556]  (0.5000)    +0.0559  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
🟣 Score select                    ✅    0.1105 [0.528]  (0.5000)    +0.0276  Vector BM25 Graph Regex   Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—suc
🟣 Score select                    ✅    0.0112 [0.503]  (0.5000)    +0.0028  Vector      Graph         Apes.docx                                 Orangutans are the only great apes found
🟣 Score select                    ✅    0.0075 [0.502]  (0.5000)    +0.0019  Vector BM25 Graph         Apes.docx                                 Gorillas are the largest living primates
🟣 Score select                    ✅    0.0067 [0.502]  (0.5000)    +0.0017  Vector BM25 Graph         Apes.docx                                 Chimpanzees (Pan troglodytes) are found
🟣 Score select                    ✅    0.0051 [0.501]  (0.5000)    +0.0013  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
🟣 Score select                    ✅    0.0029 [0.501]  (0.5000)    +0.0007  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
🟣 Score select                    ✅    0.0021 [0.501]  (0.5000)    +0.0005  Vector BM25 Graph         Fish.txt                                   what fish are fish are a diverse group
🟣 Score select                    ✅    0.0017 [0.500]  (0.5000)    +0.0004  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
🟣 Score select                    ✅    0.0014 [0.500]  (0.5000)    +0.0003  Vector BM25 Graph         Fish.txt                                  diversity: fish include jawless fishes (
🟣 Score select                    ✅    0.0013 [0.500]  (0.5000)    +0.0003  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
🟣 Score select                    ✅    0.0010 [0.500]  (0.5000)    +0.0003  Vector BM25 Graph         Cats.md                                   - free feeding can lead to obesity. - pu
🟣 Score select                    ✅    0.0008 [0.500]  (0.5000)    +0.0002  Vector BM25 Graph         Cats.md                                   cats have lived alongside humans for tho
🟣 Score select                    ✅    0.0008 [0.500]  (0.5000)    +0.0002  Vector BM25 Graph         Apes.docx                                 Great apes (Hominidae) are the closest l
🟣 Score select                    ✅    0.0007 [0.500]  (0.5000)    +0.0002  Vector BM25 Graph         Dogs.png                                  tasks. understanding breed-specific ende
🟣 Score select                    ✅    0.0005 [0.500]  (0.5000)    +0.0001  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
🟣 Score select                    ✅    0.0005 [0.500]  (0.5000)    +0.0001  Vector      Graph         Cats.md                                   cats suffered during periods of supersti
🟣 Score select                    ✅    0.0004 [0.500]  (0.5000)    +0.0001  Vector      Graph         Fish.txt                                  terminology explained cold-blooded: a co
🟣 Score select                    ✅    0.0004 [0.500]  (0.5000)    +0.0001  Vector                    Dogs.png                                  , comprehensive text dogs dogs accompani
🟣 Score select                    ✅    0.0004 [0.500]  (0.5000)    +0.0001  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
🟣 Score select                    ✅    0.0003 [0.500]  (0.5000)    +0.0001  Vector      Graph         Cats.md                                   - kingdom: animalia - phylum: chordata -
🟣 Score select                    ✅    0.0003 [0.500]  (0.5000)    +0.0001  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
🟣 Score select                    ✅    0.0003 [0.500]  (0.5000)    +0.0001  Vector      Graph         Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernati
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0001  Vector      Graph         Fish.txt                                  ectotherm: the preferred scientific term
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0001  Vector      Graph         Cats.md                                   these breeds emerged without heavy human
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   most adult cats are lactose intolerant.
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0000  Vector BM25 Graph         Cats.md                                   even well-fed cats hunt. the sequence: 1
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats were revered, often associated with
🟣 Score select                    ✅    0.0002 [0.500]  (0.5000)    +0.0000  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector                    Apes.docx                                 All great apes demonstrate remarkable co
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector                    Cats.md                                   - climbing structures - scratching posts
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector BM25 Graph         Cats.md                                   indoor cats often live 12–18 years; some
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector                    Cats.md                                   a deep, structured exploration of domest
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   outdoor access: - pros: exercise, stimul
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector BM25 Graph         Cats.md                                   cats are often described as solitary, bu
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   they have a righting reflex, but falls f
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   free-roaming cats are estimated to kill
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector                    Cats.md                                   - puzzle toys - training sessions - nove
🟣 Score select                    ✅    0.0001 [0.500]  (0.5000)    +0.0000  Vector                    Cats.md                                   spaying/neutering is essential to reduce
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Lions.pptx                                Slide 2: Physical Characteristics Male l
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats dominate digital culture: - memes (
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats are crepuscular hunters with: - a h
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats are complex, adaptable, and endless
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats can detect frequencies up to ~64 kh
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats were not bred for specific tasks ea
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    Cats.md                                   - human interaction - multi-cat househol
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats communicate through: - vocalization
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   kittens engage in: - stalking. - pouncin
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   - dental disease - kidney disease - hype
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   the mechanism of purring is still debate
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   - regular veterinary checkups - vaccinat
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Cats.md                                   cats form strong bonds but express affec
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector      Graph         Fish.txt                                  poikilotherm: a term describing organism
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                - **processors**: dual enterprise platin
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    BlazingFast_Workstation.md                the blazingfast workstation is an enterp
🔵 Strategy: default              66 chunks selected after thresholding  0.5000  single-chunk boost ×1.25
🔵 Chunk selector: Score          Ranked     Selected 50 chunks.
🔵 Selected                       50 chunks selected
   ⚪ Selected                        #   Score [Sigmoid]  File                                      Text
   ⚪ Selected                       --------------------------------------------------------------------------------------------------------------------------------------------
   ⚪ Selected                        1    1.0000 [0.728]  Hedgehogs.pdf                             PDF page 1 suburban gardens and agricultural edges. They are native to
   ⚪ Selected                        2    0.9669 [0.722]  Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs are small, nocturnal mammals kn
   ⚪ Selected                        3    0.5285 [0.627]  Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations include a keen sense of smell
   ⚪ Selected                        4    0.3579 [0.587]  Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, decisive bite to subdue it. Season
   ⚪ Selected                        5    0.1120 [0.528]  Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—such as netting, open drains, and
   ⚪ Selected                        6    0.0113 [0.503]  Apes.docx                                 Orangutans are the only great apes found in Asia, inhabiting the rainf
   ⚪ Selected                        7    0.0095 [0.502]  Elephants.jpg                             elephants large, long-lived mammals family elephantidae, represented t
   ⚪ Selected                        8    0.0076 [0.502]  Apes.docx                                 Gorillas are the largest living primates, with males weighing up to 20
   ⚪ Selected                        9    0.0068 [0.502]  Apes.docx                                 Chimpanzees (Pan troglodytes) are found across equatorial Africa. They
   ⚪ Selected                       10    0.0052 [0.501]  Fish.txt                                  ecological roles and importance fish are central to aquatic food webs:
   ⚪ Selected                       11    0.0029 [0.501]  Lions.pptx                                Slide 3: Hunting and Diet Lions are cooperative hunters, working toget
   ⚪ Selected                       12    0.0021 [0.501]  Fish.txt                                   what fish are fish are a diverse group of aquatic vertebrates that li
   ⚪ Selected                       13    0.0017 [0.500]  Lions.pptx                                Slide 1: Lions: The King of the Savanna Lions (Panthera leo) are the o
   ⚪ Selected                       14    0.0014 [0.500]  Fish.txt                                  diversity: fish include jawless fishes (lampreys, hagfish), cartilagin
   ⚪ Selected                       15    0.0013 [0.500]  Lions.pptx                                Slide 4: Conservation Status Lions are classified as Vulnerable by the
   ⚪ Selected                       16    0.0012 [0.500]  LionsAndApes.xlsx                         category detail value taxonomy scientific name panthera leo taxonomy f
   ⚪ Selected                       17    0.0010 [0.500]  Cats.md                                   - free feeding can lead to obesity. - puzzle feeders encourage natural
   ⚪ Selected                       18    0.0008 [0.500]  Cats.md                                   cats have lived alongside humans for thousands of years, yet they rema
   ⚪ Selected                       19    0.0008 [0.500]  Apes.docx                                 Great apes (Hominidae) are the closest living relatives to humans, sha
   ⚪ Selected                       20    0.0007 [0.500]  Dogs.png                                  tasks. understanding breed-specific endencies essential training, welf
   ⚪ Selected                       21    0.0005 [0.500]  Cats.md                                   as placental mammals, cats give live birth and nurse their young with
   ⚪ Selected                       22    0.0005 [0.500]  Cats.md                                   cats suffered during periods of superstition, especially black cats. i
   ⚪ Selected                       23    0.0004 [0.500]  Fish.txt                                  terminology explained cold-blooded: a common everyday phrase meaning a
   ⚪ Selected                       24    0.0004 [0.500]  Dogs.png                                  , comprehensive text dogs dogs accompanied humans tens thousands years
   ⚪ Selected                       25    0.0003 [0.500]  Cats.md                                   cats require: - taurine - arachidonic acid - preformed vitamin a - hig
   ⚪ Selected                       26    0.0003 [0.500]  Cats.md                                   - kingdom: animalia - phylum: chordata - class: mammalia (mammals) - o
   ⚪ Selected                       27    0.0003 [0.500]  Apes.docx                                 All great ape species are either endangered or critically endangered.
   ⚪ Selected                       28    0.0003 [0.500]  Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernation physiology, and the impacts
   ⚪ Selected                       29    0.0002 [0.500]  Fish.txt                                  ectotherm: the preferred scientific term for animals that rely mainly
   ⚪ Selected                       30    0.0002 [0.500]  Cats.md                                   these breeds emerged without heavy human intervention: - maine coon -
   ⚪ Selected                       31    0.0002 [0.500]  Cats.md                                   most adult cats are lactose intolerant. [section: the comprehensive fe
   ⚪ Selected                       32    0.0002 [0.500]  Cats.md                                   even well-fed cats hunt. the sequence: 1. detect 2. stalk 3. chase 4.
   ⚪ Selected                       33    0.0001 [0.500]  Cats.md                                   cats were revered, often associated with: - bastet (goddess of protect
   ⚪ Selected                       34    0.0001 [0.500]  Dogs.png                                  impact dogs human well-being. despite many strengths, dogs also face c
   ⚪ Selected                       35    0.0001 [0.500]  Apes.docx                                 All great apes demonstrate remarkable cognitive abilities including se
   ⚪ Selected                       36    0.0001 [0.500]  Cats.md                                   - climbing structures - scratching posts - window perches [section: th
   ⚪ Selected                       37    0.0001 [0.500]  Cats.md                                   indoor cats often live 12–18 years; some exceed 20. [section: the comp
   ⚪ Selected                       38    0.0001 [0.500]  Cats.md                                   a deep, structured exploration of domestic cats (felis catus) [section
   ⚪ Selected                       39    0.0001 [0.500]  Cats.md                                   outdoor access: - pros: exercise, stimulation - cons: predation, traff
   ⚪ Selected                       40    0.0001 [0.500]  Cats.md                                   cats are often described as solitary, but this is an oversimplificatio
   ⚪ Selected                       41    0.0001 [0.500]  Cats.md                                   they have a righting reflex, but falls from low heights can be dangero
   ⚪ Selected                       42    0.0000 [0.500]  Cats.md                                   free-roaming cats are estimated to kill billions of birds and small ma
   ⚪ Selected                       43    0.0000 [0.500]  Cats.md                                   whiskers are deeply rooted sensory hairs that detect: - airflow change
   ⚪ Selected                       44    0.0000 [0.500]  Cats.md                                   - puzzle toys - training sessions - novel objects [section: the compre
   ⚪ Selected                       45    0.0000 [0.500]  Cats.md                                   spaying/neutering is essential to reduce: - stray populations - shelte
   ⚪ Selected                       46    0.0000 [0.500]  Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gestation lasts about 110 days, pr
   ⚪ Selected                       47    0.0000 [0.500]  Lions.pptx                                Slide 2: Physical Characteristics Male lions weigh between 150-250 kg
   ⚪ Selected                       48    0.0000 [0.500]  Cats.md                                   cats dominate digital culture: - memes (“i can has cheezburger?”) - vi
   ⚪ Selected                       49    0.0000 [0.500]  Cats.md                                   cats are crepuscular hunters with: - a high density of rod photorecept
   ⚪ Selected                       50    0.0000 [0.500]  Cats.md                                   cats are complex, adaptable, and endlessly fascinating creatures. thei
   ⚪ KeywordActionSplitter          Final context query_id=0: 50 associated chunks for 'What do hedgehogs eat?' (association is not proof of answer support)
   ⚪ KeywordActionSplitter          Final context query_id=1: 50 associated chunks for 'Where do hedgehogs live?' (association is not proof of answer support)
   ⚪ ChunkSelect                    After chunk selection: 50/66 kept
🔵 Selection Confidence           🟡 MEDIUM C_final=0.77 evidence_chunks=50 rerank_fallback=off
🔵 Retrieval Orchestration        language: en      retrieval completed chosen=50 context_chars=41902
🔵 KeywordActionSplitter          Final answer instruction: Complete user request:
↳                                 what do hedgehogs eat and where do they live? Create a table.
↳
↳                                 Self-contained answer instruction:
↳                                 Using only the retrieved context, describe what hedgehogs eat and where they live in a table.
↳
↳                                 Requested operation to execute on the retrieved evidence:
↳                                 Create a table.
↳                                 Perform the operation, not a description of how to perform it. For a requested list, use Markdown bullets; for a requested table, use a Markdown table, unless the
↳                                 user explicitly specifies another format.
↳
↳                                 Add evidence phrases.
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] context=32768 reserved_sys=1024 prompt≈8479 → max_output_tokens=2048
🔵 Resolved token params          max_output_tokens(api: max_tokens)=2048  num_ctx=32768  (override: max_tokens=None  num_ctx=None)
🔵 LLM Plan                       Answer generation: producing the final grounded response from retrieved context chunks.
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] num_ctx=32768 prompt≈8479 → num_predict=2048
🔵 Call LLM                       Stage: Run user prompt
🔵 Call LLM                       Model: openai/qwen2_5_7b_instruct prompt template: _PROMPT_CHAT purpose: Inference
🔵 Call LLM                       options: {'temperature': 0.1, 'top_k': 40, 'top_p': 0.92, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_7b_instruct took 00:17
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_compliance_language): HIGH (confidence: 100%, threshold: 60%)
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🟢 Cache build Regex Banned       Built compiled Regex cache with 80 entries for language english stage: PIPELINE_CHECK
🟢 KeyWrdChk Summary              Phrase                         dpt / brth     Regex+Levenshtein ➡️    Jaccard ➡️              BM25 ➡️                 Keybert
🟡 KeyWrdChk Summary              child                          1/4    1/4      -/-                    -/-                    -/-                    0.4004/0.4000
🟡 KeyWrdChk Summary              diagnosis                      1/4    1/4      -/-                    -/-                    -/-                    0.4045/0.4000
🟡 KeyWrdChk Summary              ethnicity                      1/4    1/4      -/-                    -/-                    -/-                    0.4215/0.4000
🟡 KeyWrdChk Summary              hack                           1/4    1/4      -/-                    -/-                    -/-                    0.4023/0.4000
🟡 KeyWrdChk Summary              security question              1/4    1/4      -/-                    -/-                    -/-                    0.4219/0.4000
🟢 KeyWrdChk Depth                1 algos passed threshold vs. required 4
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 4
   ⚪ Grounding                      Inference answer before grounding validation:
   ↳                                 | Diet | Description |
   ↳                                 |------|-------------|
   ↳                                 | Invertebrates | Beetles, caterpillars, earthworms, slugs, and snails |
   ↳                                 | Small vertebrates | Amphibians, small rodents |
   ↳                                 | Fallen fruit | When available |
   ↳                                 | Vegetation | When available |
   ↳
   ↳                                 | Habitat | Description |
   ↳                                 |---------|-------------|
   ↳                                 | Suburban gardens and agricultural edges | Native to much of Europe, parts of Asia, and Africa |
   ↳                                 | Hedgerows | Provide cover and abundant invertebrate prey |
   ↳                                 | Compost heaps | Provide cover and invertebrate prey |
   ↳                                 | Mixed vegetation | Provide cover and invertebrate prey |
   ↳                                 | Urban and suburban areas | Can be hospitable when green spaces, hedges, and insect-rich gardens are available |
   ↳
   ↳                                 ### Sources
   ↳                                 - Hedgehogs.pdf
   ↳                                   - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf
   ↳                                   - Page: 1
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_grounding_language): HIGH (confidence: 100%, threshold: 60%)
🟡 Grounding                      Grounding requirements were not satisfied by retrieved context; generated list, table, or Mermaid diagram shown unverified.
🔵 Deleted chat context           Deleted chat context for MyFirstChat in Test_embedder_bge_m3_ChatContext
🔵 Add chat context               Upserted turn 1 for chat_name=MyFirstChat file_tag='' lang='english' to Test_embedder_bge_m3_ChatContext
   ⚪ Retrieval Orchestration        Stored query history forms orig='what do hedgehogs eat and where do they live? Create a table.' t1='what do hedgehogs eat and where do they live? Create a table.'
   ↳                                 t2='what do hedgehogs eat and where do they live? Create a table.' effective_reason=''
🔵 TokenBudget                    /v1/models did not yield max_model_len for 'openai/qwen2_5_14b_instruct'; retrying via LiteLLM /model/info endpoint
🔵 TokenBudget                    Detected context_length=8192 for openai/qwen2_5_14b_instruct via vllm adapter
🔵 TokenBudget                    Model 'openai/qwen2_5_14b_instruct': context_limit=8192 (below cap 32768 — using detected value)
🔵 TokenBudget                    Caller did not supply context_size; resolved num_ctx=8192 for openai/qwen2_5_14b_instruct
🔵 TokenBudget                    Authoritative budget=256 vs caller estimate=0; using authoritative value
🔵 TokenBudget                    [openai/qwen2_5_14b_instruct] num_ctx=8192 prompt≈421 → num_predict=256
🔵 Call LLM                       Stage: Answer-text LLM review
🔵 Call LLM                       Model: openai/qwen2_5_14b_instruct prompt template: None purpose: Judge
🔵 Call LLM                       options: {'temperature': 0.0, 'num_predict': 256, 'num_ctx': 8192} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_14b_instruct took 00:02
🟡 VisualMarker                   _mark_sources called: 50 chunk(s)
🔵 VisualMarker                   Hedgehogs.pdf: 83 new highlights on physical pages 1, 2.
🔵 VisualMarker                   Prepared 1 highlighted document(s) (in memory)
🔵 Marked sources                 1 highlighted document(s) (click a link to open; use 'Save As' in the viewer to keep a copy):
   📎 Hedgehogs.pdf (highlighted): file:///D:/RAG-LCC/tmp/ragchat/rag_marked_32odazso/Hedgehogs_marked.pdf
🔵 Answer preview                 Formatted browser page (click to open; expires after 1800 s or at shutdown):
   http://127.0.0.1:11436/marked/d7eE91N98OsAg3xwIdDfx8VbKKCe-A3YhRSlSYDxlkY.html?token=YqgWVhJDVvZVhEK4pfq6LreCoNgwlq5v
Original query: what do hedgehogs eat and where do they live? Create a table.
💬 >
 ⚠️  Not fully grounded: The generated output is shown below, but some or all claims could not be verified against the retrieved context. Treat it as unverified, not as confirmed facts. Source
 references are not verified citations.


  Diet               Description
  ───────────────────────────────────────────────────────────────────────
  Invertebrates      Beetles, caterpillars, earthworms, slugs, and snails
  Small vertebrates  Amphibians, small rodents
  Fallen fruit       When available
  Vegetation         When available



  Habitat                                  Description
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Suburban gardens and agricultural edges  Native to much of Europe, parts of Asia, and Africa
  Hedgerows                                Provide cover and abundant invertebrate prey
  Compost heaps                            Provide cover and invertebrate prey
  Mixed vegetation                         Provide cover and invertebrate prey
  Urban and suburban areas                 Can be hospitable when green spaces, hedges, and insect-rich gardens are available


 Sources

  • Hedgehogs.pdf
     • FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf ([open](file:///D:/RAG-LCC/TestDocs/Hedgehogs.pdf))
     • Page: 1
 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 Answer confidence: 🟠 LOW

 Heuristic Confidence:

  • Note                    : grounding failed; generated list, table, or Mermaid diagram shown unverified. Retrieval metrics describe selected chunks, not verified answer support
  • C_final=0.00            : final confidence in [0,1] after penalties.
  • C_top=0.36              : weighted best-chunk signal (0.50 x top_score).
  • C_mean_top3=0.21        : weighted top-3 stability (0.30 x mean_top3).
  • C_coverage=0.15         : weighted evidence coverage (0.15 x coverage).
  • C_local=0.05            : weighted local-source share (0.05 x local_share).
  • C_fallback_penalty=0.00 : subtraction when low-confidence rerank fallback was used.
  • top_score=0.73          : strongest selected chunk confidence.
  • mean_top3=0.69          : average confidence of the top 3 selected chunks.
  • coverage=1.00           : selected_chunks / final_chunks_to_llm, capped at 1.00.
  • local_share=1.00        : fraction of selected chunks from local files.
  • evidence_chunks=50      : number of selected evidence chunks.
  • rerank_fallback=off     : whether fallback rerank logic was triggered.

 Static retrieval diagnostics (not answer-quality verification):

  • Evidence relevance   : heuristic=0.74 Top retrieval scores and fallback status; does not verify answer accuracy.
  • Retrieval coverage   : heuristic=1.00 Selected chunk count relative to the target, plus evidence quantity; does not measure answer completeness.
  • Evidence availability: heuristic=0.80 Retrieval relevance and chunk availability; does not verify citations or claim-to-source support.

 LLM review - answer text only (judge model: openai/qwen2_5_14b_instruct):

  • Internal consistency: llm=1.00 Judge rating of contradictions within the answer; does not verify factual correctness.
  • Explanation clarity : not assessed Judge rating of how understandable the explanation is; does not verify reasoning validity. Subjective ratings, not probabilities. The judge sees neither the
    question nor source evidence and may share the generator's blind spots. Accuracy, answer completeness, reasoning validity, and citation support: not assessed.


---
### Document metadata

- **Hedgehogs.pdf**
    - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf
    - Creator: Microsoft® Word for Microsoft 365
    - Producer: Microsoft® Word for Microsoft 365
    - DocCreated: 2026-02-17 12:32:04+01:00
    - DocModified: 2026-02-17 12:32:04+01:00
    - Pages: 1, 2
- **Apes.docx**
    - FilePath: D:/RAG-LCC/TestDocs/Apes.docx
    - Author: python-docx
    - DocCreated: 2013-12-23 23:15:00+00:00
    - DocModified: 2013-12-23 23:15:00+00:00
- **Elephants.jpg**
    - FilePath: D:/RAG-LCC/TestDocs/Elephants.jpg
    - FileSizeBytes: 282341
    - FileModified: 2026-06-23 14:17:15.110956
- **Fish.txt**
    - FilePath: D:/RAG-LCC/TestDocs/Fish.txt
    - FileSizeBytes: 3474
    - FileModified: 2026-06-23 14:17:15.094329
- **Lions.pptx**
    - FilePath: D:/RAG-LCC/TestDocs/Lions.pptx
    - DocCreated: 2013-01-27 09:14:16
    - DocModified: 2013-01-27 09:15:58
    - LastModifiedBy: Steve Canny
- **LionsAndApes.xlsx**
    - FilePath: D:/RAG-LCC/TestDocs/LionsAndApes.xlsx
    - Author: openpyxl
    - DocCreated: 2026-04-15 12:24:52
    - DocModified: 2026-04-15 12:24:52
- **Cats.md**
    - FilePath: D:/RAG-LCC/TestDocs/Cats.md
    - FileSizeBytes: 7323
    - FileModified: 2026-06-23 14:17:15.094329
- **Dogs.png**
    - FilePath: D:/RAG-LCC/TestDocs/Dogs.png
    - FileSizeBytes: 634064
    - FileModified: 2026-06-23 14:17:15.110956
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
type new: your question to start a new topic  · type summarize: [file name or path] (alias: summary:) to summarize a file (no argument: pinned document; no pin: open the path picker)  · b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat
💬 Your actual query>

Exiting HarinezumIgel RAG chat.

🟢 RAG_CHAT_QUERIES:              0
🔵 Logs                           Created xlsx file D:\RAG-LCC\logs\RAGChat\RAGChat_HUMAN_REVIEW_20261119_094842.xlsx
🔵 Logs                           Have a look at RAG_CHAT_QUERIES .xlsx / .csv file D:\RAG-LCC\logs\RAGChat\RAGChat_HUMAN_REVIEW_20261119_094842.xlsx /
↳                                 D:\RAG-LCC\logs\RAGChat\RAGChat_HUMAN_REVIEW_20261119_094842.csv

(.venv) PS D:\RAG-LCC>
```
