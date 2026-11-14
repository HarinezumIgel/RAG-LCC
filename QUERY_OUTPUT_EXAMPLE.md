# Query Output Example

## Current confidence display

Please note that the actual output includes more color; copying and pasting from the VS Code terminal may reduce it.

```text
(.venv) PS D:\RAG-LCC> python .\src\Apps\RAGChat.py
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
🦔            Version v0.5.2.1/1645 2026-11-14                          🦔
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
💬 Your actual query>  create a table with the important properties of the blazingfast workstation

🔵 LangDetect                     Detected language: English (en) — confidence: 87% (threshold: 60%)
🔵 LangDetect                     Confidence level (default): HIGH (confidence: 87%, threshold: 60%)
🔵 Call LLM                       Stage: Encode
🔵 Call LLM                       Model: bge-m3-embedder prompt template: None purpose: Embedding 1 text(s) via http://192.168.100.50:4000/v1/embeddings
🟢 Cache build Regex Banned       Built compiled Regex cache with 80 entries for language english stage: PROMPT_CHECK
🟢 Cache build Jaccard Banned     Built Jaccard n-gram cache with 80 entries for language english
🟢 BM25 Scorer Cache              Built BM25 banlist cache with 80 entries for language english
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🟢 Cache build Keybert Scorer     Built KeyBert embeddings cache with 80 entries
🟢 KeyWrdChk Summary              Phrase                         dpt / brth     Regex+Levenshtein ➡️    Jaccard ➡️              BM25 ➡️                 Keybert
🟡 KeyWrdChk Summary              buffer overflow                1/2    1/3      -/-                    -/-                    -/-                    0.4614/0.4500
🟡 KeyWrdChk Summary              data processor                 1/2    1/3      -/-                    -/-                    -/-                    0.4545/0.4500
🟢 KeyWrdChk Depth                1 algos passed threshold vs. required 2
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 3
🔵 TokenBudget                    /v1/models did not yield max_model_len for 'llama_guard3_8b'; retrying via LiteLLM /model/info endpoint
🔵 TokenBudget                    Detected context_length=1024 for llama_guard3_8b via vllm adapter
🔵 TokenBudget                    Model 'llama_guard3_8b': context_limit=1024 (below cap 32768 — using detected value)
🔵 TokenBudget                    [llama_guard3_8b] context=1024 reserved_sys=64 prompt≈14 → max_output_tokens=64
🔵 LLM Plan                       Prompt validation: running the guard model to classify the user prompt as allow/block.
🔵 TokenBudget                    [llama_guard3_8b] num_ctx=1024 prompt≈14 → num_predict=64
🔵 Call LLM                       Stage: Check provided prompt
🔵 Call LLM                       Model: llama_guard3_8b prompt template: _PROMPT_CHECK_CHAT_LLAMA_GUARD purpose: Prompt check
🔵 Call LLM                       options: {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': 64, 'num_ctx': 1024} streaming: False
🔵 Call LLM                       Elapsed time calling: llama_guard3_8b took 00:00
🟢 CheckPrompt                    Provided prompt is considered compliant by: llama_guard3_8b. Reason: Detected safe keywords
🔵                                Chatter RAG Query LMM: openai/qwen2_5_7b_instruct
🔵 Retrieval Orchestration        Chosen orchestration flow: NATIVE_ALL
🔵 Retrieval Orchestration        start retrieval turn
🟢 Chroma Collection              Using Chroma DB collection Test_embedder_bge_m3
🔵 VectorStore                    Set Chroma vector store. Name: Test_embedder_bge_m3 Path: D:\RAG-LCC\chromadb\docs\Test_embedder_bge_m3
🔵 Retrieval Orchestration        prepare session context
🔵 Retrieval Orchestration        normalize user query
🔵 UserQuery                      Original user query: 'create a table with the important properties of the blazingfast workstation'
🔵 LangDetect                     Detected language: English (en) — confidence: 87% (threshold: 60%)
🔵 LangDetect                     Confidence level (raw_user_query): HIGH (confidence: 87%, threshold: 60%)
   ⚪ QueryPlan                      Step 1: detected query language=english; query='create a table with the important properties of the blazingfast workstation'
🔵 QueryRewrite                   No conversation history — skipping rewrite
🔵 LangDetect                     Detected language: English (en) — confidence: 87% (threshold: 60%)
🔵 LangDetect                     Confidence level (rewrite_query): HIGH (confidence: 87%, threshold: 60%)
   ⚪ QueryPlan                      Step 2: rewrite completed; language=english; query='create a table with the important properties of the blazingfast workstation'
🔵 LangDrift                      raw_query_language=english  orig_translated_query_en='create a table with the important properties of the blazingfast workstation'
🔵 LangDrift                      rewrite_language=english  rewritten_query='create a table with the important properties of the blazingfast workstation'
🔵 LangDrift                      post_rewrite_query_en='create a table with the important properties of the blazingfast workstation'
🔵 FinalQuery                     Final query for retrieval: 'create a table with the important properties of the blazingfast workstation' (unchanged)
   ⚪ QueryPlan                      Step 3: generated 0 alternatives after rewrite: []
🔵 Retrieval Orchestration        query rewrite flow:
↳                                 seed_en='create a table with the important properties of the blazingfast workstation'
↳                                 rewritten='create a table with the important properties of the blazingfast workstation'
↳                                 final_en='create a table with the important properties of the blazingfast workstation'
🔵 Retrieval Orchestration        language: en      normalized query final='create a table with the important properties of the blazingfast workstation' alternate_count=0
🔵 Retrieval Orchestration        language: en      apply retrieval gates
🔵 KeywordActionSplitter          model='openai/qwen2_5_7b_instruct' language='english' inputs={'original_request': 'create a table with the important properties of the blazingfast workstation',
↳                                 'translated_request': 'create a table with the important properties of the blazingfast workstation', 'accepted_rewrite': 'create a table with the important properties
↳                                 of the blazingfast workstation'}
🔵 TokenBudget                    Authoritative budget=2048 vs caller estimate=1024; using authoritative value
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] num_ctx=32768 prompt≈1346 → num_predict=2048
🔵 Call LLM                       Stage: Keyword/action splitting
🔵 Call LLM                       Model: openai/qwen2_5_7b_instruct prompt template: _PROMPT_KEYWORD_ACTION_SPLITTER purpose: Local retrieval decomposition
🔵 Call LLM                       options: {'temperature': 0.05, 'top_k': 10, 'top_p': 0.9, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_7b_instruct took 00:06
🔵 KeywordActionSplitter          LLM call + validation: 6.792 s
🟢 KeywordActionSplitter          Validated decomposition:
↳                                 Retrieval queries (1):
↳                                 [query_id=0] What are the important properties of the blazingfast workstation?
↳                                 Action: Create a table of the important properties of the blazingfast workstation.
↳                                 Answer instruction: Using only the retrieved context, create a table listing the important properties of the blazingfast workstation.
↳                                 Entities: blazingfast workstation
↳                                 Keywords: important properties, workstation, specifications
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
🔵 Retrieval Orchestration        original-language vector leg: off lang=en query='create a table with the important properties of the blazingfast workstation' reason=disabled by orchestration flow
🔵 Retrieval Orchestration        language: en      stage plan flow=NATIVE_ALL mode=ALL guardrail=off original_leg=off rewrite=on pronouns=on rerank=on low_score_fallback=on low_recall_rescue=on
↳                                 grounding=on indexed_shape=on indexed_translate=on indexed_wordnet=off alt_queries=0 retrievers=v=on,b=on,g=on,r=on
🔵 Retrieval Orchestration        effective stage gates: flow=NATIVE_ALL mode=ALL web_mode=local_only local_stage=on web_stage=off
🔵 Retrieval Orchestration        language: en      retrieve local candidates mode=ALL
   ⚪ KeywordActionSplitter          Local query_id=0: 'What are the important properties of the blazingfast workstation?'
🔵 Retrieval Orchestration        discovered corpus language buckets: [de, en] active config buckets: [de, en, es, fr, it]
   ⚪ QueryPlan                      Step 4: document languages=['de', 'en']; active/present=['de', 'en']; metadata limit=whole collection
🔵 Retrieval Orchestration        query language flow: user=en current=en rewrite=en retrieval=en
🔵 Retrieval Orchestration        retriever: Vector     query: 'What are the important properties of the blazingfast workstation?' dispatch query
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test_embedder_bge_m3 for language en
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance                Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6921     0.3079    0.3079  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              2        0.6551     0.3449    0.3449  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              3        0.6526     0.3474    0.3474  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              4        0.6484     0.3516    0.3516  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              5        0.6314     0.3686    0.3686  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              6        0.6158     0.3842    0.3842  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                              7        0.3345     0.6655    0.6655  Vector                     Cats.md
   ⚪ Chroma                              8        0.3315     0.6685    0.6685  Vector                     Lions.pptx
   ⚪ Chroma                              9        0.3297     0.6703    0.6703  Vector                     Lions.pptx
   ⚪ Chroma                             10        0.3292     0.6708    0.6708  Vector                     Apes.docx
   ⚪ Chroma                             11        0.3274     0.6726    0.6726  Vector                     Cats.md
   ⚪ Chroma                             12        0.3262     0.6738    0.6738  Vector                     Cats.md
   ⚪ Chroma                             13        0.3246     0.6754    0.6754  Vector                     Apes.docx
   ⚪ Chroma                             14        0.3244     0.6756    0.6756  Vector                     Cats.md
   ⚪ Chroma                             15        0.3146     0.6854    0.6854  Vector                     Cats.md
   ⚪ Chroma                             16        0.3075     0.6925    0.6925  Vector                     Elephants.jpg
   ⚪ Chroma                             17        0.3075     0.6925    0.6925  Vector                     Cats.md
   ⚪ Chroma                             18        0.3074     0.6926    0.6926  Vector                     Kamele.txt
   ⚪ Chroma                             19        0.3049     0.6951    0.6951  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             20        0.3022     0.6978    0.6978  Vector                     Dogs.png
   ⚪ Chroma                             21        0.3015     0.6985    0.6985  Vector                     Cats.md
   ⚪ Chroma                             22        0.3011     0.6989    0.6989  Vector                     Dogs.png
   ⚪ Chroma                             23        0.2991     0.7009    0.7009  Vector                     Apes.docx
   ⚪ Chroma                             24        0.2981     0.7019    0.7019  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             25        0.2980     0.7020    0.7020  Vector                     Pferde.pdf
   ⚪ Chroma                             26        0.2955     0.7045    0.7045  Vector                     Cats.md
   ⚪ Chroma                             27        0.2955     0.7045    0.7045  Vector                     Pferde.pdf
   ⚪ Chroma                             28        0.2926     0.7074    0.7074  Vector                     Kamele.txt
   ⚪ Chroma                             29        0.2890     0.7110    0.7110  Vector                     Cats.md
   ⚪ Chroma                             30        0.2869     0.7131    0.7131  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             31        0.2839     0.7161    0.7161  Vector                     LionsAndApes.xlsx
   ⚪ Chroma                             32        0.2832     0.7168    0.7168  Vector                     Cats.md
   ⚪ Chroma                             33        0.2812     0.7188    0.7188  Vector                     Cats.md
   ⚪ Chroma                             34        0.2749     0.7251    0.7251  Vector                     Apes.docx
   ⚪ Chroma                             35        0.2749     0.7251    0.7251  Vector                     Kamele.txt
   ⚪ Chroma                             36        0.2747     0.7253    0.7253  Vector                     Kamele.txt
   ⚪ Chroma                             37        0.2745     0.7255    0.7255  Vector                     Pferde.pdf
   ⚪ Chroma                             38        0.2743     0.7257    0.7257  Vector                     Cats.md
   ⚪ Chroma                             39        0.2738     0.7262    0.7262  Vector                     Cats.md
   ⚪ Chroma                             40        0.2724     0.7276    0.7276  Vector                     Apes.docx
   ⚪ Chroma                             41        0.2712     0.7288    0.7288  Vector                     Cats.md
   ⚪ Chroma                             42        0.2707     0.7293    0.7293  Vector                     Cats.md
   ⚪ Chroma                             43        0.2704     0.7296    0.7296  Vector                     Cats.md
   ⚪ Chroma                             44        0.2683     0.7317    0.7317  Vector                     Lions.pptx
   ⚪ Chroma                             45        0.2676     0.7324    0.7324  Vector                     Fish.txt
   ⚪ Chroma                             46        0.2644     0.7356    0.7356  Vector                     Cats.md
   ⚪ Chroma                             47        0.2638     0.7362    0.7362  Vector                     Cats.md
   ⚪ Chroma                             48        0.2636     0.7364    0.7364  Vector                     Cats.md
   ⚪ Chroma                             49        0.2620     0.7380    0.7380  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             50        0.2603     0.7397    0.7397  Vector                     Fish.txt
   ⚪ Chroma                             51        0.2602     0.7398    0.7398  Vector                     Dogs.png
   ⚪ Chroma                             52        0.2599     0.7401    0.7401  Vector                     Lions.pptx
   ⚪ Chroma                             53        0.2598     0.7402    0.7402  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             54        0.2598     0.7402    0.7402  Vector                     Cats.md
   ⚪ Chroma                             55        0.2559     0.7441    0.7441  Vector                     Cats.md
   ⚪ Chroma                             56        0.2540     0.7460    0.7460  Vector                     Cats.md
   ⚪ Chroma                             57        0.2505     0.7495    0.7495  Vector                     Cats.md
   ⚪ Chroma                             58        0.2502     0.7498    0.7498  Vector                     Apes.docx
   ⚪ Chroma                             59        0.2490     0.7510    0.7510  Vector                     Kamele.txt
   ⚪ Chroma                             60        0.2487     0.7513    0.7513  Vector                     Cats.md
   ⚪ Chroma                             61        0.2478     0.7522    0.7522  Vector                     Cats.md
   ⚪ Chroma                             62        0.2473     0.7527    0.7527  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             63        0.2452     0.7548    0.7548  Vector                     Fish.txt
   ⚪ Chroma                             64        0.2445     0.7555    0.7555  Vector                     Pferde.pdf
   ⚪ Chroma                             65        0.2422     0.7578    0.7578  Vector                     Fish.txt
   ⚪ Chroma                             66        0.2387     0.7613    0.7613  Vector                     Cats.md
   ⚪ Chroma                             67        0.2379     0.7621    0.7621  Vector                     Kamele.txt
   ⚪ Chroma                             68        0.2369     0.7631    0.7631  Vector                     Cats.md
   ⚪ Chroma                             69        0.2341     0.7659    0.7659  Vector                     Fish.txt
   ⚪ Chroma                             70        0.2236     0.7764    0.7764  Vector                     Fish.txt
   ⚪ Chroma                             71        0.2233     0.7767    0.7767  Vector                     Cats.md
   ⚪ Chroma                             72        0.2202     0.7798    0.7798  Vector                     Cats.md
   ⚪ Chroma                             73        0.2181     0.7819    0.7819  Vector                     Lions.pptx
   ⚪ Chroma                             74        0.2144     0.7856    0.7856  Vector                     Cats.md
   ⚪ Chroma                             75        0.2088     0.7912    0.7912  Vector                     Cats.md
   ⚪ Chroma                             76        0.1854     0.8146    0.8146  Vector                     Cats.md
🟢 Chroma                         Querying Chroma DB query returned 76 chunks
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=en retrievers=BM25/Graph/Regex shaping=on changed=0
↳                                 primary_query source=REWRITTEN_QUERY lang=en->en
↳                                   original='What are the important properties of the blazingfast workstation?'
↳                                   staged='What are the important properties of the blazingfast workstation?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->en
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->en count=0
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=de retrievers=BM25/Graph/Regex shaping=on changed=1
↳                                 primary_query source=REWRITTEN_QUERY lang=en->de
↳                                   original='What are the important properties of the blazingfast workstation?'
↳                                   staged='Was sind die wichtigen Eigenschaften der blazingfast Workstation?'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->de
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->de count=0
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: en      query: 'What are the important properties of the blazingfast workstation?'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: de      query: 'Was sind die wichtigen Eigenschaften der blazingfast Workstation?'
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language en
🟢 BM25                           Loaded persisted BM25 index (76 chunks, 1876 terms)
🟢 BM25                           BM25 retrieval returned 60 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        9.7139         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                2        6.9004         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                3        6.3437         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                4        6.3180         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                5        5.7467         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                6        5.3022         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                7        4.4309         BM25                Fish.txt
   ⚪ BM25                                8        4.0033         BM25                Apes.docx
   ⚪ BM25                                9        3.6318         BM25                Lions.pptx
   ⚪ BM25                               10        3.5538         BM25                Apes.docx
   ⚪ BM25                               11        3.4592         BM25                Hedgehogs.pdf
   ⚪ BM25                               12        3.3426         BM25                Hedgehogs.pdf
   ⚪ BM25                               13        3.3381         BM25                Cats.md
   ⚪ BM25                               14        3.2830         BM25                Cats.md
   ⚪ BM25                               15        3.2460         BM25                Apes.docx
   ⚪ BM25                               16        3.1372         BM25                Cats.md
   ⚪ BM25                               17        3.1165         BM25                Lions.pptx
   ⚪ BM25                               18        3.0589         BM25                Fish.txt
   ⚪ BM25                               19        3.0561         BM25                Cats.md
   ⚪ BM25                               20        3.0410         BM25                Lions.pptx
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=en; query='What are the important properties of the blazingfast workstation?'; hits={'BM25': 60, 'Graph': 0, 'Regex': 0}
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language de
🟢 BM25                           BM25 retrieval returned 10 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1       12.7400         BM25                Kamele.txt
   ⚪ BM25                                2       11.5524         BM25                Kamele.txt
   ⚪ BM25                                3       11.0521         BM25                Kamele.txt
   ⚪ BM25                                4       10.9409         BM25                Kamele.txt
   ⚪ BM25                                5       10.1036         BM25                Pferde.pdf
   ⚪ BM25                                6        9.8842         BM25                Kamele.txt
   ⚪ BM25                                7        9.7692         BM25                Pferde.pdf
   ⚪ BM25                                8        8.5805         BM25                Pferde.pdf
   ⚪ BM25                                9        6.1966         BM25                Pferde.pdf
   ⚪ BM25                               10        3.9042         BM25                Kamele.txt
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=de; query='Was sind die wichtigen Eigenschaften der blazingfast Workstation?'; hits={'BM25': 10, 'Graph': 0, 'Regex': 0}
🔵 Graph                          Querying graph index on collection Test_embedder_bge_m3 for language en
🟢 Graph                          Loaded persisted graph index (76 chunks, 1115 entities)
🟢 Graph                          Graph retrieval returned 4 chunks
   ⚪ Graph                             Pos    GraphScore                Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
   ⚪ Graph                               1       56.0000              Graph          BlazingFast_Workstation.md
   ⚪ Graph                               2       31.0000              Graph          BlazingFast_Workstation.md
   ⚪ Graph                               3       27.0000              Graph          BlazingFast_Workstation.md
   ⚪ Graph                               4       26.0000              Graph          BlazingFast_Workstation.md
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=en; query='What are the important properties of the blazingfast workstation?'; hits={'BM25': 0, 'Graph': 4, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language en
🟢 Regex                          Loaded persisted regex index (76 chunks, 293 verbs, 925 nouns)
🟢 Regex                          Regex retrieval returned 6 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ Regex                               1        0.6667         0         2                    Regex    BlazingFast_Workstation.md
   ⚪ Regex                               2        0.3333         0         1                    Regex    BlazingFast_Workstation.md
   ⚪ Regex                               3        0.3333         0         1                    Regex    BlazingFast_Workstation.md
   ⚪ Regex                               4        0.3333         0         1                    Regex    BlazingFast_Workstation.md
   ⚪ Regex                               5        0.3333         0         1                    Regex    BlazingFast_Workstation.md
   ⚪ Regex                               6        0.3333         0         1                    Regex    BlazingFast_Workstation.md
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=en; query='What are the important properties of the blazingfast workstation?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 6}
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
   ⚪ QueryPlan                      Step 6: retriever=Graph; language=de; query='Was sind die wichtigen Eigenschaften der blazingfast Workstation?'; hits={'BM25': 0, 'Graph': 10, 'Regex': 0}
🔵 Regex                          Querying regex index on collection Test_embedder_bge_m3 for language de
🟢 Regex                          Regex retrieval returned 0 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=Regex; language=de; query='Was sind die wichtigen Eigenschaften der blazingfast Workstation?'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
   ⚪ KeywordActionSplitter          query_id=0 hits: {'vector': 76, 'bm25': 70, 'graph': 14, 'regex': 6}
   ⚪ KeywordActionSplitter          BM25 keyword query: 'important properties workstation specifications'
   ⚪ QueryPlan                      Step 4: document languages=['de', 'en']; active/present=['de', 'en']; metadata limit=whole collection
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=en retrievers=BM25/Graph/Regex shaping=on changed=0
↳                                 primary_query source=REWRITTEN_QUERY lang=en->en
↳                                   original='important properties workstation specifications'
↳                                   staged='important properties workstation specifications'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->en
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->en count=0
🔵 Retrieval Orchestration        idx_stage flow=NATIVE_ALL language=de retrievers=BM25/Graph/Regex shaping=on changed=1
↳                                 primary_query source=REWRITTEN_QUERY lang=en->de
↳                                   original='important properties workstation specifications'
↳                                   staged='Wichtige Eigenschaften Workstation Spezifikationen'
↳                                 guardrail_query source=TRANSLATED_QUERY lang=en->de
↳                                   value=N/A (disabled by flow NATIVE_ALL: guardrail leg is off)
↳                                 alternatives source=REWRITTEN_QUERY lang=en->de count=0
🔵 Retrieval Orchestration        retrievers='BM25' language: en      query: 'important properties workstation specifications'
🔵 Retrieval Orchestration        retrievers='BM25' language: de      query: 'Wichtige Eigenschaften Workstation Spezifikationen'
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language en
🟢 BM25                           BM25 retrieval returned 7 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        7.0391         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                2        6.3437         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                3        5.0746         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                4        4.7873         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                5        4.4512         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                6        4.0059         BM25                BlazingFast_Workstation.md
   ⚪ BM25                                7        2.8734         BM25                Hedgehogs.pdf
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=en; query='important properties workstation specifications'; hits={'BM25': 7, 'Graph': 0, 'Regex': 0}
🔵 BM25                           Querying bm25 index on collection Test_embedder_bge_m3 for language de
🟢 BM25                           BM25 retrieval returned 0 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ QueryPlan                      Step 6: retriever=BM25; language=de; query='Wichtige Eigenschaften Workstation Spezifikationen'; hits={'BM25': 0, 'Graph': 0, 'Regex': 0}
🔵 Retrieval Orchestration        language: en      local docs fetched vector=76 bm25=70 graph=14 regex=6
🔵 Retrieval Orchestration        language: en      web stage skipped (web_mode=local_only)
🔵 Retrieval Orchestration        language: en      merge candidates and build context
🔵 Merge                          Language-scoped RRF candidate pool languages=2 per_language={'en': 66, 'de': 10} interleaved=76/100
   ⚪ Merge                             Pos    RRFScore                            Retrievers  File
   ⚪ Merge                          ----------------------------------------------------------------------------------
   ⚪ Merge                               1      0.0656              Vector BM25 Graph Regex   BlazingFast_Workstation.md
   ⚪ Merge                               2      0.0477              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                               3      0.0638              Vector BM25 Graph Regex   BlazingFast_Workstation.md
   ⚪ Merge                               4      0.0472              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                               5      0.0633              Vector BM25 Graph Regex   BlazingFast_Workstation.md
   ⚪ Merge                               6      0.0470              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                               7      0.0625              Vector BM25 Graph Regex   BlazingFast_Workstation.md
   ⚪ Merge                               8      0.0465              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                               9      0.0464              Vector BM25       Regex   BlazingFast_Workstation.md
   ⚪ Merge                              10      0.0464              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              11      0.0462              Vector BM25       Regex   BlazingFast_Workstation.md
   ⚪ Merge                              12      0.0462              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              13      0.0272              Vector BM25               Lions.pptx
   ⚪ Merge                              14      0.0457              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              15      0.0267              Vector BM25               Cats.md
   ⚪ Merge                              16      0.0457              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              17      0.0265              Vector BM25               Lions.pptx
   ⚪ Merge                              18      0.0437              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              19      0.0259              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              20      0.0429              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              21      0.0259              Vector BM25               Apes.docx
   ⚪ Merge                              22      0.0256              Vector BM25               Apes.docx
   ⚪ Merge                              23      0.0255              Vector BM25               Apes.docx
   ⚪ Merge                              24      0.0253              Vector BM25               Cats.md
   ⚪ Merge                              25      0.0252              Vector BM25               Apes.docx
   ⚪ Merge                              26      0.0252              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              27      0.0244              Vector BM25               Cats.md
   ⚪ Merge                              28      0.0241              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              29      0.0240              Vector BM25               Cats.md
   ⚪ Merge                              30      0.0238              Vector BM25               Lions.pptx
   ⚪ Merge                              31      0.0235              Vector BM25               Apes.docx
   ⚪ Merge                              32      0.0234              Vector BM25               Cats.md
   ⚪ Merge                              33      0.0233              Vector BM25               Lions.pptx
   ⚪ Merge                              34      0.0233              Vector BM25               Fish.txt
   ⚪ Merge                              35      0.0231              Vector BM25               Cats.md
   ⚪ Merge                              36      0.0231              Vector BM25               Cats.md
   ⚪ Merge                              37      0.0230              Vector BM25               Cats.md
   ⚪ Merge                              38      0.0230              Vector BM25               Cats.md
   ⚪ Merge                              39      0.0229              Vector BM25               Cats.md
   ⚪ Merge                              40      0.0229              Vector BM25               Cats.md
   ⚪ Merge                              41      0.0227              Vector BM25               Cats.md
   ⚪ Merge                              42      0.0225              Vector BM25               Fish.txt
   ⚪ Merge                              43      0.0225              Vector BM25               Apes.docx
   ⚪ Merge                              44      0.0217              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              45      0.0209              Vector BM25               Cats.md
   ⚪ Merge                              46      0.0205              Vector BM25               Cats.md
   ⚪ Merge                              47      0.0203              Vector BM25               Cats.md
   ⚪ Merge                              48      0.0201              Vector BM25               Cats.md
   ⚪ Merge                              49      0.0200              Vector BM25               Cats.md
   ⚪ Merge                              50      0.0200              Vector BM25               Fish.txt
   ⚪ Merge                              51      0.0200              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              52      0.0199              Vector BM25               Cats.md
   ⚪ Merge                              53      0.0197              Vector BM25               Cats.md
   ⚪ Merge                              54      0.0197              Vector BM25               Cats.md
   ⚪ Merge                              55      0.0194              Vector BM25               Cats.md
   ⚪ Merge                              56      0.0193              Vector BM25               Cats.md
   ⚪ Merge                              57      0.0193              Vector BM25               Cats.md
   ⚪ Merge                              58      0.0190              Vector BM25               Cats.md
   ⚪ Merge                              59      0.0188              Vector BM25               Cats.md
   ⚪ Merge                              60      0.0188              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              61      0.0187              Vector BM25               Lions.pptx
   ⚪ Merge                              62      0.0185              Vector BM25               Cats.md
   ⚪ Merge                              63      0.0183              Vector BM25               Cats.md
   ⚪ Merge                              64      0.0181              Vector BM25               Cats.md
   ⚪ Merge                              65      0.0177              Vector BM25               Cats.md
   ⚪ Merge                              66      0.0177              Vector BM25               Fish.txt
   ⚪ Merge                              67      0.0173              Vector BM25               Cats.md
   ⚪ Merge                              68      0.0171              Vector BM25               Cats.md
   ⚪ Merge                              69      0.0170              Vector BM25               Cats.md
   ⚪ Merge                              70      0.0168              Vector BM25               Fish.txt
   ⚪ Merge                              71      0.0132              Vector                    Elephants.jpg
   ⚪ Merge                              72      0.0127              Vector                    Dogs.png
   ⚪ Merge                              73      0.0123              Vector                    Dogs.png
   ⚪ Merge                              74      0.0115              Vector                    LionsAndApes.xlsx
   ⚪ Merge                              75      0.0096              Vector                    Dogs.png
   ⚪ Merge                              76      0.0087              Vector                    Fish.txt
🔵 ChunkDedup                     Removed 1 near-duplicate chunk(s) (threshold=0.85, include_web=on, kept 75)
🔵 Rerank                         Pre-rerank pool local=75 web=0 total=75 original_leg=off lang=en added[V/B/G/R]=0/0/0/0
🔵 Rerank                         Rerank query: 'create a table with the important properties of the blazingfast workstation'
🔵 Rerank                         Passing language=en chunks=66 to reranker; query='What are the important properties of the blazingfast workstation?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 66 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=en top5=[1:BlazingFast_Workstation.md (raw=0.9609), 2:BlazingFast_Workstation.md (raw=0.7682), 3:BlazingFast_Workstation.md (raw=0.7367),
↳                                 4:BlazingFast_Workstation.md (raw=0.6828), 5:BlazingFast_Workstation.md (raw=0.6590)]
🔵 Rerank                         Passing language=de chunks=9 to reranker; query='Was sind die wichtigen Eigenschaften der blazingfast Workstation?'
🔵 Call LLM                       Stage: Cross-encode
🔵 Call LLM                       Model: bge-reranker prompt template: None purpose: Scoring 9 (query, passage) pair(s) via http://192.168.100.50:4000/v1/rerank
🔵 Rerank                         language=de top5=[1:Kamele.txt (raw=0.0001), 2:Pferde.pdf (raw=0.0000), 3:Pferde.pdf (raw=0.0000), 4:Pferde.pdf (raw=0.0000), 5:Kamele.txt (raw=0.0000)]
🔵 Rerank summary                 Language           Chunks    Top raw  Mean top3   Mean raw
🔵 Rerank summary                 en                     66     0.9609     0.8219     0.0618
🔵 Rerank summary                 de                      9     0.0001     0.0000     0.0000
🔵 Rerank                         Selected language=en by highest top raw reranker score=0.9609; top chunk='BlazingFast_Workstation.md'; passing 66 matching-language local candidates + 0 web
↳                                 candidates to chunk selection.
   ⚪ Rerank                            Pos    RawScore    AdjScore                Retrievers  File                                      Text
   ⚪ Rerank                         --------------------------------------------------------------------------------------------------------------
   ⚪ Rerank                              1      0.9609      1.0000  Vector BM25 Graph Regex   BlazingFast_Workstation.md                the blazingfast workstation is an enterp
   ⚪ Rerank                              2      0.7682      0.7994  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
   ⚪ Rerank                              3      0.7367      0.7666  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
   ⚪ Rerank                              4      0.6828      0.7106  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **processors**: dual enterprise platin
   ⚪ Rerank                              5      0.6590      0.6858  Vector BM25       Regex   BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
   ⚪ Rerank                              6      0.2692      0.2801  Vector BM25       Regex   BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
   ⚪ Rerank                              7      0.0000      0.0000  Vector                    Dogs.png                                  tasks. understanding breed-specific ende
   ⚪ Rerank                              8      0.0000      0.0000  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
   ⚪ Rerank                              9      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs a
   ⚪ Rerank                             10      0.0000      0.0000  Vector                    Dogs.png                                  , comprehensive text dogs dogs accompani
   ⚪ Rerank                             11      0.0000      0.0000  Vector                    Elephants.jpg                             elephants large, long-lived mammals fami
   ⚪ Rerank                             12      0.0000      0.0000  Vector BM25               Cats.md                                   - climbing structures - scratching posts
   ⚪ Rerank                             13      0.0000      0.0000  Vector BM25               Cats.md                                   - puzzle toys - training sessions - nove
   ⚪ Rerank                             14      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations
   ⚪ Rerank                             15      0.0000      0.0000  Vector BM25               Fish.txt                                  ecological roles and importance fish are
   ⚪ Rerank                             16      0.0000      0.0000  Vector                    Dogs.png                                  impact dogs human well-being. despite ma
   ⚪ Rerank                             17      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—suc
   ⚪ Rerank                             18      0.0000      0.0000  Vector BM25               Cats.md                                   as placental mammals, cats give live bir
   ⚪ Rerank                             19      0.0000      0.0000  Vector BM25               Cats.md                                   cats are complex, adaptable, and endless
   ⚪ Rerank                             20      0.0000      0.0000  Vector BM25               Fish.txt                                   what fish are fish are a diverse group
   ⚪ Rerank                             21      0.0000      0.0000  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
   ⚪ Rerank                             22      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, deci
   ⚪ Rerank                             23      0.0000      0.0000  Vector BM25               Cats.md                                   cats are crepuscular hunters with: - a h
   ⚪ Rerank                             24      0.0000      0.0000  Vector BM25               Fish.txt                                  diversity: fish include jawless fishes (
   ⚪ Rerank                             25      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 suburban gardens and agricult
   ⚪ Rerank                             26      0.0000      0.0000  Vector BM25               Cats.md                                   - kingdom: animalia - phylum: chordata -
   ⚪ Rerank                             27      0.0000      0.0000  Vector BM25               Cats.md                                   they have a righting reflex, but falls f
   ⚪ Rerank                             28      0.0000      0.0000  Vector BM25               Cats.md                                   - free feeding can lead to obesity. - pu
   ⚪ Rerank                             29      0.0000      0.0000  Vector BM25               Cats.md                                   outdoor access: - pros: exercise, stimul
   ⚪ Rerank                             30      0.0000      0.0000  Vector BM25               Fish.txt                                  terminology explained cold-blooded: a co
   ⚪ Rerank                             31      0.0000      0.0000  Vector BM25               Cats.md                                   cats are often described as solitary, bu
   ⚪ Rerank                             32      0.0000      0.0000  Vector BM25               Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
   ⚪ Rerank                             33      0.0000      0.0000  Vector BM25               Cats.md                                   whiskers are deeply rooted sensory hairs
   ⚪ Rerank                             34      0.0000      0.0000  Vector BM25               Cats.md                                   cats have lived alongside humans for tho
   ⚪ Rerank                             35      0.0000      0.0000  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
   ⚪ Rerank                             36      0.0000      0.0000  Vector BM25               Fish.txt                                  ectotherm: the preferred scientific term
   ⚪ Rerank                             37      0.0000      0.0000  Vector BM25               Cats.md                                   most adult cats are lactose intolerant.
   ⚪ Rerank                             38      0.0000      0.0000  Vector BM25               Cats.md                                   cats were revered, often associated with
   ⚪ Rerank                             39      0.0000      0.0000  Vector BM25               Apes.docx                                 Great apes (Hominidae) are the closest l
   ⚪ Rerank                             40      0.0000      0.0000  Vector                    Fish.txt                                  poikilotherm: a term describing organism
   ⚪ Rerank                             41      0.0000      0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernati
   ⚪ Rerank                             42      0.0000      0.0000  Vector BM25               Cats.md                                   cats dominate digital culture: - memes (
   ⚪ Rerank                             43      0.0000      0.0000  Vector BM25               Cats.md                                   cats require: - taurine - arachidonic ac
   ⚪ Rerank                             44      0.0000      0.0000  Vector BM25               Apes.docx                                 Orangutans are the only great apes found
   ⚪ Rerank                             45      0.0000      0.0000  Vector BM25               Cats.md                                   - human interaction - multi-cat househol
   ⚪ Rerank                             46      0.0000      0.0000  Vector BM25               Apes.docx                                 Gorillas are the largest living primates
   ⚪ Rerank                             47      0.0000      0.0000  Vector BM25               Apes.docx                                 Chimpanzees (Pan troglodytes) are found
   ⚪ Rerank                             48      0.0000      0.0000  Vector BM25               Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
   ⚪ Rerank                             49      0.0000      0.0000  Vector BM25               Lions.pptx                                Slide 4: Conservation Status Lions are c
   ⚪ Rerank                             50      0.0000      0.0000  Vector BM25               Cats.md                                   even well-fed cats hunt. the sequence: 1
🔵 Rerank                         Reranking with bge-reranker returned 66 chunks
🔵 Chunk selection                Strategy 'DEFAULT' → ScoreRankedSelector
🔵 Score select                   logit+ = effective logit after single-chunk boost (raw in raw_rerank_score)
🟣 Score select                         Logit [Sigmoid]       Thr      ΔProb  Retrievers                File                                      Text
🟣 Score select                   --------------------------------------------------------------------------------------------------------------------
🟣 Score select                    ✅    0.9609 [0.723]  (0.5000)    +0.2233  Vector BM25 Graph Regex   BlazingFast_Workstation.md                the blazingfast workstation is an enterp
🟣 Score select                    ✅    0.7682 [0.683]  (0.5000)    +0.1831  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
🟣 Score select                    ✅    0.7367 [0.676]  (0.5000)    +0.1763  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
🟣 Score select                    ✅    0.6828 [0.664]  (0.5000)    +0.1644  Vector BM25 Graph Regex   BlazingFast_Workstation.md                - **processors**: dual enterprise platin
🟣 Score select                    ✅    0.6590 [0.659]  (0.5000)    +0.1590  Vector BM25       Regex   BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
🟣 Score select                    ✅    0.2692 [0.567]  (0.5000)    +0.0669  Vector BM25       Regex   BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
🟣 Score select                   ⬆✅    0.2232+[0.556]  (0.5000)    +0.0556  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
🟣 Score select                   ⬆✅    0.2232+[0.556]  (0.5000)    +0.0556  Vector                    Elephants.jpg                             elephants large, long-lived mammals fami
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    Dogs.png                                  tasks. understanding breed-specific ende
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs a
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    Dogs.png                                  , comprehensive text dogs dogs accompani
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - climbing structures - scratching posts
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - puzzle toys - training sessions - nove
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Fish.txt                                  ecological roles and importance fish are
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    Dogs.png                                  impact dogs human well-being. despite ma
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—suc
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   as placental mammals, cats give live bir
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats are complex, adaptable, and endless
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Fish.txt                                   what fish are fish are a diverse group
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, deci
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats are crepuscular hunters with: - a h
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Fish.txt                                  diversity: fish include jawless fishes (
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 1 suburban gardens and agricult
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - kingdom: animalia - phylum: chordata -
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   they have a righting reflex, but falls f
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - free feeding can lead to obesity. - pu
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   outdoor access: - pros: exercise, stimul
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Fish.txt                                  terminology explained cold-blooded: a co
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats are often described as solitary, bu
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   whiskers are deeply rooted sensory hairs
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats have lived alongside humans for tho
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Lions.pptx                                Slide 1: Lions: The King of the Savanna
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Fish.txt                                  ectotherm: the preferred scientific term
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   most adult cats are lactose intolerant.
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats were revered, often associated with
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 Great apes (Hominidae) are the closest l
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector                    Fish.txt                                  poikilotherm: a term describing organism
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernati
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats dominate digital culture: - memes (
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats require: - taurine - arachidonic ac
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 Orangutans are the only great apes found
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - human interaction - multi-cat househol
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 Gorillas are the largest living primates
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 Chimpanzees (Pan troglodytes) are found
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Lions.pptx                                Slide 4: Conservation Status Lions are c
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   even well-fed cats hunt. the sequence: 1
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   these breeds emerged without heavy human
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats can detect frequencies up to ~64 kh
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   the mechanism of purring is still debate
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats communicate through: - vocalization
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 All great ape species are either endange
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats suffered during periods of supersti
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   spaying/neutering is essential to reduce
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - regular veterinary checkups - vaccinat
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats were not bred for specific tasks ea
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   cats form strong bonds but express affec
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Apes.docx                                 All great apes demonstrate remarkable co
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   a deep, structured exploration of domest
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   indoor cats often live 12–18 years; some
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   free-roaming cats are estimated to kill
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   - dental disease - kidney disease - hype
🟣 Score select                    ✅    0.0000 [0.500]  (0.5000)    +0.0000  Vector BM25               Cats.md                                   kittens engage in: - stalking. - pouncin
🔵 Strategy: default              66 chunks selected after thresholding  0.5000  single-chunk boost ×1.25
🔵 Chunk selector: Score          Ranked     Selected 50 chunks.
🔵 Selected                       50 chunks selected
   ⚪ Selected                        #   Score [Sigmoid]  File                                      Text
   ⚪ Selected                       --------------------------------------------------------------------------------------------------------------------------------------------
   ⚪ Selected                        1    1.0000 [0.723]  BlazingFast_Workstation.md                the blazingfast workstation is an enterprise-grade computing system en
   ⚪ Selected                        2    0.7994 [0.683]  BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-datacenter rtx 6000 ada genera
   ⚪ Selected                        3    0.7666 [0.676]  BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2 pcie gen 5 ssds, configured b
   ⚪ Selected                        4    0.7106 [0.664]  BlazingFast_Workstation.md                - **processors**: dual enterprise platinum 8500-series processors. pro
   ⚪ Selected                        5    0.6858 [0.659]  BlazingFast_Workstation.md                - **power delivery**: dual 2000w redundant titanium-rated power suppli
   ⚪ Selected                        6    0.2801 [0.567]  BlazingFast_Workstation.md                - **os support**: certified for ubuntu 22.04 lts and windows 11 pro fo
   ⚪ Selected                        7    0.0000 [0.500]  Dogs.png                                  tasks. understanding breed-specific endencies essential training, welf
   ⚪ Selected                        8    0.0000 [0.500]  LionsAndApes.xlsx                         category detail value taxonomy scientific name panthera leo taxonomy f
   ⚪ Selected                        9    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 1 Hedgehog Overview Hedgehogs are small, nocturnal mammals kn
   ⚪ Selected                       10    0.0000 [0.500]  Dogs.png                                  , comprehensive text dogs dogs accompanied humans tens thousands years
   ⚪ Selected                       11    0.0000 [0.500]  Elephants.jpg                             elephants large, long-lived mammals family elephantidae, represented t
   ⚪ Selected                       12    0.0000 [0.500]  Cats.md                                   - climbing structures - scratching posts - window perches [section: th
   ⚪ Selected                       13    0.0000 [0.500]  Cats.md                                   - puzzle toys - training sessions - novel objects [section: the compre
   ⚪ Selected                       14    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 1 tissues. Sensory adaptations include a keen sense of smell
   ⚪ Selected                       15    0.0000 [0.500]  Fish.txt                                  ecological roles and importance fish are central to aquatic food webs:
   ⚪ Selected                       16    0.0000 [0.500]  Dogs.png                                  impact dogs human well-being. despite many strengths, dogs also face c
   ⚪ Selected                       17    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 2 areas, and garden hazards—such as netting, open drains, and
   ⚪ Selected                       18    0.0000 [0.500]  Cats.md                                   as placental mammals, cats give live birth and nurse their young with
   ⚪ Selected                       19    0.0000 [0.500]  Cats.md                                   cats are complex, adaptable, and endlessly fascinating creatures. thei
   ⚪ Selected                       20    0.0000 [0.500]  Fish.txt                                   what fish are fish are a diverse group of aquatic vertebrates that li
   ⚪ Selected                       21    0.0000 [0.500]  Lions.pptx                                Slide 2: Physical Characteristics Male lions weigh between 150-250 kg
   ⚪ Selected                       22    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 2 detect prey and a rapid, decisive bite to subdue it. Season
   ⚪ Selected                       23    0.0000 [0.500]  Cats.md                                   cats are crepuscular hunters with: - a high density of rod photorecept
   ⚪ Selected                       24    0.0000 [0.500]  Fish.txt                                  diversity: fish include jawless fishes (lampreys, hagfish), cartilagin
   ⚪ Selected                       25    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 1 suburban gardens and agricultural edges. They are native to
   ⚪ Selected                       26    0.0000 [0.500]  Cats.md                                   - kingdom: animalia - phylum: chordata - class: mammalia (mammals) - o
   ⚪ Selected                       27    0.0000 [0.500]  Cats.md                                   they have a righting reflex, but falls from low heights can be dangero
   ⚪ Selected                       28    0.0000 [0.500]  Cats.md                                   - free feeding can lead to obesity. - puzzle feeders encourage natural
   ⚪ Selected                       29    0.0000 [0.500]  Cats.md                                   outdoor access: - pros: exercise, stimulation - cons: predation, traff
   ⚪ Selected                       30    0.0000 [0.500]  Fish.txt                                  terminology explained cold-blooded: a common everyday phrase meaning a
   ⚪ Selected                       31    0.0000 [0.500]  Cats.md                                   cats are often described as solitary, but this is an oversimplificatio
   ⚪ Selected                       32    0.0000 [0.500]  Lions.pptx                                Slide 3: Hunting and Diet Lions are cooperative hunters, working toget
   ⚪ Selected                       33    0.0000 [0.500]  Cats.md                                   whiskers are deeply rooted sensory hairs that detect: - airflow change
   ⚪ Selected                       34    0.0000 [0.500]  Cats.md                                   cats have lived alongside humans for thousands of years, yet they rema
   ⚪ Selected                       35    0.0000 [0.500]  Lions.pptx                                Slide 1: Lions: The King of the Savanna Lions (Panthera leo) are the o
   ⚪ Selected                       36    0.0000 [0.500]  Fish.txt                                  ectotherm: the preferred scientific term for animals that rely mainly
   ⚪ Selected                       37    0.0000 [0.500]  Cats.md                                   most adult cats are lactose intolerant. [section: the comprehensive fe
   ⚪ Selected                       38    0.0000 [0.500]  Cats.md                                   cats were revered, often associated with: - bastet (goddess of protect
   ⚪ Selected                       39    0.0000 [0.500]  Apes.docx                                 Great apes (Hominidae) are the closest living relatives to humans, sha
   ⚪ Selected                       40    0.0000 [0.500]  Fish.txt                                  poikilotherm: a term describing organisms whose internal temperature v
   ⚪ Selected                       41    0.0000 [0.500]  Hedgehogs.pdf                             PDF page 2 behavioral ecology, hibernation physiology, and the impacts
   ⚪ Selected                       42    0.0000 [0.500]  Cats.md                                   cats dominate digital culture: - memes (“i can has cheezburger?”) - vi
   ⚪ Selected                       43    0.0000 [0.500]  Cats.md                                   cats require: - taurine - arachidonic acid - preformed vitamin a - hig
   ⚪ Selected                       44    0.0000 [0.500]  Apes.docx                                 Orangutans are the only great apes found in Asia, inhabiting the rainf
   ⚪ Selected                       45    0.0000 [0.500]  Cats.md                                   - human interaction - multi-cat households (when compatible) [section:
   ⚪ Selected                       46    0.0000 [0.500]  Apes.docx                                 Gorillas are the largest living primates, with males weighing up to 20
   ⚪ Selected                       47    0.0000 [0.500]  Apes.docx                                 Chimpanzees (Pan troglodytes) are found across equatorial Africa. They
   ⚪ Selected                       48    0.0000 [0.500]  Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gestation lasts about 110 days, pr
   ⚪ Selected                       49    0.0000 [0.500]  Lions.pptx                                Slide 4: Conservation Status Lions are classified as Vulnerable by the
   ⚪ Selected                       50    0.0000 [0.500]  Cats.md                                   even well-fed cats hunt. the sequence: 1. detect 2. stalk 3. chase 4.
   ⚪ KeywordActionSplitter          Final context query_id=0: 50 associated chunks for 'What are the important properties of the blazingfast workstation?' (association is not proof of answer support)
   ⚪ ChunkSelect                    After chunk selection: 50/66 kept
🔵 Selection Confidence           🟡 MEDIUM C_final=0.77 evidence_chunks=50 rerank_fallback=off
🔵 Retrieval Orchestration        language: en      retrieval completed chosen=50 context_chars=43202
🔵 KeywordActionSplitter          Final answer instruction: Complete user request:
↳                                 create a table with the important properties of the blazingfast workstation
↳
↳                                 Self-contained answer instruction:
↳                                 Using only the retrieved context, create a table listing the important properties of the blazingfast workstation.
↳
↳                                 Requested operation to execute on the retrieved evidence:
↳                                 Create a table of the important properties of the blazingfast workstation.
↳                                 Perform the operation, not a description of how to perform it. For a requested list, use Markdown bullets; for a requested table, use a Markdown table, unless the
↳                                 user explicitly specifies another format.
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] context=32768 reserved_sys=1024 prompt≈8470 → max_output_tokens=2048
🔵 Resolved token params          max_output_tokens(api: max_tokens)=2048  num_ctx=32768  (override: max_tokens=None  num_ctx=None)
🔵 LLM Plan                       Answer generation: producing the final grounded response from retrieved context chunks.
🔵 TokenBudget                    [openai/qwen2_5_7b_instruct] num_ctx=32768 prompt≈8470 → num_predict=2048
🔵 Call LLM                       Stage: Run user prompt
🔵 Call LLM                       Model: openai/qwen2_5_7b_instruct prompt template: _PROMPT_CHAT purpose: Inference
🔵 Call LLM                       options: {'temperature': 0.1, 'top_k': 40, 'top_p': 0.92, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_7b_instruct took 00:50
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_compliance_language): HIGH (confidence: 100%, threshold: 60%)
🔵 vLLM                           Using vLLM embeddings 'bge-m3-embedder' at http://192.168.100.50:4000/v1
🟢 Cache build Regex Banned       Built compiled Regex cache with 80 entries for language english stage: PIPELINE_CHECK
🟢 KeyWrdChk Summary              Phrase                         dpt / brth     Regex+Levenshtein ➡️    Jaccard ➡️              BM25 ➡️                 Keybert
🟡 KeyWrdChk Summary              account balance                1/4    1/4      -/-                    -/-                    -/-                    0.4264/0.4000
🟡 KeyWrdChk Summary              account number                 1/4    1/4      -/-                    -/-                    -/-                    0.4193/0.4000
🟡 KeyWrdChk Summary              auth token                     ******    1/4      -/-                    -/-                    -/-                    0.4205/0.4000
🟡 KeyWrdChk Summary              backdoor                       1/4    1/4      -/-                    -/-                    -/-                    0.4117/0.4000
🟡 KeyWrdChk Summary              bank account                   1/4    1/4      -/-                    -/-                    -/-                    0.4148/0.4000
🟡 KeyWrdChk Summary              buffer overflow                1/4    1/4      -/-                    -/-                    -/-                    0.4769/0.4000
🟡 KeyWrdChk Summary              credit card number             1/4    1/4      -/-                    -/-                    -/-                    0.4322/0.4000
🟡 KeyWrdChk Summary              credit score                   1/4    1/4      -/-                    -/-                    -/-                    0.4291/0.4000
🟡 KeyWrdChk Summary              data controller                1/4    1/4      -/-                    -/-                    -/-                    0.4508/0.4000
🟡 KeyWrdChk Summary              data processor                 1/4    1/4      -/-                    -/-                    -/-                    0.5379/0.4000
🟡 KeyWrdChk Summary              data subject                   1/4    1/4      -/-                    -/-                    -/-                    0.4200/0.4000
🟡 KeyWrdChk Summary              diagnosis                      1/4    1/4      -/-                    -/-                    -/-                    0.4233/0.4000
🟡 KeyWrdChk Summary              explosive                      1/4    1/4      -/-                    -/-                    -/-                    0.4485/0.4000
🟡 KeyWrdChk Summary              id card number                 1/4    1/4      -/-                    -/-                    -/-                    0.4555/0.4000
🟡 KeyWrdChk Summary              id number                      1/4    1/4      -/-                    -/-                    -/-                    0.4380/0.4000
🟡 KeyWrdChk Summary              loan decision                  1/4    1/4      -/-                    -/-                    -/-                    0.4134/0.4000
🟡 KeyWrdChk Summary              malware                        1/4    1/4      -/-                    -/-                    -/-                    0.4308/0.4000
🟡 KeyWrdChk Summary              routing number                 1/4    1/4      -/-                    -/-                    -/-                    0.4432/0.4000
🟡 KeyWrdChk Summary              salary amount                  1/4    1/4      -/-                    -/-                    -/-                    0.4342/0.4000
🟡 KeyWrdChk Summary              security question              1/4    1/4      -/-                    -/-                    -/-                    0.4526/0.4000
🟢 KeyWrdChk Depth                1 algos passed threshold vs. required 4
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 4
   ⚪ Grounding                      Inference answer before grounding validation:
   ↳                                 | Property                          | Description                                                                 |
   ↳                                 |-----------------------------------|------------------------------------------------------------------------------|
   ↳                                 | **Graphics**                      | Accommodates up to quad-datacenter RTX 6000 Ada generation GPUs with 48GB VRAM per GPU (192GB total). Supports distributed
   ↳                                 inference, fine-tuning of 70B+ parameter language models, and hardware-accelerated document classification. |
   ↳                                 | **Tensor Cores**                  | Embedded fourth-generation tensor cores drastically reduce processing times for matrix multiplication, a core requirement for
   ↳                                 embedding generation.                                               |
   ↳                                 | **Storage Subsystem**             | 8x 4TB NVMe U.2 PCIe Gen 5 SSDs configured in RAID 10 for 16TB of redundant, ultra-fast local storage. Delivers sequential
   ↳                                 read speeds up to 28,000 MB/s and extremely low latency for database lookups. |
   ↳                                 | **Networking**                    | Integrated dual 100Gbe SFP28 network interfaces for high-throughput cluster environments, alongside an independent
   ↳                                 out-of-band management interface (IPMI).                                      |
   ↳                                 | **Expansion**                     | 6x PCIe Gen 5 x16 slots, providing sufficient lanes for multiple accelerator cards, dedicated hardware RAID controllers, or
   ↳                                 additional network interface cards (NICs).                       |
   ↳                                 | **Processors**                    | Dual enterprise platinum 8500-series processors providing up to 120 physical cores (240 threads) per system. Base clock speed
   ↳                                 is 2.8 GHz, with single-core turbo frequencies up to 4.6 GHz.           |
   ↳                                 | **Memory**                        | Supports up to 2TB of ECC DDR5 RAM running at 4800 MHz across 16 DIMM slots. Optimized for loading large vector databases and
   ↳                                 LLM context windows entirely into RAM.                           |
   ↳                                 | **Power Delivery**                | Dual 2000W redundant titanium-rated power supplies (N+1 configuration). Requires a 200V-240V circuit due to the massive power
   ↳                                 draw at full compute load.                                 |
   ↳                                 | **Thermals**                      | A specialized closed-loop liquid cooling system ensures both GPUs and CPUs remain below 70°C even under sustained 24/7
   ↳                                 workloads, preventing thermal throttling.                             |
   ↳                                 | **OS Support**                    | Certified for Ubuntu 22.04 LTS and Windows 11 Pro for Workstations.                                                      |
   ↳                                 | **Pre-installed Stack**           | Ships optionally with a fully configured AI stack, including CUDA 12, PyTorch, Docker, and pre-configured Python environments
   ↳                                 tuned for RAG (retrieval-augmented generation) pipeline development. |
   ↳
   ↳                                 ### Sources
   ↳                                 - BlazingFast_Workstation.md
   ↳                                   - FilePath: D:/RAG-LCC/TestDocs/BlazingFast_Workstation.md
   ↳                                   - FileModified: 2026-06-23 14:17:15.094329
   ↳                                   - FileName: BlazingFast_Workstation.md
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_grounding_language): HIGH (confidence: 100%, threshold: 60%)
🔵 Deleted chat context           Deleted chat context for MyFirstChat in Test_embedder_bge_m3_ChatContext
🔵 Add chat context               Upserted turn 1 for chat_name=MyFirstChat file_tag='' lang='english' to Test_embedder_bge_m3_ChatContext
   ⚪ Retrieval Orchestration        Stored query history forms orig='create a table with the important properties of the blazingfast workstation' t1='create a table with the important properties of
   ↳                                 the blazingfast workstation' t2='create a table with the important properties of the blazingfast workstation' effective_reason=''
🔵 TokenBudget                    /v1/models did not yield max_model_len for 'openai/qwen2_5_14b_instruct'; retrying via LiteLLM /model/info endpoint
🔵 TokenBudget                    Detected context_length=8192 for openai/qwen2_5_14b_instruct via vllm adapter
🔵 TokenBudget                    Model 'openai/qwen2_5_14b_instruct': context_limit=8192 (below cap 32768 — using detected value)
🔵 TokenBudget                    Caller did not supply context_size; resolved num_ctx=8192 for openai/qwen2_5_14b_instruct
🔵 TokenBudget                    Authoritative budget=256 vs caller estimate=0; using authoritative value
🔵 TokenBudget                    [openai/qwen2_5_14b_instruct] num_ctx=8192 prompt≈640 → num_predict=256
🔵 Call LLM                       Stage: Answer-text LLM review
🔵 Call LLM                       Model: openai/qwen2_5_14b_instruct prompt template: None purpose: Judge
🔵 Call LLM                       options: {'temperature': 0.0, 'num_predict': 256, 'num_ctx': 8192} streaming: False
🔵 Call LLM                       Elapsed time calling: openai/qwen2_5_14b_instruct took 00:01
🟡 VisualMarker                   _mark_sources called: 50 chunk(s)
🔵 VisualMarker                   Prepared 1 highlighted document(s) (in memory)
🔵 Marked sources                 1 highlighted document(s) (click a link to open; use 'Save As' in the viewer to keep a copy):
   📎 BlazingFast_Workstation.md (highlighted): file:///D:/RAG-LCC/tmp/rag_marked_k32dzhns/BlazingFast_Workstation_marked.md
Original query: create a table with the important properties of the blazingfast workstation
💬 >  | Property                          | Description                                                                 |
💬 >  |-----------------------------------|------------------------------------------------------------------------------|
💬 >  | **Graphics**                      | Accommodates up to quad-datacenter RTX 6000 Ada generation GPUs with 48GB VRAM per GPU (192GB total). Supports distributed inference, fine-tuning of 70B+
💬 >  parameter language models, and hardware-accelerated document classification. |
💬 >  | **Tensor Cores**                  | Embedded fourth-generation tensor cores drastically reduce processing times for matrix multiplication, a core requirement for embedding generation. |
💬 >  | **Storage Subsystem**             | 8x 4TB NVMe U.2 PCIe Gen 5 SSDs configured in RAID 10 for 16TB of redundant, ultra-fast local storage. Delivers sequential read speeds up to 28,000 MB/s and
💬 >  extremely low latency for database lookups. |
💬 >  | **Networking**                    | Integrated dual 100Gbe SFP28 network interfaces for high-throughput cluster environments, alongside an independent out-of-band management interface (IPMI).
💬 >  |
💬 >  | **Expansion**                     | 6x PCIe Gen 5 x16 slots, providing sufficient lanes for multiple accelerator cards, dedicated hardware RAID controllers, or additional network interface
💬 >  cards (NICs). |
💬 >  | **Processors**                    | Dual enterprise platinum 8500-series processors providing up to 120 physical cores (240 threads) per system. Base clock speed is 2.8 GHz, with single-core
💬 >  turbo frequencies up to 4.6 GHz. |
💬 >  | **Memory**                        | Supports up to 2TB of ECC DDR5 RAM running at 4800 MHz across 16 DIMM slots. Optimized for loading large vector databases and LLM context windows entirely
💬 >  into RAM. |
💬 >  | **Power Delivery**                | Dual 2000W redundant titanium-rated power supplies (N+1 configuration). Requires a 200V-240V circuit due to the massive power draw at full compute load. |
💬 >  | **Thermals**                      | A specialized closed-loop liquid cooling system ensures both GPUs and CPUs remain below 70°C even under sustained 24/7 workloads, preventing thermal
💬 >  throttling. |
💬 >  | **OS Support**                    | Certified for Ubuntu 22.04 LTS and Windows 11 Pro for Workstations. |
💬 >  | **Pre-installed Stack**           | Ships optionally with a fully configured AI stack, including CUDA 12, PyTorch, Docker, and pre-configured Python environments tuned for RAG
💬 >  (retrieval-augmented generation) pipeline development. |
💬 >
💬 >  ### Sources
💬 >  • BlazingFast_Workstation.md
💬 >    • FilePath: D:/RAG-LCC/TestDocs/BlazingFast_Workstation.md ([open](file:///D:/RAG-LCC/TestDocs/BlazingFast_Workstation.md))
💬 >    • FileModified: 2026-06-23 14:17:15.094329
💬 >    • FileName: BlazingFast_Workstation.md
💬 >
💬 >  ---
💬 >
💬 >  **Answer confidence:**
💬 >  🟡 MEDIUM
💬 >
💬 >  Heuristic Confidence:
💬 >  • C_final=0.77            : final confidence in [0,1] after penalties.
💬 >  • C_top=0.36              : weighted best-chunk signal (0.50 x top_score).
💬 >  • C_mean_top3=0.21        : weighted top-3 stability (0.30 x mean_top3).
💬 >  • C_coverage=0.15         : weighted evidence coverage (0.15 x coverage).
💬 >  • C_local=0.05            : weighted local-source share (0.05 x local_share).
💬 >  • C_fallback_penalty=0.00 : subtraction when low-confidence rerank fallback was used.
💬 >  • top_score=0.72          : strongest selected chunk confidence.
💬 >  • mean_top3=0.69          : average confidence of the top 3 selected chunks.
💬 >  • coverage=1.00           : selected_chunks / final_chunks_to_llm, capped at 1.00.
💬 >  • local_share=1.00        : fraction of selected chunks from local files.
💬 >  • evidence_chunks=50      : number of selected evidence chunks.
💬 >  • rerank_fallback=off     : whether fallback rerank logic was triggered.
💬 >
💬 >  Static retrieval diagnostics (not answer-quality verification):
💬 >  • Evidence relevance   : heuristic=0.74
💬 >    Top retrieval scores and fallback status; does not verify answer accuracy.
💬 >  • Retrieval coverage   : heuristic=1.00
💬 >    Selected chunk count relative to the target, plus evidence quantity; does not measure answer completeness.
💬 >  • Evidence availability: heuristic=0.80
💬 >    Retrieval relevance and chunk availability; does not verify citations or claim-to-source support.
💬 >
💬 >  LLM review - answer text only (judge model: openai/qwen2_5_14b_instruct):
💬 >  • Internal consistency: llm=1.00
💬 >    Judge rating of contradictions within the answer; does not verify factual correctness.
💬 >  • Explanation clarity : not assessed
💬 >    Judge rating of how understandable the explanation is; does not verify reasoning validity.
💬 >  Subjective ratings, not probabilities. The judge sees neither the question nor source evidence and may share the generator's blind spots.
💬 >  Accuracy, answer completeness, reasoning validity, and citation support: not assessed.


---
### Document metadata

- **BlazingFast_Workstation.md**
    - FilePath: D:/RAG-LCC/TestDocs/BlazingFast_Workstation.md
    - FileSizeBytes: 2610
    - FileModified: 2026-06-23 14:17:15.094329
- **Dogs.png**
    - FilePath: D:/RAG-LCC/TestDocs/Dogs.png
    - FileSizeBytes: 634064
    - FileModified: 2026-06-23 14:17:15.110956
- **LionsAndApes.xlsx**
    - FilePath: D:/RAG-LCC/TestDocs/LionsAndApes.xlsx
    - Author: openpyxl
    - DocCreated: 2026-04-15 12:24:52
    - DocModified: 2026-04-15 12:24:52
- **Hedgehogs.pdf**
    - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf
    - Creator: Microsoft® Word for Microsoft 365
    - Producer: Microsoft® Word for Microsoft 365
    - DocCreated: 2026-02-17 12:32:04+01:00
    - DocModified: 2026-02-17 12:32:04+01:00
    - Pages: 1, 2
- **Elephants.jpg**
    - FilePath: D:/RAG-LCC/TestDocs/Elephants.jpg
    - FileSizeBytes: 282341
    - FileModified: 2026-06-23 14:17:15.110956
- **Cats.md**
    - FilePath: D:/RAG-LCC/TestDocs/Cats.md
    - FileSizeBytes: 7323
    - FileModified: 2026-06-23 14:17:15.094329
- **Fish.txt**
    - FilePath: D:/RAG-LCC/TestDocs/Fish.txt
    - FileSizeBytes: 3474
    - FileModified: 2026-06-23 14:17:15.094329
- **Lions.pptx**
    - FilePath: D:/RAG-LCC/TestDocs/Lions.pptx
    - DocCreated: 2013-01-27 09:14:16
    - DocModified: 2013-01-27 09:15:58
    - LastModifiedBy: Steve Canny
- **Apes.docx**
    - FilePath: D:/RAG-LCC/TestDocs/Apes.docx
    - Author: python-docx
    - DocCreated: 2013-12-23 23:15:00+00:00
    - DocModified: 2013-12-23 23:15:00+00:00
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
type new: your question to start a new topic  · type summarize: [file name or path] (alias: summary:) to summarize a file (no argument: pinned document; no pin: open the path picker)  · b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat
💬 Your actual query>
```
