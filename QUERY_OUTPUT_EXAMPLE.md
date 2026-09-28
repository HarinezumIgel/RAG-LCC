# Query Output Example

``` Text
🔵 Config path                    Checking configured filesystem path slots (root guard + presence check).
🔵 Config path                    For file-like slots, present means the containing directory exists.
🔵 Config path                    DOC_DIR -> D:\RAG-LCC\TestDocs present: YES
🔵 Config path                    LOG_FILE -> D:\RAG-LCC\compliance.log present (containing dir): YES
🔵 Config path                    _ABSOLUTE_PATH -> D:\RAG-LCC present: YES
🔵 Config path                    _CUSTOM_NLTK_DATA_DIRECTORY -> D:\RAG-LCC\AppData\Roaming\nltk_data\corpora\stopwords present: YES
🔵 Config path                    _EXCLUSIONS_DIR -> D:\RAG-LCC\Exclusions present: YES
🔵 Config path                    _HF_HOME -> C:\Users\YourUser\.cache present: YES
🔵 Config path                    _HF_HUB_CACHE -> C:\Users\YourUser\.cache\.hf-cache present: YES
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
🦔            Version v0.5.2.0/1516 2026-09-28                          🦔
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
🟢 HF cache                       HF home: C:\Users\YourUser\.cache HF hub cache: C:\Users\YourUser\.cache\.hf-cache

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
🔵 Tokenizer Resources            Default tokenizer resources directory: C:\Users\YourUser\.local\share\resources (directory not present yet)
🟢 spaCy License                  Consent valid for 5 configured model(s)
🟢 HF Download                    _MODELS._EMBED: model already downloaded with matching revision and config hash. Skipping download.
🟢 HF Download                    _MODELS._CROSS: model already downloaded with matching revision and config hash. Skipping download.
   ⚪ WordNet Synonyms               Expanded 81 phrases → 139 (+60 synonyms, depth=1, max/phrase=1)
🟢 BM25Scorer                     Initialized BM25 Scorer (algo=BM25) with 139 base phrase(s)
🔵 HF                             Try to load snowflake/snowflake-arctic-embed-l-v2.0 revision 'None' key: snowflake/snowflake-arctic-embed-l-v2.0_none_cuda:0_torch.float32 from cache.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 8248.45it/s]
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🔵 TokenBudget                    Detected context_length=32768 for mistral:7b via ollama adapter
🔵 TokenBudget                    Model 'mistral:7b': context_limit=32768 (below cap 32768 — using detected value)
🔵 HF                             Try to load cross-encoder 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' revision 'None' device=cuda:0 from cache.
🟢 HF Download                    _MODELS._CROSS: model already downloaded with matching revision and config hash. Skipping download.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 5479.79it/s]
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🟢 Regex                          Aux-lemma cache built for 5 active language(s)
→ Loaded defaults for 'DEFAULT':
  ▶ Debug:        debug_level=30  debug_mode='ge'
  ▶ Chat Context: use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
  ▶ Talk with:    collection='Test'  chat_name='MyFirstChat'
  ▶ File Input:   file=None  path=None  metadata=None  file_cap=15
  ▶ Visual:       mark_text=True
  ▶ Retrieval:
    ▶ Strategies:   strategy='DEFAULT'  orchestrator_flow=None  retrieve_mode='ALL'  rerank=True  threshold=0.5
    ▶ Weights:      vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0
    ▶ Web:          web_search='local_only'  web_weight=0.5  fetch_page_content='snippets only'  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5
    ▶ Chunk takes:  fetch_k=100  context_chunks=50
    ▶ LLM:          temperature=0.1  top_p=0.92  top_k=40
    ▶ Output:       max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
🔵 Masker                         Loaded _MASKING_REGEXES from configuration
🟢 Banned Word Check              *** Banned word check status by stage and algorithm:
🟢 Banned Word Check              - PROMPT_CHECK  : Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅  Prompt Check: ✅
🟢 Banned Word Check              - PIPELINE_CHECK: Jaccard: ✅  BM25: ✅  Regex+Levenshtein: ✅  Keybert: ✅
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

  ▶ Debug:        debug_level=30  debug_mode='ge'
  ▶ Chat Context: use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
  ▶ Talk with:    collection='Test'  chat_name='MyFirstChat'
  ▶ File Input:   file=None  path=None  metadata=None  file_cap=15
  ▶ Visual:       mark_text=True
  ▶ Retrieval:
    ▶ Strategies:   strategy='DEFAULT'  orchestrator_flow='THOROUGH_QUERY_REWRITE'  retrieve_mode='ALL'  rerank=True  threshold=0.5
    ▶ Weights:      vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0  regex_weight=1.0
    ▶ Web:          web_search='local_only'  web_weight=0.5  fetch_page_content='snippets only'  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5
    ▶ Chunk takes:  fetch_k=100  context_chunks=50
    ▶ LLM:          temperature=0.1  top_p=0.92  top_k=40
    ▶ Output:       max_output_tokens='14366'  context_size='32768'  terminal_line_size=200
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat  · type new: your question to start a new topic
💬 Your actual query>  what do hedgehogs eat and where do they live

🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (default): LOW (confidence: 42%, threshold: 64%)
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PROMPT_CHECK
🟢 Cache build Jaccard Banned     Built Jaccard n-gram cache with 139 entries for language english
🟢 BM25 Scorer Cache              Built BM25 banlist cache with 139 entries for language english
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 1573.36it/s]
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
🔵 UserQuery                      Original user query: 'what do hedgehogs eat and where do they live'
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (raw_user_query): LOW (confidence: 42%, threshold: 64%)
🔵 QueryRewrite                   No conversation history — skipping rewrite
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (rewrite_query): LOW (confidence: 42%, threshold: 64%)
🔵 LangDrift                      raw_query_language=english  orig_translated_query_en='what do hedgehogs eat and where do they live'
🔵 LangDrift                      rewrite_language=english  rewritten_query='what do hedgehogs eat and where do they live'
🔵 LangDrift                      post_rewrite_query_en='what do hedgehogs eat and where do they live'
🔵 FinalQuery                     Final query for retrieval: 'what do hedgehogs eat and where do they live' (unchanged)
🔵 LLM Plan                       Alternate queries: generating up to 3 retrieval variants from the normalized query.
🔵 TokenBudget                    Authoritative budget=2048 vs caller estimate=256; using authoritative value
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈219 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_QUERY_EXPAND stage: Multi-query expansion
🔵 Call LLM                       options: {'temperature': 0.5, 'top_k': 40, 'top_p': 0.95, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:10
   ⚪ MultiQuery                     Alternate queries (3)
   ↳                                 1: 'Habitat and diet of hedgehogs'
   ↳                                 2: 'Feeding habits and preferred habitats of hedgehogs'
   ↳                                 3: 'What do hedgehogs consume and where do they reside'
🔵 Retrieval Orchestration        query rewrite flow:
↳                                 seed_en='what do hedgehogs eat and where do they live'
↳                                 rewritten='what do hedgehogs eat and where do they live'
↳                                 final_en='what do hedgehogs eat and where do they live'
🔵 Retrieval Orchestration        language: en      normalized query final='what do hedgehogs eat and where do they live' alternate_count=3
🔵 Retrieval Orchestration        language: en      apply retrieval gates
🔵 Retrieval Orchestration        Query rewrite stage            Activated                    knob=use_query_rewrite
🔵 Retrieval Orchestration        Pronoun substitution stage     Activated                    knob=use_pronoun_substitution
🔵 Retrieval Orchestration        Secondary query stage          Activated                    knob=use_secondary_query
🔵 Retrieval Orchestration        Original-language vector stage Activated                    knob=use_original_language_vector
🔵 Retrieval Orchestration        Indexed query shaping stage    Activated                    knob=shape_indexed_queries
🔵 Retrieval Orchestration        Vector alternates stage        Activated                    knob=use_vector_alternates
🔵 Retrieval Orchestration        Grounding stage                Activated                    knob=run_grounding
🔵 Retrieval Orchestration        Rerank stage                   Activated                    knob=run_rerank
🔵 Retrieval Orchestration        Low-score fallback action      Activated                    knob=run_low_score_fallback
🔵 Retrieval Orchestration        Low-recall rescue action       Activated                    knob=run_low_recall_rescue
🔵 Retrieval Orchestration        Local stage                    Activated                    knob=run_local_stage
🔵 Retrieval Orchestration        Web stage                      Activated                    knob=run_web_stage
🔵 Retrieval Orchestration        Vector retriever stage         Activated                    knob=run_vector
🔵 Retrieval Orchestration        BM25 retriever stage           Activated                    knob=run_bm25
🔵 Retrieval Orchestration        Graph retriever stage          Activated                    knob=run_graph
🔵 Retrieval Orchestration        Regex retriever stage          Activated                    knob=run_regex
🔵 Retrieval Orchestration        original-language vector leg: off lang=en query='what do hedgehogs eat and where do they live' reason=user query already in English
🔵 Retrieval Orchestration        language: en      stage plan flow=THOROUGH_QUERY_REWRITE mode=ALL guardrail=off original_leg=off rewrite=on pronouns=on rerank=on low_score_fallback=on
↳                                 low_recall_rescue=on grounding=on indexed_shape=on alt_queries=3 retrievers=v=on,b=on,g=on,r=on
🔵 Retrieval Orchestration        effective stage gates: flow=THOROUGH_QUERY_REWRITE mode=ALL web_mode=local_only local_stage=on web_stage=off
🔵 Retrieval Orchestration        language: en      retrieve local candidates mode=ALL
🔵 Retrieval Orchestration        query language flow: user=en current=en rewrite=en retrieval=en
🔵 Retrieval Orchestration        retriever: Vector     query: 'what do hedgehogs eat and where do they live' dispatch query
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test for language en
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance                Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6246     0.3754    0.3754  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              2        0.5503     0.4497    0.4497  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              3        0.4291     0.5709    0.5709  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              4        0.4041     0.5959    0.5959  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              5        0.3788     0.6212    0.6212  Vector                     Hedgehogs.pdf
   ⚪ Chroma                              6        0.3410     0.6590    0.6590  Vector                     Cats.md
   ⚪ Chroma                              7        0.3022     0.6978    0.6978  Vector                     Cats.md
   ⚪ Chroma                              8        0.2948     0.7052    0.7052  Vector                     Apes.docx
   ⚪ Chroma                              9        0.2915     0.7085    0.7085  Vector                     Kamele.txt
   ⚪ Chroma                             10        0.2877     0.7123    0.7123  Vector                     Fish.txt
   ⚪ Chroma                             11        0.2725     0.7275    0.7275  Vector                     Lions.pptx
   ⚪ Chroma                             12        0.2635     0.7365    0.7365  Vector                     Cats.md
   ⚪ Chroma                             13        0.2571     0.7429    0.7429  Vector                     Cats.md
   ⚪ Chroma                             14        0.2438     0.7562    0.7562  Vector                     Hedgehogs.pdf
   ⚪ Chroma                             15        0.2421     0.7579    0.7579  Vector                     Lions.pptx
   ⚪ Chroma                             16        0.2406     0.7594    0.7594  Vector                     Cats.md
   ⚪ Chroma                             17        0.2396     0.7604    0.7604  Vector                     Cats.md
   ⚪ Chroma                             18        0.2353     0.7647    0.7647  Vector                     Cats.md
   ⚪ Chroma                             19        0.2304     0.7696    0.7696  Vector                     Cats.md
   ⚪ Chroma                             20        0.2285     0.7715    0.7715  Vector                     Cats.md
   ⚪ Chroma                             21        0.2259     0.7741    0.7741  Vector                     Cats.md
   ⚪ Chroma                             22        0.2219     0.7781    0.7781  Vector                     Cats.md
   ⚪ Chroma                             23        0.2203     0.7797    0.7797  Vector                     Cats.md
   ⚪ Chroma                             24        0.2169     0.7831    0.7831  Vector                     Fish.txt
   ⚪ Chroma                             25        0.2159     0.7841    0.7841  Vector                     Cats.md
   ⚪ Chroma                             26        0.2155     0.7845    0.7845  Vector                     Cats.md
   ⚪ Chroma                             27        0.2150     0.7850    0.7850  Vector                     Cats.md
   ⚪ Chroma                             28        0.2136     0.7864    0.7864  Vector                     Cats.md
   ⚪ Chroma                             29        0.2130     0.7870    0.7870  Vector                     Cats.md
   ⚪ Chroma                             30        0.2110     0.7890    0.7890  Vector                     Cats.md
   ⚪ Chroma                             31        0.2105     0.7895    0.7895  Vector                     Apes.docx
   ⚪ Chroma                             32        0.2105     0.7895    0.7895  Vector                     Cats.md
   ⚪ Chroma                             33        0.2086     0.7914    0.7914  Vector                     Elephants.jpg
   ⚪ Chroma                             34        0.2067     0.7933    0.7933  Vector                     Cats.md
   ⚪ Chroma                             35        0.2026     0.7974    0.7974  Vector                     Cats.md
   ⚪ Chroma                             36        0.1994     0.8006    0.8006  Vector                     Cats.md
   ⚪ Chroma                             37        0.1988     0.8012    0.8012  Vector                     Kamele.txt
   ⚪ Chroma                             38        0.1851     0.8149    0.8149  Vector                     Dogs.png
   ⚪ Chroma                             39        0.1847     0.8153    0.8153  Vector                     Dogs.png
   ⚪ Chroma                             40        0.1820     0.8180    0.8180  Vector                     Cats.md
   ⚪ Chroma                             41        0.1796     0.8204    0.8204  Vector                     Cats.md
   ⚪ Chroma                             42        0.1770     0.8230    0.8230  Vector                     Cats.md
   ⚪ Chroma                             43        0.1769     0.8231    0.8231  Vector                     Cats.md
   ⚪ Chroma                             44        0.1691     0.8309    0.8309  Vector                     Dogs.png
   ⚪ Chroma                             45        0.1587     0.8413    0.8413  Vector                     Kamele.txt
   ⚪ Chroma                             46        0.1546     0.8454    0.8454  Vector                     Cats.md
   ⚪ Chroma                             47        0.1428     0.8572    0.8572  Vector                     Apes.docx
   ⚪ Chroma                             48        0.1421     0.8579    0.8579  Vector                     Fish.txt
   ⚪ Chroma                             49        0.1374     0.8626    0.8626  Vector                     Apes.docx
   ⚪ Chroma                             50        0.1355     0.8645    0.8645  Vector                     Kamele.txt
   ⚪ Chroma                             51        0.1323     0.8677    0.8677  Vector                     Cats.md
   ⚪ Chroma                             52        0.1322     0.8678    0.8678  Vector                     Cats.md
   ⚪ Chroma                             53        0.1307     0.8693    0.8693  Vector                     Fish.txt
   ⚪ Chroma                             54        0.1290     0.8710    0.8710  Vector                     Cats.md
   ⚪ Chroma                             55        0.1218     0.8782    0.8782  Vector                     Cats.md
   ⚪ Chroma                             56        0.1185     0.8815    0.8815  Vector                     Pferde.pdf
   ⚪ Chroma                             57        0.1179     0.8821    0.8821  Vector                     Kamele.txt
   ⚪ Chroma                             58        0.1089     0.8911    0.8911  Vector                     Cats.md
   ⚪ Chroma                             59        0.1035     0.8965    0.8965  Vector                     Pferde.pdf
   ⚪ Chroma                             60        0.1019     0.8981    0.8981  Vector                     Fish.txt
   ⚪ Chroma                             61        0.1014     0.8986    0.8986  Vector                     Apes.docx
   ⚪ Chroma                             62        0.0943     0.9057    0.9057  Vector                     Kamele.txt
   ⚪ Chroma                             63        0.0922     0.9078    0.9078  Vector                     Cats.md
   ⚪ Chroma                             64        0.0889     0.9111    0.9111  Vector                     Lions.pptx
   ⚪ Chroma                             65        0.0835     0.9165    0.9165  Vector                     Pferde.pdf
   ⚪ Chroma                             66        0.0811     0.9189    0.9189  Vector                     Cats.md
   ⚪ Chroma                             67        0.0767     0.9233    0.9233  Vector                     Lions.pptx
   ⚪ Chroma                             68        0.0658     0.9342    0.9342  Vector                     Cats.md
   ⚪ Chroma                             69        0.0635     0.9365    0.9365  Vector                     Apes.docx
   ⚪ Chroma                             70        0.0634     0.9366    0.9366  Vector                     Fish.txt
   ⚪ Chroma                             71        0.0480     0.9520    0.9520  Vector                     Pferde.pdf
   ⚪ Chroma                             72        0.0452     0.9548    0.9548  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             73        0.0214     0.9786    0.9786  Vector                     LionsAndApes.xlsx
   ⚪ Chroma                             74        0.0131     0.9869    0.9869  Vector                     Lions.pptx
   ⚪ Chroma                             75        0.0026     0.9974    0.9974  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             76       -0.0026     1.0026    1.0026  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             77       -0.0053     1.0053    1.0053  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             78       -0.0269     1.0269    1.0269  Vector                     BlazingFast_Workstation.md
   ⚪ Chroma                             79       -0.0336     1.0336    1.0336  Vector                     BlazingFast_Workstation.md
🟢 Chroma                         Querying Chroma DB query returned 79 chunks
🔵 MultiQuery                     3 Alternate queries for vector retrieval:
↳                                 1: 'Habitat and diet of hedgehogs'
↳                                 2: 'Feeding habits and preferred habitats of hedgehogs'
↳                                 3: 'What do hedgehogs consume and where do they reside'
🔵 Retrieval Orchestration        discovered corpus language buckets: [de, en] active config buckets: [de, en, es, fr, it]
🔵 Retrieval Orchestration        idx_stage l='en' r='B/G/R' tr=0
↳                                 primary_query='what do hedgehogs eat and where do they live'
↳                                 stage_primary_query='what do hedgehogs eat and where do they live'
↳                                 guardrail_query='what do hedgehogs eat and where do they live'
↳                                 stage_guardrail_query='what do hedgehogs eat and where do they live'
🔵 Retrieval Orchestration        idx_stage l='de' r='B/G/R' tr=1
↳                                 primary_query='what do hedgehogs eat and where do they live'
↳                                 stage_primary_query='Was essen Igel und wo leben sie?'
↳                                 guardrail_query='what do hedgehogs eat and where do they live'
↳                                 stage_guardrail_query='Was essen Igel und wo leben sie?'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: en      query: 'what do hedgehogs eat and where do they live'
🔵 Retrieval Orchestration        retrievers='BM25/Graph/Regex' language: de      query: 'Was essen Igel und wo leben sie?'
🔵 BM25                           Querying bm25 index on collection Test for language en
🟢 BM25                           Loaded persisted BM25 index (79 chunks, 1907 terms)
🟢 BM25                           BM25 retrieval returned 57 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        9.6676         BM25                Fish.txt
   ⚪ BM25                                2        9.3282         BM25                Lions.pptx
   ⚪ BM25                                3        5.6766         BM25                Fish.txt
   ⚪ BM25                                4        5.2986         BM25                Hedgehogs.pdf
   ⚪ BM25                                5        5.0903         BM25                Cats.md
   ⚪ BM25                                6        5.0854         BM25                Hedgehogs.pdf
   ⚪ BM25                                7        4.3203         BM25                Apes.docx
   ⚪ BM25                                8        3.5794         BM25                Hedgehogs.pdf
   ⚪ BM25                                9        3.5759         BM25                Hedgehogs.pdf
   ⚪ BM25                               10        3.5759         BM25                Hedgehogs.pdf
   ⚪ BM25                               11        3.4480         BM25                Apes.docx
   ⚪ BM25                               12        3.4333         BM25                Cats.md
   ⚪ BM25                               13        3.3983         BM25                Cats.md
   ⚪ BM25                               14        3.0725         BM25                Cats.md
   ⚪ BM25                               15        2.7674         BM25                Cats.md
   ⚪ BM25                               16        2.6521         BM25                Apes.docx
   ⚪ BM25                               17        2.4462         BM25                Lions.pptx
   ⚪ BM25                               18        2.3544         BM25                Fish.txt
   ⚪ BM25                               19        2.3501         BM25                Cats.md
   ⚪ BM25                               20        2.3338         BM25                Elephants.jpg
🔵 BM25                           Querying bm25 index on collection Test for language de
🟢 BM25                           BM25 retrieval returned 10 chunks
   ⚪ BM25                              Pos     BM25Score                Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        7.4167         BM25                Kamele.txt
   ⚪ BM25                                2        6.7486         BM25                Kamele.txt
   ⚪ BM25                                3        6.4373         BM25                Kamele.txt
   ⚪ BM25                                4        5.2806         BM25                Pferde.pdf
   ⚪ BM25                                5        4.8235         BM25                Kamele.txt
   ⚪ BM25                                6        3.3619         BM25                Pferde.pdf
   ⚪ BM25                                7        3.0296         BM25                Pferde.pdf
   ⚪ BM25                                8        2.9275         BM25                Kamele.txt
   ⚪ BM25                                9        2.7825         BM25                Kamele.txt
   ⚪ BM25                               10        2.7406         BM25                Pferde.pdf
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
🟢 Regex                          Regex retrieval returned 11 chunks
   ⚪ Regex                             Pos    RegexScore   VerbHit   NounHit                Retrievers   File
   ⚪ Regex                          ----------------------------------------------------------------------------------------------
   ⚪ Regex                               1        0.6000         1         0                    Regex    Apes.docx
   ⚪ Regex                               2        0.6000         1         0                    Regex    Apes.docx
   ⚪ Regex                               3        0.6000         1         0                    Regex    Apes.docx
   ⚪ Regex                               4        0.6000         1         0                    Regex    Cats.md
   ⚪ Regex                               5        0.6000         1         0                    Regex    Cats.md
   ⚪ Regex                               6        0.6000         1         0                    Regex    Cats.md
   ⚪ Regex                               7        0.6000         1         0                    Regex    Cats.md
   ⚪ Regex                               8        0.6000         1         0                    Regex    Elephants.jpg
   ⚪ Regex                               9        0.6000         1         0                    Regex    Fish.txt
   ⚪ Regex                              10        0.6000         1         0                    Regex    Fish.txt
   ⚪ Regex                              11        0.6000         1         0                    Regex    Lions.pptx
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
🔵 Retrieval Orchestration        language: en      local docs fetched vector=79 bm25=67 graph=60 regex=11
🔵 Retrieval Orchestration        language: en      web stage skipped (web_mode=local_only, flow_gate=True)
🔵 Retrieval Orchestration        language: en      merge candidates and build context
🟢 Merge                          Reciprocal Rank Fusion (RRF) produced 79 local chunks
   ⚪ Merge                             Pos    RRFScore                            Retrievers  File
   ⚪ Merge                          ----------------------------------------------------------------------------------
   ⚪ Merge                               1      0.0605              Vector BM25 Graph Regex   Fish.txt
   ⚪ Merge                               2      0.0588              Vector BM25 Graph Regex   Apes.docx
   ⚪ Merge                               3      0.0587              Vector BM25 Graph Regex   Cats.md
   ⚪ Merge                               4      0.0573              Vector BM25 Graph Regex   Cats.md
   ⚪ Merge                               5      0.0561              Vector BM25 Graph Regex   Fish.txt
   ⚪ Merge                               6      0.0555              Vector BM25 Graph Regex   Apes.docx
   ⚪ Merge                               7      0.0498              Vector BM25 Graph Regex   Apes.docx
   ⚪ Merge                               8      0.0498              Vector BM25 Graph Regex   Cats.md
   ⚪ Merge                               9      0.0493              Vector BM25 Graph Regex   Elephants.jpg
   ⚪ Merge                              10      0.0476              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              11      0.0469              Vector BM25 Graph Regex   Cats.md
   ⚪ Merge                              12      0.0467              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              13      0.0453              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              14      0.0444              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              15      0.0434              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              16      0.0425              Vector BM25 Graph         Hedgehogs.pdf
   ⚪ Merge                              17      0.0411              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              18      0.0404              Vector BM25       Regex   Lions.pptx
   ⚪ Merge                              19      0.0380              Vector BM25 Graph         Cats.md
   ⚪ Merge                              20      0.0367              Vector BM25 Graph         Cats.md
   ⚪ Merge                              21      0.0363              Vector BM25 Graph         Cats.md
   ⚪ Merge                              22      0.0358              Vector BM25 Graph         Cats.md
   ⚪ Merge                              23      0.0349              Vector BM25 Graph         Cats.md
   ⚪ Merge                              24      0.0346              Vector BM25 Graph         Cats.md
   ⚪ Merge                              25      0.0343              Vector BM25 Graph         Cats.md
   ⚪ Merge                              26      0.0338              Vector BM25 Graph         Cats.md
   ⚪ Merge                              27      0.0334              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              28      0.0329              Vector BM25 Graph         Cats.md
   ⚪ Merge                              29      0.0329              Vector BM25 Graph         Apes.docx
   ⚪ Merge                              30      0.0327              Vector BM25 Graph         Cats.md
   ⚪ Merge                              31      0.0327              Vector BM25 Graph         Cats.md
   ⚪ Merge                              32      0.0322              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              33      0.0322              Vector BM25 Graph         Cats.md
   ⚪ Merge                              34      0.0321              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              35      0.0319              Vector BM25 Graph         Fish.txt
   ⚪ Merge                              36      0.0317              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              37      0.0315              Vector BM25 Graph         Cats.md
   ⚪ Merge                              38      0.0315              Vector BM25 Graph         Cats.md
   ⚪ Merge                              39      0.0303              Vector BM25 Graph         Cats.md
   ⚪ Merge                              40      0.0294              Vector BM25 Graph         Cats.md
   ⚪ Merge                              41      0.0287              Vector BM25 Graph         Cats.md
   ⚪ Merge                              42      0.0285              Vector BM25 Graph         Lions.pptx
   ⚪ Merge                              43      0.0282              Vector BM25 Graph         Cats.md
   ⚪ Merge                              44      0.0279              Vector BM25 Graph         Cats.md
   ⚪ Merge                              45      0.0278              Vector BM25 Graph         Cats.md
   ⚪ Merge                              46      0.0276              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              47      0.0269              Vector BM25 Graph         Cats.md
   ⚪ Merge                              48      0.0266              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              49      0.0255              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              50      0.0254              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              51      0.0252              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              52      0.0251              Vector BM25 Graph         Kamele.txt
   ⚪ Merge                              53      0.0248              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              54      0.0248              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              55      0.0245              Vector BM25               Hedgehogs.pdf
   ⚪ Merge                              56      0.0244              Vector BM25 Graph         Pferde.pdf
   ⚪ Merge                              57      0.0228              Vector BM25               Cats.md
   ⚪ Merge                              58      0.0218              Vector BM25               Cats.md
   ⚪ Merge                              59      0.0216              Vector      Graph         Cats.md
   ⚪ Merge                              60      0.0213              Vector      Graph         Dogs.png
   ⚪ Merge                              61      0.0209              Vector BM25               Cats.md
   ⚪ Merge                              62      0.0207              Vector      Graph         Dogs.png
   ⚪ Merge                              63      0.0206              Vector      Graph         Dogs.png
   ⚪ Merge                              64      0.0202              Vector      Graph         Fish.txt
   ⚪ Merge                              65      0.0190              Vector BM25               Apes.docx
   ⚪ Merge                              66      0.0187              Vector      Graph         Cats.md
   ⚪ Merge                              67      0.0184              Vector BM25               Lions.pptx
   ⚪ Merge                              68      0.0179              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              69      0.0177              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              70      0.0172              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              71      0.0167              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              72      0.0162              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              73      0.0161              Vector BM25               BlazingFast_Workstation.md
   ⚪ Merge                              74      0.0123              Vector                    Cats.md
   ⚪ Merge                              75      0.0115              Vector                    Cats.md
   ⚪ Merge                              76      0.0106              Vector                    Cats.md
   ⚪ Merge                              77      0.0105              Vector                    Cats.md
   ⚪ Merge                              78      0.0088              Vector                    Cats.md
   ⚪ Merge                              79      0.0075              Vector                    LionsAndApes.xlsx
🔵 ChunkDedup                     Removed 1 near-duplicate chunk(s) (threshold=0.85, include_web=on, kept 78)
🔵 Rerank                         Pre-rerank pool local=78 web=0 total=78 original_leg=off lang=en added[V/B/G/R]=0/0/0/0
   ⚪ Rerank                            Pos    RawScore    AdjScore                Retrievers  File                                      Text
   ⚪ Rerank                         --------------------------------------------------------------------------------------------------------------
   ⚪ Rerank                              1      2.8074      1.0000  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
   ⚪ Rerank                              2      2.4219      0.9695  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
   ⚪ Rerank                              3      0.9349      0.8518  Vector BM25 Graph         Kamele.txt                                ernährung kamele sind pflanzenfresser (h
   ⚪ Rerank                              4     -0.5777      0.7321  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
   ⚪ Rerank                              5     -0.6281      0.7282  Vector BM25 Graph         Kamele.txt                                sie liefern milch, fleisch, wolle und le
   ⚪ Rerank                              6     -0.8446      0.7110  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
   ⚪ Rerank                              7     -1.0079      0.6981  Vector BM25 Graph         Kamele.txt                                weitere körperliche anpassungen: - dicht
   ⚪ Rerank                              8     -1.3370      0.6721  Vector BM25 Graph Regex   Apes.docx                                 Gorillas are the largest living primates
   ⚪ Rerank                              9     -2.0804      0.6132  Vector BM25 Graph Regex   Elephants.jpg                             elephants large, long-lived mammals fami
   ⚪ Rerank                             10     -2.0842      0.6129  Vector BM25 Graph         Kamele.txt                                kamele – überlebenskünstler der wüste ka
   ⚪ Rerank                             11     -2.7209      0.5626  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
   ⚪ Rerank                             12     -3.0684      0.5351  Vector BM25 Graph         Apes.docx                                 Orangutans are the only great apes found
   ⚪ Rerank                             13     -3.2950      0.5171  Vector BM25 Graph Regex   Apes.docx                                 Chimpanzees (Pan troglodytes) are found
   ⚪ Rerank                             14     -3.4943      0.5014  Vector BM25 Graph         Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
   ⚪ Rerank                             15     -3.5013      0.5008  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
   ⚪ Rerank                             16     -4.3843      0.4309  Vector      Graph         Dogs.png                                  tasks. understanding breed-specific ende
   ⚪ Rerank                             17     -4.4854      0.4229  Vector BM25               Apes.docx                                 All great apes demonstrate remarkable co
   ⚪ Rerank                             18     -4.5428      0.4184  Vector BM25 Graph         Kamele.txt                                in einigen kulturen spielen kamele auch
   ⚪ Rerank                             19     -4.5796      0.4155  Vector BM25 Graph Regex   Cats.md                                   cats have lived alongside humans for tho
   ⚪ Rerank                             20     -4.6147      0.4127  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
   ⚪ Rerank                             21     -4.6751      0.4079  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
   ⚪ Rerank                             22     -4.7720      0.4003  Vector BM25 Graph Regex   Fish.txt                                   what fish are fish are a diverse group
   ⚪ Rerank                             23     -4.8073      0.3975  Vector BM25 Graph         Cats.md                                   cats are crepuscular hunters with: - a h
   ⚪ Rerank                             24     -4.9275      0.3880  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
   ⚪ Rerank                             25     -5.0120      0.3813  Vector BM25 Graph         Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
   ⚪ Rerank                             26     -5.0869      0.3753  Vector      Graph         Dogs.png                                  , comprehensive text dogs dogs accompani
   ⚪ Rerank                             27     -5.1383      0.3713  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
   ⚪ Rerank                             28     -5.4808      0.3442  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
   ⚪ Rerank                             29     -5.5839      0.3360  Vector BM25 Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
   ⚪ Rerank                             30     -5.6293      0.3324  Vector BM25 Graph         Cats.md                                   cats can detect frequencies up to ~64 kh
   ⚪ Rerank                             31     -5.7747      0.3209  Vector BM25       Regex   Lions.pptx                                Slide 1: Lions: The King of the Savanna
   ⚪ Rerank                             32     -5.8761      0.3129  Vector BM25 Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
   ⚪ Rerank                             33     -5.9337      0.3083  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
   ⚪ Rerank                             34     -6.1741      0.2893  Vector BM25 Graph         Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
   ⚪ Rerank                             35     -6.2788      0.2810  Vector BM25 Graph         Cats.md                                   cats communicate through: - vocalization
   ⚪ Rerank                             36     -6.4414      0.2682  Vector BM25 Graph Regex   Fish.txt                                  diversity: fish include jawless fishes (
   ⚪ Rerank                             37     -6.4713      0.2658  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
   ⚪ Rerank                             38     -6.7015      0.2476  Vector BM25 Graph         Cats.md                                   kittens engage in: - stalking. - pouncin
   ⚪ Rerank                             39     -6.8070      0.2392  Vector BM25 Graph         Cats.md                                   modern domestic cats descend from the ne
   ⚪ Rerank                             40     -6.8083      0.2391  Vector BM25 Graph         Cats.md                                   cats were not bred for specific tasks ea
   ⚪ Rerank                             41     -6.8271      0.2376  Vector BM25 Graph Regex   Apes.docx                                 Great apes (Hominidae) are the closest l
   ⚪ Rerank                             42     -6.8341      0.2371  Vector BM25 Graph         Cats.md                                   cats suffered during periods of supersti
   ⚪ Rerank                             43     -6.8916      0.2325  Vector      Graph         Cats.md                                   cats were revered, often associated with
   ⚪ Rerank                             44     -7.0111      0.2231  Vector BM25 Graph         Cats.md                                   free-roaming cats are estimated to kill
   ⚪ Rerank                             45     -7.1000      0.2161  Vector                    Cats.md                                   spaying/neutering is essential to reduce
   ⚪ Rerank                             46     -7.4386      0.1893  Vector BM25 Graph         Cats.md                                   cats are complex, adaptable, and endless
   ⚪ Rerank                             47     -7.4800      0.1860  Vector BM25 Graph Regex   Cats.md                                   cats are often described as solitary, bu
   ⚪ Rerank                             48     -7.5457      0.1808  Vector BM25               Cats.md                                   - regular veterinary checkups - vaccinat
   ⚪ Rerank                             49     -7.5485      0.1806  Vector BM25 Graph         Cats.md                                   some breeds have known predispositions:
   ⚪ Rerank                             50     -7.5760      0.1784  Vector                    Cats.md                                   - climbing structures - scratching posts
🔵 Rerank                         Reranking with cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 returned 78 chunks
🔵 Chunk selection                Strategy 'DEFAULT' → ScoreRankedSelector
🔵 Rerank select                  logit+ = effective logit after single-chunk boost (raw in raw_rerank_score)
🟣 Rerank select                        Logit [Sigmoid]       Thr      ΔProb  Retrievers                File                                      Text
🟣 Rerank select                  --------------------------------------------------------------------------------------------------------------------
🟣 Rerank select                   ❌   -9.8304 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
🟣 Rerank select                   ❌   -9.8285 [0.000]  (0.5000)    -0.4999  Vector      Graph         Fish.txt                                  poikilotherm: a term describing organism
🟣 Rerank select                   ❌   -9.8259 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
🟣 Rerank select                   ❌   -9.7291 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Pferde.pdf                                Page 2 KI-Modelle, die semantische Feinh
🟣 Rerank select                   ❌   -9.5246 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
🟣 Rerank select                   ❌   -9.4535 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                the blazingfast workstation is an enterp
🟣 Rerank select                   ❌   -9.4090 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph         Fish.txt                                  terminology explained cold-blooded: a co
🟣 Rerank select                   ❌   -9.2607 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph Regex   Cats.md                                   indoor cats often live 12–18 years; some
🟣 Rerank select                   ❌   -9.1021 [0.000]  (0.5000)    -0.4999  Vector BM25               BlazingFast_Workstation.md                - **processors**: dual enterprise platin
🟣 Rerank select                   ❌   -9.0856 [0.000]  (0.5000)    -0.4999  Vector BM25 Graph Regex   Cats.md                                   outdoor access: - pros: exercise, stimul
🟣 Rerank select                   ❌   -8.9715 [0.000]  (0.5000)    -0.4999  Vector                    Cats.md                                   a deep, structured exploration of domest
🟣 Rerank select                   ❌   -8.7612 [0.000]  (0.5000)    -0.4998  Vector BM25               Cats.md                                   - dental disease - kidney disease - hype
🟣 Rerank select                   ❌   -8.7280 [0.000]  (0.5000)    -0.4998  Vector BM25 Graph         Cats.md                                   - free feeding can lead to obesity. - pu
🟣 Rerank select                   ❌   -8.6081+[0.000]  (0.5000)    -0.4998  Vector                    LionsAndApes.xlsx                         category detail value taxonomy scientifi
🟣 Rerank select                   ❌   -8.4550 [0.000]  (0.5000)    -0.4998  Vector BM25               BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
🟣 Rerank select                   ❌   -8.1825 [0.000]  (0.5000)    -0.4997  Vector                    Cats.md                                   - puzzle toys - training sessions - nove
🟣 Rerank select                   ❌   -8.0579 [0.000]  (0.5000)    -0.4997  Vector                    Cats.md                                   - human interaction - multi-cat househol
🟣 Rerank select                   ❌   -7.9716 [0.000]  (0.5000)    -0.4997  Vector BM25 Graph         Cats.md                                   the mechanism of purring is still debate
🟣 Rerank select                   ❌   -7.9359 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Cats.md                                   cats form strong bonds but express affec
🟣 Rerank select                   ❌   -7.9168 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Cats.md                                   - kingdom: animalia - phylum: chordata -
🟣 Rerank select                   ❌   -7.8151 [0.000]  (0.5000)    -0.4996  Vector BM25               Cats.md                                   selective breeding intensified in the 19
🟣 Rerank select                   ❌   -7.7968 [0.000]  (0.5000)    -0.4996  Vector BM25               Hedgehogs.pdf                             Page 2 behavioral ecology, hibernation p
🟣 Rerank select                   ❌   -7.7342 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Cats.md                                   they have a righting reflex, but falls f
🟣 Rerank select                   ❌   -7.7298 [0.000]  (0.5000)    -0.4996  Vector BM25 Graph         Fish.txt                                  ectotherm: the preferred scientific term
🟣 Rerank select                   ❌   -7.7003 [0.000]  (0.5000)    -0.4995  Vector      Graph         Cats.md                                   cats dominate digital culture: - memes (
🟣 Rerank select                   ❌   -7.6720 [0.000]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   most adult cats are lactose intolerant.
🟣 Rerank select                   ❌   -7.6283 [0.000]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   these breeds emerged without heavy human
🟣 Rerank select                   ❌   -7.6228 [0.000]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   even well-fed cats hunt. the sequence: 1
🟣 Rerank select                   ❌   -7.5760 [0.001]  (0.5000)    -0.4995  Vector                    Cats.md                                   - climbing structures - scratching posts
🟣 Rerank select                   ❌   -7.5485 [0.001]  (0.5000)    -0.4995  Vector BM25 Graph         Cats.md                                   some breeds have known predispositions:
🟣 Rerank select                   ❌   -7.5457 [0.001]  (0.5000)    -0.4995  Vector BM25               Cats.md                                   - regular veterinary checkups - vaccinat
🟣 Rerank select                   ❌   -7.4800 [0.001]  (0.5000)    -0.4994  Vector BM25 Graph Regex   Cats.md                                   cats are often described as solitary, bu
🟣 Rerank select                   ❌   -7.4386 [0.001]  (0.5000)    -0.4994  Vector BM25 Graph         Cats.md                                   cats are complex, adaptable, and endless
🟣 Rerank select                   ❌   -7.1000 [0.001]  (0.5000)    -0.4992  Vector                    Cats.md                                   spaying/neutering is essential to reduce
🟣 Rerank select                   ❌   -7.0111 [0.001]  (0.5000)    -0.4991  Vector BM25 Graph         Cats.md                                   free-roaming cats are estimated to kill
🟣 Rerank select                   ❌   -6.8916 [0.001]  (0.5000)    -0.4990  Vector      Graph         Cats.md                                   cats were revered, often associated with
🟣 Rerank select                   ❌   -6.8341 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph         Cats.md                                   cats suffered during periods of supersti
🟣 Rerank select                   ❌   -6.8271 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph Regex   Apes.docx                                 Great apes (Hominidae) are the closest l
🟣 Rerank select                   ❌   -6.8083 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph         Cats.md                                   cats were not bred for specific tasks ea
🟣 Rerank select                   ❌   -6.8070 [0.001]  (0.5000)    -0.4989  Vector BM25 Graph         Cats.md                                   modern domestic cats descend from the ne
🟣 Rerank select                   ❌   -6.7015 [0.001]  (0.5000)    -0.4988  Vector BM25 Graph         Cats.md                                   kittens engage in: - stalking. - pouncin
🟣 Rerank select                   ❌   -6.4713 [0.002]  (0.5000)    -0.4985  Vector BM25 Graph         Apes.docx                                 All great ape species are either endange
🟣 Rerank select                   ❌   -6.4414 [0.002]  (0.5000)    -0.4984  Vector BM25 Graph Regex   Fish.txt                                  diversity: fish include jawless fishes (
🟣 Rerank select                   ❌   -6.2788 [0.002]  (0.5000)    -0.4981  Vector BM25 Graph         Cats.md                                   cats communicate through: - vocalization
🟣 Rerank select                   ❌   -6.1741 [0.002]  (0.5000)    -0.4979  Vector BM25 Graph         Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
🟣 Rerank select                   ❌   -5.9337 [0.003]  (0.5000)    -0.4974  Vector BM25 Graph         Lions.pptx                                Slide 4: Conservation Status Lions are c
🟣 Rerank select                   ❌   -5.8761 [0.003]  (0.5000)    -0.4972  Vector BM25 Graph         Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
🟣 Rerank select                   ❌   -5.7747 [0.003]  (0.5000)    -0.4969  Vector BM25       Regex   Lions.pptx                                Slide 1: Lions: The King of the Savanna
🟣 Rerank select                   ❌   -5.6293 [0.004]  (0.5000)    -0.4964  Vector BM25 Graph         Cats.md                                   cats can detect frequencies up to ~64 kh
🟣 Rerank select                   ❌   -5.5839 [0.004]  (0.5000)    -0.4963  Vector BM25 Graph         Cats.md                                   whiskers are deeply rooted sensory hairs
🟣 Rerank select                   ❌   -5.4808 [0.004]  (0.5000)    -0.4959  Vector BM25 Graph         Cats.md                                   cats require: - taurine - arachidonic ac
🟣 Rerank select                   ❌   -5.1383 [0.006]  (0.5000)    -0.4942  Vector BM25               Lions.pptx                                Slide 2: Physical Characteristics Male l
🟣 Rerank select                   ❌   -5.0869 [0.006]  (0.5000)    -0.4939  Vector      Graph         Dogs.png                                  , comprehensive text dogs dogs accompani
🟣 Rerank select                   ❌   -5.0120 [0.007]  (0.5000)    -0.4934  Vector BM25 Graph         Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
🟣 Rerank select                   ❌   -4.9275 [0.007]  (0.5000)    -0.4928  Vector BM25 Graph         Fish.txt                                  ecological roles and importance fish are
🟣 Rerank select                   ❌   -4.8073 [0.008]  (0.5000)    -0.4919  Vector BM25 Graph         Cats.md                                   cats are crepuscular hunters with: - a h
🟣 Rerank select                   ❌   -4.7720 [0.008]  (0.5000)    -0.4916  Vector BM25 Graph Regex   Fish.txt                                   what fish are fish are a diverse group
🟣 Rerank select                   ❌   -4.6751 [0.009]  (0.5000)    -0.4908  Vector BM25 Graph         Cats.md                                   as placental mammals, cats give live bir
🟣 Rerank select                   ❌   -4.6147 [0.010]  (0.5000)    -0.4902  Vector      Graph         Dogs.png                                  impact dogs human well-being. despite ma
🟣 Rerank select                   ❌   -4.5796 [0.010]  (0.5000)    -0.4898  Vector BM25 Graph Regex   Cats.md                                   cats have lived alongside humans for tho
🟣 Rerank select                   ❌   -4.5428 [0.011]  (0.5000)    -0.4895  Vector BM25 Graph         Kamele.txt                                in einigen kulturen spielen kamele auch
🟣 Rerank select                   ❌   -4.4854 [0.011]  (0.5000)    -0.4889  Vector BM25               Apes.docx                                 All great apes demonstrate remarkable co
🟣 Rerank select                   ❌   -4.3843 [0.012]  (0.5000)    -0.4877  Vector      Graph         Dogs.png                                  tasks. understanding breed-specific ende
🟣 Rerank select                   ❌   -3.5013 [0.029]  (0.5000)    -0.4707  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
🟣 Rerank select                   ❌   -3.4943 [0.029]  (0.5000)    -0.4705  Vector BM25 Graph         Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
🟣 Rerank select                   ❌   -3.2950 [0.036]  (0.5000)    -0.4643  Vector BM25 Graph Regex   Apes.docx                                 Chimpanzees (Pan troglodytes) are found
🟣 Rerank select                   ❌   -3.0684 [0.044]  (0.5000)    -0.4556  Vector BM25 Graph         Apes.docx                                 Orangutans are the only great apes found
🟣 Rerank select                   ❌   -2.7209 [0.062]  (0.5000)    -0.4383  Vector BM25 Graph         Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
🟣 Rerank select                   ❌   -2.0842 [0.111]  (0.5000)    -0.3894  Vector BM25 Graph         Kamele.txt                                kamele – überlebenskünstler der wüste ka
🟣 Rerank select                   ❌   -1.8573+[0.135]  (0.5000)    -0.3650  Vector BM25 Graph Regex   Elephants.jpg                             elephants large, long-lived mammals fami
🟣 Rerank select                   ❌   -1.3370 [0.208]  (0.5000)    -0.2920  Vector BM25 Graph Regex   Apes.docx                                 Gorillas are the largest living primates
🟣 Rerank select                   ❌   -1.0079 [0.267]  (0.5000)    -0.2326  Vector BM25 Graph         Kamele.txt                                weitere körperliche anpassungen: - dicht
🟣 Rerank select                   ❌   -0.8446 [0.301]  (0.5000)    -0.1994  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
🟣 Rerank select                   ❌   -0.6281 [0.348]  (0.5000)    -0.1520  Vector BM25 Graph         Kamele.txt                                sie liefern milch, fleisch, wolle und le
🟣 Rerank select                   ❌   -0.5777 [0.359]  (0.5000)    -0.1405  Vector BM25 Graph         Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
🟣 Rerank select                   ✅    2.8074 [0.943]  (0.5000)    +0.4431  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
🟣 Rerank select                   ✅    2.4219 [0.918]  (0.5000)    +0.4185  Vector BM25 Graph         Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
🟣 Rerank select                   ✅    0.9349 [0.718]  (0.5000)    +0.2181  Vector BM25 Graph         Kamele.txt                                ernährung kamele sind pflanzenfresser (h
🔵 LangDetect                     Detected language: English (en) — confidence: 42% below threshold 64% — falling back to English
🔵 LangDetect                     Confidence level (default): LOW (confidence: 42%, threshold: 64%)
🔵 Rerank fallback                Low-recall rescue action: Added 3 query-overlap local chunk(s) from retrieval order (input_fetch_k=100 input_context_chunks=50 fixed_k=100 min_local_pool=40
↳                                 target_min_local_hits=24 max_additional=24; local hits 3->6, pool=78).
🔵 Strategy: default              6 chunks selected after thresholding  0.5000  single-chunk boost ×1.25  low-recall rescue +3
🔵 Chunk selector: Score          Ranked     Selected 6 chunks.
🔵 Selected                       6 chunks selected
   ⚪ Selected                        #   Score [Sigmoid]  File                                      Text
   ⚪ Selected                       --------------------------------------------------------------------------------------------------------------------------------------------
   ⚪ Selected                        1    1.0000 [0.943]  Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are small, nocturnal mammals known
   ⚪ Selected                        2    0.9695 [0.918]  Hedgehogs.pdf                             Page 1 suburban gardens and agricultural edges. They are native to muc
   ⚪ Selected                        3    0.8518 [0.718]  Kamele.txt                                ernährung kamele sind pflanzenfresser (herbivoren) und fressen auch tr
   ⚪ Selected                        4    0.7321 [0.359]  Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive bite to subdue it. Seasonal s
   ⚪ Selected                        5    0.7110 [0.301]  Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations include a keen sense of smell and
   ⚪ Selected                        6    0.5008 [0.029]  Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as netting, open drains, and bon
   ⚪ ChunkSelect                    After chunk selection: 6/78 kept
🔵 Rerank Select Confidence       🟢 HIGH C_final=0.80 evidence_chunks=6 rerank_fallback=off
🔵 Retrieval Orchestration        language: en      retrieval completed chosen=6 context_chars=7491
🔵 TokenBudget                    [mistral:7b] context=32768 reserved_sys=1024 prompt≈2661 → max_output_tokens=2048
🔵 Resolved token params          max_output_tokens(api: max_tokens)=2048  num_ctx=32768  (override: max_tokens=None  num_ctx=None)
🔵 LLM Plan                       Answer generation: producing the final grounded response from retrieved context chunks.
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈2661 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_CHAT stage: Run user prompt
🔵 Call LLM                       options: {'temperature': 0.1, 'top_k': 40, 'top_p': 0.92, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:22
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 LangDetect                     Confidence level (answer_compliance_language): HIGH (confidence: 100%, threshold: 60%)
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PIPELINE_CHECK
🟢 KeyWrdChk Summary              Phrase                         dpt / brth     Regex+Levenshtein ➡️    Jaccard ➡️              BM25 ➡️                 Keybert
🟡 KeyWrdChk Summary              exploit                        2/4    2/4      2.0000/1.5000          1.0000/0.7500          -/-                    -/-
🟢 KeyWrdChk Depth                2 algos passed threshold vs. required 4
🟢 KeyWrdChk Breadth              2 algos had a score vs. required 4
🔵 Deleted chat context           Deleted chat context for MyFirstChat in Test_ChatContext
🔵 Add chat context               Upserted turn 1 for chat_name=MyFirstChat file_tag='' lang='english' to Test_ChatContext
   ⚪ Retrieval Orchestration        Stored query history forms orig='what do hedgehogs eat and where do they live' t1='what do hedgehogs eat and where do they live' t2='what do hedgehogs eat and
   ↳                                 where do they live' effective_reason=''
🔵 Masker                         0 of 15 rules produced matches and were replaced
💬 >   ### Answer
💬 >  Hedgehogs are insectivores by heritage but display flexible diets in the wild and captivity, consuming insects, small vertebrates, fruits, and vegetation. In the wild, their diet is dominated by
💬 >  invertebrates such as beetles, caterpillars, earthworms, slugs, and snails, but they will also consume amphibians, small rodents, eggs, carrion, and fallen fruit when available. Hedgehogs are
💬 >  native to much of Europe, parts of Asia, and Africa, with species adapted to local climates and food availability. They favor environments that provide cover and abundant invertebrate prey, such
💬 >  as hedgerows, compost heaps, and mixed vegetation.
💬 >
💬 >  ### Sources
💬 >  - FileName: Hedgehogs.pdf
💬 >    - FilePath: D:/RAG-LCC/TestDocs/Hedgehogs.pdf ([open](file:///D:/RAG-LCC/TestDocs/Hedgehogs.pdf))
💬 >    - Page: 1 (Page 1 suburban gardens and agricultural edges)
💬 >    - Page: 1 (Page 1 detect prey and a rapid, decisive bite to subdue it. Seasonal shifts in prey availability influence body condition and reproductive timing. In human-modified landscapes,
💬 >  hedgehogs may exploit supplementary food sources, which can alter natural foraging patterns and nutritional balance.)
💬 >    - Page: 2 (Page 2 detect prey and a rapid, decisive bite to subdue it. Seasonal shifts in prey availability influence body condition and reproductive timing. In human-modified landscapes,
💬 >  hedgehogs may exploit supplementary food sources, which can alter natural foraging patterns and nutritional balance. Reproduction and Life Cycle Reproductive timing varies by species and climate
💬 >  but generally aligns with periods of abundant food. Males may roam widely to locate receptive females. After mating, females undergo a gestation period that typically ranges from about 30 to 40
💬 >  days, depending on species. Litter sizes vary but commonly include several young, called hoglets, born blind and sparsely spined; quills harden within hours. Mothers nurse and groom their young,
💬 >  teaching them foraging skills before weaning. Juveniles)
💬 >    - Page: 1 (Page 1 tissues. Sensory adaptations include a keen sense of smell and hearing, both crucial for locating prey at night, while vision is adapted for low-light conditions rather than
💬 >  acute daytime sight. Behavior and Social Life Hedgehogs are primarily solitary and crepuscular to nocturnal. They spend daylight hours resting in nests made of leaves, grass, or burrows and
💬 >  emerge at dusk to forage. Foraging behavior is characterized by methodical exploration: hedgehogs use scent and whisker touch to locate invertebrates, often turning over leaf litter and probing
💬 >  soil. When threatened, their signature defensive posture is to roll into a ball, tucking head and limbs beneath the spines. Social interactions are limited outside of mating season. Courtship can
💬 >  involve vocalizations, circling)
💬 >
💬 >  ---
💬 >
💬 >  **Answer confidence:**
💬 >  🟢 HIGH
💬 >  - C_final=0.80            : final confidence in [0,1] after penalties.
💬 >  - C_top=0.47              : weighted best-chunk signal (0.50 x top_score).
💬 >  - C_mean_top3=0.26        : weighted top-3 stability (0.30 x mean_top3).
💬 >  - C_coverage=0.02         : weighted evidence coverage (0.15 x coverage).
💬 >  - C_local=0.05            : weighted local-source share (0.05 x local_share).
💬 >  - C_fallback_penalty=0.00 : subtraction when low-confidence rerank fallback was used.
💬 >  - top_score=0.94          : strongest selected chunk confidence.
💬 >  - mean_top3=0.86          : average confidence of the top 3 selected chunks.
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
   📎 Hedgehogs.pdf (highlighted): file:///D:/RAG-LCC/tmp/rag_marked_01dafdtx/Hedgehogs_marked.pdf
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy! / orchestrator_flow!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
 ```
