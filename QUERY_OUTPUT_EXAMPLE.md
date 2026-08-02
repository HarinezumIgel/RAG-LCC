# Query Output Example

``` Text
Chat with your documents

  ▶ Debug:        debug_level=30  debug_mode='ge'
  ▶ Chat Context: use_chat_context=True  history_keep=10  history_prune=5  rewrite_context=3  topic_summary='last'
  ▶ Talk with:    collection='Test'  chat_name='MyFirstChat'
  ▶ File Input:   file=None  path=None  file_cap=15
  ▶ Visual:       mark_text=True
  ▶ Retrieval:
    ▶ Strategies:   strategy='DEFAULT'  retrieve_mode='ALL'  rerank=True  threshold=0.6
    ▶ Weights:      vector_weight=1.0  bm25_weight=1.0  graph_weight=1.0
    ▶ Web:          web_search='local_only'  web_weight=0.5  fetch_page_content='snippets only'  bm25_pre_filter=0.1  cosine_pre_filter=0.3  web_rerank_threshold=0.5
    ▶ Chunk takes:  fetch_k=100  context_chunks=50
    ▶ LLM:          temperature=0.1  top_p=0.92  top_k=40
    ▶ Output:       max_output_tokens='14366'  context_size='32768'  terminal_line_size=180
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >threshold=0.7
 🛠️  >web_search!
? Choose web search mode: local_and_web — web + local retrieval
 🛠️  >
b: back to settings / ↵ to enter query / ↵↵ to quit RAGChat  · type new: your question to start a new topic
🌐 Internet search is ON — web + local indexes.  💬 Your actual query>  what  is the hedgehog diet

🔵 LangDetect                     Detected language: English (en) — confidence: 12% below threshold 81% — falling back to English
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PROMPT_CHECK
🟢 Cache build Jaccard Banned     Built Jaccard n-gram cache with 139 entries for language english
🟢 BM25 Scorer Cache              Built BM25 banlist cache with 139 entries for language english
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 1522.66it/s]
🟢 Cache build Keybert Scorer     Built KeyBert embeddings cache with 81 entries
🟢 KeyWrdChk Depth                0 algos passed threshold vs. required 2
🟢 KeyWrdChk Breadth              0 algos had a score vs. required 3
🔵 TokenBudget                    Detected context_length=131072 for llama-guard3:8b via ollama adapter
🔵 TokenBudget                    Model 'llama-guard3:8b' reports 131072 tokens; capped to 32768 (TOKEN_BUDGET_CONTEXT_CAP)
🔵 TokenBudget                    [llama-guard3:8b] context=32768 reserved_sys=64 prompt≈6 → max_output_tokens=64
🔵 TokenBudget                    [llama-guard3:8b] num_ctx=32768 prompt≈6 → num_predict=64
🔵 Call LLM                       Model: llama-guard3:8b prompt template: _PROMPT_CHECK_CHAT_LLAMA_GUARD stage: Check provided prompt
🔵 Call LLM                       options: {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'num_predict': 64, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: llama-guard3:8b took 00:35
🟢 CheckPrompt                    Provided prompt is considered compliant by: llama-guard3:8b. Reason: Prompt classified as safe
🔵                                Chatter RAG Query LMM: mistral:7b
🟢 Chroma Collection              Using Chroma DB collection Test
🔵 VectorStore                    Set Chroma vector store. Name: Test Path: D:\RAG-LCC\chromadb\docs\Test
🔵 UserQuery                      Original user query: 'what  is the hedgehog diet'
🔵 LangDetect                     Detected language: English (en) — confidence: 12% below threshold 81% — falling back to English
🔵 QueryRewrite                   No conversation history — skipping rewrite
🔵 LangDetect                     Detected language: English (en) — confidence: 12% below threshold 81% — falling back to English
🔵 FinalQuery                     Final query for retrieval: 'what  is the hedgehog diet' (unchanged)
🔵 TokenBudget                    Authoritative budget=2048 vs caller estimate=256; using authoritative value
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈215 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_QUERY_EXPAND stage: Multi-query expansion
🔵 Call LLM                       options: {'temperature': 0.5, 'top_k': 40, 'top_p': 0.95, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:11
   ⚪ MultiQuery                     Alternate queries (3): 1: "Hedgehog's food intake" | 2: 'Nutritional habits of hedgehogs' | 3: 'What does a hedgehog eat?'
🔵 Chroma                         Querying Chroma DB on vector store D:\RAG-LCC\chromadb\docs\Test
   ⚪ Chroma                            Pos   ChromaScore  ChromaSim  Distance         Retrievers   File
   ⚪ Chroma                         ------------------------------------------------------------------------------------------
   ⚪ Chroma                              1        0.6189     0.3811    0.3811             Vector   Hedgehogs.pdf
   ⚪ Chroma                              2        0.5571     0.4429    0.4429             Vector   Hedgehogs.pdf
   ⚪ Chroma                              3        0.4564     0.5436    0.5436             Vector   Hedgehogs.pdf
   ⚪ Chroma                              4        0.4490     0.5510    0.5510             Vector   Hedgehogs.pdf
   ⚪ Chroma                              5        0.4009     0.5991    0.5991             Vector   Hedgehogs.pdf
   ⚪ Chroma                              6        0.3191     0.6809    0.6809             Vector   Cats.md
   ⚪ Chroma                              7        0.3109     0.6891    0.6891             Vector   Cats.md
   ⚪ Chroma                              8        0.3049     0.6951    0.6951             Vector   Kamele.txt
   ⚪ Chroma                              9        0.3027     0.6973    0.6973             Vector   Lions.pptx
   ⚪ Chroma                             10        0.2722     0.7278    0.7278             Vector   Cats.md
   ⚪ Chroma                             11        0.2677     0.7323    0.7323             Vector   Cats.md
   ⚪ Chroma                             12        0.2539     0.7461    0.7461             Vector   Cats.md
   ⚪ Chroma                             13        0.2424     0.7576    0.7576             Vector   Cats.md
   ⚪ Chroma                             14        0.2409     0.7591    0.7591             Vector   Kamele.txt
   ⚪ Chroma                             15        0.2378     0.7622    0.7622             Vector   Cats.md
   ⚪ Chroma                             16        0.2373     0.7627    0.7627             Vector   Cats.md
   ⚪ Chroma                             17        0.2368     0.7632    0.7632             Vector   Kamele.txt
   ⚪ Chroma                             18        0.2356     0.7644    0.7644             Vector   Cats.md
   ⚪ Chroma                             19        0.2219     0.7781    0.7781             Vector   Cats.md
   ⚪ Chroma                             20        0.2092     0.7908    0.7908             Vector   Dogs.png
   ⚪ Chroma                             21        0.1973     0.8027    0.8027             Vector   Cats.md
   ⚪ Chroma                             22        0.1946     0.8054    0.8054             Vector   Hedgehogs.pdf
   ⚪ Chroma                             23        0.1915     0.8085    0.8085             Vector   Cats.md
   ⚪ Chroma                             24        0.1900     0.8100    0.8100             Vector   Cats.md
   ⚪ Chroma                             25        0.1876     0.8124    0.8124             Vector   Cats.md
   ⚪ Chroma                             26        0.1825     0.8175    0.8175             Vector   Cats.md
   ⚪ Chroma                             27        0.1772     0.8228    0.8228             Vector   Cats.md
   ⚪ Chroma                             28        0.1770     0.8230    0.8230             Vector   Cats.md
   ⚪ Chroma                             29        0.1748     0.8252    0.8252             Vector   Apes.docx
   ⚪ Chroma                             30        0.1692     0.8308    0.8308             Vector   Cats.md
   ⚪ Chroma                             31        0.1641     0.8359    0.8359             Vector   Cats.md
   ⚪ Chroma                             32        0.1563     0.8437    0.8437             Vector   Cats.md
   ⚪ Chroma                             33        0.1481     0.8519    0.8519             Vector   Elephants.jpg
   ⚪ Chroma                             34        0.1465     0.8535    0.8535             Vector   Cats.md
   ⚪ Chroma                             35        0.1459     0.8541    0.8541             Vector   Kamele.txt
   ⚪ Chroma                             36        0.1456     0.8544    0.8544             Vector   Dogs.png
   ⚪ Chroma                             37        0.1424     0.8576    0.8576             Vector   Fish.txt
   ⚪ Chroma                             38        0.1400     0.8600    0.8600             Vector   Cats.md
   ⚪ Chroma                             39        0.1389     0.8611    0.8611             Vector   Cats.md
   ⚪ Chroma                             40        0.1360     0.8640    0.8640             Vector   Cats.md
   ⚪ Chroma                             41        0.1355     0.8645    0.8645             Vector   Cats.md
   ⚪ Chroma                             42        0.1345     0.8655    0.8655             Vector   Lions.pptx
   ⚪ Chroma                             43        0.1330     0.8670    0.8670             Vector   Cats.md
   ⚪ Chroma                             44        0.1310     0.8690    0.8690             Vector   Cats.md
   ⚪ Chroma                             45        0.1293     0.8707    0.8707             Vector   Fish.txt
   ⚪ Chroma                             46        0.1241     0.8759    0.8759             Vector   Cats.md
   ⚪ Chroma                             47        0.1210     0.8790    0.8790             Vector   Cats.md
   ⚪ Chroma                             48        0.1167     0.8833    0.8833             Vector   Cats.md
   ⚪ Chroma                             49        0.1159     0.8841    0.8841             Vector   Fish.txt
   ⚪ Chroma                             50        0.1135     0.8865    0.8865             Vector   Cats.md
   ⚪ Chroma                             51        0.1112     0.8888    0.8888             Vector   Dogs.png
   ⚪ Chroma                             52        0.1106     0.8894    0.8894             Vector   Cats.md
   ⚪ Chroma                             53        0.1099     0.8901    0.8901             Vector   Kamele.txt
   ⚪ Chroma                             54        0.1046     0.8954    0.8954             Vector   Apes.docx
   ⚪ Chroma                             55        0.1026     0.8974    0.8974             Vector   Fish.txt
   ⚪ Chroma                             56        0.1020     0.8980    0.8980             Vector   Cats.md
   ⚪ Chroma                             57        0.0952     0.9048    0.9048             Vector   LionsAndApes.xlsx
   ⚪ Chroma                             58        0.0950     0.9050    0.9050             Vector   Kamele.txt
   ⚪ Chroma                             59        0.0918     0.9082    0.9082             Vector   Fish.txt
   ⚪ Chroma                             60        0.0878     0.9122    0.9122             Vector   Cats.md
   ⚪ Chroma                             61        0.0865     0.9135    0.9135             Vector   Pferde.pdf
   ⚪ Chroma                             62        0.0861     0.9139    0.9139             Vector   Lions.pptx
   ⚪ Chroma                             63        0.0763     0.9237    0.9237             Vector   Fish.txt
   ⚪ Chroma                             64        0.0744     0.9256    0.9256             Vector   Pferde.pdf
   ⚪ Chroma                             65        0.0692     0.9308    0.9308             Vector   Pferde.pdf
   ⚪ Chroma                             66        0.0660     0.9340    0.9340             Vector   Lions.pptx
   ⚪ Chroma                             67        0.0596     0.9404    0.9404             Vector   Apes.docx
   ⚪ Chroma                             68        0.0556     0.9444    0.9444             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             69        0.0530     0.9470    0.9470             Vector   Cats.md
   ⚪ Chroma                             70        0.0429     0.9571    0.9571             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             71        0.0408     0.9592    0.9592             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             72        0.0377     0.9623    0.9623             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             73        0.0353     0.9647    0.9647             Vector   Lions.pptx
   ⚪ Chroma                             74        0.0300     0.9700    0.9700             Vector   Apes.docx
   ⚪ Chroma                             75        0.0253     0.9747    0.9747             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             76        0.0208     0.9792    0.9792             Vector   BlazingFast_Workstation.md
   ⚪ Chroma                             77        0.0200     0.9800    0.9800             Vector   Apes.docx
   ⚪ Chroma                             78        0.0020     0.9980    0.9980             Vector   Apes.docx
   ⚪ Chroma                             79       -0.0143     1.0143    1.0143             Vector   Pferde.pdf
🟢 Chroma                         Querying Chroma DB query returned 79 chunks
🔵 BM25                           Querying BM25 index on collection Test
🟢 BM25                           Loaded persisted BM25 index (79 chunks, 2100 terms)
🟢 BM25                           BM25 retrieval returned 59 chunks
   ⚪ BM25                              Pos     BM25Score         Retrievers   File
   ⚪ BM25                           -----------------------------------------------------------------------
   ⚪ BM25                                1        5.5677               BM25   Hedgehogs.pdf
   ⚪ BM25                                2        4.2910               BM25   Hedgehogs.pdf
   ⚪ BM25                                3        4.0014               BM25   Lions.pptx
   ⚪ BM25                                4        2.8573               BM25   Cats.md
   ⚪ BM25                                5        2.8080               BM25   Cats.md
   ⚪ BM25                                6        2.7903               BM25   Fish.txt
   ⚪ BM25                                7        2.6267               BM25   BlazingFast_Workstation.md
   ⚪ BM25                                8        2.6267               BM25   Cats.md
   ⚪ BM25                                9        2.5801               BM25   Hedgehogs.pdf
   ⚪ BM25                               10        2.5445               BM25   Hedgehogs.pdf
   ⚪ BM25                               11        2.5432               BM25   BlazingFast_Workstation.md
   ⚪ BM25                               12        2.4661               BM25   Cats.md
   ⚪ BM25                               13        2.3264               BM25   Cats.md
   ⚪ BM25                               14        2.2321               BM25   Lions.pptx
   ⚪ BM25                               15        1.8504               BM25   Cats.md
   ⚪ BM25                               16        1.5815               BM25   Fish.txt
   ⚪ BM25                               17        0.5462               BM25   Apes.docx
   ⚪ BM25                               18        0.5193               BM25   Apes.docx
   ⚪ BM25                               19        0.5192               BM25   Lions.pptx
   ⚪ BM25                               20        0.5137               BM25   Cats.md
🔵 Graph                          Querying graph index on collection Test
🟢 Graph                          Loaded persisted graph index (79 chunks, 1163 entities)
🟢 Graph                          Graph retrieval returned 0 chunks
   ⚪ Graph                             Pos    GraphScore         Retrievers   File
   ⚪ Graph                          -----------------------------------------------------------------------
🔵 Web                            Querying web search...
🔵 Web                            Internet search started — backend: 'duckduckgo'  query: 'what  is the hedgehog diet'
🔵 Web                            Internet search completed — 10 result(s) returned.
🟢 Web                            Web search returned 10 results
   ⚪ Web                               Pos      WebScore         Retrievers   URL
   ⚪ Web                            -----------------------------------------------------------------------
   ⚪ Web                                 1        1.0000                Web   https://www.hedgehogstreet.org/about-hedgehogs/diet/
   ⚪ Web                                 2        0.9000                Web   https://www.petmd.com/exotic/what-do-hedgehogs-eat
   ⚪ Web                                 3        0.8000                Web   https://www.chewy.com/education/small-pet/hedgehog/what-do-hedgehogs-eat
   ⚪ Web                                 4        0.7000                Web   https://www.thesprucepets.com/what-do-hedgehogs-eat-4588705
   ⚪ Web                                 5        0.6000                Web   https://vcahospitals.com/know-your-pet/hedgehogs---feeding
   ⚪ Web                                 6        0.5000                Web   https://a-z-animals.com/animals/hedgehog/what-do-hedgehogs-eat/
   ⚪ Web                                 7        0.4000                Web   https://www.hedgehogworld.com/what-do-hedgehogs-eat/
   ⚪ Web                                 8        0.3000                Web   https://dorsethedgehogrescue.org/what-do-hedgehogs-eat/
   ⚪ Web                                 9        0.2000                Web   https://www.arkwildlife.co.uk/blogs/wildlife-guides/what-do-hedgehogs-eat-and-drink
   ⚪ Web                                10        0.1000                Web   https://www.woodlandtrust.org.uk/blog/2024/03/what-hedgehogs-eat/
   ⚪ WebPreFilter                   Pre-filtering 10 web result(s) — bm25_pre_filter=0.100, cosine_pre_filter=0.300
   ⚪ WebPreFilter                   After BM25 pre-filter: 10/10 kept
   ⚪ WebPF/BM25                     Status       Pos       Score   URL                                       Snippet
   ⚪ WebPF/BM25                     ----------------------------------------------------------------------------------------------------
   ⚪ WebPF/BM25                       KEPT         1      0.1771   https://www.hedgehogstreet.org/about-hed  1 month ago - Wild hedgehogs eat a wide range of natural foo
   ⚪ WebPF/BM25                       KEPT         2      0.1506   https://www.petmd.com/exotic/what-do-hed  March 18, 2024 - Hedgehogs are omnivores, but they need rela
   ⚪ WebPF/BM25                       KEPT         3      2.5657   https://www.chewy.com/education/small-pe  January 19, 2026 - Insects should be a big part of any hedge
   ⚪ WebPF/BM25                       KEPT         4      1.0687   https://www.thesprucepets.com/what-do-he  June 4, 2025 - She now works with a team of other experience
   ⚪ WebPF/BM25                       KEPT         5      0.1628   https://vcahospitals.com/know-your-pet/h  In the wild, hedgehogs eat a diverse selection of insects as
   ⚪ WebPF/BM25                       KEPT         6      0.7564   https://a-z-animals.com/animals/hedgehog  July 14, 2025 - Until recently, hedgehogs used to be conside
   ⚪ WebPF/BM25                       KEPT         7      4.2562   https://www.hedgehogworld.com/what-do-he  January 1, 2024 - From a very high level, the most important
   ⚪ WebPF/BM25                       KEPT         8      0.8545   https://dorsethedgehogrescue.org/what-do  The most important invertebrates in their diet are worms, be
   ⚪ WebPF/BM25                       KEPT         9      1.1338   https://www.arkwildlife.co.uk/blogs/wild  June 10, 2024 - Hedgehogs are opportunistic feeders and will
   ⚪ WebPF/BM25                       KEPT        10      2.3532   https://www.woodlandtrust.org.uk/blog/20  Insects and other invertebrates are the hedgehog’s main natu
   ⚪ WebPreFilter                   After cosine pre-filter: 10/10 kept
   ⚪ WebPF/Cosine                   Status       Pos       Score   URL                                       Snippet
   ⚪ WebPF/Cosine                   ----------------------------------------------------------------------------------------------------
   ⚪ WebPF/Cosine                     KEPT         1      0.7298   https://www.hedgehogstreet.org/about-hed  1 month ago - Wild hedgehogs eat a wide range of natural foo
   ⚪ WebPF/Cosine                     KEPT         2      0.6929   https://www.petmd.com/exotic/what-do-hed  March 18, 2024 - Hedgehogs are omnivores, but they need rela
   ⚪ WebPF/Cosine                     KEPT         3      0.6858   https://www.chewy.com/education/small-pe  January 19, 2026 - Insects should be a big part of any hedge
   ⚪ WebPF/Cosine                     KEPT         4      0.7871   https://www.thesprucepets.com/what-do-he  June 4, 2025 - She now works with a team of other experience
   ⚪ WebPF/Cosine                     KEPT         5      0.6921   https://vcahospitals.com/know-your-pet/h  In the wild, hedgehogs eat a diverse selection of insects as
   ⚪ WebPF/Cosine                     KEPT         6      0.6449   https://a-z-animals.com/animals/hedgehog  July 14, 2025 - Until recently, hedgehogs used to be conside
   ⚪ WebPF/Cosine                     KEPT         7      0.7529   https://www.hedgehogworld.com/what-do-he  January 1, 2024 - From a very high level, the most important
   ⚪ WebPF/Cosine                     KEPT         8      0.7149   https://dorsethedgehogrescue.org/what-do  The most important invertebrates in their diet are worms, be
   ⚪ WebPF/Cosine                     KEPT         9      0.6638   https://www.arkwildlife.co.uk/blogs/wild  June 10, 2024 - Hedgehogs are opportunistic feeders and will
   ⚪ WebPF/Cosine                     KEPT        10      0.8149   https://www.woodlandtrust.org.uk/blog/20  Insects and other invertebrates are the hedgehog’s main natu
🟢 Merge                          Reciprocal Rank Fusion (RRF) produced 79 local chunks
   ⚪ Merge                             Pos    RRFScore                     Retrievers  File
   ⚪ Merge                          ----------------------------------------------------------------------------------
   ⚪ Merge                               1      0.0328                    Vector,BM25  Hedgehogs.pdf
   ⚪ Merge                               2      0.0323                    Vector,BM25  Hedgehogs.pdf
   ⚪ Merge                               3      0.0304                    Vector,BM25  Lions.pptx
   ⚪ Merge                               4      0.0302                    Vector,BM25  Hedgehogs.pdf
   ⚪ Merge                               5      0.0299                    Vector,BM25  Hedgehogs.pdf
   ⚪ Merge                               6      0.0297                    Vector,BM25  Cats.md
   ⚪ Merge                               7      0.0270                    Vector,BM25  Cats.md
   ⚪ Merge                               8      0.0256                    Vector,BM25  Cats.md
   ⚪ Merge                               9      0.0256                    Vector,BM25  Cats.md
   ⚪ Merge                              10      0.0256                    Vector,BM25  Cats.md
   ⚪ Merge                              11      0.0250                    Vector,BM25  Cats.md
   ⚪ Merge                              12      0.0250                    Vector,BM25  Cats.md
   ⚪ Merge                              13      0.0247                    Vector,BM25  Cats.md
   ⚪ Merge                              14      0.0247                    Vector,BM25  Cats.md
   ⚪ Merge                              15      0.0246                    Vector,BM25  Cats.md
   ⚪ Merge                              16      0.0245                    Vector,BM25  Cats.md
   ⚪ Merge                              17      0.0243                    Vector,BM25  Cats.md
   ⚪ Merge                              18      0.0241                    Vector,BM25  Cats.md
   ⚪ Merge                              19      0.0241                    Vector,BM25  Apes.docx
   ⚪ Merge                              20      0.0240                    Vector,BM25  Cats.md
   ⚪ Merge                              21      0.0238                    Vector,BM25  Fish.txt
   ⚪ Merge                              22      0.0238                    Vector,BM25  Cats.md
   ⚪ Merge                              23      0.0235                    Vector,BM25  Cats.md
   ⚪ Merge                              24      0.0235                    Vector,BM25  Cats.md
   ⚪ Merge                              25      0.0232                    Vector,BM25  Cats.md
   ⚪ Merge                              26      0.0227                    Vector,BM25  BlazingFast_Workstation.md
   ⚪ Merge                              27      0.0225                    Vector,BM25  Lions.pptx
   ⚪ Merge                              28      0.0223                    Vector,BM25  Fish.txt
   ⚪ Merge                              29      0.0223                    Vector,BM25  Cats.md
   ⚪ Merge                              30      0.0217                    Vector,BM25  BlazingFast_Workstation.md
   ⚪ Merge                              31      0.0216                    Vector,BM25  Cats.md
   ⚪ Merge                              32      0.0212                    Vector,BM25  Hedgehogs.pdf
   ⚪ Merge                              33      0.0211                    Vector,BM25  Cats.md
   ⚪ Merge                              34      0.0210                    Vector,BM25  Lions.pptx
   ⚪ Merge                              35      0.0210                    Vector,BM25  Cats.md
   ⚪ Merge                              36      0.0206                    Vector,BM25  Cats.md
   ⚪ Merge                              37      0.0205                    Vector,BM25  Cats.md
   ⚪ Merge                              38      0.0203                    Vector,BM25  Lions.pptx
   ⚪ Merge                              39      0.0202                    Vector,BM25  Apes.docx
   ⚪ Merge                              40      0.0200                    Vector,BM25  Cats.md
   ⚪ Merge                              41      0.0198                    Vector,BM25  Cats.md
   ⚪ Merge                              42      0.0198                    Vector,BM25  Cats.md
   ⚪ Merge                              43      0.0196                    Vector,BM25  Apes.docx
   ⚪ Merge                              44      0.0196                    Vector,BM25  Fish.txt
   ⚪ Merge                              45      0.0192                    Vector,BM25  Cats.md
   ⚪ Merge                              46      0.0191                    Vector,BM25  Cats.md
   ⚪ Merge                              47      0.0190                    Vector,BM25  Cats.md
   ⚪ Merge                              48      0.0190                    Vector,BM25  Fish.txt
   ⚪ Merge                              49      0.0190                    Vector,BM25  Cats.md
   ⚪ Merge                              50      0.0185                    Vector,BM25  Cats.md
   ⚪ Merge                              51      0.0184                    Vector,BM25  Fish.txt
   ⚪ Merge                              52      0.0182                    Vector,BM25  Apes.docx
   ⚪ Merge                              53      0.0179                    Vector,BM25  Cats.md
   ⚪ Merge                              54      0.0178                    Vector,BM25  Cats.md
   ⚪ Merge                              55      0.0174                    Vector,BM25  Apes.docx
   ⚪ Merge                              56      0.0167                    Vector,BM25  Lions.pptx
   ⚪ Merge                              57      0.0164                    Vector,BM25  Cats.md
   ⚪ Merge                              58      0.0160                    Vector,BM25  BlazingFast_Workstation.md
   ⚪ Merge                              59      0.0159                    Vector,BM25  BlazingFast_Workstation.md
   ⚪ Merge                              60      0.0156                         Vector  Hedgehogs.pdf
   ⚪ Merge                              61      0.0147                         Vector  Kamele.txt
   ⚪ Merge                              62      0.0135                         Vector  Kamele.txt
   ⚪ Merge                              63      0.0130                         Vector  Kamele.txt
   ⚪ Merge                              64      0.0125                         Vector  Dogs.png
   ⚪ Merge                              65      0.0108                         Vector  Elephants.jpg
   ⚪ Merge                              66      0.0105                         Vector  Kamele.txt
   ⚪ Merge                              67      0.0104                         Vector  Dogs.png
   ⚪ Merge                              68      0.0090                         Vector  Dogs.png
   ⚪ Merge                              69      0.0088                         Vector  Kamele.txt
   ⚪ Merge                              70      0.0085                         Vector  LionsAndApes.xlsx
   ⚪ Merge                              71      0.0085                         Vector  Kamele.txt
   ⚪ Merge                              72      0.0083                         Vector  Pferde.pdf
   ⚪ Merge                              73      0.0081                         Vector  Fish.txt
   ⚪ Merge                              74      0.0081                         Vector  Pferde.pdf
   ⚪ Merge                              75      0.0080                         Vector  Pferde.pdf
   ⚪ Merge                              76      0.0077                         Vector  BlazingFast_Workstation.md
   ⚪ Merge                              77      0.0074                         Vector  BlazingFast_Workstation.md
   ⚪ Merge                              78      0.0073                         Vector  Apes.docx
   ⚪ Merge                              79      0.0072                         Vector  Pferde.pdf
🟢 Merge                          Appended 10 web result(s) → reranker pool: 89 chunks
🔵 ChunkDedup                     Removed 1 near-duplicate chunk(s) (threshold=0.85, kept 88)
   ⚪ Rerank                            Pos    RawScore    AdjScore         Retrievers  File                                      Text
   ⚪ Rerank                         --------------------------------------------------------------------------------------------------------------
   ⚪ Rerank                              1      3.7515      0.9518        Vector,BM25  Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
   ⚪ Rerank                              2      2.6117      0.8622        Vector,BM25  Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
   ⚪ Rerank                              3      0.7378      0.7150             Vector  Kamele.txt                                ernährung kamele sind pflanzenfresser (h
   ⚪ Rerank                              4     -1.3736      0.5491             Vector  Kamele.txt                                sie liefern milch, fleisch, wolle und le
   ⚪ Rerank                              5     -1.5884      0.5322             Vector  Kamele.txt                                kamele – überlebenskünstler der wüste ka
   ⚪ Rerank                              6     -1.8131      0.5145        Vector,BM25  Apes.docx                                 Gorillas are the largest living primates
   ⚪ Rerank                              7     -1.8529      0.5114        Vector,BM25  Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
   ⚪ Rerank                              8     -1.9144      0.5066        Vector,BM25  Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
   ⚪ Rerank                              9      4.3651      0.5000                Web  dorsethedgehogrescue.org                  [S] The most important invertebrates in
   ⚪ Rerank                             10      3.8494      0.4797                Web  www.thesprucepets.com                     [S] June 4, 2025 - She now works with a
   ⚪ Rerank                             11      3.8372      0.4793                Web  vcahospitals.com                          [S] In the wild, hedgehogs eat a diverse
   ⚪ Rerank                             12     -2.3301      0.4739             Vector  Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
   ⚪ Rerank                             13     -2.3998      0.4684             Vector  Elephants.jpg                             elephants large, long-lived mammals fami
   ⚪ Rerank                             14      3.0235      0.4473                Web  www.hedgehogstreet.org                    [S] 1 month ago - Wild hedgehogs eat a w
   ⚪ Rerank                             15     -2.8906      0.4299        Vector,BM25  Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
   ⚪ Rerank                             16      2.5243      0.4277                Web  www.woodlandtrust.org.uk                  [S] Insects and other invertebrates are
   ⚪ Rerank                             17     -2.9708      0.4236             Vector  Kamele.txt                                in einigen kulturen spielen kamele auch
   ⚪ Rerank                             18      2.2764      0.4179                Web  www.chewy.com                             [S] January 19, 2026 - Insects should be
   ⚪ Rerank                             19      2.2717      0.4178                Web  www.arkwildlife.co.uk                     [S] June 10, 2024 - Hedgehogs are opport
   ⚪ Rerank                             20     -3.0971      0.4136        Vector,BM25  Cats.md                                   whiskers are deeply rooted sensory hairs
   ⚪ Rerank                             21     -3.1091      0.4127        Vector,BM25  Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
   ⚪ Rerank                             22     -3.1286      0.4112             Vector  Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
   ⚪ Rerank                             23      1.2747      0.3786                Web  a-z-animals.com                           [S] July 14, 2025 - Until recently, hedg
   ⚪ Rerank                             24     -3.5498      0.3781        Vector,BM25  Apes.docx                                 Orangutans are the only great apes found
   ⚪ Rerank                             25     -3.6941      0.3667             Vector  Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
   ⚪ Rerank                             26     -3.7396      0.3632        Vector,BM25  Cats.md                                   cats are crepuscular hunters with: - a h
   ⚪ Rerank                             27     -3.8559      0.3540             Vector  Kamele.txt                                weitere körperliche anpassungen: - dicht
   ⚪ Rerank                             28     -3.9675      0.3453        Vector,BM25  Cats.md                                   free-roaming cats are estimated to kill
   ⚪ Rerank                             29     -3.9706      0.3450        Vector,BM25  Fish.txt                                   what fish are fish are a diverse group
   ⚪ Rerank                             30     -4.0397      0.3396             Vector  Apes.docx                                 Chimpanzees (Pan troglodytes) are found
   ⚪ Rerank                             31     -4.0582      0.3381             Vector  Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
   ⚪ Rerank                             32      0.2056      0.3366                Web  www.petmd.com                             [S] March 18, 2024 - Hedgehogs are omniv
   ⚪ Rerank                             33     -4.1045      0.3345        Vector,BM25  Cats.md                                   cats require: - taurine - arachidonic ac
   ⚪ Rerank                             34     -4.1543      0.3306        Vector,BM25  Fish.txt                                  ecological roles and importance fish are
   ⚪ Rerank                             35     -4.3374      0.3162             Vector  Dogs.png                                  tasks. understanding breed-specific ende
   ⚪ Rerank                             36     -0.5064      0.3086                Web  www.hedgehogworld.com                     [S] January 1, 2024 - From a very high l
   ⚪ Rerank                             37     -4.4591      0.3066        Vector,BM25  Apes.docx                                 All great ape species are either endange
   ⚪ Rerank                             38     -4.5073      0.3028        Vector,BM25  Cats.md                                   as placental mammals, cats give live bir
   ⚪ Rerank                             39     -4.5483      0.2996        Vector,BM25  Lions.pptx                                Slide 2: Physical Characteristics Male l
   ⚪ Rerank                             40     -4.6883      0.2886        Vector,BM25  Cats.md                                   cats can detect frequencies up to ~64 kh
   ⚪ Rerank                             41     -4.7271      0.2856             Vector  Fish.txt                                  poikilotherm: a term describing organism
   ⚪ Rerank                             42     -4.7457      0.2841        Vector,BM25  Fish.txt                                  diversity: fish include jawless fishes (
   ⚪ Rerank                             43     -4.7467      0.2840        Vector,BM25  Cats.md                                   cats have lived alongside humans for tho
   ⚪ Rerank                             44     -4.7871      0.2808        Vector,BM25  Cats.md                                   indoor cats often live 12–18 years; some
   ⚪ Rerank                             45     -4.9049      0.2716        Vector,BM25  Cats.md                                   cats suffered during periods of supersti
   ⚪ Rerank                             46     -5.0079      0.2635        Vector,BM25  Apes.docx                                 All great apes demonstrate remarkable co
   ⚪ Rerank                             47     -5.0134      0.2631        Vector,BM25  BlazingFast_Workstation.md                the blazingfast workstation is an enterp
   ⚪ Rerank                             48     -5.1025      0.2561             Vector  Dogs.png                                  , comprehensive text dogs dogs accompani
   ⚪ Rerank                             49     -5.1038      0.2560        Vector,BM25  Cats.md                                   a deep, structured exploration of domest
   ⚪ Rerank                             50     -5.1231      0.2545        Vector,BM25  Apes.docx                                 Great apes (Hominidae) are the closest l
🔵 Rerank                         Reranking with cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 returned 88 chunks
🔵 Chunk selection                Strategy 'DEFAULT' → ScoreRankedSelector
🔵 Rerank select                  logit+ = effective logit after single-chunk boost (raw in raw_rerank_score)
🟣 Rerank select                        Logit [Sigmoid]       Thr      ΔProb         Retrievers  File                                      Text
🟣 Rerank select                  --------------------------------------------------------------------------------------------------------------------
🟣 Rerank select                   ❌   -8.3614 [0.000]  (0.7000)    -0.6998             Vector  BlazingFast_Workstation.md                - **os support**: certified for ubuntu 2
🟣 Rerank select                   ❌   -8.2327 [0.000]  (0.7000)    -0.6997        Vector,BM25  BlazingFast_Workstation.md                - **graphics**: accommodates up to quad-
🟣 Rerank select                   ❌   -7.7304 [0.000]  (0.7000)    -0.6996        Vector,BM25  BlazingFast_Workstation.md                - **processors**: dual enterprise platin
🟣 Rerank select                   ❌   -7.4892 [0.001]  (0.7000)    -0.6994        Vector,BM25  Cats.md                                   cats dominate digital culture: - memes (
🟣 Rerank select                   ❌   -7.4751 [0.001]  (0.7000)    -0.6994             Vector  BlazingFast_Workstation.md                - **storage subsystem**: 8x 4tb nvme u.2
🟣 Rerank select                   ❌   -7.0086 [0.001]  (0.7000)    -0.6991        Vector,BM25  Cats.md                                   - climbing structures - scratching posts
🟣 Rerank select                   ❌   -6.8646 [0.001]  (0.7000)    -0.6990        Vector,BM25  Cats.md                                   - free feeding can lead to obesity. - pu
🟣 Rerank select                   ❌   -6.8599 [0.001]  (0.7000)    -0.6990        Vector,BM25  Cats.md                                   - puzzle toys - training sessions - nove
🟣 Rerank select                   ❌   -6.8034 [0.001]  (0.7000)    -0.6989        Vector,BM25  BlazingFast_Workstation.md                - **power delivery**: dual 2000w redunda
🟣 Rerank select                   ❌   -6.7551 [0.001]  (0.7000)    -0.6988        Vector,BM25  Cats.md                                   - dental disease - kidney disease - hype
🟣 Rerank select                   ❌   -6.5794 [0.001]  (0.7000)    -0.6986        Vector,BM25  Cats.md                                   outdoor access: - pros: exercise, stimul
🟣 Rerank select                   ❌   -6.4369 [0.002]  (0.7000)    -0.6984             Vector  Pferde.pdf                                Page 2 KI-Modelle, die semantische Feinh
🟣 Rerank select                   ❌   -6.4280 [0.002]  (0.7000)    -0.6984        Vector,BM25  Cats.md                                   cats form strong bonds but express affec
🟣 Rerank select                   ❌   -6.2086 [0.002]  (0.7000)    -0.6980        Vector,BM25  Cats.md                                   most adult cats are lactose intolerant.
🟣 Rerank select                   ❌   -6.2034+[0.002]  (0.7000)    -0.6980             Vector  LionsAndApes.xlsx                         category detail value taxonomy scientifi
🟣 Rerank select                   ❌   -6.1860 [0.002]  (0.7000)    -0.6979        Vector,BM25  Cats.md                                   - human interaction - multi-cat househol
🟣 Rerank select                   ❌   -6.1783 [0.002]  (0.7000)    -0.6979        Vector,BM25  Cats.md                                   cats communicate through: - vocalization
🟣 Rerank select                   ❌   -6.1340 [0.002]  (0.7000)    -0.6978        Vector,BM25  Hedgehogs.pdf                             Page 2 behavioral ecology, hibernation p
🟣 Rerank select                   ❌   -6.0615 [0.002]  (0.7000)    -0.6977        Vector,BM25  Cats.md                                   cats were not bred for specific tasks ea
🟣 Rerank select                   ❌   -6.0457 [0.002]  (0.7000)    -0.6976        Vector,BM25  Lions.pptx                                Slide 1: Lions: The King of the Savanna
🟣 Rerank select                   ❌   -5.9882 [0.003]  (0.7000)    -0.6975        Vector,BM25  Cats.md                                   selective breeding intensified in the 19
🟣 Rerank select                   ❌   -5.9332 [0.003]  (0.7000)    -0.6974        Vector,BM25  Cats.md                                   modern domestic cats descend from the ne
🟣 Rerank select                   ❌   -5.9216 [0.003]  (0.7000)    -0.6973        Vector,BM25  Cats.md                                   kittens engage in: - stalking. - pouncin
🟣 Rerank select                   ❌   -5.8964 [0.003]  (0.7000)    -0.6973        Vector,BM25  Fish.txt                                  terminology explained cold-blooded: a co
🟣 Rerank select                   ❌   -5.7050 [0.003]  (0.7000)    -0.6967        Vector,BM25  Cats.md                                   - kingdom: animalia - phylum: chordata -
🟣 Rerank select                   ❌   -5.6500 [0.004]  (0.7000)    -0.6965        Vector,BM25  Cats.md                                   cats are often described as solitary, bu
🟣 Rerank select                   ❌   -5.6269 [0.004]  (0.7000)    -0.6964        Vector,BM25  Lions.pptx                                Slide 4: Conservation Status Lions are c
🟣 Rerank select                   ❌   -5.5806 [0.004]  (0.7000)    -0.6962        Vector,BM25  Cats.md                                   the mechanism of purring is still debate
🟣 Rerank select                   ❌   -5.5803 [0.004]  (0.7000)    -0.6962             Vector  Dogs.png                                  impact dogs human well-being. despite ma
🟣 Rerank select                   ❌   -5.5781 [0.004]  (0.7000)    -0.6962        Vector,BM25  Cats.md                                   cats were revered, often associated with
🟣 Rerank select                   ❌   -5.5372 [0.004]  (0.7000)    -0.6961        Vector,BM25  Cats.md                                   these breeds emerged without heavy human
🟣 Rerank select                   ❌   -5.4527 [0.004]  (0.7000)    -0.6957        Vector,BM25  Fish.txt                                  ectotherm: the preferred scientific term
🟣 Rerank select                   ❌   -5.4373 [0.004]  (0.7000)    -0.6957        Vector,BM25  Cats.md                                   they have a righting reflex, but falls f
🟣 Rerank select                   ❌   -5.4204 [0.004]  (0.7000)    -0.6956        Vector,BM25  Cats.md                                   - regular veterinary checkups - vaccinat
🟣 Rerank select                   ❌   -5.3809 [0.005]  (0.7000)    -0.6954        Vector,BM25  Cats.md                                   some breeds have known predispositions:
🟣 Rerank select                   ❌   -5.3673 [0.005]  (0.7000)    -0.6954        Vector,BM25  Cats.md                                   spaying/neutering is essential to reduce
🟣 Rerank select                   ❌   -5.3414 [0.005]  (0.7000)    -0.6952        Vector,BM25  Cats.md                                   even well-fed cats hunt. the sequence: 1
🟣 Rerank select                   ❌   -5.2313 [0.005]  (0.7000)    -0.6947        Vector,BM25  Cats.md                                   cats are complex, adaptable, and endless
🟣 Rerank select                   ❌   -5.1231 [0.006]  (0.7000)    -0.6941        Vector,BM25  Apes.docx                                 Great apes (Hominidae) are the closest l
🟣 Rerank select                   ❌   -5.1038 [0.006]  (0.7000)    -0.6940        Vector,BM25  Cats.md                                   a deep, structured exploration of domest
🟣 Rerank select                   ❌   -5.1025 [0.006]  (0.7000)    -0.6940             Vector  Dogs.png                                  , comprehensive text dogs dogs accompani
🟣 Rerank select                   ❌   -5.0134 [0.007]  (0.7000)    -0.6934        Vector,BM25  BlazingFast_Workstation.md                the blazingfast workstation is an enterp
🟣 Rerank select                   ❌   -5.0079 [0.007]  (0.7000)    -0.6934        Vector,BM25  Apes.docx                                 All great apes demonstrate remarkable co
🟣 Rerank select                   ❌   -4.9049 [0.007]  (0.7000)    -0.6926        Vector,BM25  Cats.md                                   cats suffered during periods of supersti
🟣 Rerank select                   ❌   -4.7871 [0.008]  (0.7000)    -0.6917        Vector,BM25  Cats.md                                   indoor cats often live 12–18 years; some
🟣 Rerank select                   ❌   -4.7467 [0.009]  (0.7000)    -0.6914        Vector,BM25  Cats.md                                   cats have lived alongside humans for tho
🟣 Rerank select                   ❌   -4.7457 [0.009]  (0.7000)    -0.6914        Vector,BM25  Fish.txt                                  diversity: fish include jawless fishes (
🟣 Rerank select                   ❌   -4.7271 [0.009]  (0.7000)    -0.6912             Vector  Fish.txt                                  poikilotherm: a term describing organism
🟣 Rerank select                   ❌   -4.6883 [0.009]  (0.7000)    -0.6909        Vector,BM25  Cats.md                                   cats can detect frequencies up to ~64 kh
🟣 Rerank select                   ❌   -4.5483 [0.010]  (0.7000)    -0.6895        Vector,BM25  Lions.pptx                                Slide 2: Physical Characteristics Male l
🟣 Rerank select                   ❌   -4.5073 [0.011]  (0.7000)    -0.6891        Vector,BM25  Cats.md                                   as placental mammals, cats give live bir
🟣 Rerank select                   ❌   -4.4591 [0.011]  (0.7000)    -0.6886        Vector,BM25  Apes.docx                                 All great ape species are either endange
🟣 Rerank select                   ❌   -4.3374 [0.013]  (0.7000)    -0.6871             Vector  Dogs.png                                  tasks. understanding breed-specific ende
🟣 Rerank select                   ❌   -4.1543 [0.015]  (0.7000)    -0.6845        Vector,BM25  Fish.txt                                  ecological roles and importance fish are
🟣 Rerank select                   ❌   -4.1045 [0.016]  (0.7000)    -0.6838        Vector,BM25  Cats.md                                   cats require: - taurine - arachidonic ac
🟣 Rerank select                   ❌   -4.0582 [0.017]  (0.7000)    -0.6830             Vector  Pferde.pdf                                Page 1 Schließlich lohnt sich ein Blick
🟣 Rerank select                   ❌   -4.0397 [0.017]  (0.7000)    -0.6827             Vector  Apes.docx                                 Chimpanzees (Pan troglodytes) are found
🟣 Rerank select                   ❌   -3.9706 [0.019]  (0.7000)    -0.6815        Vector,BM25  Fish.txt                                   what fish are fish are a diverse group
🟣 Rerank select                   ❌   -3.9675 [0.019]  (0.7000)    -0.6814        Vector,BM25  Cats.md                                   free-roaming cats are estimated to kill
🟣 Rerank select                   ❌   -3.8559 [0.021]  (0.7000)    -0.6793             Vector  Kamele.txt                                weitere körperliche anpassungen: - dicht
🟣 Rerank select                   ❌   -3.7396 [0.023]  (0.7000)    -0.6768        Vector,BM25  Cats.md                                   cats are crepuscular hunters with: - a h
🟣 Rerank select                   ❌   -3.6941 [0.024]  (0.7000)    -0.6757             Vector  Pferde.pdf                                Page 1 Stimmungen zu vermitteln. Für Men
🟣 Rerank select                   ❌   -3.5498 [0.028]  (0.7000)    -0.6721        Vector,BM25  Apes.docx                                 Orangutans are the only great apes found
🟣 Rerank select                   ❌   -3.1286 [0.042]  (0.7000)    -0.6581             Vector  Pferde.pdf                                Page 1 Erweiterter deutscher Testtext üb
🟣 Rerank select                   ❌   -3.1091 [0.043]  (0.7000)    -0.6573        Vector,BM25  Lions.pptx                                Slide 5: Lion Cubs and Reproduction Gest
🟣 Rerank select                   ❌   -3.0971 [0.043]  (0.7000)    -0.6568        Vector,BM25  Cats.md                                   whiskers are deeply rooted sensory hairs
🟣 Rerank select                   ❌   -2.9708 [0.049]  (0.7000)    -0.6512             Vector  Kamele.txt                                in einigen kulturen spielen kamele auch
🟣 Rerank select                   ❌   -2.8906 [0.053]  (0.7000)    -0.6474        Vector,BM25  Hedgehogs.pdf                             Page 2 areas, and garden hazards—such as
🟣 Rerank select                   ❌   -2.3301 [0.089]  (0.7000)    -0.6113             Vector  Hedgehogs.pdf                             Page 2 detect prey and a rapid, decisive
🟣 Rerank select                   ❌   -2.1767+[0.102]  (0.7000)    -0.5981             Vector  Elephants.jpg                             elephants large, long-lived mammals fami
🟣 Rerank select                   ❌   -1.9144 [0.128]  (0.7000)    -0.5715        Vector,BM25  Lions.pptx                                Slide 3: Hunting and Diet Lions are coop
🟣 Rerank select                   ❌   -1.8529 [0.136]  (0.7000)    -0.5645        Vector,BM25  Hedgehogs.pdf                             Page 1 tissues. Sensory adaptations incl
🟣 Rerank select                   ❌   -1.8131 [0.140]  (0.7000)    -0.5597        Vector,BM25  Apes.docx                                 Gorillas are the largest living primates
🟣 Rerank select                   ❌   -1.5884 [0.170]  (0.7000)    -0.5304             Vector  Kamele.txt                                kamele – überlebenskünstler der wüste ka
🟣 Rerank select                   ❌   -1.3736 [0.202]  (0.7000)    -0.4980             Vector  Kamele.txt                                sie liefern milch, fleisch, wolle und le
🟣 Rerank select                   ❌   -0.5064 [0.376]  (0.5000)    -0.1240                Web  www.hedgehogworld.com                     January 1, 2024 - From a very high level
🟣 Rerank select                   ❌    0.7378 [0.677]  (0.7000)    -0.0235             Vector  Kamele.txt                                ernährung kamele sind pflanzenfresser (h
🟣 Rerank select                   ✅    4.3651 [0.987]  (0.5000)    +0.4874                Web  dorsethedgehogrescue.org                  The most important invertebrates in thei
🟣 Rerank select                   ✅    3.8494 [0.979]  (0.5000)    +0.4792                Web  www.thesprucepets.com                     June 4, 2025 - She now works with a team
🟣 Rerank select                   ✅    3.8372 [0.979]  (0.5000)    +0.4789                Web  vcahospitals.com                          In the wild, hedgehogs eat a diverse sel
🟣 Rerank select                   ✅    3.7515 [0.977]  (0.7000)    +0.2771        Vector,BM25  Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are s
🟣 Rerank select                   ✅    3.0235 [0.954]  (0.5000)    +0.4536                Web  www.hedgehogstreet.org                    1 month ago - Wild hedgehogs eat a wide
🟣 Rerank select                   ✅    2.6117 [0.932]  (0.7000)    +0.2316        Vector,BM25  Hedgehogs.pdf                             Page 1 suburban gardens and agricultural
🟣 Rerank select                   ✅    2.5243 [0.926]  (0.5000)    +0.4258                Web  www.woodlandtrust.org.uk                  Insects and other invertebrates are the
🟣 Rerank select                   ✅    2.2764 [0.907]  (0.5000)    +0.4069                Web  www.chewy.com                             January 19, 2026 - Insects should be a b
🟣 Rerank select                   ✅    2.2717 [0.907]  (0.5000)    +0.4065                Web  www.arkwildlife.co.uk                     June 10, 2024 - Hedgehogs are opportunis
🟣 Rerank select                   ✅    1.2747 [0.782]  (0.5000)    +0.2815                Web  a-z-animals.com                           July 14, 2025 - Until recently, hedgehog
🟣 Rerank select                   ✅    0.2056 [0.551]  (0.5000)    +0.0512                Web  www.petmd.com                             March 18, 2024 - Hedgehogs are omnivores
🔵 Strategy: default              11 chunks remain after applying sigmoid(raw logit) ≥ threshold  local=0.7000  web=0.5000  single-chunk boost ×1.25
🔵 Chunk selector: Score          Ranked     Selected 11 chunks.
🔵 Selected                       11 chunks selected
   ⚪ Selected                        #   Score [Sigmoid]  File                                      Text
   ⚪ Selected                       --------------------------------------------------------------------------------------------------------------------------------------------
   ⚪ Selected                        1    0.9518 [0.977]  Hedgehogs.pdf                             Page 1 Hedgehog Overview Hedgehogs are small, nocturnal mammals known
   ⚪ Selected                        2    0.8622 [0.932]  Hedgehogs.pdf                             Page 1 suburban gardens and agricultural edges. They are native to muc
   ⚪ Selected                        3    0.5000 [0.987]  dorsethedgehogrescue.org                  The most important invertebrates in their diet are worms, beetles, slu
   ⚪ Selected                        4    0.4797 [0.979]  www.thesprucepets.com                     June 4, 2025 - She now works with a team of other experienced vets to
   ⚪ Selected                        5    0.4793 [0.979]  vcahospitals.com                          In the wild, hedgehogs eat a diverse selection of insects as well as s
   ⚪ Selected                        6    0.4473 [0.954]  www.hedgehogstreet.org                    1 month ago - Wild hedgehogs eat a wide range of natural foods, but th
   ⚪ Selected                        7    0.4277 [0.926]  www.woodlandtrust.org.uk                  Insects and other invertebrates are the hedgehog’s main natural food s
   ⚪ Selected                        8    0.4179 [0.907]  www.chewy.com                             January 19, 2026 - Insects should be a big part of any hedgehog’s diet
   ⚪ Selected                        9    0.4178 [0.907]  www.arkwildlife.co.uk                     June 10, 2024 - Hedgehogs are opportunistic feeders and will eat frogs
   ⚪ Selected                       10    0.3786 [0.782]  a-z-animals.com                           July 14, 2025 - Until recently, hedgehogs used to be considered insect
   ⚪ Selected                       11    0.3366 [0.551]  www.petmd.com                             March 18, 2024 - Hedgehogs are omnivores, but they need relatively hig
   ⚪ ChunkSelect                    After chunk selection: 11/88 kept
🔵 TokenBudget                    [mistral:7b] context=32768 reserved_sys=1024 prompt≈3000 → max_output_tokens=2048
🔵 Resolved token params          max_output_tokens(api: max_tokens)=2048  num_ctx=32768  (override: max_tokens=None  num_ctx=None)
🔵 TokenBudget                    [mistral:7b] num_ctx=32768 prompt≈3000 → num_predict=2048
🔵 Call LLM                       Model: mistral:7b prompt template: _PROMPT_CHAT stage: Run user prompt
🔵 Call LLM                       options: {'temperature': 0.1, 'top_k': 40, 'top_p': 0.92, 'num_predict': 2048, 'num_ctx': 32768} streaming: False
🔵 Call LLM                       Elapsed time calling: mistral:7b took 00:12
🔵 LangDetect                     Detected language: English (en) — confidence: 100% (threshold: 60%)
🔵 HF                             Reusing cached embeddings for snowflake/snowflake-arctic-embed-l-v2.0 rev='None' device=cuda:0 dtype=torch.float32
🟢 Cache build Regex Banned       Built compiled Regex cache with 139 entries for language english stage: PIPELINE_CHECK
🟢 KeyWrdChk Depth                0 algos passed threshold vs. required 4
🟢 KeyWrdChk Breadth              1 algos had a score vs. required 4
🔵 Deleted chat context           Deleted chat context for MyFirstChat in Test_ChatContext
🔵 Add chat context               Upserted turn 1 for chat_name=MyFirstChat file_tag='' lang='english' to Test_ChatContext
🔵 Masker                         0 of 15 rules produced matches and were replaced
💡>   The context confirms that the diet of a hedgehog primarily consists of invertebrates such as worms, beetles, slugs, caterpillars, millipedes, and earwigs. In addition to
💡>  these, they may also consume amphibians, small rodents, eggs, carrion, and fallen fruit when available. Commercial foods specifically designed for hedgehogs can also be fed to
💡>  them, with a preference for products containing ample protein but little fat. Cat food can also be offered as an option.
💡>
💡>  ### Sources
💡>  - Hedgehogs.pdf (D:/RAG-LCC/TestDocs/Hedgehogs.pdf)
💡>  - dorsethedgehogrescue.org
💡>  - www.thesprucepets.com
💡>  - vcahospitals.com
💡>  - www.hedgehogstreet.org
💡>  - www.woodlandtrust.org.uk
💡>  - www.chewy.com
💡>  - www.arkwildlife.co.uk
💡>  - a-z-animals.com
💡>  - www.petmd.com
🟡 VisualMarker                   _mark_sources called: 11 chunk(s)
🔵 VisualMarker                   Prepared 1 highlighted document(s) (in memory)
🔵 Marked sources                 1 highlighted document(s) (click a link to open; use 'Save As' in the viewer to keep a copy):
   📎 Hedgehogs.pdf (highlighted): file:///D:/RAG-LCC/tmp/rag_marked_z9swvmrs/Hedgehogs_marked.pdf
🔵 Web sources                    9 web source(s):
   🌐 https://dorsethedgehogrescue.org/what-do-hedgehogs-eat/
   🌐 https://www.thesprucepets.com/what-do-hedgehogs-eat-4588705
   🌐 https://vcahospitals.com/know-your-pet/hedgehogs---feeding
   🌐 https://www.hedgehogstreet.org/about-hedgehogs/diet/
   🌐 https://www.woodlandtrust.org.uk/blog/2024/03/what-hedgehogs-eat/
   🌐 https://www.chewy.com/education/small-pet/hedgehog/what-do-hedgehogs-eat
   🌐 https://www.arkwildlife.co.uk/blogs/wildlife-guides/what-do-hedgehogs-eat-and-drink
   🌐 https://a-z-animals.com/animals/hedgehog/what-do-hedgehogs-eat/
   🌐 https://www.petmd.com/exotic/what-do-hedgehogs-eat
help? for help   show? for current values
key=value to set (e.g. strategy=default)   key! to pick (e.g. strategy!)   key- to unset (e.g. file-)   strategy*preset for quick defaults (e.g. strategy*narrow)
Press ↵ on an empty line to proceed to your query prompt
 🛠️  >
 ```
