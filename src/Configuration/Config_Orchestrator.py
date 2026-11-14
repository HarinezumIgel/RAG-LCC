"""Retrieval orchestration flow presets.

A *flow* is a named recipe that tells the orchestrator, for every user turn,
which version of the question to search with, which search engines to run and
which post-processing steps (rerank, rescue, grounding, ...) to apply.

Users pick one flow per session (CLI: ``orchestrator_flow!`` / API field
``orchestrator_flow``).  Individual knobs of the selected flow can still be
overridden per session afterwards (CLI ``<knob>=value``, API request fields).
Flows are defined only here; the GUI picker never invents new flows.

A longer, narrative version of these explanations, with experiment recipes,
is in ORCHESTRATION_FLOWS.md at the repository root.

===============================================================================
INTRODUCTION - the words used in this file
===============================================================================

Query versions (what text is sent to a search engine)
-----------------------------------------------------
Every turn produces up to three versions of the user's question.  The keys
main_query_source, secondary_query_source and rerank_query_source pick one of
them.  They are created in this order:

    user types --> ORIGINAL_QUERY
                     |  translate to the retrieval language (english mode)
                     v
                   TRANSLATED_QUERY
                     |  LLM rewrite with chat history (use_query_rewrite)
                     v
                   REWRITTEN_QUERY

ORIGINAL_QUERY
    Exactly what the user typed, in the user's language.  No pronoun
    resolution, no rewrite, no translation.
TRANSLATED_QUERY
    The ORIGINAL_QUERY translated to the *retrieval language*, BEFORE any
    rewrite.  With retrieval_language="native" nothing is translated, so it
    equals the ORIGINAL_QUERY.
REWRITTEN_QUERY
    The TRANSLATED_QUERY after the LLM *rewrite*.  The rewrite makes the
    question self-contained using the chat history: it performs *pronoun
    substitution* ("it" -> "the Eiffel Tower") and removes chatter.  With
    the rewrite switched off, REWRITTEN_QUERY equals the TRANSLATED_QUERY.

Example - previous turn was about the Eiffel Tower, user now types
"Wann wurde er gebaut?":

  Version           retrieval_language=english   retrieval_language=native
  ----------------  ---------------------------  --------------------------
  ORIGINAL_QUERY    Wann wurde er gebaut?        Wann wurde er gebaut?
  TRANSLATED_QUERY  When was it built?           Wann wurde er gebaut?
  REWRITTEN_QUERY   When was the Eiffel Tower    Wann wurde der Eiffelturm
                    built?                       gebaut?

Retrieval language (key "retrieval_language")
---------------------------------------------
english  The question is translated to English first; rewrite, vector search
         and rerank all work in English.  Best with English-only models.
native   The question stays in the user's language through rewrite, vector
         search and rerank.  Needs a multilingual embedder and reranker,
         e.g. embedder bge_m3 or snowflake (arctic-embed-l-v2.0) with
         reranker bge_reranker or mmarco (_ACTIVE_EMBED / _ACTIVE_CROSS in
         Config_Models.py; any combination works).  WordNet expansion is
         skipped for non-English text because WordNet is English-only.

Retrievers (the search engines)
-------------------------------
vector   Semantic search: finds chunks with similar *meaning* (embeddings in
         ChromaDB).  Works across wording differences.
bm25     Keyword search: finds chunks that contain the same *words*.
graph    Entity co-occurrence search: finds chunks that mention the same
         people/places/things together.
regex    Verb/noun pattern search: finds chunks with matching word patterns.
BM25, graph and regex are called *indexed retrievers*.  Their indexes are
built per document language ("language bucket"), so the query has to be
prepared ("shaped") per bucket: stop words removed, stemmed for that
language and - if allowed - translated into that bucket's language.

Leg (a single retrieval pass with one query version)
----------------------------------------------------
A *leg* is one run of the enabled retrievers with ONE query text.  A turn can
have up to three legs; all their hits are merged (Reciprocal Rank Fusion)
before reranking:
  main leg       Always runs.  Uses "main_query_source" and every enabled
                 retriever (vector/bm25/graph/regex).
  secondary leg  Optional safety net ("guardrail").  Runs the same retrievers
                 again with "secondary_query_source".  Typical use: if the LLM
                 rewrite drifts away from the question, the un-rewritten
                 TRANSLATED_QUERY still finds the right chunks.  It is skipped
                 automatically when its text is identical to the main query.
  original-language vector leg
                 Optional.  An extra VECTOR-only search with the ORIGINAL_QUERY
                 (the user's own words).  Helps multilingual embedders find
                 chunks written in the user's language.  Skipped automatically
                 when the user writes English, the language is unknown, or the
                 text equals the main query.

Alternates (MultiQuery)
-----------------------
The MultiQuery LLM writes a few paraphrases of the REWRITTEN_QUERY.  With
"use_vector_alternates" each paraphrase runs as an extra vector search inside
the main leg.  Indexed retrievers use the paraphrases too when
_INDEXED_QUERY["use_alternates"] is True, query shaping is on and the main leg
uses the REWRITTEN_QUERY.

After retrieval
---------------
rerank             A cross-encoder re-scores every merged candidate against
                   "rerank_query_source" and keeps the best ones.
low-score fallback If every rerank score is below the threshold, keep the
                   retrieval order instead of returning nothing.
low-recall rescue  If the reranker keeps too few chunks, add local chunks that
                   share words with the query from already represented files.
                   If no table survived, reserve one slot for the best
                   retrieval-ranked table fragment, even without word overlap.
grounding          Ties answer sentences back to source chunks and (with
                   mark_text) highlights them in the PDFs.
confidence_evaluation
                   Quality scoring of the answer only (off | heuristic | llm |
                   both).  Does NOT change which chunks are retrieved.

===============================================================================
FLOW KEYS - every flow below lists all keys explicitly
===============================================================================
Text keys:

  Key                     Allowed values                      Default
  ----------------------  ----------------------------------  ----------------
  retrieval_language      english | native                    english
  main_query_source       ORIGINAL_QUERY | TRANSLATED_QUERY   REWRITTEN_QUERY
                          | REWRITTEN_QUERY
  secondary_query_source  same three query versions           TRANSLATED_QUERY
  rerank_query_source     same three query versions           REWRITTEN_QUERY
  force_retrieve_mode     "" | VECTOR | BM25 | GRAPH | REGEX  "" (none)
                          | VECTOR_BM25 | ... | ALL | WEB
  confidence_evaluation   off | heuristic | llm | both        both

  retrieval_language      Language the question is searched in (see
                          "Retrieval language" above).
  main_query_source       Query version used by the main leg.
  secondary_query_source  Query version used by the secondary leg.
  rerank_query_source     Query version the reranker compares chunks with.
                          Use ORIGINAL_QUERY only with a multilingual
                          reranker.
  force_retrieve_mode     "" = use the run_* switches below.  A local mode
                          (VECTOR, BM25, VECTOR_BM25, ...) additionally
                          limits the local retrievers to that mode; ALL
                          does not limit; WEB skips local retrieval (needs
                          WEB_SEARCH_MODE=1).  Full list:
                          _ALLOWED_RETRIEVE_MODES in
                          Config_RAGChat_Strategies.py.
  confidence_evaluation   See "After retrieval".

On/off keys (True / False).  A key that is MISSING counts as True; a
misspelled key is rejected at startup with a ConfigurationError.

  Key                                  Meaning when True
  -----------------------------------  ---------------------------------------
  use_query_rewrite                    LLM rewrite (REWRITTEN_QUERY differs
                                       from TRANSLATED_QUERY)
  use_pronoun_substitution             pronoun resolution inside the rewrite
  use_secondary_query                  run the secondary leg
  use_original_language_vector         run the original-language vector leg
  use_vector_alternates                MultiQuery paraphrases for the vector
                                       search
  shape_indexed_queries                per-language query shaping for
                                       BM25/Graph/Regex
  translate_indexed_queries            allow translation into each bucket
                                       language
  expand_indexed_queries_with_wordnet  add English WordNet synonyms
  run_vector                           run the vector retriever
  run_bm25                             run the BM25 retriever
  run_graph                            run the graph retriever
  run_regex                            run the regex retriever
  run_rerank                           baseline rerank state (session
                                       "rerank" wins)
  run_low_score_fallback               see "After retrieval"
  run_low_recall_rescue                see "After retrieval"
  run_grounding                        see "After retrieval"

Every text key and on/off key can be overridden for one session:
CLI  "<key>=value" (on/off keys use their enable_* or friendly names),
API  request fields (main_query_source, rerank_query_source, ...).

===============================================================================
HOW TO EXPERIMENT
===============================================================================
ENGLISH_ALL is the reference ("baseline").  Each ENGLISH_ALL_* flow changes
exactly ONE thing, so the difference in answer quality measures that one
feature.  ENGLISH_VECTOR / ENGLISH_BM25 / ORIGINAL_VECTOR are single-retriever
baselines.  NATIVE_ALL is the multilingual-model setup.

===============================================================================
MAKING A FLOW MULTILINGUAL (running the ENGLISH_* experiments natively)
===============================================================================
The ENGLISH_* flows translate every question to English first.  To run the
same experiment in the user's own language, change ONE thing:
retrieval_language from "english" to "native".

Prerequisite: a multilingual embedder AND reranker must be active
(_ACTIVE_EMBED = bge_m3 | snowflake, _ACTIVE_CROSS = bge_reranker | mmarco in
Config_Models.py).  Changing the embedder requires re-indexing.

Three ways to switch, from quick test to permanent:
  1. One session, CLI:  pick the flow FIRST, then override the language
                          orchestrator_flow=ENGLISH_BM25
                          retrieval_language=native
                        (picking a flow resets retrieval_language to the
                        flow's value, so the order matters).
  2. One request, API:  "orchestrator_flow": "ENGLISH_BM25",
                        "retrieval_language": "native"
  3. Permanently:       copy the flow below, give it a new name (e.g.
                        NATIVE_BM25) and set "retrieval_language": "native".
                        Keep everything else identical so the comparison
                        with the English flow stays clean.

What each flow does in native mode:

  Flow                    With retrieval_language = native
  ----------------------  ---------------------------------------------------
  ENGLISH_ALL             identical to NATIVE_ALL
  ENGLISH_ALL_MULTILEG    secondary leg searches with the user's own,
                          un-rewritten words; the original-language vector
                          leg skips itself (same text as the secondary leg)
  ENGLISH_ALL_WORDNET     WordNet only expands English questions; for other
                          languages it behaves like NATIVE_ALL
  ENGLISH_ALL_NO_REWRITE  searches with the user's words exactly as typed
                          (all three query versions are identical)
  ENGLISH_ALL_NO_RERANK   works unchanged, in the user's language
  ENGLISH_VECTOR          works unchanged, in the user's language
  ENGLISH_BM25            works unchanged; the query is translated into each
                          document bucket's language
  ORIGINAL_VECTOR         already uses the user's words for the vector
                          search; in native mode only the rewrite differs
                          from ENGLISH_VECTOR (native)
"""

# -----------------------------------------------------------------------------
# Flow selector
# -----------------------------------------------------------------------------
# Restrict which flows users may pick.  Empty list = all flows defined in
# _ORCHESTRATION_FLOWS below.
_ALLOWED_ORCHESTRATION_FLOWS: list[str] = []

# Flow used when the session/request does not pick one.  The service config
# (Config_RAGChatService.py) inherits this value unless it overrides it.
_DEFAULT_ORCHESTRATION_FLOW = "NATIVE_ALL"

# Old flow names that still work (CLI, API, saved scripts).  They are resolved
# to the new name before validation.  Removed flows have no alias.
_ORCHESTRATION_FLOW_ALIASES: dict[str, str] = {
    "BGE_ENC_XENC": "NATIVE_ALL",
    "THOROUGH_QUERY_REWRITE": "ENGLISH_ALL_MULTILEG",
    "ORIGINAL_LANGUAGE_VECTOR_ONLY": "ORIGINAL_VECTOR",
}

# -----------------------------------------------------------------------------
# Indexed retriever (BM25/Graph/Regex) query preparation - applies to all flows
# -----------------------------------------------------------------------------
_INDEXED_QUERY: dict[str, str | int | bool] = {
    # How many chunk metadata records are scanned to discover which document
    # languages (buckets) exist.  0 = scan the whole (filtered) collection;
    # a positive number caps the scan for speed on huge collections.
    "language_detection_limit": 0,
    # What to do when translating the query into a bucket language fails:
    #   "stop"          abort retrieval for this turn with an error
    #   "skip_language" skip that bucket and continue with the others
    "translation_failure_policy": "skip_language",
    # True: MultiQuery paraphrases also run against BM25/Graph/Regex (only when
    # the main leg uses REWRITTEN_QUERY and shape_indexed_queries is on).
    "use_alternates": True,
}

# Query versions a flow or a session override may choose from.
_ALLOWED_ORCHESTRATION_QUERY_SOURCES = [
    "REWRITTEN_QUERY",
    "TRANSLATED_QUERY",
    "ORIGINAL_QUERY",
]

# -----------------------------------------------------------------------------
# Flow profiles
# -----------------------------------------------------------------------------
_ORCHESTRATION_FLOWS: dict[str, dict[str, str | bool]] = {
    # =========================================================================
    # NATIVE_ALL
    # For: a multilingual embedder (bge_m3 or snowflake) + a multilingual
    #      reranker (bge_reranker or mmarco), in any combination.
    # Story for one turn:
    #   1. Will NOT translate: the question stays in the user's language.
    #   2. Will rewrite it in that language (pronoun substitution on).
    #   3. Main leg will retrieve with the rewritten native query
    #      (REWRITTEN_QUERY) from vector (+ paraphrases), BM25, graph and
    #      regex.  For document buckets in another language the indexed
    #      query is translated into that bucket's language.  No WordNet
    #      (English-only).
    #   4. No secondary leg, no original-language vector leg (the main query
    #      already is in the user's language).
    #   5. Will rerank all hits against the rewritten native query, then apply
    #      low-score fallback, low-recall rescue and grounding.
    # Compare with ENGLISH_ALL to measure "native multilingual vs. translate
    # everything to English".
    # =========================================================================
    "NATIVE_ALL": {
        "retrieval_language": "native",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_ALL  - the experiment baseline
    # For: English-only embedder/reranker, or as the reference for ablations.
    # Story for one turn:
    #   1. Will translate the question to English (TRANSLATED_QUERY).
    #   2. Will rewrite the English question (pronoun substitution on)
    #      -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from vector
    #      (+ paraphrases), BM25, graph and regex; indexed queries are shaped
    #      and, for non-English buckets, translated.  No WordNet.
    #   4. Single leg only: no secondary leg, no original-language vector leg.
    #   5. Will rerank against REWRITTEN_QUERY, then fallback, rescue,
    #      grounding.
    # Multilingual version: NATIVE_ALL (identical except
    # retrieval_language=native).
    # =========================================================================
    "ENGLISH_ALL": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_ALL_MULTILEG
    # = ENGLISH_ALL + two extra legs.  Maximum recall, slowest.
    # Story for one turn:
    #   1. Will translate the question to English (TRANSLATED_QUERY).
    #   2. Will rewrite it -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from all four retrievers
    #      (+ paraphrases), exactly like ENGLISH_ALL.
    #   4. Secondary leg will retrieve AGAIN from all four retrievers, this
    #      time with the un-rewritten English translation (TRANSLATED_QUERY)
    #      - a safety net in case the rewrite changed the meaning.
    #   5. Original-language vector leg will run one more vector search with
    #      the user's own words (ORIGINAL_QUERY) when the user did not write
    #      English.
    #   6. Will merge all legs, rerank against REWRITTEN_QUERY, then fallback,
    #      rescue, grounding.
    # Compare with ENGLISH_ALL to measure what the extra legs add.
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top); the secondary leg then uses the user's own
    # words.
    # =========================================================================
    "ENGLISH_ALL_MULTILEG": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": True,
        "use_original_language_vector": True,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_ALL_WORDNET
    # = ENGLISH_ALL + WordNet synonym expansion for BM25/Graph/Regex.
    # Story for one turn:
    #   1. Will translate to English, 2. will rewrite -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from all four
    #      retrievers; the English indexed queries additionally get WordNet
    #      synonyms
    #      ("car" -> "car automobile auto").  Vector search is unchanged.
    #   4. Single leg.  5. Rerank against REWRITTEN_QUERY, fallback, rescue,
    #      grounding.
    # Compare with ENGLISH_ALL to measure synonym expansion on keyword recall.
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top); WordNet then only helps English questions.
    # =========================================================================
    "ENGLISH_ALL_WORDNET": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": True,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_ALL_NO_REWRITE
    # = ENGLISH_ALL without the LLM rewrite (and therefore without pronoun
    # substitution).
    # Story for one turn:
    #   1. Will translate to English (TRANSLATED_QUERY).
    #   2. Will NOT rewrite: REWRITTEN_QUERY is simply the English translation.
    #      Follow-up questions like "and when was it built?" are searched
    #      as-is.
    #   3. Main leg will retrieve with that translation from all four
    #      retrievers (+ paraphrases of it).
    #   4. Single leg.  5. Rerank against the translation, fallback, rescue,
    #      grounding.
    # Compare with ENGLISH_ALL to measure what the rewrite step is worth
    # (especially on multi-turn conversations).
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top); the user's words are then used exactly as
    # typed.
    # =========================================================================
    "ENGLISH_ALL_NO_REWRITE": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": True,
        "use_query_rewrite": False,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": False,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_ALL_NO_RERANK
    # = ENGLISH_ALL without the cross-encoder rerank.
    # Story for one turn:
    #   1. Will translate to English, 2. will rewrite -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from all four
    #      retrievers.
    #   4. Single leg.
    #   5. Will NOT rerank: the fused retrieval order is used directly.
    #      Low-score fallback and low-recall rescue depend on rerank, so they
    #      are off too.
    #      Grounding still runs.
    # Compare with ENGLISH_ALL to measure what the reranker is worth.
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top).
    # =========================================================================
    "ENGLISH_ALL_NO_RERANK": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": True,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": True,
        "run_graph": True,
        "run_regex": True,
        "run_rerank": False,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": False,
        "run_low_recall_rescue": False,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_VECTOR  - classic dense-retrieval baseline
    # Story for one turn:
    #   1. Will translate to English, 2. will rewrite -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from the VECTOR
    #      retriever only, without paraphrases.  BM25, graph and regex do
    #      not run.
    #   4. Single leg.  5. Rerank against REWRITTEN_QUERY, fallback, rescue,
    #      grounding.
    # Compare with ENGLISH_ALL to measure what the hybrid (keyword/graph/regex)
    # retrievers add; compare with ENGLISH_BM25 for dense vs. sparse.
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top).
    # =========================================================================
    "ENGLISH_VECTOR": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": False,
        "translate_indexed_queries": False,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": False,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": False,
        "run_graph": False,
        "run_regex": False,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ENGLISH_BM25  - classic keyword baseline
    # Story for one turn:
    #   1. Will translate to English, 2. will rewrite -> REWRITTEN_QUERY.
    #   3. Main leg will retrieve with REWRITTEN_QUERY from BM25 only.  The
    #      query is shaped per language bucket and translated for non-English
    #      buckets.
    #      Vector, graph and regex do not run.
    #   4. Single leg.  5. Rerank against REWRITTEN_QUERY, fallback, rescue,
    #      grounding.
    # Compare with ENGLISH_VECTOR for keyword vs. semantic search.
    # Multilingual: set retrieval_language=native (see MAKING A FLOW
    # MULTILINGUAL at the top).
    # =========================================================================
    "ENGLISH_BM25": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "REWRITTEN_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": True,
        "translate_indexed_queries": True,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": False,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": False,
        "run_bm25": True,
        "run_graph": False,
        "run_regex": False,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
    # =========================================================================
    # ORIGINAL_VECTOR  (formerly ORIGINAL_LANGUAGE_VECTOR_ONLY)
    # For: testing a multilingual embedder with the user's untouched words.
    # Story for one turn:
    #   1. Will translate to English and 2. will rewrite -> REWRITTEN_QUERY;
    #      both are still produced because the reranker and the answer use
    #      them.
    #   3. Main leg will retrieve with the ORIGINAL_QUERY (user's own words, no
    #      rewrite, no pronoun substitution) from the VECTOR retriever only,
    #      without paraphrases.
    #   4. Single leg.
    #   5. Will rerank against the English REWRITTEN_QUERY, then fallback,
    #      rescue, grounding.
    # Differs from ENGLISH_VECTOR only in main_query_source, so comparing the
    # two measures "user's language vs. English translation" for the embedder.
    # Already multilingual for the vector search; with retrieval_language=
    # native the reranker also works in the user's language.
    # =========================================================================
    "ORIGINAL_VECTOR": {
        "retrieval_language": "english",
        "force_retrieve_mode": "",
        "main_query_source": "ORIGINAL_QUERY",
        "secondary_query_source": "TRANSLATED_QUERY",
        "use_secondary_query": False,
        "use_original_language_vector": False,
        "shape_indexed_queries": False,
        "translate_indexed_queries": False,
        "expand_indexed_queries_with_wordnet": False,
        "use_vector_alternates": False,
        "use_query_rewrite": True,
        "use_keyword_action_splitter": True,
        "use_pronoun_substitution": True,
        "run_vector": True,
        "run_bm25": False,
        "run_graph": False,
        "run_regex": False,
        "run_rerank": True,
        "rerank_query_source": "REWRITTEN_QUERY",
        "run_low_score_fallback": True,
        "run_low_recall_rescue": True,
        "run_grounding": True,
        "confidence_evaluation": "both",
    },
}
