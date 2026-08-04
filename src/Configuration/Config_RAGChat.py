import os
from typing import Any

from Gui.Colors import \
    MARKED_DOCS_ANSWER_ANSI_COLOR as _DEFAULT_ANSWER_ANSI_COLOR
from Gui.Colors import \
    MARKED_DOCS_ANSWER_MARK_COLOR as _DEFAULT_ANSWER_MARK_COLOR
from Gui.Colors import MARKED_DOCS_HIGHLIGHT_COLOR as _DEFAULT_HIGHLIGHT_COLOR

# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned.py, Config_Models.py, Config_Global.py
# - Entries starting with _ cannot be overwritten using CLI arguments
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# Do not change _FRIENDLY_NAME
_FRIENDLY_NAME = "RAGChat"

# -----------------------------------------------------------------------------
# Web search — per-app audit logs
# All shared web-search switches (WEB_SEARCH_MODE env var, _WEB_SEARCH dict,
# WEB_SEARCH_INTENT_EXTENSIONS) are in Config_WebSearch.py.
# _QUERY_LOG:  Append-only audit log written for every web-search attempt
#   (including blocked ones).  Written to the RAGChat log directory.
# -----------------------------------------------------------------------------

_LOG_DIRECTORY = os.path.join(
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    ),
    "logs",
    "RAGChat",
)

# -----------------------------------------------------------------------------
# Chunk selection strategy
# -----------------------------------------------------------------------------
_ALLOWED_STRATEGIES = ["ULTRA_WIDE", "WIDE", "BALANCED_FILE_CAP", "NARROW", "DEFAULT"]
_ACTIVE_CHUNK_SELECT_STRATEGY = "DEFAULT"  # Default: pick one of _ALLOWED_STRATEGIES
_ALLOWED_RETRIEVE_MODES = [
    "VECTOR",  # Embedding-based retrieval only
    "BM25",  # Keyword-based retrieval only
    "GRAPH",  # Entity co-occurrence graph only
    "VECTOR_BM25",  # Vector + BM25 fused via RRF
    "VECTOR_GRAPH",  # Vector + graph fused via RRF
    "BM25_GRAPH",  # BM25 + graph fused via RRF
    "ALL",  # Vector + BM25 + graph fused via RRF (all retrieval algorithms)
    "WEB",  # Web search only; skips all local indexes (requires web_search enabled)
]  # Retrieval mode

_QUERY_LOG: str = os.path.join(_LOG_DIRECTORY, "queries.log")

# Append-only log for WebSearchFilter intent-classifier decisions (both web
# and local queries).  Set to "" to disable.
_INTENT_FILTER_LOG: str = os.path.join(_LOG_DIRECTORY, "intent_filter.log")

# -----------------------------------------------------------------------------
# Chat history and user defaults
# -----------------------------------------------------------------------------
_HISTORY_DIRECTORY = r"history"  # Where to store chat histories and metadata

# Keys to extract# -----------------------------------------------------------------------------
# Keyword extraction
# -----------------------------------------------------------------------------
_KEY_BERT = {
    "TOP_N_FIRST": 100,  # Keywords from first  KeyBERT pass
    "TOP_N_SECOND": 60,  # Keywords from second KeyBERT pass
}

_CLASSIFICATION_KEYS = []

_DEFAULT_CHAT_NAME = "MyFirstChat"  # Fallback user identifier

# -----------------------------------------------------------------------------
# Declared preferred response language of the user (lowercase NLTK language name,
# e.g. "english", "german", "french"). Must match a value in
# _ARGOS_DEFINITIONS.LANG_CODE_TO_NAME. Used by FileUtils.get_user_text_language()
# -----------------------------------------------------------------------------
# Retrieval-Augmented Generation strategy profiles
# -----------------------------------------------------------------------------
_STRATEGIES: dict[str, dict[str, int | float | bool | str]] = {
    "NARROW": {  # Precision-oriented — only strong semantic matches
        "final_chunks_to_llm": 20,  # Max chunks after selection — small for focused answers
        "retriever_k": 80,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.60,  # cross-encoder confidence floor; sigmoid(logit) ≥ 0.60 needs a clearly positive logit (precision). If no chunk clears it, rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 8192,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 20,  # Narrow token sampling — focused word choices
        "top_p": 0.8,  # Tight nucleus sampling — conservative candidate pool
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 0,  # Graph disabled — SingleDocumentSelector discards cross-file results anyway
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 5,  # Max 5 chunks per file — prevents large files filling all 20 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 5,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode: VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    },
    "BALANCED_FILE_CAP": {  # Balanced precision / recall with per-file chunk cap
        "final_chunks_to_llm": 40,  # Moderate selection window
        "retriever_k": 60,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.55,  # cross-encoder confidence floor; sigmoid(logit) ≥ 0.55 (slightly positive logit). If no chunk clears it, rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 40,  # Moderate token sampling — some variety
        "top_p": 0.92,  # Moderate nucleus sampling
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 10,  # Max 10 chunks per file — enforces actual diversity across files
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode: VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    },
    "DEFAULT": {  # General-purpose balanced retrieval
        "final_chunks_to_llm": 50,  # Moderate selection window
        "retriever_k": 100,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.50,  # cross-encoder confidence floor at the neutral logit (sigmoid(0)=0.50). If no chunk clears it the rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 40,  # Moderate token sampling — some variety
        "top_p": 0.92,  # Moderate nucleus sampling
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 15,  # Max 15 chunks per file — prevents large docs filling all 40 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode: VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    },
    "WIDE": {  # Recall-oriented — exploratory search across many chunks
        "final_chunks_to_llm": 60,  # Large selection window — more context for the LLM
        "retriever_k": 160,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.50,  # cross-encoder confidence floor at the neutral logit (sigmoid(0)=0.50); favors recall. If no chunk clears it the rerank is skipped and chunks fall back to retrieval (RRF) order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 100,  # Broad token sampling — 100 candidates per step
        "top_p": 0.97,  # Wide nucleus sampling — most of the probability mass
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 20,  # Loose cap — breadth allowed but no file takes more than 20 of 60 slots
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode: VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    },
    "ULTRA_WIDE": {  # Diagnostic / exploratory — very high recall, high cost
        "final_chunks_to_llm": 1500,  # Very large selection window (high computational cost)
        "retriever_k": 3000,  # Retriever candidates fetched per store before fusion/reranking
        "threshold": 0.45,  # cross-encoder confidence floor below the neutral logit; keeps weak (slightly negative logit) matches and rarely falls back to retrieval order
        "max_output_tokens": 14366,  # Upper bound on generated output tokens
        "temperature": 0.1,  # Low temperature — near-deterministic output
        "top_k": 100,  # Broad token sampling — 100 candidates per step
        "top_p": 0.97,  # Wide nucleus sampling — most of the probability mass
        "rerank": 1,  # Cross-encoder reranking enabled
        "vector_weight": 1,  # RRF weight for vector retriever (1 = full, 0 = off)
        "bm25_weight": 1,  # RRF weight for BM25 retriever (1 = full, 0 = off)
        "graph_weight": 1,  # RRF weight for graph retriever (1 = full, 0 = off)
        "web_weight": 0.5,  # Weight of web search results in RRF fusion (0.0–1.0)
        "filelim": 0,  # No per-file chunk limit
        "use_chat_context": True,  # Include previous conversation turns in retrieval and rewrite context
        "turns": 10,  # Max stored chat turns before pruning
        "prune_batch": 5,  # Oldest turns summarized per prune pass
        "max_history_turns": 3,  # Recent turns sent to the query rewriter
        "TOPIC_SUMMARY_MODE": "last",  # "last" = most recent ASSISTANT turn; "all" = all turns joined
        "retrieve_mode": "ALL",  # Retrieval mode: VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    },
}

# -----------------------------------------------------------------------------
# Prompt template for chat responses
# -----------------------------------------------------------------------------
_PROMPT_CHAT = """
CRITICAL: You must ONLY use information found in the context below.
Do NOT use your training knowledge, do NOT guess, do NOT infer beyond what the context states.
If an attribute, fact, or relationship is not explicitly present in the context for a given entity,
you MUST NOT supply it from outside knowledge, not even as a parenthetical, hedge, or aside
(e.g. do NOT write "X is also Y" or "X is generally Y" when the context does not say so).
State plainly that the context is silent on that point and stop.

CRITICAL RULE ON SOURCE CITATIONS:
You MUST NEVER cite a source ([Source: X]) for any fact unless that exact claim
appears in that source's retrieved text.
Citing a source for something it does not contain is a more serious error than saying
"the context is silent on this." If you know a fact from training but cannot find it in
the context, you MUST NOT write "according to the sources" — you must admit the context
does not cover it.

ONE EXCEPTION — DIRECT LOGICAL INFERENCE FROM CONTEXT (NEGATIVE INFERENCES ONLY):
If the context provides an explicit, reasonably complete description of a characteristic
(e.g. diet, habitat, function) FOR THE SPECIFIC ENTITY ASKED ABOUT, and the query asks
whether something absent from that description is part of that characteristic,
you MAY answer NEGATIVELY using one direct inference step —
citing the specific context statement as evidence.
Example: context states "whales eat krill, fish, and squid"; query asks "do whales eat insects?" →
you may answer "No — according to the sources, whales eat krill, fish, and squid; insects are
not part of their described diet."
This exception applies ONLY to negative conclusions drawn from what the context explicitly states
about the same entity. It does NOT permit asserting that something IS the case from training
knowledge, and it does NOT apply when the context contains no information about the queried entity.
Do NOT extend to multi-step reasoning, classification, or any fact not derivable in one step.
If the context is entirely empty or absolutely irrelevant to the query, you MUST respond with
EXACTLY these two lines and NOTHING else — no metadata, no explanation, no rephrasing:

I couldn't find relevant information to answer your query.
Try increasing retriever_k, top_k and lower threshold or change strategy.

Do NOT alter, summarize, or add to those two lines.

Context:
---------------------
{context}
---------------------

IMPORTANT:
You are permitted to extract and aggregate lists of entities (like animals) from multiple chunks.
Treat any inquisitive query about a category
(e.g. "what animals are discussed", "what files mention X", "which products are listed", "what topics are covered")
AS AN EXHAUSTIVE LIST REQUEST — fully equivalent to "list all <entities>".
When asked to list or extract entities, you MUST be ABSOLUTELY EXHAUSTIVE.
You MUST systematically scan every single chunk and extract every single relevant instance mentioned,
including those in lists, examples, sub-categories, or those mentioned only briefly or in passing.
However, you MUST preserve strict factual accuracy: do NOT generalize facts from one entity to others.
For example, if the text states that entity A has property P,
do not assume that entity B also has property P unless its own chunk explicitly says so.
Only output the failure message if the context provides NO relevant entities or information.

ATTRIBUTE-FILTERED LIST QUERIES (e.g. "what mammals are discussed", "which files mention reptiles",
"list the open-source products"):
  EXHAUSTIVE applies ONLY to entities for which the requested attribute (mammal, reptile, open-source, …)
  is EXPLICITLY stated in a chunk's Content. The attribute MUST appear in the chunk text itself for
  that specific entity — not inferred from the entity's name, not supplied from your training data,
  not derived from biological / commonsense / world knowledge.
  If a chunk mentions an entity but never states the attribute, that entity is NOT included in the answer.
  Do NOT add parenthetical justifications such as "(belongs to class Mammalia)", "(is also a mammal)",
  "(German for horses, which are mammals)". Such parentheticals are outside knowledge and are FORBIDDEN.
  Translating an entity's name into the query language is allowed; classifying it is NOT.

MANDATORY PROCEDURE for list/extraction queries:
STEP 1 — The first lines of the Context block list the DISTINCT SOURCE FILES.
         You MUST treat that list as the authoritative, complete enumeration of source files.
         Do NOT skip any file in that list. Do NOT shorten it.
STEP 2 — Determine the relevant entities by reading the Content of each chunk.
         Entities must be grounded in the chunk text itself, not guessed from the FileName.
         The FileName is metadata only — use it to group or label evidence, never as proof
         that an entity is discussed.
         Chunks in languages other than English still count as evidence; translate the entity
         name into the query's language (e.g. a German term for an entity should be translated
         into the query's language before being listed).
STEP 3 — Confirm each entity is actually discussed in at least one chunk's Content
         (chunks in languages other than English still count as evidence).
STEP 4 — Emit the full list. Do NOT stop after 3–5 items. Do NOT collapse similar items.
         Before answering, verify your answer mentions EVERY file from the DISTINCT SOURCE FILES list.

OUTPUT FORMAT — your response MUST contain TWO sections in this exact order,
and you MUST NOT skip either section:

### Answer
A complete, direct answer to the query, written in Markdown.
Synthesize the exact evidence from the context.
Do not invent connections or attributes not present in the text.
If the query asks "which of X, Y, Z are <attribute>",
you MUST explicitly state for EVERY entity listed in the query
whether the context confirms the attribute, denies it, or is silent on it — never omit an entity.
This section MUST contain at least one full sentence and MUST NOT be empty or replaced by metadata.

### Sources
A bullet list of the metadata fields for EVERY distinct FileName you used to answer
(one bullet group per distinct FileName — do not repeat the same FileName):
  - FileName
  - FilePath
  - Page (use the printed page label shown in the source header, if available)

Query:
{input}

### Answer
"""

# Legacy aliases – both point to the unified prompt above
_PROMPT_CHAT_MISTRAL = _PROMPT_CHAT
_PROMPT_CHAT_LLAMA = _PROMPT_CHAT

# ── Query Rewrite ──────────────────────────────────────────────────────────────

_PROMPT_TOPIC_DETECT = """You are a conversational query analyser for a retrieval system.

Your job is to decide whether the current user utterance depends on the previous turn,
and to produce retrieval-ready rewrites in either case.

You MUST output STRICT JSON. No commentary, no markdown, no preamble.

### Inputs
Previous user utterance : {previous_user_utterance}
Rolling topic summary   : {rolling_topic_summary}
Current user utterance  : {current_user_utterance}

### Output schema
{{
  "depends_on_previous_turn": <boolean>,
  "confidence": <float 0.0-1.0>,
  "reasoning": <string - one sentence, not chain-of-thought>,
  "contextual_rewrite": <string | null>,
  "standalone_rewrite": <string>,
  "salient_referents": <list of strings>
}}

### Rules

RULE 1 - Detect dependency via semantics, not lexical overlap.
  Both utterances may be in different languages or paraphrased.
  Look for ellipsis, anaphora, or implicit reference:
    "those", "that", "it", "they", "which ones", "more", "expand",
    German: "diese", "jene", "sie", "es",
    French: "ceux", "ca", "ils",
    Spanish: "esos", "ellos",
    Italian: "quelli", "essi"
  If the current utterance introduces a new semantic intent that does not rely
  on any prior entity, set depends_on_previous_turn = false.

RULE 2 - Prefer false negatives over false positives.
  When in doubt, treat the utterance as standalone (depends = false).
  It is better to miss a follow-up than to inject stale context into a new topic.

RULE 3 - contextual_rewrite.
  Populate ONLY when depends_on_previous_turn = true.
  Inline every salient entity from the rolling topic summary that the current
  utterance implicitly refers to. Replace all pronouns and demonstratives.
  Must be a complete, self-contained retrieval query.
  If depends_on_previous_turn = false, contextual_rewrite MUST be null.
  IMPORTANT: inline only domain entities (names, products, concepts).
  Never inline page numbers, file names, URLs, or source citations — see RULE 10.

RULE 4 - standalone_rewrite.
  Always populate. Must be a clean, retrieval-ready query with no conversational
  artifacts ("those", "they", "more", "expand", etc.).
  Do NOT invent entities not stated or clearly implied in the current utterance.
  STRICT when depends_on_previous_turn = false AND salient_referents = []:
    - Do NOT resolve any pronoun ("it", "its", "they", "those", etc.) by guessing
      a referent. No prior context is available to ground the resolution.
    - Do NOT introduce any named entity, proper noun, or specific domain term
      not present verbatim in the current utterance.
    - Preserving ambiguity is correct. Inventing a plausible entity is wrong.
    - allowed  : "What are RAM specifications?"
    - forbidden : "What are the RAM specifications of hedgehogs?"

RULE 5 - salient_referents.
  List the specific entities from the rolling topic summary that the current
  utterance implicitly refers to. Empty list when depends = false.

RULE 6 - Language.
  Both rewrites must be in the same language as the current user utterance.
  Do NOT translate. Entity names from the topic summary are copied verbatim.

RULE 7 - Output only the JSON object. No explanation outside it.

RULE 8 - Case variants are the same token.
  Treat words that differ only in capitalisation as identical.
  "ram" and "RAM" are the same word; evaluate dependency and topic relevance
  based on the surrounding context and the rolling topic summary, not on case.
  Do NOT interpret a lowercase word as a different concept solely because it
  also happens to be the lowercase form of an acronym or proper noun in the
  previous topic context (or vice versa).
  Example: if the prior topic is hedgehogs and the current query is
  "do they have ram", treat "ram" with the same semantic weight as "RAM"
  (i.e. evaluate whether RAM/ram is plausibly part of the hedgehog topic,
  not whether the lowercase spelling suggests a different animal-related word).

RULE 9 - Preserve binary question form; never introduce meta-descriptor nouns.
  When the current utterance is a binary (yes/no) question — including forms
  such as "do X have Y?", "does X have Y?", "tell me whether X …",
  "is X a …?", "are X …?", "can X …?" — the standalone_rewrite MUST keep the
  same binary question form.
  Do NOT convert a binary question into a WH-question ("What are the …?",
  "How many …?", "Which …?").
  Do NOT introduce any of the following meta-descriptor nouns as the head of
  the rewrite unless they were present verbatim in the original utterance:
    characteristics, characteristic, specifications, specification, specs, spec,
    features, feature, properties, property, capabilities, capability,
    parameters, parameter, configuration, settings, setting, details, detail,
    requirements, requirement, overview, information, info, attributes, attribute.
  Incorrect: standalone_rewrite = "What are the characteristics of bee stingers?"
  Correct  : standalone_rewrite = "Do bees have stingers?"
  Incorrect: standalone_rewrite = "What are the features of hedgehog spines?"
  Correct  : standalone_rewrite = "Do hedgehogs have spines?"

RULE 10 - Never embed retrieval metadata in rewrites.
  Do NOT copy page numbers, file names, URLs, section numbers, chunk IDs, or
  any other citation / source reference from the rolling topic summary or
  previous answer into either rewrite.
  Rewrites must express only the user's semantic intent, not where information
  was previously found.
  Incorrect: contextual_rewrite = "… PCIe slots found on pages 58 and 64 of ts_p620_user_guide.pdf"
  Correct  : contextual_rewrite = "… PCIe slots available on the Lenovo P620 workstation"
  The banned patterns include (but are not limited to):
    "on page N", "pages N and M", "in file X", "section N", any filename with
    an extension (.pdf, .docx, .txt, …), any URL.
"""

_QUERY_REWRITE: dict[str, Any] = {
    "enabled": True,
    "topic_confidence_threshold": 0.5,  # Minimum LLM confidence (0.0-1.0) to use the contextual
    # rewrite (depends_on_previous_turn=True path). Below this
    # threshold standalone_rewrite is used instead.
    "TOPIC_SUMMARY_MODE": "last",  # Controls rolling_topic_summary passed to the topic-detect LLM.
    # "last" = ASSISTANT block from the most recent history turn only.
    # "all"  = ASSISTANT blocks from all turns in the history window.
    # Retrieval gate: noun heads that signal an attribute/property query without
    # a concrete entity anchor.  If the rewritten query's root NP head matches one
    # of these and contains no named entity or proper noun, retrieval is blocked
    # and a clarification message is returned instead.  The list is intentionally
    # small and domain-agnostic — do not add specific attribute names here.
    "meta_descriptors": [
        "specifications",
        "specification",
        "specs",
        "spec",
        "details",
        "detail",
        "features",
        "feature",
        "properties",
        "property",
        "characteristics",
        "characteristic",
        "capabilities",
        "capability",
        "parameters",
        "parameter",
        "configuration",
        "settings",
        "setting",
        "requirements",
        "requirement",
        "information",
        "info",
        "overview",
    ],
    # Normalise the user query to English BEFORE query rewriting and retrieval.
    # Applies to both vector and BM25 paths so HYBRID fusion stays consistent.
    # Also helps the rewriter LLM resolve English pronouns reliably and prevents
    # zero cross-lingual token overlap in BM25.
    #
    # TRANSLATION_BACKEND selects the engine:
    #   "argos"  — offline Argos Translate (OPUS-MT based; lighter but lower quality
    #              on short/colloquial sentences). Requires the language pair to be
    #              installed (see _ARGOS_DEFINITIONS.ARGOS_LANGUAGES).
    #   "m2m100" — facebook/m2m100_418M via Hugging Face Transformers (MIT, ~1.7 GB).
    #              Better quality on short queries; lazy-loaded singleton.
    #   "off"    — no translation, query sent as-is.
    "TRANSLATION_BACKEND": "m2m100",
    "LLM_PARAM": {
        "temperature": 0.05,
        "top_k": 10,
        "top_p": 0.9,
        "num_predict": 256,
        "use_ollama_gpu": True,
        "streaming": False,
    },
}

# ── Multi-Query Expansion ──────────────────────────────────────────────────────

_PROMPT_QUERY_EXPAND = """You are a query expansion assistant for a retrieval system.

Given the retrieval query below, generate {num_variants} alternative phrasings.
The goal is maximum RETRIEVAL DIVERSITY: each variant should use different vocabulary
so that together they cover synonyms, technical equivalents, and domain-specific
terminology that a relevant document might use instead of the original wording.

Rules:
- Prefer domain-specific terms a technical document would actually contain
  (e.g. "thermal conditions" → also try "operating temperature", "operating environment",
  "temperature range", "environmental specifications", "ambient temperature limits").
- Use different wording, synonyms, and sentence structure across variants.
- Each variant must be a complete, self-contained retrieval query.
- Do NOT add new facts, entities, or assumptions not present in the original query.
- Do NOT produce meta-descriptor queries (e.g. "What are the characteristics of X?").
- Keep the same language as the input query.

Output ONLY a JSON array of strings. No commentary, no markdown, no preamble.
Example for num_variants=3: ["variant one", "variant two", "variant three"]

Query: {query}
"""

_MULTI_QUERY: dict[str, Any] = {
    "enabled": True,
    "num_variants": 3,  # Number of alternate queries to generate per turn
    "LLM_PARAM": {
        "temperature": 0.5,  # Higher than rewrite — diversity matters here
        "top_k": 40,
        "top_p": 0.95,
        "num_predict": 256,
        "use_ollama_gpu": True,
        "streaming": False,
    },
}

# ── Chunk Near-Duplicate Removal ───────────────────────────────────────────────

_CHUNK_DEDUP: dict[str, Any] = {
    "enabled": True,
    # Jaccard similarity threshold: chunks sharing >= this fraction of word
    # tokens are considered near-duplicates; the lower-ranked one is dropped.
    "threshold": 0.85,
}

# -----------------------------------------------------------------------------
# Terminal line size — RAGChat-specific override.
# Expressed as a dict so the width switches automatically based on the live
# session debug_level (resolved by QueryParts._resolve_terminal_line_size and
# PrettyWriter.terminal_line_size — both check this key at use time).
#   debug:    wide to accommodate algo tables when debug_level > 0
#   no_debug: normal conversation width
# RAGChatService inherits this via its `from Config_RAGChat import *`.
# All other apps receive the flat baseline from Config_Global.py.
# -----------------------------------------------------------------------------
TERMINAL_LINE_SIZE = {
    "debug": 180,
    "no_debug": 100,
}

# Multiplicative score boost applied during threshold filtering to chunks whose
# source file appears only once in the reranked candidate pool.  Such files
# have a single shot at clearing the threshold while large documents may
# contribute dozens of chunks; this levels the playing field.
# Set to 1.0 (or remove) to disable.  Default: 1.25
SINGLE_CHUNK_SCORE_BOOST = 1.25

# -----------------------------------------------------------------------------
# Visual-marker colours
# Canonical defaults live in Gui/Colors.py.
#
# _MARKED_DOCS_COLORS drives all visual-marker colours in one place:
# - highlight: source-document chunk highlight colour
# - answer_mark: grounded/effective answer spans in Markdown / HTML
# - answer_ansi: grounded/effective answer spans in CLI (ANSI SGR parameters)
# -----------------------------------------------------------------------------
_MARKED_DOCS_COLORS: dict[str, str] = {
    "highlight": _DEFAULT_HIGHLIGHT_COLOR,
    "answer_mark": _DEFAULT_ANSWER_MARK_COLOR,
    "answer_ansi": _DEFAULT_ANSWER_ANSI_COLOR,
}

# -----------------------------------------------------------------------------
# Answer grounding sensitivity (applies to both RAGChat and RAGChatService)
#
# Grounding marks answer sentences as "effective" only when they overlap
# retrieved chunk text. Tuning these values changes strictness:
# - Lower min_sentence_tokens: more short sentences can be marked.
# - Lower min_fragment_len: shorter chunk lines can anchor a match.
# - Lower min_overlap_window: requires shorter contiguous overlap for paraphrases.
# -----------------------------------------------------------------------------
_MARKED_DOCS_GROUNDING: dict[str, Any] = {
    "min_sentence_tokens": 5,
    "min_fragment_len": 12,
    "min_overlap_window": 5,  # Contiguous token window for paraphrase grounding
}
