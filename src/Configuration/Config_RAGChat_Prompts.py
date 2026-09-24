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

FINAL RESPONSE LANGUAGE:
Write your final answer in: {response_language}
If a source quote is in another language, you may quote it verbatim but keep
the explanatory narrative in {response_language}.

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
Retrieval language      : {output_language}

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

RULE 6 - Retrieval language contract.
  Both rewrites MUST be in {output_language}.
  Do NOT mirror the user's language when {output_language} differs.
  This is a monolingual retrieval pipeline: output retrieval-language rewrites only.

RULE 6b - Preserve entity surface forms.
  Keep named entities, product names, policy IDs, document titles, acronyms,
  and quoted strings unchanged whenever possible.
  Translate only the surrounding sentence frame into {output_language}.

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
