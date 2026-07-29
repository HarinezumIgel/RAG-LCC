"""Query rewrite module for topic-change detection in multi-turn chat."""

import json
import re
import time
from typing import Any

# POS tags whose tokens qualify as content words for the grounding check.
_CONTENT_POS: frozenset[str] = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})

from AI.LLMCaller import LLMCaller
from AI.TokenBudget import TokenBudget
from Chat.ChatContext import ChatContext
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import BRIGHT_MAGENTA, ORANGE, VIOLET
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class PromptRewrite(SingletonMixin):

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
        nlp: Any = None,  # injectable spaCy model for tests
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()
        self.llmCaller: LLMCaller = LLMCaller()
        self.tokenBudget: TokenBudget = TokenBudget()
        self.chatContext: ChatContext = ChatContext()
        self.perf_logger: PerfLogger = PerfLogger()

        # Read model and prompt config from dedicated _LLM_REWRITE_PROMPT role
        llm_args: dict[str, Any] = self.helpers.get_model_args(
            "_ACTIVE_LLM_REWRITE_PROMPT"
        )
        self.llm_model: str = llm_args["MODEL"]
        rewrite_prompt_var: str = llm_args["PROMPT_TOPIC_DETECT"]
        self.prompt_template: str
        self.prompt_name: str | None
        self.prompt_template, self.prompt_name = self.cfg.indirect_get(
            rewrite_prompt_var
        )

        # Read dedicated LLM params
        self.enabled: bool = self.cfg.get_bool("_QUERY_REWRITE.enabled")
        self.temperature: float = self.cfg.get_float(
            "_QUERY_REWRITE.LLM_PARAM.temperature"
        )
        self.top_k: int = self.cfg.get_int("_QUERY_REWRITE.LLM_PARAM.top_k")
        self.top_p: float = self.cfg.get_float("_QUERY_REWRITE.LLM_PARAM.top_p")
        self.num_predict: int = self.cfg.get_int("_QUERY_REWRITE.LLM_PARAM.num_predict")
        self.use_gpu: bool = self.cfg.get_bool(
            "_QUERY_REWRITE.LLM_PARAM.use_ollama_gpu"
        )
        self.streaming: bool = self.cfg.get_bool("_QUERY_REWRITE.LLM_PARAM.streaming")
        self.topic_confidence_threshold: float = self.cfg.get_float(
            "_QUERY_REWRITE.topic_confidence_threshold"
        )
        self.topic_summary_mode: str = (
            self.cfg.get_str("_QUERY_REWRITE.TOPIC_SUMMARY_MODE") or "last"
        )
        # Load spaCy for pronoun POS detection in the grounding check.
        # Shares the same model name as GraphRetriever / RetrievalGate;
        # spaCy caches loaded models per process so there is no duplicate cost.
        if nlp is not None:
            self._nlp: Any = nlp
        else:
            spacy_model: str = (
                self.cfg.get_str("_GRAPH_INDEX.spacy_model") or "en_core_web_sm"
            )
            try:
                import spacy  # type: ignore[import-untyped]

                self._nlp = spacy.load(spacy_model)
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}"
                ) from exc

    def rewrite(self, session: Session) -> str:
        """Detect topic continuity and rewrite the user query for retrieval.

        Returns contextual_rewrite when the LLM determines the current
        utterance depends on the previous turn with sufficient confidence,
        standalone_rewrite otherwise. Falls back to the original query on
        parse failure or LLM error.
        """
        # Performance logging
        self.perf_logger.log(
            "PromptRewrite.rewrite",
            "chat",
            f"start query expansion original_len={len(session.query or '')}",
        )
        _t0 = time.perf_counter()

        # EXAMPLE: session.query = "does it have spines"
        original_query: str = session.query or ""
        # EXAMPLE: original_query = "does it have spines"

        # Reset one-shot underspecified flag from the previous turn.
        session.rewrite_was_underspecified = False

        if not self.enabled:
            return original_query

        if session.force_skip_rewrite:
            self.pretty.write(
                "I",
                "QueryRewrite",
                "Topic switch flagged - skipping rewrite for this turn.",
            )
            session.last_topic_referents = (
                None  # stale referents invalid after topic switch
            )
            return original_query

        # Fetch conversation history (only the most recent turns)
        history_docs = self.chatContext.fetch_context_docs(session)
        if not history_docs:
            self.pretty.write(
                "I",
                "QueryRewrite",
                "No conversation history \u2014 skipping rewrite",
            )
            return original_query

        max_ht: int = session.max_history_turns or 0
        if max_ht > 0:
            history_docs = history_docs[-max_ht:]

        if not history_docs:
            self.pretty.write(
                "I",
                "QueryRewrite",
                "No conversation history \u2014 skipping rewrite",
            )
            return original_query

        # ----- Extract previous_user_utterance from the most recent turn -----
        last_doc_content: str = history_docs[-1].page_content
        previous_user_utterance: str = ""
        for line in last_doc_content.splitlines():
            if line.startswith("USER:"):
                previous_user_utterance = line[len("USER:") :].strip()
        # EXAMPLE: previous_user_utterance = "what is the blazingfast?"

        # ----- Build rolling_topic_summary -----
        # Prefer LLM-extracted referents saved from the previous turn - they are
        # already distilled entities, not raw prose, which avoids topic bleed from
        # verbose or incorrect assistant responses.  Fall back to parsing ASSISTANT
        # blocks from history on the first rewrite in a session or after a topic switch.
        stored_referents: list[str] | None = getattr(
            session, "last_topic_referents", None
        )
        if stored_referents:
            rolling_topic_summary = "Key entities from previous turn: " + ", ".join(
                stored_referents
            )
        else:
            tsm: str = (
                getattr(session, "topic_summary_mode", None) or self.topic_summary_mode
            )
            if tsm == "all":
                assistant_blocks: list[str] = []
                for doc in history_docs:
                    doc_lines = doc.page_content.splitlines()
                    asst_idx = next(
                        (
                            i
                            for i, ln in enumerate(doc_lines)
                            if ln.startswith("ASSISTANT:")
                        ),
                        -1,
                    )
                    if asst_idx >= 0:
                        block = "\n".join(doc_lines[asst_idx:]).strip()
                        if block:
                            assistant_blocks.append(block)
                rolling_topic_summary = (
                    "\n".join(assistant_blocks) if assistant_blocks else "(none)"
                )
            else:
                # "last" mode: ASSISTANT block from the most recent turn only
                last_lines = last_doc_content.splitlines()
                asst_idx = next(
                    (
                        i
                        for i, ln in enumerate(last_lines)
                        if ln.startswith("ASSISTANT:")
                    ),
                    -1,
                )
                if asst_idx >= 0:
                    rolling_topic_summary = (
                        "\n".join(last_lines[asst_idx:]).strip() or "(none)"
                    )
                    # EXAMPLE: rolling_topic_summary = "ASSISTANT: blazingfast is a ..."
                else:
                    rolling_topic_summary = "(none)"

        # Build the topic-detect prompt
        # NOTE: We intentionally omit the actual file path from current_user_utterance.
        # The chat history is already scoped to the same file_tag by ChromaDB's where-filter
        # (see ChatContext._fetch_context_docs), so the LLM does not need the path to detect
        # topic continuity.  Injecting a real path caused the LLM to embed it — sometimes
        # translated — into the rewritten query (e.g. "[File: Pferde.pdf] tell me about
        # spiders" → "Tell me about spiders in D:/RAG-LCC/TestDocs/Horses.pdf").
        file_tag = session.file_name or session.file_path or ""
        current_ctx = "[File filter active]" if file_tag else "[No file filter]"
        current_user_utterance = f"{current_ctx} {original_query}"
        # EXAMPLE: current_user_utterance = "[No file filter] does it have spines"
        formatted: str = self.prompt_template.format(
            previous_user_utterance=previous_user_utterance,
            rolling_topic_summary=rolling_topic_summary,
            current_user_utterance=current_user_utterance,
        )

        effective_ctx: int = self.tokenBudget.get_effective_context_limit(
            self.llm_model, session
        )

        ollama_options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "num_ctx": effective_ctx,
        }
        if not self.use_gpu:
            ollama_options["num_gpu"] = 0

        self.pretty.write(
            "I",
            "QueryRewrite",
            f"Topic detection using {len(history_docs)} history turns",
        )

        try:
            result: dict[str, str] = self.llmCaller.call_llm(
                model=self.llm_model,
                prompt=formatted,
                ollama_options=ollama_options,
                answer_is_json=True,
                template_name=self.prompt_name,
                streaming=self.streaming,
                stage="Topic detect / query rewrite",
            )
        except Exception as exc:
            self.pretty.write(
                "E",
                "QueryRewrite",
                f"LLM call failed - using original query: {exc}",
            )
            return original_query

        raw: str = result.get("content", "").strip()
        if raw.startswith("[File:") or raw.startswith("[No file"):
            idx = raw.find("]")
            if idx != -1:
                raw = raw[idx + 1 :].strip()
        if not raw:
            self.pretty.write(
                "I",
                "QueryRewrite",
                "Empty response - using original query",
            )
            return original_query

        # Parse JSON response
        try:
            data: dict[str, Any] = json.loads(raw)
        except Exception:
            self.pretty.write(
                "I",
                "QueryRewrite",
                f"JSON parse failed - using original query. Raw: {raw[:120]}",
            )
            return original_query

        # EXAMPLE (happy path) — LLM correctly identifies the pronoun dependency:
        #   {"depends_on_previous_turn": true, "confidence": 0.95,
        #    "reasoning": "'it' refers to 'blazingfast' from the previous turn",
        #    "contextual_rewrite": "does blazingfast have spines",
        #    "standalone_rewrite": "does blazingfast have spines",
        #    "salient_referents": ["blazingfast"]}
        #
        # EXAMPLE (failure path) — LLM incorrectly claims no dependency:
        #   {"depends_on_previous_turn": false, "confidence": 0.3,
        #    "reasoning": "query seems standalone",
        #    "contextual_rewrite": null,
        #    "standalone_rewrite": "does blazingfast have spines",
        #    "salient_referents": []}
        depends: bool = bool(data.get("depends_on_previous_turn", False))
        confidence: float = float(data.get("confidence", 0.0))
        reasoning: str = str(data.get("reasoning", ""))
        contextual: str | None = data.get("contextual_rewrite") or None
        standalone: str = str(data.get("standalone_rewrite", original_query)).strip()
        referents: list[str] = list(data.get("salient_referents", []))
        # EXAMPLE (happy path): depends=True, confidence=0.95, referents=["blazingfast"]
        #   contextual="does blazingfast have spines"
        # EXAMPLE (failure path): depends=False, confidence=0.3, referents=[]
        #   contextual=None, standalone="does blazingfast have spines"

        # Persist referents for the next turn - they replace full ASSISTANT prose
        # as the rolling_topic_summary, keeping context compact and entity-focused.
        # Exception: if this turn turns out to be gate-blocked (underspecified), we
        # preserve the referents from the *previous* turn so the next real turn still
        # sees the correct entity anchor rather than a clarification message as context.
        previous_referents: list[str] | None = session.last_topic_referents
        session.last_topic_referents = referents if referents else None

        # Strip file-context tag from standalone (safety)
        if standalone.startswith("[File:") or standalone.startswith("[No file"):
            s_idx = standalone.find("]")
            if s_idx != -1:
                standalone = standalone[s_idx + 1 :].strip()

        # ----- Post-hoc grounding check -----
        # When depends=False and referents=[] the model had no prior context to
        # ground a pronoun resolution.  If the standalone rewrite nevertheless
        # introduced new content words not present in the original query, the
        # model hallucinated an entity.  Catch this deterministically and fall
        # back to the original query with pronouns stripped.
        #
        # EXAMPLE (happy path): depends=True => condition is False => block skipped entirely.
        #   Execution jumps straight to the decision logic below.
        #
        # EXAMPLE (failure path): depends=False, referents=[] => condition is True => enter block.
        if not depends and not referents:
            orig_doc = self._nlp(original_query)
            # 3rd-person and demonstrative pronouns — those that require a prior
            # referent from context.  1st/2nd-person forms are excluded because
            # they refer to conversation participants, not to prior entities.
            orig_pronouns = {
                tok.text.lower()
                for tok in orig_doc
                if tok.pos_ == "PRON" and "3" in tok.morph.get("Person", [])
            }
            # EXAMPLE: orig_pronouns = {"it"}  — spaCy tags "it" as PRON / Person=3
            if orig_pronouns:
                orig_words = {
                    tok.lemma_.lower()
                    for tok in orig_doc
                    if tok.pos_ in _CONTENT_POS and len(tok.text) > 2
                }
                # EXAMPLE: orig_words = {"have", "spines"}  ("it" is PRON, "does" < 3 chars)
                sa_doc = self._nlp(standalone)
                sa_words = {
                    tok.lemma_.lower()
                    for tok in sa_doc
                    if tok.pos_ in _CONTENT_POS and len(tok.text) > 2
                }
                # EXAMPLE: sa_words = {"blazingfast", "have", "spines"}
                invented = sa_words - orig_words
                # EXAMPLE: invented = {"blazingfast"}  — entity name not present in original query
                if invented:
                    # Distinguish genuine hallucinations from legitimate history resolutions.
                    # e.g. "can they fly?" after "do bees have stings?" → standalone="can bees fly?"
                    # "bee" is not in the original but IS in the chat history — not a hallucination.
                    history_words = {
                        tok.lemma_.lower()
                        for tok in self._nlp(rolling_topic_summary)
                        if tok.pos_ in _CONTENT_POS and len(tok.text) > 2
                    }
                    true_hallucinations = invented - history_words
                    if not true_hallucinations:
                        # All "invented" words are resolutions from context — accept the rewrite.
                        self.pretty.write(
                            "I",
                            "QueryRewrite",
                            f"Standalone grounded via history: resolved={sorted(invented)} — accepting rewrite",
                        )
                    else:
                        grounding_score = 1.0 - len(true_hallucinations) / max(
                            len(sa_words), 1
                        )
                        self.pretty.write(
                            "W",
                            "QueryRewrite",
                            f"Standalone rejected: grounding_score={grounding_score:.2f}"
                            f"  invented={sorted(true_hallucinations)} - falling back to sanitized original",
                        )
                        # Pronoun is unresolvable — mark underspecified and fall back.
                        session.rewrite_was_underspecified = True
                        # Restore previous referents so the NEXT real turn still sees
                        # the correct entity anchor rather than a clarification message.
                        session.last_topic_referents = previous_referents
                        # Sanitize: strip detected pronouns from original, collapse spaces
                        pronoun_re = re.compile(
                            rf"\b(?:{'|'.join(re.escape(p) for p in orig_pronouns)})\b",
                            re.IGNORECASE,
                        )
                        sanitized = pronoun_re.sub("", original_query)
                        # EXAMPLE: "does it have spines" -> "does  have spines" ("it" stripped)
                        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,.")
                        # EXAMPLE: sanitized = "does have spines"  (double-space collapsed)
                        standalone = sanitized if sanitized else original_query
                        # EXAMPLE: standalone = "does have spines"  — safe fallback
                else:
                    # Pronoun present, standalone rewrite introduced no new words at all
                    # (i.e. the pronoun was simply dropped). Without a resolved referent
                    # the query is underspecified.
                    session.rewrite_was_underspecified = True
                    session.last_topic_referents = previous_referents
            else:
                # No 3rd-person pronouns — the query is already fully self-contained.
                # Discard the LLM's standalone_rewrite; when depends=False there is
                # no pronoun to resolve, so any entity the model adds is hallucinated
                # (e.g. "Do camels exist in Igel?" for "Do you have any information
                # about camels?" after a previous turn that mentioned Igel).
                standalone = original_query

        # Always log the JSON decision fields
        self.pretty.write(
            "I",
            "QueryRewrite",
            f"depends={depends}  confidence={confidence:.2f}  referents={referents}",
        )
        self.pretty.write(
            "I",
            "QueryRewrite",
            f"reasoning: {reasoning!r}",
        )

        threshold: float = self.topic_confidence_threshold

        # Decision logic
        # EXAMPLE (happy path): depends=True, confidence=0.95 >= threshold, contextual set
        #   => chosen = "does blazingfast have spines"  (pronoun resolved from history)
        # EXAMPLE (failure path after grounding): standalone = "does have spines" (safe fallback)
        #   depends=False => falls to else branch => chosen = "does have spines"
        if depends and confidence >= threshold and contextual:
            chosen = contextual.strip()
            if chosen.startswith("[File:") or chosen.startswith("[No file"):
                c_idx = chosen.find("]")
                if c_idx != -1:
                    chosen = chosen[c_idx + 1 :].strip()
            label = ""
        elif depends and confidence >= threshold:
            # depends=True, confidence sufficient, but contextual_rewrite was null
            chosen = standalone
            label = " (standalone - contextual_rewrite was null)"
        else:
            chosen = standalone
            label = " (standalone - no prior context needed)"

        if chosen.lower() == original_query.lower():
            self.pretty.write(
                "I",
                "QueryRewrite",
                f"Query unchanged{label}",
                color=BRIGHT_MAGENTA,
            )
        else:
            self.pretty.write(
                "I",
                "QueryRewrite",
                f"'{original_query}' -> '{chosen}'{label}",
                color=BRIGHT_MAGENTA,
            )
            self.pretty.write(
                "I",
                "QueryRewrite",
                f"Final query: {chosen}",
                color=VIOLET,
            )
            if not label:  # contextual path only
                self.pretty.write(
                    "I",
                    "QueryRewrite",
                    "If the answer does not match your intent, set max_history_turns=1 or use_chat_context=false.",
                    color=ORANGE,
                )

        if DebugHelper.check_session(session, 60):
            self.pretty.write(
                "D",
                "QueryRewrite",
                f"Topic-detect prompt:\n{formatted}",
            )

        # Performance logging stop
        elapsed = time.perf_counter() - _t0
        self.perf_logger.log(
            "PromptRewrite.rewrite",
            "chat",
            f"stop  query expansion chosen={chosen[:50]!r} elapsed={elapsed:.3f}s",
        )

        return chosen
