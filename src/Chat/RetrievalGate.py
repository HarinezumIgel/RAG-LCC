"""RetrievalGate — decides whether a rewritten query is eligible for retrieval.

If the gate triggers it populates ``session.clarification_response`` with a
human-readable message and returns ``True`` (blocked).  The caller
(RAGChatImpl._retrieve) should then return ``("", 0)`` immediately.

Gate signals (any single one is enough to block):
  1. spaCy meta-descriptor: query root NP head is a meta-descriptor word
     (specifications, details, features, …) with no named entity or proper noun.
  2. spaCy pronoun: query contains a 3rd-person pronoun (it/they/she/…) with
     no proper noun or named entity anchor — catches unresolved references even
     when the rewrite was skipped (e.g. first turn with no history).
  3. Session flag: ``session.rewrite_was_underspecified`` was set by
     PromptRewrite when it detected an unresolved pronoun and had no referents.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Config.Config import Config
    from Globals.Session import Session
    from Gui.PrettyWriter import PrettyWriter


class RetrievalGate:
    """Stateless helper — create once and call ``check()`` per turn."""

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        nlp: Any = None,  # injectable spaCy model for tests
    ) -> None:
        from Config.Config import Config
        from Gui.PrettyWriter import PrettyWriter

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()

        raw_descriptors: Any = self.cfg.get("_QUERY_REWRITE.meta_descriptors", [])
        self.meta_descriptors: frozenset[str] = frozenset(
            str(d).lower() for d in raw_descriptors if isinstance(raw_descriptors, list)
        )

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
                from Commons.Exceptions import ModelLoadError

                raise ModelLoadError(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, session: "Session") -> bool:
        """Evaluate the gate for the current turn.

        Returns ``True`` when retrieval should be blocked and
        ``session.clarification_response`` has been populated.
        Returns ``False`` when retrieval may proceed normally.
        """
        # EXAMPLE context:
        #   Previous utterance : "what is the blazingfast?"
        #   Current utterance  : "does it have spines"
        #   session.query is set by the caller to the value returned by PromptRewrite.
        #
        # HAPPY PATH  — PromptRewrite resolved "it" to "blazingfast":
        #   session.query = "does blazingfast have spines"
        #   session.rewrite_was_underspecified = False
        #
        # FAILURE PATH — PromptRewrite was underspecified (LLM missed the dependency):
        #   session.query = "does have spines"   (pronoun stripped by PromptRewrite sanitizer)
        #   session.rewrite_was_underspecified = True
        #
        # FIRST-TURN PATH — no history, PromptRewrite skipped entirely:
        #   session.query = "does it have spines"  (raw original, pronoun still present)
        #   session.rewrite_was_underspecified = False
        query: str = session.query or ""

        spacy_triggered, attribute_phrase = self._spacy_signal(query)
        pronoun_triggered, pronoun_text = self._pronoun_signal(query)
        # EXAMPLE (happy path) : spacy_triggered=False, pronoun_triggered=False, flag_triggered=False
        # EXAMPLE (failure path): spacy_triggered=False, pronoun_triggered=False, flag_triggered=True
        # EXAMPLE (first-turn) : spacy_triggered=False, pronoun_triggered=True("it"), flag_triggered=False
        flag_triggered: bool = bool(session.rewrite_was_underspecified)

        if not spacy_triggered and not pronoun_triggered and not flag_triggered:
            # EXAMPLE (happy path): all False -> gate open, return False immediately
            return False

        # Build a context-aware clarification message.
        if attribute_phrase:
            msg = (
                f"{attribute_phrase.capitalize()} can vary — "
                f"could you clarify what subject or item you'd like information about?"
            )
        elif pronoun_text:
            # EXAMPLE (first-turn): pronoun_text="it"
            #   msg = 'I\'m not sure what "it" refers to — could you clarify ...'
            msg = (
                f'I\'m not sure what "{ pronoun_text}" refers to — '
                f"could you clarify what you'd like information about?"
            )
        else:
            # EXAMPLE (failure path): no pronoun in sanitized query, but flag set
            #   msg = "Could you clarify your query — what would you like information about?"
            msg = (
                "Could you clarify your query — what would you like information about?"
            )

        session.clarification_response = msg

        trigger: list[str] = []
        if spacy_triggered:
            trigger.append("spaCy:no-entity-anchor")
        if pronoun_triggered:
            trigger.append("spaCy:unresolved-pronoun")
        if flag_triggered:
            trigger.append("flag:rewrite_was_underspecified")
        # EXAMPLE (failure path): trigger=["flag:rewrite_was_underspecified"]
        # EXAMPLE (first-turn):   trigger=["spaCy:unresolved-pronoun"]
        self.pretty.write(
            "I",
            "RetrievalGate",
            f"Retrieval blocked [{', '.join(trigger)}] — clarification requested"
            f" (query={query!r})",
        )
        return True
        # EXAMPLE (failure path): returns True -> caller skips retrieval, shows clarification
        # EXAMPLE (first-turn):   returns True -> caller skips retrieval, shows clarification

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _WH_RE = re.compile(r"^(what|which|who|how)\b", re.IGNORECASE)

    def _pronoun_signal(self, query: str) -> tuple[bool, str]:
        """Return (triggered, pronoun_text) when the query contains a
        3rd-person pronoun with no proper noun or named entity anchor.

        Works regardless of question form (does/is/can/what/…) so it catches
        cases like 'does it have spines?' that _spacy_signal misses.
        """
        if not query.strip():
            return False, ""

        doc: Any = self._nlp(query)
        has_anchor = any(tok.pos_ == "PROPN" or tok.ent_type_ for tok in doc)
        # EXAMPLE (happy path, query="does blazingfast have spines"):
        #   "blazingfast" -> PROPN => has_anchor=True -> return (False, "") immediately
        # EXAMPLE (failure path, query="does have spines"):
        #   no PROPN, no entity -> has_anchor=False -> continue to pronoun scan
        #   no 3rd-person pronoun present ("it" was already stripped) -> return (False, "")
        # EXAMPLE (first-turn, query="does it have spines"):
        #   no PROPN, no entity -> has_anchor=False -> continue to pronoun scan
        if has_anchor:
            return False, ""

        for tok in doc:
            if tok.pos_ == "PRON" and "3" in tok.morph.get("Person", []):
                # EXAMPLE (first-turn): tok.text="it", PRON / Person=3 -> return (True, "it")
                # If a common noun precedes this pronoun in the same query it has an
                # intra-query antecedent and is not unresolved.
                # EXAMPLE: "Can bees fly and how long they can do so?"
                #   "bees" (NOUN, i<"they".i) -> has_preceding_noun=True -> skip "they"
                # EXAMPLE: "does it have spines?" -> no noun before "it" -> unresolved
                if any(t.i < tok.i and t.pos_ == "NOUN" for t in doc):
                    continue
                return True, tok.text
        # EXAMPLE (failure path): loop ends with no match -> return (False, "")
        return False, ""

    def _spacy_signal(self, query: str) -> tuple[bool, str]:
        """Return (triggered, attribute_phrase).

        triggered       — True when the query is an attribute query with no
                          concrete entity anchor.
        attribute_phrase — the matched meta-descriptor phrase (for the message),
                          or "" if not triggered.
        """
        if not query.strip():
            return False, ""

        doc: Any = self._nlp(query)

        # Require at least a wh-question shape; bare imperatives can pass through.
        if not self._WH_RE.match(query):
            # EXAMPLE (all paths): "does blazingfast…" / "does have spines" / "does it have spines"
            #   None start with what/which/who/how -> returns (False, "") immediately for all paths.
            #   _spacy_signal never fires for these example queries.
            return False, ""

        # Check whether any token is a proper noun or named entity.
        has_anchor = any(tok.pos_ == "PROPN" or tok.ent_type_ for tok in doc)
        if has_anchor:
            return False, ""

        # Look for a meta-descriptor as the head of any noun chunk.
        for chunk in doc.noun_chunks:
            if (
                chunk.root.lemma_.lower() in self.meta_descriptors
                or chunk.root.text.lower() in self.meta_descriptors
            ):
                return True, chunk.root.text.lower()

        return False, ""
