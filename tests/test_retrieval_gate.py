# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportUnusedVariable=false
"""Tests for Chat.RetrievalGate.

Covers:
  * _pronoun_signal — original unresolved-pronoun detection
  * _pronoun_signal — noun-preceding-pronoun fix (common nouns as intra-query antecedents)
  * _spacy_signal   — meta-descriptor / no-anchor detection
  * check()         — integration of all signals + session flag
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Load spaCy once for the whole module.  Skip everything if not installed.
# ---------------------------------------------------------------------------

try:
    import spacy  # type: ignore[import-untyped]

    _nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    _nlp = None

pytestmark = pytest.mark.skipif(
    _nlp is None,
    reason="spaCy en_core_web_sm not installed — run: python -m spacy download en_core_web_sm",
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_META_DESCRIPTORS = [
    "specifications",
    "details",
    "features",
    "specs",
    "info",
    "information",
]


class StubConfig:
    def get(self, key, default=None):
        if key == "_QUERY_REWRITE.meta_descriptors":
            return _META_DESCRIPTORS
        return default

    def get_str(self, key, default=""):
        if key == "_GRAPH_INDEX.spacy_model":
            return "en_core_web_sm"
        return default

    def get_bool(self, key, default=False):
        return default


class StubPretty:
    def write(self, *a, **kw) -> None:
        pass


class StubSession:
    """Minimal session for check() tests."""

    def __init__(self, query: str = "", underspecified: bool = False) -> None:
        self.query = query
        self.rewrite_was_underspecified = underspecified
        self.clarification_response: str = ""


def _make_gate():
    from Chat.RetrievalGate import RetrievalGate

    return RetrievalGate(cfg=StubConfig(), pretty=StubPretty(), nlp=_nlp)


# ===========================================================================
# _pronoun_signal — baseline (unresolved pronoun, no anchor in query)
# ===========================================================================


class TestPronounSignalBaseline:
    """Queries that contain a 3rd-person pronoun with no antecedent should trigger."""

    @pytest.mark.parametrize(
        "query,expected_pronoun",
        [
            ("does it have spines?", "it"),
            ("can it fly?", "it"),
            ("is it venomous?", "it"),
            ("can they swim?", "they"),
            ("do they migrate?", "they"),
            ("what does it eat?", "it"),
        ],
    )
    def test_unresolved_pronoun_triggers(self, query, expected_pronoun):
        gate = _make_gate()
        triggered, pronoun_text = gate._pronoun_signal(query)
        assert triggered, f"Expected trigger for {query!r}"
        assert pronoun_text == expected_pronoun

    @pytest.mark.parametrize(
        "query",
        [
            "what do dolphins eat",
            "do bees have stings?",
            "how fast can a cheetah run?",
            "are hedgehogs nocturnal?",
        ],
    )
    def test_no_pronoun_does_not_trigger(self, query):
        gate = _make_gate()
        triggered, _ = gate._pronoun_signal(query)
        assert not triggered, f"Expected no trigger for {query!r}"

    def test_proper_noun_anchor_suppresses_trigger(self):
        """PROPN in query already suppresses via has_anchor — no change to this behaviour."""
        gate = _make_gate()
        triggered, _ = gate._pronoun_signal("does blazingfast have spines?")
        assert not triggered


# ===========================================================================
# _pronoun_signal — noun-preceding-pronoun fix
# ===========================================================================


class TestPronounSignalNounPrecedesPronoun:
    """When a common noun precedes the pronoun in the same query, the pronoun
    has an intra-query antecedent and must NOT be flagged as unresolved."""

    @pytest.mark.parametrize(
        "query",
        [
            "Can bees fly and how long they can do so?",
            "do bees have stings and how do they use them?",
            "hedgehogs are nocturnal — do they hibernate?",
            "what do hedgehogs eat and can they store food?",
            "dogs are mammals and they have fur",
            "bees are insects; they live in hives",
        ],
    )
    def test_noun_before_pronoun_does_not_trigger(self, query):
        gate = _make_gate()
        triggered, pronoun = gate._pronoun_signal(query)
        assert (
            not triggered
        ), f"Pronoun {pronoun!r} should be resolved by preceding noun in {query!r}"

    @pytest.mark.parametrize(
        "query",
        [
            # Pronoun comes BEFORE any noun — still unresolved
            "can they fly?",
            "does it eat plants?",
            # Note: "do they migrate in winter?" is intentionally excluded —
            # spaCy tags "winter" as a DATE entity, which sets has_anchor=True
            # and suppresses the pronoun check (pre-existing behaviour).
        ],
    )
    def test_pronoun_before_noun_still_triggers(self, query):
        """Pronoun appears before any noun → still unresolved."""
        gate = _make_gate()
        triggered, _ = gate._pronoun_signal(query)
        assert triggered, f"Expected trigger for {query!r}"

    def test_rewritten_query_bees_example(self):
        """Exact regression for the query-rewrite output that was falsely blocked."""
        gate = _make_gate()
        triggered, _ = gate._pronoun_signal("Can bees fly and how long they can do so?")
        assert not triggered


# ===========================================================================
# _spacy_signal — meta-descriptor / no-anchor
# ===========================================================================


class TestSpacySignal:

    @pytest.mark.parametrize(
        "query",
        [
            "what are the specifications",
            "what are the details",
            "what are the features",
        ],
    )
    def test_meta_descriptor_no_anchor_triggers(self, query):
        gate = _make_gate()
        triggered, phrase = gate._spacy_signal(query)
        assert triggered
        assert phrase != ""

    @pytest.mark.parametrize(
        "query",
        [
            "does blazingfast have spines?",
            "can dolphins swim fast?",
            "do hedgehogs have spines?",
        ],
    )
    def test_non_wh_query_does_not_trigger(self, query):
        """_spacy_signal only fires for wh-questions (what/which/who/how)."""
        gate = _make_gate()
        triggered, _ = gate._spacy_signal(query)
        assert not triggered

    def test_empty_query_does_not_trigger(self):
        gate = _make_gate()
        assert gate._spacy_signal("") == (False, "")
        assert gate._pronoun_signal("") == (False, "")


# ===========================================================================
# check() — integration
# ===========================================================================


class TestCheck:

    def test_clean_query_passes_gate(self):
        gate = _make_gate()
        session = StubSession(query="do bees have stings?")
        assert gate.check(session) is False
        assert session.clarification_response == ""

    def test_unresolved_pronoun_blocks(self):
        gate = _make_gate()
        session = StubSession(query="can it fly?")
        assert gate.check(session) is True
        assert "it" in session.clarification_response

    def test_intra_query_noun_antecedent_passes(self):
        """The noun-fix: 'Can bees fly and how long they can do so?' must pass the gate."""
        gate = _make_gate()
        session = StubSession(query="Can bees fly and how long they can do so?")
        assert gate.check(session) is False

    def test_underspecified_flag_blocks(self):
        gate = _make_gate()
        session = StubSession(query="can fly", underspecified=True)
        assert gate.check(session) is True
        assert session.clarification_response != ""

    def test_clarification_message_mentions_pronoun(self):
        gate = _make_gate()
        session = StubSession(query="does it eat insects?")
        gate.check(session)
        assert '"it"' in session.clarification_response

    def test_clarification_message_for_they(self):
        gate = _make_gate()
        session = StubSession(query="can they migrate?")
        gate.check(session)
        assert '"they"' in session.clarification_response

    def test_proper_noun_query_passes_gate(self):
        gate = _make_gate()
        session = StubSession(query="does blazingfast have spines?")
        assert gate.check(session) is False
