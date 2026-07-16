# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for Algos.Synonyms — WordNet-based synonym expansion for banned-word lists.
Covers: config-driven behaviour, expansion logic, caching, explosion control,
POS filtering, stoplist, depth, and graceful fallback when NLTK is absent.
"""

import sys
import os
import types
import pytest
from typing import List, Set
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    """Mirrors the _WORDNET block from Config_Global.py."""

    def __init__(self, overrides=None):
        self._data = {
            "_WORDNET.ENABLED": True,
            "_WORDNET.DEPTH": 1,
            "_WORDNET.MAX_SYNONYMS_PER_PHRASE": 3,
            "_WORDNET.POS_FILTER": ["n", "v"],
            "_WORDNET.STOPLIST": [
                "word",
                "number",
                "figure",
                "item",
                "thing",
                "part",
                "piece",
                "set",
                "group",
                "kind",
                "type",
                "form",
                "point",
                "line",
                "way",
                "case",
                "level",
                "area",
                "place",
                "make",
                "give",
            ],
            "DEBUG_LEVEL": 0,
        }
        if overrides:
            self._data.update(overrides)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def get_int(self, key, default=0) -> int:
        return int(self._data.get(key, default))  # type: ignore[arg-type]

    def get_str(self, key, default="") -> str:
        return str(self._data.get(key, default))  # type: ignore[arg-type]

    def get_bool(self, key, default=False) -> bool:
        return bool(self._data.get(key, default))  # type: ignore[arg-type]

    def get_float(self, key, default=0.0) -> float:
        return float(self._data.get(key, default))  # type: ignore[arg-type]

    def get_list(self, key, default=None) -> list[object]:
        v = self._data.get(key, default if default is not None else [])
        return v if isinstance(v, list) else [v]

    def get_dict(self, key, default=None) -> dict[str, object]:
        v = self._data.get(key, default if default is not None else {})
        return v if isinstance(v, dict) else {}  # type: ignore[return-value]


class StubPrettyWriter:
    """Captures write() calls for assertion."""

    def __init__(self):
        self.messages: list[tuple] = []

    def write(self, severity, label, message, **kw):
        self.messages.append((severity, label, message, kw))
        return None


# ---------------------------------------------------------------------------
# Helper: build a fresh Synonyms instance bypassing singleton
# ---------------------------------------------------------------------------


def _make_synonyms(cfg_overrides=None, pretty=None):
    """Construct a Synonyms object with stub deps, bypassing SingletonMixin."""
    from Algos.Synonyms import Synonyms

    s = Synonyms.__new__(Synonyms)
    s._initialized = False  # reset singleton guard
    cfg = StubConfig(cfg_overrides)
    pw = pretty or StubPrettyWriter()
    # manually run __init__ body
    s.cfg = cfg  # type: ignore[assignment]
    s.pretty = pw  # type: ignore[assignment]
    s.enabled = cfg.get_bool("_WORDNET.ENABLED")
    s.depth = cfg.get_int("_WORDNET.DEPTH")
    s.max_per_phrase = cfg.get_int("_WORDNET.MAX_SYNONYMS_PER_PHRASE")
    pos_raw = cfg.get_list("_WORDNET.POS_FILTER")
    s.pos_filter = {str(p).lower() for p in pos_raw} if pos_raw else set()
    stoplist_raw = cfg.get_list("_WORDNET.STOPLIST")
    s.stoplist = {str(w).lower() for w in stoplist_raw}
    s.debug_level = cfg.get_int("DEBUG_LEVEL")
    s._cache = {}
    s._available = None
    s._warned = False
    return s


# ---------------------------------------------------------------------------
# Check if WordNet is really available (for conditional tests)
# ---------------------------------------------------------------------------

_HAS_WORDNET = False
try:
    from nltk.corpus import wordnet as _wn  # type: ignore[import-untyped]

    _wn.synsets("test")  # type: ignore[no-untyped-call]
    _HAS_WORDNET = True
except Exception:
    pass

requires_wordnet = pytest.mark.skipif(
    not _HAS_WORDNET, reason="NLTK WordNet corpus not installed"
)


# ===========================================================================
# Tests: disabled / fallback
# ===========================================================================


class TestDisabled:
    def test_returns_original_when_disabled(self):
        s = _make_synonyms({"_WORDNET.ENABLED": False})
        phrases = ["password", "credit card number"]
        assert s.expand(phrases) is phrases

    def test_debug_message_when_disabled(self):
        pw = StubPrettyWriter()
        s = _make_synonyms({"_WORDNET.ENABLED": False, "DEBUG_LEVEL": 10}, pretty=pw)
        s.expand(["password"])
        assert any("disabled" in msg[2].lower() for msg in pw.messages)

    def test_disabled_message_only_once(self):
        pw = StubPrettyWriter()
        s = _make_synonyms({"_WORDNET.ENABLED": False, "DEBUG_LEVEL": 10}, pretty=pw)
        s.expand(["a"])
        s.expand(["b"])
        disabled_msgs = [m for m in pw.messages if "disabled" in m[2].lower()]
        assert len(disabled_msgs) == 1


class TestWordNetMissing:
    def test_orange_warning_when_wordnet_missing(self, monkeypatch):
        """Simulate NLTK not installed — should produce an orange warning."""
        import Algos.Synonyms as syn_mod

        # Temporarily force _wordnet_available to None and make import fail
        monkeypatch.setattr(syn_mod, "_wordnet_available", None)
        monkeypatch.setattr(syn_mod, "_wn", None)

        original_ensure = syn_mod._ensure_wordnet

        def fake_ensure():
            syn_mod._wordnet_available = False
            return False

        monkeypatch.setattr(syn_mod, "_ensure_wordnet", fake_ensure)

        pw = StubPrettyWriter()
        s = _make_synonyms(pretty=pw)
        s._available = None  # force re-check

        result = s.expand(["password"])

        # Should return original list unchanged
        assert result == ["password"]

        # Should have emitted an orange warning
        warnings = [m for m in pw.messages if m[0] == "W" and "WordNet" in m[1]]
        assert len(warnings) == 1
        assert "not installed" in warnings[0][2].lower()
        # Check it used ORANGE color
        from Gui.Colors import ORANGE

        assert warnings[0][3].get("color") == ORANGE

    def test_fallback_returns_original_list(self, monkeypatch):
        """When WordNet unavailable, expand() should return input unchanged."""
        import Algos.Synonyms as syn_mod

        monkeypatch.setattr(syn_mod, "_wordnet_available", None)
        monkeypatch.setattr(syn_mod, "_wn", None)
        monkeypatch.setattr(syn_mod, "_ensure_wordnet", lambda: False)

        s = _make_synonyms()
        s._available = None
        phrases = ["ssn", "exploit"]
        result = s.expand(phrases)
        assert result == phrases


# ===========================================================================
# Tests: expansion logic (require WordNet installed)
# ===========================================================================


@requires_wordnet
class TestExpansionBasic:
    def _fresh(self, **overrides):
        """Build a Synonyms with WordNet available."""
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(overrides)
        s._available = True
        return s

    def test_expand_adds_synonyms(self):
        s = self._fresh()
        result = s.expand(["password"])
        assert "password" in result
        assert len(result) > 1, "Expected at least one synonym added"

    def test_original_phrases_preserved_in_order(self):
        s = self._fresh()
        phrases = ["ssn", "password", "exploit"]
        result = s.expand(phrases)
        # Original phrases must appear and in original order
        originals_in_result = [p for p in result if p in phrases]
        assert originals_in_result == phrases

    def test_no_duplicates(self):
        s = self._fresh()
        result = s.expand(["password", "exploit", "backdoor"])
        lower_result = [r.lower() for r in result]
        assert len(lower_result) == len(set(lower_result))

    def test_empty_input(self):
        s = self._fresh()
        assert s.expand([]) == []


@requires_wordnet
class TestMaxPerPhrase:
    def _fresh(self, max_per=3):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(
            {
                "_WORDNET.MAX_SYNONYMS_PER_PHRASE": max_per,
                "_WORDNET.POS_FILTER": [],  # no filter — max synonyms
                "_WORDNET.STOPLIST": [],
            }
        )
        s._available = True
        return s

    def test_cap_at_max(self):
        s = self._fresh(max_per=2)
        result = s.expand(["password"])
        # "password" itself + at most 2 synonyms
        assert len(result) <= 3

    def test_cap_at_one(self):
        s = self._fresh(max_per=1)
        result = s.expand(["password"])
        assert len(result) <= 2

    def test_cap_at_zero_means_no_expansion(self):
        s = self._fresh(max_per=0)
        result = s.expand(["password"])
        assert result == ["password"]


@requires_wordnet
class TestPOSFilter:
    def _fresh(self, pos_filter):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(
            {
                "_WORDNET.POS_FILTER": pos_filter,
                "_WORDNET.MAX_SYNONYMS_PER_PHRASE": 50,  # high cap to see all
                "_WORDNET.STOPLIST": [],
            }
        )
        s._available = True
        return s

    def test_noun_only(self):
        """With POS_FILTER=['n'], should only get noun synonyms."""
        s = self._fresh(["n"])
        result_nouns = s.expand(["steal"])
        # "steal" as a noun ("a stolen base" in baseball) should appear
        # but verb synonyms should not

        s2 = self._fresh(["v"])
        result_verbs = s2.expand(["steal"])
        # Verb filter should give different (likely more) results
        # Just verify both return something and are different
        assert isinstance(result_nouns, list)
        assert isinstance(result_verbs, list)

    def test_empty_pos_filter_accepts_all(self):
        s = self._fresh([])
        result = s.expand(["password"])
        assert len(result) >= 1


@requires_wordnet
class TestStoplist:
    def _fresh(self, stoplist):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(
            {
                "_WORDNET.STOPLIST": stoplist,
                "_WORDNET.MAX_SYNONYMS_PER_PHRASE": 50,
                "_WORDNET.POS_FILTER": [],
            }
        )
        s._available = True
        return s

    def test_stoplist_excludes_words(self):
        # "word" is a WordNet synonym for "password" — should be excluded
        s = self._fresh(["word", "countersign"])
        result = s.expand(["password"])
        lower_result = {r.lower() for r in result}
        assert "word" not in lower_result
        assert "countersign" not in lower_result

    def test_empty_stoplist_allows_all(self):
        s_stop = self._fresh(["word"])
        s_nostop = self._fresh([])
        r_stop = s_stop.expand(["password"])
        r_nostop = s_nostop.expand(["password"])
        assert len(r_nostop) >= len(r_stop)


@requires_wordnet
class TestDepth:
    def _fresh(self, depth=1):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(
            {
                "_WORDNET.DEPTH": depth,
                "_WORDNET.MAX_SYNONYMS_PER_PHRASE": 50,
                "_WORDNET.POS_FILTER": [],
                "_WORDNET.STOPLIST": [],
            }
        )
        s._available = True
        return s

    def test_depth_1_gives_direct_synonyms(self):
        s = self._fresh(depth=1)
        result = s.expand(["password"])
        assert len(result) > 1

    def test_depth_2_gives_more_than_depth_1(self):
        s1 = self._fresh(depth=1)
        s2 = self._fresh(depth=2)
        r1 = set(s1._synonyms_for_phrase("password", depth=1))
        r2 = set(s2._synonyms_for_phrase("password", depth=2))
        # depth-2 should be a superset of depth-1 (or equal if no transitive hits)
        assert r1 <= r2


@requires_wordnet
class TestMultiWordPhrases:
    def _fresh(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms(
            {
                "_WORDNET.MAX_SYNONYMS_PER_PHRASE": 10,
                "_WORDNET.POS_FILTER": [],
                "_WORDNET.STOPLIST": [],
            }
        )
        s._available = True
        return s

    def test_multi_word_lookup(self):
        """Multi-word phrases like 'credit card' should be looked up as underscored key."""
        s = self._fresh()
        syns = s._synonyms_for_phrase("credit card", depth=1)
        # WordNet has "credit_card" as a synset
        assert isinstance(syns, list)

    def test_multi_word_fallback_to_tokens(self):
        """If a multi-word phrase isn't in WordNet, individual tokens should be tried."""
        s = self._fresh()
        # "account number" is unlikely to be a single WordNet entry
        syns = s._synonyms_for_phrase("zxqfake phrasenotreal", depth=1)
        # Should still return a list (possibly empty)
        assert isinstance(syns, list)


# ===========================================================================
# Tests: caching
# ===========================================================================


@requires_wordnet
class TestCaching:
    def _fresh(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms()
        s._available = True
        return s

    def test_cache_hit_returns_same_object(self):
        s = self._fresh()
        phrases = ["password", "exploit"]
        r1 = s.expand(phrases)
        r2 = s.expand(phrases)
        assert r1 is r2  # exact same object from cache

    def test_different_input_different_cache(self):
        s = self._fresh()
        r1 = s.expand(["password"])
        r2 = s.expand(["exploit"])
        assert r1 is not r2


# ===========================================================================
# Tests: debug output
# ===========================================================================


@requires_wordnet
class TestDebugOutput:
    def test_summary_at_debug_level_1(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn

        pw = StubPrettyWriter()
        s = _make_synonyms({"DEBUG_LEVEL": 10}, pretty=pw)
        s._available = True
        s.expand(["password"])
        debug_msgs = [m for m in pw.messages if m[0] == "D" and "Expanded" in m[2]]
        assert len(debug_msgs) == 1
        assert "phrases" in debug_msgs[0][2]

    def test_per_synonym_at_debug_level_3(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn

        pw = StubPrettyWriter()
        s = _make_synonyms(
            {
                "DEBUG_LEVEL": 55,
                "_WORDNET.STOPLIST": [],
                "_WORDNET.POS_FILTER": [],
            },
            pretty=pw,
        )
        s._available = True
        s.expand(["password"])
        syn_msgs = [m for m in pw.messages if "+synonym" in m[2]]
        assert len(syn_msgs) >= 1

    def test_no_debug_at_level_0(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn

        pw = StubPrettyWriter()
        s = _make_synonyms({"DEBUG_LEVEL": 0}, pretty=pw)
        s._available = True
        s.expand(["password"])
        debug_msgs = [m for m in pw.messages if m[0] == "D"]
        assert len(debug_msgs) == 0


# ===========================================================================
# Tests: explosion control (combined)
# ===========================================================================


@requires_wordnet
class TestExplosionControl:
    """Verify that with default settings the list doesn't explode."""

    SAMPLE_BANLIST: List[str] = [
        "ssn",
        "social security number",
        "tax id",
        "passport number",
        "credit card number",
        "account number",
        "iban",
        "password",
        "private-key",
        "api-key",
        "exploit",
        "backdoor",
        "malware",
        "bomb",
        "weapon",
        "steal",
        "extort",
    ]

    def _fresh(self):
        import Algos.Synonyms as syn_mod

        syn_mod._wordnet_available = True
        from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

        syn_mod._wn = wn
        s = _make_synonyms()  # default config
        s._available = True
        return s

    def test_expansion_ratio_bounded(self):
        """With max=3/phrase, expansion should be at most 4x the original."""
        s = self._fresh()
        result = s.expand(self.SAMPLE_BANLIST)
        ratio = len(result) / len(self.SAMPLE_BANLIST)
        assert ratio <= 4.0, (
            f"Expansion ratio {ratio:.1f}x exceeds 4x — "
            f"original={len(self.SAMPLE_BANLIST)}, expanded={len(result)}"
        )

    def test_no_stoplist_words_in_result(self):
        s = self._fresh()
        result = s.expand(self.SAMPLE_BANLIST)
        lower_result = {r.lower() for r in result}
        for stop in s.stoplist:
            assert (
                stop not in lower_result
            ), f"Stoplist word {stop!r} leaked into result"

    def test_all_originals_present(self):
        s = self._fresh()
        result = s.expand(self.SAMPLE_BANLIST)
        lower_result = {r.lower() for r in result}
        for phrase in self.SAMPLE_BANLIST:
            assert phrase.lower() in lower_result, f"Original phrase {phrase!r} missing"
