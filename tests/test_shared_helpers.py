# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for SharedHelpers pure-logic methods: normalize, tokenize, char_ngrams, jaccard, containment.
Uses DI to inject a stub Config so no real config file is needed.
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Compliance.SharedHelpers import SharedHelpers

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    """Absorbs all write() calls silently."""

    def write(self, *a, **kw):
        return None


class StubConfig:
    """Returns safe defaults for SharedHelpers config keys."""

    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
            "_LEET_MAP": {"@": "a", "$": "s", "0": "o", "1": "i", "3": "e"},
            "_CONFUSABLES": {},
        }
        return mapping.get(key, default)

    def get_int(self, key, default=0) -> int:
        return int(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_str(self, key, default="") -> str:
        return str(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_bool(self, key, default=False) -> bool:
        return bool(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_float(self, key, default=0.0) -> float:
        return float(self.get(key, default))  # type: ignore[reportArgumentType]

    def get_list(self, key, default=None) -> list[object]:
        return self.get(key, default if default is not None else [])  # type: ignore[reportReturnType]

    def get_dict(self, key, default=None) -> dict[str, object]:
        return self.get(key, default if default is not None else {})  # type: ignore[reportReturnType]


@pytest.fixture(autouse=True)
def reset_shared_helpers():
    """Reset the SharedHelpers singleton before each test."""
    SharedHelpers._reset()  # type: ignore[reportPrivateUsage]
    yield
    SharedHelpers._reset()  # type: ignore[reportPrivateUsage]


@pytest.fixture
def sh():
    """Return a SharedHelpers instance with stub deps."""
    return SharedHelpers(cfg=StubConfig(), pretty=StubPrettyWriter())  # type: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_empty_string(self, sh):
        assert sh.normalize("") == ""

    def test_lowercases(self, sh):
        assert sh.normalize("HELLO World") == "hello world"

    def test_nfkc_compatibility(self, sh):
        # NFKC decomposes ﬁ ligature to 'fi'
        assert "fi" in sh.normalize("ﬁnd")

    def test_collapses_whitespace(self, sh):
        assert sh.normalize("hello   world") == "hello world"

    def test_leet_map_applied(self, sh):
        result = sh.normalize("h3llo")
        assert result == "hello"

    def test_strips_trailing_whitespace(self, sh):
        assert sh.normalize("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_empty_string(self, sh):
        assert sh.tokenize("") == []

    def test_simple_sentence(self, sh):
        tokens = sh.tokenize("Hello World 123")
        assert tokens == ["hello", "world", "123"]

    def test_punctuation_stripped(self, sh):
        tokens = sh.tokenize("hello, world! test.")
        assert tokens == ["hello", "world", "test"]

    def test_mixed_case_lowered(self, sh):
        tokens = sh.tokenize("FoO BAR")
        assert tokens == ["foo", "bar"]


# ---------------------------------------------------------------------------
# char_ngrams
# ---------------------------------------------------------------------------


class TestCharNgrams:
    def test_bigrams(self, sh):
        grams = sh.char_ngrams("abcd", 2, 2)
        assert grams == ["ab", "bc", "cd"]

    def test_range(self, sh):
        grams = sh.char_ngrams("abc", 1, 2)
        assert grams == ["a", "b", "c", "ab", "bc"]

    def test_empty_string(self, sh):
        assert sh.char_ngrams("", 2, 3) == []

    def test_n_larger_than_text(self, sh):
        grams = sh.char_ngrams("ab", 3, 3)
        assert grams == []


# ---------------------------------------------------------------------------
# jaccard (static)
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets(self):
        assert SharedHelpers.jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint_sets(self):
        assert SharedHelpers.jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        # {a, b} & {b, c} = {b}, union = {a, b, c} -> 1/3
        score = SharedHelpers.jaccard(["a", "b"], ["b", "c"])
        assert abs(score - 1 / 3) < 1e-9

    def test_empty_lists(self):
        assert SharedHelpers.jaccard([], []) == 0.0


# ---------------------------------------------------------------------------
# containment (static)
# ---------------------------------------------------------------------------


class TestContainment:
    def test_full_containment(self):
        # {a, b, c} & {a, b} = {a, b}, min(3, 2) = 2 -> 2/2 = 1.0
        assert SharedHelpers.containment(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert SharedHelpers.containment(["a"], ["b"]) == 0.0

    def test_partial(self):
        # {a, b, c} & {a, d} = {a}, min(3, 2) = 2 -> 1/2 = 0.5
        assert SharedHelpers.containment(["a", "b", "c"], ["a", "d"]) == 0.5

    def test_empty(self):
        assert SharedHelpers.containment([], ["a"]) == 0.0
