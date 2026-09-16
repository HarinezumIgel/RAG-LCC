# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for Strategies.RegexRetriever - index lifecycle, term extraction,
strict/fallback query behavior, persistence, and filter matching.
"""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Strategies.RegexRetriever import RegexRetriever, _RegexIndexData

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    """Minimal Config stub returning regex index defaults."""

    _DEFAULTS: Dict[str, Any] = {
        "_REGEX_INDEX.REGEX_INDEX_DIR": "/tmp/regex",
        "_REGEX_INDEX.spacy_model": "en_core_web_sm",
        "_REGEX_INDEX.max_candidates": 50,
        "_REGEX_INDEX.min_token_chars": 3,
        "_REGEX_INDEX.noun_pos_tags": ["NOUN", "PROPN"],
        "_REGEX_INDEX.exclude_auxiliaries": True,
        "_REGEX_INDEX.aux_lemmas": [
            "be",
            "do",
            "have",
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "should",
            "will",
            "would",
        ],
        "_REGEX_INDEX.require_both_when_available": True,
        "_REGEX_INDEX.min_verb_hits": 1,
        "_REGEX_INDEX.min_noun_hits": 1,
        "_REGEX_INDEX.fallback_to_verb_only": True,
        "_REGEX_INDEX.fallback_min_verb_hits": 1,
        "_REGEX_INDEX.verb_weight": 2.0,
        "_REGEX_INDEX.noun_weight": 1.0,
        "_REGEX_INDEX.both_match_bonus": 0.25,
        "_REGEX_INDEX.fallback_score_scale": 0.6,
        "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
    }

    def get(self, key, default=None):
        return self._DEFAULTS.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self._DEFAULTS.get(key, default))

    def get_str(self, key: str, default: str = "") -> str:
        return str(self._DEFAULTS.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._DEFAULTS.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self._DEFAULTS.get(key, default))

    def get_list(self, key: str, default=None) -> list:
        val = self._DEFAULTS.get(key)
        if val is not None:
            return list(val)
        return default if default is not None else []

    def get_dict(self, key: str, default=None) -> dict:
        return default if default is not None else {}


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


class _FakeToken:
    """Mimics a spaCy token with .lemma_ and .pos_."""

    def __init__(self, lemma: str, pos: str) -> None:
        self.lemma_ = lemma
        self.pos_ = pos


class _FakeDoc:
    """Mimics an iterable spaCy Doc of tokens."""

    def __init__(self, tokens: List[_FakeToken]) -> None:
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class StubNLP:
    """Fake spaCy NLP returning pre-configured tokens by exact text key."""

    def __init__(self, token_map: Dict[str, List[tuple[str, str]]]) -> None:
        self._map = token_map

    def __call__(self, text: str) -> _FakeDoc:
        pairs = self._map.get(text, [])
        tokens = [_FakeToken(lemma, pos) for lemma, pos in pairs]
        return _FakeDoc(tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    RegexRetriever._reset()  # type: ignore[reportPrivateUsage]
    yield
    RegexRetriever._reset()  # type: ignore[reportPrivateUsage]


def _make_retriever(nlp: Any = None, **overrides: Any) -> RegexRetriever:
    """Create a RegexRetriever with stub deps (bypasses real spaCy load)."""
    if nlp is None:
        nlp = StubNLP({})
    r = RegexRetriever.__new__(RegexRetriever)
    r._initialized = True
    r.cfg = overrides.get("cfg", StubConfig())
    r.pretty = overrides.get("pretty", StubPrettyWriter())
    r.perf_logger = MagicMock()
    r._data = _RegexIndexData()

    r._regex_index_dir = overrides.get("regex_index_dir", "/tmp/regex")
    r._max_candidates = overrides.get("max_candidates", 50)
    r._min_token_chars = overrides.get("min_token_chars", 3)
    r._exclude_auxiliaries = overrides.get("exclude_auxiliaries", True)
    r._noun_pos_tags = set(overrides.get("noun_pos_tags", ["NOUN", "PROPN"]))
    r._aux_lemmas = set(
        overrides.get(
            "aux_lemmas",
            [
                "be",
                "do",
                "have",
                "can",
                "could",
                "may",
                "might",
                "must",
                "shall",
                "should",
                "will",
                "would",
            ],
        )
    )
    r._require_both_when_available = overrides.get("require_both_when_available", True)
    r._min_verb_hits = overrides.get("min_verb_hits", 1)
    r._min_noun_hits = overrides.get("min_noun_hits", 1)
    r._fallback_to_verb_only = overrides.get("fallback_to_verb_only", True)
    r._fallback_min_verb_hits = overrides.get("fallback_min_verb_hits", 1)
    r._verb_weight = overrides.get("verb_weight", 2.0)
    r._noun_weight = overrides.get("noun_weight", 1.0)
    r._both_match_bonus = overrides.get("both_match_bonus", 0.25)
    r._fallback_score_scale = overrides.get("fallback_score_scale", 0.6)
    r._spacy_model = "en_core_web_sm"
    r._nlp = nlp
    file_utils = overrides.get("file_utils", MagicMock())
    file_utils.is_safe_delete_path.return_value = True
    r._file_utils = file_utils

    RegexRetriever._instance = r  # type: ignore[reportPrivateUsage]
    return r


def _add(r: RegexRetriever, chunks: List[Dict[str, Any]]) -> None:
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]
    r.add_chunks(ids, texts, metas)
    r._data.collection_name = "test_coll"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

TOKEN_MAP: Dict[str, List[tuple[str, str]]] = {
    "Bees build hives in forests.": [
        ("bee", "NOUN"),
        ("build", "VERB"),
        ("hive", "NOUN"),
        ("forest", "NOUN"),
    ],
    "Birds migrate across continents.": [
        ("bird", "NOUN"),
        ("migrate", "VERB"),
        ("continent", "NOUN"),
    ],
    "Hives protect colonies.": [
        ("hive", "NOUN"),
        ("protect", "VERB"),
        ("colony", "NOUN"),
    ],
    "Auxiliary do have be.": [
        ("do", "AUX"),
        ("have", "AUX"),
        ("be", "AUX"),
    ],
    "Bees repair hives.": [
        ("bee", "NOUN"),
        ("repair", "VERB"),
        ("hive", "NOUN"),
    ],
    "build hives": [
        ("build", "VERB"),
        ("hive", "NOUN"),
    ],
    "migrate continents": [
        ("migrate", "VERB"),
        ("continent", "NOUN"),
    ],
    "migrate hive": [
        ("migrate", "VERB"),
        ("hive", "NOUN"),
    ],
    "build quickly": [
        ("build", "VERB"),
        ("quickly", "ADV"),
    ],
    "be do have": [
        ("be", "AUX"),
        ("do", "AUX"),
        ("have", "AUX"),
    ],
    "tiny tokens": [
        ("an", "NOUN"),
        ("x1", "NOUN"),
        ("123", "NOUN"),
        ("cat", "NOUN"),
    ],
}

CHUNKS = [
    {
        "id": "c1",
        "text": "Bees build hives in forests.",
        "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
    },
    {
        "id": "c2",
        "text": "Birds migrate across continents.",
        "meta": {"FilePath": "birds.txt", "FileName": "birds.txt"},
    },
    {
        "id": "c3",
        "text": "Hives protect colonies.",
        "meta": {"FilePath": "colonies.txt", "FileName": "colonies.txt"},
    },
    {
        "id": "c4",
        "text": "Auxiliary do have be.",
        "meta": {"FilePath": "aux.txt", "FileName": "aux.txt"},
    },
]


# ===========================================================================
# _RegexIndexData
# ===========================================================================


class TestRegexIndexData:
    def test_initial_state(self):
        data = _RegexIndexData()
        assert data.chunk_verbs == {}
        assert data.chunk_nouns == {}
        assert data.chunk_metas == {}
        assert data.chunk_texts == {}
        assert data.collection_name == ""
        assert data.doc_count_at_build == 0


# ===========================================================================
# Singleton / directory helpers
# ===========================================================================


class TestSingletonAndDirs:
    def test_is_loaded_for_empty(self):
        r = _make_retriever()
        assert r.is_loaded_for("test_coll") is False

    def test_is_loaded_for_after_add(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:1])
        assert r.is_loaded_for("test_coll") is True
        assert r.is_loaded_for("other_coll") is False

    def test_get_index_dir_alias(self):
        r = _make_retriever(regex_index_dir="/tmp/regex_root")
        assert r.get_regex_dir("abc") == os.path.join("/tmp/regex_root", "abc")
        assert r.get_index_dir("abc") == r.get_regex_dir("abc")


# ===========================================================================
# _extract_terms, _passes_strict_gate, _score
# ===========================================================================


class TestTermExtractionAndScoring:
    def test_extract_terms_excludes_aux_by_default(self):
        nlp = StubNLP({"q": [("be", "AUX"), ("run", "VERB"), ("dog", "NOUN")]})
        r = _make_retriever(nlp=nlp, exclude_auxiliaries=True)
        verbs, nouns = r._extract_terms("q")
        assert "be" not in verbs
        assert "run" in verbs
        assert "dog" in nouns

    def test_extract_terms_can_include_aux(self):
        nlp = StubNLP({"q": [("have", "AUX"), ("dog", "NOUN")]})
        r = _make_retriever(nlp=nlp, exclude_auxiliaries=False)
        verbs, nouns = r._extract_terms("q")
        assert "have" in verbs
        assert "dog" in nouns

    def test_extract_terms_respects_min_chars_and_alpha(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP), min_token_chars=3)
        verbs, nouns = r._extract_terms("tiny tokens")
        # an (len=2), x1 (len=2), 123 (no alpha) are filtered; cat survives
        assert verbs == set()
        assert nouns == {"cat"}

    def test_strict_gate_require_both(self):
        r = _make_retriever(require_both_when_available=True)
        assert (
            r._passes_strict_gate(
                q_has_verbs=True,
                q_has_nouns=True,
                verb_hits=1,
                noun_hits=1,
            )
            is True
        )
        assert (
            r._passes_strict_gate(
                q_has_verbs=True,
                q_has_nouns=True,
                verb_hits=1,
                noun_hits=0,
            )
            is False
        )

    def test_strict_gate_relaxed_either_channel(self):
        r = _make_retriever(require_both_when_available=False)
        assert (
            r._passes_strict_gate(
                q_has_verbs=True,
                q_has_nouns=True,
                verb_hits=1,
                noun_hits=0,
            )
            is True
        )

    def test_score_formula_with_bonus(self):
        r = _make_retriever(verb_weight=2.0, noun_weight=1.0, both_match_bonus=0.25)
        # verb_cov=0.5, noun_cov=0.5 => 2*0.5 + 1*0.5 + 0.25 = 1.75
        score = r._score(
            verb_hits=1, noun_hits=1, query_verb_count=2, query_noun_count=2
        )
        assert abs(score - 1.75) < 1e-9


# ===========================================================================
# add_chunks / remove_by_filepath
# ===========================================================================


class TestAddRemove:
    def test_add_chunks_stores_terms_and_meta(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:2])
        assert set(r._data.chunk_verbs["c1"]) == {"build"}
        assert set(r._data.chunk_nouns["c1"]) == {"bee", "hive", "forest"}
        assert r._data.chunk_metas["c2"]["FilePath"] == "birds.txt"
        assert r._data.chunk_texts["c2"] == "Birds migrate across continents."

    def test_remove_by_filepath_removes_matching_chunks(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS)
        r.remove_by_filepath("birds.txt")
        assert "c2" not in r._data.chunk_texts
        assert "c1" in r._data.chunk_texts

    def test_remove_by_filepath_noop_when_missing(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:2])
        before = dict(r._data.chunk_texts)
        r.remove_by_filepath("missing.txt")
        assert r._data.chunk_texts == before

    def test_remove_then_add_simulates_reingest(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:2])
        r.remove_by_filepath("animals.txt")
        r.add_chunks(
            ["c1_v2"],
            ["Bees repair hives."],
            [{"FilePath": "animals.txt", "FileName": "animals.txt"}],
        )
        assert "c1" not in r._data.chunk_texts
        assert "c1_v2" in r._data.chunk_texts

    def test_skips_removal_when_guard_rejects_path(self):
        file_utils = MagicMock()
        file_utils.is_safe_delete_path.return_value = False
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP), file_utils=file_utils)
        _add(r, CHUNKS)

        before = dict(r._data.chunk_texts)
        r.remove_by_filepath("/")

        file_utils.is_safe_delete_path.assert_called_once_with("/")
        assert r._data.chunk_texts == before


# ===========================================================================
# query
# ===========================================================================


class TestQuery:
    def test_empty_index_returns_empty(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        assert r.query("build hives", k=10) == []

    def test_query_with_no_terms_returns_empty(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS)
        assert r.query("be do have", k=10) == []

    def test_strict_match_returns_expected_chunk(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP), require_both_when_available=True)
        _add(r, CHUNKS)
        docs = r.query("build hives", k=10)
        assert len(docs) >= 1
        assert docs[0].id == "c1"
        assert docs[0].metadata["regex_match_mode"] == "strict"
        assert docs[0].metadata["regex_verb_hits"] == 1
        assert docs[0].metadata["regex_noun_hits"] == 1

    def test_fallback_verb_only_when_strict_has_no_hits(self):
        r = _make_retriever(
            nlp=StubNLP(TOKEN_MAP),
            require_both_when_available=True,
            fallback_to_verb_only=True,
        )
        _add(r, CHUNKS)
        docs = r.query("migrate hive", k=10)
        ids = {d.id for d in docs}
        assert "c2" in ids
        c2 = next(d for d in docs if d.id == "c2")
        assert c2.metadata["regex_match_mode"] == "fallback_verb_only"

    def test_no_fallback_when_disabled(self):
        r = _make_retriever(
            nlp=StubNLP(TOKEN_MAP),
            require_both_when_available=True,
            fallback_to_verb_only=False,
        )
        _add(r, CHUNKS)
        assert r.query("migrate hive", k=10) == []

    def test_relaxed_gate_can_match_either_channel(self):
        r = _make_retriever(
            nlp=StubNLP(TOKEN_MAP),
            require_both_when_available=False,
            fallback_to_verb_only=False,
        )
        _add(r, CHUNKS)
        docs = r.query("migrate hive", k=10)
        ids = [d.id for d in docs]
        # c2 (verb hit) should rank before c1/c3 (noun-only hits)
        assert "c2" in ids
        assert ids[0] == "c2"

    def test_file_filter_simple_eq(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS)
        docs = r.query("build hives", k=10, file_filter={"FileName": "animals.txt"})
        assert all(d.metadata["FileName"] == "animals.txt" for d in docs)

    def test_file_filter_dollar_eq(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS)
        docs = r.query(
            "migrate continents",
            k=10,
            file_filter={"FilePath": {"$eq": "birds.txt"}},
        )
        assert all(d.metadata["FilePath"] == "birds.txt" for d in docs)

    def test_top_k_and_max_candidates_caps_results(self):
        r = _make_retriever(
            nlp=StubNLP(TOKEN_MAP),
            require_both_when_available=False,
            max_candidates=1,
        )
        _add(r, CHUNKS)
        docs = r.query("migrate hive", k=10)
        assert len(docs) <= 1

    def test_query_result_has_pipeline_compatible_metadata(self):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS)
        docs = r.query("build hives", k=10)
        assert len(docs) == 1
        meta = docs[0].metadata
        assert "regex_score" in meta
        assert "regex_verb_hits" in meta
        assert "regex_noun_hits" in meta
        assert "regex_match_mode" in meta
        assert "regex_verb_matches" in meta
        assert "regex_noun_matches" in meta
        assert meta["bm25_score"] == 0.0
        assert meta["chroma_score"] == meta["regex_score"]
        assert meta["chroma_sim"] == 1.0


# ===========================================================================
# _matches_filter
# ===========================================================================


class TestMatchesFilter:
    def test_simple_match(self):
        assert RegexRetriever._matches_filter({"a": 1}, {"a": 1}) is True

    def test_simple_no_match(self):
        assert RegexRetriever._matches_filter({"a": 1}, {"a": 2}) is False

    def test_dollar_eq_match(self):
        assert RegexRetriever._matches_filter({"a": "x"}, {"a": {"$eq": "x"}}) is True

    def test_dollar_eq_no_match(self):
        assert RegexRetriever._matches_filter({"a": "x"}, {"a": {"$eq": "y"}}) is False


# ===========================================================================
# persistence / load-or-rebuild / ingest
# ===========================================================================


class TestPersistenceAndLifecycle:
    def test_persist_and_load_round_trip(self, tmp_path):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:2])
        r._data.collection_name = "my_coll"
        r._data.doc_count_at_build = 2

        r.persist(str(tmp_path))
        index_path = os.path.join(str(tmp_path), RegexRetriever.INDEX_FILENAME)
        assert os.path.isfile(index_path)

        RegexRetriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        r2._load(index_path)

        assert r2._data.collection_name == "my_coll"
        assert r2._data.doc_count_at_build == 2
        assert r2._data.chunk_texts == r._data.chunk_texts
        assert r2._data.chunk_verbs == r._data.chunk_verbs
        assert r2._data.chunk_nouns == r._data.chunk_nouns

    def test_load_or_rebuild_uses_persisted_when_fresh(self, tmp_path):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:2])
        r._data.collection_name = "my_coll"
        r._data.doc_count_at_build = 7
        r.persist(str(tmp_path))

        RegexRetriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        coll = MagicMock()
        coll.count.return_value = 7
        r2.load_or_rebuild(str(tmp_path), "my_coll", coll)

        assert len(r2._data.chunk_texts) == 2
        assert r2._data.collection_name == "my_coll"

    def test_load_or_rebuild_stale_rebuilds_from_collection(self, tmp_path):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        _add(r, CHUNKS[:1])
        r._data.collection_name = "my_coll"
        r._data.doc_count_at_build = 1
        r.persist(str(tmp_path))

        RegexRetriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        coll = MagicMock()
        coll.count.return_value = 999
        coll.get.return_value = {
            "ids": ["x1"],
            "documents": ["Bees repair hives."],
            "metadatas": [{"FilePath": "animals.txt", "FileName": "animals.txt"}],
        }
        r2.load_or_rebuild(str(tmp_path), "my_coll", coll)

        assert len(r2._data.chunk_texts) == 1
        assert "x1" in r2._data.chunk_texts

    def test_load_or_rebuild_without_file_rebuilds(self, tmp_path):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        coll = MagicMock()
        coll.count.return_value = 2
        coll.get.return_value = {
            "ids": ["a1", "a2"],
            "documents": [
                "Bees build hives in forests.",
                "Birds migrate across continents.",
            ],
            "metadatas": [
                {"FilePath": "animals.txt", "FileName": "animals.txt"},
                {"FilePath": "birds.txt", "FileName": "birds.txt"},
            ],
        }
        r.load_or_rebuild(str(tmp_path), "coll_x", coll)
        assert len(r._data.chunk_texts) == 2
        assert r._data.collection_name == "coll_x"

    def test_build_and_persist_full_rebuild(self, tmp_path):
        r = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        coll = MagicMock()
        coll.count.return_value = 2
        coll.get.return_value = {
            "ids": ["b1", "b2"],
            "documents": [
                "Bees build hives in forests.",
                "Hives protect colonies.",
            ],
            "metadatas": [
                {"FilePath": "animals.txt", "FileName": "animals.txt"},
                {"FilePath": "colonies.txt", "FileName": "colonies.txt"},
            ],
        }
        r.build_and_persist(str(tmp_path), "full_coll", coll)

        index_path = os.path.join(str(tmp_path), RegexRetriever.INDEX_FILENAME)
        assert os.path.isfile(index_path)

        RegexRetriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever(nlp=StubNLP(TOKEN_MAP))
        r2._load(index_path)
        assert len(r2._data.chunk_texts) == 2
        assert r2._data.collection_name == "full_coll"

    def test_ingest_file_removes_old_and_adds_new(self, tmp_path):
        r = _make_retriever(
            nlp=StubNLP(TOKEN_MAP),
            regex_index_dir=str(tmp_path),
        )
        _add(r, CHUNKS[:3])
        r._data.collection_name = "test_coll"

        coll = MagicMock()
        coll.count.return_value = 3
        r.ingest_file(
            "animals.txt",
            "test_coll",
            coll,
            ["c1_v2"],
            ["Bees repair hives."],
            [{"FilePath": "animals.txt", "FileName": "animals.txt"}],
        )

        assert "c1" not in r._data.chunk_texts
        assert "c1_v2" in r._data.chunk_texts
        assert "c2" in r._data.chunk_texts

        index_path = os.path.join(
            str(tmp_path),
            "test_coll",
            RegexRetriever.INDEX_FILENAME,
        )
        assert os.path.isfile(index_path)
