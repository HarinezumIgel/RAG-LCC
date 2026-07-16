# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""
Tests for Strategies.GraphRetriever — index build, incremental update,
BFS query, persistence, filter matching, and singleton behaviour.
"""

import gzip
import os
import pickle
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents.base import Document as LangchainDocument

from Strategies.GraphRetriever import GraphRetriever, _GraphIndexData

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    """Minimal Config stub returning graph index defaults."""

    _DEFAULTS: Dict[str, Any] = {
        "_GRAPH_INDEX.entity_types": ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"],
        "_GRAPH_INDEX.max_hops": 2,
        "_GRAPH_INDEX.max_candidates": 50,
        "_GRAPH_INDEX.min_edge_weight": 1,
        "_GRAPH_INDEX.spacy_model": "en_core_web_sm",
        "_GRAPH_INDEX.noun_chunk_min_chars": 3,
        "_GRAPH_INDEX.noun_chunk_drop_leading": "[({<",
        "_GRAPH_INDEX.GRAPH_INDEX_DIR": "/tmp/graph",
    }

    def get(self, key, default=None):
        return self._DEFAULTS.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        val = self._DEFAULTS.get(key, default)
        return int(val)

    def get_str(self, key: str, default: str = "") -> str:
        val = self._DEFAULTS.get(key, default)
        return str(val)

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


class _FakeEnt:
    """Mimics a spaCy span with .text and .label_."""

    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class _FakeChunk:
    """Mimics a spaCy noun chunk span with .text."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    """Mimics a spaCy Doc with .ents and .noun_chunks."""

    def __init__(
        self, ents: List[_FakeEnt], noun_chunks: List[_FakeChunk] | None = None
    ) -> None:
        self.ents = ents
        self.noun_chunks = noun_chunks or []


class StubNLP:
    """Fake spaCy NLP that returns pre-configured entities and noun chunks."""

    def __init__(
        self,
        entity_map: Dict[str, List[tuple]],
        chunk_map: Dict[str, List[str]] | None = None,
    ) -> None:
        # entity_map: text -> [(entity_text, label), ...]
        # chunk_map:  text -> [noun_chunk_text, ...]
        self._map = entity_map
        self._chunk_map: Dict[str, List[str]] = chunk_map or {}

    def __call__(self, text: str) -> _FakeDoc:
        pairs = self._map.get(text, [])
        ents = [_FakeEnt(e, lbl) for e, lbl in pairs]
        chunks = [_FakeChunk(c) for c in self._chunk_map.get(text, [])]
        return _FakeDoc(ents, chunks)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    GraphRetriever._reset()  # type: ignore[reportPrivateUsage]
    yield
    GraphRetriever._reset()  # type: ignore[reportPrivateUsage]


def _make_retriever(nlp: Any = None, **overrides: Any) -> GraphRetriever:
    """Create a GraphRetriever with stub deps (bypasses real spaCy load)."""
    if nlp is None:
        nlp = StubNLP({})
    r = GraphRetriever.__new__(GraphRetriever)
    r._initialized = True
    r.cfg = overrides.get("cfg", StubConfig())
    r.pretty = overrides.get("pretty", StubPrettyWriter())
    r._nlp = nlp
    r._data = _GraphIndexData()
    r._entity_types = overrides.get(
        "entity_types", ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"]
    )
    r._max_hops = overrides.get("max_hops", 2)
    r._max_candidates = overrides.get("max_candidates", 50)
    r._min_edge_weight = overrides.get("min_edge_weight", 1)
    r._spacy_model = "en_core_web_sm"
    r._noun_chunk_min_chars = overrides.get("noun_chunk_min_chars", 3)
    r._noun_chunk_drop_leading = overrides.get("noun_chunk_drop_leading", "[({<")
    # Register as the singleton instance
    GraphRetriever._instance = r  # type: ignore[reportPrivateUsage]
    return r


def _add(r: GraphRetriever, chunks: List[Dict[str, Any]]) -> None:
    """Add chunks (list of {id, text, meta}) and mark collection name."""
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]
    r.add_chunks(ids, texts, metas)
    r._data.collection_name = "test_coll"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

ENTITY_MAP: Dict[str, List[tuple]] = {
    "Apple is led by Tim Cook in Cupertino.": [
        ("Apple", "ORG"),
        ("Tim Cook", "PERSON"),
        ("Cupertino", "GPE"),
    ],
    "Tim Cook announced new products at WWDC.": [
        ("Tim Cook", "PERSON"),
        ("WWDC", "ORG"),
    ],
    "Google and Microsoft compete globally.": [("Google", "ORG"), ("Microsoft", "ORG")],
    "No entities here.": [],
    "Tim Cook": [("Tim Cook", "PERSON")],
    "Apple": [("Apple", "ORG")],
    "Google": [("Google", "ORG")],
}

CHUNKS = [
    {
        "id": "c1",
        "text": "Apple is led by Tim Cook in Cupertino.",
        "meta": {"FilePath": "tech.txt", "FileName": "tech.txt"},
    },
    {
        "id": "c2",
        "text": "Tim Cook announced new products at WWDC.",
        "meta": {"FilePath": "tech.txt", "FileName": "tech.txt"},
    },
    {
        "id": "c3",
        "text": "Google and Microsoft compete globally.",
        "meta": {"FilePath": "market.txt", "FileName": "market.txt"},
    },
    {
        "id": "c4",
        "text": "No entities here.",
        "meta": {"FilePath": "other.txt", "FileName": "other.txt"},
    },
]


# ===========================================================================
# _GraphIndexData
# ===========================================================================


class TestGraphIndexData:
    def test_initial_state(self):
        data = _GraphIndexData()
        assert data.chunk_entities == {}
        assert data.chunk_metas == {}
        assert data.chunk_texts == {}
        assert data.adjacency == {}
        assert data.entity_to_chunks == {}
        assert data.collection_name == ""
        assert data.doc_count_at_build == 0


# ===========================================================================
# Singleton
# ===========================================================================


class TestSingleton:
    def test_is_loaded_for_empty(self):
        r = _make_retriever()
        assert r.is_loaded_for("test_coll") is False

    def test_is_loaded_for_after_add(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])
        assert r.is_loaded_for("test_coll") is True
        assert r.is_loaded_for("other_coll") is False

    def test_reset_clears_instance(self):
        r = _make_retriever()
        GraphRetriever._reset()  # type: ignore[reportPrivateUsage]
        assert GraphRetriever._instance is None  # type: ignore[reportPrivateUsage]


# ===========================================================================
# add_chunks + _rebuild_derived
# ===========================================================================


class TestAddChunks:
    def test_chunk_entities_stored(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])
        # c1 has Apple, Tim Cook, Cupertino — all lowercased
        assert set(r._data.chunk_entities["c1"]) == {"apple", "tim cook", "cupertino"}

    def test_entity_to_chunks_mapping(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        # "tim cook" appears in both c1 and c2
        assert set(r._data.entity_to_chunks["tim cook"]) == {"c1", "c2"}

    def test_adjacency_co_occurrence(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])  # c1: apple, tim cook, cupertino — all co-occur once
        assert r._data.adjacency["apple"]["tim cook"] == 1
        assert r._data.adjacency["tim cook"]["cupertino"] == 1
        assert r._data.adjacency["apple"]["cupertino"] == 1

    def test_adjacency_accumulates_across_chunks(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        # "tim cook" + "apple" co-occur in c1; "tim cook" + "wwdc" in c2
        assert r._data.adjacency["tim cook"]["apple"] == 1
        assert r._data.adjacency["tim cook"]["wwdc"] == 1

    def test_chunk_with_no_entities(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, [CHUNKS[3]])  # "No entities here."
        assert r._data.chunk_entities["c4"] == []
        # No adjacency entries for entity-free chunk
        assert (
            "c4" not in r._data.entity_to_chunks
            or r._data.entity_to_chunks.get("c4") == []
        )

    def test_chunk_metas_stored(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS)
        assert r._data.chunk_metas["c3"]["FilePath"] == "market.txt"

    def test_chunk_texts_stored(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])
        assert r._data.chunk_texts["c1"] == CHUNKS[0]["text"]


# ===========================================================================
# _rebuild_derived
# ===========================================================================


class TestRebuildDerived:
    def test_adjacency_symmetric(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])
        for ent_a, neighbours in r._data.adjacency.items():
            for ent_b, weight in neighbours.items():
                assert (
                    r._data.adjacency[ent_b][ent_a] == weight
                ), f"Asymmetric edge: {ent_a} <-> {ent_b}"

    def test_rebuild_idempotent(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        adj_before = {k: dict(v) for k, v in r._data.adjacency.items()}
        r._rebuild_derived()
        assert r._data.adjacency == adj_before


# ===========================================================================
# remove_by_filepath
# ===========================================================================


class TestRemoveByFilepath:
    def test_removes_correct_chunks(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS)
        r.remove_by_filepath("tech.txt")
        assert "c1" not in r._data.chunk_entities
        assert "c2" not in r._data.chunk_entities
        assert "c3" in r._data.chunk_entities  # unaffected

    def test_derived_rebuilt_after_remove(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        # tim cook appears in both c1 (tech.txt) and c2 (tech.txt)
        r.remove_by_filepath("tech.txt")
        assert "tim cook" not in r._data.entity_to_chunks
        assert "tim cook" not in r._data.adjacency

    def test_noop_when_filepath_missing(self):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        count_before = len(r._data.chunk_entities)
        r.remove_by_filepath("nonexistent.txt")
        assert len(r._data.chunk_entities) == count_before

    def test_partial_remove_preserves_shared_entity(self):
        """After removing c1 only, 'tim cook' should still map to c2."""
        # c1 and c2 are both in tech.txt — remove all at once via filepath
        # So make a separate chunk in another file that also has tim cook
        extra_entity_map = {
            **ENTITY_MAP,
            "Tim Cook visits London.": [("Tim Cook", "PERSON"), ("London", "GPE")],
        }
        extra_chunk = {
            "id": "c5",
            "text": "Tim Cook visits London.",
            "meta": {"FilePath": "news.txt", "FileName": "news.txt"},
        }
        nlp = StubNLP(extra_entity_map)
        r = _make_retriever(nlp=nlp)
        _add(r, [CHUNKS[0], extra_chunk])  # c1 (tech.txt) + c5 (news.txt)
        r.remove_by_filepath("tech.txt")
        # tim cook still in c5
        assert "tim cook" in r._data.entity_to_chunks
        assert "c5" in r._data.entity_to_chunks["tim cook"]
        assert "c1" not in r._data.chunk_entities


# ===========================================================================
# query
# ===========================================================================


class TestQuery:
    def test_returns_empty_when_no_index(self):
        r = _make_retriever(nlp=StubNLP({}))
        docs = r.query("Apple CEO", k=10)
        assert docs == []

    def test_returns_empty_on_no_entity_seeds(self):
        nlp = StubNLP({**ENTITY_MAP, "no match query": []})
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])
        docs = r.query("no match query", k=10)
        assert docs == []

    def test_1_hop_retrieval(self):
        """Query for 'Apple' should find c1 (contains apple entity)."""
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=1)
        _add(r, CHUNKS[:2])
        docs = r.query("Apple", k=10)
        doc_ids = {d.id for d in docs}
        assert "c1" in doc_ids

    def test_2_hop_retrieval(self):
        """'Apple' -> 'tim cook' (1-hop) -> c2 (2-hop via tim cook)."""
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2)
        _add(r, CHUNKS[:2])
        docs = r.query("Apple", k=10)
        doc_ids = {d.id for d in docs}
        # c1 is direct hit; c2 is reachable via tim cook
        assert "c1" in doc_ids
        assert "c2" in doc_ids

    def test_score_ordering(self):
        """Chunk with more entity overlap should rank higher."""
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2)
        _add(r, CHUNKS[:2])
        docs = r.query("Tim Cook", k=10)
        # Both c1 and c2 have tim cook; scores should be equal or c1 higher (3 entities vs 2)
        assert len(docs) >= 1
        scores = [d.metadata["graph_score"] for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_graph_score_in_metadata(self):
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2)
        _add(r, CHUNKS[:1])
        docs = r.query("Apple", k=10)
        assert len(docs) >= 1
        for d in docs:
            assert "graph_score" in d.metadata
            assert "bm25_score" in d.metadata
            assert d.metadata["bm25_score"] == 0.0
            assert "chroma_score" in d.metadata

    def test_file_filter(self):
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2)
        _add(r, CHUNKS[:3])
        filt = {"FilePath": {"$eq": "market.txt"}}
        docs = r.query("Google", k=10, file_filter=filt)
        for d in docs:
            assert d.metadata["FilePath"] == "market.txt"

    def test_k_cap(self):
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2)
        _add(r, CHUNKS[:3])
        docs = r.query("Tim Cook", k=1)
        assert len(docs) <= 1

    def test_min_edge_weight_prunes_hops(self):
        """With min_edge_weight=99, no edges qualify → 2-hop expansion fails."""
        r = _make_retriever(nlp=StubNLP(ENTITY_MAP), max_hops=2, min_edge_weight=99)
        _add(r, CHUNKS[:2])
        docs = r.query("Apple", k=10)
        doc_ids = {d.id for d in docs}
        # c1 is a direct hit (seed 'apple' maps to c1), but c2 shouldn't be reached
        # because the edge apple->tim cook has weight 1 < 99
        assert "c2" not in doc_ids


# ===========================================================================
# Persistence (gzip pickle round-trip)
# ===========================================================================


class TestPersistLoad:
    def test_round_trip(self, tmp_path):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:2])

        graph_dir = str(tmp_path)
        r.persist(graph_dir)

        index_path = os.path.join(graph_dir, GraphRetriever.INDEX_FILENAME)
        assert os.path.isfile(index_path)

        # Load into a fresh retriever
        GraphRetriever._reset()  # type: ignore[reportPrivateUsage]
        r2 = _make_retriever(nlp=nlp)
        r2._load(index_path)

        assert r2._data.collection_name == r._data.collection_name
        assert set(r2._data.chunk_entities.keys()) == {"c1", "c2"}
        assert r2._data.adjacency == r._data.adjacency

    def test_index_filename_constant(self):
        assert GraphRetriever.INDEX_FILENAME == "graph_index.pkl.gz"

    def test_persist_creates_directories(self, tmp_path):
        nlp = StubNLP(ENTITY_MAP)
        r = _make_retriever(nlp=nlp)
        _add(r, CHUNKS[:1])
        nested = str(tmp_path / "deep" / "nested")
        r.persist(nested)
        assert os.path.isfile(os.path.join(nested, GraphRetriever.INDEX_FILENAME))


# ===========================================================================
# NOUN_CHUNK sentinel
# ===========================================================================


class TestNounChunk:
    """NOUN_CHUNK in entity_types enables noun-phrase extraction alongside NER."""

    _CHUNK_MAP: Dict[str, List[str]] = {
        "The hedgehog is a spiny mammal.": ["the hedgehog", "a spiny mammal"],
        "A fox lives in the forest.": ["a fox", "the forest"],
        "The hedgehog": ["the hedgehog"],
    }
    _CHUNK_ENTITY_MAP: Dict[str, List[tuple]] = {}  # no named entities in these texts

    _NC_CHUNKS = [
        {
            "id": "nc1",
            "text": "The hedgehog is a spiny mammal.",
            "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
        },
        {
            "id": "nc2",
            "text": "A fox lives in the forest.",
            "meta": {"FilePath": "animals.txt", "FileName": "animals.txt"},
        },
    ]

    def test_noun_chunks_indexed(self):
        nlp = StubNLP(self._CHUNK_ENTITY_MAP, self._CHUNK_MAP)
        r = _make_retriever(nlp=nlp, entity_types=["NOUN_CHUNK"])
        _add(r, self._NC_CHUNKS)
        assert "the hedgehog" in r._data.chunk_entities["nc1"]
        assert "a spiny mammal" in r._data.chunk_entities["nc1"]

    def test_noun_chunk_query_finds_chunk(self):
        nlp = StubNLP(self._CHUNK_ENTITY_MAP, self._CHUNK_MAP)
        r = _make_retriever(nlp=nlp, entity_types=["NOUN_CHUNK"])
        _add(r, self._NC_CHUNKS)
        docs = r.query("The hedgehog", k=10)
        doc_ids = {d.id for d in docs}
        assert "nc1" in doc_ids

    def test_noun_chunk_mixed_with_ner(self):
        """NOUN_CHUNK and NER labels can coexist."""
        combined_entity_map = {
            **self._CHUNK_ENTITY_MAP,
            "Apple makes the best hedgehog apps.": [("Apple", "ORG")],
            "Apple": [("Apple", "ORG")],
        }
        combined_chunk_map = {
            **self._CHUNK_MAP,
            "Apple makes the best hedgehog apps.": ["the best hedgehog apps"],
        }
        mixed_chunk = {
            "id": "nc3",
            "text": "Apple makes the best hedgehog apps.",
            "meta": {"FilePath": "tech.txt", "FileName": "tech.txt"},
        }
        nlp = StubNLP(combined_entity_map, combined_chunk_map)
        r = _make_retriever(nlp=nlp, entity_types=["ORG", "NOUN_CHUNK"])
        _add(r, [mixed_chunk])
        ents = r._data.chunk_entities["nc3"]
        assert "apple" in ents  # from NER
        assert "the best hedgehog apps" in ents  # from noun chunk

    def test_noun_chunks_deduplicated(self):
        """Same noun phrase appearing as both NER and noun chunk is not duplicated."""
        both_map = {"Apple Inc. is a company.": [("Apple Inc.", "ORG")]}
        both_chunk_map = {"Apple Inc. is a company.": ["apple inc.", "a company"]}
        chunk = {
            "id": "dup1",
            "text": "Apple Inc. is a company.",
            "meta": {"FilePath": "f.txt", "FileName": "f.txt"},
        }
        nlp = StubNLP(both_map, both_chunk_map)
        r = _make_retriever(nlp=nlp, entity_types=["ORG", "NOUN_CHUNK"])
        _add(r, [chunk])
        ents = r._data.chunk_entities["dup1"]
        assert ents.count("apple inc.") == 1

    def test_without_noun_chunk_sentinel_ignores_chunks(self):
        """Without NOUN_CHUNK sentinel, noun chunks are not extracted."""
        nlp = StubNLP(self._CHUNK_ENTITY_MAP, self._CHUNK_MAP)
        r = _make_retriever(nlp=nlp, entity_types=["PERSON", "ORG"])  # no NOUN_CHUNK
        _add(r, self._NC_CHUNKS)
        # No NER entities + no noun chunks → empty entity lists
        assert r._data.chunk_entities["nc1"] == []
        assert r._data.chunk_entities["nc2"] == []
