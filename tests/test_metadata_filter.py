# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for metadata-based retrieval filtering and the answer metadata block."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)


def _make_doc(meta):
    class _Doc:
        def __init__(self, m):
            self.metadata = m
            self.page_content = "x"

    return _Doc(meta)


# ---------------------------------------------------------------------------
# RAGChatImpl._chroma_where — flat filter → ChromaDB $and normalization
# ---------------------------------------------------------------------------


class TestChromaWhere:
    def _fn(self):
        from Chat.RAGChatImpl import RAGChatImpl

        return RAGChatImpl._chroma_where

    def test_none_passthrough(self):
        assert self._fn()(None) is None

    def test_empty_passthrough(self):
        assert self._fn()({}) == {}

    def test_single_field_unchanged(self):
        flt = {"Author": {"$eq": "x"}}
        assert self._fn()(flt) == flt

    def test_multi_field_wrapped_in_and(self):
        flt = {"Author": {"$eq": "x"}, "DocTitle": {"$eq": "y"}}
        assert self._fn()(flt) == {
            "$and": [{"Author": {"$eq": "x"}}, {"DocTitle": {"$eq": "y"}}]
        }


# ---------------------------------------------------------------------------
# Helpers.build_document_metadata_md — answer metadata section
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def helpers():
    from Helpers.Helpers import Helpers

    return Helpers()


class TestMetadataBlock:
    def test_shows_fields_and_pages(self, helpers):
        docs = [
            _make_doc(
                {
                    "FileName": "a.pdf",
                    "FilePath": "/docs/a.pdf",
                    "Author": "Jane",
                    "PageNumber": 1,
                    "PageLabel": "i",
                }
            )
        ]
        md = helpers.build_document_metadata_md(docs)
        assert "### Document metadata" in md
        assert "a.pdf" in md
        assert "Author: Jane" in md
        assert "Pages: i" in md  # printed label preferred over physical number

    def test_empty_attributes_skipped(self, helpers):
        docs = [
            _make_doc(
                {
                    "FileName": "a.pdf",
                    "FilePath": "/docs/a.pdf",
                    "Author": "",
                    "DocTitle": "Title",
                }
            )
        ]
        md = helpers.build_document_metadata_md(docs)
        assert "Author" not in md
        assert "DocTitle: Title" in md

    def test_all_files_listed_even_without_metadata(self, helpers):
        docs = [_make_doc({"FileName": "bare.txt", "FilePath": "/docs/bare.txt"})]
        md = helpers.build_document_metadata_md(docs)
        assert "bare.txt" in md

    def test_web_sources_skipped(self, helpers):
        docs = [
            _make_doc({"FileName": "site", "FilePath": "http://x", "Source": "web"})
        ]
        md = helpers.build_document_metadata_md(docs)
        assert md == ""

    def test_empty_chosen_returns_empty(self, helpers):
        assert helpers.build_document_metadata_md([]) == ""


# ---------------------------------------------------------------------------
# MetadataPicker._collect_fields — aggregate fields, exclude internal keys
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self, include=None, limit=None):
        return {"metadatas": self._metadatas}


class TestMetadataPickerCollect:
    def test_aggregates_and_excludes_internal(self, monkeypatch):
        from Gui.MetadataPicker import MetadataPicker

        MetadataPicker._reset()
        picker = MetadataPicker()
        try:
            fake = _FakeCollection(
                [
                    {"Author": "A", "FileHash": "h", "MyChunk": 0, "PageLabel": "i"},
                    {"Author": "B", "chunk_id": "c", "PageLabel": "ii", "Empty": ""},
                ]
            )
            # ROOT is a real dir so the isdir() guard passes without patching os.
            monkeypatch.setattr(
                picker.chromaDBHelper,
                "change_chroma_collection",
                lambda name: ("Test", ROOT),
            )
            monkeypatch.setattr(
                picker.chromaDBHelper,
                "get_chroma_client_and_collection",
                lambda d, n: (None, fake),
            )
            fields = picker._collect_fields("Test")
            assert fields["Author"] == {"A", "B"}
            assert fields["PageLabel"] == {"i", "ii"}
            # Internal / empty-value keys excluded.
            for excluded in ("FileHash", "MyChunk", "chunk_id", "Empty"):
                assert excluded not in fields
        finally:
            MetadataPicker._reset()


# ---------------------------------------------------------------------------
# BM25 + Graph apply the same flat metadata filter (parity with vector path)
# ---------------------------------------------------------------------------


class TestRetrieverMatchers:
    """The combined file+metadata filter is a flat multi-key dict that both
    BM25 and Graph AND via their identical _matches_filter predicate."""

    def _matchers(self):
        from Strategies.BM25Retriever import BM25Retriever
        from Strategies.GraphRetriever import GraphRetriever

        return BM25Retriever._matches_filter, GraphRetriever._matches_filter

    def test_metadata_key_matches(self):
        for match in self._matchers():
            assert match({"Author": "Jane"}, {"Author": {"$eq": "Jane"}}) is True
            assert match({"Author": "Bob"}, {"Author": {"$eq": "Jane"}}) is False

    def test_missing_metadata_key_rejected(self):
        for match in self._matchers():
            assert match({"FileName": "a.pdf"}, {"Author": {"$eq": "Jane"}}) is False

    def test_combined_file_and_metadata_anded(self):
        flt = {"FileName": {"$eq": "a.pdf"}, "Author": {"$eq": "Jane"}}
        meta_ok = {"FileName": "a.pdf", "Author": "Jane"}
        meta_wrong_author = {"FileName": "a.pdf", "Author": "Bob"}
        meta_wrong_file = {"FileName": "b.pdf", "Author": "Jane"}
        for match in self._matchers():
            assert match(meta_ok, flt) is True
            assert match(meta_wrong_author, flt) is False
            assert match(meta_wrong_file, flt) is False

    def test_bm25_and_graph_parity(self):
        bm25_match, graph_match = self._matchers()
        cases = [
            ({"Author": "X", "DocCreated": "2020"}, {"Author": {"$eq": "X"}}),
            ({"Author": "X"}, {"Author": {"$eq": "X"}, "DocCreated": {"$eq": "2020"}}),
            ({"PageLabel": "iii"}, {"PageLabel": {"$eq": "iii"}}),
        ]
        for meta, flt in cases:
            assert bm25_match(meta, flt) == graph_match(meta, flt)
