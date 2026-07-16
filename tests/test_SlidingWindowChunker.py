# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for the SlidingWindowChunker."""

import sys
import os
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.Chunkers.SlidingWindowChunker import SlidingWindowChunker


def _make_chunker(max_chunk_size: int = 20, overlap_sentences: int = 2):
    """Create a SlidingWindowChunker with mocked config."""
    cfg = MagicMock()
    helpers = MagicMock()
    file_utils = MagicMock()

    helpers.get_chunker_config_slot.return_value = "_CHUNKERS.SLIDING_WINDOW"

    def get_int(key, default=0):
        if "MAX_CHUNK_SIZE" in key:
            return max_chunk_size
        if "OVERLAP_SENTENCES" in key:
            return overlap_sentences
        return default

    cfg.get_int.side_effect = get_int
    cfg.get_list.return_value = ["\n", " ", "."]

    file_utils.count_words.side_effect = lambda t: len(t.split())

    return SlidingWindowChunker(cfg=cfg, helpers=helpers, file_utils=file_utils)


META = {"FileName": "test.pdf", "FilePath": "/test.pdf", "FileType": "pdf"}


class TestSlidingWindowBasics:
    def test_empty_content(self):
        chunker = _make_chunker()
        assert chunker.chunk("", META) == ([], None)

    def test_single_sentence(self):
        chunker = _make_chunker(max_chunk_size=50)
        docs, _ = chunker.chunk("Hello world.", META)
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world."

    def test_chunk_size_property(self):
        chunker = _make_chunker(max_chunk_size=128)
        assert chunker.chunk_size == 128


class TestOverlap:
    def test_overlap_shares_sentences(self):
        # 4 short sentences, window of 10 words, overlap 1
        chunker = _make_chunker(max_chunk_size=10, overlap_sentences=1)
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
        docs, _ = chunker.chunk(text, META)

        # With overlap, later chunks should start with content from previous chunk
        contents = [d.page_content for d in docs]
        assert len(contents) >= 2

        # Each chunk after the first should share the tail of the previous
        for i in range(1, len(contents)):
            # The overlap sentence should appear at the start of next chunk
            # and at the end of the previous chunk
            prev_words = contents[i - 1].split()
            curr_words = contents[i].split()
            # There should be some word overlap
            prev_tail = (
                set(prev_words[-5:]) if len(prev_words) >= 5 else set(prev_words)
            )
            curr_head = set(curr_words[:5]) if len(curr_words) >= 5 else set(curr_words)
            assert (
                len(prev_tail & curr_head) > 0
            ), f"No overlap between chunk {i-1} tail and chunk {i} head"

    def test_no_overlap(self):
        chunker = _make_chunker(max_chunk_size=10, overlap_sentences=0)
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
        docs, _ = chunker.chunk(text, META)
        # Without overlap, should behave like SentenceWindowChunker
        assert len(docs) >= 1

    def test_overlap_larger_than_chunk_still_progresses(self):
        # overlap_sentences larger than sentences in a chunk — must still advance
        chunker = _make_chunker(max_chunk_size=5, overlap_sentences=10)
        text = "One two three. Four five six. Seven eight nine."
        docs, _ = chunker.chunk(text, META)
        # Must produce chunks and not loop forever
        assert len(docs) >= 2


class TestMetadata:
    def test_mychunk_increments(self):
        chunker = _make_chunker(max_chunk_size=10, overlap_sentences=1)
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
        docs, _ = chunker.chunk(text, META)
        for i, doc in enumerate(docs):
            assert doc.metadata["MyChunk"] == i

    def test_metadata_preserved(self):
        chunker = _make_chunker(max_chunk_size=50)
        docs, _ = chunker.chunk("Hello world.", META)
        assert docs[0].metadata["FileName"] == "test.pdf"
        assert docs[0].metadata["FileType"] == "pdf"

    def test_uuid_assigned(self):
        chunker = _make_chunker(max_chunk_size=50)
        docs, _ = chunker.chunk("Hello world.", META)
        assert docs[0].id is not None
        assert len(docs[0].id) == 36  # UUID format
