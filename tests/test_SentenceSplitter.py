# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false
"""Tests for the SentenceSplitter class."""

import sys
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from Strategies.Chunkers.SentenceSplitter import SentenceSplitter as _SentenceSplitter

# split_sentences is now an instance method; use a shared singleton instance so
# the existing static-style call sites below keep working unchanged.
SentenceSplitter = _SentenceSplitter()


# ── Classic prose boundaries ──────────────────────────────────────────────


class TestProseBoundaries:
    def test_period(self):
        assert SentenceSplitter.split_sentences("Hello world. Foo bar.") == [
            "Hello world.",
            "Foo bar.",
        ]

    def test_question_mark(self):
        assert SentenceSplitter.split_sentences("What? Why?") == ["What?", "Why?"]

    def test_exclamation(self):
        assert SentenceSplitter.split_sentences("Wow! Amazing!") == ["Wow!", "Amazing!"]

    def test_mixed_punctuation(self):
        result = SentenceSplitter.split_sentences("Hello world. How are you? Fine!")
        assert result == ["Hello world.", "How are you?", "Fine!"]

    def test_no_boundaries_single_segment(self):
        text = "just one long sentence without punctuation"
        assert SentenceSplitter.split_sentences(text) == [text]


# ── Newline boundaries ───────────────────────────────────────────────────


class TestNewlines:
    def test_single_newline_before_uppercase(self):
        # Newline before uppercase → new sentence / item → split
        result = SentenceSplitter.split_sentences("Line one\nLine two\nLine three")
        assert result == ["Line one", "Line two", "Line three"]

    def test_single_newline_before_lowercase_no_split(self):
        # Newline before lowercase → PDF line-wrap → no split
        result = SentenceSplitter.split_sentences(
            "The system has seven expansion\nslots for configuration"
        )
        assert result == ["The system has seven expansion\nslots for configuration"]

    def test_single_newline_before_digit_no_split(self):
        # Newline before digit → PDF line continuation → no split
        # (numbered items like "6. text" stay with surrounding content)
        result = SentenceSplitter.split_sentences("Items:\n1 First\n2 Second")
        assert result == ["Items:\n1 First\n2 Second"]

    def test_paragraph_breaks(self):
        result = SentenceSplitter.split_sentences(
            "Para one\n\nPara two\n\n\nPara three"
        )
        assert result == ["Para one", "Para two", "Para three"]


# ── PDF artifact boundaries ──────────────────────────────────────────────


class TestPDFArtifacts:
    def test_semicolons_preserved(self):
        # Semicolons no longer split — keeps related spec items together
        result = SentenceSplitter.split_sentences(
            "Slot 1 is PCIe x16; Slot 2 is PCIe x8"
        )
        assert result == ["Slot 1 is PCIe x16; Slot 2 is PCIe x8"]

    def test_colon_before_uppercase_preserved(self):
        # Colons no longer split — keeps label:value pairs intact
        result = SentenceSplitter.split_sentences("Memory: DDR4 ECC registered")
        assert result == ["Memory: DDR4 ECC registered"]

    def test_colon_before_lowercase_no_split(self):
        # "colon: lowercase" should NOT split — common in prose
        result = SentenceSplitter.split_sentences("such as: alpha beta gamma")
        assert len(result) == 1

    def test_double_spaces_preserved(self):
        # Double spaces no longer split — keeps PDF table rows intact
        result = SentenceSplitter.split_sentences("Column A  Column B  Column C")
        assert result == ["Column A  Column B  Column C"]

    def test_tabs_preserved(self):
        # Tabs no longer split — keeps PDF table rows intact
        result = SentenceSplitter.split_sentences("Field1\t\tField2\t\tField3")
        assert result == ["Field1\t\tField2\t\tField3"]

    def test_bullet_dash(self):
        text = "Features:\n- PCIe 4.0\n- DDR4 memory\n- NVMe storage"
        result = SentenceSplitter.split_sentences(text)
        assert "PCIe 4.0" in result
        assert "DDR4 memory" in result
        assert "NVMe storage" in result

    def test_bullet_dot(self):
        text = "Items:\n• First item\n• Second item"
        result = SentenceSplitter.split_sentences(text)
        assert "First item" in result
        assert "Second item" in result


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self):
        assert SentenceSplitter.split_sentences("") == []

    def test_whitespace_only(self):
        assert SentenceSplitter.split_sentences("   \n\n   ") == []

    def test_single_word(self):
        assert SentenceSplitter.split_sentences("Hello") == ["Hello"]

    def test_abbreviations_with_period(self):
        # "U.S." — only 1 char before each period, so no split.
        # The 2-char look-behind keeps abbreviations intact.
        result = SentenceSplitter.split_sentences("The U.S. government announced it.")
        assert result == ["The U.S. government announced it."]

    def test_mixed_pdf_and_prose(self):
        text = (
            "Chapter 3: System Board\n\n"
            "The system board has 7 PCIe slots. "
            "Slot 1 is x16; Slot 2 is x8.\n"
            "• Supports PCIe 4.0\n"
            "• Hot-plug capable"
        )
        result = SentenceSplitter.split_sentences(text)
        # Should produce multiple segments, not one giant blob
        assert len(result) >= 3

    def test_numbered_list_labels_not_orphaned(self):
        # PDF numbered-item labels like "6." must NOT become orphan chunks
        text = "6. usb connectors 7. ps/2 mouse connector 8. ethernet connector"
        result = SentenceSplitter.split_sentences(text)
        assert result == [text]

    def test_pcie_slot_spec_stays_together(self):
        # Real P620 system board text — slot specs must not fragment
        text = (
            "44 pcie 4.0 x8 card slot 2 "
            "45 pcie 4.0 x 16 card slot 1 "
            "46 memory fan connector"
        )
        result = SentenceSplitter.split_sentences(text)
        assert len(result) == 1
