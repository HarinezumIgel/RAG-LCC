# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for Masker.mask() — pure regex-based text masking with pre-set _specs.
"""

import re
from unittest.mock import Mock

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.Masker import Masker

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def __init__(self, debug_level=0):
        self.debug_level = debug_level

    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": self.debug_level,
            "_LEET_MAP": {},
            "_CONFUSABLES": {},
        }
        return mapping.get(key, default)

    def get_int(self, key, default=0):
        return int(self.get(key, default))

    def get_str(self, key, default=""):
        return str(self.get(key, default))

    def get_bool(self, key, default=False):
        return bool(self.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.get(key, default))

    def get_list(self, key, default=None):
        return self.get(key, default if default is not None else [])

    def get_dict(self, key, default=None):
        return self.get(key, default if default is not None else {})


class StubPrettyWriter:
    def write(self, *a, **kw):
        return None


@pytest.fixture(autouse=True)
def reset_masker():
    Masker._reset()
    yield
    Masker._reset()


def _make_masker(specs, apply_masking=True, debug_level=0):
    """Build a Masker singleton with injected _specs, bypassing config-driven init."""
    m = Masker.__new__(Masker)
    m._initialized = True
    m.cfg = StubConfig(debug_level)
    m.pretty = StubPrettyWriter()
    m.apply_masking = apply_masking
    # _specs is a list of (compiled_pattern, replacement_string, rule_name)
    m.specs = specs
    return m


# ---------------------------------------------------------------------------
# mask()
# ---------------------------------------------------------------------------


class TestMask:
    @pytest.mark.parametrize(
        ("debug_level", "visible"),
        [
            (0, False),
            (30, False),
            (39, False),
            (40, True),
            (41, True),
            ("le 39", False),
            ("le 40", True),
            ("le 41", True),
            ("is 30", False),
            ("is 40", True),
            ("is 41", False),
        ],
    )
    @pytest.mark.parametrize(
        ("text", "expected", "matches"),
        [("hello world", "hello world", 0), ("secret", "***", 1)],
    )
    def test_match_summary_respects_debug_level(
        self, debug_level, visible, text, expected, matches, monkeypatch
    ):
        specs = [(re.compile(r"secret"), "***", "secret")]
        m = _make_masker(specs, debug_level=debug_level)
        write = Mock()
        monkeypatch.setattr(m.pretty, "write", write)

        assert m.mask(text) == expected

        summaries = [
            call.args
            for call in write.call_args_list
            if "rules produced matches and were replaced" in call.args[2]
        ]
        assert summaries == (
            [
                (
                    "I",
                    "Masker",
                    f"{matches} of 1 rules produced matches and were replaced",
                )
            ]
            if visible
            else []
        )

    def test_no_specs_returns_unchanged(self):
        m = _make_masker([])
        assert m.mask("hello world") == "hello world"

    def test_single_rule_applied(self):
        specs = [(re.compile(r"\d{3}-\d{2}-\d{4}"), "***-**-****", "ssn")]
        m = _make_masker(specs)
        assert m.mask("SSN: 123-45-6789") == "SSN: ***-**-****"

    def test_multiple_rules_applied_in_order(self):
        specs = [
            (re.compile(r"\bpassword\b", re.IGNORECASE), "[REDACTED]", "password"),
            (re.compile(r"\b\d{4}\b"), "****", "four_digits"),
        ]
        m = _make_masker(specs)
        result = m.mask("password is 1234")
        assert result == "[REDACTED] is ****"

    def test_apply_masking_false_skips(self):
        specs = [(re.compile(r"secret"), "***", "secret")]
        m = _make_masker(specs, apply_masking=False)
        assert m.mask("this is secret") == "this is secret"

    def test_empty_text_returns_empty(self):
        specs = [(re.compile(r"x"), "y", "rule")]
        m = _make_masker(specs)
        assert m.mask("") == ""

    def test_email_masking(self):
        specs = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]", "email")]
        m = _make_masker(specs)
        assert m.mask("Contact user@example.com please") == "Contact [EMAIL] please"

    def test_overlapping_patterns(self):
        # First rule masks digits, second rule tries to mask SSN (already masked)
        specs = [
            (re.compile(r"\d"), "#", "digits"),
            (re.compile(r"\d{3}-\d{2}-\d{4}"), "[SSN]", "ssn"),
        ]
        m = _make_masker(specs)
        result = m.mask("SSN: 123-45-6789")
        # digits rule fires first, replacing all digits with #
        assert result == "SSN: ###-##-####"
