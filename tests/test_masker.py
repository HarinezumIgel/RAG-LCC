# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false
"""
Tests for Masker.mask() — pure regex-based text masking with pre-set _specs.
"""

import re
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Algos.Masker import Masker

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubConfig:
    def get(self, key, default=None):
        mapping = {
            "DEBUG_LEVEL": 0,
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


def _make_masker(specs, apply_masking=True):
    """Build a Masker singleton with injected _specs, bypassing config-driven init."""
    m = Masker.__new__(Masker)
    m._initialized = True
    m.cfg = StubConfig()
    m.pretty = StubPrettyWriter()
    m.apply_masking = apply_masking
    # _specs is a list of (compiled_pattern, replacement_string, rule_name)
    m._specs = specs
    return m


# ---------------------------------------------------------------------------
# mask()
# ---------------------------------------------------------------------------


class TestMask:
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
