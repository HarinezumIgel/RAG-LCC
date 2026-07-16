# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportReturnType=false
"""
Tests for FileUtils lingua-based language detection.

Covers _detect_lang_iso, get_text_language, and get_user_text_language.

The real lingua LanguageDetector (expensive singleton) is never loaded.
A fake detector is injected directly into the FileUtils class variable
_lingua_detector so each test runs in microseconds.

Config keys exercised:
    _LANGUAGE_DETECTION.MIN_WORDS          (default 3)
    _LANGUAGE_DETECTION.MIN_CONFIDENCE     (default 0.60)
    _LANGUAGE_DETECTION.CONF_FULL_WORDS    (default 10)
    _ARGOS_DEFINITIONS.LANG_CODE_TO_NAME
"""

import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Helpers.FileUtils import FileUtils

# ---------------------------------------------------------------------------
# Fake lingua LanguageDetector infrastructure
# ---------------------------------------------------------------------------


class _FakeIso:
    """Mimics lingua's IsoCode639_1 enum: .name returns the uppercase code."""

    def __init__(self, name: str) -> None:
        self.name = name.upper()


class _FakeLang:
    def __init__(self, iso: str) -> None:
        self.iso_code_639_1 = _FakeIso(iso)


class _FakeLinguaResult:
    def __init__(self, iso: str, value: float) -> None:
        self.language = _FakeLang(iso)
        self.value = value


class _FakeDetector:
    """Returns a fixed ranked list regardless of input text."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    def compute_language_confidence_values(self, text: str) -> list[_FakeLinguaResult]:
        return [_FakeLinguaResult(iso, prob) for iso, prob in self._results]


class _BrokenDetector:
    """Always raises to simulate a detector crash."""

    def compute_language_confidence_values(self, text: str) -> list[Any]:
        raise RuntimeError("simulated detector failure")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubPrettyWriter:
    def __init__(self) -> None:
        self.messages: list[tuple[Any, ...]] = []

    def write(self, *a: Any, **kw: Any) -> None:
        self.messages.append(a)


LANG_CODE_TO_NAME = {
    "de": "german",
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "nl": "dutch",
    "it": "italian",
    "af": "afrikaans",
}


class StubConfig:
    """Minimal config stub for language-detection tests."""

    def __init__(
        self,
        min_words: int = 3,
        min_confidence: float = 0.60,
        conf_full_words: int = 10,
    ) -> None:
        self._ld: dict[str, Any] = {
            "_LANGUAGE_DETECTION.MIN_WORDS": min_words,
            "_LANGUAGE_DETECTION.MIN_CONFIDENCE": min_confidence,
            "_LANGUAGE_DETECTION.CONF_FULL_WORDS": conf_full_words,
        }

    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self._ld.get(key, default))

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self._ld.get(key, default))

    def get_dict(self, key: str, default: Any = None) -> dict:
        if key == "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME":
            return LANG_CODE_TO_NAME
        return default or {}

    def get_str(self, key: str, default: str = "") -> str:
        return default

    def get(self, key: Any, default: Any = None) -> Any:
        return default


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _make_fu(
    detector_results: list[tuple[str, float]] | None = None,
    min_words: int = 3,
    min_confidence: float = 0.60,
    conf_full_words: int = 10,
) -> FileUtils:
    """Build a FileUtils with injected stubs.  The real lingua model is never loaded."""
    fu = object.__new__(FileUtils)
    fu.pretty = StubPrettyWriter()
    fu.cfg = StubConfig(min_words, min_confidence, conf_full_words)
    FileUtils._lingua_detector = _FakeDetector(detector_results or [])
    # Prevent auto-detection of installed Argos codes (SharedHelpers not available).
    FileUtils._argos_codes = set()
    # Reset reverse-name lookup cache to avoid cross-test pollution.
    FileUtils._name_to_code = None
    return fu


@pytest.fixture(autouse=True)
def _reset_class_state():
    """Restore FileUtils class-level singletons before and after every test."""
    original_detector = FileUtils._lingua_detector
    original_codes = FileUtils._argos_codes
    original_name_to_code = FileUtils._name_to_code
    yield
    FileUtils._lingua_detector = original_detector
    FileUtils._argos_codes = original_codes
    FileUtils._name_to_code = original_name_to_code


# ===========================================================================
# _detect_lang_iso
# ===========================================================================


class TestDetectLangIso:

    # --- too-short branch ---------------------------------------------------

    def test_single_word_is_too_short(self):
        fu = _make_fu()
        lang, fell_back, too_short, det, conf, eff_conf = fu._detect_lang_iso("Test")
        assert lang == "en"
        assert fell_back is True
        assert too_short is True
        assert det == []
        assert conf == 0.0

    def test_two_words_is_too_short(self):
        fu = _make_fu()
        lang, fell_back, too_short, *_ = fu._detect_lang_iso("hello world")
        assert too_short is True
        assert lang == "en"

    def test_exactly_min_words_is_not_too_short(self):
        """3 words == MIN_WORDS; detection is attempted (may still fall back on conf)."""
        fu = _make_fu(detector_results=[("en", 0.95)])
        lang, fell_back, too_short, *_ = fu._detect_lang_iso("do bees sting")
        assert too_short is False

    # --- effective_conf scaling ---------------------------------------------

    def test_at_min_words_threshold_is_0_90(self):
        """At exactly MIN_WORDS the effective threshold should be 0.90."""
        fu = _make_fu(detector_results=[("de", 0.85)])
        _, _, _, _, _, eff_conf = fu._detect_lang_iso("do bees sting")  # 3 words
        assert eff_conf == pytest.approx(0.90)

    def test_at_conf_full_words_threshold_is_min_confidence(self):
        """At CONF_FULL_WORDS the effective threshold equals MIN_CONFIDENCE."""
        fu = _make_fu(detector_results=[("de", 0.65)])
        # 10 words
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        _, _, _, _, _, eff_conf = fu._detect_lang_iso(text)
        assert eff_conf == pytest.approx(0.60)

    def test_midpoint_threshold_interpolates_linearly(self):
        """At 6 words (midpoint between 3 and 10) the threshold is interpolated."""
        # t = (6-3)/(10-3) = 3/7 ≈ 0.4286
        # effective_conf = 0.90 - 0.4286 * (0.90 - 0.60) ≈ 0.7714
        fu = _make_fu(detector_results=[("de", 0.80)])
        text = "sind dies saeugetiere und welche tiere"  # 6 words
        _, _, _, _, _, eff_conf = fu._detect_lang_iso(text)
        expected_t = (6 - 3) / (10 - 3)
        expected_eff = 0.90 - expected_t * (0.90 - 0.60)
        assert eff_conf == pytest.approx(expected_eff, abs=1e-6)

    def test_beyond_conf_full_words_clamps_to_min_confidence(self):
        """More than CONF_FULL_WORDS words still gets exactly MIN_CONFIDENCE."""
        fu = _make_fu(detector_results=[("de", 0.65)])
        text = "das ist ein sehr langer text damit wir genug woerter haben hier heute"
        _, _, _, _, _, eff_conf = fu._detect_lang_iso(text)
        assert eff_conf == pytest.approx(0.60)

    # --- detection outcomes -------------------------------------------------

    def test_confident_detection_accepted(self):
        """Confidence above threshold → language accepted."""
        fu = _make_fu(detector_results=[("de", 0.70)])
        text = (
            "was sind saeugetiere und welche tiere gehoeren dazu und warum"  # 10 words
        )
        lang, fell_back, *_ = fu._detect_lang_iso(text)
        assert lang == "de"
        assert fell_back is False

    def test_short_text_high_bar_rejects_moderate_confidence(self):
        """3-word text requires 0.90; confidence of 0.85 should fall back."""
        fu = _make_fu(detector_results=[("de", 0.85)])
        lang, fell_back, *_ = fu._detect_lang_iso("do bees sting")
        assert lang == "en"
        assert fell_back is True

    def test_low_confidence_falls_back_to_english(self):
        """Confidence well below threshold → fall back to English."""
        fu = _make_fu(detector_results=[("de", 0.30)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        lang, fell_back, too_short, *_ = fu._detect_lang_iso(text)
        assert lang == "en"
        assert fell_back is True
        assert too_short is False

    def test_detector_exception_falls_back_gracefully(self):
        """Any exception from the detector silently falls back to English."""
        fu = object.__new__(FileUtils)
        fu.pretty = StubPrettyWriter()
        fu.cfg = StubConfig()
        FileUtils._lingua_detector = _BrokenDetector()
        FileUtils._argos_codes = set()
        FileUtils._name_to_code = None

        lang, fell_back, too_short, det, *_ = fu._detect_lang_iso(
            "das ist ein test und wir haben viele woerter im text hier"
        )
        assert lang == "en"
        assert fell_back is True
        assert det == []

    def test_empty_detector_results_falls_back(self):
        """Empty detection list (e.g. non-alphabetic input) → English fallback."""
        fu = _make_fu(detector_results=[])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        lang, fell_back, *_ = fu._detect_lang_iso(text)
        assert lang == "en"
        assert fell_back is True

    # --- runner-up promotion ------------------------------------------------

    def test_runner_up_promoted_when_top_not_installed(self):
        """Top language not installed; close runner-up in installed_codes is promoted."""
        fu = _make_fu(detector_results=[("nl", 0.80), ("de", 0.50)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        lang, *_ = fu._detect_lang_iso(text, installed_codes={"de"})
        assert lang == "de"

    def test_runner_up_not_promoted_when_too_weak(self):
        """Runner-up below 50 % of top's confidence is not promoted."""
        # 0.35 < 0.80 * 0.50 = 0.40 → not promoted
        fu = _make_fu(detector_results=[("nl", 0.80), ("de", 0.35)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        lang, *_ = fu._detect_lang_iso(text, installed_codes={"de"})
        assert lang == "nl"

    def test_runner_up_not_promoted_when_only_one_result(self):
        """Exactly one detection result; runner-up logic requires at least two."""
        fu = _make_fu(detector_results=[("nl", 0.80)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        lang, *_ = fu._detect_lang_iso(text, installed_codes={"de"})
        assert lang == "nl"

    def test_no_runner_up_when_installed_codes_empty(self):
        """Empty installed_codes set disables promotion (falsy guard)."""
        fu = _make_fu(detector_results=[("nl", 0.80), ("de", 0.50)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        lang, *_ = fu._detect_lang_iso(text, installed_codes=set())
        assert lang == "nl"


# ===========================================================================
# get_text_language
# ===========================================================================


class TestGetTextLanguage:

    def test_too_short_returns_en_iso(self):
        fu = _make_fu()
        assert fu.get_text_language("Test", output="iso-639") == "en"

    def test_too_short_returns_english_nltk(self):
        fu = _make_fu()
        assert fu.get_text_language("Test") == "english"

    def test_confident_detection_iso(self):
        fu = _make_fu(detector_results=[("de", 0.75)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        assert fu.get_text_language(text, output="iso-639") == "de"

    def test_confident_detection_nltk_name(self):
        fu = _make_fu(detector_results=[("de", 0.75)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        assert fu.get_text_language(text) == "german"

    def test_low_confidence_returns_en_iso(self):
        fu = _make_fu(detector_results=[("de", 0.20)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        assert fu.get_text_language(text, output="iso-639") == "en"

    def test_unknown_code_falls_through_as_code(self):
        """ISO code with no LANG_CODE_TO_NAME mapping is returned as-is."""
        fu = _make_fu(detector_results=[("sw", 0.85)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        assert fu.get_text_language(text) == "sw"

    def test_logs_detection_result(self):
        fu = _make_fu(detector_results=[("de", 0.75)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        fu.get_text_language(text, output="iso-639")
        labels = [m[1] for m in fu.pretty.messages]
        assert "LangDetect" in labels

    def test_logs_fallback_on_short_text(self):
        fu = _make_fu()
        fu.get_text_language("Test")
        labels = [m[1] for m in fu.pretty.messages]
        assert "LangDetect" in labels


# ===========================================================================
# get_user_text_language
# ===========================================================================


class TestGetUserTextLanguage:

    def test_too_short_uses_native_lang(self):
        """Single word + declared native language → use native language."""
        fu = _make_fu()
        result = fu.get_user_text_language(
            "Test", output="iso-639", native_lang="german", installed_codes={"de"}
        )
        assert result == "de"

    def test_too_short_no_native_returns_english(self):
        fu = _make_fu()
        assert fu.get_user_text_language("Test", output="iso-639") == "en"

    def test_confident_detection_overrides_native(self):
        """High-confidence detection wins even when native_lang differs."""
        fu = _make_fu(detector_results=[("fr", 0.80)])
        text = "das ist ein test und wir haben viele woerter im text hier"
        result = fu.get_user_text_language(
            text, output="iso-639", native_lang="german", installed_codes={"de", "fr"}
        )
        assert result == "fr"

    def test_low_confidence_promotes_installed_runnerup(self):
        """Fell-back detection: installed runner-up is promoted over English."""
        # At 7 words: effective_conf ≈ 0.77; af=0.55 < 0.77 → fell_back
        # candidate pool = {"de","en"}; de=0.30, en not in det → best="de"
        fu = _make_fu(detector_results=[("af", 0.55), ("de", 0.30)])
        text = "sind dies saeugetiere und welche tiere sind"  # 7 words
        result = fu.get_user_text_language(
            text, output="iso-639", installed_codes={"de"}
        )
        assert result == "de"

    def test_low_confidence_stays_english_when_no_installed_candidates(self):
        """Fell-back and no installed candidates in det list → stays English."""
        fu = _make_fu(detector_results=[("nl", 0.35)])
        text = "do bees have stingers around here and why"  # 9 words
        result = fu.get_user_text_language(
            text, output="iso-639", installed_codes=set()
        )
        assert result == "en"

    def test_iso_and_nltk_output_formats(self):
        """Verify both output format options for a confident detection."""
        fu = _make_fu(detector_results=[("de", 0.80)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        assert fu.get_user_text_language(text, output="iso-639") == "de"
        assert fu.get_user_text_language(text, output="nltk") == "german"

    def test_native_lang_not_used_when_detection_confident(self):
        """When confidence is above threshold, native_lang has no effect."""
        fu = _make_fu(detector_results=[("de", 0.80)])
        text = "was sind saeugetiere und welche tiere gehoeren dazu und warum"
        result_with = fu.get_user_text_language(
            text, output="iso-639", native_lang="french", installed_codes={"de", "fr"}
        )
        result_without = fu.get_user_text_language(text, output="iso-639")
        assert result_with == result_without == "de"
