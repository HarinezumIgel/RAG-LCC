import hashlib
import json
import logging
import os
import re
import shutil
import string
import tempfile
import uuid
from pathlib import Path
from typing import Any, ClassVar

import json5
import nltk  # type: ignore[reportMissingTypeStubs]
from nltk.corpus import stopwords  # type: ignore[reportMissingTypeStubs]

from Config.Config import Config
from Globals.CounterInstance import FailedCount, ProcessedCount
from Globals.Globals import Globals
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper


class _LangResult:
    """Typed container for a single language-detection result.

    Provides ``.lang`` (ISO 639-1 string) and ``.prob`` (float confidence)
    so the rest of FileUtils can treat results from any backend uniformly.
    """

    __slots__ = ("lang", "prob")

    def __init__(self, lang: str, prob: float) -> None:
        self.lang = lang
        self.prob = prob


class FileUtils:
    # Utility class for file operations and text processing.
    _stopwords_cache: ClassVar[dict[str, list[str]]] = {}

    def __init__(
        self, *, cfg: "Config | None" = None, pretty: "PrettyWriter | None" = None
    ):
        """
        Initialize FileUtils with counters, config, and stopwords setup.
        """
        self.failedCounter: FailedCount = FailedCount()
        self.processedCounter: ProcessedCount = ProcessedCount()
        self.globalsInstance: Globals = Globals()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.custom_ntlk_data_dir: str = self.cfg.get_str("_CUSTOM_NLTK_DATA_DIRECTORY")
        self.stopwords_download: str | None = os.environ.get(
            "NLTK_STOPWORDS_DOWNLOAD", "0"
        )
        if self.custom_ntlk_data_dir not in nltk.data.path:  # type: ignore[reportUnknownMemberType]
            nltk.data.path.append(self.custom_ntlk_data_dir)  # type: ignore[reportUnknownMemberType]
        self.label_alias: dict[str, str] = self.cfg.get_dict("_LABEL_ALIAS")

    # Example: use FileUtils.hash_file when needed
    def hash_module(self, mod: Any, algo: str = "sha256") -> str:
        """
        Compute hash of a module file using the specified algorithm.
        """
        path: str | None = getattr(mod, "__file__", None)
        if not path:
            raise ValueError(
                f"Module {mod.__name__} has no file (built-in or extension)."
            )
        return self.hash_file(path)

    def hash_file(self, file_path: str, chunk_size: int = 8_192) -> str:
        """
        Compute SHA256 hash of a file.
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                h.update(chunk)
        return h.hexdigest()

    def compute_hash(self, config_slot: Any) -> str:
        """
        Compute SHA256 hash of a config slot (dict, list, tuple, or str).
        """
        if isinstance(config_slot, (dict, list, tuple)):
            text: str = json.dumps(config_slot, sort_keys=True, separators=(",", ":"))
        else:
            text = str(config_slot)
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def setDebug(self) -> None:
        """
        Set debug logging levels for various modules based on config.
        """
        if self.cfg.get_bool("HF_DEBUG"):
            self.pretty.write("I", "", "HF_DEBUG mode is enabled.")
            logging.basicConfig(level=logging.DEBUG)
            os.environ["HF_DEBUG"] = "1"
            os.environ["HF_HUB_LOG_LEVEL"] = "DEBUG"
            os.environ["TRANSFORMERS_VERBOSITY"] = "DEBUG"
        if DebugHelper.check(self.cfg, 55):
            self.pretty.write(
                "I", "", f"DEBUG mode is set to {DebugHelper.level(self.cfg)}."
            )
            logging.basicConfig(level=logging.DEBUG)
        if self.cfg.get_bool("URL_DEBUG") or DebugHelper.check(self.cfg, 55):
            logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.DEBUG)
            logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)
            logging.getLogger("httpcore").setLevel(logging.DEBUG)
        else:
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Text utilities
    def count_words(self, text: str) -> int:
        """
        Count the number of words in a string.
        """
        return len(text.split())

    def clean_text(self, text: str, char_mapping: dict[str, str]) -> str:
        """
        Clean and normalize text using character mapping and punctuation removal.
        """
        translation_table: dict[int, str] = {
            ord(ch): repl for ch, repl in char_mapping.items()
        }
        normalized_text: str = text.translate(translation_table)
        punctuation_pattern: str = "[" + re.escape(string.punctuation) + "]"
        normalized_text = re.sub(punctuation_pattern, " ", normalized_text)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        return normalized_text

    # Cached set of installed Argos Translate language codes.
    _argos_codes: ClassVar[set[str] | None] = None
    # Cached reverse map (NLTK name -> ISO code) built from LANG_CODE_TO_NAME.
    _name_to_code: ClassVar[dict[str, str] | None] = None
    # Lazy-initialised lingua LanguageDetector singleton (expensive to build).
    _lingua_detector: ClassVar[Any] = None

    @classmethod
    def _get_lingua_detector(cls) -> Any:
        """Return the singleton lingua LanguageDetector, building it on first call."""
        if cls._lingua_detector is None:
            from lingua import \
                LanguageDetectorBuilder  # lazy — keeps startup fast

            cls._lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
        return cls._lingua_detector

    def _detect_lang_iso(
        self,
        text: str,
        installed_codes: "set[str] | None" = None,
    ) -> tuple[str, bool, bool, list[Any], float, float]:
        """Run lingua-language-detector with a word-count-scaled confidence
        threshold + installed-codes runner-up promotion.

        Returns ``(lang_iso, fell_back, too_short, det, conf, effective_conf)``
        where ``lang_iso`` is the resolved ISO code (``"en"`` on fallback),
        ``det`` is a list of :class:`_LangResult` objects (may be empty), and
        ``effective_conf`` is the threshold actually applied — higher for short
        text, lower for long text.
        """
        min_conf: float = self.cfg.get_float("_LANGUAGE_DETECTION.MIN_CONFIDENCE")
        min_words: int = self.cfg.get_int("_LANGUAGE_DETECTION.MIN_WORDS")
        conf_full_words: int = self.cfg.get_int("_LANGUAGE_DETECTION.CONF_FULL_WORDS")
        det: list[Any] = []
        fell_back: bool = False
        too_short: bool = False

        word_count: int = len(text.split())

        # Word-count-scaled threshold: starts at 0.90 for short text and
        # decreases linearly to min_conf at conf_full_words.  Short text
        # produces noisy signals even with full-accuracy lingua; the high
        # bar at low word counts prevents spurious non-English detections.
        _SHORT_CONF: float = 0.90
        if word_count >= conf_full_words:
            effective_conf: float = min_conf
        else:
            t: float = max(
                0.0,
                (word_count - min_words) / max(1, conf_full_words - min_words),
            )
            effective_conf = _SHORT_CONF - t * (_SHORT_CONF - min_conf)

        if word_count < min_words:
            lang: str = "en"
            fell_back = True
            too_short = True
        else:
            try:
                detector = FileUtils._get_lingua_detector()
                det = [
                    _LangResult(r.language.iso_code_639_1.name.lower(), r.value)
                    for r in detector.compute_language_confidence_values(text)
                    if r.value > 0.0
                ]
                if det and det[0].prob >= effective_conf:
                    lang = det[0].lang
                else:
                    lang = "en"
                    fell_back = True
            except Exception:
                lang = "en"
                fell_back = True

        # Auto-detect installed codes once from the SharedHelpers singleton.
        if installed_codes is None:
            if FileUtils._argos_codes is None:
                try:
                    from Compliance.SharedHelpers import \
                        SharedHelpers  # noqa: E402 — lazy to avoid circular deps

                    sh = SharedHelpers()
                    FileUtils._argos_codes = set(sh.get_installed_langs().keys())
                except Exception:
                    FileUtils._argos_codes = set()
            installed_codes = FileUtils._argos_codes

        if installed_codes and lang not in installed_codes:
            try:
                if len(det) >= 2:
                    top_prob: float = det[0].prob
                    for alt in det[1:]:
                        if alt.lang in installed_codes and alt.prob >= top_prob * 0.5:
                            lang = alt.lang
                            break
            except Exception:
                pass

        conf: float = det[0].prob if det else 0.0
        return lang, fell_back, too_short, det, conf, effective_conf

    def _name_to_iso(self, lang_name: str) -> str | None:
        """Reverse-lookup ISO code for an NLTK language name (e.g. ``"german"``
        \u2192 ``"de"``). Returns ``None`` if the name is unknown."""
        if FileUtils._name_to_code is None:
            lang_map: dict[str, str] = self.cfg.get_dict(
                "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
            )
            FileUtils._name_to_code = {v.lower(): k for k, v in lang_map.items()}
        return FileUtils._name_to_code.get(lang_name.lower())

    def _log_lang_detection(
        self,
        text: str,
        lang: str,
        fell_back: bool,
        too_short: bool,
        det: list[Any],
        conf: float,
        installed_codes: "set[str] | None",
        native_lang: str | None = None,
        override_applied: bool = False,
        override_source: str = "native",
        effective_conf: float | None = None,
    ) -> None:
        """Emit the standard LangDetect log lines for a single detection.

        Centralises all messaging (info / fallback warnings / native-language
        override / language-not-installed warning) so both public language
        detection methods share identical log output.
        """
        min_conf: float = self.cfg.get_float("_LANGUAGE_DETECTION.MIN_CONFIDENCE")
        min_words: int = self.cfg.get_int("_LANGUAGE_DETECTION.MIN_WORDS")
        display_threshold: float = (
            effective_conf if effective_conf is not None else min_conf
        )
        lang_map: dict[str, str] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        friendly: str = lang_map.get(lang, lang).capitalize()

        if override_applied:
            reason: str = (
                f"text too short ({len(text.split())} words < {min_words})"
                if too_short
                else f"top guess confidence {conf:.0%} below threshold {display_threshold:.0%}"
            )
            if override_source == "runnerup":
                # Find the chosen language's probability in the det list for
                # the log line.
                promoted_prob: float = 0.0
                for d in det:
                    if d.lang == lang:
                        promoted_prob = d.prob
                        break
                self.pretty.write(
                    "I",
                    "LangDetect",
                    f"Promoted runner-up '{friendly}' ({lang}) at {promoted_prob:.0%} — {reason}",
                )
            else:
                self.pretty.write(
                    "I",
                    "LangDetect",
                    f"Using declared native language '{native_lang}' ({lang}) — {reason}",
                )
        elif fell_back:
            if too_short:
                self.pretty.write(
                    "I",
                    "LangDetect",
                    f"Text too short for reliable detection ({len(text.split())} words < {min_words}) — falling back to English",
                )
            else:
                raw_lang: str = det[0].lang if det else "?"
                raw_friendly: str = lang_map.get(raw_lang, raw_lang).capitalize()
                self.pretty.write(
                    "I",
                    "LangDetect",
                    f"Detected language: {raw_friendly} ({raw_lang}) — confidence: {conf:.0%} below threshold {display_threshold:.0%} — falling back to English",
                )
        else:
            self.pretty.write(
                "I",
                "LangDetect",
                f"Detected language: {friendly} ({lang}) — confidence: {conf:.0%} (threshold: {display_threshold:.0%})",
            )

        # Heads-up when the resolved language has no offline translation
        # package installed locally. Downstream policy (e.g. AIHelpers'
        # check_language_support) may still reject the prompt; this log is
        # informational so the user sees WHY it was rejected.
        if installed_codes and lang not in installed_codes and lang != "en":
            self.pretty.write(
                "I",
                "LangDetect",
                f"Language '{friendly}' ({lang}) is NOT installed (no Argos "
                f"package present); offline translation unavailable for this "
                f"language.",
            )

    def get_text_language(
        self,
        text: str,
        output: str = "nltk",
        installed_codes: "set[str] | None" = None,
    ) -> str:
        """Detect language of text and return either ISO or NLTK language name.

        Args:
            text: The text to detect the language of.
            output: ``"iso-639"`` for the raw ISO 639-1 code, anything else
                for the NLTK stopword name (e.g. ``"german"``).
            installed_codes: Optional set of ISO codes for locally installed
                Argos Translate packages. When supplied (or auto-detected)
                and the top detection is *not* installed but a close
                runner-up *is*, the runner-up is preferred.  This avoids
                misclassifying e.g. short German text as Dutch when only
                ``de`` is installed.
        """
        lang, fell_back, too_short, det, conf, eff_conf = self._detect_lang_iso(
            text, installed_codes
        )
        # Resolve installed_codes the same way _detect_lang_iso did, so the
        # logger can emit the "not installed" warning without re-querying.
        codes: set[str] | None = installed_codes
        if codes is None:
            codes = FileUtils._argos_codes
        self._log_lang_detection(
            text,
            lang,
            fell_back,
            too_short,
            det,
            conf,
            codes,
            effective_conf=eff_conf,
        )

        if output == "iso-639":
            return lang
        lang_map: dict[str, str] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        return lang_map.get(lang, lang)

    def get_user_text_language(
        self,
        text: str,
        output: str = "nltk",
        native_lang: str | None = None,
        installed_codes: "set[str] | None" = None,
    ) -> str:
        """Detect language of a USER-entered query/prompt with low-confidence
        rescue logic.

        Same return semantics as :meth:`get_text_language` but adds a smarter
        fallback: when the top langdetect guess is below the confidence
        threshold, instead of returning ``"english"`` we pick the highest
        probability candidate in the langdetect result list, restricted to
        ``en`` plus installed Argos languages. ``native_lang`` (if declared)
        participates as an extra candidate — its probability is taken from
        the detection list when present.

        Genuinely English short queries therefore stay English (``en`` is in
        the pool), while short non-English queries like ``"sind dies
        säugetiere?"`` (where langdetect ranks Afrikaans first at 57 %
        but German second at 30 %) get promoted to German.

        When the text is too short to attempt detection at all, ``native_lang``
        is used directly if declared, else English.

        High-confidence detections always win.

        Args:
            text: The user query.
            output: ``"iso-639"`` for raw ISO code, else NLTK name.
            native_lang: Lowercase NLTK language name (e.g. ``"german"``) of
                the user's declared preferred language. ``None`` disables the
                override (behaviour identical to :meth:`get_text_language`).
            installed_codes: See :meth:`get_text_language`.
        """
        lang, fell_back, too_short, det, conf, eff_conf = self._detect_lang_iso(
            text, installed_codes
        )

        override_applied: bool = False
        override_source: str = "native"
        if fell_back:
            # Low-confidence fallback. Pick the highest-probability candidate
            # from the detection result list, restricted to English plus
            # installed Argos languages. native_lang participates as an extra
            # candidate — its probability is read from the detection list
            # when present. English stays in the pool so genuinely English
            # short/ambiguous queries don't get promoted to a sibling
            # language by accident.
            codes_for_pool: set[str] | None = installed_codes
            if codes_for_pool is None:
                codes_for_pool = FileUtils._argos_codes or set()

            native_iso: str | None = None
            if native_lang:
                native_iso = self._name_to_iso(native_lang)

            # Build {iso: prob} from det, filtered to allowed candidates.
            allowed: set[str] = set(codes_for_pool) | {"en"}
            if native_iso:
                allowed.add(native_iso)
            candidates: dict[str, float] = {}
            for d in det:
                d_lang: str = d.lang
                if d_lang in allowed:
                    candidates[d_lang] = d.prob

            if candidates:
                best_lang: str = max(candidates, key=lambda k: candidates[k])
                if best_lang != "en":
                    lang = best_lang
                    override_applied = True
                    override_source = (
                        "native" if best_lang == native_iso else "runnerup"
                    )
            elif too_short and native_iso:
                # No detection list to mine (text too short to even try);
                # trust the declared language.
                lang = native_iso
                override_applied = True
                override_source = "native"

        codes: set[str] | None = installed_codes
        if codes is None:
            codes = FileUtils._argos_codes
        self._log_lang_detection(
            text,
            lang,
            fell_back,
            too_short,
            det,
            conf,
            codes,
            native_lang=native_lang,
            override_applied=override_applied,
            override_source=override_source,
            effective_conf=eff_conf,
        )

        if output == "iso-639":
            return lang
        lang_map: dict[str, str] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        return lang_map.get(lang, lang)

    def removeStopwords(self, text: str, stopWords: set[str]) -> str:
        """
        Remove stopwords from text.
        """
        words: list[str] = text.split()
        filteredText: str = " ".join(
            [word for word in words if word.lower() not in stopWords]
        )
        return filteredText

    def get_stopwords(self, text: str) -> list[str]:
        """
        Retrieve stopwords for detected language, downloading if needed.
        """
        stopword_lang: str = self.get_text_language(text, "nltk")
        if stopword_lang in self._stopwords_cache:
            return self._stopwords_cache[stopword_lang]
        try:
            nltk.data.find("corpora/stopwords")  # type: ignore[reportUnknownMemberType]
        except LookupError:
            if self.stopwords_download == "1":
                try:
                    nltk.download("stopwords", quiet=True)  # type: ignore[reportUnknownMemberType]
                except Exception as e:
                    self.pretty.write(
                        "W", "Stopwords", f"Failed to download stopwords corpus: {e}"
                    )
                    self._stopwords_cache[stopword_lang] = []
                    return []
            else:
                self.pretty.write(
                    "W",
                    "Stopwords",
                    "Stopwords corpus not found and internet access is disabled.",
                )
                self._stopwords_cache[stopword_lang] = []
                return []
        try:
            stop_words: list[str] = stopwords.words(stopword_lang)  # type: ignore[reportUnknownMemberType]
            self._stopwords_cache[stopword_lang] = stop_words
            return stop_words
        except Exception as e:
            self.pretty.write(
                "W", "Stopwords", f"Could not load stopwords for '{stopword_lang}': {e}"
            )
            self._stopwords_cache[stopword_lang] = []
            return []

    # Example of delegating to FileUtils
    def delete_path(self, path: str) -> bool:
        """
        Delete a file or directory at the given path.
        """
        return self.delete_file_or_dir(path)

    def delete_file_or_dir(self, filepath: str) -> bool:
        """
        Deletes a file or directory at the specified filepath.

        A project-root containment check prevents deletion of paths
        outside the project directory (path jailbreak guard).

        Args:
            filepath (str): The full path to the file or directory to be deleted.

        Returns:
            bool: True if deletion succeeded or if the file/directory does not exist,
                False if an error occurred.
        """
        if not filepath:
            self.pretty.write("E", "", f"No filepath specified.")
            return False

        abs_fp = os.path.normpath(os.path.abspath(filepath))
        raw_absolute_path = self.cfg.get_str("_ABSOLUTE_PATH")
        project_root = os.path.normpath(os.path.abspath(raw_absolute_path))

        # If project root could not be resolved, refuse all deletions
        if not raw_absolute_path or not project_root:
            self.pretty.write(
                "E",
                "Path Guard",
                "Cannot resolve project root (_ABSOLUTE_PATH). "
                "Refusing all deletions as a safety measure.",
            )
            return False

        # Refuse if the configured project root is itself a drive/fs root
        _, root_tail = os.path.splitdrive(project_root)
        if len(root_tail) <= 2:
            self.pretty.write(
                "E",
                "Path Guard",
                f"Project root resolves to a drive or filesystem root ('{project_root}'). "
                "Refusing all deletions as a safety measure.",
            )
            return False

        # Block drive roots / very short paths (e.g. "C:\" or "/")
        _, tail = os.path.splitdrive(abs_fp)
        if len(tail) <= 2:
            self.pretty.write(
                "E",
                "Path Guard",
                f"Refusing to delete root or drive path '{abs_fp}'. "
                "This looks like a path jailbreak attempt.",
            )
            return False

        # Block any path outside the project root
        if not abs_fp.startswith(project_root + os.sep) and abs_fp != project_root:
            self.pretty.write(
                "E",
                "Path Guard",
                f"Refusing to delete '{abs_fp}': path is outside the project root "
                f"'{project_root}'. This looks like a path jailbreak attempt.",
            )
            return False

        if os.path.exists(filepath):
            try:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    self.pretty.write("O", "", f"Directory '{filepath}' deleted.")
                else:
                    os.remove(filepath)
                    self.pretty.write("O", "", f"File '{filepath}' deleted.")
                return True
            except Exception as e:
                self.pretty.write("E", "", f"Error deleting '{filepath}': {e}")
                return False
        else:
            self.pretty.write("I", "", f"Path '{filepath}' does not exist.")
            return True

    def randomTempFilename(self, suffix: str) -> str:
        """
        Generates a unique temporary filename with the given suffix.

        :param suffix: The file suffix (e.g., ".docx", ".pptx", ".xlsx").
        :return: A unique file path in the system's temporary directory.
        """
        temp_dir: str = tempfile.gettempdir()
        unique_name: str = uuid.uuid4().hex  # Generates a random hex string
        return os.path.join(temp_dir, unique_name + suffix)

    def create_abs_path(
        self, relative_or_absolute_path: str, must_exist: bool = False
    ) -> str:
        """
        Resolves a path to an absolute path and checks its existence.

        Raises FileNotFoundError if the file does not exist.
        """
        # Turn “../../foo.xls” into “C:\…\foo.xls”
        abs_path: str = os.path.abspath(relative_or_absolute_path)

        # Guard: ensure the target exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: '{abs_path}'")

        # Optionally, check it’s a file (not a directory)
        if must_exist and not os.path.isfile(abs_path):
            raise FileNotFoundError(f"[ERR] Path is not a file: '{abs_path}'")

        return abs_path

    def normalize_path(self, path_str: str) -> str:
        """
        Normalize a path to absolute, using forward slashes.
        """
        p: Path = Path(path_str).expanduser().resolve()
        # Always use forward slashes
        return p.as_posix()

    def try_parse_json(self, text: str) -> tuple[bool, Any]:
        """
        Try strict json, then json5, then small repairs, then ast.literal_eval.

        Raises ValueError if nothing parseable is found.
        """
        if text is None:  # type: ignore[reportUnnecessaryComparison]
            raise ValueError("No text to parse")
        # strict JSON
        try:
            return True, json.loads(text)
        except Exception:
            pass
        # json5 (more permissive)
        try:
            return True, json5.loads(text)  # type: ignore[reportUnknownVariableType]
        except Exception:
            pass
        # quick repairs
        repaired = text.replace("'", '"')
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)
        try:
            return True, json.loads(repaired)
        except Exception:
            pass
        return False, text

    def is_file_or_path(self, filename: str) -> str:
        """
        Determine if filename is a file, directory, or path (Windows/relative).
        """
        p = Path(filename)

        # Check if there's a drive letter (indicative of an absolute Windows path)
        if p.drive:
            if p.suffix:  # file extension exists → file with full path
                return "FileWithPath"
            else:
                return "Directory"
        else:
            # It's a relative path or file name
            if p.suffix:  # relative file with extension
                return "File"
            else:
                # Relative file name without extension
                return "file"


def build_csv_path(friendly_name: str, status: str, stamp: str, log_dir: str) -> str:
    """Build a CSV log-file path: ``{log_dir}/{friendly_name}_{status}_{stamp}.csv``"""
    return os.path.join(log_dir, f"{friendly_name}_{status}_{stamp}.csv")
