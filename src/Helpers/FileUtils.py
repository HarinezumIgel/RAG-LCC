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
from langdetect import DetectorFactory  # type: ignore[reportMissingTypeStubs]
from langdetect import detect_langs  # type: ignore[reportUnknownVariableType]
from nltk.corpus import stopwords  # type: ignore[reportMissingTypeStubs]

from Config.Config import Config
from Globals.CounterInstance import FailedCount, ProcessedCount
from Globals.Globals import Globals
from Gui.PrettyWriter import PrettyWriter

# Make langdetect deterministic (it uses random sampling internally).
DetectorFactory.seed = 0  # type: ignore[reportAttributeAccessIssue]


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
        if self.cfg.get_int("DEBUG_LEVEL") >= 50:
            self.pretty.write("I", "", f"DEBUG mode is set to {self.cfg.get_int('DEBUG_LEVEL')}.")  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            logging.basicConfig(level=logging.DEBUG)
        if self.cfg.get_bool("URL_DEBUG") or self.cfg.get_int("DEBUG_LEVEL") >= 50:
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
        # Use detect_langs for a single call; fall back to 'en' when the top
        # result is below the confidence threshold (common for short queries).
        min_conf: float = self.cfg.get_float(
            "_ARGOS_DEFINITIONS.LANG_DETECT_MIN_CONFIDENCE"
        )
        min_chars: int = self.cfg.get_int("_ARGOS_DEFINITIONS.LANG_DETECT_MIN_CHARS")
        det: list[Any] = []
        fell_back: bool = False
        too_short: bool = False
        if len(text.strip()) < min_chars:
            lang: str = "en"
            fell_back = True
            too_short = True
        else:
            try:
                det = detect_langs(text)  # type: ignore[reportUnknownVariableType]
                if det[0].prob >= min_conf:  # type: ignore[reportUnknownMemberType]
                    lang = det[0].lang  # type: ignore[reportUnknownMemberType]
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
                    top_prob: float = det[0].prob  # type: ignore[reportUnknownMemberType]
                    for alt in det[1:]:
                        if alt.lang in installed_codes and alt.prob >= top_prob * 0.5:  # type: ignore[reportUnknownMemberType]
                            lang = alt.lang  # type: ignore[reportUnknownMemberType]
                            break
            except Exception:
                pass

        # Show detected language
        lang_map_log: dict[str, str] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        friendly: str = lang_map_log.get(lang, lang).capitalize()  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        conf: float = det[0].prob if det else 0.0  # type: ignore[reportUnknownMemberType]
        if fell_back:
            if too_short:
                self.pretty.write(
                    "W",
                    "LangDetect",
                    f"Text too short for reliable detection ({len(text.strip())} chars < {min_chars}) — falling back to English",
                )
            else:
                raw_lang: str = str(det[0].lang) if det else "?"  # type: ignore[reportUnknownMemberType]
                raw_friendly: str = lang_map_log.get(raw_lang, raw_lang).capitalize()
                self.pretty.write(
                    "W",
                    "LangDetect",
                    f"Detected language: {raw_friendly} ({raw_lang}) — confidence: {conf:.0%} below threshold {min_conf:.0%} — falling back to English",
                )
        else:
            self.pretty.write(
                "I",
                "LangDetect",
                f"Detected language: {friendly} ({lang}) — confidence: {conf:.0%} (threshold: {min_conf:.0%})",
            )

        if output == "iso-639":
            return lang  # type: ignore[reportUnknownVariableType]
        lang_map: dict[str, str] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        return lang_map.get(lang, lang)  # type: ignore[reportUnknownVariableType, reportUnknownArgumentType]

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
