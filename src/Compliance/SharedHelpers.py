import getpass
import hashlib
import importlib.util
import os
import re
import socket
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Commons.StartupCommons import suppress_argos_logging
from Compliance.ArgosSpacySentencizerPatch import \
    prepare_argos_runtime_for_patch

prepare_argos_runtime_for_patch()

from argostranslate import package, translate  # type: ignore[import-untyped]

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import ORANGE, RED, RESET, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.LanguageConfig import (get_active_language_codes,
                                    get_lang_code_to_name,
                                    get_lang_name_to_code)


def _incomplete_argos_package_dirs() -> list[Path]:
    """Return package directories missing their required metadata file."""
    try:
        return [
            package_dir
            for package_root in package.settings.package_dirs
            for package_dir in Path(package_root).iterdir()
            if package_dir.is_dir() and not (package_dir / "metadata.json").is_file()
        ]
    except Exception:
        return []


class SharedHelpers(SingletonMixin):
    """
    Normalization, tokenization, n-grams, translation and shared compiled-regex cache.
    """

    _INSTALLED_KIND_WIDTH = 5
    _INSTALLED_NAME_WIDTH = 24
    _INSTALLED_CODE_WIDTH = 5

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        emit_startup_inventory: bool = True,
        emit_startup_warnings: bool = True,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        suppress_argos_logging()

        self.lang_load_failed: bool = False
        try:
            # Enumerate installed local packages
            langs = translate.get_installed_languages()

        except Exception as exc:
            langs = []
            self.lang_load_failed = True
            pretty = pretty or PrettyWriter(always_on=True)
            warning = (
                f"Installed language inventory unavailable: "
                f"get_installed_languages() failed: {exc!r}"
            )
            if isinstance(exc, FileNotFoundError) and "no metadata.json" in str(exc):
                incomplete_dirs = _incomplete_argos_package_dirs()
                if incomplete_dirs:
                    warning += (
                        "; consider cleaning incomplete package directory "
                        "(metadata.json is missing): "
                        + ", ".join(str(path) for path in incomplete_dirs)
                    )
            pretty.write(
                "W",
                "Argos Translate",
                warning,
                color=ORANGE,
            )

        self.installed_langs: Dict[str, Any] = {
            getattr(l, "code", "").lower(): l for l in langs
        }
        # compiled regex cache keyed by (pattern_text, flags)
        self.regex_compile_cache: Dict[Tuple[str, int], re.Pattern[str]] = {}
        # translation cache keyed by (text, target_lang, source_lang)
        self.translation_cache: Dict[Tuple[str, str, str, bool], str] = {}
        # translated wordlist cache keyed by (tuple(wordlist_en), lang)
        self.translated_list_cache: Dict[Tuple[Tuple[str, ...], str], List[str]] = {}
        # Warning dedupe buckets keyed by category.
        self._warned_by_category: Dict[str, set[str]] = {}
        # Backward-compatible alias sets used by existing code paths/tests.
        self.warned_langs: set[str] = self._warning_bucket(
            "argos.target-language-missing"
        )
        self.warned_unsupported_langs: set[str] = self._warning_bucket(
            "argos.runtime-language-unsupported"
        )
        self.warned_pairs: set[str] = self._warning_bucket(
            "argos.translation-pair-missing"
        )
        self.warned_runtime_pairs: set[str] = self._warning_bucket(
            "argos.translation-runtime-failure"
        )
        self.warned_spacy_model_unavailable: set[str] = self._warning_bucket(
            "spacy.model-unavailable"
        )
        self.pretty: PrettyWriter = pretty or PrettyWriter(always_on=True)
        self.cfg: Config = cfg or Config()
        self._emit_startup_inventory: bool = emit_startup_inventory
        self._emit_startup_warnings: bool = emit_startup_warnings

        if self._emit_startup_inventory and DebugHelper.check(self.cfg, 10):
            self._log_installed_argos_languages(langs)
            self._log_installed_spacy_packages()
            self.pretty.write("N", "", "")
        self.leet_map: dict[str, Any] = self.cfg.get_dict("_LEET_MAP")
        self.confusables: dict[str, Any] = self.cfg.get_dict("_CONFUSABLES")
        # Reverse of LANG_CODE_TO_NAME: NLTK name → ISO code
        self.lang_name_to_code: Dict[str, str] = get_lang_name_to_code(self.cfg)
        if self._emit_startup_warnings:
            self._warn_missing_active_argos_languages_once()
            self._warn_missing_active_spacy_models_once()
        # Module-level cache so the identity is captured only once per process
        self.identity_cache: Optional[Dict[str, object]] = None

    def _warn_missing_active_argos_languages_once(self) -> None:
        """Warn once at startup when active config languages are not installed."""
        active_codes = sorted(
            code
            for code in get_active_language_codes(self.cfg)
            if code and not code.startswith("en")
        )
        if not active_codes:
            return

        missing_codes = [
            code for code in active_codes if code not in self.installed_langs
        ]
        if not missing_codes:
            return

        self.pretty.write(
            "W",
            "Argos Translate",
            "Configured active language(s) not installed locally: "
            + ", ".join(missing_codes)
            + ". Translation-dependent checks will fall back to English until "
            "packages are installed. Run src/Scripts/ArgosTranslatePackages.py "
            "install.",
            color=ORANGE,
        )

    def _warn_missing_active_spacy_models_once(self) -> None:
        """Warn once at startup when active language spaCy packages are unavailable."""
        try:
            models_by_language: dict[str, Any] = self.cfg.get_dict(
                "_SPACY_MODELS_BY_ACTIVE_LANGUAGE",
                {},
                silent=True,
            )
        except TypeError:
            # Test stubs may expose a narrower ``get_dict(key, default)`` signature.
            models_by_language = self.cfg.get_dict(
                "_SPACY_MODELS_BY_ACTIVE_LANGUAGE",
                {},
            )
        if not models_by_language:
            return

        active_codes = sorted(get_active_language_codes(self.cfg))
        missing_labels: list[str] = []
        for language_code in active_codes:
            model_name = str(models_by_language.get(language_code, "") or "").strip()
            if not model_name:
                continue

            warn_key = f"{language_code}:{model_name}"
            if not self._is_spacy_package_installed(model_name):
                if self._should_warn_once(
                    "spacy.model-unavailable",
                    warn_key,
                ):
                    missing_labels.append(warn_key)

        if not missing_labels:
            return

        self.pretty.write(
            "W",
            "spaCy",
            "Configured active spaCy model package(s) unavailable: "
            + ", ".join(missing_labels)
            + ". Retrieval components may fall back or fail depending on retriever. "
            "Run src/Scripts/SpacyLanguageModels.py install.",
            color=ORANGE,
        )

    def get_installed_langs(self) -> Dict[str, Any]:
        """Return the map of installed Argos Translate language codes to language objects."""
        return self.installed_langs

    def _warning_bucket(self, category: str) -> set[str]:
        """Return the mutable warning-key bucket for *category*."""
        normalized_category = str(category or "").strip().lower() or "default"
        bucket: set[str] | None = self._warned_by_category.get(normalized_category)
        if bucket is None:
            bucket = set[str]()
            self._warned_by_category[normalized_category] = bucket
        return bucket

    def _should_warn_once(self, category: str, key: str) -> bool:
        """Return True once per ``(category, key)`` pair."""
        normalized_key = str(key or "").strip().lower()
        if not normalized_key:
            return False
        bucket = self._warning_bucket(category)
        if normalized_key in bucket:
            return False
        bucket.add(normalized_key)
        return True

    def is_language_supported(self, lang: str) -> bool:
        """Return *True* if *lang* is English or is installed in Argos Translate.

        Accepts both ISO-639 codes (``"de"``) and NLTK names (``"german"``).
        English is always treated as supported because the banlists are
        already in English and no translation is required.
        """
        code: str = (lang or "en").lower()
        code = self.lang_name_to_code.get(code, code)
        if code.startswith("en"):
            return True
        return code in self.installed_langs

    def check_language_support(self, language: str, file_path: str = "?") -> str | None:
        """Check whether *language* is installed and return the configured action.

        Returns ``None`` when the language is supported or the configured
        action is ``FALLBACK_EN`` (caller should continue normally).
        Returns ``"NOT_OK"`` when the caller must reject the document.
        A warning is logged automatically.
        """
        if self.is_language_supported(language):
            return None
        action: str = self.cfg.get_str(
            "UNSUPPORTED_LANGUAGE_ACTION", "FALLBACK_EN"
        ).upper()
        lang_code = self.lang_name_to_code.get(
            (language or "").lower(), (language or "").lower()
        )
        warning_key = lang_code or (language or "").lower()

        if action == "NOT_OK":
            if self._should_warn_once(
                "argos.runtime-language-unsupported",
                warning_key,
            ):
                self.pretty.write(
                    "W",
                    "Language",
                    f"Language '{language}' not installed — "
                    f"skipping {file_path} (UNSUPPORTED_LANGUAGE_ACTION=NOT_OK)",
                    color=ORANGE,
                )
            return "NOT_OK"

        if self._should_warn_once(
            "argos.runtime-language-unsupported",
            warning_key,
        ):
            self.pretty.write(
                "W",
                "Language",
                f"Language '{language}' not installed — "
                "falling back to English-only checks "
                "(UNSUPPORTED_LANGUAGE_ACTION=FALLBACK_EN). "
                "Run src/Scripts/ArgosTranslatePackages.py install to add the "
                "required language pairs.",
                color=ORANGE,
            )
        return None

    def refresh_installed_languages(self) -> None:
        """Re-scan installed Argos Translate packages and update the language map.

        Called after ArgosDownloader installs new packages so that translation
        is available immediately without requiring a restart.
        """
        try:
            langs = translate.get_installed_languages()
        except Exception:
            langs = []
        self.installed_langs = {getattr(l, "code", "").lower(): l for l in langs}

    @staticmethod
    def _is_spacy_package_installed(package_name: str) -> bool:
        """Return True if the given spaCy model package can be imported."""
        try:
            return importlib.util.find_spec(package_name) is not None
        except Exception:
            return False

    def _format_installed_inventory_line(
        self,
        *,
        kind: str,
        name: Any,
        code: Any,
        suffix: str = "",
    ) -> str:
        """Build one aligned inventory line for Argos/spaCy status output."""
        kind_text = str(kind or "").strip()
        name_text = str(name or "").strip()
        code_text = str(code or "").strip().lower()
        suffix_text = str(suffix or "").strip()

        base = (
            f"{kind_text:<{self._INSTALLED_KIND_WIDTH}} "
            f"Installed {name_text:<{self._INSTALLED_NAME_WIDTH}} "
            f"({code_text:<{self._INSTALLED_CODE_WIDTH}})"
        )
        if suffix_text:
            return f"{base} {suffix_text}"
        return base

    def _log_installed_argos_languages(self, langs: List[Any]) -> None:
        """Emit debug lines for installed Argos language packages."""
        for lang in langs:
            self.pretty.write(
                "D",
                "Argos Translate",
                self._format_installed_inventory_line(
                    kind="Lang:",
                    name=getattr(lang, "name", ""),
                    code=getattr(lang, "code", ""),
                ),
            )

    def _log_installed_spacy_packages(self) -> None:
        """Emit debug lines for installed configured spaCy model packages."""
        models_by_language: dict[str, Any] = self.cfg.get_dict(
            "_SPACY_MODELS_BY_ACTIVE_LANGUAGE", {}, silent=True
        )
        if not models_by_language:
            return

        code_to_name = get_lang_code_to_name(self.cfg)
        active_codes = sorted(get_active_language_codes(self.cfg))

        for language_code in active_codes:
            model_name = str(models_by_language.get(language_code, "")).strip()
            if not model_name:
                continue
            if not self._is_spacy_package_installed(model_name):
                continue

            language_name = str(
                code_to_name.get(language_code, language_code)
            ).capitalize()
            self.pretty.write(
                "D",
                "spaCy",
                self._format_installed_inventory_line(
                    kind="Pkg:",
                    name=model_name,
                    code=language_code,
                    suffix=language_name,
                ),
            )

    # -------------------------
    # Normalization / token helpers
    # -------------------------
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        t = unicodedata.normalize("NFKC", text).casefold()
        t = "".join(self.confusables.get(ch, ch) for ch in t)
        t = "".join(self.leet_map.get(ch, ch) for ch in t)
        return re.sub(r"\s+", " ", t).strip()

    # ----------------------------
    # Tokenization helper (robust regex)
    # ----------------------------
    def tokenize(self, text: str) -> List[str]:
        # keep alphanumerics only, lowercase
        if not text:
            return []
        toks = re.findall(r"[A-Za-z0-9]+", text)
        return [t.lower() for t in toks if t.strip()]

    def char_ngrams(self, text: str, n_min: int, n_max: int) -> List[str]:
        UNSET = object()
        if n_min is UNSET or n_max is UNSET:
            msg: str = "n_min and n_max must be set"
            self.pretty.write("E", "char_ngrams", msg, color=RED)
            raise ValueError(msg)
        grams: List[str] = []
        L = len(text)
        for n in range(n_min, n_max + 1):
            if n <= 0:
                continue
            for i in range(L - n + 1):
                grams.append(text[i : i + n])
        return grams

    @staticmethod
    def jaccard(a: List[str], b: List[str]) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if A and B else 0.0

    @staticmethod
    def containment(a: List[str], b: List[str]) -> float:
        A, B = set(a), set(b)
        return len(A & B) / min(len(A), len(B)) if A and B else 0.0

    # -------------------------
    # Translation / banlist helpers (cached)
    # -------------------------
    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        *,
        warn_on_fallback: bool = True,
    ) -> str:
        if not text or not target_lang:
            return text
        key = (
            text,
            (target_lang or "").lower(),
            (source_lang or "auto").lower(),
            bool(warn_on_fallback),
        )
        cached = self.translation_cache.get(key)
        if cached is not None:
            return cached

        translator = self._get_translation(
            source_lang,
            target_lang,
            emit_warning=warn_on_fallback,
        )
        if translator is None:
            translated = text
            if warn_on_fallback:
                src_norm: str = self.lang_name_to_code.get(
                    (source_lang or "").lower(), (source_lang or "").lower()
                )
                tgt_norm: str = self.lang_name_to_code.get(
                    (target_lang or "").lower(), (target_lang or "").lower()
                )
                if src_norm and src_norm != "auto" and src_norm != tgt_norm:
                    pair_key: str = f"{src_norm}->{tgt_norm}"
                    if self._should_warn_once(
                        "argos.translation-pair-missing",
                        pair_key,
                    ):
                        self.pretty.write(
                            "W",
                            "Translate",
                            f"No Argos translation pair {src_norm}->{tgt_norm} "
                            f"installed - returning text unchanged. Ensure both "
                            f"languages are listed in "
                            f"_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES, add "
                            f"('{src_norm}', '{tgt_norm}') to "
                            f"_ARGOS_DEFINITIONS.ARGOS_LANGUAGES and run "
                            f"src/Scripts/ArgosTranslatePackages.py to install it.",
                            color=ORANGE,
                        )
        else:
            try:
                translated = translator.translate(text)
            except Exception as exc:
                translated = text
                if warn_on_fallback:
                    src_norm: str = self.lang_name_to_code.get(
                        (source_lang or "").lower(), (source_lang or "").lower()
                    )
                    tgt_norm: str = self.lang_name_to_code.get(
                        (target_lang or "").lower(), (target_lang or "").lower()
                    )
                    pair_key = f"{src_norm or 'auto'}->{tgt_norm or '?'}"
                    if self._should_warn_once(
                        "argos.translation-runtime-failure",
                        pair_key,
                    ):
                        self.pretty.write(
                            "W",
                            "Translate",
                            f"Argos translation failed for {pair_key} "
                            f"({exc.__class__.__name__}: {exc}) — using English "
                            "fallback/unchanged text. This can happen when Argos "
                            "package cache files are missing. Run "
                            "src/Scripts/ArgosTranslatePackages.py install "
                            "to restore required language pairs.",
                            color=ORANGE,
                        )

        self.translation_cache[key] = translated
        return translated

    def _get_translation(
        self,
        source_code: str,
        target_code: str,
        *,
        emit_warning: bool = True,
    ) -> Any:
        source_code = (source_code or "auto").lower()
        target_code = (target_code or "").lower()
        # Normalise NLTK names ("german") → ISO codes ("de")
        source_code = self.lang_name_to_code.get(source_code, source_code)
        target_code = self.lang_name_to_code.get(target_code, target_code)
        if not target_code:
            return None

        # --- lazy retry if initial language load failed ---
        if self.lang_load_failed and not self.installed_langs:
            try:
                langs = translate.get_installed_languages()
                self.installed_langs = {
                    getattr(l, "code", "").lower(): l for l in langs
                }
                self.lang_load_failed = False
            except Exception:
                pass

        # --- try to find a locally installed translation pair ---
        src_lang = self.installed_langs.get(source_code)
        tgt_lang = self.installed_langs.get(target_code)
        if src_lang and tgt_lang:
            try:
                translated = src_lang.get_translation(tgt_lang)
                if translated:
                    return translated
            except Exception:
                pass

        # Fallback source probing is only safe when source is unknown/auto.
        # If a concrete source language is provided and that exact pair is
        # missing, return None so callers can handle the miss explicitly.
        allow_source_fallback: bool = (
            source_code in {"", "auto"} or source_code not in self.installed_langs
        )
        if tgt_lang and allow_source_fallback:
            for src in self.installed_langs.values():
                src_code = str(getattr(src, "code", "")).lower()
                if not src_code or src_code == target_code:
                    continue
                try:
                    translated = src.get_translation(tgt_lang)
                    if translated:
                        return translated
                except Exception:
                    continue

        # --- no installed pair found ---
        # Only emit the target-language warning when the target itself is
        # missing. Pair-specific misses are logged by translate_text().
        if (
            emit_warning
            and tgt_lang is None
            and self._should_warn_once(
                "argos.target-language-missing",
                target_code,
            )
        ):
            self.pretty.write(
                "W",
                "Translate Banned",
                f"Language '{target_code}' not installed locally — "
                f"banned words to target language translation skipped, using English fallback. "
                "Run src/Scripts/ArgosTranslatePackages.py install to add the required "
                "language pairs and refresh consent metadata.",
                color=ORANGE,
            )
        return None

    def merge_banlists(
        self, banlist_en: List[str], translated_phrases: List[str] | None = None
    ) -> List[str]:
        """
        Return a merged banlist.

        - If `lang == "en"` returns the translated list lowercased (or the English source lowercased if no translated list provided).
        - If `lang != "en"` merges `self.banlist_en` with `translated`, lowercasing all words,
          preserving order (English first), and removing duplicates.
        """

        # lowercase inputs
        src = [w.lower() for w in banlist_en]
        trans = [w.lower() for w in (translated_phrases or [])]

        # merge while preserving order and removing duplicates
        merged: List[str] = []
        seen: set[str] = set()
        for w in src + trans:
            if w and w not in seen:
                seen.add(w)
                merged.append(w)
        return merged

    def get_translated_wordlist(
        self, wordlist_en: List[str], language: str, algo: str
    ) -> List[str]:
        lang: str = (language or "en").lower()
        lang = self.lang_name_to_code.get(lang, lang)
        wordlist_key: tuple[str, ...] = tuple(wordlist_en or [])
        cache_key: tuple[tuple[str, ...], str] = (wordlist_key, lang)
        cached: list[str] | None = self.translated_list_cache.get(cache_key)
        if cached is not None:
            return cached

        if lang.startswith("en"):
            normalized: list[str] = [self.normalize(p) for p in wordlist_en]
            self.translated_list_cache[cache_key] = normalized
            return normalized

        translated_phrases: List[str] = []
        for p in wordlist_en:
            tp: str = self.translate_text(
                p,
                target_lang=lang,
                source_lang="en",
                warn_on_fallback=False,
            )
            if DebugHelper.check(self.cfg, 40):
                self.pretty.write(
                    "D",
                    f"Cache build {algo} Translated",
                    f"Cached Lang: {lang} Source: {p:<40} Translated: {tp:<40}",
                )
            translated_phrases.append(self.normalize(tp))

        merged: List[str] = self.merge_banlists(wordlist_en, translated_phrases)
        self.translated_list_cache[cache_key] = merged
        return merged

    # -------------------------
    # Regex compile cache
    # -------------------------
    def compile_regex(self, pattern: str, flags: int = 0) -> re.Pattern[str]:
        """
        Compile and cache regex Pattern objects keyed by (pattern, flags).
        """
        key: tuple[str, int] = (pattern, flags)
        cp: re.Pattern[str] | None = self.regex_compile_cache.get(key)
        if cp is not None:
            return cp
        # compile outside lock to avoid holding lock for long operations
        compiled: re.Pattern[str] = re.compile(pattern, flags)
        existing: re.Pattern[str] | None = self.regex_compile_cache.get(key)
        if existing is None:
            self.regex_compile_cache[key] = compiled
            return compiled
        return existing

    def capture_acceptance_identity_once(self) -> Dict[str, object]:
        """
        Capture or prompt for the identity to use as 'accepted_by'.
        Executed only once per process and result is cached.
        Returns dict with accepted_by, accepted_by_source, accepted_by_verified, host, pid.
        """
        if self.identity_cache is not None:
            return self.identity_cache

        # Try to get git user email first
        git_user: str | None = None
        try:
            git_user = (
                subprocess.check_output(
                    ["git", "config", "user.email"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            if not git_user:
                git_user = None
        except Exception:
            git_user = None

        # Fall back to OS user
        try:
            os_user: str = getpass.getuser()
        except Exception:
            os_user = os.getenv("_USER") or os.getenv("USERNAME") or "unknown-user"

        # Prefer git email if available
        accepted_by: str = (git_user or os_user or "").strip()
        accepted_by_source: str = "git" if git_user else "os"
        if not accepted_by:
            accepted_by = "unknown-user"
            accepted_by_source = "os"
        accepted_by_verified: bool = False  # Only true if SSO/signing is implemented

        # Allow interactive override (only once per process)
        try:
            print(
                f"Detected identity for acceptance: {accepted_by} (source: {accepted_by_source})"
            )
            override = input(
                f"{YELLOW}Press Enter to accept as-is or type your email/ID to override: {RESET}"
            ).strip()
            if override:
                accepted_by = override
                accepted_by_source = "interactive"
                accepted_by_verified = False
        except Exception:
            # Non-interactive: keep detected identity
            pass

        accepted_by = (accepted_by or "").strip() or "unknown-user"
        accepted_by_source = (accepted_by_source or "").strip() or "os"
        host = (socket.gethostname() or "").strip() or "unknown-host"

        identity: dict[str, object] = {
            "accepted_by": accepted_by,
            "accepted_by_source": accepted_by_source,
            "accepted_by_verified": accepted_by_verified,
            "host": host,
            "pid": os.getpid(),
        }
        self.identity_cache = identity
        return identity

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """SHA-256 hash of canonicalised text (CRLF→LF, stripped)."""
        return hashlib.sha256(
            text.replace("\r\n", "\n").strip().encode("utf-8")
        ).hexdigest()
