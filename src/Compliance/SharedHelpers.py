import getpass
import hashlib
import os
import re
import socket
import subprocess
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from Helpers.Helpers import Helpers

Helpers().prepare_argos_runtime_dirs()

from argostranslate import translate  # type: ignore[import-untyped]

from Commons.SingletonMixin import SingletonMixin
from Commons.StartupCommons import suppress_argos_logging
from Config.Config import Config
from Gui.Colors import ORANGE, RED, RESET, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper


class SharedHelpers(SingletonMixin):
    """
    Normalization, tokenization, n-grams, translation and shared compiled-regex cache.
    """

    def __init__(
        self, *, cfg: "Config | None" = None, pretty: "PrettyWriter | None" = None
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
            pretty.write(
                "W",
                "Argos Translate",
                f"get_installed_languages() failed: {exc!r} — "
                f"translation will retry on first use",
                color=ORANGE,
            )

        self.installed_langs: Dict[str, Any] = {
            getattr(l, "code", "").lower(): l for l in langs
        }
        # compiled regex cache keyed by (pattern_text, flags)
        self.regex_compile_cache: Dict[Tuple[str, int], re.Pattern[str]] = {}
        # translation cache keyed by (text, target_lang, source_lang)
        self.translation_cache: Dict[Tuple[str, str, str], str] = {}
        # translated wordlist cache keyed by (tuple(wordlist_en), lang)
        self.translated_list_cache: Dict[Tuple[Tuple[str, ...], str], List[str]] = {}
        # track languages for which a "not installed" warning has been issued
        self.warned_langs: set[str] = set()
        # track src->tgt pairs for which a "pair not installed" warning has been issued
        self.warned_pairs: set[str] = set()
        self.pretty: PrettyWriter = pretty or PrettyWriter(always_on=True)
        self.cfg: Config = cfg or Config()

        if DebugHelper.check(self.cfg, 10):
            for lang in langs:
                self.pretty.write(
                    "D",
                    "Argos Translate",
                    f"Lang: Installed {lang.name:<20} ({lang.code:<5})",
                )
            self.pretty.write("N", "", "")
        self.leet_map: dict[str, Any] = self.cfg.get_dict("_LEET_MAP")
        self.confusables: dict[str, Any] = self.cfg.get_dict("_CONFUSABLES")
        # Reverse of LANG_CODE_TO_NAME: NLTK name → ISO code
        code_to_name: dict[str, Any] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        self.lang_name_to_code: Dict[str, str] = {
            str(name).lower(): code for code, name in code_to_name.items()
        }
        # Module-level cache so the identity is captured only once per process
        self.identity_cache: Optional[Dict[str, object]] = None

    def get_installed_langs(self) -> Dict[str, Any]:
        """Return the map of installed Argos Translate language codes to language objects."""
        return self.installed_langs

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
        if action == "NOT_OK":
            self.pretty.write(
                "W",
                "Language",
                f"Language '{language}' not installed — "
                f"skipping {file_path} (UNSUPPORTED_LANGUAGE_ACTION=NOT_OK)",
                color=ORANGE,
            )
            return "NOT_OK"
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
        if DebugHelper.check(self.cfg, 10):
            for lang in langs:
                self.pretty.write(
                    "D",
                    "Argos Translate",
                    f"Lang: Installed {lang.name:<20} ({lang.code:<5})",
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
        self, text: str, target_lang: str, source_lang: str = "auto"
    ) -> str:
        if not text or not target_lang:
            return text
        key = (text, (target_lang or "").lower(), (source_lang or "auto").lower())
        cached = self.translation_cache.get(key)
        if cached is not None:
            return cached

        translator = self._get_translation(source_lang, target_lang)
        if translator is None:
            translated = text
            # One-time, source-aware warning: distinguishes "pair not installed"
            # (e.g. de→en when only en→de is installed) from the existing
            # target-only warning emitted by _get_translation. Without this,
            # the user-query normalisation path silently returned the original
            # text and the caller saw "FinalQuery: ... (unchanged)".
            src_norm: str = self.lang_name_to_code.get(
                (source_lang or "").lower(), (source_lang or "").lower()
            )
            tgt_norm: str = self.lang_name_to_code.get(
                (target_lang or "").lower(), (target_lang or "").lower()
            )
            if src_norm and src_norm != "auto" and src_norm != tgt_norm:
                pair_key: str = f"{src_norm}->{tgt_norm}"
                if pair_key not in self.warned_pairs:
                    self.warned_pairs.add(pair_key)
                    self.pretty.write(
                        "W",
                        "Translate",
                        f"No Argos translation pair {src_norm}\u2192{tgt_norm} "
                        f"installed \u2014 returning text unchanged. Add "
                        f"('{src_norm}', '{tgt_norm}') to "
                        f"_ARGOS_DEFINITIONS.ARGOS_LANGUAGES and run "
                        f"src/scripts/ArgosTranslatePackages.py to install it.",
                        color=ORANGE,
                    )
        else:
            try:
                translated = translator.translate(text)
            except Exception:
                translated = text

        self.translation_cache[key] = translated
        return translated

    def _get_translation(self, source_code: str, target_code: str) -> Any:
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

        # fallback: try every installed source → target
        if tgt_lang:
            for src in self.installed_langs.values():
                try:
                    translated = src.get_translation(tgt_lang)
                    if translated:
                        return translated
                except Exception:
                    continue

        # --- no installed pair found ---
        if target_code not in self.warned_langs:
            self.warned_langs.add(target_code)
            stanza_download = os.environ.get("ARGOS_STANZA_DOWNLOAD", "0").strip()
            if stanza_download != "1":
                self.pretty.write(
                    "W",
                    "Translate Banned",
                    f"Language '{target_code}' not installed locally — "
                    f"banned words to target language translation skipped, "
                    f"using English fallback. Set ARGOS_STANZA_DOWNLOAD=1 to allow network downloads "
                    f"or run src/scripts/ArgosTranslatePackages.py to install languages "
                    f"defined in Config_Global _ARGOS_DEFINITIONS.ARGOS_LANGUAGES)",
                    color=ORANGE,
                )
            else:
                self.pretty.write(
                    "W",
                    "Translate Banned",
                    f"Language '{target_code}' not installed — "
                    f"banned words to target language translation skipped, using English fallback. "
                    f"(Add the language pair to ARGOS_LANGUAGES and reinstall packages)",
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
            tp: str = self.translate_text(p, target_lang=lang, source_lang="en")
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
