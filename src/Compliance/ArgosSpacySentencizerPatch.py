"""Patch argostranslate SPACY chunk mode to use blingfire sentence splitting.

Argos Translate 1.11 initializes SPACY sentence splitting by calling
``argostranslate.networking.cache_spacy()``, which downloads ``xx_sent_ud_sm``.
RAG-LCC hard-forces ``ARGOS_CHUNK_TYPE=SPACY`` internally while:
1) replacing the cache hook with an offline local cache builder, and
2) monkey-patching ``SpacySentencizerSmall`` to use blingfire.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, cast

from Commons.DriveRootGuard import is_drive_root

_FALLBACK_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _fallback_sentence_split(text: str) -> list[str]:
    """Best-effort sentence split used when blingfire is unavailable/fails."""
    if not text:
        return []
    parts = [
        part.strip() for part in _FALLBACK_SENTENCE_SPLIT_RE.split(text) if part.strip()
    ]
    return parts or [text]


def _is_safe_delete_target(path: Path) -> bool:
    """Return False when *path* resolves to a drive/filesystem root."""
    try:
        abs_path = str(path.resolve())
    except Exception:
        abs_path = os.path.abspath(str(path))
    return not is_drive_root(abs_path)


def prepare_argos_runtime_for_patch() -> None:
    """Prepare Argos runtime directories, then apply the sentencizer patch."""
    from Helpers.Helpers import Helpers

    Helpers().prepare_argos_runtime_dirs()
    patch_argostranslate_spacy_cache()


def patch_argostranslate_spacy_cache() -> None:
    """Monkey-patch Argos SPACY mode for offline, model-free sentence splitting.

    The patch is idempotent. RAG-LCC enforces the Argos SPACY slot and patches
    that slot to use blingfire sentence splitting.
    """
    os.environ["ARGOS_CHUNK_TYPE"] = "SPACY"

    try:
        from argostranslate import (networking,  # type: ignore[import-untyped]
                                    settings)
    except Exception:
        return

    if not getattr(networking, "_rag_lcc_spacy_cache_patched", False):

        def _cache_spacy_no_model_download() -> Path | None:
            try:
                import spacy

                spacy_cache = Path(settings.cache_dir / "spacy")
                marker = spacy_cache / ".rag_lcc_spacy_blank_xx"

                # Keep a local blank pipeline cache and never call
                # `spacy.cli.download("xx_sent_ud_sm")`.
                if marker.exists() and (spacy_cache / "meta.json").exists():
                    return spacy_cache

                if spacy_cache.exists():
                    if not _is_safe_delete_target(spacy_cache):
                        return None
                    shutil.rmtree(spacy_cache, ignore_errors=True)
                spacy_cache.mkdir(parents=True, exist_ok=True)

                nlp = spacy.blank("xx")
                nlp.to_disk(spacy_cache)
                marker.write_text("spacy.blank('xx')\n", encoding="utf-8")
                return spacy_cache
            except Exception:
                return None

        patched_cache: Callable[[], Path | None] = _cache_spacy_no_model_download
        networking.cache_spacy = patched_cache  # type: ignore[assignment]
        setattr(networking, "_rag_lcc_spacy_cache_patched", True)

    try:
        import argostranslate.sbd as sbd  # type: ignore[import-untyped]
    except Exception:
        return

    if getattr(sbd, "_rag_lcc_sentencizer_patched", False):
        return

    def _init_blingfire_sentencizer(self: Any, pkg: Any) -> None:
        self.pkg = pkg

    def _split_with_regex_fallback(self: Any, text: str) -> list[str]:
        if not text:
            return []
        return _fallback_sentence_split(text)

    def _regex_fallback_label(self: Any) -> str:
        return "RegexFallbackSentencizer"

    try:
        from blingfire import text_to_sentences  # type: ignore[import-untyped]
    except Exception:
        setattr(sbd.SpacySentencizerSmall, "__init__", _init_blingfire_sentencizer)
        setattr(
            sbd.SpacySentencizerSmall, "split_sentences", _split_with_regex_fallback
        )
        setattr(sbd.SpacySentencizerSmall, "__str__", _regex_fallback_label)
        setattr(sbd, "_rag_lcc_blingfire_sentencizer_patched", False)
        setattr(sbd, "_rag_lcc_sentencizer_backend", "regex")
        setattr(sbd, "_rag_lcc_sentencizer_patched", True)
        return

    def _split_with_blingfire(self: Any, text: str) -> list[str]:
        if not text:
            return []
        try:
            sentences_text = cast(str, text_to_sentences(text))
            sentences = sentences_text.splitlines()
            cleaned: list[str] = [line.strip() for line in sentences if line.strip()]
            return cleaned or _fallback_sentence_split(text)
        except Exception:
            return _fallback_sentence_split(text)

    def _blingfire_label(self: Any) -> str:
        return "BlingfireSentencizer"

    setattr(sbd.SpacySentencizerSmall, "__init__", _init_blingfire_sentencizer)
    setattr(sbd.SpacySentencizerSmall, "split_sentences", _split_with_blingfire)
    setattr(sbd.SpacySentencizerSmall, "__str__", _blingfire_label)
    setattr(sbd, "_rag_lcc_sentencizer_backend", "blingfire")
    setattr(sbd, "_rag_lcc_sentencizer_patched", True)
    setattr(sbd, "_rag_lcc_blingfire_sentencizer_patched", True)
