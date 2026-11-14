"""Utilities for active-language configuration and Argos pair filtering."""

from __future__ import annotations

from typing import Any, Dict


def get_lang_code_to_name(cfg: Any) -> dict[str, str]:
    """Return normalized ISO code -> language name mapping."""
    mapping: dict[str, str] = {"en": "english"}
    raw_mapping: dict[str, Any] = cfg.indirect_dict(
        "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME", {}, silent=True
    )
    for code, name in raw_mapping.items():
        normalized_code = str(code or "").strip().lower()
        normalized_name = str(name or "").strip().lower()
        if normalized_code and normalized_name:
            mapping[normalized_code] = normalized_name
    return mapping


def get_lang_name_to_code(cfg: Any) -> Dict[str, str]:
    """Return normalized language token -> ISO code mapping."""
    name_to_code: Dict[str, str] = {
        "en": "en",
        "english": "en",
    }
    code_to_name = get_lang_code_to_name(cfg)
    for code, name in code_to_name.items():
        normalized_code = str(code or "").strip().lower()
        normalized_name = str(name or "").strip().lower()
        if normalized_code:
            name_to_code[normalized_code] = normalized_code
        if normalized_code and normalized_name:
            name_to_code[normalized_name] = normalized_code
    return name_to_code


def normalize_language_code(cfg: Any, language: Any) -> str:
    """Normalize language code/name to lowercase ISO code when possible."""
    text = str(language or "").strip().lower()
    if not text:
        return ""
    return get_lang_name_to_code(cfg).get(text, text)


def get_active_language_codes(cfg: Any) -> set[str]:
    """Return active language codes from ACTIVE_LANGUAGES, defaulting to English."""
    name_to_code = get_lang_name_to_code(cfg)

    active_raw: list[Any] = cfg.indirect_list(
        "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES", ["en"], silent=True
    )
    active_codes: set[str] = set()
    for item in active_raw:
        token = str(item or "").strip().lower()
        if not token:
            continue
        normalized = name_to_code.get(token, token)
        if normalized:
            active_codes.add(normalized)

    if not active_codes:
        active_codes = {"en"}

    active_codes.add("en")
    return active_codes


def get_active_argos_pairs(cfg: Any) -> list[tuple[str, str]]:
    """Return configured Argos pairs filtered to active language codes."""
    active_codes = get_active_language_codes(cfg)
    name_to_code = get_lang_name_to_code(cfg)
    raw_pairs: list[Any] = cfg.indirect_list(
        "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES", [], silent=True
    )

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)):
            continue
        if len(pair) != 2:
            continue

        src_token = str(pair[0] or "").strip().lower()
        dst_token = str(pair[1] or "").strip().lower()
        if not src_token or not dst_token:
            continue

        src_code = name_to_code.get(src_token, src_token)
        dst_code = name_to_code.get(dst_token, dst_token)
        if src_code not in active_codes or dst_code not in active_codes:
            continue

        normalized_pair = (src_code, dst_code)
        if normalized_pair in seen:
            continue
        seen.add(normalized_pair)
        pairs.append(normalized_pair)

    return pairs
