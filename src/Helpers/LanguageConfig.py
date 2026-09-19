"""Utilities for active-language configuration and Argos pair filtering."""

from __future__ import annotations

from typing import Any, Dict


def _cfg_get_list(cfg: Any, key: str, default: list[Any] | None = None) -> list[Any]:
    fallback = default if default is not None else []
    indirect_getter = getattr(cfg, "indirect_list", None)
    if callable(indirect_getter):
        try:
            value = indirect_getter(key, fallback, silent=True)
        except TypeError:
            value = indirect_getter(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, list):
            return value
        return fallback

    getter = getattr(cfg, "get_list", None)
    if callable(getter):
        try:
            value = getter(key, fallback, silent=True)
        except TypeError:
            value = getter(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, list):
            return value
        return fallback

    generic_get = getattr(cfg, "get", None)
    if callable(generic_get):
        try:
            value = generic_get(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, list):
            return value
    return fallback


def _cfg_get_dict(
    cfg: Any,
    key: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = default if default is not None else {}
    indirect_getter = getattr(cfg, "indirect_dict", None)
    if callable(indirect_getter):
        try:
            value = indirect_getter(key, fallback, silent=True)
        except TypeError:
            value = indirect_getter(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, dict):
            return value
        return fallback

    getter = getattr(cfg, "get_dict", None)
    if callable(getter):
        try:
            value = getter(key, fallback, silent=True)
        except TypeError:
            value = getter(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, dict):
            return value
        return fallback

    generic_get = getattr(cfg, "get", None)
    if callable(generic_get):
        try:
            value = generic_get(key, fallback)
        except Exception:
            value = fallback
        if isinstance(value, dict):
            return value
    return fallback


def get_lang_code_to_name(cfg: Any) -> dict[str, str]:
    """Return normalized ISO code -> language name mapping."""
    mapping: dict[str, str] = {"en": "english"}
    raw_mapping = _cfg_get_dict(cfg, "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME", {})
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

    active_raw = _cfg_get_list(cfg, "_ARGOS_DEFINITIONS.ACTIVE_LANGUAGES", ["en"])
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
    raw_pairs = _cfg_get_list(cfg, "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES", [])

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
