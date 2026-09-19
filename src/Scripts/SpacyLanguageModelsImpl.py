"""Shared spaCy model-package operations used by compliance and CLI wrappers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, cast

from Gui.Colors import ORANGE, RESET
from Helpers.LanguageConfig import (get_active_language_codes,
                                    normalize_language_code)


def print_install_scope_note() -> None:
    """Explain spaCy model license scope before installation checks."""
    print(f"{ORANGE}Installed spaCy models may be distributed under licenses{RESET}")
    print(f"{ORANGE}different from the spaCy library itself.{RESET}")
    print(f"{ORANGE}Refer to the model metadata for license details.{RESET}")


def configured_spacy_models(cfg: Any) -> list[str]:
    """Return deduped configured spaCy model package names."""
    active_codes = get_active_language_codes(cfg)
    bm25_model = _read_indirect_str(cfg, "_BM25_INDEX.spacy_model")
    graph_model = _read_indirect_str(cfg, "_GRAPH_INDEX.spacy_model")
    regex_model = _read_indirect_str(cfg, "_REGEX_INDEX.spacy_model")
    bm25_by_lang = _read_indirect_dict(cfg, "_BM25_INDEX.spacy_models_by_language")
    graph_by_lang = _read_indirect_dict(cfg, "_GRAPH_INDEX.spacy_models_by_language")
    regex_by_lang = _read_indirect_dict(cfg, "_REGEX_INDEX.spacy_models_by_language")

    by_lang_models: list[str] = []
    for mapping in (bm25_by_lang, graph_by_lang, regex_by_lang):
        for language, model in mapping.items():
            language_code = normalize_language_code(cfg, language)
            if language_code and language_code not in active_codes:
                continue
            model_name = str(model or "").strip()
            if model_name:
                by_lang_models.append(model_name)

    models = [m for m in (bm25_model, graph_model, regex_model) if m] + by_lang_models
    if not models:
        return ["en_core_web_sm"]
    return sorted(set(models))


def is_model_installed(model: str) -> bool:
    """Return True when the model package is importable."""
    try:
        return importlib.util.find_spec(model) is not None
    except Exception:
        return False


def model_location(model: str) -> str:
    """Return resolved model package location, or empty string if missing."""
    try:
        spec = importlib.util.find_spec(model)
    except Exception:
        spec = None
    if spec is None:
        return ""
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if location:
                return str(location)
    return str(spec.origin or "")


def missing_models(models: Sequence[str]) -> list[str]:
    """Return models that are not installed in the active environment."""
    return [model for model in models if not is_model_installed(model)]


def installed_models(models: Sequence[str]) -> list[str]:
    """Return configured models that are currently installed."""
    return [model for model in models if is_model_installed(model)]


def install_models(
    models: Sequence[str],
    *,
    on_install_error: Callable[[str, int], None] | None = None,
) -> bool:
    """Install requested spaCy model packages via python -m spacy download."""
    all_ok = True
    for model in models:
        print(f"  Installing spaCy model {model} ...")
        cmd = [sys.executable, "-m", "spacy", "download", model]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            all_ok = False
            if on_install_error is None:
                print(f"Failed to install spaCy model '{model}' (exit={rc})")
            else:
                on_install_error(model, rc)
    return all_ok


def remove_models(
    models: Sequence[str],
    *,
    on_uninstall_error: Callable[[str, int], None] | None = None,
) -> tuple[int, int]:
    """Uninstall provided spaCy model packages via pip uninstall."""
    removed = 0
    total = 0
    for model in models:
        total += 1
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", model]
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            removed += 1
            continue
        if on_uninstall_error is None:
            print(f"pip uninstall failed for {model} (exit={rc})")
        else:
            on_uninstall_error(model, rc)
    return removed, total


def show_status(models: Sequence[str]) -> None:
    """Print configured spaCy models and whether they are installed."""
    print(f"\nConfigured spaCy models ({len(models)}):")
    for model in models:
        marker = "installed" if is_model_installed(model) else "missing"
        where = model_location(model)
        suffix = f" [{where}]" if where else ""
        print(f"  {model}: {marker}{suffix}")


def spacy_model_license_hint(model_name: str) -> str:
    """Return license hint from local spaCy model metadata."""
    try:
        spec = importlib.util.find_spec(model_name)
    except Exception:
        return "Unknown"

    if spec is None or spec.origin is None:
        return "Unknown (not installed)"

    model_dir = Path(spec.origin).resolve().parent
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        return "Unknown (meta.json missing)"

    try:
        meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "Unknown (meta.json unreadable)"

    if not isinstance(meta_raw, dict):
        return "Unknown (meta.json format)"

    meta = cast(Dict[str, Any], meta_raw)
    license_value = str(meta.get("license", "")).strip()
    return license_value or "Unknown (license not declared)"


def report_spacy_model_license_hints(models: Sequence[str]) -> None:
    """Print per-model license hints from local model metadata."""
    if not models:
        return

    print("\nspaCy model license hints (local model metadata):")
    for model_name in models:
        hint = spacy_model_license_hint(model_name)
        print(f"  {model_name}: {hint}")


def _read_indirect_str(cfg: Any, key: str) -> str:
    indirect_get = getattr(cfg, "indirect_get", None)
    if callable(indirect_get):
        try:
            raw_value = indirect_get(key, "")
            value_obj: object
            if isinstance(raw_value, tuple) and raw_value:
                value_obj = cast(object, raw_value[0])
            else:
                value_obj = cast(object, raw_value)
            return str("" if value_obj is None else value_obj).strip()
        except Exception:
            pass

    direct_get = getattr(cfg, "get", None)
    if callable(direct_get):
        return str(direct_get(key, "") or "").strip()
    return ""


def _read_indirect_dict(cfg: Any, key: str) -> Dict[str, Any]:
    indirect_dict = getattr(cfg, "indirect_dict", None)
    if callable(indirect_dict):
        try:
            mapping = indirect_dict(key, {})
            if isinstance(mapping, dict):
                return cast(Dict[str, Any], mapping)
        except Exception:
            pass

    direct_get = getattr(cfg, "get", None)
    if callable(direct_get):
        mapping = direct_get(key, {})
        if isinstance(mapping, dict):
            return cast(Dict[str, Any], mapping)
    return {}
