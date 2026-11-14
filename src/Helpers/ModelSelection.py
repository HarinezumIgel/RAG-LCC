"""Resolve the _ACTIVE_* model selectors of Config_Models.py to _MODELS entries.

Config_Models.py is the single source of truth: every top-level name there
starting with ``_ACTIVE_`` selects an impl key of ``_MODELS``. For most
selectors the role is the selector name without ``_ACTIVE``
(``_ACTIVE_EMBED`` -> ``_EMBED``). ``_ACTIVE_ENDPOINT`` is the exception: it
selects the provider impl ("ollama" / "vllm") whose single role is named
after the provider (``_OLLAMA`` / ``_VLLM``).
Values are read through the merged config so overrides still apply.
"""

from typing import Any, Protocol, cast

import Configuration.Config_Models as Config_Models

ACTIVE_PREFIX = "_ACTIVE"
ENDPOINT_SELECTOR = "_ACTIVE_ENDPOINT"


class _ConfigLike(Protocol):
    def get_str(self, key: str, default: str = "") -> str: ...

    def get_dict(self, key: str, default: Any = None) -> Any: ...


def model_selectors() -> list[str]:
    """All _ACTIVE_* selector names declared in Config_Models.py."""
    return [
        name for name in vars(Config_Models) if name.startswith(f"{ACTIVE_PREFIX}_")
    ]


def resolve_model_entry(
    cfg: _ConfigLike, selector: str, role: str | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(impl, role, raw_entry)`` for an _ACTIVE_* selector or impl key.

    For a direct impl key (e.g. ``"vllm"``) *role* must be given.
    Raises ValueError when the impl or role does not exist in _MODELS.
    """
    if selector.startswith(ACTIVE_PREFIX):
        impl = cfg.get_str(selector)
        role = role or selector[len(ACTIVE_PREFIX) :]
    else:
        impl = selector
        if role is None:
            raise ValueError(
                "get_model_args expects an _ACTIVE_* selector or an explicit lookup key"
            )
    models: dict[str, Any] = cfg.get_dict("_MODELS") or {}
    if impl not in models:
        raise ValueError(f"Unknown impl: {impl!r} (from {selector!r})")
    impl_set: Any = models[impl]
    if not isinstance(impl_set, dict) or role not in impl_set:
        raise ValueError(f"Role {role!r} not found in impl {impl!r}")
    return impl, role, cast(dict[str, Any], impl_set[role])


def active_model_entries(cfg: _ConfigLike) -> dict[str, dict[str, Any]]:
    """Return ``{"impl.role": raw_entry}`` for every _MODELS entry in use.

    - Models (_LLM, _EMBED, ...): the impl chosen by their _ACTIVE_* selector.
    - Provider entries (those with a PROVIDER key): only the one selected by
      _ACTIVE_ENDPOINT, resolved like Helpers.get_active_endpoint_args
      ("ollama" -> _OLLAMA, "vllm" -> _VLLM).
    - Service entries (any other role, e.g. _OPENWEBUI, _RAGCHATSERVICE):
      fixed, have no selector and are always in use.

    Entries are returned as configured, i.e. with both "OLLAMA" / "VLLM"
    provider slots and without USED_BY filtering (see apply_provider_slot and
    the caller). Empty selectors are skipped. Raises ValueError listing every
    selector that does not resolve.
    """
    entries: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    model_roles: set[str] = set()
    for selector in model_selectors():
        if selector != ENDPOINT_SELECTOR:
            model_roles.add(selector[len(ACTIVE_PREFIX) :])
        impl = str(cfg.get_str(selector) or "").strip()
        if not impl:
            continue
        try:
            if selector == ENDPOINT_SELECTOR:
                _, role, entry = resolve_model_entry(cfg, impl, f"_{impl.upper()}")
            else:
                _, role, entry = resolve_model_entry(cfg, selector)
            entries[f"{impl}.{role}"] = entry
        except ValueError as exc:
            problems.append(f"{selector} = {impl!r}: {exc}")
    if problems:
        raise ValueError("; ".join(problems))
    models: dict[str, Any] = cfg.get_dict("_MODELS") or {}
    for impl, roles in models.items():
        if not isinstance(roles, dict):
            continue
        for role, entry in cast(dict[str, Any], roles).items():
            if (
                isinstance(entry, dict)
                and role not in model_roles
                and "PROVIDER" not in entry
            ):
                entries[f"{impl}.{role}"] = cast(dict[str, Any], entry)
    return entries


def provider_overrides(entry: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Return a copy of the *endpoint* slot (``"OLLAMA"`` / ``"VLLM"``) of *entry*."""
    slot: Any = entry.get(endpoint.strip().upper())
    return dict(cast(dict[str, Any], slot)) if isinstance(slot, dict) else {}


def apply_provider_slot(entry: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Flatten the provider slots of a role entry for the active endpoint.

    A role entry may hold per-provider settings in nested dicts named after the
    endpoint in upper case (``"OLLAMA": {"MODEL": ...}``, ``"VLLM": {...}``).
    The slot of *endpoint* overrides the role-level keys; the other provider
    slots are dropped.  Returns a new dict; *entry* is not modified.
    """
    flat = {k: v for k, v in entry.items() if not isinstance(v, dict)}
    flat.update(provider_overrides(entry, endpoint))
    return flat
