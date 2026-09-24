from typing import Any, cast

from Commons.Exceptions import ConfigPathError


def _is_cli_overrideable_slot(name: str) -> bool:
    """Return True when a constant name is eligible for CLI overrides."""
    return (
        name.isupper()
        and not name.startswith("_")
        and not name.startswith("$")
        and not name.startswith("INTERNAL_")
    )


def extract_cli_overrideable_constants(module: Any) -> dict[str, Any]:
    """Extract top-level constants that can be surfaced as CLI flags."""
    if module is None:
        return {}

    module_vars = cast(dict[str, Any], vars(module))
    return {
        name: value
        for name, value in module_vars.items()
        if _is_cli_overrideable_slot(name) and not isinstance(value, dict)
    }


def app_uses_load_retrievers_cli_scope(app_module: Any | None) -> bool:
    """Return True when app policy includes Config_Load_Retrievers CLI keys."""
    module_name = str(getattr(app_module, "__name__", ""))
    return module_name in {
        "Configuration.Config_RAGLoad",
        "Configuration.Config_RAGChat",
        "Configuration.Config_RAGChatService",
    }


def build_allowed_cli_overrides(
    global_module: Any,
    app_module: Any | None,
    load_retrievers_module: Any | None = None,
) -> dict[str, Any]:
    """Build allowed CLI constants from Global + optional modules."""
    allowed = extract_cli_overrideable_constants(global_module)
    if load_retrievers_module is not None:
        allowed.update(extract_cli_overrideable_constants(load_retrievers_module))
    if app_module is not None:
        allowed.update(extract_cli_overrideable_constants(app_module))
    return allowed


def validate_cli_overrides(
    raw_args: dict[str, Any],
    allowed_slots: set[str],
) -> None:
    """Fail fast when a provided CLI override is outside policy."""
    invalid_keys: list[str] = []

    for key, value in raw_args.items():
        if value is None:
            continue

        key_up = str(key).upper()
        if not _is_cli_overrideable_slot(key_up) or key_up not in allowed_slots:
            invalid_keys.append(key_up)

    if invalid_keys:
        shown = ", ".join(sorted(set(invalid_keys)))
        raise ConfigPathError(
            "Unsupported CLI override(s): "
            f"{shown}. Allowed overrides must come from Config_Global and "
            "the active app config module, and must not start with '_' or '$'."
        )


def normalize_cli_overrides(
    raw_args: dict[str, Any],
    allowed_slots: set[str],
) -> dict[str, Any]:
    """Normalize, validate, then keep only policy-allowed CLI args."""
    raw_upper = {str(key).upper(): value for key, value in raw_args.items()}
    validate_cli_overrides(raw_upper, allowed_slots)
    return {key: value for key, value in raw_upper.items() if key in allowed_slots}
