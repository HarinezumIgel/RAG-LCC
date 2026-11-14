"""Single source of truth for orchestration-flow names, keys and validation.

Lightweight on purpose (no pipeline imports) so Session, QueryParts, the API
handler, the evaluator and the orchestrator can all share it.
"""

from __future__ import annotations

from typing import Any, cast

from Commons.Exceptions import ConfigurationError

# Used only when the configuration defines no flows at all.
FALLBACK_ORCHESTRATION_FLOW = "NATIVE_ALL"

QUERY_SOURCES: tuple[str, ...] = (
    "REWRITTEN_QUERY",
    "TRANSLATED_QUERY",
    "ORIGINAL_QUERY",
)
DEFAULT_MAIN_QUERY_SOURCE = "REWRITTEN_QUERY"
DEFAULT_SECONDARY_QUERY_SOURCE = "TRANSLATED_QUERY"
DEFAULT_RERANK_QUERY_SOURCE = "REWRITTEN_QUERY"

RETRIEVAL_LANGUAGE_MODES: tuple[str, ...] = ("ENGLISH", "NATIVE")
DEFAULT_RETRIEVAL_LANGUAGE_MODE = "ENGLISH"

# Flow config on/off key -> Session / OrchestrationFlowProfile field.
FLOW_SWITCH_KEY_TO_FIELD: dict[str, str] = {
    "use_keyword_action_splitter": "enable_keyword_action_splitter",
    "use_query_rewrite": "enable_query_rewrite",
    "use_pronoun_substitution": "enable_pronoun_substitution",
    "use_secondary_query": "enable_guardrail_leg",
    "use_original_language_vector": "enable_original_language_vector_leg",
    "use_vector_alternates": "enable_vector_alternate_queries",
    "shape_indexed_queries": "enable_indexed_query_shaping",
    "translate_indexed_queries": "enable_indexed_query_translation",
    "expand_indexed_queries_with_wordnet": "enable_indexed_query_wordnet_expansion",
    "run_vector": "enable_vector_retriever",
    "run_bm25": "enable_bm25_retriever",
    "run_graph": "enable_graph_retriever",
    "run_regex": "enable_regex_retriever",
    "run_rerank": "enable_rerank",
    "run_low_score_fallback": "enable_low_score_fallback",
    "run_low_recall_rescue": "enable_low_recall_rescue",
    "run_grounding": "enable_grounding",
}

FLOW_VALUE_KEYS: tuple[str, ...] = (
    "force_retrieve_mode",
    "main_query_source",
    "secondary_query_source",
    "rerank_query_source",
    "retrieval_language",
    "confidence_evaluation",
)

KNOWN_FLOW_KEYS: frozenset[str] = frozenset(FLOW_SWITCH_KEY_TO_FIELD) | frozenset(
    FLOW_VALUE_KEYS
)

LOCAL_RETRIEVERS: tuple[str, ...] = ("VECTOR", "BM25", "GRAPH", "REGEX")


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _cfg_dict(cfg: Any, key: str) -> dict[str, Any]:
    return cfg.get_dict(key, {}) if cfg is not None else {}


def _cfg_list(cfg: Any, key: str) -> list[Any]:
    return cfg.get_list(key, []) if cfg is not None else []


def _cfg_str(cfg: Any, key: str) -> str:
    return cfg.get_str(key, "").strip() if cfg is not None else ""


def normalize_name(name: Any) -> str:
    return str(name or "").strip().upper()


# ---------------------------------------------------------------------------
# Flow catalog
# ---------------------------------------------------------------------------


def normalize_flow_map(raw: Any) -> dict[str, dict[str, Any]]:
    """Return {FLOW_NAME: profile_dict} with upper-cased names."""
    flow_map: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return flow_map
    for key_obj, value in cast(dict[Any, Any], raw).items():
        key = normalize_name(key_obj)
        if key and isinstance(value, dict):
            flow_map[key] = cast(dict[str, Any], value)
    return flow_map


def flow_aliases(cfg: Any) -> dict[str, str]:
    """Return {OLD_NAME: NEW_NAME} from _ORCHESTRATION_FLOW_ALIASES."""
    aliases: dict[str, str] = {}
    for old, new in _cfg_dict(cfg, "_ORCHESTRATION_FLOW_ALIASES").items():
        old_key, new_key = normalize_name(old), normalize_name(new)
        if old_key and new_key:
            aliases[old_key] = new_key
    return aliases


def resolve_flow_alias(name: Any, cfg: Any) -> str:
    """Map a (possibly legacy) flow name to its current name, upper-cased."""
    key = normalize_name(name)
    return flow_aliases(cfg).get(key, key)


def load_flow_map(cfg: Any) -> dict[str, dict[str, Any]]:
    """Read, normalize and validate _ORCHESTRATION_FLOWS."""
    flow_map = normalize_flow_map(_cfg_dict(cfg, "_ORCHESTRATION_FLOWS"))
    validate_flow_profiles(
        flow_map,
        allowed_retrieve_modes=allowed_retrieve_modes(cfg),
        allowed_query_sources=allowed_query_sources(cfg),
    )
    return flow_map


def allowed_flow_names(cfg: Any) -> list[str]:
    """Flows a user may select: _ALLOWED_ORCHESTRATION_FLOWS or all defined."""
    allowed: list[str] = []
    for item in _cfg_list(cfg, "_ALLOWED_ORCHESTRATION_FLOWS"):
        name = resolve_flow_alias(item, cfg)
        if name and name not in allowed:
            allowed.append(name)
    if not allowed:
        allowed = list(normalize_flow_map(_cfg_dict(cfg, "_ORCHESTRATION_FLOWS")))
    return allowed or [FALLBACK_ORCHESTRATION_FLOW]


def default_flow_name(cfg: Any, allowed: list[str] | None = None) -> str:
    """Configured _DEFAULT_ORCHESTRATION_FLOW, falling back to the first allowed flow."""
    allowed_names = allowed if allowed is not None else allowed_flow_names(cfg)
    configured = resolve_flow_alias(_cfg_str(cfg, "_DEFAULT_ORCHESTRATION_FLOW"), cfg)
    if configured and (not allowed_names or configured in allowed_names):
        return configured
    if allowed_names:
        return allowed_names[0]
    return configured or FALLBACK_ORCHESTRATION_FLOW


def select_flow_name(cfg: Any, requested: Any) -> str:
    """Resolve aliases and fall back to the default for unknown/disallowed names."""
    allowed = allowed_flow_names(cfg)
    name = resolve_flow_alias(requested, cfg)
    if name and name in allowed:
        return name
    return default_flow_name(cfg, allowed)


def flow_profile_raw(cfg: Any, flow_name: Any) -> dict[str, Any]:
    """Return the validated raw profile dict of one flow ({} when unknown)."""
    return load_flow_map(cfg).get(resolve_flow_alias(flow_name, cfg), {})


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def allowed_retrieve_modes(cfg: Any) -> set[str]:
    return {
        normalize_name(item)
        for item in _cfg_list(cfg, "_ALLOWED_RETRIEVE_MODES")
        if normalize_name(item)
    }


def allowed_query_sources(cfg: Any) -> list[str]:
    allowed: list[str] = []
    for item in _cfg_list(cfg, "_ALLOWED_ORCHESTRATION_QUERY_SOURCES"):
        key = normalize_name(item)
        if key and key not in allowed:
            allowed.append(key)
    return allowed or list(QUERY_SOURCES)


def normalize_query_source(value: Any, allowed: list[str], default: str) -> str:
    key = normalize_name(value)
    if key in allowed:
        return key
    return default if default in allowed else allowed[0]


def normalize_retrieval_language(value: Any) -> str | None:
    """Return ENGLISH/NATIVE for a valid value (case-insensitive), else None."""
    key = normalize_name(value)
    return key if key in RETRIEVAL_LANGUAGE_MODES else None


def mode_flags_for_local_retrievers(
    retrieve_mode: str | None,
) -> tuple[bool, bool, bool, bool]:
    """Return enabled flags for Vector/BM25/Graph/Regex for a retrieve mode."""
    mode_upper = normalize_name(retrieve_mode) or "VECTOR"
    if mode_upper == "ALL":
        return True, True, True, True
    parts = set(mode_upper.split("_"))
    return cast(
        tuple[bool, bool, bool, bool],
        tuple(name in parts for name in LOCAL_RETRIEVERS),
    )


def narrows_local_retrievers(retrieve_mode: str | None) -> bool:
    """True when an explicit mode restricts local retrievers (not ALL/WEB/empty)."""
    mode = normalize_name(retrieve_mode)
    return bool(mode) and mode not in {"ALL", "WEB"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_flow_profiles(
    flow_map: dict[str, dict[str, Any]],
    *,
    allowed_retrieve_modes: set[str] | None = None,
    allowed_query_sources: list[str] | None = None,
) -> None:
    """Raise ConfigurationError listing every invalid flow key or value."""
    from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator

    sources = allowed_query_sources or list(QUERY_SOURCES)
    problems: list[str] = []
    for flow_name, profile in flow_map.items():
        for key, value in profile.items():
            where = f"_ORCHESTRATION_FLOWS[{flow_name!r}][{key!r}]"
            if key not in KNOWN_FLOW_KEYS:
                problems.append(f"{where}: unknown key")
            elif key in FLOW_SWITCH_KEY_TO_FIELD:
                if not isinstance(value, bool):
                    problems.append(f"{where}: must be True or False, got {value!r}")
            elif key == "force_retrieve_mode":
                mode = normalize_name(value)
                if (
                    mode
                    and allowed_retrieve_modes
                    and mode not in allowed_retrieve_modes
                ):
                    problems.append(
                        f"{where}: {value!r} not in _ALLOWED_RETRIEVE_MODES "
                        f"({', '.join(sorted(allowed_retrieve_modes))})"
                    )
            elif key.endswith("_query_source"):
                if normalize_name(value) not in sources:
                    problems.append(f"{where}: {value!r} not in {', '.join(sources)}")
            elif key == "retrieval_language":
                if normalize_retrieval_language(value) is None:
                    problems.append(f"{where}: {value!r} must be english or native")
            elif key == "confidence_evaluation":
                try:
                    QualityDimensionEvaluator.validate_mode(str(value))
                except ValueError as exc:
                    problems.append(f"{where}: {exc}")
    if problems:
        raise ConfigurationError(
            "Invalid orchestration flow configuration:\n  " + "\n  ".join(problems)
        )


def validate_flow_config(cfg: Any) -> None:
    """Validate flows, aliases, the allowed list and the default flow."""
    flow_map = load_flow_map(cfg)
    if not flow_map:
        return
    problems: list[str] = []
    for old, new in flow_aliases(cfg).items():
        if new not in flow_map:
            problems.append(
                f"_ORCHESTRATION_FLOW_ALIASES[{old!r}] -> unknown flow {new!r}"
            )
        if old in flow_map:
            problems.append(
                f"_ORCHESTRATION_FLOW_ALIASES[{old!r}] shadows a defined flow"
            )
    for item in _cfg_list(cfg, "_ALLOWED_ORCHESTRATION_FLOWS"):
        name = resolve_flow_alias(item, cfg)
        if name not in flow_map:
            problems.append(f"_ALLOWED_ORCHESTRATION_FLOWS: unknown flow {item!r}")
    default_name = resolve_flow_alias(_cfg_str(cfg, "_DEFAULT_ORCHESTRATION_FLOW"), cfg)
    if default_name and default_name not in flow_map:
        problems.append(f"_DEFAULT_ORCHESTRATION_FLOW: unknown flow {default_name!r}")
    if problems:
        raise ConfigurationError(
            "Invalid orchestration flow configuration:\n  " + "\n  ".join(problems)
        )
