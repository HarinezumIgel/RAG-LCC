"""Weighted intent classifier — safety gate for all retrieval queries.

Runs on every query regardless of retrieval path (web or local).  A query that
would be safe to answer from a local document store is also safe to send to a
web search engine, and vice versa.  Having one gate for both paths ensures that
bypassing web search does not open a loophole for local retrieval.

Architecture
------------
* **Singleton** — built once per process via :meth:`get_instance`.
* **Baseline immutable** — ``_BASELINE_*`` constants are compiled at module
  load time and cannot be weakened by configuration.  Configuration
  (``WEB_SEARCH_INTENT_EXTENSIONS`` in ``Config_WebSearch.py``) is *additive
  only*: extra entity terms or extra categories can be appended, and
  thresholds can only be *tightened* (lowered), never relaxed.
* **Scoring** — each query accumulates a weighted integer score:
    - entity matches add their category weight
    - intent-pattern matches (synthesis, acquisition, …) add their weight
    - mitigation-pattern matches (definitional, safety framing, …) subtract
  The final clamped score is compared against ``refuse`` / ``warn`` thresholds.
* **Hard-rule override** — when ``_HARD_RULE = True`` (the default) a query
  that matches any entity *and* any synthesis/weaponization intent pattern is
  REFUSE regardless of score.

Outcomes
--------
``ALLOW``                   — score below warn threshold and no hard rule.
``ALLOW_WITH_SAFETY_FRAMING`` — score in [warn, refuse) range.
``REFUSE``                  — score >= refuse threshold or hard rule fired.

Logging
-------
Decisions are written to ``intent_filter.log`` (path from ``_INTENT_FILTER_LOG``
in Config_RAGChat.py / Config_RAGChatService.py).  The caller is responsible for
writing the ``HUMAN_REVIEW`` CSV row; the filter only manages the text log.
"""

from __future__ import annotations

import datetime
import os
import re
import threading
from typing import Any, cast

# ---------------------------------------------------------------------------
# Baseline entity catalogue — immutable; configuration may only extend this.
# weight: integer points added to score on ANY entity match within the category.
# ---------------------------------------------------------------------------
_BASELINE_ENTITIES: dict[str, dict[str, Any]] = {
    "chemical_weapons": {
        "weight": 35,
        "entities": [
            "sarin",
            "novichok",
            "vx",
            "mustard gas",
            "nerve agent",
            "tabun",
            "soman",
        ],
    },
    "explosives": {
        "weight": 35,
        "entities": [
            "dirty bomb",
            "nuclear device",
            "radiological weapon",
            "plastic explosive",
            "c4",
            "semtex",
            "rdx",
            "tatp",
            "hmtd",
            "ammonium nitrate bomb",
        ],
    },
    "bioweapons": {
        "weight": 35,
        "entities": [
            "ricin",
            "anthrax spore",
            "botulinum",
            "weaponized pathogen",
            "plague bacteria",
            "smallpox virus",
        ],
    },
    "illicit_substances": {
        "weight": 25,
        "entities": [
            "fentanyl",
            "methamphetamine",
            "meth",
            "heroin",
            "crack cocaine",
            "carfentanil",
            "opioid analogue",
        ],
    },
    "csam_terms": {
        "weight": 35,
        "entities": [
            "csam",
            "child porn",
            "child sex",
            "lolita porn",
            "underage sex",
        ],
    },
    "trafficking_targets": {
        "weight": 30,
        "entities": [
            "human trafficking",
            "forced labor",
            "sex trafficking",
        ],
    },
    "violence_solicitation": {
        "weight": 30,
        "entities": [
            "hitman",
            "assassin",
            "hired killer",
            "contract killer",
        ],
    },
}

# ---------------------------------------------------------------------------
# Baseline intent patterns — how the query is framed.
# ---------------------------------------------------------------------------
_BASELINE_INTENT_PATTERNS: dict[str, dict[str, Any]] = {
    "synthesis_manufacture": {
        "pattern": re.compile(
            r"(?i)\b(synthesize|synthesis|manufacture|cook|make|produce|create|prepare|"
            r"brew|fabricate|assemble)\b"
        ),
        "weight": 40,
    },
    "acquisition_trafficking": {
        "pattern": re.compile(
            r"(?i)\b(buy|purchase|acquire|obtain|source|procure|order|import|smuggle|"
            r"traffic)\b"
        ),
        "weight": 35,
    },
    "hire_violence": {
        "pattern": re.compile(
            r"(?i)\b(hire|pay|contract|find|recruit)\b.{0,40}"
            r"\b(hitman|assassin|killer|murderer)\b"
        ),
        "weight": 45,
    },
    "procedural_request": {
        "pattern": re.compile(
            r"(?i)\b(step[- ]by[- ]step|instructions?|tutorial|guide|how[- ]to|"
            r"recipe|procedure|protocol|method)\b"
        ),
        "weight": 30,
    },
    "optimization_scaling": {
        "pattern": re.compile(
            r"(?i)\b(increase\s+(yield|potency|purity)|scale\s+up|maximize\s+output|"
            r"bulk\s+(production|synthesis)|industrial[- ]scale)\b"
        ),
        "weight": 40,
    },
    "weaponization": {
        "pattern": re.compile(
            r"(?i)\b(weaponize|deliver(y)?\s+mechanism|aerosolize|disperse|deploy|"
            r"detonate|arm(ing)?|payload)\b"
        ),
        "weight": 50,
    },
    "evasion_jailbreak": {
        "pattern": re.compile(
            r"(?i)\b(bypass|evade|circumvent|avoid\s+detection|undetectable|"
            r"without\s+(getting\s+caught|leaving\s+traces)|off[- ]the[- ]record)\b"
        ),
        "weight": 30,
    },
}

# ---------------------------------------------------------------------------
# Mitigation patterns — benign framing that reduces the score.
# ---------------------------------------------------------------------------
_BASELINE_MITIGATION_PATTERNS: dict[str, dict[str, Any]] = {
    "definitional": {
        "pattern": re.compile(
            r"(?i)\b(what\s+is|define|definition\s+of|explain|describe|"
            r"molar\s+mass|molecular\s+weight|chemical\s+formula|"
            r"structure\s+of|properties\s+of)\b"
        ),
        "weight": -20,
    },
    "safety_framing": {
        "pattern": re.compile(
            r"(?i)\b(safety|hazard|risk|danger|first\s+aid|antidote|treatment|"
            r"decontamination|protective\s+equipment|exposure\s+limit|"
            r"emergency\s+response)\b"
        ),
        "weight": -15,
    },
    "medical_educational": {
        "pattern": re.compile(
            r"(?i)\b(medical|clinical|pharmacology|toxicology|forensic|"
            r"academic|research|scientific\s+literature|peer[- ]reviewed|"
            r"journal|study|thesis)\b"
        ),
        "weight": -10,
    },
    "academic_forensics": {
        "pattern": re.compile(
            r"(?i)\b(history\s+of|historical|detection\s+(method|technique)|"
            r"forensic\s+analysis|autopsy|case\s+study|legal\s+context)\b"
        ),
        "weight": -10,
    },
}

# ---------------------------------------------------------------------------
# Baseline thresholds and hard-rule switch.
# ---------------------------------------------------------------------------
_BASELINE_THRESHOLDS: dict[str, int] = {"refuse": 60, "warn": 30}
_HARD_RULE: bool = True

# Intent categories for which the hard-rule fires (entity + synthesis/weaponize
# must both match).
_HARD_RULE_INTENT_GATES: frozenset[str] = frozenset(
    {
        "synthesis_manufacture",
        "weaponization",
        "optimization_scaling",
        "hire_violence",
    }
)
_HARD_RULE_ENTITY_GATES: frozenset[str] = frozenset(
    {
        "chemical_weapons",
        "explosives",
        "bioweapons",
        "csam_terms",
    }
)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class WebSearchFilter:
    """Weighted intent classifier — see module docstring."""

    _instance: "WebSearchFilter | None" = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction / singleton access
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        extensions_cfg: "dict[str, Any] | None" = None,
        log_path: str = "",
        log_verbose: bool = False,
    ) -> None:
        # Build merged entity catalogue
        self._entities: dict[str, dict[str, Any]] = _merge_entities(
            _BASELINE_ENTITIES, extensions_cfg or {}
        )
        # Build compiled intent / mitigation patterns (baseline only — extensions
        # do not add patterns, only entity terms).
        self._intent_patterns = _BASELINE_INTENT_PATTERNS
        self._mitigation_patterns = _BASELINE_MITIGATION_PATTERNS
        # Thresholds — extensions may only tighten (lower) them.
        self._thresholds: dict[str, int] = _apply_threshold_overrides(
            _BASELINE_THRESHOLDS, extensions_cfg or {}
        )
        self._log_path: str = log_path
        self._log_verbose: bool = log_verbose

    @classmethod
    def get_instance(
        cls,
        *,
        extensions_cfg: "dict[str, Any] | None" = None,
        log_path: str = "",
        log_verbose: bool = False,
    ) -> "WebSearchFilter":
        """Return the singleton, building it on first call.

        Subsequent calls are no-ops — the instance is returned as-is.
        The first caller (usually the app init path) should supply all
        parameters; later callers can omit them.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        extensions_cfg=extensions_cfg,
                        log_path=log_path,
                        log_verbose=log_verbose,
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_query(
        self, query: str, *, path: str = "web"
    ) -> tuple[int, str, list[str]]:
        """Score *query* and return ``(score, outcome, reasons)``.

        Parameters
        ----------
        query:
            The final query string (after translation/rewrite).
        path:
            ``"web"`` or ``"local"`` — written to the log, not used in scoring.

        Returns
        -------
        score:
            Clamped integer [0, 200].
        outcome:
            ``"ALLOW"`` / ``"ALLOW_WITH_SAFETY_FRAMING"`` / ``"REFUSE"``.
        reasons:
            List of signal labels that contributed (entity categories + intent
            pattern names that fired, prefixed by ``"-"`` for mitigations).
        """
        q_lower = query.lower()
        score = 0
        reasons: list[str] = []

        # ── Entity matching ──
        matched_entity_categories: set[str] = set()
        for cat_name, cat in self._entities.items():
            for term in cat["entities"]:
                if _term_matches(term, q_lower):
                    matched_entity_categories.add(cat_name)
                    break  # one match per category is enough

        for cat_name in matched_entity_categories:
            score += self._entities[cat_name]["weight"]
            reasons.append(cat_name)

        # ── Intent pattern matching ──
        matched_intent: set[str] = set()
        for pat_name, pat_cfg in self._intent_patterns.items():
            if pat_cfg["pattern"].search(query):
                score += pat_cfg["weight"]
                reasons.append(pat_name)
                matched_intent.add(pat_name)

        # ── Mitigation pattern matching ──
        for pat_name, pat_cfg in self._mitigation_patterns.items():
            if pat_cfg["pattern"].search(query):
                score += pat_cfg["weight"]  # weight is negative
                reasons.append(f"-{pat_name}")

        # ── Hard-rule check ──
        hard_rule_fired = (
            _HARD_RULE
            and bool(matched_entity_categories & _HARD_RULE_ENTITY_GATES)
            and bool(matched_intent & _HARD_RULE_INTENT_GATES)
        )

        # ── Clamp and determine outcome ──
        score = max(0, min(score, 200))
        refuse_threshold = self._thresholds["refuse"]
        warn_threshold = self._thresholds["warn"]

        # Score-based REFUSE requires at least one entity match — a query with
        # only generic procedural intent but no dangerous subject (e.g. "step by
        # step instructions for manufacturing rock candy") must never be refused,
        # only warned.  The hard rule already requires entity + intent.
        if hard_rule_fired or (score >= refuse_threshold and matched_entity_categories):
            outcome = "REFUSE"
        elif score >= warn_threshold:
            outcome = "ALLOW_WITH_SAFETY_FRAMING"
        else:
            outcome = "ALLOW"

        self.log_decision(
            query=query,
            score=score,
            outcome=outcome,
            reasons=reasons,
            path=path,
            hard_rule=hard_rule_fired,
        )
        return score, outcome, reasons

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        *,
        query: str,
        score: int,
        outcome: str,
        reasons: list[str],
        path: str,
        collection: str = "",
        hard_rule: bool = False,
    ) -> None:
        """Append one pipe-separated line to ``intent_filter.log``.

        Skipped when ``_log_path`` is empty.
        """
        if not self._log_path:
            return
        if outcome == "ALLOW" and not self._log_verbose:
            return
        try:
            parent = os.path.dirname(self._log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            reasons_str = "+".join(reasons) if reasons else ""
            line = (
                f"{ts}"
                f" | path={path}"
                f" | outcome={outcome}"
                f" | score={score}"
                f" | hard_rule={hard_rule}"
                f" | collection={collection}"
                f" | query={query!r}"
                f" | reasons={reasons_str!r}"
                "\n"
            )
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass  # log failure must never abort a query


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _term_matches(term: str, q_lower: str) -> bool:
    """Return True if *term* appears as a whole-word match in *q_lower*.

    Multi-word terms are matched as subsequences (each word must appear).
    Single-word terms are matched as whole words via a simple boundary check.
    """
    words = term.lower().split()
    if len(words) == 1:
        # whole-word boundary check without re overhead
        idx = q_lower.find(words[0])
        while idx != -1:
            before = idx == 0 or not q_lower[idx - 1].isalnum()
            after = (
                idx + len(words[0]) >= len(q_lower)
                or not q_lower[idx + len(words[0])].isalnum()
            )
            if before and after:
                return True
            idx = q_lower.find(words[0], idx + 1)
        return False
    # multi-word: all words must appear somewhere in the query
    return all(w in q_lower for w in words)


def _merge_entities(
    baseline: dict[str, dict[str, Any]],
    extensions_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Merge baseline entity catalogue with config extensions.

    Rules:
    * Existing baseline categories are preserved; their entity lists are
      extended (never replaced or shrunk).
    * Extra categories from ``entity_categories_extra`` are added verbatim
      provided they do not collide with baseline category names (collision
      → silently ignored to prevent shadowing the baseline).
    """
    import copy

    merged: dict[str, dict[str, Any]] = copy.deepcopy(baseline)

    entity_extensions: dict[str, Any] = extensions_cfg.get("entity_extensions", {})
    for cat_name, extra_terms in entity_extensions.items():
        if cat_name in merged and isinstance(extra_terms, list):
            existing: set[str] = set(merged[cat_name]["entities"])
            terms: list[str] = cast("list[str]", extra_terms)
            merged[cat_name]["entities"] = list(existing | set(terms))

    extra_categories: dict[str, dict[str, Any]] = extensions_cfg.get(
        "entity_categories_extra", {}
    )
    for cat_name, cat_cfg in extra_categories.items():
        if cat_name not in merged:  # never shadow baseline
            merged[cat_name] = cat_cfg

    return merged


def _apply_threshold_overrides(
    baseline: dict[str, int],
    extensions_cfg: dict[str, Any],
) -> dict[str, int]:
    """Return thresholds after applying overrides from config.

    Configuration may only *tighten* (lower) thresholds.  An override that
    would raise a threshold above the baseline is silently ignored.
    """
    overrides: dict[str, Any] = extensions_cfg.get("threshold_overrides", {})
    result = dict(baseline)
    for key, val in overrides.items():
        if key in result and isinstance(val, int) and val < result[key]:
            result[key] = val
    return result
