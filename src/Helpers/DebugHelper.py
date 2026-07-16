from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Config.Config import Config


class DebugHelper:
    """Thin helper for evaluating DEBUG_LEVEL guards.

    Instantiate once per component (or share an instance) and call the
    check methods instead of repeating ``cfg.get_int("DEBUG_LEVEL") >= N``
    at every call site.

    Usage::

        from Helpers.DebugHelper import DebugHelper
        self.dbg = DebugHelper(self.cfg)

        if self.dbg.on(30):        # DEBUG_LEVEL >= 30  (unconditional)
            self.pretty.write(...)

        if self.dbg.only(45):      # DEBUG_LEVEL == 45  (unconditional)
            self.pretty.write(...)

        if self.dbg.active(30):    # respects DEBUG_MODE ("ge", "is", or "le")
            self.pretty.write(...)

    Parsing::

        level, mode = DebugHelper.parse("is 30")   # → (30, "is")
        level, mode = DebugHelper.parse("ge 30")   # → (30, "ge")
        level, mode = DebugHelper.parse("30")      # → (30, "is")  plain int defaults to is
        level, mode = DebugHelper.parse(30)        # → (30, "is")
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg

    def _parsed(self) -> tuple[int, str]:
        """Read DEBUG_LEVEL from config (raw string or int) and return (level, mode)."""
        raw = self._cfg.get_str("DEBUG_LEVEL", "none")
        return DebugHelper.parse(raw)

    # ── unconditional checks ─────────────────────────────────────────────────

    def on(self, level: int) -> bool:
        """Return True when DEBUG_LEVEL >= *level* (always uses >= regardless of mode)."""
        configured, _ = self._parsed()
        return configured >= level

    def only(self, level: int) -> bool:
        """Return True when DEBUG_LEVEL == *level* exactly (always uses == regardless of mode)."""
        configured, _ = self._parsed()
        return configured == level

    # ── mode-aware check ─────────────────────────────────────────────────────

    def active(self, level: int) -> bool:
        """Return True when the guard at *level* fires given the current DEBUG_LEVEL and DEBUG_MODE.

        - ``DEBUG_MODE = "ge"``  →  ``DEBUG_LEVEL >= level``  (default, activates this level and all below)
        - ``DEBUG_MODE = "is"``  →  ``DEBUG_LEVEL == level``  (activates this exact level only)
        - ``DEBUG_MODE = "le"``  →  ``DEBUG_LEVEL <= level``  (activates this level and all above)
        """
        configured, mode = self._parsed()
        if mode == "is":
            return configured == level
        if mode == "le":
            return configured <= level
        return configured >= level  # "ge" — default

    # ── parser ───────────────────────────────────────────────────────────────

    @staticmethod
    def level(cfg: Config) -> int:
        """Return the numeric debug level from config, parsing combined strings.

        Use ``check()`` for mode-aware guards; use this only when you need the
        raw integer (e.g. ``> 0`` presence checks, storing in a variable).
        """
        raw = cfg.get_str("DEBUG_LEVEL", "none")
        lvl, _ = DebugHelper.parse(raw)
        return lvl

    @staticmethod
    def check(cfg: Config, level: int) -> bool:
        """Mode-aware debug guard — replaces ``DebugHelper.level(cfg) >= level``.

        Reads ``DEBUG_LEVEL`` from config and applies the comparison mode:

        - ``"ge"``  →  ``configured >= level``  (default, this level and above)
        - ``"is"``  →  ``configured == level``  (exact level only)
        - ``"le"``  →  ``configured <= level``  (this level and below)
        """
        raw = cfg.get_str("DEBUG_LEVEL", "none")
        configured, mode = DebugHelper.parse(raw)
        if mode == "is":
            return configured == level
        if mode == "le":
            return configured <= level
        return configured >= level  # "ge" — default

    @staticmethod
    def check_session(session: Any, level: int) -> bool:
        """Mode-aware guard that reads level and mode directly from a Session object.

        Use instead of ``(session.debug_level or 0) >= level`` so that
        ``debug_mode`` ("ge", "is", "le") is respected.

        The session value may be stored as an int or a numeric string, and the
        mode may be missing or None; both are normalized safely.
        """
        raw_level = getattr(session, "debug_level", None)
        try:
            configured = int(raw_level) if raw_level not in (None, "") else 0
        except (TypeError, ValueError):
            configured = 0

        raw_mode = getattr(session, "debug_mode", "ge")
        mode = str(raw_mode or "ge").strip().lower()
        if mode in ("eq", "=="):
            mode = "is"
        if mode in (">=", "ge"):
            return configured >= level
        if mode in ("<=", "le"):
            return configured <= level
        if mode == "is":
            return configured == level
        return configured >= level

    @staticmethod
    def parse(raw: "str | int") -> tuple[int, str]:
        """Parse a raw debug-level specifier into ``(level, mode)``.

        Accepted formats (case-insensitive):

        =============  =========  ============================================
        Input          Result     Meaning
        =============  =========  ============================================
        ``30``         (30, "ge") plain int → >= (default)
        ``"30"``       (30, "ge") plain string number
        ``"ge 30"``    (30, "ge") explicit greater-equal
        ``">= 30"``    (30, "ge") alternative spelling
        ``"is 30"``    (30, "is") exact match
        ``"eq 30"``    (30, "is") alias for is
        ``"== 30"``    (30, "is") alternative spelling
        ``"le 30"``    (30, "le") less-equal (this level and below)
        ``"<= 30"``    (30, "le") alternative spelling
        ``"none"``     (0, "is")  alias for silent
        ``""``         (0, "is")  empty string → silent
        =============  =========  ============================================

        Raises ``ValueError`` on unrecognised input.
        """
        if isinstance(raw, int):
            return (0, "is") if raw == 0 else (raw, "ge")

        text: str = str(raw).strip().lower()

        # Named alias — empty string or "none" → silent
        if text in ("", "none"):
            return 0, "is"

        # Two-token form: "<mode> <number>"
        parts: list[str] = text.split(None, 1)
        if len(parts) == 2:
            mode_token, num_str = parts
            if mode_token in ("ge", ">="):
                mode = "ge"
            elif mode_token in ("is", "eq", "=="):
                mode = "is"
            elif mode_token in ("le", "<="):
                mode = "le"
            else:
                raise ValueError(
                    f"Unknown debug mode '{mode_token}'. "
                    "Use 'ge' / '>=' for greater-equal, 'is' / 'eq' / '==' for exact match, "
                    "or 'le' / '<=' for less-equal."
                )
            try:
                return int(num_str), mode
            except ValueError:
                raise ValueError(f"Debug level must be an integer, got '{num_str}'.")

        # Single token: plain integer — default mode is 'ge' (>=)
        try:
            n = int(text)
            return (0, "is") if n == 0 else (n, "ge")
        except ValueError:
            raise ValueError(
                f"Invalid debug level '{raw}'. "
                "Use a number (e.g. '30'), 'ge 30' (>=), 'is 30' (==), 'le 30' (<=), or 'none' (silent)."
            )
