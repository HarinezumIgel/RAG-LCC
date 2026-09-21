"""Terminal-safe symbols with automatic fallback on Windows.

Windows PowerShell (and many Windows console fonts) cannot render colour
emoji such as 🟢 or 🦔.  This module detects the situation and exposes
plain-text or ANSI-coloured alternatives that render correctly everywhere.

The auto-detected value is stored once in Config (key ``_USE_EMOJI``)
by ``StartupCommons.common_start``.  All public helpers read it back
from Config each time they are called.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from Gui.Colors import BRIGHT_BLUE, GREEN, MAGENTA, RED, RESET, WHITE, YELLOW

if TYPE_CHECKING:
    from Config.Config import Config


class Symbols:
    """Encapsulates terminal-safe symbol/icon helpers with emoji auto-detection."""

    # ── private config key (not exposed outside this class) ──────────
    _CFG_KEY: str = "_USE_EMOJI"

    # ── detection ────────────────────────────────────────────────────

    @staticmethod
    def emoji_supported() -> bool:
        """
        Heuristic: emoji render correctly on non-Windows, and on Windows
        only inside the VS Code integrated terminal (TERM_PROGRAM=vscode).
        Windows Terminal, legacy conhost, and PowerShell ISE cannot render
        them reliably.
        """
        if sys.platform != "win32":
            return True
        # VS Code sets TERM_PROGRAM=vscode in its integrated terminal.
        if os.environ.get("TERM_PROGRAM", "").lower() == "vscode":
            return True
        return False

    # Cached at class-definition time; used as fallback before Config exists.
    _detected: bool = emoji_supported()

    # ── emoji icon and ANSI fallback tables ──────────────────────────

    _EMOJI_ICONS: dict[str, str] = {
        "OK": "\U0001f7e2 ",  # 🟢
        "O": "\U0001f7e2 ",
        "INFO": "\U0001f535 ",  # 🔵
        "I": "\U0001f535 ",
        "WARNING": "\U0001f7e1 ",  # 🟡
        "W": "\U0001f7e1 ",
        "ERROR": "\U0001f534 ",  # 🔴
        "E": "\U0001f534 ",
        "ATTENTION": "\U0001f7e3 ",  # 🟣
        "A": "\U0001f7e3 ",
        "DEBUG": "\u26aa ",  # ⚪
        "D": "\u26aa ",
        "NEUTRAL": "   ",
        "N": "   ",
    }

    _ANSI_COLOURS: dict[str, str] = {
        "OK": GREEN,
        "O": GREEN,
        "INFO": BRIGHT_BLUE,
        "I": BRIGHT_BLUE,
        "WARNING": YELLOW,
        "W": YELLOW,
        "ERROR": RED,
        "E": RED,
        "ATTENTION": MAGENTA,
        "A": MAGENTA,
        "DEBUG": WHITE,
        "D": WHITE,
        "NEUTRAL": "",
        "N": "",
    }

    # Inline icons used in plain text output (outside PrettyWriter severity labels).
    # Unknown keys fall back to "INFO" so callers always get a safe neutral symbol.
    _INLINE_ICONS: dict[str, tuple[str, str]] = {
        "INFO": ("\u2139\ufe0f ", "\u2139 "),
        "WARNING": ("\u26a0\ufe0f ", "\u26a0 "),
        "WEB": ("\U0001f310 ", "\U0001f310 "),
        "DOC": ("\U0001f4c4 ", "\U0001f4c4 "),
        "CHAT": ("\U0001f4ac ", "\U0001f4ac "),
        "SEARCH": ("\U0001f50d ", "\U0001f50d "),
        "LINK": ("\U0001f4ce ", "\U0001f4ce "),
    }

    @staticmethod
    def store_emoji_preference(cfg: Config) -> bool:
        """Detect emoji support and persist the flag into *cfg*.

        Returns the detected boolean so callers can act on it.
        """
        emoji_ok: bool = Symbols.emoji_supported()
        cfg.set(Symbols._CFG_KEY, emoji_ok, create_missing=True)
        return emoji_ok

    @staticmethod
    def _use_emoji() -> bool:
        """Read the flag from Config.  Falls back to ``_detected`` if Config
        is not yet initialised.

        The import of Config is deliberately lazy (inside the function body)
        because of a circular dependency:
        Config -> PrettyWriter -> Symbols.  A top-level import would fail.
        """
        try:
            from Config.Config import Config  # lazy — avoids circular import

            val = Config().get(Symbols._CFG_KEY, Symbols._detected, silent=True)
            return bool(val)
        except Exception:
            return Symbols._detected

    # ── private helpers ──────────────────────────────────────────────

    @staticmethod
    def _fb(colour: str, restore: str) -> str:
        """Coloured bullet + restore (3 display cells, matching emoji icons)."""
        return f"{colour}\u25cf{restore}  "  # ● + 2 spaces = 3 display cells

    # ── severity icons (used by PrettyWriter) ────────────────────────

    @staticmethod
    def severity_icon(severity: str, restore_colour: str = RESET) -> str:
        """Return a 2-3 char icon string suitable for terminal display."""
        s = severity.strip().upper()
        if Symbols._use_emoji():
            return Symbols._EMOJI_ICONS.get(s, "   ")
        colour = Symbols._ANSI_COLOURS.get(s, "")
        if not colour:
            return "   "
        return Symbols._fb(colour, restore_colour)

    # ── inline symbols (Summarizer / ChunkSelector / Accumulator) ────

    @staticmethod
    def sym_ok() -> str:
        return "\u2705" if Symbols._use_emoji() else f"{GREEN}[Y]{RESET}"  # ✅

    @staticmethod
    def sym_fail() -> str:
        return "\u274c" if Symbols._use_emoji() else f"{RED}[X]{RESET}"  # ❌

    @staticmethod
    def sym_neutral() -> str:
        return (
            "\U0001f537" if Symbols._use_emoji() else f"{BRIGHT_BLUE}[-]{RESET}"
        )  # 🔷

    @staticmethod
    def sym_icon(kind: str) -> str:
        """Return inline icon by key with neutral INFO fallback when unknown."""
        key = (kind or "").strip().upper()
        emoji_icon, fallback_icon = Symbols._INLINE_ICONS.get(
            key,
            Symbols._INLINE_ICONS["INFO"],
        )
        return emoji_icon if Symbols._use_emoji() else fallback_icon

    @staticmethod
    def sym_warning() -> str:
        """Warning symbol for inline text without ASCII fallback."""
        return Symbols.sym_icon("WARNING")

    @staticmethod
    def sym_arrow() -> str:
        return "\u27a1\ufe0f " if Symbols._use_emoji() else "=> "  # ➡️

    @staticmethod
    def sym_continuation() -> str:
        """Marker shown at the start of wrapped continuation lines."""
        return "\u21b3 " if Symbols._use_emoji() else "-> "  # ↳

    @staticmethod
    def sym_banner_char() -> str:
        return "\U0001f994" if Symbols._use_emoji() else "*"  # 🦔
