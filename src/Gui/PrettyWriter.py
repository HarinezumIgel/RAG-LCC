import logging
import re
import textwrap
import time
from typing import Any

from Globals.Globals import Globals
from Gui.Colors import *
from Gui.Symbols import Symbols

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _fallback_wcswidth(s: Any) -> int:
    return len(s)


try:
    from wcwidth import wcswidth as _wcswidth  # type: ignore[assignment]

    wcswidth: Any = _wcswidth  # type: ignore[reportUnknownVariableType]
except Exception:
    wcswidth: Any = _fallback_wcswidth  # type: ignore[no-redef]


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so wcswidth counts display chars only."""
    return _ANSI_RE.sub("", text)


class PrettyWriter:
    """
    Utility class for pretty terminal and log output with icons, colors, and wrapping.
    """

    def __init__(
        self, logger: logging.Logger | None = None, *, cfg: Any = None
    ) -> None:
        """
        :param logger: an optional Python Logger; if provided,
                       calls to write() will also emit log entries.
        :param cfg:    optional Config instance (injected for testing).
        """
        from Config.Config import Config

        self.cfg: Config = cfg if isinstance(cfg, Config) else Config()
        self.terminal_line_size: int = self.cfg.get_int("TERMINAL_LINE_SIZE", 160)
        self.logger: logging.Logger | None = logger
        self.globalsInstance: Globals = Globals()

    def _get_icon(self, severity: str, restore_colour: str = RESET) -> str:
        """
        Map severity string to a terminal-safe icon.
        Delegates to Gui.Symbols which auto-detects emoji support.
        """
        return Symbols.severity_icon(severity, restore_colour)

    def write(
        self,
        severity: str,
        label: str,
        message: str,
        indent_level: int = 1,
        label_width: int = 30,
        max_line_length: int | None = None,
        sleep_time: int = 0,
        color: str | None = None,
    ) -> str:
        """
        Print a formatted message to terminal and optionally to logger.
        Handles icons, colors, wrapping, and severity mapping.
        """
        if severity in ("D", "DEBUG"):
            indent_level = 3

        if max_line_length is None:
            max_line_length = self.terminal_line_size

        indent: str = "" if indent_level == 1 else " " * indent_level
        message = str(message)

        # Determine the colour to restore after the icon
        restore = color if color else RESET
        icon: str = self._get_icon(severity, restore_colour=restore)

        # Build the initial prefix (with a trailing space before the message)
        prefix: str = f"{indent}{icon}{label:<{label_width}} "

        # Continuation marker to show on wrapped lines instead of the icon
        continuation: str = Symbols.sym_continuation()

        # Strip ANSI escapes for accurate display-width measurement
        prefix_display_width: int = wcswidth(_strip_ansi(prefix))
        indent_display_width: int = wcswidth(indent)
        continuation_display_width: int = wcswidth(continuation)

        # Number of spaces to add after indent+continuation so the total display width
        # equals the prefix display width (so wrapped text lines up under the message)
        pad_spaces: int = prefix_display_width - (
            indent_display_width + continuation_display_width
        )
        if pad_spaces < 0:
            # fallback: ensure at least one space after continuation
            pad_spaces = 1

        subsequent_indent: str = indent + continuation + (" " * pad_spaces)

        # Strip ANSI escapes from the message so textwrap counts only
        # visible characters; this prevents premature wrapping when the
        # message contains inline ANSI-coloured tokens (e.g. sym_ok).
        stripped_message: str = _strip_ansi(message)
        available: int = max(1, max_line_length - prefix_display_width)
        wrapped: list[str] = textwrap.wrap(
            stripped_message,
            width=available,
            break_long_words=False,
            break_on_hyphens=False,
        )

        if len(wrapped) <= 1:
            # No wrapping needed — keep original message with ANSI codes
            formatted = prefix + message
        else:
            # Wrapping needed — use ANSI-stripped lines for correct layout
            parts: list[str] = [prefix + wrapped[0]]
            for wl in wrapped[1:]:
                parts.append(subsequent_indent + wl)
            formatted = "\n".join(parts)

        if sleep_time:
            time.sleep(sleep_time)
        if color is None:
            print(formatted)
        else:
            print(f"{color}{formatted}{RESET}")

        # ——— Emit to runlog if logger was provided ———
        if self.logger is None:
            self.logger = self.globalsInstance.get_logger()
        if self.logger:
            lvl: str = severity.strip().upper()
            # map severity to Python log levels
            level_map: dict[str, int] = {
                "OK": logging.INFO,
                "O": logging.INFO,
                "INFO": logging.INFO,
                "I": logging.INFO,
                "WARNING": logging.WARNING,
                "W": logging.WARNING,
                "ERROR": logging.ERROR,
                "E": logging.ERROR,
                "ATTENTION": logging.WARNING,
                "A": logging.WARNING,
                "DEBUG": logging.DEBUG,
                "D": logging.DEBUG,
                "NEUTRAL": logging.INFO,
                "N": logging.INFO,
            }
            log_level: int = level_map.get(lvl, logging.INFO)
            # log just the label + message (icons aren’t needed in file)
            self.logger.log(log_level, f"{label}: {message}")

        return formatted
