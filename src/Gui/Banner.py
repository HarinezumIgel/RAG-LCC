import time
import unicodedata

from Config.Config import Config
from Gui.Colors import BRIGHT_ORANGE, RESET
from Gui.Symbols import Symbols


class Banner:
    def __init__(self, cfg: Config) -> None:
        self.cfg: Config = cfg or Config()
        self.friendly_name: str = cfg.get_str("_FRIENDLY_NAME")
        self.version: str = cfg.get_str("_VERSION")

    def _char_display_width(self, ch: str) -> int:
        """
        Heuristic display width for a single character.
        Uses east_asian_width: 'F' or 'W' -> 2, others -> 1.
        This covers most emoji and wide glyphs.
        """
        if not ch:
            return 0
        w = unicodedata.east_asian_width(ch)
        return 2 if w in ("F", "W") else 1

    def _display_width(self, s: str) -> int:
        """Return the sum of display widths for all characters in s."""
        return sum(self._char_display_width(ch) for ch in s)

    def _center_to_display_width(self, text: str, target_width: int) -> str:
        """
        Center `text` into a string of spaces whose display width equals target_width.
        Returns a plain string of spaces + text + spaces (no special padding chars).
        """
        text = str(text)
        text_w = self._display_width(text)
        if text_w >= target_width:
            # truncate by characters until fits (preserve codepoints)
            truncated = ""
            cur = 0
            for ch in text:
                ch_w = self._char_display_width(ch)
                if cur + ch_w > target_width:
                    break
                truncated += ch
                cur += ch_w
            text = truncated
            text_w = cur

        total_pad = target_width - text_w
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return " " * left_pad + text + " " * right_pad

    def startup_banner(self, display_width: int = 60) -> None:
        """
        Print a framed banner using the hedgehog emoji for the frame when
        emoji are supported, otherwise fall back to '*'.
        Uses display-width aware centering so all lines visually align.
        :param display_width: desired total display width in terminal columns
        """
        frame_char = Symbols.sym_banner_char()

        # compute display width of one frame char
        frame_char_disp_w = self._char_display_width(frame_char)

        # number of frame characters needed to fill the desired display width
        char_count = display_width // frame_char_disp_w
        total_display_width = char_count * frame_char_disp_w

        # inner display width (space available for text)
        inner_display_width = total_display_width - 2 * frame_char_disp_w
        if inner_display_width < 0:
            inner_display_width = 0

        # Build visual border (frame_char repeated to fill display_width)
        border = frame_char * char_count

        # Build an "empty" inner line (frame_char + spaces + frame_char)
        empty_inner = frame_char + " " * inner_display_width + frame_char

        # Content lines
        title = f"HarinezumIgel {self.friendly_name}"
        version = f"Version {self.version}"
        content_lines = [
            title,
            "An **experimental** lab environment",
            "to test settings and filter chains",
            version,
        ]

        # Prepare centered content using display-aware centering
        centered_lines: list[str] = []
        for t in content_lines:
            centered = self._center_to_display_width(t, inner_display_width)
            centered_lines.append(f"{frame_char}{centered}{frame_char}")

        # Print banner
        print(f"{BRIGHT_ORANGE}\n\n{border}")
        print(empty_inner)
        for ln in centered_lines:
            print(ln)
        print(empty_inner)
        print(f"{border}\n\n{RESET}")

        time.sleep(3)
