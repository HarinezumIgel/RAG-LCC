"""
LicensePager — shared terminal pager for interactive license display.

Used by Compliance.py and ArgosDownloader.py.
Setup.py and NLTK_Stopwords_WordNet.py are stdlib-only standalone scripts
and keep their own simpler less-based _page_text.
"""

import re
import shutil
from typing import Any

from Gui.Colors import RESET, YELLOW

_END_OF_LICENSE = (
    "\n\n" + "-" * 70 + "\n" "  >>>>>  End of license  <<<<<\n" + "-" * 70 + "\n"
)


def _compute_height() -> int:
    try:
        h = shutil.get_terminal_size().lines - 2
        return h if h >= 5 else 15
    except Exception:
        return 15


def show_license(text: str) -> None:
    """Display license text page-by-page with an end-of-license marker.

    Uses rich (markdown / HTML rendering) when available, with a plain
    fallback.  The end-of-license banner is always appended so users who
    read to the end see a clear confirmation.
    """
    text = text + _END_OF_LICENSE

    try:
        from rich.console import Console
        from rich.markdown import Markdown

        rich_available: bool = True
    except Exception:
        Console: Any = None
        Markdown: Any = None
        rich_available: bool = False

    lines: list[str] = text.splitlines()
    height: int = _compute_height()

    if len(lines) <= height:
        if rich_available:
            console = Console()
            console.clear()
            wants_md = (
                text.lstrip().startswith("#")
                or "```" in text
                or bool(re.search(r"^- ", text, re.M))
            )
            if wants_md:
                console.print(Markdown(text, code_theme="monokai", hyperlinks=True))
            else:
                console.print(text)
        else:
            print(text)
        return

    console: Any = Console() if rich_available else None
    content: str = text

    if content.lstrip().startswith("<"):
        try:
            import html2text as _html2text

            h = _html2text.HTML2Text()
            h.ignore_images = True
            content = h.handle(content)
        except Exception:
            content = re.sub(r"<[^>]+>", "", content)

    wants_md = (
        content.lstrip().startswith("#")
        or "```" in content
        or bool(re.search(r"^- ", content, re.M))
    )

    if wants_md and rich_available:
        console.clear()
        with console.pager(styles=True):
            console.print(Markdown(content, code_theme="monokai", hyperlinks=True))
        return

    lines = content.splitlines()
    index: int = 0
    while index < len(lines):
        if rich_available:
            console.clear()
            page = lines[index : index + height]
            console.print("\n".join(page))
        else:
            page = lines[index : index + height]
            print("\n".join(page))
        index += height
        if index >= len(lines):
            break
        key = (
            input(f"{YELLOW}[Enter] next page, [q] quit view: {RESET}").strip().lower()
        )
        if key == "q":
            return
