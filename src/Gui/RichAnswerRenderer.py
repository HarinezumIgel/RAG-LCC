"""Rich-based rendering for CLI Markdown answers."""

from __future__ import annotations

import re
import sys
from typing import Self

from markdown_it.token import Token
from rich.align import Align
from rich.ansi import AnsiDecoder
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.markdown import (CodeBlock, Heading, Markdown, MarkdownContext,
                           MarkdownElement, Paragraph, TableDataElement)
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

from Gui.Colors import RESET, YELLOW

_SGR_RE = re.compile(r"\033\[([0-9;]*)m")

# Invisible Markdown marker separating the answer body from the trailing
# metadata (confidence notice, diagnostics). The body is rendered with its own
# background so the actual answer stands out.
ANSWER_META_SEPARATOR = "\n\n[//]: # (answer-meta)\n\n"


class _AnsiCodeBlock(CodeBlock):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        if _SGR_RE.search(code):
            # Decode before wrapping/padding so escape codes occupy no cells.
            yield Padding(Text.from_ansi(code, tab_size=4), 1)
        else:
            yield from super().__rich_console__(console, options)


class _AnsiElement(MarkdownElement):
    ansi_decoder: AnsiDecoder

    @classmethod
    def _with_decoder(cls, element: MarkdownElement, markdown: Markdown) -> Self:
        assert isinstance(element, cls)
        assert isinstance(markdown, _AnswerMarkdown)
        element.ansi_decoder = markdown.ansi_decoder
        return element

    def on_text(self, context: MarkdownContext, text: str | Text) -> None:
        if isinstance(text, str):
            decoded = self.ansi_decoder.decode_line(text)
            decoded.stylize_before(context.current_style)
            text = decoded
        super().on_text(context, text)


class _AnsiParagraph(_AnsiElement, Paragraph):
    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> _AnsiParagraph:
        return cls._with_decoder(super().create(markdown, token), markdown)


class _AnsiHeading(_AnsiElement, Heading):
    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> _AnsiHeading:
        return cls._with_decoder(super().create(markdown, token), markdown)


class _AnsiTableData(_AnsiElement, TableDataElement):
    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> _AnsiTableData:
        return cls._with_decoder(super().create(markdown, token), markdown)


class _AnswerMarkdown(Markdown):
    ansi_decoder: AnsiDecoder

    elements = {
        **Markdown.elements,
        "fence": _AnsiCodeBlock,
        "code_block": _AnsiCodeBlock,
        "paragraph_open": _AnsiParagraph,
        "heading_open": _AnsiHeading,
        "td_open": _AnsiTableData,
        "th_open": _AnsiTableData,
    }

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # ANSI color spans can cross Markdown tokens and paragraph boundaries.
        self.ansi_decoder = AnsiDecoder()
        # Rich's default inline-code background hides the answer background.
        with console.use_theme(
            Theme({"markdown.code": Style(bold=True, color="cyan")})
        ):
            yield from super().__rich_console__(console, options)


class _AnswerOutput:
    def __init__(self, renderable: RenderableType) -> None:
        self.renderable = renderable

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # Rich shares Style objects whose cached ANSI codes are not keyed by
        # console colour capability. Isolate styles after all layout is done.
        styles: dict[Style, Style] = {}
        for segment in console.render(self.renderable, options):
            style = segment.style
            if style is None:
                yield segment
                continue
            if style not in styles:
                styles[style] = Style(
                    color=style.color,
                    bgcolor=style.bgcolor,
                    bold=style.bold,
                    dim=style.dim,
                    italic=style.italic,
                    underline=style.underline,
                    blink=style.blink,
                    blink2=style.blink2,
                    reverse=style.reverse,
                    conceal=style.conceal,
                    strike=style.strike,
                    underline2=style.underline2,
                    frame=style.frame,
                    encircle=style.encircle,
                    overline=style.overline,
                    link=style.link,
                    meta=style.meta,
                )
            yield Segment(segment.text, styles[style], segment.control)


class RichAnswerRenderer:
    """Render an answer as terminal Markdown without changing its source text."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console

    def render(
        self,
        answer: str,
        terminal_width: int,
        *,
        prefix: str = "",
        original_query: str | None = None,
        color: str = "",
        background: str = "",
        foreground: str = "",
        body_background: str = "",
    ) -> None:
        """Print Markdown with Rich; Mermaid fences remain readable source blocks."""
        console = self.console or Console(file=sys.stdout)
        background_width = console.width
        style = self._style(color)
        style += self._style(background, foreground_only=False)
        style += self._style(foreground, foreground_only=True)
        body_style = style
        body_ansi_bg = background
        if body_background:
            body_style = style + self._style(body_background, foreground_only=False)
            body_ansi_bg = body_background

        if original_query is not None:
            # Plain print with explicit ANSI: the yellow query line must not
            # depend on Rich terminal detection (e.g. under piped output).
            print(f"{YELLOW}Original query: {original_query}{RESET}")
        if prefix:
            console.print(Text(prefix.rstrip()))

        body, separator, meta = answer.partition(ANSWER_META_SEPARATOR)
        self._print_markdown(
            console,
            body,
            terminal_width,
            background_width,
            body_style,
            body_ansi_bg,
            foreground,
        )
        if separator:
            self._print_markdown(
                console,
                meta,
                terminal_width,
                background_width,
                style,
                background,
                foreground,
            )

    @staticmethod
    def _print_markdown(
        console: Console,
        text: str,
        terminal_width: int,
        background_width: int,
        style: Style,
        ansi_background: str,
        ansi_foreground: str,
    ) -> None:
        """Render one Markdown section, keeping the block background intact."""
        if style:
            # Grounder marks end with an ANSI reset which would punch a hole
            # in the block background for the rest of that segment (ragged
            # right edges). Re-apply the block colors after every reset.
            repair = ansi_background + ansi_foreground
            if repair:
                text = text.replace("\033[0m", "\033[0m" + repair)
            # "ansi_dark" draws code blocks without their own background, so
            # fenced blocks (e.g. Mermaid) inherit the block background, and
            # Padding(expand=True) paints the background across full terminal
            # lines, including blank lines between blocks.
            markdown = _AnswerMarkdown(text, code_theme="ansi_dark", hyperlinks=True)
            console.print(
                _AnswerOutput(
                    Padding(
                        Align(markdown, align="left", width=terminal_width),
                        (0, 1),
                        style=style,
                        expand=True,
                    )
                ),
                width=background_width,
            )
        else:
            # Grounder ANSI spans are intentionally retained through parsing.
            markdown = _AnswerMarkdown(text, code_theme="monokai", hyperlinks=True)
            console.print(_AnswerOutput(markdown), width=terminal_width)

    @staticmethod
    def _style(ansi: str, *, foreground_only: bool = True) -> Style:
        if not ansi:
            return Style()

        foreground: Color | None = None
        background: Color | None = None
        for match in _SGR_RE.finditer(ansi):
            codes = [int(code or "0") for code in match.group(1).split(";")]
            index = 0
            while index < len(codes):
                code = codes[index]
                if code == 0:
                    foreground = None
                    background = None
                elif code in (38, 48) and index + 1 < len(codes):
                    target = "foreground" if code == 38 else "background"
                    mode = codes[index + 1]
                    if mode == 5 and index + 2 < len(codes):
                        value = Color.from_ansi(codes[index + 2])
                        index += 2
                    elif mode == 2 and index + 4 < len(codes):
                        value = Color.from_rgb(
                            codes[index + 2], codes[index + 3], codes[index + 4]
                        )
                        index += 4
                    else:
                        index += 1
                        continue
                    if target == "foreground":
                        foreground = value
                    else:
                        background = value
                elif 30 <= code <= 37 or 90 <= code <= 97:
                    foreground = Color.from_ansi(code - 30 if code < 90 else code - 82)
                elif 40 <= code <= 47 or 100 <= code <= 107:
                    background = Color.from_ansi(code - 40 if code < 100 else code - 92)
                index += 1

        if foreground_only:
            return Style(color=foreground)
        return Style(bgcolor=background)
