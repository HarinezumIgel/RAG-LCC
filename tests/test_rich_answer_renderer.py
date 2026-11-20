from __future__ import annotations

import io
import re

import pytest
from rich.cells import cell_len
from rich.color import Color
from rich.console import Console
from rich.text import Text

from Grounding.Grounder import Grounder
from Gui.Colors import ORANGE, TURQUOISE, YELLOW
from Gui.RichAnswerRenderer import ANSWER_META_SEPARATOR, RichAnswerRenderer
from Gui.Symbols import Symbols


def _plain(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


@pytest.mark.parametrize("partial", [False, True])
def test_unverified_answer_label_colour_in_cli(partial: bool):
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=100, force_terminal=True, color_system="truecolor")
    )
    renderer.render(
        Grounder.unverified_answer_notice(cli=True, partial=partial)
        + "Generated prose.",
        100,
        foreground="\033[38;2;220;220;220m",
    )
    rendered = output.getvalue()
    expected_label = (
        "Answer not completely verified." if partial else "Answer not verified."
    )
    label_line = next(line for line in rendered.splitlines() if expected_label in line)
    assert (ORANGE if partial else TURQUOISE)[2:-1] in label_line
    icon = Symbols.sym_icon("WARNING" if partial else "EXCLAMATION")
    assert icon.replace("\ufe0f", "").strip() in _plain(label_line)
    assert "Generated prose." in _plain(rendered)


@pytest.mark.parametrize("terminal_emoji", [False, True])
def test_partial_warning_preserves_emoji_presentation_for_openwebui(
    monkeypatch: pytest.MonkeyPatch,
    terminal_emoji: bool,
) -> None:
    monkeypatch.setattr(
        Symbols,
        "sym_icon",
        lambda _name: "\u26a0\ufe0f " if terminal_emoji else "\u26a0 ",
    )

    notice = Grounder.unverified_answer_notice(partial=True)
    assert notice.startswith("\u26a0\ufe0f **Answer not completely verified.**")
    cli_notice = Grounder.unverified_answer_notice(partial=True, cli=True)
    assert cli_notice.startswith(ORANGE + "\u26a0 **Answer not completely verified.**")
    assert "\ufe0f" not in cli_notice


def test_rich_renderer_displays_markdown_table_list_and_mermaid_source():
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=100, force_terminal=True, color_system="standard")
    )
    answer = (
        "## Results\n\n"
        "| Animal | Habitat |\n| --- | --- |\n| Hedgehogs | gardens |\n\n"
        "- Small mammals\n"
        "  - Hedgehogs\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  A[Hedgehogs] --> B[Gardens]\n"
        "```"
    )

    renderer.render(answer, 100, prefix="> ")

    rendered = _plain(output.getvalue())
    assert "Results" in rendered
    assert "Animal" in rendered and "Habitat" in rendered
    assert "Hedgehogs" in rendered and "gardens" in rendered
    assert rendered.count("•") >= 2
    assert "flowchart LR" in rendered
    assert "A[Hedgehogs] --> B[Gardens]" in rendered


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("table", [False, True])
def test_grounding_preserves_table_cells_and_separate_bullets(
    marked: bool, table: bool
):
    from Config.Config import Config
    from Grounding.Grounder import Grounder

    evidence = [
        "Cats have lived alongside humans for thousands of years.",
        "Dogs need training, exercise, and responsible ownership.",
        "Hedgehogs are small nocturnal mammals with distinctive coats of spines.",
    ]
    animals = ["Cats", "Dogs", "Hedgehogs"]
    if table:
        answer = "| Animal | Mammal Status | Evidence Phrase |\n| --- | --- | --- |\n"
        answer += "\n".join(
            f'| {animal} | Yes | "{quote}" |'
            for animal, quote in zip(animals, evidence)
        )
    else:
        answer = "\n".join(
            f'- **{animal}** - Mammal: Yes - Evidence: "{quote}"'
            for animal, quote in zip(animals, evidence)
        )

    output = io.StringIO()
    console = Console(file=output, width=120, force_terminal=True, color_system="256")
    display = (
        Grounder(cfg=Config()).ground_answer_cli(
            answer,
            [evidence[0], evidence[2]],
            ansi_codes="48;5;214",
        )
        if marked
        else answer
    )
    assert _plain(display) == answer
    RichAnswerRenderer(console).render(display, 120, background="\033[48;5;238m")

    rendered = _plain(output.getvalue())
    if not table:
        assert rendered.count("•") == len(animals)
    for animal, quote in zip(animals, evidence):
        assert " ".join(quote.split()) in " ".join(rendered.split())
        line = next(line for line in rendered.splitlines() if quote in line)
        assert re.search(rf"\b{animal}\b\s+(?:-\s+Mammal:\s+)?Yes\b", line)
    if marked:
        assert "48;5;214" in output.getvalue()


@pytest.mark.parametrize("background", ["", "\033[48;5;238m"])
def test_rich_renderer_preserves_grounding_ansi_marks_across_color_systems(
    background: str,
):
    for color_system in ("standard", "256", "truecolor", "standard", "256"):
        output = io.StringIO()
        console = Console(
            file=output,
            width=80,
            force_terminal=True,
            color_system=color_system,
        )
        renderer = RichAnswerRenderer(console)

        renderer.render(
            "\033[48;5;214mSupported claim.\033[0m",
            80,
            background=background,
        )

        decoded = Text.from_ansi(output.getvalue().splitlines()[0])
        offset = decoded.plain.index("Supported claim.")
        expected = Color.from_ansi(11 if color_system == "standard" else 214)
        for index in range(offset, offset + len("Supported claim.")):
            assert (
                decoded.get_style_at_offset(
                    console,
                    index,
                ).bgcolor
                == expected
            )


def test_rich_renderer_background_spans_every_output_line():
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=60, force_terminal=True, color_system="truecolor")
    )
    answer = (
        "## Answer\n\n"
        "Fish are aquatic.\n\n"
        "| A | B |\n| - | - |\n| x | y |\n\n"
        "Closing sentence."
    )

    renderer.render(answer, 60, background="\033[48;2;45;45;45m")

    bg = "48;2;45;45;45"
    lines = [line for line in output.getvalue().splitlines() if line]
    assert len(lines) > 5
    # Every line — including blank spacer lines and the area next to the
    # table — must carry the answer background so the block is continuous.
    assert all(bg in line for line in lines)


@pytest.mark.parametrize("width", [30, 80])
@pytest.mark.parametrize("background", ["", "\033[48;5;238m"])
@pytest.mark.parametrize("grounded", [False, True])
def test_inline_command_inherits_background_without_breaking_padding(
    width: int,
    background: str,
    grounded: bool,
):
    from Grounding.Grounder import Grounder

    output = io.StringIO()
    console = Console(file=output, width=width, force_terminal=True, color_system="256")
    answer = Grounder.no_evidence_message()
    if grounded:
        answer = "\033[48;5;214m" + answer + "\033[0m"
    answer += "\n\nFollowing paragraph."

    RichAnswerRenderer(console).render(answer, width, background=background)

    lines = output.getvalue().splitlines()
    expected_background = Color.from_ansi(238) if background else None
    expected_command_background = (
        Color.from_ansi(214) if grounded else expected_background
    )
    command_line = next(line for line in lines if "new:" in _plain(line))
    decoded = Text.from_ansi(command_line)
    start = decoded.plain.index("new:")
    for offset in range(start, start + len("new:")):
        style = decoded.get_style_at_offset(console, offset)
        assert style.bgcolor == expected_command_background
        assert style.bold
    assert "`" not in decoded.plain
    assert "\033" not in decoded.plain
    for line in lines:
        decoded = Text.from_ansi(line)
        assert cell_len(decoded.plain) <= width
        if background:
            assert cell_len(decoded.plain) == width
            assert decoded.get_style_at_offset(console, width - 1).bgcolor == (
                expected_background
            )
        if "Following" in decoded.plain:
            assert (
                decoded.get_style_at_offset(
                    console, decoded.plain.index("Following")
                ).bgcolor
                == expected_background
            )
    # Answer rendering must not alter styles used by other console output.
    assert console.get_style("markdown.code").bgcolor == Color.parse("black")


def test_rich_renderer_wraps_at_configured_width_and_fills_terminal_width():
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=60, force_terminal=True, color_system="truecolor")
    )
    answer = "word " * 40

    renderer.render(answer, 30, background="\033[48;2;45;45;45m")

    lines = [line for line in output.getvalue().splitlines() if line]
    plain_lines = [_plain(line) for line in lines]
    assert lines
    assert all(len(line) == 60 for line in plain_lines)
    assert max(len(line.strip()) for line in plain_lines) <= 30


def test_rich_renderer_repairs_grounding_reset_holes_in_background():
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=60, force_terminal=True, color_system="truecolor")
    )
    answer = (
        "```mermaid\nflowchart LR\n"
        "  \033[48;5;214mA[Hedgehogs] --> B[Gardens]\033[0m\n"
        "```"
    )

    renderer.render(answer, 60, background="\033[48;2;45;45;45m")

    # The grounding reset must be followed by the block background again so
    # the rest of the line (padding) is not left unstyled (ragged edge).
    assert "\033[0m\033[48;2;45;45;45m" in output.getvalue()
    assert "\033[48;5;214m" in output.getvalue()


def test_rich_renderer_gives_answer_body_a_distinct_background():
    output = io.StringIO()
    renderer = RichAnswerRenderer(
        Console(file=output, width=60, force_terminal=True, color_system="truecolor")
    )
    answer = (
        "The grounded answer body."
        + ANSWER_META_SEPARATOR
        + "**Answer confidence:** LOW"
    )

    renderer.render(
        answer,
        60,
        background="\033[48;2;45;45;45m",
        body_background="\033[48;2;32;42;60m",
    )

    rendered = output.getvalue()
    body_line = next(line for line in rendered.splitlines() if "body" in line)
    meta_line = next(line for line in rendered.splitlines() if "confidence" in line)
    assert "48;2;32;42;60" in body_line  # body gets the distinct background
    assert "48;2;45;45;45" in meta_line  # metadata keeps the standard one
    assert "48;2;32;42;60" not in meta_line
    assert "answer-meta" not in rendered  # the separator itself is invisible


@pytest.mark.parametrize("terminal_width", [30, 60, 100])
def test_grounded_mermaid_fills_every_terminal_cell(terminal_width: int):
    output = io.StringIO()
    console = Console(
        file=output, width=100, force_terminal=True, color_system="truecolor"
    )
    renderer = RichAnswerRenderer(console)
    claims = [
        "    Lions -->|Mammals| Yes",
        "    Hedgehogs -->|Mammals| Yes",
        "    Elephants -->|Mammals| Yes",
    ]
    answer = (
        "```mermaid\nflowchart LR\n"
        + "\n".join(f"\033[48;5;214m{claim}\033[0m" for claim in claims)
        + "\n```"
        + ANSWER_META_SEPARATOR
        + "**Answer confidence:** HIGH"
    )

    renderer.render(
        answer,
        terminal_width,
        background="\033[48;2;45;45;45m",
        body_background="\033[48;2;32;42;60m",
    )

    lines = output.getvalue().splitlines()
    body_lines = lines[:-1]
    assert all(len(_plain(line)) == 100 for line in lines)
    for line in body_lines:
        decoded = Text.from_ansi(line)
        for offset in range(len(decoded)):
            assert decoded.get_style_at_offset(console, offset).bgcolor in (
                Color.from_rgb(32, 42, 60),
                Color.from_ansi(214),
            )
    visible = "".join("".join(_plain(line) for line in body_lines).split())
    for claim in claims:
        assert "".join(claim.split()) in visible
    assert "\033[48;5;214m" in output.getvalue()


@pytest.mark.parametrize("width", [80, 300])
@pytest.mark.parametrize(
    ("level", "status_color"),
    [
        ("HIGH", Color.from_ansi(2)),
        ("MEDIUM", Color.from_ansi(3)),
        ("LOW", Color.from_rgb(255, 165, 0)),
    ],
)
def test_confidence_colors_only_status_and_preserves_normal_details(
    width: int,
    level: str,
    status_color: Color,
):
    from Chat.Chatter import Chatter
    from Chat.ConfidenceEvaluator import AnswerConfidenceResult

    output = io.StringIO()
    console = Console(
        file=output, width=width, force_terminal=True, color_system="truecolor"
    )
    notice = Chatter._build_answer_confidence_notice_cli(
        AnswerConfidenceResult(
            level=level,
            score=0.62,
            components={},
            summary=(
                f"{level}\n\nHeuristic Confidence:\n\n"
                "- C_final=0.62: final confidence in [0,1] after penalties.\n\n"
                "Subjective ratings, not probabilities. The judge sees neither "
                "the question nor source evidence and may share the generator's "
                "blind spots. Accuracy, answer completeness, reasoning validity, "
                "and citation support: not assessed."
            ),
        )
    )

    RichAnswerRenderer(console).render(
        "Answer body." + ANSWER_META_SEPARATOR + notice,
        width,
        background="\033[48;2;45;45;45m",
        foreground="\033[38;2;220;220;220m",
        body_background="\033[48;2;32;42;60m",
    )

    lines = output.getvalue().splitlines()
    assert all(cell_len(_plain(line)) == width for line in lines)
    expected_colors = {
        level: status_color,
        "Heuristic": Color.from_rgb(220, 220, 220),
        "C_final": Color.from_rgb(220, 220, 220),
        "Subjective": Color.from_rgb(220, 220, 220),
    }
    seen: set[str] = set()
    for line in lines[1:]:
        decoded = Text.from_ansi(line)
        for offset in range(len(decoded)):
            assert decoded.get_style_at_offset(
                console, offset
            ).bgcolor == Color.from_rgb(45, 45, 45)
        for word, expected_color in expected_colors.items():
            if word in decoded.plain:
                seen.add(word)
                offset = decoded.plain.index(word)
                assert (
                    decoded.get_style_at_offset(console, offset).color == expected_color
                )
    assert seen == set(expected_colors)


@pytest.mark.parametrize(
    "markup",
    [
        "Supported **claim**.",
        "## Supported claim",
        "- Supported claim",
        "| Claim |\n| --- |\n| Supported claim |",
    ],
)
def test_grounded_markdown_fills_terminal_and_retains_highlights(markup: str):
    output = io.StringIO()
    console = Console(
        file=output, width=80, force_terminal=True, color_system="truecolor"
    )
    marked = markup.replace("Supported", "\033[48;5;214mSupported")
    marked = marked.replace("claim", "claim\033[0m")

    RichAnswerRenderer(console).render(marked, 80, background="\033[48;2;45;45;45m")

    lines = output.getvalue().splitlines()
    assert all(cell_len(_plain(line)) == 80 for line in lines)
    supported = next(line for line in lines if "Supported" in line)
    decoded = Text.from_ansi(supported)
    for word in ("Supported", "claim"):
        offset = decoded.plain.index(word)
        assert decoded.get_style_at_offset(console, offset).bgcolor == Color.from_ansi(
            214
        )
    offset = decoded.plain.index("claim")
    if "**claim**" in markup:
        assert decoded.get_style_at_offset(console, offset).bold
    for line in lines:
        decoded = Text.from_ansi(line)
        assert all(
            decoded.get_style_at_offset(console, offset).bgcolor is not None
            for offset in range(len(decoded))
        )
