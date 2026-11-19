from __future__ import annotations

import io
import re
from html import unescape
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from Config.Config import Config
from Grounding.Grounder import Grounder
from Grounding.MermaidGrounder import MermaidGrounder
from Gui.GroundedAnswerHtmlRenderer import GroundedAnswerHtmlRenderer
from Gui.RichAnswerRenderer import RichAnswerRenderer


class _AnswerMarkConfig(Config):
    def get_str(self, key: str, default: str = "", *, silent: bool = False) -> str:
        if key == "_MARKED_DOCS_COLORS.answer_ansi":
            return "48;5;214"
        return super().get_str(key, default, silent=silent)


@pytest.fixture
def grounder():
    Grounder._reset()
    instance = Grounder(cfg=_AnswerMarkConfig())
    yield instance
    Grounder._reset()


class _EnglishDetector:
    def __call__(self, _text: str, _native_lang: str, _stage: str) -> str:
        return "english"


def _session():
    return SimpleNamespace(
        last_chosen_chunks=[],
        retrieval_language="english",
        debug_level=0,
        debug_mode="none",
        grounding_proxy_text="",
        grounding_proxy_sentences={},
    )


EVIDENCE = "Hedgehogs live in gardens and eat insects."
DIAGRAM = (
    "```mermaid\n" "flowchart LR\n" "  A[Hedgehogs] -->|live in| B[Gardens]\n" "```"
)


def test_flowchart_edge_becomes_ordered_grounding_claim():
    blocks = MermaidGrounder().find_blocks(DIAGRAM)

    assert len(blocks) == 1
    assert blocks[0].parseable
    assert blocks[0].claims == ("Hedgehogs live in Gardens",)


def test_mermaid_fence_language_is_case_insensitive():
    blocks = MermaidGrounder().find_blocks(DIAGRAM.replace("mermaid", "Mermaid"))

    assert len(blocks) == 1
    assert blocks[0].parseable


def test_only_grounded_mermaid_claims_pass_answer_guard(grounder):
    session = _session()
    assert grounder.has_grounded_evidence(
        session, DIAGRAM, [EVIDENCE], "english", _EnglishDetector()
    )

    unsupported = DIAGRAM.replace("live in", "manufacture")
    assert not grounder.has_grounded_evidence(
        session, unsupported, [EVIDENCE], "english", _EnglishDetector()
    )


def test_partially_grounded_mermaid_passes_but_marks_only_supported_lines(grounder):
    # Unsupported diagram claims stay unmarked without rejecting supported ones.
    two_edges = DIAGRAM.replace(
        "  A[Hedgehogs] -->|live in| B[Gardens]",
        "  A[Hedgehogs] -->|live in| B[Gardens]\n"
        "  B[Gardens] -->|manufacture| C[Purple spaceships]",
    )
    session = _session()
    assert grounder.has_grounded_evidence(
        session, two_edges, [EVIDENCE], "english", _EnglishDetector()
    )

    marked = grounder.mark_answer_cli(session, two_edges, [EVIDENCE])
    assert "\033[48;5;214m  A[Hedgehogs] -->|live in| B[Gardens]\033[0m" in marked
    assert "\033[48;5;214m  B[Gardens]" not in marked
    assert "Purple spaceships]\033[0m" not in marked


VERDICT_DIAGRAM = (
    "```mermaid\n"
    "flowchart LR\n"
    "    Cats -->|Mammals| Yes\n"
    "    Fish -->|Not mammals| No\n"
    "```"
)
VERDICT_EVIDENCE = [
    "Cats are placental mammals that nurse their young.",
    "Fish are prey for birds, mammals, and other fish.",
]


def test_verdict_diagram_grounds_via_single_chunk_co_occurrence(grounder):
    session = _session()
    assert grounder.has_grounded_evidence(
        session, VERDICT_DIAGRAM, VERDICT_EVIDENCE, "english", _EnglishDetector()
    )

    marked = grounder.mark_answer_cli(session, VERDICT_DIAGRAM, VERDICT_EVIDENCE)
    assert "\033[48;5;214m    Cats -->|Mammals| Yes\033[0m" in marked


def test_verdict_claims_match_singular_plural_and_abbreviated_labels(grounder):
    # Diagram labels rarely match evidence word forms exactly: the docs say
    # "mammal" (singular) and "chimpanzees" while the diagram says "Mammals"
    # and "Chimps".
    diagram = (
        "```mermaid\n"
        "flowchart LR\n"
        "    Hedgehogs -->|Mammals| Yes\n"
        "    Chimps -->|Mammals| Yes\n"
        "```"
    )
    evidence = [
        "The hedgehog is a small spiny mammal found in gardens.",
        "Chimpanzees are mammals closely related to humans.",
    ]
    assert grounder.has_grounded_evidence(
        _session(), diagram, evidence, "english", _EnglishDetector()
    )


def test_short_stems_do_not_prefix_match_unrelated_words(grounder):
    # "Cats" must not ground against "catalog" — short stems require equality.
    diagram = "```mermaid\nflowchart LR\n    Cats -->|Mammals| Yes\n```"
    evidence = ["The catalog lists every mammal species."]
    assert not grounder.has_grounded_evidence(
        _session(), diagram, evidence, "english", _EnglishDetector()
    )


def test_verdict_diagram_with_hallucinated_subject_leaves_line_unmarked(grounder):
    # "Spaceships" is hallucinated, but the Fish claim is supported, so the
    # diagram passes with only the supported line highlighted.
    session = _session()
    hallucinated = VERDICT_DIAGRAM.replace("Cats", "Spaceships")
    assert grounder.has_grounded_evidence(
        session, hallucinated, VERDICT_EVIDENCE, "english", _EnglishDetector()
    )

    marked = grounder.mark_answer_cli(session, hallucinated, VERDICT_EVIDENCE)
    assert "\033[48;5;214m    Fish -->|Not mammals| No\033[0m" in marked
    assert "\033[48;5;214m    Spaceships" not in marked


def test_fully_ungrounded_diagram_fails_closed(grounder):
    diagram = (
        "```mermaid\n"
        "flowchart LR\n"
        "    Spaceships -->|Mammals| Yes\n"
        "    Unicorns -->|Mammals| Yes\n"
        "```"
    )
    assert not grounder.has_grounded_evidence(
        _session(), diagram, VERDICT_EVIDENCE, "english", _EnglishDetector()
    )


def test_verdict_claim_tokens_must_co_occur_in_one_chunk(grounder):
    # "Cats" and "Mammals" live in different chunks: assembling the claim
    # across chunks is not evidence for the relationship.
    split_evidence = ["Cats purr loudly.", "Mammals nurse their young."]
    diagram = "```mermaid\nflowchart LR\n    Cats -->|Mammals| Yes\n```"
    assert not grounder.has_grounded_evidence(
        _session(), diagram, split_evidence, "english", _EnglishDetector()
    )


@pytest.mark.parametrize(
    "diagram",
    [
        "```mermaid\nsequenceDiagram\nA->>B: hello\n```",
        "```mermaid\nflowchart LR\nA[Hedgehogs] -->",
    ],
)
def test_unparsed_mermaid_is_rejected(grounder, diagram):
    assert not grounder.has_grounded_evidence(
        _session(), diagram, [EVIDENCE], "english", _EnglishDetector()
    )


def test_diagram_is_not_mistaken_for_a_short_answer(grounder):
    assert not grounder.is_answer_too_short(DIAGRAM)


def test_cli_marks_only_fully_grounded_mermaid_blocks(grounder):
    session = _session()

    marked = grounder.mark_answer_cli(session, DIAGRAM, [EVIDENCE])
    assert "\033[48;5;214m  A[Hedgehogs] -->|live in| B[Gardens]\033[0m" in marked

    unsupported = grounder.mark_answer_cli(
        session, DIAGRAM.replace("live in", "manufacture"), [EVIDENCE]
    )
    assert "\033[" not in unsupported


def test_html_marks_grounded_prose_without_marking_mermaid_source(grounder):
    session = _session()
    answer = (
        "Hedgehogs live in gardens and eat insects. "
        "Unicorns live in castles.\n\n" + DIAGRAM
    )

    marked = grounder.mark_answer_html(session, answer, [EVIDENCE])

    assert (
        '<mark class="grounded">Hedgehogs live in gardens and eat insects.</mark>'
        in marked
    )
    assert "Unicorns live in castles." in marked
    assert '<mark class="grounded">Unicorns live in castles.</mark>' not in marked
    assert "```mermaid\nflowchart LR" in marked
    assert '<mark class="grounded">  A[Hedgehogs]' not in marked


def test_html_renderer_formats_markdown_and_enables_grounded_mermaid(grounder):
    session = _session()
    answer = (
        "Hedgehogs live in gardens and eat insects.\n\n"
        "| Animal | Habitat |\n| --- | --- |\n| Hedgehogs | gardens |\n\n" + DIAGRAM
    )

    page, csp = GroundedAnswerHtmlRenderer().render(
        answer, grounder, session, [EVIDENCE]
    )

    assert "<table>" in page and "<th>Animal</th>" in page
    assert '<link rel="icon" type="image/png" href="data:image/png;base64,' in page
    assert (
        '<mark class="grounded">Hedgehogs live in gardens and eat insects.</mark>'
        in page
    )
    assert 'class="language-mermaid"' in page
    assert "mermaid@11.4.1" in page
    assert 'securityLevel: "strict"' in page
    assert "nonce-" in csp
    assert "connect-src 'none'" in csp


@pytest.mark.parametrize(
    "items",
    [
        [
            "**PCIe x16 Card Slot 1**: Located on the system board, this slot is designed for full-length PCIe x16 cards.",
            "**PCIe x16 Card Slot 2**: Another full-length PCIe x16 slot on the system board.",
            "**PCIe x16 Card Slot 3**: A third full-length PCIe x16 slot on the system board.",
            "**PCIe x16 Card Slot 4**: A fourth full-length PCIe x16 slot on the system board.",
            "**PCIe x8 Card Slot 5**: A half-length PCIe x8 slot on the system board.",
            "**PCIe x8 Card Slot 6**: Another half-length PCIe x8 slot on the system board.",
        ],
        [
            "PCIe card slots support 6 PCIe slots with 4x PCIe 4.0 x16 slots plus 2x PCIe 4.0 x8 slots.",
            "Slot 1: PCIe 4.0 x16, full height, full length, 75W, double-width, by CPU.",
        ],
    ],
)
@pytest.mark.parametrize("verified", [True, False])
def test_html_lists_keep_each_item_separate_with_grounding_marks(
    grounder,
    items,
    verified,
):
    answer = "\n".join(f"- {item}" for item in items)

    page, _ = GroundedAnswerHtmlRenderer().render(
        answer,
        grounder,
        _session(),
        [answer],
        verified=verified,
    )

    assert page.count("<ul>") == 1
    assert page.count("<li>") == len(items)
    assert page.count("</li>") == len(items)
    assert page.count("<strong>") == sum(item.startswith("**") for item in items)
    assert page.count('<mark class="grounded">') == len(items)
    assert '<mark class="grounded">- ' not in page


def test_html_grounding_preserves_ordered_and_nested_bullets(grounder):
    answer = (
        "1. **First slot** accepts full length expansion cards.\n"
        "   * **Child slot** supports smaller expansion cards.\n"
        "2. **Second slot** accepts another full length expansion card."
    )

    page, _ = GroundedAnswerHtmlRenderer().render(
        answer,
        grounder,
        _session(),
        [answer],
    )

    assert page.count("<ol>") == 1
    assert page.count("<ul>") == 1
    assert page.count("<li>") == 3
    assert page.count("<strong>") == 3
    assert page.count('<mark class="grounded">') == 3


def test_html_bullet_highlights_keep_the_original_translation_lookup(grounder):
    answer = "- Igel leben in Gaerten und fressen Insekten."
    page, _ = GroundedAnswerHtmlRenderer().render(
        answer,
        grounder,
        _session(),
        [EVIDENCE],
        sentence_translations={answer: EVIDENCE},
    )

    assert (
        '<li><mark class="grounded">Igel leben in Gaerten und fressen Insekten.</mark></li>'
        in page
    )


@pytest.mark.parametrize("verified", [True, False])
def test_html_renderer_quotes_parenthesized_flowchart_labels(grounder, verified):
    diagram = (
        "```mermaid\nflowchart LR\n"
        "Cat[Mammal (confirmed)] -->|Cats.md| Cat\n"
        "Dog[Mammal (unknown)] -->|Dogs.png| Dog\n"
        "Lion[Mammal (confirmed)] -->|Lions.pptx| Lion\n"
        "Ape[Mammal (confirmed)] -->|Apes.docx| Ape\n"
        "Hedgehog[Mammal (confirmed)] -->|Hedgehogs.pdf| Hedgehog\n"
        "Elephant[Mammal (confirmed)] -->|Elephants.jpg| Elephant\n```"
    )
    page, _ = GroundedAnswerHtmlRenderer().render(
        diagram,
        grounder,
        _session(),
        [],
        verified=verified,
    )
    decoded_page = unescape(page)
    for animal in ("Cat", "Lion", "Ape", "Hedgehog", "Elephant"):
        assert f'{animal}["Mammal (confirmed)"]' in decoded_page
    assert 'Dog["Mammal (unknown)"]' in decoded_page
    assert '<mark class="grounded">' not in page


@pytest.mark.parametrize(
    "source",
    [
        'flowchart LR\nA["Mammal (confirmed)"] --> B[Animal]',
        'flowchart LR\nA("Mammal (confirmed)") --> B((Animal))',
        'flowchart LR\nA[Animal] -->|"B[Mammal (unknown)]"| B',
        "flowchart LR\nA[Animal] %% B[Mammal (unknown)]",
        "sequenceDiagram\nAlice->>Bob: B[Mammal (unknown)]",
    ],
)
def test_html_renderer_preserves_other_mermaid_syntax(grounder, source):
    from markdown_it import MarkdownIt

    answer = f"```mermaid\n{source}\n```"
    expected_body = MarkdownIt().render(answer)
    page, _ = GroundedAnswerHtmlRenderer().render(
        answer,
        grounder,
        _session(),
        [],
        verified=False,
    )
    assert expected_body in page


def test_cli_builds_grounded_browser_page_with_security_headers(grounder):
    from Chat.Chatter import Chatter
    from Globals.Session import Session

    session = Session()
    session.answer_grounding_verified = True
    session.last_answer_content = (
        "Hedgehogs live in gardens and eat insects.\n\n" + DIAGRAM
    )
    session.chunk_texts_for_grounding = [EVIDENCE]
    chatter = object.__new__(Chatter)
    chatter.grounder = grounder

    document = Chatter._grounded_answer_browser_document(chatter, session)

    assert document is not None
    name, data, headers = document
    assert name == "GroundedAnswer.html"
    page = data.decode("utf-8")
    assert '<mark class="grounded">' in page
    assert 'class="language-mermaid"' in page
    assert "nonce-" in headers["Content-Security-Policy"]
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Frame-Options"] == "DENY"

    # An unverified answer with no evidence still has an unmarked preview.
    session.answer_grounding_verified = False
    session.browser_answer_content = session.last_answer_content
    session.last_answer_content = grounder.no_evidence_message()
    session.chunk_texts_for_grounding = []
    document = Chatter._grounded_answer_browser_document(chatter, session)
    assert document is not None
    name, data, _ = document
    page = data.decode("utf-8")
    assert name == "AnswerPreview.html"
    assert "Unverified answer preview" in page
    assert 'class="language-mermaid"' in page
    assert "Hedgehogs live in gardens" in page
    assert '<mark class="grounded">' not in page
    assert grounder.no_evidence_message() not in page


@pytest.mark.parametrize(
    "channel", ["cli", "cli_unverified", "service", "service_stream"]
)
def test_html_answer_pages_include_full_confidence_without_grounding_it(
    grounder, channel
):
    from Api.MarkedDocsService import register_grounded_answer
    from Api.MarkedDocsStore import MarkedDocsStore
    from Chat.Chatter import Chatter
    from Chat.ConfidenceEvaluator import AnswerConfidenceResult, ConfidenceEvaluator
    from Globals.Session import Session

    confidence = AnswerConfidenceResult(
        level="LOW",
        score=0.0,
        components={},
        summary=ConfidenceEvaluator().build_confidence_summary(
            level="LOW",
            score=0.0,
            c_top=0.33,
            c_mean_top3=0.19,
            c_coverage=0.15,
            c_local=0.05,
            c_fallback_penalty=0.0,
            top_score=0.66,
            mean_top3=0.64,
            coverage=1.0,
            local_share=1.0,
            evidence_chunks=50,
            fallback_triggered=False,
            note="Retrieval metrics do not verify answer support. <script>alert(1)</script>",
            quality_dimensions={
                "evidence_relevance": {"heuristic": 0.69},
                "internal_consistency": {"llm": 0.50},
            },
            judge_model="openai/qwen2_5_14b_instruct",
        ),
    )
    session = Session()
    session.answer_grounding_verified = channel != "cli_unverified"
    session.answer_confidence_result = confidence
    session.chunk_texts_for_grounding = [EVIDENCE, confidence.summary]
    answer = EVIDENCE + "\n\n" + DIAGRAM
    session.last_answer_content = answer + Chatter._build_answer_confidence_notice(
        confidence,
        monospace=channel == "service_stream",
    )
    if channel.startswith("cli"):
        session.browser_answer_content = answer
        chatter = object.__new__(Chatter)
        chatter.grounder = grounder
        document = chatter._grounded_answer_browser_document(session)
        assert document is not None
        page = document[1].decode("utf-8")
    else:
        store = MarkedDocsStore()
        link = register_grounded_answer(
            session,
            grounder,
            session.last_answer_content,
            store,
            "https://host",
        )
        token = link.split("/marked/", 1)[1].split(".html", 1)[0]
        entry = store.get(token)
        assert entry is not None
        page = entry.data.decode("utf-8")

    assert '<mark class="grounded">' in page
    assert 'class="language-mermaid"' in page
    assert page.count("Answer confidence:") == 1
    _, confidence_html = page.split('<section class="answer-confidence">', 1)
    assert confidence.summary.splitlines()[0] in confidence_html
    for metric in (
        "C_final=0.00",
        "C_top=0.33",
        "C_mean_top3=0.19",
        "C_coverage=0.15",
        "C_local=0.05",
        "C_fallback_penalty=0.00",
        "top_score=0.66",
        "mean_top3=0.64",
        "coverage=1.00",
        "local_share=1.00",
        "evidence_chunks=50",
        "rerank_fallback=off",
        "Heuristic Confidence:",
        "Static retrieval diagnostics (not answer-quality verification):",
        "LLM review - answer text only (judge model: openai/qwen2_5_14b_instruct):",
        "citation support: not assessed.",
    ):
        assert metric in confidence_html
    assert '<mark class="grounded">' not in confidence_html
    assert "<script>alert(1)</script>" not in confidence_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in confidence_html


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (EVIDENCE + " Unicorns manufacture purple spaceships.", False),
        ("Unicorns manufacture purple spaceships. " + EVIDENCE, False),
        (EVIDENCE + " Cats purr.", True),
        (EVIDENCE + " Unicorns fly.", False),
        ("- Hedgehogs\n- Unicorns", True),
        ("1. Unicorns\n2. Hedgehogs", True),
        ("- Unicorns\n  - Hedgehogs", True),
        ("- Unicorns\n- Spaceships", False),
        ("Unicorns fly.\n\n- Hedgehogs", False),
        (EVIDENCE + "\n\n- Hedgehogs\n- Unicorns", True),
        (EVIDENCE + "\n\n- Unicorns", False),
        ("### Answer\n- Hedgehogs\n\n### Sources\n- Unicorns", True),
        ("- Unicorns\n\n### Sources\n- Hedgehogs", False),
        ("```text\n- Hedgehogs\n- Unicorns\n```", False),
        (DIAGRAM + "\n\nUnicorns manufacture purple spaceships.", False),
        (DIAGRAM + "\n\n" + EVIDENCE, True),
        (DIAGRAM + "\n\n" + DIAGRAM.replace("Hedgehogs", "Unicorns"), True),
    ],
)
def test_guard_distinguishes_all_prose_from_partial_lists_and_diagrams(
    grounder,
    answer,
    expected,
):
    assert (
        grounder.has_grounded_evidence(
            _session(), answer, [EVIDENCE, "Cats purr."], "english", _EnglishDetector()
        )
        is expected
    )


@pytest.mark.parametrize("answer", ["Hedgehogs", "- Hedgehogs", DIAGRAM])
def test_short_answers_cannot_pass_without_evidence(grounder, answer):
    assert not grounder.is_answer_too_short(answer)
    assert not grounder.has_grounded_evidence(
        _session(), answer, [], "english", _EnglishDetector()
    )


def test_translation_cannot_hide_unsupported_prose(grounder, monkeypatch):
    def translate(_session, _body, _lang, _targets, is_grounded):
        return is_grounded(EVIDENCE + " Unicorns manufacture purple spaceships.")

    monkeypatch.setattr(grounder, "_check_answer_in_document_languages", translate)
    assert not grounder.has_grounded_evidence(
        _session(),
        "Igel leben in Gaerten. Einhoerner bauen Raumschiffe.",
        [EVIDENCE],
        "german",
        lambda *_args: "german",
    )


def test_preview_with_marking_disabled_escapes_html_without_grounding_marks(
    grounder, monkeypatch
):
    def unexpected_marks(*_args, **_kwargs):
        pytest.fail("Unverified preview requested grounding marks")

    monkeypatch.setattr(grounder, "mark_answer_html", unexpected_marks)
    page, csp = GroundedAnswerHtmlRenderer().render(
        "<script>alert('unsafe')</script>\n\n" + DIAGRAM,
        grounder,
        _session(),
        [EVIDENCE],
        verified=False,
        mark_supported=False,
    )
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    assert 'class="language-mermaid"' in page
    assert "unverified and may contain unsupported claims" in page
    assert "connect-src 'none'" in csp


def test_unverified_browser_preview_preserves_only_supported_cli_spans(grounder):
    from Chat.Chatter import Chatter
    from Globals.Session import Session

    supported = "Hedgehogs live in gardens and eat insects."
    unsupported = "Purple spacecraft manufacture impossible engines."
    session = Session()
    session.answer_grounding_verified = False
    session.browser_answer_content = f"- {supported}\n- {unsupported}"
    session.chunk_texts_for_grounding = [EVIDENCE]
    chatter = object.__new__(Chatter)
    chatter.grounder = grounder

    document = chatter._grounded_answer_browser_document(session)

    assert document is not None
    name, data, _ = document
    page = data.decode("utf-8")
    assert name == "AnswerPreview.html"
    assert "<title>Not fully grounded</title>" in page
    assert "<h1>Not fully grounded</h1>" in page
    assert "Not fully grounded:" in page
    assert f'<mark class="grounded">{supported}</mark>' in page
    assert f"<li>{unsupported}</li>" in page
    assert page.count('<mark class="grounded">') == 1
    assert not session.answer_grounding_verified
    cli = grounder.mark_answer_cli(
        session,
        session.browser_answer_content,
        session.chunk_texts_for_grounding,
    )
    assert f"- \033[48;5;214m{supported}\033[0m" in cli
    assert f"\033[48;5;214m- {unsupported}" not in cli

    document = chatter._grounded_answer_browser_document(session, mark_supported=False)
    assert document is not None
    assert '<mark class="grounded">' not in document[1].decode("utf-8")


def test_browser_diagram_marks_supported_cats_but_not_unsupported_lions(grounder):
    diagram = (
        "```mermaid\nflowchart LR\n"
        "Cats -->|Mammals| Yes\nLions -->|Mammals| Yes\n"
        "classDef modelColour fill:orange;\nclass Lions modelColour;\n```"
    )
    page, _ = GroundedAnswerHtmlRenderer().render(
        diagram,
        grounder,
        _session(),
        ["Cats are mammals that nurse their young."],
    )
    assert "class Cats,Yes groundedEvidence;" in page
    assert "class Cats,Lions" not in page
    assert "class Lions" not in page
    assert "modelColour" not in page
    assert "linkStyle 0 stroke:#ef6c00" in page
    assert "linkStyle 0,1" not in page
    assert 'primaryColor: "#f5f5f5"' in page


@pytest.mark.parametrize("mark_supported", [False, True])
def test_browser_table_keeps_evidence_in_the_third_cell(grounder, mark_supported):
    rows = [
        ("Cats", "Yes", "cats have lived alongside humans for thousands of years"),
        (
            "Dogs",
            "Yes",
            "tasks. dogs need training, exercise, and responsible ownership",
        ),
        ("Hedgehogs", "Yes", "Hedgehogs are small, nocturnal mammals"),
        ("Fish", "No", "fish are central to aquatic food webs"),
    ]
    answer = "| Animal | Mammal Status | Evidence Phrase |\n| --- | --- | --- |\n"
    answer += "\n".join(
        f'| {name} | {status} | "{quote}" |' for name, status, quote in rows
    )
    evidence = [
        rows[0][2],
        "dogs need training, exercise, and responsible ownership",
        rows[2][2],
    ]

    page, _ = GroundedAnswerHtmlRenderer().render(
        answer,
        grounder,
        _session(),
        evidence,
        verified=False,
        mark_supported=mark_supported,
    )

    body = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    cells_by_row = [
        re.findall(r"<td>(.*?)</td>", row)
        for row in re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)
    ]
    assert [
        [unescape(re.sub(r"</?mark\b[^>]*>", "", cell)) for cell in cells]
        for cells in cells_by_row
    ] == [[name, status, f'"{quote}"'] for name, status, quote in rows]
    if mark_supported:
        assert '<mark class="grounded">' in cells_by_row[0][2]
        assert '"tasks. <mark class="grounded">' in unescape(cells_by_row[1][2])
        assert '<mark class="grounded">' in cells_by_row[2][2]
        assert '<mark class="grounded">' not in cells_by_row[3][2]
        assert "<title>Not fully grounded</title>" in page
    else:
        assert '<mark class="grounded">' not in body


def test_mermaid_grounding_styles_keep_edge_indices_and_explicit_node_ids(grounder):
    diagram = (
        "```mermaid\nflowchart LR\n"
        "A[Lions] -->|Mammals| Y[Yes]\n"
        "B[Cats] -->|Mammals| Y\n```"
    )
    marked = grounder.mark_answer_html(
        _session(),
        diagram,
        ["Cats are mammals that nurse their young."],
    )
    assert "class B,Y groundedEvidence;" in marked
    assert "linkStyle 1 stroke:#ef6c00" in marked


@pytest.mark.parametrize("color_system", ["standard", "256"])
def test_rich_renderer_keeps_grounding_marks_on_mermaid_source(grounder, color_system):
    marked = grounder.mark_answer_cli(_session(), DIAGRAM, [EVIDENCE])
    output = io.StringIO()
    console = Console(
        file=output, width=100, force_terminal=True, color_system=color_system
    )
    renderer = RichAnswerRenderer(console)

    renderer.render(marked, 100)

    decoded = Text.from_ansi(output.getvalue())
    claim = "A[Hedgehogs] -->|live in| B[Gardens]"
    offset = decoded.plain.index(claim)
    for index in range(offset, offset + len(claim)):
        assert decoded.get_style_at_offset(console, index).bgcolor is not None
