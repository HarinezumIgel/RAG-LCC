"""Summary source hints must locate verified passages, never establish support."""

import io
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from Grounding.Grounder import Grounder, tokenize
from Config.Config import Config
from Grounding.SourceReferences import SummaryStatement, parse_summary, summary_payload
from VisualMarkers import ChunkSnippet
from VisualMarkers.PdfVisualMarker import PdfVisualMarker, _match_word_sequence


@pytest.fixture
def grounder():
    Grounder._reset()
    instance = Grounder()
    yield instance
    Grounder._reset()


SOURCE = "Storage for program variables is reserved by memory allocation."
ANSWER = "Memory allocation reserves storage for program variables."
STATEMENTS = [SummaryStatement(ANSWER, (7,))]


def test_pcie_lane_spacing_preserves_width():
    assert tokenize("PCIe 4.0 x16") == tokenize("PCIe 4.0 x 16")
    assert tokenize("PCIe 4.0 x16") != tokenize("PCIe 4.0 x8")


def test_pcie_row_with_spaced_lane_width_becomes_orange_source_snippet(
    grounder, tmp_path
):
    source = tmp_path / "source.pdf"
    source.touch()
    row = "Row 12: 45 | PCIe 4.0 x 16 card slot 1"
    answer = "- PCIe 4.0 x16 card slot 1 (located at slot 45)"
    doc = Document(
        page_content=row, metadata={"FilePath": str(source), "PageNumber": 12}
    )
    session = SimpleNamespace(
        last_answer_content=answer,
        chunk_texts_for_grounding=[row],
        grounding_proxy_text="",
    )

    assert grounder.find_grounded_sentences(answer, [row]) == [answer]
    snippets = grounder.build_grounded_snippets(session, [doc], "orange")
    assert [(s.text, s.page_number, s.color) for s in snippets[str(source)]] == [
        (row, 12, "orange"),
    ]


def test_relaxed_match_returns_verbatim_passage(grounder):
    assert not grounder.find_grounded_sentences(ANSWER, [SOURCE])
    assert grounder.summary_support(ANSWER, [SOURCE], [7]) == {ANSWER: [(0, SOURCE)]}


def test_verified_citation_narrows_candidates(grounder):
    assert grounder.summary_support(ANSWER, [SOURCE, SOURCE], [6, 7], STATEMENTS) == {
        ANSWER: [(1, SOURCE)],
    }


def test_verified_citation_avoids_indexing_unrelated_chunks(grounder, monkeypatch):
    get_index = Mock(wraps=grounder._index_cache.get)
    monkeypatch.setattr(grounder._index_cache, "get", get_index)
    grounder.summary_support(
        ANSWER, ["Unrelated compiler documentation.", SOURCE], [6, 7], STATEMENTS
    )
    assert get_index.call_count == 1
    assert get_index.call_args.args[0] == SOURCE


@pytest.mark.parametrize("citation", [(7,), (999,), ()])
def test_bad_or_missing_citation_falls_back_to_verified_sources(grounder, citation):
    support = grounder.summary_support(
        ANSWER,
        ["Unrelated compiler documentation.", SOURCE],
        [7, 8],
        [SummaryStatement(ANSWER, citation)],
    )
    assert support == {ANSWER: [(1, SOURCE)]}


def test_real_chunk_id_is_not_evidence(grounder):
    assert not grounder.summary_support(
        ANSWER, ["Unrelated compiler documentation."], [7], STATEMENTS
    )


@pytest.mark.parametrize(
    "answer, source",
    [
        ("Memory allocation does not reserve storage for program variables.", SOURCE),
        (
            "Compiler version 4.8.2 optimizes program execution.",
            "Compiler version 4.4.1 optimizes program execution.",
        ),
        (
            "This chapter describes examples and provides information.",
            "This chapter provides examples and describes information.",
        ),
        (
            "The encryption algorithm was compiled in GCC 4.4.1.",
            "The graphics interface was compiled in GCC 4.4.1.",
        ),
    ],
)
def test_misleading_overlap_is_not_marked(grounder, answer, source):
    assert not grounder.summary_support(answer, [source], [7])


def test_match_is_bounded_not_scattered_across_chunk(grounder):
    source = "memory allocation " + "unrelated " * 60 + "storage program variables"
    assert not grounder.summary_support(ANSWER, [source], [7])


def test_reference_does_not_create_token_overlap():
    assert parse_summary(summary_payload(STATEMENTS)) == STATEMENTS
    assert tokenize(STATEMENTS[0].text) == tokenize(ANSWER)


def test_summary_policy_does_not_leak_into_next_question():
    from Globals.Session import Session

    session = object.__new__(Session)
    session.summary_grounding = True
    session.summary_statements = STATEMENTS
    session.reset_turn_state()
    assert session.summary_grounding is False
    assert session.summary_statements == []


def test_translated_statement_keeps_integer_context_hints(grounder):
    original = "Die Speicherverwaltung reserviert Platz."
    session = SimpleNamespace(
        summary_statements=[SummaryStatement(original, (7,))],
    )
    mapped = grounder._summary_statements(session, {original: ANSWER})
    assert mapped == STATEMENTS
    assert grounder.summary_support(ANSWER, [SOURCE, SOURCE], [6, 7], mapped) == {
        ANSWER: [(1, SOURCE)],
    }


def test_duplicate_statement_merges_candidate_ids(grounder):
    statements = [SummaryStatement(ANSWER, (6,)), *STATEMENTS]
    assert grounder.summary_support(ANSWER, [SOURCE, SOURCE], [6, 7], statements) == {
        ANSWER: [(0, SOURCE), (1, SOURCE)],
    }


def test_only_verified_clause_is_highlighted(grounder, monkeypatch):
    original = grounder.cfg.get_str
    monkeypatch.setattr(
        grounder.cfg,
        "get_str",
        lambda key, *args, **kwargs: (
            "48;5;208"
            if key == "_MARKED_DOCS_COLORS.answer_ansi"
            else original(key, *args, **kwargs)
        ),
    )
    clause = "Memory allocation reserves storage for program variables"
    unsupported = "flying elephants manufacture purple spaceships on distant planets."
    answer = f"{clause}, {unsupported}"
    session = SimpleNamespace(
        summary_grounding=True,
        last_chosen_chunks=[],
        debug_level=0,
        debug_mode="none",
    )
    support = grounder.summary_support(answer, [SOURCE], [1])
    assert clause in support
    assert answer not in support
    marked = grounder.mark_answer_cli(session, answer, [SOURCE])
    assert f"\033[48;5;208m{clause}\033[0m, {unsupported}" == marked


def test_cli_masking_happens_after_grounding(grounder, monkeypatch):
    original = grounder.cfg.get_str
    monkeypatch.setattr(
        grounder.cfg,
        "get_str",
        lambda key, *args, **kwargs: (
            "48;5;208"
            if key == "_MARKED_DOCS_COLORS.answer_ansi"
            else original(key, *args, **kwargs)
        ),
    )
    session = SimpleNamespace(
        summary_grounding=False,
        debug_level=0,
        debug_mode="none",
    )
    answer = "Memory allocation reserves storage for program variables."
    masked = grounder.mark_answer_cli(
        session,
        answer,
        [answer],
        text_transform=lambda text: text.replace("Memory allocation", "[MASKED]"),
    )

    assert (
        masked
        == "\033[48;5;208m[MASKED] reserves storage for program variables.\033[0m"
    )


def test_structured_references_stay_attached_to_their_statement(grounder):
    support = grounder.summary_support(
        ANSWER,
        [SOURCE, SOURCE],
        [6, 7],
        [SummaryStatement("Unrelated statement.", (6,)), *STATEMENTS],
    )
    assert support == {ANSWER: [(1, SOURCE)]}


def test_summary_cli_and_source_marking_use_same_passage(
    grounder, tmp_path, monkeypatch
):
    original = grounder.cfg.get_str
    monkeypatch.setattr(
        grounder.cfg,
        "get_str",
        lambda key, *args, **kwargs: (
            "48;5;208"
            if key == "_MARKED_DOCS_COLORS.answer_ansi"
            else original(key, *args, **kwargs)
        ),
    )
    source = tmp_path / "source.pdf"
    source.touch()
    doc = Document(
        page_content=SOURCE,
        metadata={
            "FilePath": str(source),
            "PageNumber": 3,
            "SummaryChunkId": 7,
        },
    )
    session = SimpleNamespace(
        summary_grounding=True,
        last_answer_content=ANSWER,
        chunk_texts_for_grounding=[SOURCE],
        last_chosen_chunks=[doc],
        grounding_proxy_text="",
        debug_level=0,
        debug_mode="none",
        summary_statements=STATEMENTS,
    )
    assert grounder.has_local_spans(session)
    marked = grounder.mark_answer_cli(session, ANSWER, [SOURCE])
    assert "\033[" in marked and ANSWER in marked
    snippets = grounder.build_grounded_snippets(session, [doc], "orange")
    assert [(s.text, s.page_number, s.color) for s in snippets[str(source)]] == [
        (SOURCE, 3, "orange"),
    ]


def test_repeated_passage_on_distinct_pages_is_not_dropped(grounder, tmp_path):
    source = tmp_path / "source.pdf"
    source.touch()
    docs = [
        Document(
            page_content=SOURCE, metadata={"FilePath": str(source), "PageNumber": p}
        )
        for p in (1, 2)
    ]
    session = SimpleNamespace(
        summary_grounding=True,
        last_answer_content=ANSWER,
        chunk_texts_for_grounding=[SOURCE, SOURCE],
        grounding_proxy_text="",
    )
    snippets = grounder.build_grounded_snippets(session, docs, "orange")
    assert [s.page_number for s in snippets[str(source)]] == [1, 2]


def test_fragment_attribution_preserves_aggregate_order_and_deduplication(grounder):
    repeated = SOURCE.upper()
    chunk = f"{SOURCE}\n{repeated}\nA separate passage describes compiler optimization."
    answers = [SOURCE, "Unrelated galaxies contain distant stars.", SOURCE]

    fragments, by_sentence = grounder._grounding_fragments(answers, chunk)

    assert fragments == [SOURCE]
    assert by_sentence == [[SOURCE], [], [SOURCE]]
    assert grounder.find_grounding_fragments_in_chunk(answers, chunk) == fragments


@pytest.mark.parametrize("chunk", ["", "Unrelated galaxies contain distant stars."])
def test_fragment_fallback_keeps_answer_text_without_source_attribution(
    grounder, chunk
):
    fragments, by_sentence = grounder._grounding_fragments([ANSWER], chunk)

    assert fragments == [ANSWER]
    assert by_sentence == [[]]


def test_debug_source_marking_reuses_fragment_attribution(
    grounder, tmp_path, monkeypatch
):
    from Helpers.DebugHelper import DebugHelper

    source = tmp_path / "source.pdf"
    source.touch()
    doc = Document(
        page_content=SOURCE, metadata={"FilePath": str(source), "PageNumber": 3}
    )
    session = SimpleNamespace(
        last_answer_content=SOURCE,
        chunk_texts_for_grounding=[SOURCE],
        grounding_proxy_text="",
    )
    monkeypatch.setattr(DebugHelper, "check", lambda *_args: True)
    fragments = Mock(wraps=grounder._grounding_fragments)
    monkeypatch.setattr(grounder, "_grounding_fragments", fragments)
    writer = Mock()
    monkeypatch.setattr(grounder, "pretty", writer)

    snippets = grounder.build_grounded_snippets(session, [doc], "orange")

    assert fragments.call_count == 1
    assert snippets[str(source)][0].text == SOURCE
    assert any("answer[0]" in str(call) for call in writer.write.call_args_list)


def test_summary_support_keeps_ids_aligned_after_local_file_filtering(
    grounder, tmp_path
):
    source = tmp_path / "source.pdf"
    source.touch()
    docs = [
        Document(
            page_content="Unrelated compiler documentation.",
            metadata={
                "Source": "web",
                "FilePath": "https://example.test",
                "SummaryChunkId": 6,
            },
        ),
        Document(
            page_content=SOURCE,
            metadata={
                "FilePath": str(source),
                "PageNumber": 3,
                "SummaryChunkId": 7,
            },
        ),
    ]
    session = SimpleNamespace(
        summary_grounding=True,
        last_answer_content=ANSWER,
        chunk_texts_for_grounding=[doc.page_content for doc in docs],
        grounding_proxy_text="",
        summary_statements=STATEMENTS,
    )

    snippets = grounder.build_grounded_snippets(session, docs, "orange")

    assert list(snippets) == [str(source)]
    assert [
        (snippet.text, snippet.page_number) for snippet in snippets[str(source)]
    ] == [
        (SOURCE, 3),
    ]


def _word(text, x, top):
    return {"text": text, "x0": x, "x1": x + 20, "top": top, "bottom": top + 10}


def test_multiline_match_has_separate_line_rectangles():
    words = [
        _word("memory", 10, 10),
        _word("allocation", 35, 10),
        _word("reserves", 10, 40),
        _word("storage", 35, 40),
    ]
    assert _match_word_sequence(words, 100, "memory allocation reserves storage") == [
        (10, 80, 55, 90),
        (10, 50, 55, 60),
    ]


def test_separated_columns_do_not_get_one_wide_rectangle():
    words = [_word("memory", 10, 10), _word("storage", 200, 10)]
    assert _match_word_sequence(words, 100, "memory storage") == [
        (10, 80, 30, 90),
        (200, 80, 220, 90),
    ]


@pytest.fixture
def source_pdf(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            ),
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 20 250 Td (Storage for program variables) Tj "
        b"0 -20 Td (is reserved by memory allocation.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "source.pdf"
    writer.write(path)
    return path


def test_actual_pdf_marks_relaxed_source_and_reports_pages(
    grounder, source_pdf, caplog
):
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject

    doc = Document(
        page_content=SOURCE,
        metadata={
            "FilePath": str(source_pdf),
            "PageNumber": 1,
            "SummaryChunkId": 7,
        },
    )
    session = SimpleNamespace(
        summary_grounding=True,
        last_answer_content=ANSWER,
        grounding_proxy_text="",
        chunk_texts_for_grounding=[SOURCE],
    )
    snippets = grounder.build_grounded_snippets(session, [doc], "orange")[
        str(source_pdf)
    ]
    with caplog.at_level(logging.INFO):
        output = PdfVisualMarker().mark_to_bytes(source_pdf, snippets)
    reader = PdfReader(io.BytesIO(output))
    refs = reader.pages[0]["/Annots"]
    assert isinstance(refs, ArrayObject)
    annotations = [ref.get_object() for ref in refs]
    assert len(annotations) == 2
    assert all(a["/Subtype"] == "/Highlight" for a in annotations)
    assert all(float(a["/Rect"][3]) - float(a["/Rect"][1]) < 15 for a in annotations)
    assert "2 new highlights on physical pages 1" in caplog.text
    assert reader.metadata and reader.metadata["/RAGHighlightCount"] == "2"
    assert reader.metadata["/RAGHighlightPages"] == "1"


def test_actual_pdf_highlights_render(source_pdf):
    pdfium = pytest.importorskip("pypdfium2")
    from PIL import ImageChops

    output = PdfVisualMarker().mark_to_bytes(
        source_pdf,
        [ChunkSnippet(text=SOURCE, page_number=1, color="orange")],
    )
    with pdfium.PdfDocument(output) as doc:
        page = doc[0]
        marked = page.render(draw_annots=True).to_pil()
        plain = page.render(draw_annots=False).to_pil()
        assert ImageChops.difference(marked, plain).getbbox() is not None


def test_zero_new_highlights_warns_even_when_pdf_has_existing_annotations(
    source_pdf, caplog
):
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject

    marker = PdfVisualMarker()
    marked = marker.mark_to_bytes(
        source_pdf, [ChunkSnippet(text=SOURCE, page_number=1)]
    )
    source_pdf.write_bytes(marked)
    with caplog.at_level(logging.INFO):
        output = marker.mark_to_bytes(
            source_pdf,
            [ChunkSnippet(text="Completely absent phrase", page_number=1)],
        )
    assert "zero new highlights; 1 unmatched snippets" in caplog.text
    annotations = PdfReader(io.BytesIO(output)).pages[0]["/Annots"]
    assert isinstance(annotations, ArrayObject)
    assert len(annotations) == 2


def test_worker_preserves_highlight_count_and_physical_pages(source_pdf):
    from pypdf import PdfReader
    from VisualMarkers.MarkWorker import mark_in_subprocess

    output = mark_in_subprocess(
        str(source_pdf),
        [ChunkSnippet(text=SOURCE, page_number=1)],
        "orange",
        30,
    )
    metadata = PdfReader(io.BytesIO(output)).metadata
    assert metadata and metadata["/RAGHighlightCount"] == "2"
    assert metadata["/RAGHighlightPages"] == "1"


@pytest.mark.parametrize(
    "found, parallel", [(True, False), (False, False), (True, True)]
)
def test_cli_reports_new_counts_and_pages(source_pdf, found, parallel):
    from Chat.RAGChatImpl import RAGChatImpl

    snippet = ChunkSnippet(
        text=SOURCE if found else "Completely absent phrase", page_number=1
    )
    stub = Mock(spec=RAGChatImpl)
    stub.cfg = Config()
    stub.pretty = Mock()
    stub._mark_pdfs_in_parallel.return_value = {}
    if parallel:
        stub._mark_pdfs_in_parallel.return_value = {
            str(source_pdf): PdfVisualMarker().mark_to_bytes(source_pdf, [snippet]),
        }
    output = RAGChatImpl._produce_marked_bytes(
        stub,
        {str(source_pdf): [snippet]},
        {},
        "orange",
    )
    assert len(output) == 1
    expected = (
        "2 new highlights on physical pages 1." if found else "zero new highlights."
    )
    assert any(expected in str(call) for call in stub.pretty.write.call_args_list)


def test_summary_without_verified_passages_warns(grounder, source_pdf, monkeypatch):
    from Chat.RAGChatImpl import RAGChatImpl
    from Globals.Session import Session

    monkeypatch.setattr(grounder, "build_grounded_snippets", lambda *args: {})
    stub = Mock(spec=RAGChatImpl)
    stub.cfg = Config()
    stub.pretty = Mock()
    stub._group_chunks_by_file.return_value = {str(source_pdf): []}
    stub._resolve_mark_colors.return_value = ("yellow", "orange")
    stub._produce_marked_bytes.return_value = []
    session = Mock(spec=Session)
    session.debug_level = 0
    session.debug_mode = "none"
    RAGChatImpl._mark_sources(stub, session, [], summary=True)
    assert any(
        "No verified source passages for this summary; zero new highlights."
        in str(call)
        for call in stub.pretty.write.call_args_list
    )
