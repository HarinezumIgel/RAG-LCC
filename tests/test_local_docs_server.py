from __future__ import annotations

import ast
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener
from urllib.parse import parse_qs, urlsplit, urlunsplit

import pytest

from Chat.LocalDocsServer import LocalDocsServer
from Chat.MarkedDocsViewer import open_grounded_answer_html, open_marked_documents


class _PrettyRecorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, *args: Any, **kwargs: Any) -> None:
        self.lines.append(" ".join(str(arg) for arg in args))


@pytest.fixture
def server() -> Iterator[LocalDocsServer]:
    listener = LocalDocsServer(port=0)
    try:
        yield listener
    finally:
        listener.close()


def _get(url: str) -> Any:
    return build_opener(ProxyHandler({})).open(url, timeout=2)


@pytest.mark.parametrize(
    ("filename", "label", "description"),
    [
        ("GroundedAnswer.html", "Grounded answer", "Grounded browser page"),
        ("AnswerPreview.html", "Answer preview", "Formatted browser page"),
    ],
)
def test_grounded_page_link_serves_html_without_writing_or_opening(
    server: LocalDocsServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
    label: str,
    description: str,
) -> None:
    import Chat.MarkedDocsViewer as viewer

    def unexpected_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Page opened before the user clicked")

    monkeypatch.setattr(viewer, "_open_with_os", unexpected_open)
    pretty = _PrettyRecorder()
    page = b"<!doctype html><h1>Grounded answer</h1>"
    url = open_grounded_answer_html(
        filename,
        page,
        pretty,
        local_server=server,
        response_headers={"Content-Security-Policy": "default-src 'none'"},
    )

    assert isinstance(url, str)
    assert url.startswith("http://127.0.0.1:")
    assert url in capsys.readouterr().out
    assert any("expires after 1800 s or at shutdown" in line for line in pretty.lines)
    assert any(label in line and description in line for line in pretty.lines)
    assert not list(tmp_path.iterdir())
    with _get(url) as response:
        assert response.status == 200
        assert response.read() == page
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"
        assert (
            response.headers["Content-Disposition"] == f'inline; filename="{filename}"'
        )
        assert response.headers["Content-Security-Policy"] == "default-src 'none'"
        assert response.headers["Cache-Control"] == "private, no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_chatter_offers_a_working_http_link(
    server: LocalDocsServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Chat.Chatter import Chatter
    from Gui.PrettyWriter import PrettyWriter

    chatter = object.__new__(Chatter)
    chatter.pretty = PrettyWriter(always_on=True)
    chatter.local_docs_server = server
    page = b"<!doctype html><h1>Grounded answer</h1>"
    chatter._open_grounded_answer_page(
        ("GroundedAnswer.html", page, {"Referrer-Policy": "no-referrer"})
    )
    output = capsys.readouterr().out
    assert "file://" not in output
    url = re.search(r"http://127\.0\.0\.1:\d+/\S+", output)
    assert url is not None
    with _get(url.group()) as response:
        assert response.read() == page
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"
        assert response.headers["Referrer-Policy"] == "no-referrer"


def test_only_html_uses_http_while_other_formats_keep_file_links(
    server: LocalDocsServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import Chat.MarkedDocsViewer as viewer

    monkeypatch.setattr(viewer, "supports_osc8", lambda: True)
    monkeypatch.setattr(
        "builtins.input", lambda *args: pytest.fail("Unexpected document picker")
    )
    open_marked_documents(
        [
            ("source.v2 with spaces.html", b"<mark>Source</mark>"),
            ("source.pdf", b"%PDF-1.4"),
            ("source.\u65e5\u672c\u8a9e.html", b"<h1>Unicode filename</h1>"),
            ("source.HTM", b"<p>HTML</p>"),
            ("notes.txt", b"Notes"),
            ("notes.md", b"# Notes"),
            ("document.docx", b"Word document"),
        ],
        _PrettyRecorder(),
        project_root=str(tmp_path),
        _register_cleanup=False,
        local_server=server,
    )
    output = capsys.readouterr().out
    urls = re.findall(r"http://127\.0\.0\.1:\d+/\S+", output)
    assert len(urls) == 3
    assert output.count("file://") == 4
    assert len(server.store) == 3
    files = {
        path.name: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert files == {
        "source_marked.pdf": b"%PDF-1.4",
        "notes_marked.txt.md": b"Notes",
        "notes_marked.md": b"# Notes",
        "document_marked.docx": b"Word document",
    }
    for url, expected in zip(
        urls,
        [b"<mark>Source</mark>", b"<h1>Unicode filename</h1>", b"<p>HTML</p>"],
    ):
        with _get(url) as response:
            assert response.read() == expected
            if expected == b"<h1>Unicode filename</h1>":
                assert "filename*=UTF-8''" in response.headers["Content-Disposition"]


def test_non_html_picker_is_preserved_with_listener_enabled(
    server: LocalDocsServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import Chat.MarkedDocsViewer as viewer

    opened: list[Path] = []
    responses = iter(["1", ""])
    monkeypatch.setattr(viewer, "supports_osc8", lambda: False)
    monkeypatch.setattr(viewer, "_open_with_os", opened.append)
    monkeypatch.setattr("builtins.input", lambda *args: next(responses))
    open_marked_documents(
        [("source.html", b"<h1>Source</h1>"), ("notes.md", b"# Notes")],
        _PrettyRecorder(),
        project_root=str(tmp_path),
        _register_cleanup=False,
        local_server=server,
    )
    assert len(opened) == 1
    assert opened[0].name == "notes_marked.md"
    assert opened[0].read_bytes() == b"# Notes"
    assert len(server.store) == 1


def test_expiry_releases_bytes_without_waiting_for_a_request() -> None:
    listener = LocalDocsServer(port=0, ttl_seconds=1)
    try:
        url = listener.put(b"temporary", "answer.html")
        deadline = time.monotonic() + 4
        while listener.store.total_bytes and time.monotonic() < deadline:
            time.sleep(0.05)
        assert listener.store.total_bytes == 0
        assert len(listener.store) == 0
        with pytest.raises(HTTPError) as error:
            _get(url)
        assert error.value.code == 404
    finally:
        listener.close()


def test_shutdown_clears_documents_and_closes_the_socket(
    server: LocalDocsServer,
) -> None:
    url = server.put(b"temporary", "answer.html")
    server.close()
    server.close()
    assert server.store.total_bytes == 0
    assert len(server.store) == 0
    with pytest.raises(URLError):
        _get(url)
    with pytest.raises(RuntimeError, match="closed"):
        server.put(b"late", "answer.html")


@pytest.mark.parametrize("path", ["/", "/marked/unknown.html", "/../README.md"])
def test_unknown_links_do_not_expose_files(server: LocalDocsServer, path: str) -> None:
    with pytest.raises(HTTPError) as error:
        _get(server.base_url + path)
    assert error.value.code == 404


@pytest.mark.parametrize("capability", ["", "?token=wrong", "?token=%E6%97%A5"])
def test_missing_or_invalid_session_capability_does_not_consume_document(
    server: LocalDocsServer,
    capability: str,
) -> None:
    url = server.put(b"private", "answer.html")
    path = url.split("?", 1)[0]
    with pytest.raises(HTTPError) as error:
        _get(path + capability)
    assert error.value.code == 404
    with _get(url) as response:
        assert response.read() == b"private"
        assert response.headers["Referrer-Policy"] == "no-referrer"


def test_session_capabilities_are_not_interchangeable(server: LocalDocsServer) -> None:
    other = LocalDocsServer(port=0)
    try:
        url = server.put(b"private", "answer.html")
        other_url = other.put(b"other", "answer.html")
        parts = urlsplit(url)
        assert (
            parse_qs(parts.query)["token"]
            != parse_qs(urlsplit(other_url).query)["token"]
        )
        forged = urlunsplit(parts._replace(query=urlsplit(other_url).query))
        with pytest.raises(HTTPError) as error:
            _get(forged)
        assert error.value.code == 404
        with _get(url) as response:
            assert response.read() == b"private"
    finally:
        other.close()


@pytest.mark.parametrize("extra", ["&token=", "&token=wrong"])
def test_duplicate_session_capabilities_are_rejected(
    server: LocalDocsServer,
    extra: str,
) -> None:
    url = server.put(b"private", "answer.html")
    with pytest.raises(HTTPError) as error:
        _get(url + extra)
    assert error.value.code == 404
    with _get(url) as response:
        assert response.read() == b"private"


def test_publication_failure_is_reported_without_file_fallback(
    tmp_path: Path,
) -> None:
    listener = LocalDocsServer(port=0, max_total_bytes=1)
    pretty = _PrettyRecorder()
    try:
        result = open_grounded_answer_html(
            "answer.html",
            b"too large",
            pretty,
            local_server=listener,
        )
        assert result is None
        assert any("exceeds cache capacity" in line for line in pretty.lines)
        assert not list(tmp_path.iterdir())
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("enabled", "fail_query"), [(True, False), (True, True), (False, False)]
)
def test_ragchat_run_wires_and_closes_the_listener(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    fail_query: bool,
    server: LocalDocsServer,
) -> None:
    # Load the entry-point method without running module-level model startup.
    path = Path(__file__).parents[1] / "src" / "Apps" / "RAGChat.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RAGChat"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    namespace: dict[str, Any] = {
        "os": os,
        "LocalDocsServer": LocalDocsServer,
        "NetworkTracer": None,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    run: Callable[[Any], None] = namespace["_run"]
    monkeypatch.setenv("SERVE_RAGCHAT_GROUNDED_DOCS", str(int(enabled)))
    monkeypatch.setenv("RAG_LCC_NW_TRACE", "0")
    listeners: list[LocalDocsServer] = []
    chatter = SimpleNamespace(local_docs_server=None)
    port = urlsplit(server.base_url).port
    assert port is not None
    server.close()

    def query() -> None:
        listener = chatter.local_docs_server
        if not enabled:
            assert listener is None
            return
        assert isinstance(listener, LocalDocsServer)
        assert listener.base_url == f"http://127.0.0.1:{port}"
        listeners.append(listener)
        with _get(listener.put(b"answer", "answer.html")) as response:
            assert response.read() == b"answer"
        if fail_query:
            raise RuntimeError("query failed")

    shell = SimpleNamespace(
        cfg=SimpleNamespace(get_dict=lambda *args: {"ttl_seconds": 10, "port": port}),
        chatter=chatter,
        _showIntro=lambda: None,
        _run_query=query,
        _show_results=lambda: None,
    )
    if fail_query:
        with pytest.raises(RuntimeError, match="query failed"):
            run(shell)
    else:
        run(shell)
    assert chatter.local_docs_server is None
    if enabled:
        assert listeners[0].store.ttl_seconds == 10
        assert listeners[0].store.total_bytes == 0
        with pytest.raises(URLError):
            _get(listeners[0].base_url)
    else:
        assert listeners == []


def test_occupied_document_port_fails_without_switching_ports(
    server: LocalDocsServer,
) -> None:
    port = urlsplit(server.base_url).port
    assert port is not None
    with pytest.raises(OSError):
        LocalDocsServer(port=port)
    with _get(server.put(b"still available", "answer.html")) as response:
        assert response.read() == b"still available"
