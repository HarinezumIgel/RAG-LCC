import ast
import json
import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.responses import Response

from Api.ChatCompletionHandler import ChatCompletionRequest, _complianceResponse
from Api.MarkedDocsService import marked_document_url, register_marked_documents
from Api.MarkedDocsStore import (
    MarkedDocsStore,
    configure_default,
    get_default,
    reset_default,
)
from DocumentServing.MarkedDocsServer import MarkedDocsServer
from Globals.Session import Session


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_settings", [False, True])
def test_invalid_api_request_returns_settings_and_logs_reference(
    stream: bool,
    with_settings: bool,
) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "Apps" / "RAGChatService.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_validationErrorHandler"
    )
    handler.decorator_list = []
    writer = Mock()
    settings = Mock()
    settings.format_session_snapshot.return_value = "\n\nSETTINGS_BLOCK"
    service = SimpleNamespace(
        serviceCfgDebugLevel=0,
        queryParts=settings if with_settings else None,
    )
    namespace = {
        "self": service,
        "Request": Request,
        "RequestValidationError": RequestValidationError,
        "Response": Response,
        "json": json,
        "uuid": uuid,
        "RED": "",
        "PrettyWriter": lambda: writer,
        "_complianceResponse": _complianceResponse,
    }
    exec(
        compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    app = FastAPI()
    app.add_exception_handler(
        RequestValidationError, namespace["_validationErrorHandler"]
    )

    @app.post("/v1/chat/completions")
    async def endpoint(request: ChatCompletionRequest) -> dict[str, str]:
        raise AssertionError("Invalid request reached endpoint")

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "documents",
                "messages": [],
                "temperature": "not-a-number",
                "stream": stream,
            },
        )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(
            event["choices"][0]["delta"].get("content", "") for event in events
        )
        assert "data: [DONE]" in response.text
    else:
        content = response.json()["choices"][0]["message"]["content"]
    assert "Invalid request parameter" in content
    assert "temperature" in content
    assert "Reference: ragchat-" in content
    assert ("SETTINGS_BLOCK" in content) is with_settings
    warning = writer.write.call_args_list[0].args
    assert warning[0] == "W"
    assert warning[1] == "RAGChatService invalid API request:"
    assert warning[2].split("Reference: ", 1)[1] in content
    if with_settings:
        settings.print_values.assert_called_once()
        settings.format_session_snapshot.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_service_registers_capability_protected_document_routes(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "Apps" / "RAGChatService.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_register_marked_docs"
    )
    writer = Mock()
    namespace = {
        "os": os,
        "MarkedDocsServer": MarkedDocsServer,
        "configure_marked_docs_store": configure_default,
        "PrettyWriter": lambda: writer,
        "RED": "",
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    app = FastAPI()
    cfg = Mock()
    cfg.get_dict.return_value = {
        "cors_origins": ["https://ui.example.test"],
        "single_use": True,
    }
    cfg.get_str.return_value = "127.0.0.1"
    cfg.get_int.return_value = 11435
    monkeypatch.setenv("SERVE_RAGCHATSERVICE_GROUNDED_DOCS", str(int(enabled)))
    reset_default()
    try:
        namespace["_register_marked_docs"](SimpleNamespace(cfg=cfg, app=app))
        store = get_default()
        if not enabled:
            assert store is None
            with TestClient(app) as client:
                assert client.get("/marked/unknown.html?token=wrong").status_code == 404
            return
        assert store is not None
        session = object.__new__(Session)
        session.marked_documents = [("source.pdf", b"private document")]
        block = register_marked_documents(
            session, store, "https://rag.example.test/prefix"
        )
        url = block.split("](", 1)[1].split(")", 1)[0]
        assert url.startswith("https://rag.example.test/prefix/marked/")
        path = url.removeprefix("https://rag.example.test/prefix")
        with TestClient(app) as client:
            for query in (
                "",
                "?token=wrong",
                "?token=%E6%97%A5",
                path[path.index("?") :] + "&token=wrong",
            ):
                rejected = client.get(path.split("?", 1)[0] + query)
                assert rejected.status_code == 404
                assert rejected.headers["Cache-Control"] == "private, no-store"
                assert rejected.content != b"private document"
            response = client.get(path, headers={"Origin": "https://ui.example.test"})
            assert response.status_code == 200
            assert response.content == b"private document"
            assert (
                response.headers["Access-Control-Allow-Origin"]
                == "https://ui.example.test"
            )
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert client.get(path).status_code == 404
    finally:
        reset_default()


def test_document_routes_reject_capabilities_from_previous_service_start() -> None:
    previous = MarkedDocsStore()
    current = MarkedDocsStore()
    app = FastAPI()
    MarkedDocsServer(app, current)
    token = current.put(b"private", "answer.html")
    with TestClient(app) as client:
        old_url = marked_document_url("http://testserver", previous, token, ".html")
        assert client.get(old_url).status_code == 404
        url = marked_document_url("http://testserver", current, token, ".html")
        assert client.get(url).content == b"private"


@pytest.mark.parametrize("prefix", ["", "/proxy-prefix"])
def test_uvicorn_document_access_logs_redact_both_capabilities(prefix: str) -> None:
    MarkedDocsServer(FastAPI(), None)
    logger = logging.getLogger("uvicorn.access")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        (
            "127.0.0.1",
            "GET",
            prefix + "/marked/document-secret.html?token=session-secret",
            "1.1",
            200,
        ),
        None,
    )
    assert logger.filter(record)
    assert "document-secret" not in record.getMessage()
    assert "session-secret" not in record.getMessage()


def test_document_server_over_live_loopback_http(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = FastAPI()
    store = MarkedDocsStore()
    MarkedDocsServer(app, store)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                log_config=None,
                lifespan="off",
            )
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            port = listener.getsockname()[1]
            token = store.put(b"<h1>Private answer</h1>", "answer.html")
            url = marked_document_url(f"http://127.0.0.1:{port}", store, token, ".html")
            opener = build_opener(ProxyHandler({}))
            with caplog.at_level(logging.INFO, logger="uvicorn.access"):
                with pytest.raises(HTTPError) as error:
                    opener.open(url.split("?", 1)[0], timeout=2)
                assert error.value.code == 404
                with opener.open(url, timeout=2) as response:
                    assert response.status == 200
                    assert response.read() == b"<h1>Private answer</h1>"
                    assert response.headers["Referrer-Policy"] == "no-referrer"
            assert "/marked/<redacted>" in caplog.text
            assert token not in caplog.text
            assert store.session_token not in caplog.text
        finally:
            server.should_exit = True
            thread.join(timeout=5)
        assert not thread.is_alive()
