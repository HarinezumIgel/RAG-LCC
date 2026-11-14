import ast
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.responses import Response

from Api.ChatCompletionHandler import ChatCompletionRequest, _complianceResponse


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
