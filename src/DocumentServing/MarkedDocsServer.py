"""Document routes on RAGChatService's existing HTTP listener."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from Api.MarkedDocsService import marked_document_response
from Api.MarkedDocsStore import MarkedDocsStore


class _DocumentAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn includes the full path and query in its access-log arguments.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, path, version, status = record.args
            if isinstance(path, str) and "/marked/" in path.split("?", 1)[0]:
                record.args = (client, method, "/marked/<redacted>", version, status)
        return True


class MarkedDocsServer:
    """Register browser-accessible routes without opening another listener."""

    def __init__(
        self,
        app: FastAPI,
        store: MarkedDocsStore | None,
        *,
        cors_origins: list[str] | None = None,
    ) -> None:
        access_logger = logging.getLogger("uvicorn.access")
        if not any(isinstance(f, _DocumentAccessFilter) for f in access_logger.filters):
            access_logger.addFilter(_DocumentAccessFilter())
        if cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_methods=["GET"],
                allow_headers=["*"],
                allow_credentials=False,
            )

        @app.get("/marked/{token_filename}")
        async def get_marked_document(
            token_filename: str, request: Request
        ) -> Response:
            tokens = request.query_params.getlist("token")
            return marked_document_response(
                token_filename, store, tokens[0] if len(tokens) == 1 else ""
            )
