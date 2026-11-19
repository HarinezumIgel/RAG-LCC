"""Loopback-only, short-lived document links for the RAGChat CLI."""

from __future__ import annotations

import atexit
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

from Api.MarkedDocsService import marked_document_response, marked_document_url
from Api.MarkedDocsStore import MarkedDocsStore


class _DocumentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")

    def __init__(
        self, store: MarkedDocsStore, handler: type[BaseHTTPRequestHandler], port: int
    ) -> None:
        self.store = store
        super().__init__(("127.0.0.1", port), handler)

    def server_bind(self) -> None:
        # Windows SO_REUSEADDR permits competing listeners on the same port.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def service_actions(self) -> None:
        self.store.purge_expired()


class LocalDocsServer:
    """Share the service's token cache and responses, without exposing files."""

    def __init__(
        self,
        *,
        port: int = 11436,
        ttl_seconds: int = 1800,
        max_total_bytes: int = 200 * 1024 * 1024,
        single_use: bool = False,
    ) -> None:
        self.store = MarkedDocsStore(
            ttl_seconds=ttl_seconds,
            max_total_bytes=max_total_bytes,
            single_use=single_use,
        )
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlsplit(self.path)
                match = re.fullmatch(
                    r"/marked/([A-Za-z0-9_-]+\.[A-Za-z0-9.]+)",
                    url.path,
                )
                tokens = parse_qs(url.query, keep_blank_values=True).get("token", [])
                response = marked_document_response(
                    match.group(1) if match else "",
                    store,
                    tokens[0] if len(tokens) == 1 else "",
                )
                self.send_response(response.status_code)
                for name, value in response.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(response.body)

            def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
                # Do not log capability tokens from document URLs.
                pass

        self._server = _DocumentHTTPServer(self.store, Handler, port)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="RAGChat-local-docs",
            daemon=True,
        )
        self._closed = False
        try:
            self._thread.start()
        except RuntimeError:
            self._server.server_close()
            raise
        atexit.register(self.close)

    def put(
        self,
        data: bytes,
        filename: str,
        *,
        response_headers: Mapping[str, str] | None = None,
    ) -> str:
        if self._closed:
            raise RuntimeError("The local document listener is closed.")
        token = self.store.put(data, filename, response_headers=response_headers)
        suffix = (
            ".txt.md"
            if filename.endswith(".txt.md")
            else Path(filename).suffix or ".bin"
        )
        return marked_document_url(self.base_url, self.store, token, suffix)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self.store.clear()
        atexit.unregister(self.close)
