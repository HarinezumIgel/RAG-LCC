#!/usr/bin/env python3
import asyncio
import hmac
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import Configuration.Config_Internet_Env  # type: ignore[reportUnusedImport]  # side-effect import

if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1":
    from Commons.NetworkTracer import NetworkTracer

    NetworkTracer.enable_tracer()
else:
    NetworkTracer = None  # type: ignore[assignment,misc]

from Commons.StartupCommons import StartupCommons
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper

if os.environ.get("SERVE_OPENWEBUI_CHAT", "0") != "1":
    PrettyWriter().write(
        "E",
        "RAGChatService",
        "SERVE_OPENWEBUI_CHAT is not enabled. "
        'Set SERVE_OPENWEBUI_CHAT="1" in Config_Internet_Env.py to start the service.',
        color=RED,
    )
    sys.exit(1)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from AI.AIHelpers import AIHelpers
from Api.ChatCompletionHandler import (  # pyright: ignore[reportPrivateUsage]; pyright: ignore[reportPrivateUsage, reportUnknownVariableType]
    ChatCompletionRequest, _complianceResponse, _format_session_for_error,
    handleRequest)
from Api.MarkedDocsStore import \
    configure_default as configure_marked_docs_store
from Api.MarkedDocsStore import get_default as get_marked_docs_store
from Chat.Chatter import Chatter
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Compliance.HFDownloader import HFDownloader
from Globals.Globals import Globals
from Gui.Informer import Informer
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers


class RAGChatService:
    def __init__(self):
        self.ctx = StartupCommons.common_start(
            "RAGChatService",
            "RAGChatService – OpenWebUI-compatible REST API for RAGChat",
        )
        self.cfg = self.ctx.cfg
        self.serviceCfgDebugLevel: int = DebugHelper.level(self.cfg)
        self.executor: ThreadPoolExecutor | None = None
        self.lock: asyncio.Lock | None = None
        self.chatter: Chatter | None = None
        self.rag: RAGChatImpl | None = None
        self.queryParts: QueryParts | None = None
        self.aiHelpers: AIHelpers | None = None
        self.app = FastAPI(
            title="RAGChat OpenWebUI Service",
            description="OpenAI-compatible /v1/chat/completions endpoint backed by RAG-LCC.",
            version="0.1.0",
            lifespan=self.lifespan,
        )
        self._register_marked_docs()
        self._register_routes()
        self._register_auth_middleware()

    def _register_marked_docs(self) -> None:
        """Configure the in-memory marked-docs store and optional CORS.

        Enable/disable gate is ``SERVE_IN_MEMORY_DOCS_HTTP`` from
        Config_Internet_Env.py. Runtime options come from ``_SERVE_DOCS``.
        When the gate is disabled the store stays uninitialised and
        the /marked route returns 404 for every request.
        """
        if os.environ.get("SERVE_IN_MEMORY_DOCS_HTTP", "0") != "1":
            return
        block = self.cfg.get_dict("_SERVE_DOCS", {}) or {}
        # Fail fast: both packages are required when the feature is enabled.
        missing: list[str] = []
        try:
            import pdfplumber  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            missing.append("pdfplumber")
        try:
            import pypdf  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            missing.append("pypdf")
        if missing:
            PrettyWriter().write(
                "E",
                "RAGChatService",
                f"SERVE_IN_MEMORY_DOCS_HTTP=1 but the following required package(s) are not installed: "
                f"{', '.join(missing)}. "
                f"Install with:  pip install {' '.join(missing)}"
                f'\nSet SERVE_IN_MEMORY_DOCS_HTTP="0" in Configuration/Config_Internet_Env.py '
                f"to disable document serving and suppress this error.",
                color=RED,
            )
            sys.exit(1)
        configure_marked_docs_store(
            ttl_seconds=int(block.get("ttl_seconds", 1800)),
            max_total_bytes=int(block.get("max_total_mb", 200)) * 1024 * 1024,
            single_use=bool(block.get("single_use", False)),
        )
        host = self.cfg.get_str(
            "_MODELS.ragchatservice._RAGCHATSERVICE.HOST", "127.0.0.1"
        )
        port = self.cfg.get_int("_MODELS.ragchatservice._RAGCHATSERVICE.PORT", 11435)
        base = (
            str(block.get("public_base_url", "") or "").strip().rstrip("/")
            or f"http://{host}:{port}"
        )
        PrettyWriter().write(
            "W",
            "RAGChatService",
            f"Marked-docs in-memory service active — endpoint: {base}/marked/<token>",
        )
        cors_origins = list(block.get("cors_origins", []) or [])
        if cors_origins:
            from fastapi.middleware.cors import CORSMiddleware

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_methods=["GET"],
                allow_headers=["*"],
                allow_credentials=False,
            )

    def _register_auth_middleware(self):
        expectedKey: str = self.cfg.get_str(
            "_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY", ""
        )

        @self.app.middleware("http")
        async def _verifyBearerToken(  # pyright: ignore[reportUnusedFunction]
            request: Request, call_next: RequestResponseEndpoint
        ) -> Response:  # pyright: ignore[reportUnusedFunction]
            if request.url.path.startswith("/v1/"):
                auth: str = request.headers.get("authorization", "")
                token: str = (
                    auth.removeprefix("Bearer ").strip()
                    if auth.lower().startswith("bearer ")
                    else ""
                )
                if not hmac.compare_digest(token, expectedKey):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key."},
                    )
            return await call_next(request)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        # ---- startup ----
        globalsInstance: Globals = Globals()
        helpers: Helpers = Helpers()
        fileUtils: FileUtils = FileUtils()
        self.inf: Informer = Informer()

        hf_downloader: HFDownloader = HFDownloader()
        hf_downloader.download("_MODELS._EMBED")
        hf_downloader.download("_MODELS._CROSS")

        logger = helpers.setup_logger()
        globalsInstance.set_logger(logger)

        helpers.check_cpu_and_bits()
        fileUtils.setDebug()
        self.inf.inform()

        self.queryParts = QueryParts()
        self.aiHelpers = AIHelpers()
        self.rag = RAGChatImpl()
        self.chatter = Chatter()
        self.lock = asyncio.Lock()
        threadPoolWorkers: int = max(
            1, self.cfg.get_int("OPENWEBUI_THREAD_POOL_WORKERS", 1)
        )
        self.executor = ThreadPoolExecutor(max_workers=threadPoolWorkers)
        PrettyWriter().write(
            "I",
            "RAGChatService",
            f"ThreadPoolExecutor started with max_workers={threadPoolWorkers}",
        )

        yield

        # ---- shutdown ----
        PrettyWriter().write("I", "RAGChatService", "Shutting down …")

        self.inf.write_counter_and_csv(
            label="RAG_CHAT_QUERIES:  ",
            count=0,
            csv_key="HUMAN_REVIEW",
            log_message="Have a look at RAG_CHAT_QUERIES .xlsx / .csv file",
            failure_indication=True,
        )

        self.executor.shutdown(wait=False)
        if os.environ.get("RAG_LCC_NW_TRACE", "0") == "1":
            try:
                from Commons.NetworkTracer import NetworkTracer

                NetworkTracer.disable_tracer()
            except Exception:
                pass

    def _register_routes(self):
        @self.app.exception_handler(RequestValidationError)
        async def _validationErrorHandler(  # pyright: ignore[reportUnusedFunction]
            request: Request, exc: RequestValidationError
        ) -> Response:  # pyright: ignore[reportUnusedFunction]
            model = "rag-chat"
            stream = False
            raw = b""
            try:
                raw = await request.body()
                body = json.loads(raw)
                model = str(body.get("model", model))
                stream = bool(body.get("stream", stream))
            except Exception:
                pass
            if self.serviceCfgDebugLevel >= 5:
                body_text = (
                    raw.decode("utf-8", errors="replace") if raw else "<unreadable>"
                )
                PrettyWriter().write(
                    "D",
                    "RAGChatService validation error raw body:",
                    body_text,
                    color=RED,
                )
                PrettyWriter().write(
                    "D",
                    "RAGChatService validation errors:",
                    str(exc.errors()),
                    color=RED,
                )
            lines: list[str] = []
            for err in exc.errors():
                loc_parts = [str(p) for p in err.get("loc", ()) if str(p) != "body"]
                loc = " → ".join(loc_parts) if loc_parts else "request"
                msg = err.get("msg", "validation error")
                inp = err.get("input")
                if inp is not None and not isinstance(inp, (dict, list)):
                    lines.append(f"**{loc}**: {msg} (got `{inp!r}`)")
                else:
                    lines.append(f"**{loc}**: {msg}")
            detail = (
                "\n".join(f"- {line}" for line in lines)
                if lines
                else "Validation error"
            )
            err_id = f"ragchat-{uuid.uuid4().hex}"
            if self.queryParts is not None:
                PrettyWriter().write(
                    "W",
                    "RAGChatService validation error — active session:",
                    "",
                    color=RED,
                )
                self.queryParts.print_values()
            session_block = (
                _format_session_for_error(self.queryParts.session)
                if self.queryParts is not None
                else ""
            )
            return _complianceResponse(
                err_id,
                model,
                f"\u26a0\ufe0f **Invalid request parameter**\n\n{detail}{session_block}",
                stream=stream,
            )

        @self.app.post("/v1/chat/completions")
        async def chatCompletions(  # pyright: ignore[reportUnusedFunction]
            req: ChatCompletionRequest, request: Request
        ):
            assert self.chatter is not None
            assert self.rag is not None
            assert self.queryParts is not None
            assert self.aiHelpers is not None
            assert self.lock is not None
            assert self.executor is not None

            return await handleRequest(
                req=req,
                chatter=self.chatter,
                rag=self.rag,
                queryParts=self.queryParts,
                aiHelpers=self.aiHelpers,
                cfg=self.cfg,
                lock=self.lock,
                executor=self.executor,
                serviceCfgDebugLevel=self.serviceCfgDebugLevel,
                request=request,
            )

        @self.app.get("/marked/{token_filename}")
        async def getMarkedDocument(  # pyright: ignore[reportUnusedFunction]
            token_filename: str,
        ) -> Response:
            """Serve a previously-registered, in-memory highlighted document.

            ``token_filename`` is ``<token>.<ext>`` where the extension
            matches the source file type (e.g. ``.pdf``, ``.docx``, ``.pptx``).
            """
            _MEDIA_TYPES: dict[str, str] = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".md": "text/markdown; charset=utf-8",
                ".txt": "text/plain; charset=utf-8",
            }
            store = get_marked_docs_store()
            if store is None:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            token = token_filename.split(".", 1)[0]
            entry = store.get(token)
            if entry is None:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            safe_name = entry.filename.replace('"', "")
            suffix = Path(entry.filename).suffix.lower()
            media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")
            return Response(
                content=entry.data,
                media_type=media_type,
                headers={
                    "Content-Disposition": f'inline; filename="{safe_name}"',
                    "Cache-Control": "private, no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        _resources_dir = self.cfg.get_str("_SERVICE_RESOURCES_DIR") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resources"
        )
        _favicon_path = os.path.normpath(os.path.join(_resources_dir, "favicon.ico"))
        _favicon_bytes: bytes = (
            open(_favicon_path, "rb").read() if os.path.isfile(_favicon_path) else b""
        )

        @self.app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Response:  # pyright: ignore[reportUnusedFunction]
            if not _favicon_bytes:
                return Response(status_code=404)
            return Response(
                content=_favicon_bytes,
                media_type="image/x-icon",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        @self.app.get("/v1/models")
        async def listModels() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
            chromaDbDir: str = self.cfg.get_str("_CHROMA_DB_DIR")
            collections: List[str] = []
            if chromaDbDir and os.path.isdir(chromaDbDir):
                collections = sorted(
                    name
                    for name in os.listdir(chromaDbDir)
                    if os.path.isdir(os.path.join(chromaDbDir, name))
                    and not name.startswith(".")
                    and not name.endswith("_ChatContext")
                )
            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "rag-lcc",
                        }
                        for name in collections
                    ],
                }
            )

    def run(self):
        apiHost: str = self.cfg.get_str(
            "_MODELS.ragchatservice._RAGCHATSERVICE.HOST", "127.0.0.1"
        )  # e.g. 127.0.0.1 or 0.0.0.0 if in Docker
        apiPort: int = self.cfg.get_int(
            "_MODELS.ragchatservice._RAGCHATSERVICE.PORT", 11435
        )
        try:
            uvicorn.run(self.app, host=apiHost, port=apiPort)
        except KeyboardInterrupt:
            PrettyWriter().write("I", "RAGChatService", "Stopped by Ctrl-C.")


def main() -> None:
    service = RAGChatService()
    service.run()


if __name__ == "__main__":
    StartupCommons.run_with_top_level_handlers(main)
# ---------------------------------------------------------------------------
