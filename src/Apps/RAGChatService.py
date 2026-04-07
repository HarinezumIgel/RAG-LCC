import asyncio
import hmac
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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

if os.environ.get("SERVE_OPENWEBUI_CHAT", "0") != "1":
    PrettyWriter().write(
        "E",
        "RAGChatService",
        "SERVE_OPENWEBUI_CHAT is not enabled. "
        'Set SERVE_OPENWEBUI_CHAT="1" in Config_Internet_Env.py to start the service.',
        color=RED,
    )
    sys.exit(1)

ctx = StartupCommons.common_start(
    "RAGChatService",
    "RAGChatService – OpenWebUI-compatible REST API for RAGChat",
)
cfg = ctx.cfg
_serviceCfgDebugLevel: int = cfg.get_int("DEBUG_LEVEL", 0)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from AI.AIHelpers import AIHelpers
from Api.ChatCompletionHandler import ChatCompletionRequest, handleRequest
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
        self.serviceCfgDebugLevel: int = self.cfg.get_int("DEBUG_LEVEL", 0)
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
        self._register_routes()
        self._register_auth_middleware()

    def _register_auth_middleware(self):
        expectedKey: str = self.cfg.get_str("OPENWEBUI_API_KEY", "RAGChatService")

        @self.app.middleware("http")
        async def _verifyBearerToken(request: Request, call_next):  # type: ignore[no-untyped-def]
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
        async def _validationErrorHandler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
            if self.serviceCfgDebugLevel >= 5:
                try:
                    raw = await request.body()
                    body_text = raw.decode("utf-8", errors="replace")
                except Exception:
                    body_text = "<unreadable>"
                PrettyWriter().write(
                    "D",
                    "RAGChatService 422 raw body:",
                    body_text,
                    color=RED,
                )
                PrettyWriter().write(
                    "D",
                    "RAGChatService 422 errors:",
                    str(exc.errors()),
                    color=RED,
                )
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

        @self.app.post("/v1/chat/completions")
        async def chatCompletions(
            req: ChatCompletionRequest,
        ):  # pyright: ignore[reportUnusedFunction]
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
        apiHost: str = self.cfg.get_str("OPENWEBUI_API_HOST", "127.0.0.1")
        apiPort: int = self.cfg.get_int("OPENWEBUI_API_PORT", 11435)
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
