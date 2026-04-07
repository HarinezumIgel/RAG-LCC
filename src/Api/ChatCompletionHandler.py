import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from AI.AIHelpers import AIHelpers
from Algos.ComplianceAlgoResult import ResultsForPrint
from Chat.Chatter import Chatter
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Commons.Exceptions import CollectionNotFoundError, LLMResultError
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, MAGENTA
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.CSVWriter import CSVWriter

# ---------------------------------------------------------------------------
# Ollama option keys that map to the "options" sub-dict in the payload
# ---------------------------------------------------------------------------
_OLLAMA_OPTIONS_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "repeat_last_n",
        "repeat_penalty",
        "tfs_z",
        "min_p",
        "use_mmap",
        "use_mlock",
        "num_keep",
        "num_batch",
        "num_thread",
        "num_gpu",
    }
)

# Top-level Ollama payload keys (not inside "options")
_OLLAMA_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"think", "keep_alive", "format"})

# Prefixes used by OpenWebUI for housekeeping tasks (follow-ups, tags, titles).
# These requests should NOT go through the RAG pipeline.
_OPENWEBUI_FOLLOWUP_PREFIX = "### Task:\nSuggest 3-5 relevant follow-up questions"
_OPENWEBUI_TAGS_PREFIX = "### Task:\nGenerate 1-3 broad tags"
_OPENWEBUI_TITLE_PREFIX = "### Task:\nGenerate a concise, 3-5 word title"


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]] = ""

    def text(self) -> str:
        """Return content as plain text regardless of whether it arrived as a string or a list of parts."""
        if isinstance(self.content, list):
            parts: list[str] = []
            for part in self.content:
                if isinstance(part, dict):
                    typed_part = cast(Dict[str, Any], part)
                    val = typed_part.get("text", "")
                    parts.append(val if isinstance(val, str) else str(val))
                else:
                    parts.append(str(part))
            return " ".join(parts)
        return self.content


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow"
    )  # accept unknown fields from OpenWebUI custom params

    model: str
    messages: List[ChatMessage]
    stream: bool = False

    # Standard LLM sampling params (mapped to Session / Ollama options)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_tokens: Optional[int] = None
    num_ctx: Optional[int] = None

    # RAG-LCC specific params — can be set via OpenWebUI model "Advanced Parameters"
    strategy: Optional[str] = None
    chroma_k_value: Optional[int] = None
    rerank: Optional[bool] = None
    chroma_threshold: Optional[float] = None
    chunks_window: Optional[int] = None
    chroma_weight: Optional[float] = None
    use_chat_context: Optional[bool] = None
    chat_name: Optional[str] = None
    per_file_limit: Optional[int] = None
    debug_level: Optional[int] = None

    # Ollama options passthrough
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    tfs_z: Optional[float] = None
    min_p: Optional[float] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    num_keep: Optional[int] = None
    num_batch: Optional[int] = None
    num_thread: Optional[int] = None
    num_gpu: Optional[int] = None

    # Ollama top-level payload params
    think: Optional[bool] = None
    keep_alive: Optional[str] = None
    format: Optional[Union[str, Dict[str, Any]]] = None

    # OpenWebUI-injected fields
    chat_id: Optional[str] = None  # stable per-conversation ID sent by OpenWebUI

    # OpenAI-only — accepted but ignored
    logit_bias: Optional[Dict[str, Any]] = None
    stream_options: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_values(cls, values: Any) -> Any:
        """Coerce string values sent by OpenWebUI to the declared field types.

        OpenWebUI serialises custom "Advanced Parameters" as JSON strings
        regardless of the field's declared type (e.g. ``"60"`` instead of
        ``60``).  Without this pre-processing step, Pydantic v2's strict
        JSON-mode validation rejects the request with a 422.
        """
        if not isinstance(values, dict):
            return values

        data: dict[str, Any] = cast(dict[str, Any], values)
        _BOOL_STRINGS: dict[str, bool] = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }

        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            raw: Any = data[field_name]

            annotation: Any = field_info.annotation
            # Unwrap Optional[X] → X
            origin: Any = getattr(annotation, "__origin__", None)
            if origin is Union:
                args: list[Any] = [
                    a
                    for a in getattr(annotation, "__args__", ())
                    if a is not type(None)
                ]
                annotation = args[0] if len(args) == 1 else annotation

            try:
                if isinstance(raw, str):
                    # OpenWebUI sends custom params as JSON strings
                    if annotation is int:
                        data[field_name] = int(raw)
                    elif annotation is float:
                        data[field_name] = float(raw)
                    elif annotation is bool:
                        lowered = raw.strip().lower()
                        if lowered in _BOOL_STRINGS:
                            data[field_name] = _BOOL_STRINGS[lowered]
                elif isinstance(raw, float) and annotation is int:
                    # OpenWebUI may send JSON floats (e.g. 100.0) for int fields
                    data[field_name] = int(raw)
            except (ValueError, TypeError):
                pass  # let Pydantic report the validation error
        return data

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v]
        return v


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _buildQuey(messages: List[ChatMessage]) -> str:
    """
    Extract the user query from the messages array.
    If a system message is present, prepend it to the user query so the RAG
    prompt template still receives a simple {input} string.
    """
    system_text: str | None = None
    user_text: str = ""
    for msg in messages:
        if msg.role == "system" and system_text is None:
            system_text = msg.text().strip()
        elif msg.role == "user":
            user_text = msg.text()  # last user message wins
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


_ALLOWED_STRATEGIES: frozenset[str] = frozenset(
    {"ULTRA_WIDE", "WIDE", "MEDIUM", "NARROW"}
)


def _getLastUserText(req: ChatCompletionRequest) -> str | None:
    """Return the text of the last user message, or None."""
    if not req.messages:
        return None
    last_user = None
    for msg in req.messages:
        if msg.role == "user":
            last_user = msg
    return last_user.text() if last_user is not None else None


def _isFollowUpRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI follow-up suggestion prompt."""
    txt = _getLastUserText(req)
    return txt is not None and txt.startswith(_OPENWEBUI_FOLLOWUP_PREFIX)


def _isTagsRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI tag-generation prompt."""
    txt = _getLastUserText(req)
    return txt is not None and txt.startswith(_OPENWEBUI_TAGS_PREFIX)


def _isTitleRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI title-generation prompt."""
    txt = _getLastUserText(req)
    return txt is not None and txt.startswith(_OPENWEBUI_TITLE_PREFIX)


def _applyRequestToSession(
    req: ChatCompletionRequest,
    session: Session,
    queryParts: QueryParts,
    cfg: Config,
) -> None:
    """
    Load the strategy defaults into Session and then apply any per-request
    overrides coming from the OpenWebUI / API caller.
    """

    # Switch collection if needed
    incoming_collection = req.model
    if (
        session.collection_name is not None
        and session.collection_name != incoming_collection
    ):
        # Reset things before switching collection
        if hasattr(queryParts, "_reset_things"):
            queryParts.reset_things()
    session.collection_name = incoming_collection

    # Determine strategy: request > session > config default
    if req.strategy is not None:
        strategy_upper = req.strategy.upper()
        if strategy_upper not in _ALLOWED_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy {req.strategy!r}. Allowed: {sorted(_ALLOWED_STRATEGIES)}",
            )
        session.strategy = strategy_upper
    strategy = session.strategy or cfg.get_str("CHUNK_SELECT_STRATEGY") or "MEDIUM"
    queryParts.applyStrategyDefaults(strategy, session=session)

    def _applyOverrides() -> None:
        # RAG-LCC param overrides (applied after strategy defaults so they take precedence)
        if req.chroma_k_value is not None:
            session.chroma_k_value = req.chroma_k_value
        if req.rerank is not None:
            session.rerank = req.rerank
        if req.chroma_threshold is not None:
            session.chroma_threshold = max(0.0, min(1.0, req.chroma_threshold))
        if req.chunks_window is not None:
            session.chunks_window = req.chunks_window
        if req.chroma_weight is not None:
            session.chroma_weight = max(0.0, min(1.0, req.chroma_weight))
        if req.use_chat_context is not None:
            session.use_chat_context = req.use_chat_context
        # chat_name priority: explicit param > OpenWebUI chat_id > keep existing
        if req.chat_name is not None:
            session.chat_name = req.chat_name
        elif req.chat_id is not None:
            session.chat_name = req.chat_id
        if req.per_file_limit is not None:
            session.per_file_limit = req.per_file_limit
        if req.debug_level is not None:
            session.debug_level = req.debug_level
            cfg.set("DEBUG_LEVEL", req.debug_level)

        # Per-request LLM param overrides
        if req.temperature is not None:
            session.temperature = req.temperature
        if req.top_p is not None:
            session.top_p = req.top_p
        if req.top_k is not None:
            session.top_k = float(req.top_k)
        if req.max_tokens is not None:
            session.max_output_tokens_override = req.max_tokens
        if req.num_ctx is not None:
            session.context_size_override = req.num_ctx

    _applyOverrides()

    # Build extra Ollama options from the request
    extra_options: dict[str, Any] = {}
    for key in _OLLAMA_OPTIONS_KEYS:
        value = getattr(req, key, None)
        if value is not None:
            extra_options[key] = value
    session.extraOllamaOptions = extra_options if extra_options else None

    # Build top-level Ollama payload params from the request
    top_level: dict[str, Any] = {}
    for key in _OLLAMA_TOP_LEVEL_KEYS:
        value = getattr(req, key, None)
        if value is not None:
            top_level[key] = value
    session.ollamaTopLevelParams = top_level if top_level else None

    session.query = _buildQuey(req.messages)
    session.base_kwargs = {"k": session.chroma_k_value}


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------


async def _streamGenerator(
    chatter: Chatter,
    session: Session,
    req_id: str,
    model: str,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    lock: asyncio.Lock,
    cfg_debug_level: int = 0,
    *,
    show_algo_results: bool = False,
    prompt_check_md: str = "",
) -> AsyncGenerator[str, None]:
    """
    Runs chatter.run() in a thread executor and yields SSE chunks compatible
    with the OpenAI streaming format.

    *lock* is the request-serialisation lock whose ownership was transferred
    here by ``handleRequest``.  It is released in the ``finally`` block so
    that the singleton RAGChatImpl stays protected for the entire duration
    of ``chatter.run()``.
    """
    try:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        created = int(time.time())

        def on_chunk(text: str) -> None:
            if text:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                except RuntimeError:
                    pass  # event loop closed during shutdown

        # Yield the opening delta with role
        opening = (
            f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
            f'"model":"{model}","choices":[{{"index":0,"delta":{{"role":"assistant"}},'
            f'"finish_reason":null}}]}}'
        )
        yield f"data: {opening}\n\n"

        compliance_failed = False

        async def run_in_thread() -> None:
            nonlocal compliance_failed
            try:
                success, _ = await loop.run_in_executor(
                    executor,
                    lambda: (
                        PrettyWriter().write(
                            "I",
                            "RAGChatService worker:",
                            f"thread={threading.current_thread().name!r}  req_id={req_id!r}  streaming=True",
                            color=CYAN,
                        ),
                        chatter.run(
                            session, apiChunkHandler=on_chunk, is_streaming=True
                        ),
                    )[-1],
                )
                if success is False:
                    compliance_failed = True
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
                except RuntimeError:
                    pass  # event loop closed during shutdown

        asyncio.ensure_future(run_in_thread())

        while True:
            token = await queue.get()
            if token is None:
                break
            chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(token)}}},'
                f'"finish_reason":null}}]}}'
            )
            yield f"data: {chunk}\n\n"

        if compliance_failed:
            algo_suffix = ""
            if show_algo_results:
                answer_check_md = Accumulator().format_results_as_md("Answer Check")
                algo_suffix = "\n\n---\n### Filter chain algo results Results\n\n"
                if prompt_check_md:
                    algo_suffix += prompt_check_md + "\n\n"
                algo_suffix += answer_check_md
            error_content = "[Answer blocked by compliance check]" + algo_suffix
            error_chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(error_content)}}},'
                f'"finish_reason":"stop"}}]}}'
            )
            yield f"data: {error_chunk}\n\n"
        else:
            if show_algo_results:
                answer_check_md = Accumulator().format_results_as_md("Answer Check")
                algo_block = "\n\n---\n### Filter chain algo results Results\n\n"
                if prompt_check_md:
                    algo_block += prompt_check_md + "\n\n"
                algo_block += answer_check_md
                algo_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(algo_block)}}},'
                    f'"finish_reason":null}}]}}'
                )
                yield f"data: {algo_chunk}\n\n"
            final_chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{}},'
                f'"finish_reason":"stop"}}]}}'
            )
            yield f"data: {final_chunk}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        lock.release()


def _json_str(text: str) -> str:
    """JSON-encode a plain string value (with quotes)."""
    return json.dumps(text)


_MSG_CONTENT_LIMIT: int = 80  # max chars shown per message content in the debug summary


def _summarise_request(raw_dump: Dict[str, Any]) -> str:
    """Build a compact one-line summary of the incoming request for debug logging.

    * Drops keys whose value is ``None`` (the bulk of the noise).
    * Truncates long message ``content`` fields.
    * Keeps the output as a single readable line.
    """
    cleaned: Dict[str, Any] = {}

    for k, v in raw_dump.items():
        if v is None:
            continue
        if k == "messages" and isinstance(v, list):
            short_msgs: list[Dict[str, Any]] = []
            for msg in cast(List[Any], v):
                if not isinstance(msg, dict):
                    short_msgs.append(msg)
                    continue
                m: Dict[str, Any] = dict(cast(Dict[str, Any], msg))
                content: Any = m.get("content", "")
                if isinstance(content, str) and len(content) > _MSG_CONTENT_LIMIT:
                    m["content"] = content[:_MSG_CONTENT_LIMIT] + "…"
                short_msgs.append(m)
            cleaned[k] = short_msgs
        else:
            cleaned[k] = v

    return json.dumps(cleaned, ensure_ascii=False)


def _format_validation_details(phrase_table: List[ResultsForPrint]) -> str:
    """Build a human-readable summary of prompt-validation algo results."""
    from collections import defaultdict

    # Group results by phrase
    by_phrase: Dict[str, List[ResultsForPrint]] = defaultdict(list)
    for r in phrase_table:
        by_phrase[r.phrase].append(r)

    lines: list[str] = ["\n\n---\n**Validation details:**"]
    for phrase in sorted(by_phrase, key=str.lower):
        entries = by_phrase[phrase]
        # Extract depth/breadth from the first entry's algos_matched field
        meta = entries[0].algos_matched or ""
        lines.append(f"\n**Phrase:** `{phrase}`  (dpt/brth: {meta})")
        for e in entries:
            lines.append(f"- {e.algo}: {e.score_str}")
    return "\n".join(lines)


def _log_compliance_csv(
    csvWriter: "CSVWriter",
    bannedPhraseCollector: "BannedPhraseCollector",
    phrase_table: List[ResultsForPrint],
    human_review: bool,
    stage: str,
    *,
    session: Optional["Session"] = None,
) -> None:
    """Write a HUMAN_REVIEW CSV row, mirroring RAGChat._process_query logging."""
    status = "NOT_OK" if human_review else "OK"
    meta: Dict[str, Any] = {
        "Stage": stage,
        "Time": datetime.now(),
        "Status": status,
    }
    if session is not None:
        meta["Session"] = session.export_session_state_as_cell()

    try:
        if phrase_table:
            rows = bannedPhraseCollector.prepare_for_csv_print(phrase_table, meta)
        else:
            rows = bannedPhraseCollector.prepare_print_for_chat(meta)
        csvWriter.write_json2csv(rows, "HUMAN_REVIEW")
    except Exception:
        pass


def _complianceResponse(req_id: str, model: str, message: str, stream: bool = False):
    """Return a 200 chat-completion response carrying a compliance notice.

    When *stream* is True the response is emitted as SSE chunks so that
    OpenWebUI (which requested streaming) can parse and store it correctly.
    """
    created = int(time.time())
    if stream:

        async def _gen():
            role_chunk = json.dumps(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield f"data: {role_chunk}\n\n"
            content_chunk = json.dumps(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": message},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield f"data: {content_chunk}\n\n"
            stop_chunk = json.dumps(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield f"data: {stop_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(
        content={
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


# ---------------------------------------------------------------------------
# Main request handler
# ---------------------------------------------------------------------------


async def handleRequest(
    req: ChatCompletionRequest,
    chatter: Chatter,
    rag: RAGChatImpl,
    queryParts: QueryParts,
    aiHelpers: AIHelpers,
    cfg: Config,
    lock: asyncio.Lock,
    executor: ThreadPoolExecutor,
    serviceCfgDebugLevel: int = 0,
) -> StreamingResponse | JSONResponse:
    """
    Entry point called by the FastAPI route.  A fresh :class:`Session` is
    created for every request so that concurrent callers cannot clobber
    each other's parameters.  The *lock* still serialises access to the
    singleton :class:`RAGChatImpl` whose vector-store state lives on *self*.
    """
    # Per-request state — immune to concurrent overwrites
    session = Session()

    # -------------------------------------------------------------------
    # Fast-path: OpenWebUI housekeeping requests (follow-ups, tags, titles)
    # These are synthetic prompts that should NOT go through the RAG
    # pipeline.  Handled *before* lock.acquire() so they never block on
    # a long-running chatter.run() and never hold the lock themselves.
    # -------------------------------------------------------------------
    if _isFollowUpRequest(req):
        req_id = f"ragchat-{uuid.uuid4().hex}"
        if serviceCfgDebugLevel >= 2:
            PrettyWriter().write(
                "D",
                "RAGChatService follow-up:",
                f"Skipped RAG pipeline for OpenWebUI follow-up suggestion (model={req.model!r})",
                color=CYAN,
            )
        return _complianceResponse(
            req_id,
            req.model,
            '{"follow_ups": []}',
            stream=req.stream,
        )

    if _isTagsRequest(req):
        req_id = f"ragchat-{uuid.uuid4().hex}"
        if serviceCfgDebugLevel >= 2:
            PrettyWriter().write(
                "D",
                "RAGChatService tags:",
                f"Skipped RAG pipeline for OpenWebUI tag generation (model={req.model!r})",
                color=CYAN,
            )
        return _complianceResponse(
            req_id,
            req.model,
            '{"tags": ["General"]}',
            stream=req.stream,
        )

    if _isTitleRequest(req):
        req_id = f"ragchat-{uuid.uuid4().hex}"
        if serviceCfgDebugLevel >= 2:
            PrettyWriter().write(
                "D",
                "RAGChatService title:",
                f"Skipped RAG pipeline for OpenWebUI title generation (model={req.model!r})",
                color=CYAN,
            )
        return _complianceResponse(
            req_id,
            req.model,
            '{"title": "RAG Chat"}',
            stream=req.stream,
        )

    await lock.acquire()
    lock_released = False
    try:
        raw_dump = (
            req.model_dump()
        )  # capture before applyRequestToSession mutates session

        _applyRequestToSession(req, session, queryParts, cfg)

        # Use the debug level captured at service startup — strategies reset
        # cfg's DEBUG_LEVEL to their own value during initialization.
        debug_level = serviceCfgDebugLevel
        pretty = PrettyWriter() if debug_level >= 2 else None

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService incoming:",
                _summarise_request(raw_dump),
                color=CYAN,
            )

        if cfg.get_int("DEBUG_LEVEL", 0) >= 3:
            PrettyWriter().write(
                "D",
                "RAGChatService session:",
                (
                    f"collection={session.collection_name!r}  strategy={session.strategy!r}  "
                    f"temperature={session.temperature}  top_k={session.top_k}  top_p={session.top_p}\n"
                    f"chroma_k_value={session.chroma_k_value}  chroma_threshold={session.chroma_threshold}  "
                    f"chunks_window={session.chunks_window}  rerank={session.rerank}  "
                    f"chroma_weight={session.chroma_weight}\n"
                    f"use_chat_context={session.use_chat_context}  "
                    f"chat_context_k_value={session.chat_context_k_value}  "
                    f"turns={session.turns}  "
                    f"chat_name={session.chat_name!r}  "
                    f"chat_id={req.chat_id!r}  "
                    f"per_file_limit={session.per_file_limit}\n"
                    f"batch_size={session.batch_size}  "
                    f"debug_level={session.debug_level}  "
                    f"max_output_tokens(api: max_tokens)={session.max_output_tokens}  "
                    f"max_output_tokens_override={session.max_output_tokens_override}  "
                    f"context_size_override(api: num_ctx)={session.context_size_override}\n"
                    f"extraOllamaOptions={session.extraOllamaOptions}  "
                    f"ollamaTopLevelParams={session.ollamaTopLevelParams}"
                ),
                color=MAGENTA,
            )

        try:
            rag.set_vector_store(session)
        except CollectionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService vector store:",
                f"collection={session.collection_name!r}",
                color=CYAN,
            )

        # ----- compliance: Stage 1 – filter chain -----
        stage = "PROMPT_CHECK"
        human_review, phrase_table = aiHelpers.check_user_prompt_with_filter_chain(
            session.query or "", stage
        )
        csvWriter = CSVWriter()
        bannedPhraseCollector = BannedPhraseCollector()

        _log_compliance_csv(
            csvWriter,
            bannedPhraseCollector,
            phrase_table,
            human_review,
            stage,
        )

        # Capture prompt-check algo results for MD rendering (if enabled)
        show_algo_results = cfg.get_bool("SHOW_CLI_LIKE_ALGO_RESULTS", False)
        prompt_check_md = ""
        if show_algo_results:
            prompt_check_md = Accumulator().format_results_as_md("Prompt Check")

        if human_review:
            matched = sorted(
                {r.phrase for r in phrase_table if r.phrase},
                key=str.lower,
            )
            phrase_hint = (
                f" Your query contained content related to: {', '.join(matched)}."
                if matched
                else ""
            )
            if show_algo_results and prompt_check_md:
                detail_block = (
                    "\n\n---\n### Filter chain algo results Results\n\n"
                    + prompt_check_md
                )
            else:
                detail_block = (
                    _format_validation_details(phrase_table) if phrase_table else ""
                )
            return _complianceResponse(
                f"ragchat-{uuid.uuid4().hex}",
                req.model,
                f"I'm sorry, but I can't help with that query.{phrase_hint} "
                "Please rephrase your question or ask about a different topic."
                f"{detail_block}",
                stream=req.stream,
            )

        # ----- compliance: Stage 2 – LLM prompt guard -----
        guard_rejected, guard_reason = aiHelpers.check_prompt_with_llm_guard(
            session.query or ""
        )

        _log_compliance_csv(
            csvWriter,
            bannedPhraseCollector,
            [],
            guard_rejected,
            stage,
            session=session,
        )

        if guard_rejected:
            guard_msg = (
                "I'm sorry, but I can't help with that query. "
                "Please rephrase your question or ask about a different topic."
            )
            if show_algo_results:
                guard_msg += "\n\n---\n### Filter chain algo results Results\n\n"
                if prompt_check_md:
                    guard_msg += prompt_check_md + "\n\n"
                guard_msg += f"⚠️ {guard_reason}"
            else:
                guard_msg += f"\n\n---\n**Guard:** {guard_reason}"
            return _complianceResponse(
                f"ragchat-{uuid.uuid4().hex}",
                req.model,
                guard_msg,
                stream=req.stream,
            )

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService compliance:",
                "User prompt passed compliance check.",
                color=CYAN,
            )

        req_id = f"ragchat-{uuid.uuid4().hex}"

        if req.stream:
            if pretty is not None:
                pretty.write(
                    "D",
                    "RAGChatService dispatch:",
                    f"streaming=True  req_id={req_id!r}",
                    color=CYAN,
                )
            loop = asyncio.get_event_loop()
            # Transfer lock ownership to _streamGenerator so that the
            # singleton RAGChatImpl cannot be mutated by a concurrent
            # request while chatter.run() is still active.
            lock_released = True
            return StreamingResponse(
                _streamGenerator(
                    chatter,
                    session,
                    req_id,
                    req.model,
                    loop,
                    executor,
                    lock,
                    serviceCfgDebugLevel,
                    show_algo_results=show_algo_results,
                    prompt_check_md=prompt_check_md,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming path
        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService dispatch:",
                f"streaming=False  req_id={req_id!r}",
                color=CYAN,
            )
        chatter.is_streaming = False
        loop = asyncio.get_event_loop()
        try:

            def _run_non_streaming() -> tuple[bool, str | None]:
                PrettyWriter().write(
                    "I",
                    "RAGChatService worker:",
                    f"thread={threading.current_thread().name!r}  req_id={req_id!r}  streaming=False",
                    color=CYAN,
                )
                return chatter.run(session, is_streaming=False)

            success, answer_text = await loop.run_in_executor(
                executor, _run_non_streaming
            )
        except LLMResultError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        if not success:
            compliance_msg = (
                "I'm sorry, but I can't provide an answer to that. "
                "The response did not pass the content compliance check."
            )
            if show_algo_results:
                answer_check_md = Accumulator().format_results_as_md("Answer Check")
                algo_block = "\n\n---\n### Filter chain algo results Results\n\n"
                if prompt_check_md:
                    algo_block += prompt_check_md + "\n\n"
                algo_block += answer_check_md
                compliance_msg += algo_block
            return _complianceResponse(
                req_id,
                req.model,
                compliance_msg,
                stream=False,
            )

        created = int(time.time())
        answer_out = answer_text or ""

        if show_algo_results:
            answer_check_md = Accumulator().format_results_as_md("Answer Check")
            algo_block = "\n\n---\n### Filter chain algo results Results\n\n"
            if prompt_check_md:
                algo_block += prompt_check_md + "\n\n"
            algo_block += answer_check_md
            answer_out += algo_block

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService answer:",
                answer_out,
                color=CYAN,
            )
        return JSONResponse(
            content={
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer_out},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not lock_released:
            lock.release()
