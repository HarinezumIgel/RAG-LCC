import asyncio
import json
import ntpath
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from AI.AIHelpers import AIHelpers
from Algos.ComplianceAlgoResult import ResultsForPrint
from Api.ChatRequestSessionMapper import (_ALLOWED_STRATEGIES,
                                          _OLLAMA_OPTIONS_KEYS,
                                          _OLLAMA_TOP_LEVEL_KEYS,
                                          RequestSessionMapper, _buildQuery,
                                          _getLastUserText,
                                          _resolve_allowed_orchestration_flows,
                                          _resolve_default_orchestration_flow)
from Api.ChatResponseStreamer import ChatResponseStreamer, _json_str
from Api.MarkedDocsService import register_marked_documents
from Api.MarkedDocsStore import get_default as get_marked_docs_store
from Api.OpenWebUIHousekeeping import (_OPENWEBUI_FOLLOWUP_PREFIX,
                                       _OPENWEBUI_TAGS_PREFIX,
                                       _OPENWEBUI_TITLE_PREFIXES,
                                       OpenWebUIHousekeeping)
from Chat.Chatter import Chatter
from Chat.ContextSummarizer import ContextSummarizer
from Chat.FileSelector import ServiceFileSelector
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Commons.Exceptions import (CollectionNotFoundError, EmbedModelMismatch,
                                LLMResultError, OllamaNotRunning,
                                VllmNotRunning)
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, GREEN, MAGENTA, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.CSVWriter import CSVWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.PerfLogger import PerfLogger
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Helpers.SourcePathLinkifier import SourcePathLinkifier
from Orchestrator import FlowConfig

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
    max_output_tokens: Optional[int] = None  # CLI alias for max_tokens
    num_ctx: Optional[int] = None
    context_size: Optional[int] = None  # CLI alias for num_ctx

    # RAG-LCC specific params — can be set via OpenWebUI model "Advanced Parameters"
    # Each has both the internal API name and the CLI-facing alias so users who read
    # the session dump and enter CLI names in OpenWebUI get proper validation.
    # CLI names are canonical in QueryParts.COMMAND_SPECS (src/Chat/QueryParts.py);
    # keep aliases here in sync with that registry when adding new parameters.
    strategy: Optional[str] = None
    orchestrator_flow: Optional[str] = None
    confidence_evaluation: Optional[str] = None
    summary_length: Optional[str] = None
    summary_temperature: Optional[float] = None
    chunk_overlap: Optional[int] = None
    file: Optional[str] = None
    path: Optional[str] = None
    force_retrieve_mode: Optional[str] = None
    main_query_source: Optional[str] = None
    secondary_query_source: Optional[str] = None
    rerank_query_source: Optional[str] = None
    retrieval_language: Optional[str] = None  # english | native
    retriever_k: Optional[int] = None
    fetch_k: Optional[int] = None  # CLI alias for retriever_k
    rerank: Optional[bool] = None
    enable_keyword_action_splitter: Optional[bool] = None
    chroma_threshold: Optional[float] = None
    threshold: Optional[float] = None  # CLI alias for chroma_threshold
    final_chunks_to_llm: Optional[int] = None
    context_chunks: Optional[int] = None  # CLI alias for final_chunks_to_llm
    vector_weight: Optional[float] = None
    bm25_weight: Optional[float] = None
    graph_weight: Optional[float] = None
    regex_weight: Optional[float] = None

    use_chat_context: Optional[bool] = None
    chat_name: Optional[str] = None
    per_file_limit: Optional[int] = None
    file_cap: Optional[int] = None  # CLI alias for per_file_limit
    web_search: Optional[Union[bool, str]] = None
    web_weight: Optional[float] = None
    fetch_page_content: Optional[bool] = None
    web_fetch_k: Optional[int] = None
    debug_level: Optional[int] = None
    debug_mode: Optional[str] = (
        None  # "ge" (>=, default), "is" (== exact), or "le" (<= less-equal)
    )

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
        bool_strings: dict[str, bool] = {
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
                        if lowered in bool_strings:
                            data[field_name] = bool_strings[lowered]
                elif isinstance(raw, float) and annotation is int:
                    # OpenWebUI may send JSON floats (e.g. 100.0) for int fields
                    if field_name != "chunk_overlap" or raw.is_integer():
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


def _isFollowUpRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI follow-up suggestion prompt."""
    return OpenWebUIHousekeeping.classify(_getLastUserText(req)) == "follow_up"


def _isTagsRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI tag-generation prompt."""
    return OpenWebUIHousekeeping.classify(_getLastUserText(req)) == "tags"


def _isTitleRequest(req: ChatCompletionRequest) -> bool:
    """Return True if this request is an OpenWebUI title-generation prompt."""
    return OpenWebUIHousekeeping.classify(_getLastUserText(req)) == "title"


def _applyRequestToSession(
    req: ChatCompletionRequest, session: Session, queryParts: QueryParts, cfg: Config
) -> str:
    """Compatibility wrapper around the request-to-session mapper."""
    return RequestSessionMapper(queryParts, cfg).apply(req, session)


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------


def _streamGenerator(
    chatter: Chatter,
    session: Session,
    req_id: str,
    model: str,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    lock: asyncio.Lock,
    *,
    show_algo_results: bool = False,
    prompt_check_md: str = "",
    preamble: str = "",
    marked_docs_base_url: str = "",
) -> AsyncGenerator[str, None]:
    """Compatibility entry point for the response stream lifecycle."""
    return ChatResponseStreamer(
        chatter,
        session,
        req_id,
        model,
        loop,
        executor,
        lock,
        show_algo_results=show_algo_results,
        prompt_check_md=prompt_check_md,
        preamble=preamble,
        marked_docs_base_url=marked_docs_base_url,
    ).generate()


def _marked_docs_base_url(cfg: Config, request: Optional[Request]) -> str:
    """Resolve the externally-reachable base URL for /marked links.

    Priority: explicit config (``MARKED_DOCS_PUBLIC_BASE_URL``) >
    incoming request URL (proxy-aware via uvicorn's --proxy-headers) >
    bound host:port. Returns ``""`` if unresolvable.
    """
    public = (
        str((cfg.get_dict("_SERVE_DOCS", {}) or {}).get("public_base_url", "") or "")
        .strip()
        .rstrip("/")
    )
    if public:
        return public
    if request is not None:
        try:
            return str(request.base_url).rstrip("/")
        except Exception:
            pass
    host = cfg.get_str("_MODELS.ragchatservice._RAGCHATSERVICE.HOST", "127.0.0.1")
    port = cfg.get_int("_MODELS.ragchatservice._RAGCHATSERVICE.PORT", 11435)
    return f"http://{host}:{port}"


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
        elif k == "tools" and isinstance(v, list):
            tools_list = cast(List[Any], v)
            tool_names: list[str] = []
            for tool in tools_list:
                if not isinstance(tool, dict):
                    continue
                tool_dict: Dict[str, Any] = cast(Dict[str, Any], tool)
                fn = tool_dict.get("function")
                if not isinstance(fn, dict):
                    continue
                fn_dict: Dict[str, Any] = cast(Dict[str, Any], fn)
                name = fn_dict.get("name")
                if isinstance(name, str) and name.strip():
                    tool_names.append(name.strip())

            max_names = 12
            preview = tool_names[:max_names]
            cleaned[k] = {
                "count": len(tools_list),
                "names": preview,
                "truncated": len(tool_names) > max_names,
            }
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
) -> None:
    """Write a HUMAN_REVIEW CSV row, mirroring RAGChat._process_query logging."""
    status = "NOT_OK" if human_review else "OK"
    meta: Dict[str, Any] = {
        "Stage": stage,
        "Time": datetime.now(),
        "Status": status,
    }

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
    request: Optional[Request] = None,
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

    if OpenWebUIHousekeeping.is_terminal_navigation(_getLastUserText(req)):
        return _complianceResponse(
            f"ragchat-{uuid.uuid4().hex}",
            req.model,
            "Terminal settings and pickers are not available in OpenWebUI. "
            "Set file or path in Advanced Parameters to pin a document for chat "
            "and summarize:, or enter summarize: followed by a filename or path.",
            stream=req.stream,
        )

    await lock.acquire()
    lock_released = False

    # Performance logging for the entire request
    perf_logger = PerfLogger()
    perf_logger.log(
        "ChatCompletionHandler.handleRequest",
        "api",
        f"start endpoint model={req.model!r} messages={len(req.messages)} stream={req.stream}",
    )
    _t0_request = time.perf_counter()

    try:
        raw_dump = (
            req.model_dump()
        )  # capture before applyRequestToSession mutates session

        request_notice: str = _applyRequestToSession(req, session, queryParts, cfg)

        # Grounding needs the complete answer text. If grounding is active,
        # downgrade requested streaming to a buffered/non-streaming reply.
        grounding_forces_buffered_reply = bool(req.stream) and bool(
            getattr(session, "mark_text", False)
        )
        effective_stream = bool(req.stream) and not grounding_forces_buffered_reply

        # Effective debug level: per-request override (set in _applyRequestToSession)
        # takes precedence over the service startup value.
        pretty = PrettyWriter() if DebugHelper.check(cfg, 20) else None

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService incoming:",
                _summarise_request(raw_dump),
                color=CYAN,
            )
            if grounding_forces_buffered_reply:
                pretty.write(
                    "I",
                    "RAGChatService streaming:",
                    (
                        "Requested stream=True but switched to stream=False because grounding is enabled "
                        "(switches: SERVE_IN_MEMORY_DOCS_HTTP=1 for RAGChatService, mark_text for RAGChat.py)."
                    ),
                    color=YELLOW,
                )

        if DebugHelper.check(cfg, 30):
            PrettyWriter().write(
                "D",
                "RAGChatService session:",
                "Using QueryParts.print_values() session formatter",
                color=MAGENTA,
            )
            QueryParts().print_values(session=session)
            PrettyWriter().write(
                "D",
                "RAGChatService session:",
                f"chat_id={req.chat_id!r}",
                color=MAGENTA,
            )

        req_id = f"ragchat-{uuid.uuid4().hex}"

        try:
            rag.set_vector_store(session)
        except EmbedModelMismatch as exc:
            PrettyWriter().write(
                "W",
                "RAGChatService collection embedding mismatch:",
                f"{req_id}: {exc}",
                color=YELLOW,
            )
            return _complianceResponse(
                req_id,
                req.model,
                (
                    f"\u26a0\ufe0f Collection **{req.model}** is incompatible with the "
                    "active embedding model. In Open WebUI, select a collection built "
                    "with the active embedder from the model list, then retry. "
                    f"Reference: {req_id}"
                ),
                stream=req.stream,
            )
        except CollectionNotFoundError as exc:
            PrettyWriter().write(
                "E",
                "RAGChatService collection error:",
                f"{req_id}: {exc}",
                color=YELLOW,
            )
            return _complianceResponse(
                req_id,
                req.model,
                f"\u26a0\ufe0f Collection unavailable. Reference: {req_id}",
                stream=req.stream,
            )

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService vector store:",
                f"collection={session.collection_name!r}",
                color=CYAN,
            )

        # ----- compliance: Stage 1 – filter chain -----
        # Use only the raw user message (not session.query, which prepends the
        # operator-supplied system prompt).  The system prompt is trusted
        # operator content and must not be fed to the user-prompt banned-word
        # check, or it will cause false positives on legitimate operator
        # instructions that happen to contain controlled terms.
        _user_text_for_check = _getLastUserText(req) or session.query or ""
        file_selector = ServiceFileSelector(
            rag, [(message.role, message.text()) for message in req.messages]
        )
        selection = file_selector.resolve_reply(_user_text_for_check)
        command, summary_target = RAGChatImpl.parse_chat_command(_user_text_for_check)
        if selection is None and command == "summarize":
            selection = file_selector.select(session, summary_target)
        if selection is not None:
            if selection.stops_summary:
                return _complianceResponse(
                    req_id,
                    req.model,
                    request_notice + selection.message,
                    stream=req.stream,
                )
            _user_text_for_check = selection.summary_query(_user_text_for_check)
            command, _ = RAGChatImpl.parse_chat_command(_user_text_for_check)
        is_summary = command == "summarize"
        if is_summary:
            # Use the command itself, not the query with a system-prompt prefix.
            session.query = _user_text_for_check

        if not is_summary and DebugHelper.check(cfg, 31):
            _sys_msg = next(
                (m.text().strip() for m in req.messages if m.role == "system"), None
            )
            _user_preview = (
                _user_text_for_check[:200] + "…"
                if len(_user_text_for_check) > 200
                else _user_text_for_check
            )
            PrettyWriter().write(
                "D",
                "CheckPrompt input:",
                (
                    f"user_text={_user_preview!r}"
                    + (
                        f"\n  system_msg (not checked)={(_sys_msg[:200] + '…') if _sys_msg and len(_sys_msg) > 200 else _sys_msg!r}"
                        if _sys_msg
                        else "  [no system message]"
                    )
                ),
                color=CYAN,
            )

        show_algo_results = cfg.get_bool("SHOW_CLI_LIKE_ALGO_RESULTS", False)
        prompt_check_md = ""
        if not is_summary:
            stage = "PROMPT_CHECK"
            human_review, phrase_table = aiHelpers.check_user_prompt_with_filter_chain(
                _user_text_for_check, stage
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
                        "\n\n---\n### Filter chain algo results\n\n" + prompt_check_md
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
            )

            if guard_rejected:
                guard_msg = (
                    "I'm sorry, but I can't help with that query. "
                    "Please rephrase your question or ask about a different topic."
                )
                if show_algo_results:
                    guard_msg += "\n\n---\n### Filter chain algo results\n\n"
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

            # ----- compliance: Stage 3 – intent filter -----
            if_score, if_outcome, if_reasons = rag.intent_filter.score_query(
                session.query or "", path="service"
            )
            if if_outcome == "REFUSE":
                if_reason_str = ", ".join(if_reasons) if if_reasons else "intent score"
                if_msg = (
                    "I'm sorry, but I can't help with that query. "
                    "Please rephrase your question or ask about a different topic."
                )
                if show_algo_results:
                    if_msg += f"\n\n---\n\u26a0\ufe0f intent score {if_score} \u2014 {if_reason_str}"
                else:
                    if_msg += f"\n\n---\n**IntentFilter:** intent score {if_score} \u2014 {if_reason_str}"
                return _complianceResponse(
                    f"ragchat-{uuid.uuid4().hex}",
                    req.model,
                    if_msg,
                    stream=req.stream,
                )
            if if_outcome == "ALLOW_WITH_SAFETY_FRAMING" and pretty is not None:
                pretty.write(
                    "W",
                    "RAGChatService IntentFilter:",
                    f"Dual-use query detected \u2014 score={if_score}, reasons={if_reasons}. Proceeding.",
                    color=YELLOW,
                )

            # Append intent filter pass line to the algo-results block shown in OpenWebUI.
            if show_algo_results:
                if_emoji = "\u2705" if if_outcome == "ALLOW" else "\u26a0\ufe0f"
                if_reason_str = ", ".join(if_reasons) if if_reasons else ""
                if_detail = f" \u2014 {if_reason_str}" if if_reason_str else ""
                prompt_check_md += (
                    f"\n{if_emoji} **IntentFilter:** score {if_score}{if_detail}\n"
                )

        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService compliance:",
                (
                    "Summary command: prompt checks skipped; answer checks remain enabled."
                    if is_summary
                    else "User prompt passed all compliance stages."
                ),
                color=CYAN,
            )

        if effective_stream:
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
                    show_algo_results=show_algo_results,
                    prompt_check_md=prompt_check_md,
                    preamble=request_notice,
                    marked_docs_base_url=_marked_docs_base_url(cfg, request),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming path
        if pretty is not None:
            pretty.write(
                "D",
                "RAGChatService dispatch:",
                f"streaming=False  req_id={req_id!r}  (requested={bool(req.stream)})",
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
                # Pass a sink handler so Chatter keeps API-mode formatting semantics
                # (no CLI-only local file:// linkification).
                return chatter.run(
                    session,
                    apiChunkHandler=lambda _token: None,
                    is_streaming=False,
                )

            success, answer_text = await loop.run_in_executor(
                executor, _run_non_streaming
            )
        except LLMResultError as exc:
            PrettyWriter().write(
                "E",
                "RAGChatService LLM error:",
                f"{req_id}: {exc}",
                color=YELLOW,
            )
            return _complianceResponse(
                req_id,
                req.model,
                f"\u26a0\ufe0f Request could not be completed. Reference: {req_id}",
                stream=effective_stream,
            )

        if not success:
            indexed_query_error = session.indexed_query_error
            if indexed_query_error:
                return _complianceResponse(
                    req_id,
                    req.model,
                    request_notice + indexed_query_error,
                    stream=False,
                )
            compliance_msg = (
                "I'm sorry, but I can't provide an answer to that. "
                "The response did not pass the content compliance check."
            )
            if show_algo_results:
                answer_check_md = Accumulator().format_results_as_md("Answer Check")
                algo_block = "\n\n---\n### Filter chain algo results\n\n"
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
        # answer_text already has grounding applied (in Chatter.run())
        emit_cli_blocks: bool = (
            str(cfg.get_str("_FRIENDLY_NAME", "")).strip() == "RAGChatService"
        )

        algo_block = ""
        if show_algo_results:
            answer_check_md = Accumulator().format_results_as_md("Answer Check")
            algo_block = "\n\n---\n### Filter chain algo results\n\n"
            if prompt_check_md:
                algo_block += prompt_check_md + "\n\n"
            algo_block += answer_check_md

        # Marked documents are prepared in Chatter.run() via shared RAGChatImpl logic.

        _mb_base_url = _marked_docs_base_url(cfg, request)
        marked_block = register_marked_documents(
            session, get_marked_docs_store(), _mb_base_url
        )
        if marked_block:
            if emit_cli_blocks:
                chatter.print_llm_answer(marked_block, chatter.terminal_line_size)

        answer_out = request_notice + marked_block + (answer_text or "") + algo_block

        metadata_helper = getattr(rag, "helperInstance", None)
        metadata_block = (
            metadata_helper.build_document_metadata_md(
                getattr(session, "last_chosen_chunks", [])
            )
            if metadata_helper is not None
            else ""
        )
        if metadata_block:
            answer_out += metadata_block
            if emit_cli_blocks:
                chatter.print_llm_answer(metadata_block, chatter.terminal_line_size)

        # Log HTTP links to the server terminal so the admin can open them directly.
        url_map: dict[str, str] = getattr(session, "marked_docs_url_map", {})
        if url_map:
            from pathlib import Path as _Path

            for src_path, http_url in url_map.items():
                PrettyWriter().write(
                    "I",
                    "MarkedDoc",
                    f"{_Path(src_path).name}  →  {http_url}",
                    color=GREEN,
                )

        answer_out = SourcePathLinkifier.strip_inline_file_citation_links(
            answer_out,
            allowed_url_prefixes=(_mb_base_url,) if _mb_base_url else (),
        )
        answer_out = SourcePathLinkifier.linkify_source_paths_md(
            answer_out,
            allow_local_file_uri=False,
            strip_local_open_link_tail=True,
        )

        # Log successful completion
        elapsed_success = time.perf_counter() - _t0_request
        perf_logger.log(
            "ChatCompletionHandler.handleRequest",
            "api",
            f"stop  endpoint model={req.model!r} status=success elapsed={elapsed_success:.3f}s",
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

    except (OllamaNotRunning, VllmNotRunning) as exc:
        if not lock_released:
            lock.release()
            lock_released = True

        provider = getattr(exc, "provider", "backend")
        provider_name = provider.upper() if provider in {"ollama", "vllm"} else "LLM"
        err_id = f"ragchat-{uuid.uuid4().hex}"
        PrettyWriter().write(
            "E",
            "RAGChatService backend unavailable:",
            f"{err_id}: {exc}",
            color=YELLOW,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"{provider_name} backend unavailable. Reference: {err_id}"
            },
        )
    except HTTPException as exc:
        err_id = f"ragchat-{uuid.uuid4().hex}"
        PrettyWriter().write(
            "W",
            "RAGChatService parameter error:",
            f"{err_id}: {exc.detail}",
            color=YELLOW,
        )
        queryParts.print_values()
        try:
            return _complianceResponse(
                err_id,
                req.model,
                f"\u26a0\ufe0f Request parameters could not be processed. Reference: {err_id}",
                stream=req.stream,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Request could not be processed. Reference: {err_id}",
            ) from exc
    except Exception as exc:
        err_id = f"ragchat-{uuid.uuid4().hex}"
        PrettyWriter().write(
            "E",
            "RAGChatService request error:",
            f"{err_id}: {type(exc).__name__}: {exc}",
            color=YELLOW,
        )
        try:
            # Keep masked diagnostics in the server log; clients get a reference ID.
            return _complianceResponse(
                err_id,
                req.model,
                f"\u26a0\ufe0f Request could not be completed. Reference: {err_id}",
                stream=req.stream,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Request could not be completed. Reference: {err_id}",
            ) from exc
    finally:
        if not lock_released:
            lock.release()
