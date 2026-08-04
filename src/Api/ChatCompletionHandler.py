import asyncio
import json
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
from Api.MarkedDocsService import register_marked_documents
from Api.MarkedDocsStore import get_default as get_marked_docs_store
from Chat.Chatter import Chatter
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Commons.Exceptions import (CollectionNotFoundError, LLMResultError,
                                OllamaNotRunning, VllmNotRunning)
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, MAGENTA, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.CSVWriter import CSVWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.PerfLogger import PerfLogger
from Helpers.SourcePathLinkifier import SourcePathLinkifier

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
    max_output_tokens: Optional[int] = None  # CLI alias for max_tokens
    num_ctx: Optional[int] = None
    context_size: Optional[int] = None  # CLI alias for num_ctx

    # RAG-LCC specific params — can be set via OpenWebUI model "Advanced Parameters"
    # Each has both the internal API name and the CLI-facing alias so users who read
    # the session dump and enter CLI names in OpenWebUI get proper validation.
    # CLI names are canonical in QueryParts.COMMAND_SPECS (src/Chat/QueryParts.py);
    # keep aliases here in sync with that registry when adding new parameters.
    strategy: Optional[str] = None
    retriever_k: Optional[int] = None
    fetch_k: Optional[int] = None  # CLI alias for retriever_k
    rerank: Optional[bool] = None
    chroma_threshold: Optional[float] = None
    threshold: Optional[float] = None  # CLI alias for chroma_threshold
    final_chunks_to_llm: Optional[int] = None
    context_chunks: Optional[int] = None  # CLI alias for final_chunks_to_llm
    vector_weight: Optional[float] = None

    use_chat_context: Optional[bool] = None
    chat_name: Optional[str] = None
    per_file_limit: Optional[int] = None
    file_cap: Optional[int] = None  # CLI alias for per_file_limit
    retrieve_mode: Optional[str] = (
        None  # VECTOR, BM25, GRAPH, VECTOR_GRAPH, BM25_GRAPH, ALL
    )
    web_search: Optional[Union[bool, str]] = None
    web_weight: Optional[float] = None
    fetch_page_content: Optional[bool] = None
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

    Note: OpenWebUI (and similar frontends) send the full conversation history
    in ``messages``.  We intentionally ignore all but the last user turn here —
    RAG-LCC maintains its own chat history via the persisted ``history/`` files,
    bounded by ``max_history_turns``.  The two histories are independent and do
    not interfere with each other.
    """
    system_text: str | None = None
    user_text: str = ""
    for msg in messages:
        if msg.role == "system" and system_text is None:
            system_text = msg.text().strip()
        elif msg.role == "user":
            user_text = msg.text()  # last user message wins; prior turns ignored
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


_ALLOWED_STRATEGIES: frozenset[str] = frozenset(
    {"ULTRA_WIDE", "WIDE", "BALANCED_FILE_CAP", "NARROW", "DEFAULT"}
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
) -> str:
    """Apply per-request overrides to *session*.

    Returns a (possibly empty) notice string that should be prepended to the
    response when a requested feature was silently suppressed — e.g. web
    search disabled or in dry-run mode.
    """
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
    # Treat empty/whitespace-only values the same as absent (OpenWebUI may send strategy="" when unset)
    if req.strategy is not None and req.strategy.strip():
        strategy_upper = req.strategy.strip().upper()
        if strategy_upper not in _ALLOWED_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy {req.strategy!r}. Allowed: {sorted(_ALLOWED_STRATEGIES)}",
            )
        session.strategy = strategy_upper
    strategy = (
        session.strategy or cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY") or "DEFAULT"
    )
    queryParts.applyStrategyDefaults(strategy, session=session)

    web_search_notice: str = ""

    def _applyOverrides() -> None:
        nonlocal web_search_notice
        # RAG-LCC param overrides (applied after strategy defaults so they take precedence)
        retriever_k = req.fetch_k if req.fetch_k is not None else req.retriever_k
        if retriever_k is not None:
            session.retriever_k = retriever_k
        if req.rerank is not None:
            session.rerank = req.rerank
        threshold = req.threshold if req.threshold is not None else req.chroma_threshold
        if threshold is not None:
            session.chroma_threshold = max(0.0, min(1.0, threshold))
        chunks = (
            req.context_chunks
            if req.context_chunks is not None
            else req.final_chunks_to_llm
        )
        if chunks is not None:
            session.final_chunks_to_llm = chunks
        if req.vector_weight is not None:
            session.vector_weight = max(0.0, min(1.0, req.vector_weight))
        if req.use_chat_context is not None:
            session.use_chat_context = req.use_chat_context
        # chat_name priority: explicit param > OpenWebUI chat_id > keep existing
        if req.chat_name is not None:
            session.chat_name = req.chat_name
        elif req.chat_id is not None:
            session.chat_name = req.chat_id
        file_cap = req.file_cap if req.file_cap is not None else req.per_file_limit
        if file_cap is not None:
            session.per_file_limit = file_cap
        if req.retrieve_mode is not None:
            sm = req.retrieve_mode.upper()
            allowed_modes: list[str] = cfg.get_list("_ALLOWED_RETRIEVE_MODES")
            if sm in allowed_modes:
                session.retrieve_mode = sm
        if req.web_search is not None:
            # Normalize to the tri-state string used by QueryParts.
            # Bool values come from JSON true/false; strings come from OpenWebUI
            # Advanced Parameters (web_search=local_and_web / web_search=web_only).
            ws_raw = req.web_search
            if isinstance(ws_raw, bool):
                ws_raw = "local_and_web" if ws_raw else "local_only"
            else:
                # OpenWebUI custom params may include surrounding quotes,
                # e.g. web_search='local_and_web'. Normalize both styles.
                ws_raw = str(ws_raw).strip().strip("\"'").lower()
                if ws_raw in ("true", "1"):
                    ws_raw = "local_and_web"
                elif ws_raw in ("false", "0"):
                    ws_raw = "local_only"
            if ws_raw in ("local_and_web", "web_only"):
                web_mode: str = (
                    str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
                )
                if web_mode != "1":
                    web_search_notice = (
                        "\U0001f6ab **Web search unavailable** \u2014 "
                        "disabled by the administrator "
                        '(`WEB_SEARCH_MODE != "1"` in `Config_Internet_Env.py`). '
                        "Answering from the local knowledge base only.\n\n---\n\n"
                    )
                    session.web_search = False
                else:
                    session.web_search = True
                    if ws_raw == "web_only":
                        session.retrieve_mode = "WEB"
            else:  # "local_only" or unrecognised value
                session.web_search = False
                if getattr(session, "retrieve_mode", None) == "WEB":
                    # Keep session state coherent: if web search is off, WEB-only
                    # mode must be replaced with a local retrieval mode.
                    allowed_rm: list[str] = cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                    session.retrieve_mode = (
                        "ALL" if "ALL" in allowed_rm else allowed_rm[0]
                    )
        elif (
            cfg.get_bool("_OPENWEB_UI_WEBSEARCH", False)
            and str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower() == "1"
        ):
            # No explicit per-request value — apply the service-level default.
            session.web_search = True
        if req.web_weight is not None:
            session.web_weight = max(0.0, req.web_weight)
        if req.fetch_page_content is not None:
            session.fetch_page_content = req.fetch_page_content
        # Enable visual markers (yellow highlighting) and HTTP serving only when enabled.
        # Answer grounding (orange) works independently via chunk_texts_for_grounding.
        session.mark_text = os.environ.get("SERVE_IN_MEMORY_DOCS_HTTP", "0") == "1"
        if req.debug_level is not None or req.debug_mode is not None:
            level = (
                req.debug_level
                if req.debug_level is not None
                else session.debug_level or 0
            )
            mode = (
                req.debug_mode.strip().lower() if req.debug_mode is not None else None
            ) or "ge"
            if mode not in ("ge", "is", "le"):
                mode = "ge"
            session.debug_level = level
            session.debug_mode = mode
            combined = "none" if level == 0 else f"{mode} {level}"
            cfg.set("DEBUG_LEVEL", combined)

        # Per-request LLM param overrides
        if req.temperature is not None:
            session.temperature = req.temperature
        if req.top_p is not None:
            session.top_p = req.top_p
        if req.top_k is not None:
            session.top_k = float(req.top_k)
        max_out = (
            req.max_output_tokens
            if req.max_output_tokens is not None
            else req.max_tokens
        )
        if max_out is not None:
            session.max_output_tokens_override = max_out
        ctx = req.context_size if req.context_size is not None else req.num_ctx
        if ctx is not None:
            session.context_size_override = ctx

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
    session.base_kwargs = {"k": session.retriever_k}
    return web_search_notice


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
    *,
    show_algo_results: bool = False,
    prompt_check_md: str = "",
    preamble: str = "",
    marked_docs_base_url: str = "",
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

        if preamble:
            preamble_chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(preamble)}}},'
                f'"finish_reason":null}}]}}'
            )
            yield f"data: {preamble_chunk}\n\n"

        compliance_failed = False
        compliance_error: str = ""  # set if OllamaNotRunning is raised mid-stream

        async def run_in_thread() -> None:
            nonlocal compliance_failed, compliance_error
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
            except (OllamaNotRunning, VllmNotRunning) as exc:
                compliance_error = str(exc)
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
            token = SourcePathLinkifier.linkify_source_paths_md(
                token,
                allow_local_file_uri=False,
                strip_local_open_link_tail=True,
            )
            chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(token)}}},'
                f'"finish_reason":null}}]}}'
            )
            yield f"data: {chunk}\n\n"

        if compliance_error:
            compliance_content = _json_str(
                f"[Compliance service unavailable \u2014 {compliance_error}]"
            )
            error_chunk = (
                f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{compliance_content}}},'
                f'"finish_reason":"stop"}}]}}'
            )
            yield f"data: {error_chunk}\n\n"
        elif compliance_failed:
            algo_suffix = ""
            if show_algo_results:
                answer_check_md = Accumulator().format_results_as_md("Answer Check")
                algo_suffix = "\n\n---\n### Filter chain algo results\n\n"
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
                algo_block = "\n\n---\n### Filter chain algo results\n\n"
                if prompt_check_md:
                    algo_block += prompt_check_md + "\n\n"
                algo_block += answer_check_md
                algo_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(algo_block)}}},'
                    f'"finish_reason":null}}]}}'
                )
                yield f"data: {algo_chunk}\n\n"
            marked_block = register_marked_documents(
                session, get_marked_docs_store(), marked_docs_base_url
            )
            if marked_block:
                marked_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(marked_block)}}},'
                    f'"finish_reason":null}}]}}'
                )
                yield f"data: {marked_chunk}\n\n"
            metadata_helper = getattr(chatter, "helpers", None)
            metadata_block = (
                metadata_helper.build_document_metadata_md(
                    getattr(session, "last_chosen_chunks", [])
                )
                if metadata_helper is not None
                else ""
            )
            if metadata_block:
                metadata_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{_json_str(metadata_block)}}},'
                    f'"finish_reason":null}}]}}'
                )
                yield f"data: {metadata_chunk}\n\n"
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
        else:
            cleaned[k] = v

    return json.dumps(cleaned, ensure_ascii=False)


def _format_session_for_error(
    session: Session, req: Optional["ChatCompletionRequest"] = None
) -> str:
    """Return a grouped markdown code block of all session values, appended to parameter-error messages.

    When *req* is supplied the Output sub-section shows the effective per-request
    value (e.g. context_size set via OpenWebUI Advanced Parameters) rather than
    the persistent-session value, which is always None for API-only overrides.
    """
    s = session

    lbl = 14  # label column width — matches print_values() in QueryParts

    def _kv(*pairs: tuple[str, Any]) -> str:
        return "  ".join(f"{k}={v!r}" for k, v in pairs)

    def _line(label: str, *pairs: tuple[str, Any]) -> str:
        return f"  \u25b6 {label + ':':<{lbl}}{_kv(*pairs)}"

    def _sub(label: str, *pairs: tuple[str, Any]) -> str:
        return f"    \u25b6 {label + ':':<{lbl}}{_kv(*pairs)}"

    web_val = (
        "web_only"
        if getattr(s, "retrieve_mode", None) == "WEB"
        else ("local_and_web" if getattr(s, "web_search", False) else "local_only")
    )
    fp_val = (
        "fetch pages" if getattr(s, "fetch_page_content", False) else "snippets only"
    )
    cfg_local = Config()
    bm25_pf = cfg_local.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
    cos_pf = cfg_local.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0

    mot_val = str(getattr(s, "max_output_tokens", "auto"))
    if req is not None:
        req_mot = (
            req.max_output_tokens
            if req.max_output_tokens is not None
            else req.max_tokens
        )
        if req_mot is not None:
            mot_val = str(req_mot)
    elif getattr(s, "max_output_tokens_override", None) is not None:
        mot_val += f"  [override={s.max_output_tokens_override}]"

    if req is not None:
        req_ctx = req.context_size if req.context_size is not None else req.num_ctx
    else:
        req_ctx = None
    ctx_val = (
        str(req_ctx)
        if req_ctx is not None
        else (
            f"override={s.context_size_override}"
            if getattr(s, "context_size_override", None) is not None
            else "auto"
        )
    )

    output_lines = [
        "\n\n---\n**Active session values:**\n```",
        _line(
            "Debug",
            ("debug_level", getattr(s, "debug_level", None)),
            ("debug_mode", getattr(s, "debug_mode", None)),
        ),
        _line(
            "Chat Context",
            ("use_chat_context", getattr(s, "use_chat_context", None)),
            ("history_keep", getattr(s, "turns", None)),
            ("history_prune", getattr(s, "prune_batch", None)),
            ("rewrite_context", getattr(s, "max_history_turns", None)),
            ("topic_summary", getattr(s, "topic_summary_mode", None)),
        ),
        _line(
            "Talk with",
            ("collection", getattr(s, "collection_name", None)),
            ("chat_name", getattr(s, "chat_name", None)),
        ),
        _line(
            "File Input",
            ("file", getattr(s, "file_name", None)),
            ("path", getattr(s, "file_path", None)),
            ("file_cap", getattr(s, "per_file_limit", None)),
        ),
        "  \u25b6 Retrieval:",
        _sub(
            "Strategies",
            ("strategy", getattr(s, "strategy", None)),
            ("retrieve_mode", getattr(s, "retrieve_mode", None)),
            ("rerank", getattr(s, "rerank", None)),
            ("threshold", getattr(s, "chroma_threshold", None)),
        ),
        _sub(
            "Weights",
            ("vector_weight", getattr(s, "vector_weight", None)),
            ("bm25_weight", getattr(s, "bm25_weight", None)),
            ("graph_weight", getattr(s, "graph_weight", None)),
        ),
        _sub(
            "Web",
            ("web_search", web_val),
            ("web_weight", getattr(s, "web_weight", None)),
            ("fetch_page_content", fp_val),
            ("bm25_pre_filter", bm25_pf),
            ("cosine_pre_filter", cos_pf),
        ),
        _line("Visual", ("mark_text", getattr(s, "mark_text", False))),
        _sub(
            "Chunk takes",
            ("fetch_k", getattr(s, "retriever_k", None)),
            ("context_chunks", getattr(s, "final_chunks_to_llm", None)),
        ),
        _sub(
            "LLM",
            ("temperature", getattr(s, "temperature", None)),
            ("top_p", getattr(s, "top_p", None)),
            ("top_k", getattr(s, "top_k", None)),
        ),
        _sub(
            "Output",
            ("max_output_tokens", mot_val),
            ("context_size", ctx_val),
            ("terminal_line_size", getattr(s, "terminal_line_size", None)),
        ),
        "```",
    ]
    return "\n".join(output_lines)


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

        web_search_notice: str = _applyRequestToSession(req, session, queryParts, cfg)

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
            ws_log = (
                "web_only"
                if getattr(session, "retrieve_mode", None) == "WEB"
                else ("local_and_web" if session.web_search else "local_only")
            )
            bm25_pf = cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
            cos_pf = cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
            web_rthr = cfg.get_float("_WEB_SEARCH.rerank_threshold") or 0.0
            PrettyWriter().write(
                "D",
                "RAGChatService session:",
                (
                    f"collection={session.collection_name!r}  strategy={session.strategy!r}  "
                    f"temperature={session.temperature}  top_k={session.top_k}  top_p={session.top_p}\n"
                    f"retriever_k={session.retriever_k}  chroma_threshold={session.chroma_threshold}  "
                    f"final_chunks_to_llm={session.final_chunks_to_llm}  "
                    f"use_chat_context={session.use_chat_context}  "
                    f"turns={session.turns}  "
                    f"chat_name={session.chat_name!r}  "
                    f"chat_id={req.chat_id!r}  "
                    f"per_file_limit={session.per_file_limit}\n"
                    f"prune_batch={session.prune_batch}  "
                    f"max_history_turns={session.max_history_turns}  "
                    f"topic_summary_mode={session.topic_summary_mode}  "
                    f"retrieve_mode={session.retrieve_mode}  "
                    f"rerank={session.rerank}  "
                    f"debug_level={session.debug_level}  "
                    f"debug_mode={getattr(session, 'debug_mode', 'ge') or 'ge'}  "
                    f"web_search={ws_log}  "
                    f"bm25_pre_filter={bm25_pf}  cosine_pre_filter={cos_pf}  web_rerank_threshold={web_rthr}\n"
                    f"fetch_page_content={session.fetch_page_content}  "
                    f"mark_text={getattr(session, 'mark_text', False)}  "
                    f"max_output_tokens(api: max_tokens)={session.max_output_tokens}  "
                    f"max_output_tokens_override={session.max_output_tokens_override}  "
                    f"context_size_override(api: num_ctx)={session.context_size_override}\n"
                    f"extraOllamaOptions={session.extraOllamaOptions}  "
                    f"ollamaTopLevelParams={session.ollamaTopLevelParams}"
                ),
                color=MAGENTA,
            )

        req_id = f"ragchat-{uuid.uuid4().hex}"

        try:
            rag.set_vector_store(session)
        except CollectionNotFoundError as exc:
            return _complianceResponse(
                req_id,
                req.model,
                f"\u26a0\ufe0f **RAGChatService error**\n\n{exc}",
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
        stage = "PROMPT_CHECK"
        _user_text_for_check = _getLastUserText(req) or session.query or ""

        if DebugHelper.check(cfg, 31):
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
            session=session,
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
                "User prompt passed all compliance stages.",
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
                    preamble=web_search_notice,
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
            return _complianceResponse(
                req_id,
                req.model,
                f"\u26a0\ufe0f **RAGChatService error**\n\n{exc}",
                stream=effective_stream,
            )

        if not success:
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
        answer_out = web_search_notice + (answer_text or "")

        if show_algo_results:
            answer_check_md = Accumulator().format_results_as_md("Answer Check")
            algo_block = "\n\n---\n### Filter chain algo results\n\n"
            if prompt_check_md:
                algo_block += prompt_check_md + "\n\n"
            algo_block += answer_check_md
            answer_out += algo_block

        # Marked documents are prepared in Chatter.run() via shared RAGChatImpl logic.

        _mb_base_url = _marked_docs_base_url(cfg, request)
        marked_block = register_marked_documents(
            session, get_marked_docs_store(), _mb_base_url
        )
        if marked_block:
            answer_out += marked_block

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

        # Log HTTP links to the server terminal so the admin can open them directly.
        url_map: dict[str, str] = getattr(session, "marked_docs_url_map", {})
        if url_map:
            from pathlib import Path as _Path

            for src_path, http_url in url_map.items():
                PrettyWriter().write(
                    "I",
                    "MarkedDoc",
                    f"{_Path(src_path).name}  →  {http_url}",
                    color=CYAN,
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
        detail = (
            f"{provider.upper()} backend unavailable — {exc}"
            if provider in {"ollama", "vllm"}
            else f"LLM backend unavailable — {exc}"
        )
        return JSONResponse(
            status_code=503,
            content={"detail": detail},
        )
    except HTTPException as exc:
        PrettyWriter().write(
            "W",
            "RAGChatService parameter error:",
            str(exc.detail),
            color=YELLOW,
        )
        queryParts.print_values()
        err_id = f"ragchat-{uuid.uuid4().hex}"
        try:
            return _complianceResponse(
                err_id,
                req.model,
                f"\u26a0\ufe0f **Parameter error** \u2014 {exc.detail}{_format_session_for_error(queryParts.session, req)}",
                stream=req.stream,
            )
        except Exception:
            raise HTTPException(status_code=500, detail=str(exc.detail)) from exc
    except Exception as exc:
        err_id = f"ragchat-{uuid.uuid4().hex}"
        try:
            return _complianceResponse(
                err_id,
                req.model,
                f"\u26a0\ufe0f **RAGChatService error**\n\n{exc}",
                stream=req.stream,
            )
        except Exception:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not lock_released:
            lock.release()
