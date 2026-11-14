from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from Api.MarkedDocsService import register_marked_documents
from Api.MarkedDocsStore import get_default as get_marked_docs_store
from Chat.Chatter import Chatter
from Commons.Exceptions import OllamaNotRunning, VllmNotRunning
from Globals.Session import Session
from Gui.Colors import CYAN, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.Accumulator import Accumulator
from Helpers.SourcePathLinkifier import SourcePathLinkifier


def _json_str(text: str) -> str:
    """JSON-encode a plain string value (with quotes)."""
    return json.dumps(text)


class ChatResponseStreamer:
    """Own the worker task, SSE lifecycle, and transferred request lock."""

    def __init__(
        self,
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
    ) -> None:
        self.chatter = chatter
        self.session = session
        self.req_id = req_id
        self.model = model
        self.loop = loop
        self.executor = executor
        self.lock = lock
        self.show_algo_results = show_algo_results
        self.prompt_check_md = prompt_check_md
        self.preamble = preamble
        self.marked_docs_base_url = marked_docs_base_url

    async def generate(self) -> AsyncGenerator[str, None]:
        """
        Run the chat worker and emit OpenAI-compatible SSE chunks.

        The transferred lock remains held until this async generator finishes
        or is closed by a disconnected client.
        """
        chatter = self.chatter
        session = self.session
        req_id = self.req_id
        model = self.model
        loop = self.loop
        executor = self.executor
        lock = self.lock
        show_algo_results = self.show_algo_results
        prompt_check_md = self.prompt_check_md
        preamble = self.preamble
        marked_docs_base_url = self.marked_docs_base_url
        try:
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            created = int(time.time())
            emit_cli_blocks: bool = (
                str(chatter.cfg.get_str("_FRIENDLY_NAME", "")).strip()
                == "RAGChatService"
            )

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
                    if success is False and not getattr(
                        session, "indexed_query_error", None
                    ):
                        compliance_failed = True
                except (OllamaNotRunning, VllmNotRunning) as exc:
                    PrettyWriter().write(
                        "E",
                        "RAGChatService compliance error:",
                        f"{req_id}: {exc}",
                        color=YELLOW,
                    )
                    compliance_error = "backend unavailable"
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
                    f"[Compliance service unavailable. Reference: {req_id}]"
                )
                error_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{"content":{compliance_content}}},'
                    f'"finish_reason":"stop"}}]}}'
                )
                yield f"data: {error_chunk}\n\n"
            elif getattr(session, "indexed_query_error", None):
                yield (
                    f'data: {{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n'
                )
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
                    if emit_cli_blocks:
                        chatter.print_llm_answer(
                            marked_block, chatter.terminal_line_size
                        )
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
                    if emit_cli_blocks:
                        chatter.print_llm_answer(
                            metadata_block, chatter.terminal_line_size
                        )
                final_chunk = (
                    f'{{"id":"{req_id}","object":"chat.completion.chunk","created":{created},'
                    f'"model":"{model}","choices":[{{"index":0,"delta":{{}},'
                    f'"finish_reason":"stop"}}]}}'
                )
                yield f"data: {final_chunk}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            lock.release()
