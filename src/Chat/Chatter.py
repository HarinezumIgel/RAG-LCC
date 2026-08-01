import logging
import os
import re
import textwrap
from datetime import datetime
from typing import Any, Callable, Dict

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _wrap_preserving_ansi(text: str, width: int) -> list[str]:
    """Wrap text at word boundaries using ANSI-stripped width, then carry
    active ANSI codes across line breaks so background colour is not lost
    on terminals that reset background at newlines."""
    stripped = _ANSI_RE.sub("", text)
    wrapped_stripped = textwrap.wrap(
        stripped, width=width, break_long_words=False, break_on_hyphens=False
    )
    if len(wrapped_stripped) <= 1:
        return [text]

    result: list[str] = []
    pending: list[str] = []  # ANSI codes active at the start of the current line
    orig_pos = 0

    for line_idx, stripped_line in enumerate(wrapped_stripped):
        visible_needed = len(stripped_line)
        line_start = orig_pos
        visible_seen = 0
        line_codes: list[str] = list(pending)  # inherit from previous line

        while visible_seen < visible_needed and orig_pos < len(text):
            m = _ANSI_RE.match(text, orig_pos)
            if m:
                code = m.group(0)
                orig_pos = m.end()
                if code == "\033[0m":
                    line_codes.clear()
                else:
                    line_codes.append(code)
            else:
                orig_pos += 1
                visible_seen += 1

        # Skip the inter-word space that textwrap consumed at the wrap point
        while orig_pos < len(text) and text[orig_pos] == " ":
            orig_pos += 1

        line_text = text[line_start:orig_pos].rstrip()

        # Re-open inherited codes at the start of continuation lines
        if line_idx > 0 and pending:
            line_text = "".join(pending) + line_text

        # Close any still-open codes (many terminals reset background at newlines)
        if line_codes:
            line_text += "\033[0m"

        result.append(line_text)
        pending = list(line_codes)

    return result


from langchain_core.prompts import PromptTemplate

from AI.AIHelpers import AIHelpers
from AI.LLMCaller import LLMCaller
from AI.ModelOutputAdapter import ModelOutput, ModelOutputAdapter
from AI.ModelsCache import ModelsCache
from AI.TensorHelpers import TensorHelpers
from AI.TokenBudget import TokenBudget
from Algos.Masker import Masker
from Chat.ChatContext import ChatContext
from Chat.CommandProcessor import CommandProcessor
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Commons.Exceptions import LLMResultError
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Config.Config import Config
from Globals.Globals import Globals
from Globals.Session import Session
from Gui.Colors import CYAN, ORANGE, RED, RESET
from Gui.PrettyWriter import PrettyWriter
from Helpers.CSVWriter import CSVWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.SourcePathLinkifier import SourcePathLinkifier


class Chatter:
    def __init__(self) -> None:
        # your core components
        self.globalsInstance: Globals = Globals()
        self.helpers: Helpers = Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        self.logger: logging.Logger = self.helpers.setup_logger("RAGChat")
        self.globalsInstance.set_logger(self.logger)
        self.pretty: PrettyWriter = PrettyWriter()
        self.queryParts: QueryParts = QueryParts()
        self.cfg: Config = Config()
        self.csvWriter: CSVWriter = CSVWriter()
        self.bannedPhraseCollector: BannedPhraseCollector = BannedPhraseCollector()
        self.llm_model: str = self.helpers.get_model_args("_ACTIVE_LLM")["MODEL"]
        endpoint_args = self.helpers.get_active_endpoint_args()
        self.is_streaming: bool = endpoint_args.get("STREAMING_REQ", False)
        self.use_ollama_gpu: bool = endpoint_args.get("USE_GPU", True)
        self.prompt: str
        self.prompt_name: str | None
        # Indirect call
        prompt_var: str = self.helpers.get_model_args("_ACTIVE_LLM")["PROMPT_CHAT"]
        self.prompt, self.prompt_name = self.cfg.indirect_get(prompt_var)

        self.cp: CommandProcessor = CommandProcessor()
        self.rag: RAGChatImpl = RAGChatImpl()
        self.chatContext: ChatContext = ChatContext()
        self.llmCaller: LLMCaller = LLMCaller()
        self.modelOutputAdapter: ModelOutputAdapter = ModelOutputAdapter()
        self.masker: Masker = Masker()
        self.tokenBudget: TokenBudget = TokenBudget()

    def run(
        self,
        session: Session,
        *,
        apiChunkHandler: Callable[[str], None] | None = None,
        is_streaming: bool | None = None,
    ) -> tuple[bool, str | None]:
        """Run the RAG pipeline for *session* and return ``(success, answer)``.

        Parameters
        ----------
        session:
            Per-request state (collection, query, strategy, …).
        apiChunkHandler:
            Optional callback that receives streamed content tokens.
        is_streaming:
            Override the config default for ``_MODELS.ollama._OLLAMA.STREAMING_REQ``.
            When *None* the config value is used.
        """
        streaming = self.is_streaming if is_streaming is None else is_streaming

        missing = [
            name
            for name, val in (
                ("fetch_k", session.retriever_k),
                ("max_output_tokens", session.max_output_tokens),
                ("temperature", session.temperature),
                ("top_k", session.top_k),
                ("top_p", session.top_p),
                ("query", session.query),
            )
            if val is None
        ]
        if missing:
            raise ValueError(f"Session fields not set: {', '.join(missing)}")

        # Pyright can't narrow through the dynamic loop above — bind non-None locals.
        temperature: float = session.temperature  # type: ignore[reportAssignmentType]
        top_k: float = session.top_k  # type: ignore[reportAssignmentType]
        top_p: float = session.top_p  # type: ignore[reportAssignmentType]
        query: str = session.query  # type: ignore[reportAssignmentType]

        self.pretty.write("I", "", f"Chatter RAG Query LMM: {self.llm_model}")

        prompt = PromptTemplate.from_template(self.prompt)
        context, length = self.rag.retrieve(session)
        query_notice: str = self._build_query_notice(session, query)

        if length == 0:
            return self._handle_no_results(
                session, query, query_notice, apiChunkHandler
            )

        # Resolve the prompt with actual values
        formatted = prompt.format(context=context, input=session.query)
        effective_ctx, resolved_output = self._resolve_token_params(session, formatted)
        session.max_output_tokens = resolved_output

        self.pretty.write(
            "I",
            "Resolved token params",
            f"max_output_tokens(api: max_tokens)={resolved_output}  "
            f"num_ctx={effective_ctx}  "
            f"(override: max_tokens={session.max_output_tokens_override}  "
            f"num_ctx={session.context_size_override})",
        )

        # Build the unified Ollama options dict: session params + any extra options from API
        ollama_options: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": session.max_output_tokens,
            "num_ctx": effective_ctx,
        }
        if not self.use_ollama_gpu:
            ollama_options["num_gpu"] = 0
        if session.extraOllamaOptions:
            ollama_options.update(session.extraOllamaOptions)

        will_apply_grounding = apiChunkHandler is not None and getattr(
            session, "mark_text", False
        )

        # Build chunk handler — wraps real-time streaming to API client when grounding is off
        handler = self.llmCaller.make_on_chunk(ollama_options)
        if apiChunkHandler is not None and is_streaming and not will_apply_grounding:
            base_handler: Callable[[Dict[str, str]], None] = handler
            api_fn: Callable[[str], None] = apiChunkHandler

            def wrapped(
                chunk: Dict[str, str],
                bh: Callable[[Dict[str, str]], None] = base_handler,
                af: Callable[[str], None] = api_fn,
            ) -> None:
                bh(chunk)
                af(chunk.get("content", ""))

            handler = wrapped

        # Emit pre-stream notices to API client
        web_warning: str = ""
        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        if getattr(session, "web_search", False) and web_mode == "1":
            web_warning = (
                "\u26a0\ufe0f **Internet access is enabled** "
                "\u2014 this answer may include live web search results.\n\n---\n\n"
            )
            if apiChunkHandler is not None:
                apiChunkHandler(web_warning)
        if query_notice and apiChunkHandler is not None:
            apiChunkHandler(query_notice)

        if will_apply_grounding and not streaming:
            self.pretty.write(
                "I",
                "Call LLM streaming:",
                "False \u2014 document grounding active (mark_text=True, streaming downgraded)",
                color=ORANGE,
            )

        llm_result = self.llmCaller.call_llm(
            self.llm_model,
            formatted,
            ollama_options,
            answer_is_json=True,
            template_name=self.prompt_name,
            on_chunk=handler,
            streaming=streaming,
            stage="Run user prompt",
            top_level_params=session.ollamaTopLevelParams,
        )

        if isinstance(llm_result, dict) and "error" in llm_result:  # type: ignore[reportUnnecessaryIsInstance]
            msg = f"LLM error: {llm_result['error']}"
            self.pretty.write("E", "LLM", msg, color=RED)
            raise LLMResultError(msg)

        answer: ModelOutput = self.modelOutputAdapter.interpret(
            llm_result,
            self.llm_model,
            is_compliance=False,
            is_streaming=streaming,
        )
        content: str = answer.content or ""

        if self._check_answer_compliance(session, answer, content):
            return False, None

        if session.use_chat_context:
            self.chatContext.add_chat_turn(session, query, content)

        # Post-process answer content
        answer.content = self.masker.mask(content)
        if query_notice:
            answer.content = query_notice + answer.content
        if web_warning and apiChunkHandler is not None:
            answer.content = web_warning + answer.content
        if apiChunkHandler is None:
            answer.content = SourcePathLinkifier.linkify_source_paths_md(
                answer.content, allow_local_file_uri=True
            )

        # Apply answer grounding (highlight sentences traceable to source chunks)
        mark_text_enabled = getattr(session, "mark_text", False)
        cli_answer = answer.content
        if mark_text_enabled:
            chunk_texts: list[str] = list(
                getattr(session, "chunk_texts_for_grounding", []) or []
            )
            if chunk_texts:
                cli_answer = self._apply_answer_grounding(answer, chunk_texts)

        # Deliver answer: send buffered content to API client; always print CLI
        if apiChunkHandler is not None and (will_apply_grounding or not is_streaming):
            apiChunkHandler(answer.content)
        self.print_llm_answer(cli_answer, self.terminal_line_size)

        # Mark source documents with grounding highlights and show to user
        session.last_answer_content = answer.content
        if mark_text_enabled:
            chosen = getattr(session, "last_chosen_chunks", [])
            if chosen:
                try:
                    self.rag._mark_sources(session, chosen)
                except Exception as exc:
                    self.pretty.write(
                        "W", "VisualMarker", f"Visual marking failed: {exc}"
                    )

        if apiChunkHandler is None:
            chosen = getattr(session, "last_chosen_chunks", [])
            if mark_text_enabled:
                marked: list[tuple[str, bytes]] = list(
                    getattr(session, "marked_documents", []) or []
                )
                if marked:
                    self._open_marked_documents(marked)
                if chosen:
                    self._show_web_sources(chosen)
            else:
                if chosen:
                    self._show_original_sources(chosen)

        return True, answer.content

    # ------------------------------------------------------------------
    # Private helpers extracted from run()
    # ------------------------------------------------------------------

    def _build_query_notice(self, session: Session, query: str) -> str:
        """Return a Markdown notice when the effective query differs from the user's query."""
        effective_q: str | None = getattr(session, "effective_query", None)
        if not effective_q or not query or effective_q.strip() == query.strip():
            return ""
        reason = getattr(session, "effective_query_reason", None) or "changed"
        notice_labels: dict[str, str] = {
            "translated": "Translated query",
            "rewritten": "Rewritten query",
            "translated+rewritten": "Translated & rewritten query",
        }
        label = notice_labels.get(reason, "Query (changed)")
        return f'\U0001f50d *{label}: "{effective_q}"*\n\n---\n\n'

    def _handle_no_results(
        self,
        session: Session,
        query: str,
        query_notice: str,
        apiChunkHandler: Callable[[str], None] | None,
    ) -> tuple[bool, str | None]:
        """Handle the case where retrieval returned no chunks."""
        if session.clarification_response:
            clarification = "\u2754  " + session.clarification_response
            session.clarification_response = None
            if query_notice and apiChunkHandler is not None:
                apiChunkHandler(query_notice)
            if apiChunkHandler is not None:
                apiChunkHandler(clarification)
            msg = (query_notice + clarification) if query_notice else clarification
            self.print_llm_answer(msg, self.terminal_line_size, color=CYAN)
            return True, msg

        friendly_name = self.cfg.get_str("_FRIENDLY_NAME")
        strats_raw = self.cfg.get_dict("_STRATEGIES")
        strats_list: str = (
            ", ".join(sorted(strats_raw.keys()))
            if strats_raw
            else "ULTRA_WIDE, WIDE, BALANCED_FILE_CAP, NARROW, DEFAULT"
        )
        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        show_web_hint: bool = web_mode == "1"

        no_results_msg = (
            "I couldn't find relevant information to answer your query in the provided context.\n\n"
            "Metadata used: None (context is empty)"
        )
        if friendly_name == "RAGChatService":
            no_results_msg += (
                "\n\n**Hints (OpenWebUI Controls menu):**\n"
                "- Increase the `retriever_k` slider (top_k) to retrieve more candidate chunks.\n"
                "- Set `chroma_threshold` to a lower value (e.g. **0.2**) so fewer chunks are filtered out.\n"
                f"- Try a different `strategy` (current: "
                f"{session.strategy or 'default'}). "
                f"Allowed: {strats_list}."
            )
            if show_web_hint:
                no_results_msg += (
                    "\n- Open **Controls** (sliders on top right corner) and look for **+ Add Custom Parameter** at the right bottom"
                    "Overwrite **custom_pram_name** with **web_search** and **custom_param_value** with True "
                    "to search the internet for an answer."
                )
        elif friendly_name == "RAGChat":
            no_results_msg += (
                "\n\nHints:\n"
                "- Use   retriever_k  to increase top_k and retrieve more candidate chunks.\n"
                "- Use   threshold  to set a lower value (e.g. 0.2) so fewer chunks are filtered out.\n"
                f"- Use   strategy  to switch retrieval strategy (current: "
                f"{session.strategy or 'default'}).\n"
                f"        Allowed: {strats_list}.\n"
                "- Type  help?  at the command prompt for all available commands."
            )
            if show_web_hint:
                no_results_msg += "\n- Use   web_search=local_and_web to enable web search (provided your admin allows it)."

        if not query_notice and query:
            query_notice = f'\U0001f50d *Query: "{query}"*\n\n---\n\n'
        if query_notice and apiChunkHandler is not None:
            apiChunkHandler(query_notice)
        if apiChunkHandler is not None:
            apiChunkHandler(no_results_msg)
        full_no_results = (
            (query_notice + no_results_msg) if query_notice else no_results_msg
        )
        self.print_llm_answer(full_no_results, self.terminal_line_size)
        return True, full_no_results

    def _check_answer_compliance(
        self,
        session: Session,
        answer: ModelOutput,
        content: str,
    ) -> bool:
        """Run compliance checks on the LLM answer. Returns True if the answer should be rejected."""
        language: str = self.fileUtils.get_text_language(content, "ntlk")
        embedder: Any = self.models_cache.get_hf_embeddings()
        embeddings: list[Any] = embedder.embed_documents([answer.content])
        stage: str = "PIPELINE_CHECK"
        emb_tensor: Any = self.tensorHelpers.to_tensor(embeddings[0])
        human_review, _, phrase_table = self.aiHelpers.run_ensemble_checks(
            content,
            language,
            stage=stage,
            accumulate=False,
            require_keybert=False,
            embedding=emb_tensor,
        )
        status = "NOT_OK" if human_review is True else "OK"
        doc: Dict[str, Dict[str, Any]] = {
            "meta": {
                "Stage": stage,
                "Session": session.export_session_state_as_cell(),
                "Time": datetime.now(),
                "Status": status,
            }
        }
        self.csvWriter.write_json2csv(
            self.bannedPhraseCollector.prepare_for_csv_print(phrase_table, doc),
            "HUMAN_REVIEW",
        )
        if human_review:
            self.pretty.write(
                "E", "Answer check", "\u26a0\ufe0f    Answer compliance check failed."
            )
        return bool(human_review)

    def _apply_answer_grounding(
        self,
        answer: ModelOutput,
        chunk_texts: list[str],
    ) -> str:
        """Apply grounding highlights to *answer.content* and return the CLI display version.

        Always returns an ANSI-coloured string for the server terminal (both RAGChat and
        RAGChatService).  *answer.content* is never modified — the coloured version is
        only used for the local terminal print, never sent to API clients.
        """
        from VisualMarkers.AnswerGrounder import AnswerGrounder

        ansi = self.cfg.get_str("_MARKED_DOCS_COLORS.answer_ansi") or ""
        grounder = AnswerGrounder()
        return grounder.ground_answer_cli(
            answer.content or "",
            chunk_texts,
            ansi_codes=ansi,
        )

    def _resolve_token_params(
        self, session: Session, formatted: str
    ) -> tuple[int, int]:
        """Resolve effective context window and output-token budget, applying user overrides.

        Returns:
            (effective_ctx, resolved_max_output_tokens) — both guaranteed to be safe values.
        """
        # --- context window ---
        auto_ctx: int = self.tokenBudget.get_context_limit(self.llm_model)
        ctx_override: int | None = session.context_size_override
        if ctx_override is not None:
            if ctx_override > auto_ctx:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"User override context_size={ctx_override} exceeds hardware cap={auto_ctx}; "
                    f"using capped value to protect hardware",
                    color=ORANGE,
                )
                effective_ctx = auto_ctx
            else:
                effective_ctx = ctx_override
        else:
            effective_ctx = auto_ctx

        # --- output-token budget ---
        dynamic_budget: int = self.tokenBudget.compute_dynamic_max_tokens(formatted)
        override: int | None = session.max_output_tokens_override
        if override is not None:
            if override > dynamic_budget:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"User override max_output_tokens={override} exceeds computed budget={dynamic_budget}; "
                    f"using dynamic budget to avoid context overflow",
                    color=ORANGE,
                )
                return effective_ctx, dynamic_budget
            return effective_ctx, override
        return effective_ctx, dynamic_budget

    @property
    def terminal_line_size(self) -> int:
        """Read TERMINAL_LINE_SIZE live from config so runtime changes take effect immediately."""
        from Helpers.DebugHelper import DebugHelper as _DH

        raw: Any = self.cfg.get("TERMINAL_LINE_SIZE")
        if isinstance(raw, dict):
            d: dict[str, Any] = raw  # type: ignore[reportUnknownVariableType]
            key = "debug" if _DH.level(self.cfg) > 0 else "no_debug"
            return int(d.get(key, d.get("debug", 160)))
        return int(str(raw)) if raw is not None else 160

    @staticmethod
    def _supports_osc8() -> bool:
        from Chat.MarkedDocsViewer import supports_osc8

        return supports_osc8()

    def _open_marked_documents(self, marked: list[tuple[str, bytes]]) -> None:
        from Chat.MarkedDocsViewer import open_marked_documents

        open_marked_documents(
            marked,
            self.pretty,
            cyan=CYAN,
            project_root=self.cfg.get_str("_ABSOLUTE_PATH") or None,
        )

    def _show_web_sources(self, chosen: list[Any]) -> None:
        """Display web source URLs from chosen chunks as clickable terminal links."""
        web_urls: list[str] = []
        seen: set[str] = set()
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() != "web":
                continue
            url = str(meta.get("FilePath", "")).strip()
            if url and url not in seen:
                seen.add(url)
                web_urls.append(url)

        if not web_urls:
            return

        self.pretty.write(
            "I",
            "Web sources",
            f"{len(web_urls)} web source(s):",
            color=CYAN,
        )
        for url in web_urls:
            print(f"   🌐 {url}")

    def _show_original_sources(self, chosen: list[Any]) -> None:
        """Display original source file paths when mark_text=False."""
        import os
        from pathlib import Path

        # Collect unique source paths from chunks
        source_paths: set[str] = set()
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() == "web":
                continue
            file_path = str(meta.get("FilePath", "")).strip()
            if file_path and os.path.isfile(file_path):
                source_paths.add(file_path)

        if source_paths:
            sorted_paths = sorted(source_paths)
            self.pretty.write(
                "I",
                "Sources",
                f"{len(sorted_paths)} document(s), click to open:",
                color=CYAN,
            )
            for path in sorted_paths:
                abs_path = Path(path).resolve()
                print(f"   📄 {abs_path.as_uri()}")

        self._show_web_sources(chosen)

    def print_llm_answer(
        self, answer: str, terminal_width: int, prefix: str = "💡>  ", color: str = ""
    ) -> None:
        """
        Wrap `answer` by words to fit `terminal_width` and print each line prefixed.

        - Preserves existing paragraph breaks (lines from answer.splitlines()).
        - Avoids breaking words when possible.
        - Falls back safely if terminal_width is too small.
        """
        # Build optional background/foreground ANSI codes from config
        from typing import cast as _cast

        import Gui.Colors as _colors

        if color:
            style: str = color
        else:
            display_raw: Any = self.cfg.get("ANSWER_DISPLAY")
            display_cfg: dict[str, Any] = (
                _cast(dict[str, Any], display_raw)
                if isinstance(display_raw, dict)
                else {}
            )
            bg_name: str = str(display_cfg.get("bg") or "")
            fg_name: str = str(display_cfg.get("fg") or "")
            bg: str = str(getattr(_colors, bg_name, "")) if bg_name else ""
            fg: str = str(getattr(_colors, fg_name, "")) if fg_name else ""
            style = bg + fg

        avail = max(1, terminal_width - len(prefix))

        for paragraph in answer.splitlines():
            # preserve empty lines as a prefixed blank line
            if not paragraph.strip():
                print(f"{style}{prefix}{' ' * avail}{RESET}" if style else prefix)
                continue

            wrapped_lines = _wrap_preserving_ansi(paragraph, avail)

            # If wrap returned nothing (very long unbreakable word), force a fallback
            if not wrapped_lines:
                # hard-break the paragraph into chunks of avail
                wrapped_lines = [
                    paragraph[i : i + avail] for i in range(0, len(paragraph), avail)
                ]

            for line in wrapped_lines:
                if style:
                    # Re-apply the outer style after every inner RESET so that
                    # grounded-sentence spans (which close with \033[0m) don't
                    # cancel the background for the rest of the line.
                    padded = line.replace(RESET, RESET + style).ljust(avail)
                    print(f"{style}{prefix}{padded}{RESET}")
                else:
                    print(f"{prefix}{line}")
