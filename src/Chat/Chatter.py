import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict

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
from Chat.ConfidenceEvaluator import AnswerConfidenceResult
from Chat.ContextSummarizer import ContextSummarizer
from Chat.FileSelector import FileSelector
from Chat.QueryParts import QueryParts
from Chat.RAGChatImpl import RAGChatImpl
from Commons.Exceptions import (CollectionNotFoundError, LLMResultError,
                                QueryTranslationError)
from Compliance.BannedPhraseCollector import BannedPhraseCollector
from Config.Config import Config
from Globals.Globals import Globals
from Globals.Session import Session
from Grounding.Grounder import Grounder, LanguageDetector
from Gui.Colors import BROWN, CYAN, GREEN, ORANGE, RED, RESET, YELLOW
from Gui.GroundedAnswerHtmlRenderer import GroundedAnswerHtmlRenderer
from Gui.PrettyWriter import PrettyWriter
from Gui.RichAnswerRenderer import ANSWER_META_SEPARATOR, RichAnswerRenderer
from Gui.Symbols import Symbols
from Helpers.ConfidenceLogger import ConfidenceLogger
from Helpers.CSVWriter import CSVWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator
from Helpers.SourcePathLinkifier import SourcePathLinkifier

if TYPE_CHECKING:
    from Chat.LocalDocsServer import LocalDocsServer


_LIST_OR_TABLE_REQUEST = re.compile(
    r"\b(?:use|create|make|give|provide|show|return|format|output|generate|write)"
    r"\s+(?:(?:me|it|them|the results|a|an|the|markdown|numbered|bulleted|comparison)\s+){0,4}"
    r"(?:list|table|bullets)\b"
    r"|\b(?:as|in)\s+(?:a\s+)?(?:(?:markdown|numbered|bulleted|comparison)\s+)?"
    r"(?:list|table|bullets)\b"
    r"|\b(?:bullet points|(?:bulleted|numbered|markdown) list|markdown table)\b"
    r"|(?:^|[.!?;]\s*)\s*(?:please\s+)?list\s+\w"
    r"|\b(?:and|then|please)\s+list\s+\w",
    re.IGNORECASE,
)
_EVIDENCE_REQUEST = re.compile(
    r"\bevidence phrases?\b|\bwith\s+(?:supporting\s+)?evidence\b"
    r"|\b(?:add|include|give|provide|show|quote)"
    r"\s+(?:\w+\s+){0,3}(?:evidence|quotes?|excerpts?)\b",
    re.IGNORECASE,
)


class Chatter:
    def __init__(self) -> None:
        self.local_docs_server: LocalDocsServer | None = None
        # your core components
        self.globalsInstance: Globals = Globals()
        self.helpers: Helpers = Helpers()
        self.cfg: Config = Config()
        self.fileUtils: FileUtils = FileUtils()
        self.tensorHelpers: TensorHelpers = TensorHelpers()
        self.aiHelpers: AIHelpers = AIHelpers()
        self.models_cache: ModelsCache = ModelsCache()
        logger_name = self.cfg.get_str("_FRIENDLY_NAME", "RAGChat", silent=True)
        self.logger: logging.Logger = self.helpers.setup_logger(logger_name)
        self.globalsInstance.set_logger(self.logger)
        self.pretty: PrettyWriter = PrettyWriter()
        self.pretty_always: PrettyWriter = PrettyWriter(always_on=True)
        self.queryParts: QueryParts = QueryParts()
        self.csvWriter: CSVWriter = CSVWriter()
        self.confidence_logger: ConfidenceLogger = ConfidenceLogger()
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
        self.answer_renderer: RichAnswerRenderer = RichAnswerRenderer()
        self.tokenBudget: TokenBudget = TokenBudget()
        self.grounder: Grounder = Grounder()

    def run(
        self,
        session: Session,
        *,
        apiChunkHandler: Callable[[str], None] | None = None,
        is_streaming: bool | None = None,
        original_query: str | None = None,
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
        original_query:
            Original CLI input, before any document selection changes the command.
        """
        session.answer_grounding_verified = False
        session.browser_answer_content = ""
        streaming = self.is_streaming if is_streaming is None else is_streaming
        display_query = original_query if original_query is not None else session.query
        if apiChunkHandler is not None or getattr(session, "is_service", False):
            display_query = None

        command, argument = self.rag.parse_chat_command(session.query or "")
        session.summarize = command == "summarize"
        if command == "summarize":
            return self._run_document_summary(
                session, argument, apiChunkHandler, display_query
            )

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
        query: str = argument if command == "new" else session.query  # type: ignore[reportAssignmentType]

        self.pretty.write("I", "", f"Chatter RAG Query LMM: {self.llm_model}")

        prompt = PromptTemplate.from_template(self.prompt)
        try:
            context, length = self.rag.retrieve(session)
        except QueryTranslationError as exc:
            message = f"Retrieval stopped: {exc}"
            session.indexed_query_error = message
            self.pretty.write("E", "QueryPlan", message)
            if apiChunkHandler is not None:
                apiChunkHandler(message)
            self.confidence_logger.flush_turn_csv(
                session,
                outcome="translation_error",
                answer_text=message,
            )
            return False, message
        query_notice: str = self._build_query_notice(session, query)

        if length == 0:
            success, response_text = self._handle_no_results(
                session, query, query_notice, apiChunkHandler, display_query
            )
            self.confidence_logger.flush_turn_csv(
                session,
                outcome="no_results",
                answer_text=response_text or "",
            )
            return success, response_text

        # Resolve the prompt with actual values. Retrieval runs in English,
        # but response language follows user preference / detected user language.
        response_language: str = (
            session.preferred_response_language
            or getattr(session, "user_language", None)
            or "english"
        )
        split = getattr(session, "keyword_action_split", None)
        mark_text_enabled = bool(getattr(session, "mark_text", False))
        flow_grounding_enabled = bool(getattr(session, "enable_grounding", True))
        grounding_enabled = mark_text_enabled and flow_grounding_enabled
        answer_query = session.query or query
        if split is not None:
            action_instruction = ""
            if split.action_part is not None:
                action_instruction = self.cfg.get_str(
                    "_PROMPT_KEYWORD_ACTION_OPERATION"
                ).format(action_part=split.action_part)
            answer_query = self.cfg.get_str("_PROMPT_KEYWORD_ACTION_ANSWER").format(
                user_query=query,
                answer_instruction=split.answer_instruction,
                action_instruction=action_instruction,
            )
        if grounding_enabled:
            request = "\n".join(
                filter(
                    None,
                    (
                        query,
                        getattr(session, "orig_translated_query_en", None),
                        split.action_part if split is not None else None,
                    ),
                )
            )
            if _LIST_OR_TABLE_REQUEST.search(request) and not (
                _EVIDENCE_REQUEST.search(request)
                or _EVIDENCE_REQUEST.search(answer_query)
            ):
                answer_query += "\n\nAdd evidence phrases."
        if split is not None:
            self.pretty_always.write(
                "I",
                "KeywordActionSplitter",
                f"Final answer instruction: {answer_query}",
                color=BROWN,
            )
        formatted = prompt.format(
            context=context,
            input=answer_query,
            response_language=response_language,
        )
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

        will_apply_grounding = apiChunkHandler is not None and grounding_enabled

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
        if (
            query_notice
            and apiChunkHandler is not None
            and streaming
            and not will_apply_grounding
        ):
            apiChunkHandler(query_notice)

        if mark_text_enabled and not flow_grounding_enabled:
            session.marked_documents = []

        if will_apply_grounding and not streaming:
            self.pretty.write(
                "I",
                "Call LLM streaming:",
                "False \u2014 document grounding active (mark_text=True, streaming downgraded)",
                color=ORANGE,
            )

        llm_plan_detail = (
            "Answer generation: producing the final grounded response from retrieved context chunks."
            if grounding_enabled
            else "Answer generation: producing the final response from retrieved context chunks."
        )
        self.pretty.write(
            "I",
            "LLM Plan",
            llm_plan_detail,
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
            purpose="Inference",
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
            self.confidence_logger.flush_turn_csv(
                session,
                outcome="blocked_compliance",
                answer_text="",
            )
            return False, None

        if apiChunkHandler is None:
            session.browser_answer_content = self.masker.mask(content)

        grounding_notice: str = ""
        skip_grounding_for_short_answer: bool = False
        unverified_output: bool = False
        session.grounding_proxy_text = ""
        session.grounding_proxy_sentences = {}
        if grounding_enabled:
            if DebugHelper.check_session(session, 30):
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"Inference answer before grounding validation:\n{content}",
                    color=CYAN,
                )
            chunk_texts_for_grounding: list[str] = list(
                getattr(session, "chunk_texts_for_grounding", []) or []
            )
            if self.grounder.is_answer_too_short(content):
                self.pretty.write(
                    "I",
                    "Grounding",
                    "Answer too short for reliable sentence-level grounding; "
                    "returning ungrounded answer.",
                    color=ORANGE,
                )
                grounding_notice = self.grounder.short_answer_notice()
                skip_grounding_for_short_answer = True
                # Ensure no stale highlighted docs are shown from a previous query.
                session.marked_documents = []
            elif not self.grounder.has_grounded_evidence(
                session,
                content,
                chunk_texts_for_grounding,
                response_language,
                self._grounding_language_detector(session),
            ):
                web_leg_enabled = bool(getattr(session, "web_search", False)) or (
                    str(getattr(session, "retrieve_mode", "") or "").upper() == "WEB"
                )
                unverified_output = (
                    self.grounder.has_structured_answer(content) or web_leg_enabled
                )
                if unverified_output:
                    grounding_notice = self.grounder.unverified_structure_notice()
                    disposition = "generated output shown unverified"
                else:
                    content = self.grounder.no_evidence_message()
                    disposition = "generated answer withheld"
                self.pretty.write(
                    "W",
                    "Grounding",
                    f"Grounding requirements were not satisfied by retrieved context; {disposition}.",
                    color=ORANGE,
                )
                if not web_leg_enabled:
                    self.rag.confidenceEvaluator.store_answer_confidence(
                        session,
                        list(session.last_chosen_chunks),
                        note=(
                            f"grounding failed; {disposition}. "
                            "Retrieval metrics describe selected chunks, not verified answer support"
                        ),
                        emit=False,
                        grounding_failed=True,
                    )
                self.confidence_logger.log_step(
                    session,
                    step_name="grounding evidence check",
                    status="unverified" if unverified_output else "fallback",
                    confidence_level="LOW",
                    confidence_score=0.0,
                    detail=f"answer did not satisfy grounding requirements; {disposition}",
                    source="grounding",
                )
                answer.content = content
                if not unverified_output:
                    # A withheld answer must not present unrelated sources.
                    session.last_chosen_chunks = []
                    session.chunk_texts_for_grounding = []
            else:
                session.answer_grounding_verified = True

        if session.use_chat_context:
            self.chatContext.add_chat_turn(
                session,
                query,
                grounding_notice + content if unverified_output else content,
            )

        # Post-process answer content
        masked_content = self.masker.mask(content)
        answer.content = masked_content
        if grounding_notice:
            answer.content = grounding_notice + answer.content
        if query_notice:
            answer.content = query_notice + answer.content
        if web_warning and apiChunkHandler is not None:
            answer.content = web_warning + answer.content
        if apiChunkHandler is None:
            answer.content = SourcePathLinkifier.linkify_source_paths_md(
                answer.content, allow_local_file_uri=True
            )

        confidence = session.answer_confidence_result
        if confidence is None:
            raise ValueError("Retrieval did not populate answer confidence")
        self._evaluate_answer_quality(session, confidence, answer.content or "")

        # Ground the original answer text, then mask only its CLI rendering.
        cli_answer = answer.content
        if grounding_enabled and not skip_grounding_for_short_answer:
            local_chunk_texts: list[str] = list(
                getattr(session, "chunk_texts_for_grounding", []) or []
            )
            web_chunk_texts = self.grounder.collect_web_texts(session)
            evidence_texts: list[str] = local_chunk_texts + web_chunk_texts
            if evidence_texts:
                grounded_cli = self.grounder.mark_answer_cli(
                    session,
                    content,
                    evidence_texts,
                    sentence_translations=session.grounding_proxy_sentences,
                    text_transform=self.masker.mask,
                )
                cli_prefix = ""
                if web_warning and apiChunkHandler is not None:
                    cli_prefix += web_warning
                cli_prefix += query_notice + grounding_notice
                cli_answer = cli_prefix + grounded_cli
                if apiChunkHandler is None:
                    cli_answer = SourcePathLinkifier.linkify_source_paths_md(
                        cli_answer, allow_local_file_uri=True
                    )
                if grounded_cli == masked_content and DebugHelper.check_session(
                    session, 30
                ):
                    self.pretty.write(
                        "D",
                        "Grounding",
                        "Grounding active but no sentence-level overlap met thresholds "
                        "for CLI marking",
                        color=CYAN,
                    )
            elif DebugHelper.check_session(session, 30):
                self.pretty.write(
                    "D",
                    "Grounding",
                    "No eligible evidence text for CLI grounding marks "
                    "(local chunks=0, web snippets/pages=0)",
                    color=CYAN,
                )

        confidence_notice = self._build_answer_confidence_notice(
            confidence,
            monospace=apiChunkHandler is not None,
        )
        if confidence_notice:
            cli_confidence_notice = self._build_answer_confidence_notice_cli(confidence)
            answer.content = answer.content + confidence_notice
            # The separator renders the confidence block with the standard
            # background so the answer body keeps its distinct one.
            cli_answer = (
                cli_answer
                + ANSWER_META_SEPARATOR
                + (cli_confidence_notice or confidence_notice)
            )
            self.confidence_logger.log_step(
                session,
                step_name="final answer confidence",
                status="executed",
                confidence_level=confidence.level,
                confidence_score=confidence.score,
                detail=confidence.summary,
                source="answer",
            )
            if apiChunkHandler is not None and streaming and not will_apply_grounding:
                apiChunkHandler(confidence_notice)

        if apiChunkHandler is not None:
            chosen_for_cli = getattr(session, "last_chosen_chunks", [])
            selected_sources_cli = self.helpers.build_selected_sources_md(
                chosen_for_cli
            )
            if selected_sources_cli:
                cli_answer = cli_answer + selected_sources_cli

        # Deliver answer: send buffered content to API client; always print CLI
        if apiChunkHandler is not None and (will_apply_grounding or not is_streaming):
            apiChunkHandler(answer.content)

        # Mark source documents with grounding highlights and show to user
        session.last_answer_content = answer.content
        session.marked_documents = []
        if grounding_enabled and not skip_grounding_for_short_answer:
            chosen = getattr(session, "last_chosen_chunks", [])
            if chosen and self.grounder.has_local_spans(session):
                try:
                    self.rag._mark_sources(
                        session, chosen
                    )  # pyright: ignore[reportPrivateUsage]
                except Exception as exc:
                    self.pretty.write(
                        "W", "VisualMarker", f"Visual marking failed: {exc}"
                    )

        if apiChunkHandler is None:
            marked = list(getattr(session, "marked_documents", []) or [])
            if marked:
                self._open_marked_documents(marked)
            # Offer the browser view without opening it automatically.
            grounded_page = self._grounded_answer_browser_document(
                session,
                mark_supported=grounding_enabled
                and not skip_grounding_for_short_answer,
            )
            if grounded_page is not None:
                self._open_grounded_answer_page(grounded_page)

        self.print_llm_answer(
            cli_answer, self.terminal_line_size, original_query=display_query
        )

        if apiChunkHandler is None:
            chosen = getattr(session, "last_chosen_chunks", [])
            meta_md = self.helpers.build_document_metadata_md(chosen)
            if meta_md:
                print(meta_md)
            if grounding_enabled:
                marked = list(getattr(session, "marked_documents", []) or [])
                if not marked and chosen:
                    self._show_original_sources(chosen)
                elif chosen:
                    self._show_web_sources(chosen)
            else:
                if chosen:
                    self._show_original_sources(chosen)

        self.confidence_logger.flush_turn_csv(
            session,
            outcome="grounding_unverified" if unverified_output else "ok",
            answer_text=answer.content or "",
        )

        return True, answer.content

    # ------------------------------------------------------------------
    # Private helpers extracted from run()
    # ------------------------------------------------------------------

    def _run_document_summary(
        self,
        session: Session,
        target: str,
        api_chunk_handler: Callable[[str], None] | None,
        display_query: str | None,
    ) -> tuple[bool, str]:
        session.reset_turn_state()
        session.summarize = True
        session.summary_grounding = True
        session.answer_grounding_verified = False
        session.summary_statements = []
        session.last_chosen_chunks = []
        session.marked_documents = []
        session.chunk_texts_for_grounding = []
        session.last_answer_content = None
        session.grounding_proxy_text = ""
        session.grounding_proxy_sentences = {}
        self.confidence_logger.reset_turn(session)
        self.fileUtils.reset_lang_detection_events()
        try:
            documents = self.rag.retrieve_summary_documents(session, target)
            self.pretty_always.write(
                "I", "Summarize", f"Summarizing all {len(documents)} indexed chunks."
            )
            summarizer = ContextSummarizer(
                cfg=self.cfg, llm_caller=self.llmCaller, token_budget=self.tokenBudget
            )
            summary = summarizer.summarize_documents(documents, session=session)
        except (ValueError, CollectionNotFoundError) as exc:
            message = str(exc)
            session.indexed_query_error = message
            for line in message.splitlines():
                self.pretty.write("W", "", line, color=YELLOW)
            if api_chunk_handler is not None:
                api_chunk_handler(message)
            return False, message
        except LLMResultError as exc:
            message = f"Summarize: {exc}"
            session.indexed_query_error = message
            self.pretty.write("E", "Summarize", message, color=RED)
            if api_chunk_handler is not None:
                api_chunk_handler(message)
            return False, message

        title = str(documents[0].metadata.get("FileName", target))
        answer = f"## Summary of {title}\n\n{summary}"
        if self._check_answer_compliance(session, ModelOutput(content=answer), answer):
            message = "Summarize: answer compliance check failed."
            if api_chunk_handler is not None:
                api_chunk_handler(message)
            return False, message
        raw_answer = answer
        if api_chunk_handler is None:
            session.browser_answer_content = self.masker.mask(raw_answer)
        selected_file = str(documents[0].metadata.get("FilePath") or title)
        confirmation = FileSelector.confirmation(selected_file)
        history_query = session.query or "summarize:"
        if not target:
            history_query = f"summarize: {selected_file}"
        session.user_query_original = history_query
        session.last_chosen_chunks = documents
        session.chunk_texts_for_grounding = [
            document.page_content for document in documents
        ]
        response_language = session.preferred_response_language or "english"
        skip_grounding = self.grounder.is_answer_too_short(summary)
        supported = False
        if not skip_grounding:
            supported = self.grounder.has_grounded_evidence(
                session,
                summary,
                session.chunk_texts_for_grounding,
                response_language,
                self._grounding_language_detector(session),
            )
            if not supported:
                answer = "I couldn't find supporting evidence in the selected document for this summary."
                session.last_chosen_chunks = []
                session.chunk_texts_for_grounding = []
                self.pretty.write("W", "Grounding", answer, color=ORANGE)

        session.answer_grounding_verified = supported
        if supported or skip_grounding:
            answer = self.masker.mask(raw_answer)
        if skip_grounding:
            answer = self.grounder.short_answer_notice() + answer

        if session.use_chat_context:
            self.chatContext.add_chat_turn(session, history_query, answer)

        confidence = self.rag.confidenceEvaluator.store_answer_confidence(
            session,
            session.last_chosen_chunks if supported else [],
            note=(
                "Document summary: all indexed file chunks were used; no relevance "
                "ranking was performed, so retrieval relevance scores are unavailable. "
                "This heuristic score is not a factual-accuracy probability."
                if supported
                else (
                    "Summary too short for reliable grounding."
                    if skip_grounding
                    else "No grounded evidence for the summary."
                )
            ),
            emit=False,
        )
        self._evaluate_answer_quality(session, confidence, answer)
        output = ModelOutput(content=answer)
        cli_answer = answer
        if supported and session.mark_text:
            cli_answer = self.grounder.mark_answer_cli(
                session,
                raw_answer,
                session.chunk_texts_for_grounding,
                sentence_translations=session.grounding_proxy_sentences,
                text_transform=self.masker.mask,
            )
        answer += self._build_answer_confidence_notice(
            confidence, monospace=api_chunk_handler is not None
        )
        cli_confidence_notice = self._build_answer_confidence_notice_cli(confidence)
        if cli_confidence_notice:
            cli_answer += ANSWER_META_SEPARATOR + cli_confidence_notice
        answer = confirmation + answer
        cli_answer = confirmation + cli_answer
        self.confidence_logger.log_step(
            session,
            step_name="summary grounding",
            status="executed",
            confidence_level=confidence.level,
            confidence_score=confidence.score,
            detail=confidence.note,
            source="grounding",
        )
        session.last_answer_content = answer
        if supported and session.mark_text:
            try:
                self.rag._mark_sources(
                    session, documents, summary=True
                )  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:
                self.pretty.write("W", "VisualMarker", f"Visual marking failed: {exc}")
        if api_chunk_handler is None and not getattr(session, "is_service", False):
            marked = list(session.marked_documents or [])
            if marked:
                self._open_marked_documents(marked)
            if not getattr(session, "summarize", False):
                grounded_page = self._grounded_answer_browser_document(
                    session,
                    mark_supported=supported and session.mark_text,
                )
                if grounded_page is not None:
                    self._open_grounded_answer_page(grounded_page)
        self.print_llm_answer(
            cli_answer, self.terminal_line_size, original_query=display_query
        )
        if api_chunk_handler is not None:
            api_chunk_handler(answer)
        self.confidence_logger.flush_turn_csv(
            session, outcome="summary", answer_text=answer
        )
        return True, answer

    def _evaluate_answer_quality(
        self, session: Session, confidence: AnswerConfidenceResult, answer: str
    ) -> None:
        QualityDimensionEvaluator(self.cfg).evaluate_answer(
            session,
            confidence,
            answer,
            llm_caller=self.llmCaller,
            model=self.helpers.get_model_args("_ACTIVE_LLM_JUDGE")["MODEL"],
            confidence_evaluator=self.rag.confidenceEvaluator,
        )

    def _drain_lang_detection_confidence_events(self, session: Session) -> None:
        """Move buffered language-detection events into session confidence signals."""
        events = self.fileUtils.pop_lang_detection_events()
        for event in events:
            self.confidence_logger.log_language_detection(session, event)

    def _grounding_language_detector(self, session: Session) -> LanguageDetector:
        """Return the language detector used by grounding; events feed session confidence."""

        def _detect(text: str, native_lang: str, stage_label: str) -> str:
            language = self.fileUtils.get_user_text_language(
                text,
                output="nltk",
                native_lang=native_lang,
                stage_label=stage_label,
            )
            self._drain_lang_detection_confidence_events(session)
            return str(language or "")

        return _detect

    def _build_query_notice(self, session: Session, query: str) -> str:
        """Return query-change and incomplete multilingual-coverage notices."""
        notices = getattr(session, "indexed_query_notices", [])
        coverage_notice = (
            "Retrieval coverage notice:\n"
            + "\n".join(f"- {notice}" for notice in notices)
            + "\n\n---\n\n"
            if notices
            else ""
        )
        user_query_input = getattr(session, "user_query_input", None)
        if user_query_input:
            command, argument = self.rag.parse_chat_command(user_query_input)
            if command == "new":
                return (
                    coverage_notice
                    + f'{Symbols.sym_icon("SEARCH")}*Query: "{argument}"*\n\n---\n\n'
                )
        effective_q: str | None = getattr(session, "effective_query", None)
        if not effective_q or not query or effective_q.strip() == query.strip():
            return coverage_notice
        reason = getattr(session, "effective_query_reason", None) or "changed"
        notice_labels: dict[str, str] = {
            "translated": "Translated query",
            "rewritten": "Rewritten query",
            "translated+rewritten": "Translated & rewritten query",
        }
        label = notice_labels.get(reason, "Query (changed)")
        return (
            coverage_notice
            + f'{Symbols.sym_icon("SEARCH")}*{label}: "{effective_q}"*\n\n---\n\n'
        )

    @staticmethod
    def _resolve_answer_confidence_payload(
        confidence: AnswerConfidenceResult | None,
    ) -> tuple[str, str]:
        """Return (level, payload) for answer confidence notices."""
        if confidence is None:
            return "", ""
        summary = confidence.summary.strip()
        level = confidence.level.strip().upper()
        score_obj = confidence.score

        if summary:
            return level, summary
        elif level:
            return (
                level,
                f"{Symbols.sym_icon(level)}{level}\n"
                f"- C_final={float(score_obj):.2f}: final confidence in [0,1].",
            )
        else:
            return "", ""

    @staticmethod
    def _build_answer_confidence_notice(
        confidence: AnswerConfidenceResult | None,
        *,
        monospace: bool = False,
    ) -> str:
        """Return a Markdown block with the final answer confidence level."""
        _, payload = Chatter._resolve_answer_confidence_payload(confidence)
        if not payload:
            return ""

        if monospace:
            return f"\n\n---\n\n**Answer confidence:**\n```text\n{payload}\n```"

        return f"\n\n---\n\n**Answer confidence:**\n{payload}"

    @staticmethod
    def _build_answer_confidence_notice_cli(
        confidence: AnswerConfidenceResult | None,
    ) -> str:
        """Return a colorized confidence notice for terminal output only."""
        level, payload = Chatter._resolve_answer_confidence_payload(confidence)
        if not payload:
            return ""

        color = ORANGE
        if level == "HIGH":
            color = GREEN
        elif level == "MEDIUM":
            color = YELLOW

        status, newline, details = payload.partition("\n")
        return f"\n\n---\n\n**Answer confidence:**\n{color}{status}{RESET}{newline}{details}"

    def _handle_no_results(
        self,
        session: Session,
        query: str,
        query_notice: str,
        apiChunkHandler: Callable[[str], None] | None,
        display_query: str | None,
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
            self.print_llm_answer(
                msg, self.terminal_line_size, color=CYAN, original_query=display_query
            )
            return True, msg

        friendly_name = self.cfg.get_str("_FRIENDLY_NAME")
        web_mode: str = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
        show_web_hint: bool = web_mode == "1"
        web_only: bool = (
            str(getattr(session, "retrieve_mode", "") or "").upper() == "WEB"
        )

        no_results_msg = (
            "I couldn't find relevant information to answer your query in the provided context.\n\n"
            "Metadata used: None (context is empty)"
        )
        if web_only:
            web_warning = (
                "\u26a0\ufe0f **Web-only search returned no usable results.** "
                "Check the web-search log for provider errors or filtering."
            )
            self.pretty.write("W", "Web", web_warning, color=ORANGE)
            no_results_msg = f"{web_warning}\n\n---\n\n{no_results_msg}"
        if friendly_name == "RAGChatService":
            no_results_msg += (
                "\n\n**Things to try:**\n"
                "- Start a new topic by prefixing your question with `new:`.\n"
                "- Switch to a collection that contains the information you need.\n"
                "- Review your session parameters in the OpenWebUI **Controls** menu."
            )
            if show_web_hint and not web_only:
                no_results_msg += (
                    "\n- Open **Controls** (sliders on top right corner) and look for **+ Add Custom Parameter** at the right bottom"
                    "Overwrite **custom_pram_name** with **web_search** and **custom_param_value** with True "
                    "to search the internet for an answer."
                )
        elif friendly_name == "RAGChat":
            no_results_msg += (
                "\n\nHints:\n"
                "- Start a new topic by prefixing your question with `new:`.\n"
                "- Switch to a collection that contains the information you need.\n"
                "- Review your session parameters in settings (`b:`); "
                "type `help?` to see the available options."
            )
            if show_web_hint and not web_only:
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
        self.print_llm_answer(
            full_no_results, self.terminal_line_size, original_query=display_query
        )
        return True, full_no_results

    def _check_answer_compliance(
        self,
        session: Session,
        answer: ModelOutput,
        content: str,
    ) -> bool:
        """Run compliance checks on the LLM answer. Returns True if the answer should be rejected."""
        language: str = self.fileUtils.get_text_language(
            content,
            "ntlk",
            stage_label="answer_compliance_language",
        )
        self._drain_lang_detection_confidence_events(session)
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
                "Time": datetime.now(),
                "Status": status,
            }
        }
        rows = self.bannedPhraseCollector.prepare_for_csv_print(
            phrase_table,
            doc["meta"],
        )
        if not rows:
            rows = self.bannedPhraseCollector.prepare_print_for_chat(doc["meta"])
        self.csvWriter.write_json2csv(
            rows,
            "HUMAN_REVIEW",
        )
        if human_review:
            self.pretty.write(
                "E", "Answer check", "\u26a0\ufe0f    Answer compliance check failed."
            )
        return bool(human_review)

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
            _file_utils=self.fileUtils,
            local_server=getattr(self, "local_docs_server", None),
        )

    def _grounded_answer_browser_document(
        self, session: Session, *, mark_supported: bool = True
    ) -> tuple[str, bytes, dict[str, str]] | None:
        """Build an answer preview, explicitly labelled when not verified."""
        verified = bool(getattr(session, "answer_grounding_verified", False))
        browser_answer = getattr(session, "browser_answer_content", "")
        answer = str(
            browser_answer or getattr(session, "last_answer_content", "") or ""
        )
        evidence = list(getattr(session, "chunk_texts_for_grounding", []) or [])
        evidence.extend(self.grounder.collect_web_texts(session))
        if not answer.strip():
            return None
        if browser_answer:
            answer += self._build_answer_confidence_notice(
                session.answer_confidence_result
            )
        page, csp = GroundedAnswerHtmlRenderer().render(
            answer,
            self.grounder,
            session,
            evidence,
            sentence_translations=session.grounding_proxy_sentences,
            verified=verified,
            mark_supported=mark_supported,
        )
        return (
            "GroundedAnswer.html" if verified else "AnswerPreview.html",
            page.encode("utf-8"),
            {
                "Content-Security-Policy": csp,
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    def _open_grounded_answer_page(
        self, document: tuple[str, bytes, dict[str, str]]
    ) -> None:
        """Offer the grounded answer as a link that opens only on user request."""
        from Chat.MarkedDocsViewer import open_grounded_answer_html

        open_grounded_answer_html(
            document[0],
            document[1],
            self.pretty,
            cyan=CYAN,
            local_server=getattr(self, "local_docs_server", None),
            response_headers=document[2],
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
            print(f"   {Symbols.sym_icon('WEB')}{url}")

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
            link_green: str = "\033[32m"
            reset: str = "\033[0m"
            for path in sorted_paths:
                abs_path = Path(path).resolve()
                print(
                    f"{link_green}   {Symbols.sym_icon('DOC')}{abs_path.name}: "
                    f"{abs_path.as_uri()}{reset}"
                )

        self._show_web_sources(chosen)

    def print_llm_answer(
        self,
        answer: str,
        terminal_width: int,
        prefix: str | None = None,
        color: str = "",
        original_query: str | None = None,
    ) -> None:
        """Render the CLI answer as Markdown with Rich."""
        from typing import cast as _cast

        import Gui.Colors as _colors

        if prefix is None:
            prefix = f"{Symbols.sym_icon('CHAT')}>  "

        background = ""
        foreground = ""
        body_background = ""
        if not color:
            display_raw: Any = self.cfg.get("ANSWER_DISPLAY")
            display_cfg: dict[str, Any] = (
                _cast(dict[str, Any], display_raw)
                if isinstance(display_raw, dict)
                else {}
            )
            bg_name: str = str(display_cfg.get("bg") or "")
            fg_name: str = str(display_cfg.get("fg") or "")
            body_bg_name: str = str(display_cfg.get("body_bg") or "")
            background = str(getattr(_colors, bg_name, "")) if bg_name else ""
            foreground = str(getattr(_colors, fg_name, "")) if fg_name else ""
            body_background = (
                str(getattr(_colors, body_bg_name, "")) if body_bg_name else ""
            )

        renderer = getattr(self, "answer_renderer", None) or RichAnswerRenderer()
        renderer.render(
            answer,
            terminal_width,
            prefix=prefix,
            original_query=original_query,
            color=color,
            background=background,
            foreground=foreground,
            body_background=body_background,
        )
