import logging
import textwrap
from datetime import datetime
from typing import Any, Callable, Dict

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
        self.llm_model: str = self.helpers.get_model_args("_LLM")["MODEL"]
        self.is_streaming: bool = self.helpers.get_model_args("_OLLAMA").get(
            "STREAMING_REQ", False
        )
        self.use_ollama_gpu: bool = self.helpers.get_model_args("_OLLAMA").get(
            "USE_GPU", True
        )
        self.terminal_line_size: int = self.cfg.get_int("TERMINAL_LINE_SIZE")
        self.prompt: str
        self.prompt_name: str | None
        # Indirect call
        prompt_var: str = self.helpers.get_model_args("_LLM")["PROMPT_CHAT"]
        self.prompt, self.prompt_name = self.cfg.get(f"${prompt_var}")

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
                ("chroma_k_value", session.chroma_k_value),
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

        # build prompt
        prompt = PromptTemplate.from_template(self.prompt)

        # Retrieve context
        context, length = self.rag.retrieve(session)
        if length == 0:
            no_results_msg = (
                "I couldn't find relevant information to answer your query in the provided context.\n\n"
                "Metadata used: None (context is empty)"
            )
            friendly_name = self.cfg.get_str("_FRIENDLY_NAME")
            if friendly_name == "RAGChatService":
                no_results_msg += (
                    "\n\n**Hints (OpenWebUI Controls menu):**\n"
                    "- Increase the `chroma_k_value` slider (top_k) to retrieve more candidate chunks.\n"
                    "- Set `chroma_threshold` to a lower value (e.g. **0.2**) so fewer chunks are filtered out.\n"
                    f"- Try a different `strategy` (current: "
                    f"{session.strategy or 'default'}). "
                    f"Allowed: ULTRA_WIDE, WIDE, MEDIUM, NARROW."
                )
            elif friendly_name == "RAGChat":
                no_results_msg += (
                    "\n\nHints:\n"
                    "- Use  chroma_k_value  to increase top_k and retrieve more candidate chunks.\n"
                    "- Use  threshold  to set a lower value (e.g. 0.2) so fewer chunks are filtered out.\n"
                    f"- Use  strategy  to switch retrieval strategy (current: "
                    f"{session.strategy or 'default'}). "
                    f"Allowed: ULTRA_WIDE, WIDE, MEDIUM, NARROW.\n"
                    "- Type  help?  at the command prompt for all available commands."
                )
            if apiChunkHandler is not None:
                apiChunkHandler(no_results_msg)
            self.print_llm_answer(no_results_msg, self.terminal_line_size)
            return True, no_results_msg

        # Resolve the prompt with actual values
        formatted = prompt.format(context=context, input=session.query)

        # Resolve context window and output-token budget (applies user overrides with warnings).
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

        handler = self.llmCaller.make_on_chunk(ollama_options)
        # If an API chunk handler is registered, wrap the base handler so both are called
        if apiChunkHandler is not None:
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
        # for cell in handler.__closure__: print(cell.cell_contents)
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
        # Compliance hook: run checks on the received LLM answer
        content: str = answer.content or ""
        language: str = self.fileUtils.get_text_language(content, "ntlk")
        embedder: Any = self.models_cache.get_hf_embeddings()

        embeddings: list[Any] = embedder.embed_documents([answer.content])
        stage: str = "PIPELINE_CHECK"
        emb_tensor: Any = self.tensorHelpers.to_tensor(embeddings[0])
        (
            human_review,
            _,
            phrase_table,
        ) = self.aiHelpers.run_ensemble_checks(
            content,
            language,
            stage=stage,
            accumulate=False,
            require_keybert=False,
            embedding=emb_tensor,
        )
        status = "NOT_OK" if human_review is True else "OK"
        # Handle compliance failure gracefully

        doc: Dict[str, Dict[str, Any]] = {}
        doc["meta"] = {
            "Stage": stage,
            "Session": session.export_session_state_as_cell(),
            "Time": datetime.now(),
            "Status": status,
        }
        self.csvWriter.write_json2csv(
            self.bannedPhraseCollector.prepare_for_csv_print(phrase_table, doc),
            "HUMAN_REVIEW",
        )
        if human_review:
            self.pretty.write(
                "E", "Answer check", "⚠️    Answer compliance check failed."
            )
            return False, None

        # Record the chat turn
        if session.use_chat_context:
            self.chatContext.add_chat_turn(session, query, content)

        # Output the answer
        answer.content = self.masker.mask(content)
        self.print_llm_answer(answer.content, self.terminal_line_size)
        return True, answer.content

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

    def print_llm_answer(
        self, answer: str, terminal_width: int, prefix: str = "💡>  "
    ) -> None:
        """
        Wrap `answer` by words to fit `terminal_width` and print each line prefixed.

        - Preserves existing paragraph breaks (lines from answer.splitlines()).
        - Avoids breaking words when possible.
        - Falls back safely if terminal_width is too small.
        """
        # available width for text after the prefix
        print(f"\n{CYAN}{"+" * self.terminal_line_size}{RESET}")
        avail = max(1, terminal_width - len(prefix))

        for paragraph in answer.splitlines():
            # preserve empty lines as a prefixed blank line
            if not paragraph.strip():
                print(prefix)
                continue

            wrapped_lines = textwrap.wrap(
                paragraph, width=avail, break_long_words=False, break_on_hyphens=False
            )

            # If wrap returned nothing (very long unbreakable word), force a fallback
            if not wrapped_lines:
                # hard-break the paragraph into chunks of avail
                wrapped_lines = [
                    paragraph[i : i + avail] for i in range(0, len(paragraph), avail)
                ]

            for line in wrapped_lines:
                print(f"{prefix}{line}")
        print(f"{CYAN}{"+" * self.terminal_line_size}{RESET}\n")
