import logging
import textwrap
from datetime import datetime
from typing import Any, Dict

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
        self.is_streaming: bool = self.cfg.get_bool("OLLAMA_STREAMING_REQ")
        self.use_ollama_gpu: bool = self.cfg.get_bool("USE_OLLAMA_GPU")
        self.terminal_line_size: int = self.cfg.get_int("TERMINAL_LINE_SIZE")
        self.prompt: str
        self.prompt_name: str | None
        # Indirect call
        prompt_var: str = self.helpers.get_model_args("_LLM")["PROMPT_CHAT"]
        self.prompt, self.prompt_name = self.cfg.get(f"${prompt_var}")

        self.cp: CommandProcessor = CommandProcessor()
        self.rag: RAGChatImpl = RAGChatImpl()
        self.session: Session = Session()
        self.chatContext: ChatContext = ChatContext()
        self.llmCaller: LLMCaller = LLMCaller()
        self.modelOutputAdapter: ModelOutputAdapter = ModelOutputAdapter()
        self.masker: Masker = Masker()
        self.tokenBudget: TokenBudget = TokenBudget()

    def run(self) -> bool:
        missing = [
            name
            for name, val in (
                ("chroma_k_value", self.session.chroma_k_value),
                ("max_output_tokens", self.session.max_output_tokens),
                ("temperature", self.session.temperature),
                ("top_k", self.session.top_k),
                ("top_p", self.session.top_p),
                ("query", self.session.query),
            )
            if val is None
        ]
        if missing:
            raise ValueError(f"Session fields not set: {', '.join(missing)}")

        # Pyright can't narrow through the dynamic loop above — bind non-None locals.
        temperature: float = self.session.temperature  # type: ignore[reportAssignmentType]
        top_k: float = self.session.top_k  # type: ignore[reportAssignmentType]
        top_p: float = self.session.top_p  # type: ignore[reportAssignmentType]
        query: str = self.session.query  # type: ignore[reportAssignmentType]

        self.pretty.write("I", "", f"Chatter RAG Query LMM: {self.llm_model}")

        # build prompt
        prompt = PromptTemplate.from_template(self.prompt)

        # Retrieve context
        context, length = self.rag.retrieve(self.session)
        if length == 0:
            return True

        # Resolve the prompt with actual values
        formatted = prompt.format(context=context, input=self.session.query)

        # Resolve context window and output-token budget (applies user overrides with warnings).
        effective_ctx, resolved_output = self._resolve_token_params(formatted)
        self.session.max_output_tokens = resolved_output

        handler = self.llmCaller.make_on_chunk(
            temperature,
            self.session.max_output_tokens,
            top_k,
            top_p,
        )
        # for cell in handler.__closure__: print(cell.cell_contents)
        llm_result = self.llmCaller.call_llm(
            self.llm_model,
            formatted,
            temperature,
            top_k,
            top_p,
            self.session.max_output_tokens,
            answer_is_json=True,
            use_ollama_gpu=self.use_ollama_gpu,
            template_name=self.prompt_name,
            on_chunk=handler,
            streaming=self.is_streaming,
            stage="Run user prompt",
            context_size=effective_ctx,
        )

        if isinstance(llm_result, dict) and "error" in llm_result:  # type: ignore[reportUnnecessaryIsInstance]
            msg = f"LLM error: {llm_result['error']}"
            self.pretty.write("E", "LLM", msg, color=RED)
            raise LLMResultError(msg)

        answer: ModelOutput = self.modelOutputAdapter.interpret(
            llm_result,
            self.llm_model,
            is_compliance=False,
            is_streaming=self.is_streaming,
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
            "Session": self.session.export_session_state_as_cell(),
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
            return False

        # Record the chat turn
        if self.session.use_chat_context:
            self.chatContext.add_chat_turn(self.session, query, content)

        # Output the answer
        answer.content = self.masker.mask(content)
        self.print_llm_answer(answer.content, self.terminal_line_size)
        return True

    def _resolve_token_params(self, formatted: str) -> tuple[int, int]:
        """Resolve effective context window and output-token budget, applying user overrides.

        Returns:
            (effective_ctx, resolved_max_output_tokens) — both guaranteed to be safe values.
        """
        # --- context window ---
        auto_ctx: int = self.tokenBudget.get_context_limit(self.llm_model)
        ctx_override: int | None = self.session.context_size_override
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
        override: int | None = self.session.max_output_tokens_override
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
