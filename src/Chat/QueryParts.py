# Local module imports
import os
# Standard library imports
import re
from typing import Any, List, Optional

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]

inquirer: Any = _inquirer
from InquirerPy.base.control import Choice  # type: ignore[attr-defined]
from InquirerPy.validator import ValidationError  # type: ignore[attr-defined]

from AI.TokenBudget import TokenBudget
from Chat.RAGChatImpl import RAGChatImpl
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Session import Session
from Gui.CollectionPicker import CollectionPicker
from Gui.Colors import BRIGHT_BLUE, CYAN, ORANGE, RESET, YELLOW
from Gui.FileList import FileList
from Gui.HistoryManager import HistoryManager
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers


# ——— Command & Cmd types ———
class Cmd:
    def __init__(self, token: str, op: str, payload: Any = None) -> None:
        self.token: str = token
        self.op: str = op
        self.payload: Any = payload


class QueryParts(SingletonMixin):

    # Central declarative command specification
    # mode:
    #   - "normal": generic ? = ! handling
    #   - "file": file/path picker
    #   - "strategy": strategy picker/defaults
    #   - "debug": debug level picker
    #   - "help": help command
    # NOTE: CLI names here (keys) are mirrored as alias fields in ChatCompletionRequest
    # (src/Api/ChatCompletionHandler.py) so OpenWebUI Advanced Parameters accept them too.
    # When adding a new parameter here, add the corresponding alias field there as well.
    COMMAND_SPECS: dict[str, dict[str, Any]] = {
        "collection": {
            "section": "Talk with",
            "prompt": (
                "Selects the active knowledge base or document collection. "
                "This determines which embeddings, files, and Chat Context are used for retrieval. "
                "Changing the collection resets file selections and context."
            ),
            "mode": "normal",
            "type": "str",
            "attr": "collection_name",
        },
        "chat_name": {
            "section": "Talk with",
            "prompt": (
                "Sets the chat_name used for storing and retrieving Chat Context. "
                "Useful when multiple people share the same environment or when you want isolated histories."
            ),
            "mode": "chat_name",
            "type": "str",
            "attr": "chat_name",
        },
        "strategy": {
            "section": "Retrieval",
            "prompt": (
                "Chooses a predefined retrieval configuration (narrow, balanced_file_cap, wide, default). "
                "Each strategy sets defaults for chunk selection, thresholds, and sampling parameters. "
                "Use this when you want consistent behavior without manually tuning every value."
            ),
            "mode": "strategy",
            "type": "string",
            "attr": "strategy",
        },
        "retrieve_mode": {
            "section": "Retrieval",
            "prompt": (
                "Sets the retrieval mode. "
                "VECTOR: embedding-based only. "
                "BM25: keyword-based only. "
                "GRAPH: entity co-occurrence graph only. "
                "VECTOR_BM25: vector + BM25 via RRF. "
                "VECTOR_GRAPH: vector + graph via RRF. "
                "BM25_GRAPH: BM25 + graph via RRF. "
                "ALL: vector + BM25 + graph fused via RRF (all retrieval algorithms). "
                "WEB: web search only — skips all local indexes; requires WEB_SEARCH_MODE=1."
            ),
            "mode": "retrieve_mode",
            "type": "str",
            "attr": "retrieve_mode",
        },
        "rerank": {
            "section": "Retrieval",
            "prompt": (
                "Enables or disables the reranker (0 = off, 1 = on). "
                "When enabled, retrieved chunks are re\u2011scored using a more accurate cross\u2011encoder. "
                "This improves relevance at the cost of additional computation."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "rerank",
        },
        "vector_weight": {
            "section": "Retrieval",
            "prompt": (
                "Weight applied to vector (embedding) retrieval scores before RRF fusion. "
                "Set to 0.0 to effectively disable vector retrieval in the merged ranking. "
                "Relative to bm25_weight and graph_weight."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "vector_weight",
        },
        "bm25_weight": {
            "section": "Retrieval",
            "prompt": (
                "Weight applied to BM25 keyword retrieval scores before RRF fusion. "
                "Set to 0.0 to effectively disable BM25 in the merged ranking. "
                "Relative to vector_weight and graph_weight."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "bm25_weight",
        },
        "graph_weight": {
            "section": "Retrieval",
            "prompt": (
                "Weight applied to graph (entity co-occurrence) retrieval scores before RRF fusion. "
                "Set to 0.0 to effectively disable graph retrieval in the merged ranking. "
                "Relative to vector_weight and bm25_weight."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "graph_weight",
        },
        "web_search": {
            "section": "Retrieval",
            "prompt": (
                "Control web search mode. "
                "local_only = use only local indexes. "
                "local_and_web = query the web via DuckDuckGo and merge results with local retrieval. "
                "web_only = skip all local indexes; use web search only (equivalent to retrieve_mode=WEB). "
                "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (web_search=local_and_web)."
            ),
            "mode": "web_search",
            "type": "str",
            "attr": "web_search",
        },
        "web_weight": {
            "section": "Retrieval",
            "prompt": (
                "Weight applied to web search results before RRF fusion. "
                "Set to 0.0 to exclude web results from the merged ranking. "
                "Relative to vector_weight, bm25_weight, and graph_weight. "
                "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (e.g. web_weight=0.5)."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "web_weight",
        },
        "fetch_page_content": {
            "section": "Retrieval",
            "prompt": (
                "Controls how web results are used. "
                "'snippets only' = use the search snippet as the document text (fast, default). "
                "'fetch pages' = fetch the full page and extract main text (slower, more content). "
                "Can be set permanently for a model via OpenWebUI → Admin → Models → Advanced Parameters (fetch_page_content=true)."
            ),
            "mode": "fetch_page_content",
            "type": "str",
            "attr": "fetch_page_content",
        },
        "web_rerank_threshold": {
            "section": "Retrieval",
            "prompt": (
                "Minimum cross-encoder score a web result must reach after reranking. "
                "0.0 = no additional filtering (default \u2014 web results are already "
                "quality-gated by bm25_pre_filter and cosine_pre_filter). "
                "Raise for stricter post-rerank filtering (e.g. web_rerank_threshold=0.05). "
                "Unlike threshold, this applies only to web-sourced chunks."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "web_rerank_threshold",
        },
        "fetch_k": {
            "section": "Retrieval",
            "prompt": (
                "Defines how many chunks each retriever (vector, BM25, graph) fetches before filtering and fusion. "
                "Higher values increase recall but may introduce noise. "
                "Lower values improve precision but risk missing relevant context."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "retriever_k",
        },
        "context_chunks": {
            "section": "Retrieval",
            "prompt": (
                "Sets the maximum number of chunks kept after all filters and reranking. "
                "This controls how much context the model sees. "
                "A larger value helps with broad questions; a smaller one keeps responses focused."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "final_chunks_to_llm",
        },
        "threshold": {
            "section": "Retrieval",
            "section_break": "before",
            "prompt": (
                "Minimum similarity score a chunk must meet to be considered relevant. "
                "Raising the threshold increases precision but may drop borderline‑useful chunks. "
                "Lowering it increases recall but risks including irrelevant text."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "chroma_threshold",
        },
        "max_output_tokens": {
            "section": "Retrieval",
            "prompt": (
                "Cap the number of output tokens the model may generate in its reply. "
                "Leave unset to use the fully dynamic budget. "
                "A warning is shown if the value exceeds the computed budget; "
                "Ollama will then receive the dynamic value instead "
                "(e.g. max_output_tokens=512 for short answers)."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "max_output_tokens_override",
        },
        "context_size": {
            "section": "Retrieval",
            "prompt": (
                "KV-cache / context window size sent to Ollama as num_ctx. "
                "Leave unset to use the auto-detected model limit (capped by TOKEN_BUDGET_CONTEXT_CAP). "
                "Setting this higher than the hardware cap is ignored with a warning. "
                "Use a lower value to reduce VRAM usage on constrained hardware "
                "(e.g. context_size=4096)."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "context_size_override",
        },
        "terminal_line_size": {
            "section": "Retrieval",
            "prompt": (
                "Width (in characters) used for wrapping terminal output. "
                "Increase for wide terminals to reduce line wrapping in debug tables; "
                "decrease for narrow windows (e.g. terminal_line_size=120)."
            ),
            "mode": "terminal_line_size",
            "type": "int",
        },
        "temperature": {
            "section": "Retrieval",
            "prompt": (
                "Controls randomness in the model’s generation. "
                "Lower values produce deterministic, factual answers; higher values increase creativity and variation. "
                "Use low temperature for compliance, code, and precise Q&A; use higher values for brainstorming."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "temperature",
        },
        "top_p": {
            "section": "Retrieval",
            "prompt": (
                "Nucleus sampling threshold that limits token selection to the smallest set whose cumulative probability reaches p. "
                "Lower values restrict generation to highly probable tokens; higher values allow more diverse outputs. "
                "This is a softer, more adaptive alternative to top_k."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "top_p",
        },
        "top_k": {
            "section": "Retrieval",
            "prompt": (
                "Limits generation to the top k most likely next tokens. "
                "Smaller values reduce noise and hallucination risk; larger values increase creativity. "
                "Use this together with top_p to shape the model’s output behavior."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "top_k",
        },
        "file": {
            "section": "File Input",
            "prompt": (
                "Selects a specific file from the active collection to query. "
                "Useful when you want to restrict retrieval to a single document. "
                "The file picker helps you avoid typos and path issues."
            ),
            "mode": "file",
            "type": "string",
            "attr": "file_name",
        },
        "path": {
            "section": "File Input",
            "prompt": (
                "Same as 'file', but allows selecting by full path. "
                "Useful when working with external or newly added files."
            ),
            "mode": "path",
            "type": "string",
            "attr": "path_name",
        },
        "file_cap": {
            "section": "File Input",
            "prompt": (
                "Sets the maximum number of chunks allowed per file (balanced_file_cap / default strategy). "
                "Prevents large files from dominating retrieval. "
                "Helps maintain balanced context across multiple documents."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "per_file_limit",
        },
        "use_chat_context": {
            "section": "Chat Context",
            "prompt": (
                "Enables or disables retrieval from previous chat messages. "
                "When enabled, the system treats past conversation turns as additional context chunks. "
                "Useful for multi‑turn reasoning or long workflows."
            ),
            "mode": "normal",
            "type": "bool",
            "attr": "use_chat_context",
        },
        "history_keep": {
            "section": "Chat Context",
            "prompt": (
                "Pruning threshold: maximum number of context entries kept per chat. "
                "When the count exceeds this value, the oldest `history_prune` entries "
                "are summarized into one entry and removed."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "turns",
        },
        "history_prune": {
            "section": "Chat Context",
            "prompt": (
                "Pruning granularity: number of oldest entries compressed into a "
                "single summary per pruning pass. Smaller values preserve more "
                "fine-grained history; larger values compress more aggressively."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "prune_batch",
        },
        "rewrite_context": {
            "section": "Chat Context",
            "prompt": (
                "Number of most recent chat turns sent to the query rewriter. "
                "Lower values keep the rewrite focused on the latest exchange; "
                "higher values give the rewriter more conversational context. "
                "0 disables the limit (all fetched turns are used)."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "max_history_turns",
        },
        "topic_summary": {
            "section": "Chat Context",
            "prompt": (
                "Rolling topic summary mode for the query rewriter. "
                "'last' = use only the most recent ASSISTANT response as topic context; "
                "'all' = concatenate all ASSISTANT responses in the history window. "
                "'last' is lower cost and avoids stale context; 'all' gives the rewriter "
                "the full topic arc across multiple turns."
            ),
            "mode": "normal",
            "type": "str",
            "attr": "topic_summary_mode",
        },
        "debug_level": {
            "section": "Chat Context",
            "prompt": (
                "Sets the verbosity of debug output. "
                "Accepts a plain level number (e.g. '30') for >= mode, "
                "'ge 30' for explicit >= mode, 'is 30' for exact-match mode, "
                "or 'le 30' for <= mode (this level and all below). "
                "Higher levels show internal decisions, scoring, and filtering steps. "
                "Useful for diagnosing retrieval behavior or teaching how the pipeline works."
            ),
            "mode": "debug",
            "type": "debug_level",
            "attr": "debug_level",
        },
        "debug_mode": {
            "section": "Chat Context",
            "prompt": (
                "Sets the comparison mode for debug_level. "
                "'ge' activates the selected level and all below it (>=). "
                "'is' activates that exact level only (==). "
                "'le' activates that level and all above it (<=)."
            ),
            "mode": "debug_mode",
            "type": "str",
            "attr": "debug_mode",
        },
        "help": {
            "section": "General",
            "prompt": (
                "Displays detailed help for all available commands. "
                "Use help? to show this information at any time."
            ),
            "mode": "help",
        },
        "mark_text": {
            "section": "Visual",
            "prompt": (
                "When enabled, every retrieved local PDF source is highlighted "
                "in memory and opened from a short-lived temp file (auto-deleted "
                "on exit). The original files are not modified. "
                "Currently supports PDF only; other document types are "
                "skipped silently."
            ),
            "mode": "normal",
            "type": "bool",
            "attr": "mark_text",
        },
    }

    # Extra tokens that appear in regex but are not fully modeled as commands
    EXTRA_TOKENS: list[str] = ["context"]

    def __new__(cls, *args: Any, **kwargs: Any) -> "QueryParts":
        return SingletonMixin.__new__(cls)

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return

        self._initialized = True

        # Build token list for regexes
        tokens: list[str] = list(self.COMMAND_SPECS.keys()) + self.EXTRA_TOKENS
        token_alt: str = "|".join(tokens)

        # Keep original behavior, including the old "treshold" typo in CMD_RX
        self.split_regex: re.Pattern[str] = re.compile(
            rf"(?=\b({token_alt})\b\s*[\?\!\=\-\*])",
            re.IGNORECASE,
        )
        self.compiled_regex: re.Pattern[str] = re.compile(
            r"^\s*\b("
            r"use_chat_context|fetch_k|file|path|strategy|"
            r"context_chunks|treshold|threshold|"
            r"max_output_tokens|context_size|terminal_line_size|"
            r"temperature|top_p|top_k|"
            r"rerank|vector_weight|bm25_weight|graph_weight|"
            r"web_search|web_weight|web_rerank_threshold|fetch_page_content|"
            r"file_cap|collection|chat_name|context|"
            r"history_keep|history_prune|rewrite_context|topic_summary|"
            r"retrieve_mode|debug_level|debug_mode|help|show|"
            r"mark_text"
            r")\b"
            r"\s*([\?\!\=\-\*])\s*(.*)$",
            re.IGNORECASE,
        )

        self.hist: HistoryManager = HistoryManager()
        self.collectionPicker: CollectionPicker = CollectionPicker()
        self.session: Session = Session()
        self.ragChatImpl: RAGChatImpl = RAGChatImpl()
        self.fileList: FileList = FileList()
        self.cfg: Config = cfg or Config()
        self.helpers: Helpers = helpers or Helpers()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.tokenBudget: TokenBudget = TokenBudget()

        # load allowed strategies
        self.allowed_strategies: list[Any] = self.cfg.get_list("_ALLOWED_STRATEGIES")
        self.allowed_debug_levels: dict[str, Any] = self.cfg.get_dict(
            "_ALLOWED_DEBUG_LEVELS"
        )

        # derive prompt_map from COMMAND_SPECS for help/printing
        self.prompt_map: dict[str, dict[str, str]] = {
            name: {"prompt": spec["prompt"], "section": spec["section"]}
            for name, spec in self.COMMAND_SPECS.items()
            if "prompt" in spec and "section" in spec
        }

        # initial defaults
        self._defaults(
            "strategy", self.cfg.get_str("_ACTIVE_CHUNK_SELECT_STRATEGY"), True
        )

    # ——— Utility: type casting ———

    def _cast_value(self, tok: str, raw: str) -> Any:
        spec: dict[str, Any] | None = self.COMMAND_SPECS.get(tok)
        if not spec or "type" not in spec:
            return raw
        t: str = spec["type"]
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            return self.to_bool(raw)
        # default: string
        return str(raw)

    def _get_attr_name(self, tok: str) -> Optional[str]:
        spec: dict[str, Any] | None = self.COMMAND_SPECS.get(tok)
        if not spec:
            return None
        return spec.get("attr")

    # ——— Printing current values ———

    def print_values(self):
        s = self.session

        # Build values dict based on known attributes
        values: dict[str, Any] = {
            "collection": getattr(s, "collection_name", None),
            "chat_name": getattr(s, "chat_name", None),
            "strategy": getattr(s, "strategy", None),
            "retrieve_mode": getattr(s, "retrieve_mode", None),
            "fetch_k": getattr(s, "retriever_k", None),
            "context_chunks": getattr(s, "final_chunks_to_llm", None),
            "threshold": getattr(s, "chroma_threshold", None),
            "max_output_tokens": (
                f"{getattr(s, 'max_output_tokens', None)}"
                + (
                    f"  [override(api: max_tokens)={s.max_output_tokens_override}]"
                    if getattr(s, "max_output_tokens_override", None) is not None
                    else ""
                )
            ),
            "context_size": (
                f"{self.tokenBudget.get_context_limit()}"
                if getattr(s, "context_size_override", None) is None
                else f"override(api: num_ctx)={s.context_size_override}"
            ),
            "terminal_line_size": self._resolve_terminal_line_size(),
            "temperature": getattr(s, "temperature", None),
            "top_p": getattr(s, "top_p", None),
            "top_k": getattr(s, "top_k", None),
            "rerank": getattr(s, "rerank", None),
            "vector_weight": getattr(s, "vector_weight", None),
            "bm25_weight": getattr(s, "bm25_weight", None),
            "graph_weight": getattr(s, "graph_weight", None),
            "web_search": (
                "web_only"
                if getattr(s, "retrieve_mode", None) == "WEB"
                else (
                    "local_and_web" if getattr(s, "web_search", False) else "local_only"
                )
            ),
            "web_weight": getattr(s, "web_weight", None),
            "fetch_page_content": (
                "fetch pages"
                if getattr(s, "fetch_page_content", False)
                else "snippets only"
            ),
            "file": getattr(s, "file_name", None),
            "path": getattr(s, "file_path", None),
            "file_cap": getattr(s, "per_file_limit", None),
            "use_chat_context": getattr(s, "use_chat_context", None),
            "history_keep": getattr(s, "turns", None),
            "history_prune": getattr(s, "prune_batch", None),
            "rewrite_context": getattr(s, "max_history_turns", None),
            "topic_summary": getattr(s, "topic_summary_mode", None),
            "debug_level": getattr(s, "debug_level", None),
            "mark_text": getattr(s, "mark_text", False),
        }

        # Group keys by section — Retrieval, debug_level, and keys without a session
        # value (e.g. help) are rendered separately or excluded.
        grouped: dict[str, list[str]] = {}
        for key, meta in self.prompt_map.items():
            section = meta["section"]
            if section != "Retrieval" and key != "debug_level" and key in values:
                grouped.setdefault(section, []).append(key)

        lbl = 14  # label column width — shared by section headers and Retrieval sub-labels
        b_color = CYAN  # match the ++++ query border colour
        r_color = RESET

        def _label(text: str, indent: str = "  ") -> str:
            """Return coloured label padded to lbl, with given indent prefix."""
            return f"{indent}▶ {b_color}{text + ': ':<{lbl}}{r_color}"

        debug_mode = getattr(s, "debug_mode", "ge") or "ge"
        print(
            _label("Debug")
            + f"debug_level={values['debug_level']!r}  debug_mode={debug_mode!r}"
        )
        section_order = ["Chat Context", "Talk with", "File Input"]
        sorted_sections = sorted(
            grouped,
            key=lambda s: (
                section_order.index(s) if s in section_order else len(section_order),
                s,
            ),
        )
        for section_name in sorted_sections:
            header_plain = "  ▶ " + f"{section_name + ': ':<{lbl}}"
            header = _label(section_name)
            indent = " " * len(header_plain)
            line = header
            for key in grouped[section_name]:
                if key not in values:
                    continue
                # Per-key opt-in: COMMAND_SPECS[key]["section_break"] == "before"
                # forces a line break *before* this key, so wide sections stay
                # readable. Continuation lines are indented to align with the
                # first value of the section.
                spec = self.COMMAND_SPECS.get(key, {})
                if spec.get("section_break") == "before" and line != header:
                    print(line.rstrip())
                    line = indent
                line += f"{key}={values[key]!r}  "
            if line.rstrip() != header.rstrip():
                print(line.rstrip())

        # ——— Custom Retrieval block ———
        def _rline(label: str, *pairs: tuple[str, Any]) -> str:
            return _label(label, indent="    ") + "  ".join(
                f"{k}={v!r}" for k, v in pairs
            )

        vw = getattr(s, "vector_weight", None)
        bw = getattr(s, "bm25_weight", None)
        gw = getattr(s, "graph_weight", None)
        ww = getattr(s, "web_weight", None)
        bm25_pf = self.cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        cos_pf = self.cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        web_rthr = getattr(s, "web_rerank_threshold", None)
        if web_rthr is None:
            web_rthr = self.cfg.get_float("_WEB_SEARCH.rerank_threshold") or 0.0
        print(_label("Retrieval").rstrip())
        print(
            _rline(
                "Strategies",
                ("strategy", values["strategy"]),
                ("retrieve_mode", values["retrieve_mode"]),
                ("rerank", values["rerank"]),
                ("threshold", values["threshold"]),
            )
        )
        print(
            _rline(
                "Weights",
                ("vector_weight", vw),
                ("bm25_weight", bw),
                ("graph_weight", gw),
            )
        )
        print(
            _rline(
                "Web",
                ("web_search", values["web_search"]),
                ("web_weight", ww),
                ("fetch_page_content", values["fetch_page_content"]),
                ("bm25_pre_filter", bm25_pf),
                ("cosine_pre_filter", cos_pf),
                ("web_rerank_threshold", web_rthr),
            )
        )
        print(
            _rline(
                "Chunk takes",
                ("fetch_k", values["fetch_k"]),
                ("context_chunks", values["context_chunks"]),
            )
        )
        print(
            _rline(
                "LLM",
                ("temperature", values["temperature"]),
                ("top_p", values["top_p"]),
                ("top_k", values["top_k"]),
            )
        )
        print(
            _rline(
                "Output",
                ("max_output_tokens", values["max_output_tokens"]),
                ("context_size", values["context_size"]),
                ("terminal_line_size", values["terminal_line_size"]),
            )
        )

    # ——— Asking for values interactively ———

    def _ask(self, tok: str):
        s = self.session
        spec = self.COMMAND_SPECS.get(tok, {})

        mode = spec.get("mode", "normal")

        if mode in ("file", "path"):
            entry = self.fileList.select(
                f"{s.collection_name}_{s.chat_name}", "File"
            )  # now returns list of FileEntry
            if entry:
                fn, fp = entry[0].file, entry[0].path
                self.fileList.set(
                    f"{s.collection_name}_{s.chat_name}", "File", fn, fp
                )  # now returns list of FileEntry
                s.file_name = fn if mode == "file" else None
                s.file_path = fp if mode == "path" else None
                s.file_path_select = mode  # "file" or "path"
            return

        if mode == "strategy":
            choice = inquirer.select(
                message="Choose strategy:",
                choices=self.allowed_strategies,
                default=s.strategy,
            ).execute()
            s.strategy = choice
            self._base_defaults(choice)
            return

        if mode == "debug":
            choices = [
                Choice(name=label, value=value)
                for label, value in self.allowed_debug_levels.items()
            ]
            choice_value = inquirer.select(
                message="Choose debug level:",
                choices=choices,
                default=s.debug_level,
            ).execute()
            _dm = getattr(s, "debug_mode", "ge") or "ge"
            mode_choice = inquirer.select(
                message="Choose comparison mode:",
                choices=[
                    Choice(
                        name=">= level  (ge — activates this level and all below)",
                        value="ge",
                    ),
                    Choice(
                        name="<= level  (le — activates this level and all above)",
                        value="le",
                    ),
                    Choice(
                        name="== level  (is — activates this exact level only)",
                        value="is",
                    ),
                ],
                default=0 if _dm == "ge" else (1 if _dm == "le" else 2),
            ).execute()
            s.debug_level = choice_value
            s.debug_mode = mode_choice
            _combined = "none" if choice_value == 0 else f"{mode_choice} {choice_value}"
            self.cfg.set("DEBUG_LEVEL", _combined, force=True)
            return

        if mode == "debug_mode":
            _dm = getattr(s, "debug_mode", "ge") or "ge"
            mode_choice = inquirer.select(
                message="Choose comparison mode:",
                choices=[
                    Choice(
                        name=">= level  (ge — activates this level and all below)",
                        value="ge",
                    ),
                    Choice(
                        name="<= level  (le — activates this level and all above)",
                        value="le",
                    ),
                    Choice(
                        name="== level  (is — activates this exact level only)",
                        value="is",
                    ),
                ],
                default=0 if _dm == "ge" else (1 if _dm == "le" else 2),
            ).execute()
            s.debug_mode = mode_choice
            level = s.debug_level or 0
            _combined = "none" if level == 0 else f"{mode_choice} {level}"
            self.cfg.set("DEBUG_LEVEL", _combined, force=True)
            return

        if mode == "retrieve_mode":
            allowed: list[str] = self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
            choice = inquirer.select(
                message="Choose retrieve mode:",
                choices=allowed,
                default=s.retrieve_mode or "VECTOR",
            ).execute()
            s.retrieve_mode = choice
            return

        if mode == "preferred_response_language":
            # Build the language list from the Argos pair table so we only
            # offer languages the system can actually translate.
            lang_map: dict[str, str] = (
                self.cfg.get_dict("_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME") or {}
            )
            argos_pairs: list[Any] = (
                self.cfg.get_list("_ARGOS_DEFINITIONS.ARGOS_LANGUAGES") or []
            )
            iso_codes: set[str] = {"en"}
            for pair in argos_pairs:
                # Each entry is a (from, to) tuple/list.
                try:
                    iso_codes.add(str(pair[0]).lower())
                    iso_codes.add(str(pair[1]).lower())
                except (IndexError, TypeError):
                    continue
            names: list[str] = sorted(
                {lang_map.get(code, code).lower() for code in iso_codes}
            )
            if "english" not in names:
                names.insert(0, "english")
            choice = inquirer.select(
                message="Choose your preferred response language:",
                choices=names,
                default=s.preferred_response_language or "english",
            ).execute()
            s.preferred_response_language = choice
            return

        if tok == "collection":
            if s.collection_name:
                self.hist.save(f"{s.collection_name}_{s.chat_name}", "Settings")
            s.collection_name = self.collectionPicker.pick_collection()
            self.ragChatImpl.set_vector_store(s)
            self.reset_things()
            return

        if mode == "chat_name":
            existing = self._list_chat_names(s.collection_name or "")
            if existing:
                choices: list[Any] = [Choice(name=n, value=n) for n in existing]
                choices.append(Choice(name="<new chat name>", value="__new__"))
                picked = inquirer.select(
                    message="Pick a chat name (or choose <new chat name>):",
                    choices=choices,
                    default=s.chat_name if s.chat_name in existing else None,
                ).execute()
            else:
                picked = "__new__"
            if picked == "__new__":
                picked = (
                    inquirer.text(
                        message="Enter new chat name> ",
                    )
                    .execute()
                    .strip()
                )
                if not picked:
                    return
            self._read_and_apply_value(tok, picked, from_interactive=True)
            return

        if mode == "help":
            self.help()
            return

        if tok == "terminal_line_size":
            current = self._resolve_terminal_line_size()

            def _tls_validator(text: str) -> bool:
                try:
                    v = int(text)
                    if v < 40:
                        raise ValidationError(
                            message="terminal_line_size must be at least 40."
                        )
                    return True
                except ValueError:
                    raise ValidationError(
                        message="terminal_line_size must be an integer."
                    )

            val_str = (
                inquirer.text(
                    message=f"Enter terminal_line_size (current={current})> ",
                    validate=_tls_validator,
                )
                .execute()
                .strip()
            )
            self.cfg.set("TERMINAL_LINE_SIZE", int(val_str), force=True)
            return

        if mode == "web_search":
            _ws_current = (
                "web_only"
                if getattr(s, "retrieve_mode", None) == "WEB"
                else ("local_and_web" if s.web_search else "local_only")
            )
            _ws_choices = [
                Choice(name="local_only    — local indexes only", value="local_only"),
                Choice(
                    name="local_and_web — web + local retrieval", value="local_and_web"
                ),
                Choice(
                    name="web_only      — web only, skip local indexes",
                    value="web_only",
                ),
            ]
            ws_choice = inquirer.select(
                message="Choose web search mode:",
                choices=_ws_choices,
                default=_ws_current,
            ).execute()
            self._read_and_apply_value(tok, ws_choice, from_interactive=True)
            return

        if mode == "fetch_page_content":
            _allowed_fpc = ["snippets only", "fetch pages"]
            choice = inquirer.select(
                message="Choose web result mode:",
                choices=_allowed_fpc,
                default="fetch pages" if s.fetch_page_content else "snippets only",
            ).execute()
            self._read_and_apply_value(tok, choice, from_interactive=True)
            return

        if tok in (
            "fetch_k",
            "window",
            "threshold",
            "max_output_tokens",
            "context_size",
            "temperature",
            "top_p",
            "top_k",
            "file_cap",
            "rerank",
            "vector_weight",
            "bm25_weight",
            "graph_weight",
            "web_weight",
            "web_rerank_threshold",
            "use_chat_context",
            "history_keep",
            "history_prune",
            "rewrite_context",
            "retrieve_mode",
            "mark_text",
        ):
            spec = self.COMMAND_SPECS.get(tok, {})
            expected_type = spec.get("type", "str")
            attr = spec.get("attr")
            current = getattr(s, attr, None) if attr else None

            # For override attributes, show override vs auto info
            if tok == "max_output_tokens":
                override = s.max_output_tokens_override
                if override is not None:
                    prompt = f"Enter {tok} (override={override}, last_used={s.max_output_tokens})> "
                else:
                    prompt = f"Enter {tok} (current={s.max_output_tokens}, dynamic)> "
            elif tok == "context_size":
                override = s.context_size_override
                auto = self.tokenBudget.get_context_limit()
                if override is not None:
                    prompt = f"Enter {tok} (override={override}, auto={auto})> "
                else:
                    prompt = f"Enter {tok} (current={auto}, auto)> "
            else:
                prompt = f"Enter {tok} (current={current})> "

            def validator(text: str) -> bool:
                try:
                    self._validate_raw_for_type(text, expected_type)
                    return True
                except ValueError as e:
                    raise ValidationError(message=str(e))

            val = (
                inquirer.text(
                    message=prompt,
                    validate=validator,
                )
                .execute()
                .strip()
            )
            # apply via helper
            self._read_and_apply_value(tok, val, from_interactive=True)
            return

    # ——— Chat name discovery ———

    def _resolve_terminal_line_size(self) -> int:
        """Return the active terminal line size, resolving the debug/no_debug dict if needed.

        When TERMINAL_LINE_SIZE is a dict, the result is driven by the session's
        debug_level so that the width tracks the live session state rather than
        the config string (which may lag after a mid-session change).
        """
        raw = self.cfg.get("TERMINAL_LINE_SIZE")
        if isinstance(raw, dict):
            if int(getattr(self.session, "debug_level", 0) or 0) != 0:
                return int(raw.get("debug", 180))  # type: ignore[reportUnknownArgumentType]
            return int(raw.get("no_debug", 100))  # type: ignore[reportUnknownArgumentType]
        return int(str(raw)) if raw is not None else 160

    def _list_chat_names(self, collection_name: str) -> list[str]:
        """Return sorted unique chat names found in the history directory for the given collection."""
        history_dir = self.hist.path
        if not collection_name or not os.path.isdir(history_dir):
            return []
        prefix = f"{collection_name}_"
        suffixes = ("_Query.txt", "_Settings.txt")
        names: set[str] = set()
        for fname in os.listdir(history_dir):
            if not fname.startswith(prefix):
                continue
            for sfx in suffixes:
                if fname.endswith(sfx):
                    chat = fname[len(prefix) : -len(sfx)]
                    if chat:
                        names.add(chat)
                    break
        return sorted(names)

    # ——— Parsing and applying commands ———

    def handle(self, raw: str) -> List[Cmd]:
        parts = [s.strip() for s in self.split_regex.split(raw) if s.strip()]
        out: List[Cmd] = []
        for seg in parts:
            m = self.compiled_regex.match(seg)
            if not m:
                continue
            tok, op, pay = m.group(1).lower(), m.group(2), m.group(3).strip()
            pay = pay.strip("'\"")
            out.append(Cmd(tok, op, pay))
        return out

    def _apply(self, cmd: Cmd):
        if cmd.op == "?":
            if cmd.token == "show":
                self.print_values()
                return
            return self._show(cmd.token)
        if cmd.op == "=":
            return self._set(cmd.token, cmd.payload)
        if cmd.op == "!":
            return self._ask(cmd.token)
        if cmd.op == "-":
            return self._clear(cmd.token)
        if cmd.op == "*":
            return self._defaults(cmd.token, cmd.payload)

    # ——— Showing current values ———

    def _show(self, tok: str):
        if tok == "help":
            self.help()
            return

        s = self.session
        if tok == "file":
            fn, fp = self.fileList.get_current()
            print(f"[file] File: {fn} Path: {fp}")
        elif tok == "path":
            _, fp = self.fileList.get_current()
            print(f"[path] Path: {fp}")
        elif tok == "strategy":
            print(f"[strategy] {s.strategy}")
        elif tok == "chat_name":
            print(f"[chat_name] {s.chat_name}")
        elif tok == "context_chunks":
            print(f"[context_chunks] {s.final_chunks_to_llm}")
        elif tok == "fetch_k":
            print(f"[fetch_k] {s.retriever_k}")
        elif tok == "threshold":
            print(f"[threshold] {s.chroma_threshold}")
        elif tok == "max_output_tokens":
            override = s.max_output_tokens_override
            used = s.max_output_tokens
            if override is not None:
                print(f"[max_output_tokens] override={override}  last_used={used}")
            else:
                print(f"[max_output_tokens] {used}  (dynamic, no override set)")
        elif tok == "context_size":
            override = s.context_size_override
            if override is not None:
                print(
                    f"[context_size] override={override}  (auto={self.tokenBudget.get_context_limit()})"
                )
            else:
                print(
                    f"[context_size] {self.tokenBudget.get_context_limit()}  (auto, no override set)"
                )
        elif tok == "temperature":
            print(f"[temperature] {s.temperature}")
        elif tok == "top_p":
            print(f"[top_p] {s.top_p}")
        elif tok == "top_k":
            print(f"[top_k] {s.top_k}")
        elif tok == "file_cap":
            print(f"[file_cap] {s.per_file_limit}")
        elif tok == "vector_weight":
            print(f"[vector_weight] {s.vector_weight}")
        elif tok == "bm25_weight":
            print(f"[bm25_weight] {s.bm25_weight}")
        elif tok == "graph_weight":
            print(f"[graph_weight] {s.graph_weight}")
        elif tok == "web_search":
            _ws_display = (
                "web_only"
                if getattr(s, "retrieve_mode", None) == "WEB"
                else ("local_and_web" if s.web_search else "local_only")
            )
            print(f"[web_search] {_ws_display}")
        elif tok == "web_weight":
            print(f"[web_weight] {s.web_weight}")
        elif tok == "web_rerank_threshold":
            val = getattr(s, "web_rerank_threshold", None)
            if val is None:
                val = self.cfg.get_float("_WEB_SEARCH.rerank_threshold") or 0.0
            print(f"[web_rerank_threshold] {val}")
        elif tok == "fetch_page_content":
            print(
                f"[fetch_page_content] {'fetch pages' if s.fetch_page_content else 'snippets only'}"
            )
        elif tok == "rerank":
            print(f"[rerank] {s.rerank}")
        elif tok == "collection":
            print(f"[collection] {s.collection_name}")
        elif tok == "use_chat_context":
            print(f"[use_chat_context] {s.use_chat_context}")
        elif tok == "history_keep":
            print(f"[history_keep] {s.turns}")
        elif tok == "history_prune":
            print(f"[history_prune] {s.prune_batch}")
        elif tok == "rewrite_context":
            print(f"[rewrite_context] {s.max_history_turns}")
        elif tok == "topic_summary":
            print(f"[topic_summary] {s.topic_summary_mode}")
        elif tok == "retrieve_mode":
            print(f"[retrieve_mode] {s.retrieve_mode}")
        elif tok == "terminal_line_size":
            print(f"[terminal_line_size] {self._resolve_terminal_line_size()}")
        elif tok == "debug_level":
            _dm = getattr(s, "debug_mode", "ge") or "ge"
            print(f"[debug_level] {int(s.debug_level or 0)}  mode={_dm!r}")
        elif tok == "debug_mode":
            _dm = getattr(s, "debug_mode", "ge") or "ge"
            print(f"[debug_mode] {_dm!r}")
        elif tok == "mark_text":
            print(f"[mark_text] {bool(getattr(s, 'mark_text', False))}")
        elif tok == "help":
            self.help()
        else:
            print(f"⚠ Invalid command: {tok}")
            self.help()

    # ——— Resetting on collection change ———

    def reset_things(self):
        if self.session.file_name or self.session.file_path:
            print(f"⚠ Clearing current file name / file path")
            self.session.file_name, self.session.file_path = None, None
            self.session.file_path_select = None
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Query"
        )
        self.hist.load(
            f"{self.session.collection_name}_{self.session.chat_name}", "Settings"
        )

    # ——— Setting values via "=" ———

    def _set(self, tok: str, p: str):
        if p is None:  # type: ignore[reportUnnecessaryComparison]
            print("No value specified\n")
            return
        if tok in ("file", "path"):
            fn, fp = os.path.basename(p), os.path.abspath(p)
            self.fileList.set(
                f"{self.session.collection_name}_{self.session.chat_name}",
                "File",
                fn,
                fp,
            )
            self.session.file_name = fn if tok == "file" else None
            self.session.file_path = fp if tok == "path" else None
            self.session.file_path_select = tok.lower()
            return

        if tok == "strategy":
            if p.upper() not in self.allowed_strategies:
                print(
                    f"⚠ Invalid strategy '{p.upper()}'. Allowed: {self.allowed_strategies}"
                )
            else:
                # delegate to helper so defaults and side effects are consistent
                self._read_and_apply_value(
                    "strategy", p.upper(), from_interactive=False
                )
            return

        if tok == "collection":
            if self.session.collection_name:
                self.hist.save(
                    f"{self.session.collection_name}_{self.session.chat_name}",
                    "Settings",
                )
            self.session.collection_name = p
            self.ragChatImpl.set_vector_store(self.session)
            self.reset_things()
            return

        if tok == "terminal_line_size":
            try:
                val = int(p)
            except ValueError:
                print(f"\u26a0 terminal_line_size must be an integer, got {p!r}")
                return
            if val < 40:
                print(f"\u26a0 terminal_line_size={val} is too small (minimum 40).")
                return
            self.cfg.set("TERMINAL_LINE_SIZE", val, force=True)
            return

        if tok == "debug_level":
            try:
                level, mode = DebugHelper.parse(p)
            except ValueError as e:
                print(f"⚠ {e}")
                return
            allowed_values: list[int] = list(self.allowed_debug_levels.values())
            if level != 0 and level not in allowed_values:
                allowed_names = "  ".join(
                    f"{v}={k}" for k, v in self.allowed_debug_levels.items() if v != 0
                )
                print(
                    f"⚠ {level} is not a recognised debug level. Allowed: {allowed_names}"
                )
                return
            self.session.debug_level = level
            self.session.debug_mode = mode
            _combined = "none" if level == 0 else f"{mode} {level}"
            self.cfg.set("DEBUG_LEVEL", _combined, force=True)
            return

        if tok == "debug_mode":
            v = p.strip().lower()
            if v not in ("ge", "is", "le"):
                print(
                    f"⚠ Invalid debug_mode '{p}'. Allowed: ge  (>=)   is  (==)   le  (<=)"
                )
                return
            self.session.debug_mode = v
            level = self.session.debug_level or 0
            _combined = "none" if level == 0 else f"{v} {level}"
            self.cfg.set("DEBUG_LEVEL", _combined, force=True)
            return

        if tok in (
            "chat_name",
            "fetch_k",
            "context_chunks",
            "threshold",
            "max_output_tokens",
            "context_size",
            "temperature",
            "top_p",
            "top_k",
            "file_cap",
            "rerank",
            "vector_weight",
            "bm25_weight",
            "graph_weight",
            "web_search",
            "web_weight",
            "web_rerank_threshold",
            "fetch_page_content",
            "use_chat_context",
            "history_keep",
            "history_prune",
            "rewrite_context",
            "retrieve_mode",
            "mark_text",
        ):
            try:
                self._read_and_apply_value(tok, p, from_interactive=False)
            except ValueError:
                print(f"⚠ Invalid command: {tok}")
                self.help()
            return

    # ——— Clearing values ———

    def _clear(self, tok: str):
        s = self.session
        if tok in ("file", "path"):
            s.file_name, s.file_path = None, None
            s.file_path_select = None
        elif tok == "max_output_tokens":
            s.max_output_tokens_override = None
            print("[max_output_tokens] override cleared — dynamic budget will be used")
        elif tok == "context_size":
            s.context_size_override = None
            print(
                f"[context_size] override cleared — auto={self.tokenBudget.get_context_limit()} will be used"
            )

    # ——— Defaults loader for strategies ———

    def applyStrategyDefaults(
        self, strategy: str, session: "Session | None" = None
    ) -> str:
        """Public entry point for loading strategy defaults into Session. Returns the collection name."""
        return self._base_defaults(strategy, session=session)

    def _base_defaults(self, strategy: str, *, session: "Session | None" = None) -> str:
        """Read strategy config values and apply them to the session. Returns the collection name."""
        s = session or self.session
        key = strategy.upper()
        s.strategy = key  # record which strategy is active
        chunks_win = self.cfg.get_int(f"_STRATEGIES.{key}.final_chunks_to_llm")
        chroma_kval = self.cfg.get_int(f"_STRATEGIES.{key}.retriever_k")
        thr = self.cfg.get_float(f"_STRATEGIES.{key}.threshold")
        tok_val = self.cfg.get_int(f"_STRATEGIES.{key}.max_output_tokens")
        temp = self.cfg.get_float(f"_STRATEGIES.{key}.temperature")
        top_p = self.cfg.get_float(f"_STRATEGIES.{key}.top_p")
        top_k = self.cfg.get_int(f"_STRATEGIES.{key}.top_k")
        rer = self.cfg.get_int(f"_STRATEGIES.{key}.rerank")
        vw = self.cfg.get_float(f"_STRATEGIES.{key}.vector_weight")
        bw = self.cfg.get_float(f"_STRATEGIES.{key}.bm25_weight")
        gw = self.cfg.get_float(f"_STRATEGIES.{key}.graph_weight")
        ww = self.cfg.get_float(f"_STRATEGIES.{key}.web_weight")
        if self.cfg.get("COLLECTION") is not None:
            cn = self.cfg.get_str("COLLECTION")
        else:
            cn = self.cfg.get_str(f"_STRATEGIES.{key}.collection")
        fl = self.cfg.get_int(f"_STRATEGIES.{key}.filelim")
        use_ctx = self.cfg.get_bool(f"_STRATEGIES.{key}.use_chat_context")
        turn = self.cfg.get_int(f"_STRATEGIES.{key}.turns")
        bs = self.cfg.get_int(f"_STRATEGIES.{key}.prune_batch")
        mht = self.cfg.get_int(f"_STRATEGIES.{key}.max_history_turns")
        tsm = self.cfg.get_str(f"_STRATEGIES.{key}.TOPIC_SUMMARY_MODE") or "last"
        sm = self.cfg.get_str(f"_STRATEGIES.{key}.retrieve_mode") or "VECTOR"

        s.final_chunks_to_llm = chunks_win
        s.retriever_k = chroma_kval
        s.chroma_threshold = thr
        s.max_output_tokens = tok_val
        s.temperature = temp
        s.top_p = top_p
        s.top_k = top_k
        s.rerank = bool(rer)
        s.vector_weight = vw or 1.0
        s.bm25_weight = bw or 1.0
        s.graph_weight = gw or 1.0
        # web_weight can be pre-set per-strategy; web_search and fetch_page_content
        # are session-persistent knobs set deliberately by the user or via OpenWebUI
        # Advanced Parameters — strategy loading does not reset them.
        if ww:
            s.web_weight = ww
        s.per_file_limit = fl
        s.use_chat_context = use_ctx
        s.turns = turn
        s.prune_batch = bs
        s.max_history_turns = mht
        s.topic_summary_mode = tsm
        s.retrieve_mode = sm.upper()

        # Always use debug_level from Config_Global.py (respects CLI override if present)
        _debug_str = self.cfg.get_str("DEBUG_LEVEL", "none")
        _lvl, _mode = DebugHelper.parse(_debug_str)
        s.debug_level = _lvl
        s.debug_mode = _mode

        return cn

    def _defaults(self, tok: str, p: str, init_once: bool = True):
        if tok == "strategy":
            if p.upper() not in self.allowed_strategies:
                print(
                    f"⚠ Invalid value {p.upper()}. Defaults can be set only for strategies. Allowed: {self.allowed_strategies}"
                )
                return

            self.session.strategy = p
            cn = self._base_defaults(p)

            if self.session.collection_name and self.session.chat_name:
                self.hist.save(
                    f"{self.session.collection_name}_{self.session.chat_name}",
                    "Settings",
                )
            self.session.collection_name = cn
            self.reset_things()

            if init_once:
                print(f"→ Loaded defaults for '{p}':")
                self.print_values()
        else:
            print(f"⚠ Default (*) only allowed for strategy. Use e.g. strategy*default")
        return

    # ——— Help ———

    def command_help(self):
        print(
            f"⚙️{BRIGHT_BLUE}  Setup phase: enter query modifiers.  ? show, = set, * defaults, ! ask (picker), - unset (file and path) {RESET}"
        )
        print("   Values: see below")
        print("   strategy! gives you an inline ←/→ picker")
        print(
            "   file! or path! gives you an inline ←/→ picker based upon your history"
        )
        print(
            f"{YELLOW}\n   Quick defaults for strategies: narrow*, balanced_file_cap*, default*, wide* \n\n{RESET}"
        )

    def help(self):
        # Validate all entries have a section
        for key, meta in self.prompt_map.items():
            if "section" not in meta:
                raise ValueError(f"Missing 'section' for key: {key}")

        grouped: dict[str, list[tuple[str, str]]] = {}
        for key, meta in self.prompt_map.items():
            section = meta["section"]
            grouped.setdefault(section, []).append((key, meta["prompt"]))

        first = True
        for section_name in sorted(grouped):
            if not first:
                self.pretty.write("N", "", "")
            self.pretty.write("A", f"{section_name}", "")
            for key, prompt in grouped[section_name]:
                self.pretty.write("O", f"{key}", f"{prompt}")
            first = False
        self.command_help()

    def _validate_raw_for_type(self, raw: str, type_name: str) -> None:
        t = type_name or "str"
        v = raw.strip()
        if t == "int":
            if not v.lstrip("-").isdigit():
                raise ValueError("Must be an integer")
            return
        if t == "debug_level":
            try:
                DebugHelper.parse(v)
            except ValueError as e:
                raise ValueError(str(e))
            return
        if t == "float":
            try:
                float(v)
            except ValueError:
                raise ValueError("Must be a number")
            return
        if t == "bool":
            if v.lower() not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
                raise ValueError("Must be a boolean (true/false, yes/no, 1/0)")
            return
        # string: non-empty
        if v == "":
            raise ValueError("Value cannot be empty")

    def _read_and_apply_value(
        self, tok: str, raw: str, *, from_interactive: bool = False
    ):
        """
        Validate, cast and set the session attribute for tok.
        If tok is file/path this helper will not handle file list logic; callers handle pickers and file parsing.
        """
        spec = self.COMMAND_SPECS.get(tok, {})
        expected_type = spec.get("type", "str")
        # Validate
        try:
            self._validate_raw_for_type(raw, expected_type)
        except ValueError as e:
            # Interactive callers should raise ValidationError for inquirer
            if from_interactive:
                raise ValidationError(message=str(e))
            # Non-interactive callers get a ValueError
            raise

        # Cast and set
        val_cast = self._cast_value(tok, raw.strip())
        attr = self._get_attr_name(tok)
        if attr:
            setattr(self.session, attr, val_cast)
            # special side effects
            if tok == "chat_name":
                self.reset_things()
            if tok == "strategy":
                # ensure defaults are applied when setting strategy via '='
                strategy: str = val_cast
                strategy.upper()
                self._base_defaults(strategy)
            if tok == "retrieve_mode":
                allowed = self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                if val_cast.upper() not in allowed:
                    msg = f"Invalid retrieve_mode '{val_cast}'. Allowed: {', '.join(allowed)}"
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"⚠ {msg}")
                    return
                setattr(self.session, attr, val_cast.upper())
            if tok == "web_search":
                _ws_raw = val_cast.strip().lower()
                if _ws_raw not in ("local_only", "local_and_web", "web_only"):
                    msg = f"Invalid web_search value '{val_cast}'. Allowed: local_only, local_and_web, web_only."
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"\u26a0 {msg}")
                    return
                want_web = _ws_raw in ("local_and_web", "web_only")
                _web_mode = str(os.environ.get("WEB_SEARCH_MODE", "0")).strip().lower()
                if want_web and _web_mode != "1":
                    setattr(self.session, attr, False)  # undo premature general setattr
                    msg = 'Web search is disabled by administrator (WEB_SEARCH_MODE must be "1" in Config_Internet_Env.py).'
                    print(f"\u26a0 {msg}")
                    return
                # Apply: web_only sets retrieve_mode=WEB; local_and_web/local_only clear it if currently WEB.
                setattr(self.session, attr, want_web)
                if _ws_raw == "web_only":
                    self.session.retrieve_mode = "WEB"
                elif getattr(self.session, "retrieve_mode", None) == "WEB":
                    # Switching away from web_only — restore default retrieve_mode.
                    allowed_rm = self.cfg.get_list("_ALLOWED_RETRIEVE_MODES")
                    self.session.retrieve_mode = (
                        "ALL" if "ALL" in allowed_rm else allowed_rm[0]
                    )
            if tok == "fetch_page_content":
                val = val_cast.strip().lower()
                _allowed_fpc = ("snippets only", "fetch pages")
                if val not in _allowed_fpc:
                    msg = f"Invalid fetch_page_content value '{val_cast}'. Allowed: {', '.join(_allowed_fpc)}"
                    if from_interactive:
                        raise ValidationError(message=msg)
                    print(f"\u26a0 {msg}")
                    return
                setattr(self.session, attr, val == "fetch pages")
            if tok == "context_size":
                cap = self.tokenBudget.get_context_limit()
                if val_cast > cap:
                    print(
                        f"{YELLOW}⚠ context_size={val_cast} exceeds hardware cap={cap}; "
                        f"Chatter will clamp to {cap}{RESET}"
                    )
                # Warn when num_ctx is so small that the full prompt
                # (template + retrieved context + query) cannot fit.
                # 256 is the absolute floor; below 2048 things get tight.
                if val_cast <= 256:
                    print(
                        f"{ORANGE}⚠ context_size={val_cast} is dangerously low. "
                        f"Ollama uses this as the total KV-cache for input + output. "
                        f"The full prompt (instructions, retrieved context, and query) "
                        f"will be physically truncated — the model will never see "
                        f"the grounding instructions or the retrieved documents and "
                        f"will hallucinate from its training data instead.{RESET}"
                    )
                elif val_cast < 2048:
                    print(
                        f"{YELLOW}⚠ context_size={val_cast} is tight. The full prompt "
                        f"(instructions + retrieved context + query) plus the model's "
                        f"reply must all fit within {val_cast} tokens. Key instructions "
                        f"or context may be truncated, causing the model to hallucinate "
                        f"rather than answer from the provided documents.{RESET}"
                    )
        else:
            raise ValueError(f"Invalid command: {tok}")

    # ——— Bool helpers ———

    def to_bool(self, value: int | str | bool) -> bool:
        if isinstance(value, (bool, int)):
            return bool(value)
        return str(value).strip().lower() in ("true", "1", "yes", "y")
