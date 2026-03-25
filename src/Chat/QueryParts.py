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
from Gui.Colors import BRIGHT_BLUE, ORANGE, RESET, YELLOW
from Gui.FileList import FileList
from Gui.HistoryManager import HistoryManager
from Gui.PrettyWriter import PrettyWriter
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
    COMMAND_SPECS: dict[str, dict[str, Any]] = {
        "collection": {
            "section": "General",
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
            "section": "General",
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
                "Chooses a predefined retrieval configuration (narrow, medium, wide). "
                "Each strategy sets defaults for chunk selection, thresholds, and sampling parameters. "
                "Use this when you want consistent behavior without manually tuning every value."
            ),
            "mode": "strategy",
            "type": "string",
            "attr": "strategy",
        },
        "chroma_k_value": {
            "section": "Retrieval",
            "prompt": (
                "Defines how many chunks the vector store retrieves before filtering. "
                "Higher values increase recall but may introduce noise. "
                "Lower values improve precision but risk missing relevant context."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "chroma_k_value",
        },
        "chunks_window": {
            "section": "Retrieval",
            "prompt": (
                "Sets the maximum number of chunks kept after all filters and reranking. "
                "This controls how much context the model sees. "
                "A larger window helps with broad questions; a smaller one keeps responses focused."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "chunks_window",
        },
        "threshold": {
            "section": "Retrieval",
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
        "rerank": {
            "section": "Ranking",
            "prompt": (
                "Enables or disables the reranker (0 = off, 1 = on). "
                "When enabled, retrieved chunks are re‑scored using a more accurate cross‑encoder. "
                "This improves relevance at the cost of additional computation."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "rerank",
        },
        "chroma_weight": {
            "section": "Ranking",
            "prompt": (
                "Controls how much weight the vector store’s similarity score has compared to the reranker score. "
                "Higher values favor raw embeddings; lower values favor reranker judgments. "
                "Useful when tuning retrieval behavior for different document types."
            ),
            "mode": "normal",
            "type": "float",
            "attr": "chroma_weight",
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
        "filelim": {
            "section": "File Input",
            "prompt": (
                "Sets the maximum number of chunks allowed per file (medium strategy only). "
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
        "chat_context_k_value": {
            "section": "Chat Context",
            "prompt": (
                "Maximum number of chat‑history chunks retrieved when chat context is enabled. "
                "Higher values allow deeper memory; lower values keep responses focused on recent turns."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "chat_context_k_value",
        },
        "turns": {
            "section": "Chat Context",
            "prompt": (
                "Number of question/answer cycles stored in Chat Context. "
                "Controls how much conversational memory is retained across sessions."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "turns",
        },
        "batch_size": {
            "section": "Chat Context",
            "prompt": (
                "Number of chat items summarized into a new chunk during housekeeping. "
                "Larger values reduce summarization frequency; smaller values keep history more compact."
            ),
            "mode": "normal",
            "type": "int",
            "attr": "batch_size",
        },
        "debug": {
            "section": "Chat Context",
            "prompt": (
                "Sets the verbosity of debug output (0–3). "
                "Higher levels show internal decisions, scoring, and filtering steps. "
                "Useful for diagnosing retrieval behavior or teaching how the pipeline works."
            ),
            "mode": "debug",
            "type": "int",
            "attr": "debug_level",
        },
        "help": {
            "section": "General",
            "prompt": (
                "Displays detailed help for all available commands. "
                "Use help? to show this information at any time."
            ),
            "mode": "help",
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
            rf"^\s*\b(use_chat_context|chat_context_k_value|chroma_k_value|file|path|strategy|chunks_window|treshold|threshold|max_output_tokens|context_size|temperature|top_p|top_k|filelim|rerank|chroma_weight|collection|chat_name|context|turns|batch_size|debug|help)\b"
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
        self._defaults("strategy", self.cfg.get_str("CHUNK_SELECT_STRATEGY"), True)

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
            "chroma_k_value": getattr(s, "chroma_k_value", None),
            "chunks_window": getattr(s, "chunks_window", None),
            "threshold": getattr(s, "chroma_threshold", None),
            "max_output_tokens": (
                f"{getattr(s, 'max_output_tokens', None)}"
                + (
                    f"  [override={s.max_output_tokens_override}]"
                    if getattr(s, "max_output_tokens_override", None) is not None
                    else ""
                )
            ),
            "context_size": (
                f"{self.tokenBudget.get_context_limit()}"
                if getattr(s, "context_size_override", None) is None
                else f"override={s.context_size_override}"
            ),
            "temperature": getattr(s, "temperature", None),
            "top_p": getattr(s, "top_p", None),
            "top_k": getattr(s, "top_k", None),
            "rerank": getattr(s, "rerank", None),
            "chroma_weight": getattr(s, "chroma_weight", None),
            "file": getattr(s, "file_name", None),
            "path": getattr(s, "file_path", None),
            "filelim": getattr(s, "per_file_limit", None),
            "use_chat_context": getattr(s, "use_chat_context", None),
            "chat_context_k_value": getattr(s, "chat_context_k_value", None),
            "turns": getattr(s, "turns", None),
            "batch_size": getattr(s, "batch_size", None),
            "debug": getattr(s, "debug_level", None),
        }

        # Group keys by section
        grouped: dict[str, list[str]] = {}
        for key, meta in self.prompt_map.items():
            section = meta["section"]
            grouped.setdefault(section, []).append(key)

        print("▶")
        for section_name in sorted(grouped):
            line = f"  ▶ {section_name}: "
            for key in grouped[section_name]:
                if key in values:
                    line += f"{key}={values[key]!r}  "
            print(line.strip())

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
            s.debug_level = choice_value
            self.cfg.set(f"DEBUG_LEVEL", choice_value)
            return

        if tok == "collection":
            if s.collection_name:
                self.hist.save(f"{s.collection_name}_{s.chat_name}", "Settings")
            s.collection_name = self.collectionPicker.pick_collection()
            self.ragChatImpl.set_vector_store(s)
            self._reset_things()
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

        if tok in (
            "chroma_k_value",
            "window",
            "threshold",
            "max_output_tokens",
            "context_size",
            "temperature",
            "top_p",
            "top_k",
            "filelim",
            "rerank",
            "chroma_weight",
            "chat_context_k_value",
            "use_chat_context",
            "turns",
            "batch_size",
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
        elif tok == "chunks_window":
            print(f"[chunks_window] {s.chunks_window}")
        elif tok == "chroma_k_value":
            print(f"[chroma_k_value] {s.chroma_k_value}")
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
        elif tok == "filelim":
            print(f"[filelim?] {s.per_file_limit}")
        elif tok == "chroma_weight":
            print(f"[chroma_weight] {s.chroma_weight}")
        elif tok == "rerank":
            print(f"[rerank] {s.rerank}")
        elif tok == "collection":
            print(f"[collection] {s.collection_name}")
        elif tok == "chat_context_k_value":
            print(f"[chat_context_k_value] {s.chat_context_k_value}")
        elif tok == "use_chat_context":
            print(f"[use_chat_context] {s.use_chat_context}")
        elif tok == "turns":
            print(f"[turns] {s.turns}")
        elif tok == "batch_size":
            print(f"[batch_size] {s.batch_size}")
        elif tok == "debug":
            print(f"[debug] {int(s.debug_level or 0)}")
        elif tok == "help":
            self.help()
        else:
            print(f"⚠ Invalid command: {tok}")
            self.help()

    # ——— Resetting on collection change ———

    def _reset_things(self):
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
            self._reset_things()
            return

        if tok == "debug":
            self.cfg.set(f"DEBUG_LEVEL", int(p))
            self._read_and_apply_value("debug", p.upper(), from_interactive=False)
            return

        if tok in (
            "chat_name",
            "chroma_k_value",
            "chunks_window",
            "threshold",
            "max_output_tokens",
            "context_size",
            "temperature",
            "top_p",
            "top_k",
            "filelim",
            "rerank",
            "chroma_weight",
            "chat_context_k_value",
            "use_chat_context",
            "turns",
            "batch_size",
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

    def _base_defaults(self, strategy: str) -> str:
        """Read strategy config values and apply them to the session. Returns the collection name."""
        key = strategy.upper()
        chunks_win = self.cfg.get_int(f"_STRATEGIES.{key}.chunks_window")
        chroma_kval = self.cfg.get_int(f"_STRATEGIES.{key}.chroma_k_value")
        thr = self.cfg.get_float(f"_STRATEGIES.{key}.threshold")
        tok_val = self.cfg.get_int(f"_STRATEGIES.{key}.max_output_tokens")
        temp = self.cfg.get_float(f"_STRATEGIES.{key}.temperature")
        top_p = self.cfg.get_float(f"_STRATEGIES.{key}.top_p")
        top_k = self.cfg.get_int(f"_STRATEGIES.{key}.top_k")
        rer = self.cfg.get_int(f"_STRATEGIES.{key}.rerank")
        cw = self.cfg.get_float(f"_STRATEGIES.{key}.chroma_weight")
        if self.cfg.get("COLLECTION") is not None:
            cn = self.cfg.get_str("COLLECTION")
        else:
            cn = self.cfg.get_str(f"_STRATEGIES.{key}.collection")
        fl = self.cfg.get_int(f"_STRATEGIES.{key}.filelim")
        use_ctx = self.cfg.get_bool(f"_STRATEGIES.{key}.use_chat_context")
        ctx_kval = self.cfg.get_int(f"_STRATEGIES.{key}.chat_context_k_value")
        turn = self.cfg.get_int(f"_STRATEGIES.{key}.turns")
        bs = self.cfg.get_int(f"_STRATEGIES.{key}.batch_size")
        dbg = self.cfg.get_int(f"_STRATEGIES.{key}.debug_level")

        self.session.chunks_window = chunks_win
        self.session.chroma_k_value = chroma_kval
        self.session.chroma_threshold = thr
        self.session.max_output_tokens = tok_val
        self.session.temperature = temp
        self.session.top_p = top_p
        self.session.top_k = top_k
        self.session.rerank = bool(rer)
        self.session.chroma_weight = cw
        self.session.per_file_limit = fl
        self.session.use_chat_context = use_ctx
        self.session.chat_context_k_value = ctx_kval
        self.session.turns = turn
        self.session.batch_size = bs
        self.session.debug_level = dbg
        self.cfg.set("DEBUG_LEVEL", dbg)

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
            self._reset_things()

            if init_once:
                print(f"→ Loaded defaults for '{p}':")
                self.print_values()
        else:
            print(f"⚠ Default (*) only allowed for strategy. Use e.g. strategy*medium")
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
            f"{YELLOW}\n   Quick defaults for strategies: narrow*, medium*, large* \n\n{RESET}"
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
                self._reset_things()
            if tok == "strategy":
                # ensure defaults are applied when setting strategy via '='
                strategy: str = val_cast
                strategy.upper()
                self._base_defaults(strategy)
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
