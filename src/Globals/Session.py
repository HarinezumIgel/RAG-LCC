# Local module imports
from typing import Any, Optional

from AI.AIHelpers import AIHelpers
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter


class Session(SingletonMixin):

    def __init__(self) -> None:
        # Only initialize once
        if self._initialized:
            return
        self._initialized = True

        # your original setup—runs only on the very first Session() call
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.aiHelpers: AIHelpers = AIHelpers()

        self.file_name: Optional[str | None] = None
        self.file_path: Optional[str | None] = None
        self.file_path_select: Optional[str | None] = (
            None  # Internal flag only, will not be shown to user
        )
        self.query: str | None = None
        self.strategy: str | None = None
        self.chroma_k_value: int | None = None
        self.rerank: bool | None = None
        self.chunks_window: int | None = None
        self.chroma_threshold: float | None = None
        self.per_file_limit: int | None = None
        self.use_chat_context: bool | None = None
        self.chat_context_k_value: int | None = None
        self.turns: int | None = None
        self.batch_size: int | None = None
        self.chroma_weight: float | None = None
        self.chat_name: str = self.cfg.get_str("_DEFAULT_CHAT_NAME")
        self.max_output_tokens: int | None = None
        self.max_output_tokens_override: int | None = None
        self.context_size_override: int | None = None
        self.temperature: float | None = None
        self.top_k: float | None = None
        self.top_p: float | None = None
        self.base_kwargs: dict[str, Any] | None = None
        self.collection_name: str | None = None
        self.debug_level: int | None = None

    def export_session_state_as_cell(self, max_items_per_line: int = 6) -> str:
        """
        Return all session attributes as a single CSV-safe cell.
        No commas or delimiters are added. Optional line breaks
        keep the cell readable without breaking CSV structure.
        """

        # Collect only the attributes you explicitly defined
        keys = [
            "file_name",
            "file_path",
            "view_doc",
            "query",
            "strategy",
            "chroma_k_value",
            "rerank",
            "chunks_window",
            "chroma_threshold",
            "per_file_limit",
            "use_chat_context",
            "chat_context_k_value",
            "turns",
            "batch_size",
            "chroma_weight",
            "chat_name",
            "max_output_tokens",
            "temperature",
            "base_kwargs",
            "collection_name",
            "debug",
        ]

        # Build "key=value" pairs
        items: list[str] = []
        for key in keys:
            value = getattr(self, key, None)
            items.append(f"{key}={value}")

        # Insert line breaks every N items
        lines: list[str] = []
        for i in range(0, len(items), max_items_per_line):
            chunk = " | ".join(items[i : i + max_items_per_line])
            lines.append(chunk)

        # Final output: one single CSV cell
        return "\n".join(lines)
