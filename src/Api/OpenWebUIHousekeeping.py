from __future__ import annotations

from typing import Literal

_OPENWEBUI_FOLLOWUP_PREFIX = "### Task:\nSuggest 3-5 relevant follow-up questions"
_OPENWEBUI_TAGS_PREFIX = "### Task:\nGenerate 1-3 broad tags"
_OPENWEBUI_TITLE_PREFIXES = (
    "### Task:\nGenerate a concise, 3-5 word title",
    "### Task:\nGenerate a concise title summarizing the chat history.",
)

HousekeepingKind = Literal["follow_up", "tags", "title"]


class OpenWebUIHousekeeping:
    """Classify synthetic OpenWebUI prompts that bypass the RAG request path."""

    @staticmethod
    def classify(last_user_text: str | None) -> HousekeepingKind | None:
        if last_user_text is None:
            return None
        if last_user_text.startswith(_OPENWEBUI_FOLLOWUP_PREFIX):
            return "follow_up"
        if last_user_text.startswith(_OPENWEBUI_TAGS_PREFIX):
            return "tags"
        if last_user_text.startswith(_OPENWEBUI_TITLE_PREFIXES):
            return "title"
        return None

    @staticmethod
    def is_terminal_navigation(last_user_text: str | None) -> bool:
        return (last_user_text or "").strip().lower() in ("b", "b:")
