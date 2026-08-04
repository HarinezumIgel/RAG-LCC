"""Interactive metadata filter picker for RAGChat.

Reads the active ChromaDB collection's chunk metadata, aggregates the
available fields and their distinct values, and lets the user pick one or
more field/value pairs. The result is a ``{field: value}`` dict stored on the
session and applied as an extra ChromaDB ``where`` condition during retrieval.
"""

import os
from typing import Any

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]
from InquirerPy.base.control import Choice  # type: ignore[attr-defined]

inquirer: Any = _inquirer

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper

# Pipeline/internal metadata fields that must not be offered as filters.
_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "MyChunk",
        "FileHash",
        # FileName/FilePath are covered by the dedicated file/path filter;
        # offering them here would be redundant.
        "FileName",
        "FilePath",
        "chunk_id",
        "id",
        "retriever_sources",
        "chroma_score",
        "chroma_sim",
        "dist",
        "rerank_score",
        "raw_rerank_score",
        "rrf_score",
        "bm25_score",
        "graph_score",
        "position",
        "HeadingPath",
        "snippet",
    }
)

# Max chunks sampled when aggregating available metadata values.
_SAMPLE_LIMIT: int = 5000
# Sentinel returned by the value picker when the user wants to type a value.
_CUSTOM = "\0__custom__\0"


class MetadataPicker(SingletonMixin):
    """Thread-safe singleton that drives the interactive metadata filter UI."""

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()

    def pick(
        self, collection_name: "str | None", current: dict[str, str]
    ) -> dict[str, str]:
        """Return updated metadata filters after interactive selection.

        Existing *current* filters are preserved and used as defaults; the user
        may add, change, or leave them untouched.
        """
        fields = self._collect_fields(collection_name)
        if not fields:
            print("⚠ No filterable metadata found in this collection.")
            return current

        key_choices = [
            Choice(key, name=f"{key}  ({len(values)} value(s))")
            for key, values in sorted(fields.items())
        ]
        selected_keys: list[str] = (
            inquirer.checkbox(
                message="Select metadata fields to filter on:",
                choices=key_choices,
                instruction="Space to toggle, Enter to confirm",
            ).execute()
            or []
        )
        if not selected_keys:
            return current

        result: dict[str, str] = dict(current)
        for key in selected_keys:
            value = self._pick_value(key, sorted(fields[key]), current.get(key))
            if value:
                result[key] = value
        return result

    def _pick_value(
        self, key: str, values: list[str], default: "str | None"
    ) -> "str | None":
        choices = [Choice(v, name=v) for v in values]
        choices.append(Choice(_CUSTOM, name="⌨  type a custom value"))
        picked: str = inquirer.select(
            message=f"Value for '{key}':",
            choices=choices,
            default=default if default in values else None,
        ).execute()
        if picked == _CUSTOM:
            picked = inquirer.text(message=f"Enter value for '{key}':").execute()
        picked = (picked or "").strip()
        return picked or None

    def _collect_fields(self, collection_name: "str | None") -> dict[str, set[str]]:
        """Aggregate ``{field: {values}}`` from the collection's chunk metadata."""
        try:
            name, persist_dir = self.chromaDBHelper.change_chroma_collection(
                collection_name
            )
            if not persist_dir or not os.path.isdir(persist_dir):
                return {}
            _, collection = self.chromaDBHelper.get_chroma_client_and_collection(
                persist_dir, name
            )
            data: dict[str, Any] = collection.get(
                include=["metadatas"], limit=_SAMPLE_LIMIT
            )
        except Exception as exc:
            self.pretty.write(
                "W", "MetadataPicker", f"Could not read collection metadata: {exc}"
            )
            return {}

        metadatas: list[Any] = data.get("metadatas") or []
        fields: dict[str, set[str]] = {}
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            for key, value in meta.items():
                if key in _EXCLUDED_KEYS or value is None:
                    continue
                text = str(value).strip()
                if text:
                    fields.setdefault(str(key), set()).add(text)
        return fields
