import os
from dataclasses import dataclass
from typing import Any, List, Optional

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]
from InquirerPy.base.control import Choice  # type: ignore[attr-defined]

inquirer: Any = _inquirer
from Commons.SingletonMixin import SingletonMixin


# ——— record for each file entry, now scoped by store ———
@dataclass
class FileEntry:
    store: str
    file: str  # filename only (no path)
    path: str  # full filesystem path (local files only, no http:// or https://)
    usage: str  # "File" or "DocSelect"


class FileList(SingletonMixin):
    """Tracks local file history for interactive file/path picker commands.

    Only local filesystem paths are stored. Web URLs (http://, https://) are
    automatically rejected since they cannot be used for ChromaDB collection
    filtering (file= and path= commands scope retrieval to specific local files).
    """

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.entries: List[FileEntry] = []
        self.cur: Optional[FileEntry] = None
        self.current_store: Optional[str] = None
        self.usage: str = "File"

    def set_usage(self, usage: str):
        if usage not in ("File", "DocSelect"):
            raise ValueError("usage must be 'File' or 'DocSelect'")
        self.usage = usage

    def set(
        self,
        store: str,
        usage: str,
        file: str,
        path: str,
    ):
        if usage not in ("File", "DocSelect"):
            raise ValueError("usage must be 'File' or 'DocSelect'")

        # Normalize early
        file_stripped: str = file.strip()
        path_stripped: str = path.strip()

        # Reject empty/whitespace or Chat Context entries
        if not file_stripped or not path_stripped:
            return
        file = file_stripped
        path = path_stripped

        # Reject web URLs — file history is only for local files that can be
        # searched in ChromaDB collections. Web sources come from live queries.
        if path.startswith("http://") or path.startswith("https://"):
            return

        # Deduplication check
        if not any(
            e.store == store and e.file == file and e.path == path and e.usage == usage
            for e in self.entries
        ):
            entry = FileEntry(store, file, path, usage)
            self.entries.append(entry)
            self.cur = entry
            self.current_store = store

    def get_current(self) -> tuple[str, str]:
        if self.cur is None:
            return ("N/A", "N/A")
        return (self.cur.file, self.cur.path)

    def clear_current(self) -> None:
        """Drop the active file selection so get_current() reports nothing."""
        self.cur = None
        self.current_store = None

    def select_filename_inline(
        self, store: Optional[str] = None
    ) -> Optional[FileEntry]:
        store = store or self.current_store
        if not store:
            raise RuntimeError("No store specified")
        choices: list[dict[str, Any]] = [
            {"name": f"{e.file} @ {e.path}", "value": e}
            for e in self.entries
            if e.store == store and e.usage == "File"
        ]
        if not choices:
            print("⚠ No history for store", store)
            return None
        return inquirer.select(
            message=f"Select from history (store={store}):",
            choices=choices,
            instruction="Use ↑/↓ to navigate, Enter to pick",
        ).execute()

    def _browse_and_select(
        self, store: Optional[str] = None, start_path: Optional[str] = None
    ) -> List[FileEntry]:
        store = store or self.current_store
        if not store:
            raise RuntimeError("No store specified")
        current = os.path.abspath(start_path or ".")
        selected: List[FileEntry] = []

        while True:
            actions = [
                Choice("Select files here", "select"),
                Choice("Go into subdirectory", "down"),
            ]
            if os.path.dirname(current) != current:
                actions.append(Choice("Go up to parent", "up"))
            actions.append(Choice("Finish", "done"))

            action = inquirer.select(
                message=f"📁 Current: {current}", choices=actions
            ).execute()

            if action == "select":
                files = [
                    f
                    for f in os.listdir(current)
                    if os.path.isfile(os.path.join(current, f))
                    and not f.startswith(".")
                ]
                if not files:
                    print("  (no files here)")
                    continue
                picks = inquirer.checkbox(
                    message="Tick files to process:", choices=files
                ).execute()
                for fname in picks:
                    fpath = os.path.join(current, fname)
                    entry = FileEntry(store, fname, fpath, "DocSelect")
                    if not any(
                        e.store == store
                        and e.file == fname
                        and e.path == fpath
                        and e.usage == "DocSelect"
                        for e in self.entries
                    ):
                        self.entries.append(entry)
                    selected.append(entry)

            elif action == "down":
                dirs = [
                    d
                    for d in os.listdir(current)
                    if os.path.isdir(os.path.join(current, d)) and not d.startswith(".")
                ]
                if not dirs:
                    print("  (no subdirectories here)")
                    continue
                choice = inquirer.select(
                    message="Choose a subdirectory:", choices=dirs
                ).execute()
                current = os.path.join(current, choice)

            elif action == "up":
                current = os.path.dirname(current)

            else:  # done
                break

        return selected

    def select(
        self,
        store: Optional[str] = None,
        usage: Optional[str] = None,
        start_path: Optional[str] = None,
    ) -> Optional[List[FileEntry]]:
        if store is not None:
            self.current_store = store
        if usage is not None:
            self.set_usage(usage)

        if self.usage == "File":
            pick = self.select_filename_inline(store=store)
            return [pick] if pick else None

        if self.usage == "DocSelect":
            return self._browse_and_select(store=store, start_path=start_path)

        raise ValueError(f"Unknown usage: {self.usage}")
