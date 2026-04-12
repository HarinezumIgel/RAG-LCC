import os

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory

from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter


class HistoryManager:
    def __init__(self, max_length: int = 1000) -> None:
        """
        path: filename for this history (e.g. "~/.settings_history").
        max_length: how many entries to keep.
        """
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        # pick your filenames however you like (config, env, defaults…)
        self.path: str = self.cfg.get_str("_HISTORY_DIRECTORY")
        self.max_length: int = max_length
        # ensure directories exist
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._session: PromptSession[str] | None = None

    def load(self, collection: str, mode: str) -> None:
        """Create a new prompt session backed by a file history for this collection/mode."""
        path = f"{self.path}\\{collection}_{mode}.txt"
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        # Create file if missing
        if not os.path.exists(path):
            try:
                with open(path, "a", encoding="utf-8"):
                    pass
            except OSError as exc:
                raise OSError(f"Failed to create history file: {path}") from exc
            self.pretty.write("I", "HistoryManager", f"Created history file {path}")

        self._session = PromptSession[str](history=FileHistory(path))

    def prompt(self, prompt_text: str = "") -> str:
        """Show prompt with arrow-key history. Falls back to input() if no session loaded."""
        if self._session is None:
            return input(prompt_text)
        return self._session.prompt(ANSI(prompt_text))

    def save(self, collection: str, mode: str) -> None:
        """No-op: prompt_toolkit's FileHistory auto-saves on each entry."""
        _ = collection, mode
