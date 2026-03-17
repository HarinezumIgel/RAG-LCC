import os
from typing import Any

from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter

try:
    import readline as _readline  # type: ignore[attr-defined]
except ImportError:
    import pyreadline3 as _readline  # type: ignore[attr-defined,import-untyped]

readline: Any = _readline


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

    def load(self, collection: str, mode: str):
        """Clear Python's readline buffer and load history from a file.
        If the file does not exist, create it and pass the filename to any exception.
        """
        readline.clear_history()

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

        # Load history with explicit filename in the exception
        try:
            readline.read_history_file(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"History file not found: {path}") from exc
        except OSError as exc:
            raise OSError(f"Could not read history file: {path}") from exc

        readline.set_history_length(self.max_length)

    def save(self, collection: str, mode: str):
        """Write whatever’s in readline buffer back to disk."""
        path = f"{self.path}\\{collection}_{mode}.txt"
        readline.write_history_file(path)
