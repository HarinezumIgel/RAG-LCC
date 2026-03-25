import os
from typing import Any, List, Optional

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]

inquirer: Any = _inquirer

from Config.Config import Config


class CollectionPicker:
    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.base_dir: str = self.cfg.get_str("_CHROMA_DB_DIR")
        if not self.base_dir or not os.path.isdir(self.base_dir):
            raise ValueError(
                f"⚠ Invalid Chroma DB Dir: {self.base_dir}. Choose another collection or create one with RAGLoad.py --doc-dir <dir> --collection <name>"
            )

    def _list_collections(self) -> List[str]:
        return sorted(
            [
                name
                for name in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, name))
                and not name.startswith(".")
                and not name.endswith("_ChatContext")
            ]
        )

    def pick_collection(self) -> Optional[str]:
        collections = self._list_collections()
        if not collections:
            print("⚠ No collections found in _CHROMA_DB_DIR.")
            return None
        return inquirer.select(
            message="📚 Pick a Chroma collection:",
            choices=collections,
            instruction="Use ↑/↓ to navigate, Enter to pick",
        ).execute()
