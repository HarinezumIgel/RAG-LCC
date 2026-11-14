import os
from typing import Any, List, Optional

from InquirerPy import inquirer as _inquirer  # type: ignore[attr-defined]

inquirer: Any = _inquirer

from Config.Config import Config
from Gui.Picker import PickerCancelled, pick_value
from Helpers.Helpers import Helpers


class CollectionPicker:
    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.helpers: Helpers = Helpers()
        self.base_dir: str = self.cfg.get_str("_CHROMA_DB_DIR")
        if not self.base_dir or not os.path.isdir(self.base_dir):
            raise ValueError(
                f"⚠ Invalid Chroma DB Dir: {self.base_dir}. Choose another collection or create one with RAGLoad.py --doc-dir <dir> --collection <name>"
            )

    def _list_collections(self) -> List[str]:
        return self.helpers.active_embedder_collections(self.base_dir)

    def pick_collection(self) -> Optional[str]:
        active = self.cfg.get_str("_ACTIVE_EMBED", "")
        print(
            f"ℹ Showing only collections created with the active embedder "
            f"'{active}' (_ACTIVE_EMBED in Config_Models.py)."
        )
        collections = self._list_collections()
        if not collections:
            print(
                f"⚠ No collections for embedder '{active}' found in _CHROMA_DB_DIR. "
                "Create one with RAGLoad.py or switch _ACTIVE_EMBED."
            )
            return None
        try:
            return pick_value(
                inquirer,
                message="📚 Pick a Chroma collection:",
                choices=[
                    {"name": f"{collection} (embedder: {active})", "value": collection}
                    for collection in collections
                ],
                instruction="Use ↑/↓ to navigate, Enter to pick",
            )
        except PickerCancelled:
            return None
