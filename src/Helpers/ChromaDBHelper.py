# Local module imports
import os
from typing import Any, Tuple

from chromadb.api import ClientAPI
from chromadb.config import Settings

from chromadb import Collection, PersistentClient
from Commons.Exceptions import (ChromaInstallCurrentEmbeddingsMismatch,
                                EmbedModelMismatch, InvalidCollectionName)
from Config.Config import Config
from Gui.Colors import GREEN, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers


class ChromaDBHelper:
    def __init__(self) -> None:
        # Initialize helper instances
        self.pretty: PrettyWriter = PrettyWriter()
        self.helpers: Helpers = Helpers()
        self.cfg: Config = Config()
        self.fileUtils: FileUtils = FileUtils()
        self.persist_dir: str = self.cfg.get_str("_CHROMA_DB_DIR")
        self.collection_name: str = ""

    def _disable_chroma_db_telemetry(self) -> Settings:
        return Settings(anonymized_telemetry=False)

    def _persistent_chroma_client(self, persist_dir: str) -> ClientAPI:
        settings: Settings = self._disable_chroma_db_telemetry()
        return PersistentClient(path=persist_dir, settings=settings)

    def get_or_create_collection(
        self, client: ClientAPI, collection_name: str
    ) -> Collection:
        """
        Get or create a Chroma collection with HNSW index settings optimized for cosine similarity.
        """
        return client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                # More neighbors explored during index construction (balance accuracy vs speed)
                "hnsw:construction_ef": self.cfg.get_int(
                    f"{self.helpers.get_chroma_config_slot()}.NEIGHBORS_ON_LOAD"
                ),
                # Maximum links per node in HNSW graph (32 is typical)
                "hnsw:M": 32,
                # More neighbors explored during search for better recall
                "hnsw:search_ef": self.cfg.get_int(
                    f"{self.helpers.get_chroma_config_slot()}.NEIGHBORS_RETRIEVE"
                ),
            },
        )

    def chroma_coll_name_and_mkdir_or_del(
        self, mode: str, collection_name: str | None = None
    ) -> Tuple[str, str]:
        """Returns the ChromaDB collection name based on the given chunk selection strategy.
        Depending on the mode, it either creates the directory, deletes it, or does nothing.
        """
        self.collection_name, theDirectory = self.change_chroma_collection(
            collection_name
        )

        theDirectory: str = os.path.join(
            self.cfg.get_str("_CHROMA_DB_DIR"), self.collection_name
        )

        if mode == "delete":
            if os.path.exists(theDirectory):
                if self.fileUtils.delete_file_or_dir(theDirectory):
                    self.pretty.write(
                        "I", "Info", f"Deleted Chroma DB directory {theDirectory}."
                    )
        elif mode == "create":
            if os.makedirs(theDirectory, exist_ok=True):
                self.pretty.write(
                    "I", "Info", f"Created Chroma DB directory {theDirectory}."
                )
        else:
            self.pretty.write(
                "E",
                "Error",
                f"No or wrong {str(mode)} for chroma collection {self.collection_name} given",
            )
        return self.collection_name, theDirectory

    def _validate_collection_name(self, name: str) -> None:
        """
        Raise InvalidCollectionName if *name* looks like a file-system path
        (contains '/', '\\', or ':').
        """
        if any(ch in name for ch in ("/", "\\", ":")):
            self.pretty.write(
                "E",
                "Collection name",
                f"'{name}' looks like a file-system path. "
                f"Please provide a plain collection name without path separators "
                f"(e.g. 'Test', not 'C:\\\\path\\\\to\\\\Test' or './Test').",
                color=RED,
            )
            raise InvalidCollectionName(name)

    def change_chroma_collection(
        self, collection_name: str | None = None, print: bool = False
    ) -> Tuple[str, str]:
        """
        Determine the active Chroma collection name and directory.
        Preference order: session parameter > config COLLECTION > error
        """
        if collection_name is not None:
            # Use session-provided collection name
            self._validate_collection_name(collection_name)
            self.collection_name = collection_name
        elif (
            self.cfg.get("COLLECTION") is not None and self.cfg.get("COLLECTION") != ""
        ):
            # Fall back to configured default
            raw = str(self.cfg.get_str("COLLECTION"))
            self._validate_collection_name(raw)
            self.collection_name = raw
            self.pretty.write(
                "W",
                "Chroma Collection",
                f"Using default collection from ./Configuration/Config_Global.py. Key: COLLECTION value: {self.collection_name}",
            )
        else:
            # No collection specified
            self.pretty.write(
                "E",
                "Chroma Collection",
                "No chroma collection given as default or argument. Pass by argument --collection or define  ./Configuration/Config_Global.py. Key: COLLECTION",
                color=RED,
            )

        if print:
            self.pretty.write(
                "O",
                "Chroma Collection",
                f"Using Chroma DB collection {self.collection_name}",
                color=GREEN,
            )

        theDirectory: str = os.path.join(
            self.cfg.get_str("_CHROMA_DB_DIR"), self.collection_name
        )
        return self.collection_name, theDirectory

    def clean_metadata(self, meta: Any) -> dict[str, str | int | float | bool]:
        """
        Filter and convert metadata for Chroma compatibility.
        Removes None values and converts unsupported types to strings.
        """
        if not isinstance(meta, dict):
            return {}

        meta_dict: dict[str, Any] = dict(meta)  # type: ignore[reportUnknownArgumentType]
        cleaned: dict[str, str | int | float | bool] = {}
        for k, v in meta_dict.items():
            if v is None:
                continue
            # Keep supported types as-is
            if isinstance(v, (bool, int, float, str)):
                cleaned[k] = v
            else:
                # Convert complex types to string representation
                cleaned[k] = str(v)
        return cleaned

    def get_chroma_client_and_collection(
        self, persist_directory: str, collection_name: str, stamp: bool = False
    ) -> Tuple[ClientAPI, Collection]:
        """
        Initialize Chroma client and get or create a collection with cosine similarity.

        If the collection metadata already contains an ``embed_model`` or ``embed_bits``
        key that differs from the current configuration, raises ``EmbedModelMismatch``
        so that a collection built with one embedder is never queried with another.

        When ``stamp=True`` (RAGLoad write path) the current model name and bits are
        written into the collection metadata after a successful compatibility check,
        so that future RAGChat runs can detect mismatches.
        """
        client: ClientAPI = self._persistent_chroma_client(persist_directory)
        collection: Collection = self.get_or_create_collection(client, collection_name)

        # --- embedding compatibility check -----------------------------------
        stored_meta: dict[str, Any] = dict(collection.metadata or {})
        stored_model: str | None = stored_meta.get("embed_model")
        stored_bits: str | None = stored_meta.get("embed_bits")

        current_model: str = self.helpers.get_model_args("_EMBED")["MODEL"]
        current_bits: str = str(self.cfg.get_int("EMBEDDER_BITS", 32))

        mismatch: bool = (
            stored_model is not None and stored_model != current_model
        ) or (stored_bits is not None and stored_bits != current_bits)

        if mismatch:
            self.pretty.write(
                "E",
                "Chroma embedding",
                f"The collection '{collection_name}' was created with "
                f"({stored_model}, {stored_bits} bits). "
                f"Actual config is ({current_model}, {current_bits} bits).",
                color=RED,
            )
            raise EmbedModelMismatch(
                f"Collection '{collection_name}': stored embed=({stored_model} / "
                f"{stored_bits} bits), current config=({current_model} / "
                f"{current_bits} bits)"
            )

        # --- stamp current model + bits into collection metadata -------------
        # Only pass the two custom keys — never HNSW keys, which ChromaDB
        # refuses to modify after collection creation.
        if stamp:
            try:
                collection.modify(
                    metadata={"embed_model": current_model, "embed_bits": current_bits}
                )
            except ValueError as exc:
                self.pretty.write(
                    "E",
                    "Chroma embedding",
                    f"Could not stamp embedding info onto collection '{collection_name}': {exc}. "
                    f"Delete the collection directory and re-run RAGLoad to rebuild it.",
                    color=RED,
                )
                raise ChromaInstallCurrentEmbeddingsMismatch(
                    f"Collection '{collection_name}' metadata update rejected by ChromaDB: {exc}"
                ) from exc

        return client, collection
