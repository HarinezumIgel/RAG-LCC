# Local module import
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, cast

from chromadb.api import Collection  # type: ignore
# 3rd party module import
from langchain_core.documents import Document as LangchainDocument

from AI.ModelsCache import ModelsCache
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Session import Session
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.DebugHelper import DebugHelper


class ChatContext(SingletonMixin):

    def __init__(
        self, *, cfg: "Config | None" = None, pretty: "PrettyWriter | None" = None
    ) -> None:
        if self._initialized:
            return

        self.models_cache: ModelsCache = ModelsCache()
        self.cfg: Config = cfg or Config()
        self.embedder: Any = self.models_cache.get_hf_embeddings()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()

        # Chromadb collection & client
        self.client: Any = None
        # conversation state
        self.conversation_id: Optional[str] = None
        self.turn_index: int = 0
        self.initalized_collections: List[str] = []

        # RAG parameters (defaults, but overridable via session)
        self._initialized = True
        self.collection: Collection | None = None
        self.collection_name: str = ""

    def _start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Begin or resume a conversation, resetting turn counter on fresh start.
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.turn_index = 0
        return self.conversation_id

    def reset_conversation(self) -> None:
        """Discard the current conversation history and start a fresh one.

        Old ChromaDB entries are retained in the collection (natural pruning
        will remove them) but will never be fetched again because all queries
        filter by conversation_id.
        """
        self._start_conversation()

    def _init_chat_collection(self, session: "Session") -> "Collection":
        """
        Prepare or create a Chroma collection for chat turns.
        On the first in-process invocation for a given collection name,
        call self.deleted(). Uses self.initalized_collections: List[str].
        """
        # create/ensure collection directory and name
        self.collection_name, persist_dir = (
            self.chromaDBHelper.chroma_coll_name_and_mkdir_or_del(
                "create", f"{session.collection_name}_ChatContext"
            )
        )

        # get client and collection
        self.client, self.collection = (
            self.chromaDBHelper.get_chroma_client_and_collection(
                persist_dir, self.collection_name
            )
        )

        # If we've already initialized this collection in this process, skip deletion
        if self.collection_name in self.initalized_collections:
            return self.collection

        # Mark as initialized now that we are about to perform first-invocation work
        self.initalized_collections.append(self.collection_name)
        self._delete_collection(session)
        return self.collection

    def _upsert_to_collection(
        self,
        session: Session,
        docs: list[str],  # now just raw text strings
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Upsert embeddings + documents into Chroma, filtering out complex metadata.
        """
        # Clean metadata for Chroma compatibility
        cleaned_metas = [self.chromaDBHelper.clean_metadata(m) for m in metadatas]

        # Embed all docs in one go
        embeddings = self.embedder.embed_documents(docs)
        coll = self._init_chat_collection(session)
        # Add directly to the existing collection
        coll.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=cast(Any, cleaned_metas),
            documents=docs,
        )

    def add_chat_turn(
        self, session: Session, chat_message: str, assistant_response: str
    ) -> None:
        """
        Store one chat_name+assistant turn with an internal turn counter,
        and tag it with session.chat_name.
        """
        if not self.conversation_id:
            self._start_conversation()

        self._prune_chat_context(session)
        self.turn_index += 1
        self._init_chat_collection(session)

        file_tag = session.file_name or session.file_path or ""
        if file_tag:
            combined = f"[File: {file_tag}]\nUSER: {chat_message}\nASSISTANT: {assistant_response}"
        else:
            combined = f"[No file filter]\nUSER: {chat_message}\nASSISTANT: {assistant_response}"
        query_lang: str = getattr(session, "current_query_lang", None) or "english"
        meta: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chat_name": session.chat_name,
            "file_tag": file_tag,
            "query_lang": query_lang,
        }
        doc_id = str(uuid.uuid4())
        self._upsert_to_collection(
            session=session,
            docs=[combined],
            ids=[doc_id],
            metadatas=[meta],
        )

        if DebugHelper.check_session(session, 30):
            self.pretty.write(
                "I",
                "Add chat context",
                f"Upserted turn {self.turn_index} for chat_name={session.chat_name} "
                f"file_tag='{file_tag}' lang='{query_lang}' to {session.collection_name}_ChatContext",
            )

    def fetch_context_docs(self, session: Session) -> List[LangchainDocument]:
        """Public entry point for retrieving conversation context documents."""
        return self._fetch_context_docs(session)

    def _fetch_context_docs(self, session: Session) -> List[LangchainDocument]:
        """
        Pull up to `session.turns` chat turns for this chat_name,
        sort by turn_index, then return as LangchainDocument list.
        """
        if not self.conversation_id or not self.collection_name:
            return []

        coll = self._init_chat_collection(session)
        file_tag = session.file_name or session.file_path or ""
        query_lang: str = getattr(session, "current_query_lang", None) or "english"
        res = coll.get(
            where={
                "$and": [
                    {"conversation_id": self.conversation_id},
                    {"chat_name": session.chat_name},
                    {"file_tag": file_tag},
                    {"query_lang": query_lang},
                ]
            },
            include=["documents", "metadatas"],
            limit=session.turns,
        )
        texts: list[Any] = res["documents"] or []
        metas: list[Any] = res["metadatas"] or []

        # sort by turn_index, then return all (already capped at session.turns)
        batch: list[tuple[Any, Any]] = list(zip(texts, metas))
        batch.sort(key=lambda x: x[1].get("turn_index", 0))

        docs: List[LangchainDocument] = []
        for text, meta in batch:
            docs.append(
                LangchainDocument(
                    page_content=text, metadata=meta, id=str(uuid.uuid4())
                )
            )
        return docs

    def _prune_chat_context(
        self, session: Session, summary_model: Callable[[str], str] = lambda x: x
    ) -> None:
        """
        Summarize & delete older turns for this chat_name when context exceeds max_turns.
        """
        if not self.conversation_id:
            return

        coll = self._init_chat_collection(session)
        file_tag = session.file_name or session.file_path or ""
        results = coll.get(
            where={
                "$and": [
                    {"conversation_id": self.conversation_id},
                    {"chat_name": session.chat_name},
                    {"file_tag": file_tag},
                ]
            },
            include=["documents", "metadatas"],
            limit=(session.turns or 0) + 1,
        )
        ids, docs, metas = (
            results["ids"],
            results["documents"] or [],
            results["metadatas"] or [],
        )
        if len(ids) <= (session.turns or 0):
            return

        # summarize the oldest `prune_batch` turns
        to_sum = docs[: session.prune_batch]
        sum_metas = metas[: session.prune_batch]
        summary_in = "\n\n".join(to_sum)
        summary_txt = summary_model(summary_in)

        summary_id = str(uuid.uuid4())
        turns = [m["turn_index"] for m in sum_metas]
        summary_meta: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "turn_index": sum_metas[-1]["turn_index"],
            "is_summary": True,
            "summarized_turns": turns,
            "chat_name": session.chat_name,
            "file_tag": file_tag,
        }

        self._upsert_to_collection(
            session=session,
            docs=[summary_txt],
            ids=[summary_id],
            metadatas=[summary_meta],
        )

        if DebugHelper.check_session(session, 30):
            self.pretty.write(
                "I",
                "Prune chat context",
                f"Summarized {session.prune_batch} turns for chat_name {session.chat_name} in {self.collection_name}. Turns: {turns}",
            )

        # delete only the summarized entries
        ids_to_delete = ids[: session.prune_batch]
        coll.delete(ids=ids_to_delete)
        self.pretty.write(
            "I",
            "Prune chat context",
            f"Deleted {len(ids_to_delete)} summarized turns for chat_name {session.chat_name} in {self.collection_name}",
        )

    def annotate_chunks(self, hits: Any) -> list[Any]:
        chunk_docs: list[Any] = []
        for doc, sim in hits:
            score = 1.0 - sim
            doc.metadata["chroma_score"] = score
            doc.metadata["chroma_sim"] = sim
            chunk_docs.append(doc)
        return chunk_docs

    def _delete_collection(
        self,
        session: Session,
    ) -> None:
        """
        Delete the context of a chat.
        """
        if not self.conversation_id:
            return

        # delete only those fetched IDs
        assert self.collection is not None, "collection not initialized"
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
            if DebugHelper.check_session(session, 30):
                self.pretty.write(
                    "I",
                    "Deleted chat context",
                    f"Deleted chat context for {session.chat_name} in {self.collection_name}",
                )
