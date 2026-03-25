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

    def _start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Begin or resume a conversation, resetting turn counter on fresh start.
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.turn_index = 0
        return self.conversation_id

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

        combined = f"USER: {chat_message}\nASSISTANT: {assistant_response}"
        meta = {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chat_name": session.chat_name,
        }
        doc_id = str(uuid.uuid4())
        self._upsert_to_collection(
            session=session,
            docs=[combined],
            ids=[doc_id],
            metadatas=[meta],
        )

        if (session.debug_level or 0) >= 3:
            self.pretty.write(
                "I",
                "Add chat context",
                f"Upserted turn {self.turn_index} for chat_name {session.chat_name} to {session.collection_name}_ChatContext",
            )

    def _fetch_context_docs(self, session: Session) -> List[LangchainDocument]:
        """
        Pull up to `session.turns` chat turns for this chat_name,
        sort by turn_index, then return as LangchainDocument list.
        """
        if not self.conversation_id or not self.collection_name:
            return []

        coll = self._init_chat_collection(session)
        res = coll.get(
            where={
                "$and": [
                    {"conversation_id": self.conversation_id},
                    {"chat_name": session.chat_name},
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
        results = coll.get(
            where={
                "$and": [
                    {"conversation_id": self.conversation_id},
                    {"chat_name": session.chat_name},
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

        # summarize the oldest `batch_size` turns
        to_sum = docs[: session.batch_size]
        sum_metas = metas[: session.batch_size]
        summary_in = "\n\n".join(to_sum)
        summary_txt = summary_model(summary_in)

        summary_id = str(uuid.uuid4())
        turns = [m["turn_index"] for m in sum_metas]
        summary_meta = {
            "conversation_id": self.conversation_id,
            "turn_index": sum_metas[-1]["turn_index"],
            "is_summary": True,
            "summarized_turns": turns,
            "chat_name": session.chat_name,
        }

        self._upsert_to_collection(
            session=session,
            docs=[summary_txt],
            ids=[summary_id],
            metadatas=[summary_meta],
        )

        if (session.debug_level or 0) >= 3:
            self.pretty.write(
                "I",
                "Prune chat context",
                f"Summarized {session.batch_size} turns for chat_name {session.chat_name} in {self.collection_name}. Turns: {turns}",
            )

        # delete only those fetched IDs
        coll.delete(
            where={
                "$and": [
                    {"conversation_id": self.conversation_id},
                    {"chat_name": session.chat_name},
                ]
            }
        )
        self.pretty.write(
            "I",
            "Prune chat context",
            f"Deleted old turns for chat_name {session.chat_name} in {self.collection_name}",
        )

    def annotate_chunks(self, hits: Any) -> list[Any]:
        chunk_docs: list[Any] = []
        for doc, sim in hits:
            score = 1.0 - sim
            doc.metadata["chroma_score"] = score
            doc.metadata["chroma_sim"] = sim
            chunk_docs.append(doc)
        return chunk_docs

    def merge_with_chunks(self, chunk_docs: Any, session: Session) -> tuple[Any, int]:
        # Returns a new documents list
        merged = chunk_docs.copy()
        hist_docs = self._fetch_context_docs(session)
        # compute base_score from chunk_docs (fallback to 0.5)
        if chunk_docs:
            avg_score = sum(d.metadata["chroma_score"] for d in chunk_docs) / len(
                chunk_docs
            )
        else:
            avg_score = 0.5
        self.pretty.write(
            "I",
            "Merge with chat context",
            f"Average after merge with chat context is: {avg_score}",
        )
        for h in hist_docs:
            h.metadata["chroma_score"] = avg_score  # type: ignore[reportUnknownMemberType]
            # "sim" for chat context can be derived or left at (1-avg_score)
            h.metadata["chroma_sim"] = 1.0 - avg_score  # type: ignore[reportUnknownMemberType]
            h.metadata.setdefault("FileName", "Chat Context")  # type: ignore[reportUnknownMemberType]
            h.metadata.setdefault("FilePath", "Chat Context")  # type: ignore[reportUnknownMemberType]
        merged = hist_docs + merged
        return merged, len(merged)

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
            if (session.debug_level or 0) >= 3:
                self.pretty.write(
                    "I",
                    "Deleted chat context",
                    f"Deleted chat context for {session.chat_name} in {self.collection_name}",
                )
