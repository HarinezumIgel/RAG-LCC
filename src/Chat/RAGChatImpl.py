# ── Local Module Imports ──
import os
# ── Standard Library Imports ──
from typing import Any, Sequence, Tuple, cast

from chromadb.api import Collection  # type: ignore[attr-defined]
# ── LangChain Ecosystem ──
from langchain_chroma import Chroma
from langdetect import \
    detect  # type: ignore[reportMissingTypeStubs]  # noqa: F401

from AI.ModelsCache import ModelsCache
from Chat.ChatContext import ChatContext
from Commons.Exceptions import CollectionNotFoundError, RerankError
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.ChromaDBHelper import ChromaDBHelper
from Helpers.Helpers import Helpers
from Strategies.HomeBrewChunkSelector import ChunkSelectionService

# ── Third-Party Libraries ──


class RAGChatImpl(SingletonMixin):
    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Instantiate helper objects/singletons as instance attributes.
        # Cache for stopwords per language.
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = cfg or Config()
        self.helperInstance: Helpers = helpers or Helpers()
        self.chromaDBHelper: ChromaDBHelper = ChromaDBHelper()
        self.chatContext: ChatContext = ChatContext()
        self.models_cache: ModelsCache = ModelsCache()
        # Initialize the embeddings using Ollama.
        self.device: Any
        self.device, _, _, _ = self.models_cache.switch2device()
        self.embed_model_name: str = self.helperInstance.get_model_args("_EMBED")[
            "MODEL"
        ]
        self.cross_encoder_model_name: str = self.helperInstance.get_model_args(
            "_CROSS"
        )["MODEL"]
        self.x_encoder: Any = self.models_cache.get_cross_encoder()
        self.embedder: Any = self.models_cache.get_hf_embeddings()
        self.persist_directory: str | None = None
        self.vector_store: Chroma | None = None
        self.collection: Collection | None = None

    def set_vector_store(self, mySession: Session) -> bool:
        self.collection_name, self.persist_directory = (
            self.chromaDBHelper.change_chroma_collection(
                mySession.collection_name, True
            )
        )
        if self.collection_name and self.persist_directory:
            self.pretty.write(
                "I",
                "VectorStore",
                f"Set Chroma vector store. Name: {self.collection_name} Path: {self.persist_directory}",
            )

        if not os.path.exists(self.persist_directory):
            msg = (
                f"Collection '{self.collection_name}' not found at {self.persist_directory}. "
                f"Create a new collection running RAGLoad.py --collection MyCollection "
                f"or provide an existing collection running RAGChat.py --collection existingCollection"
            )
            self.pretty.write("E", "Collection", msg, color=RED)
            raise CollectionNotFoundError(msg)

        # Load Chroma client and collection from persisted directory
        self.client, self.collection = (
            self.chromaDBHelper.get_chroma_client_and_collection(
                self.persist_directory, self.collection_name
            )
        )
        # Initialize vector store with cosine similarity metric
        self.vector_store = Chroma(
            embedding_function=self.embedder,
            client=self.client,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return True

    def _rerank(self, mySession: Session, all_docs: list[Any]) -> list[Any]:
        alpha: float = mySession.chroma_weight or 0.5

        # Prepare query-document pairs for re-ranking
        candidates: list[Any] = all_docs
        pairs = [
            (
                f"Query: {mySession.query}\n"
                f"ChromaScore={doc.metadata['chroma_score']:.3f}\n"
                f"Chunk: {doc.page_content}",
                "",
            )
            for doc in candidates
        ]

        if len(pairs) == 0:
            self.pretty.write(
                "W",
                "Rerank",
                f"Reranking with {self.cross_encoder_model_name} returned {len(pairs)} chunks",
            )
            return pairs

        # Get re-ranking scores from cross-encoder model
        try:
            raw_rerank = self.x_encoder.predict(pairs, show_progress_bar=False)
        except RuntimeError as e:
            self.pretty.write(
                "E",
                "Rerank",
                f"Reranking failed due to a model input mismatch. "
                f"Did you run Load.py with a different _CHROMA_EMBED_PARAMS.CHUNK_SIZE than RAGChat.py is using now? "
                f"Details: {e}",
                color=RED,
            )
            raise RerankError

        # Extract original Chroma scores for blending
        chroma_raw: list[float] = [doc.metadata["chroma_score"] for doc in candidates]

        # MIN-MAX normalize both score streams to [0,1]
        eps: float = 1e-8
        min_c, max_c = min(chroma_raw), max(chroma_raw)
        norm_chroma = [(c - min_c) / (max_c - min_c + eps) for c in chroma_raw]

        min_r, max_r = min(raw_rerank), max(raw_rerank)
        norm_rerank = [(r - min_r) / (max_r - min_r + eps) for r in raw_rerank]

        # Blend Chroma and re-rank scores using weighted alpha parameter
        combined_norm: list[float] = []
        for doc, c_n, r_n in zip(candidates, norm_chroma, norm_rerank):
            combo = alpha * c_n + (1 - alpha) * r_n
            combined_norm.append(combo)
            doc.metadata["rerank_score"] = combo

        # Sort documents by combined score in descending order
        reranked: list[Any] = sorted(
            candidates, key=lambda d: d.metadata["rerank_score"], reverse=True  # type: ignore[reportUnknownLambdaType, reportUnknownMemberType]
        )

        # debug print
        if (mySession.debug_level or 0) >= 1:
            header = "{:>6} {:>12} {}"
            row = "{:>6} {:>12.4f} {}"

            # Print header row
            self.pretty.write("D", "Rerank", header.format("Pos", "Score", "Text"))
            self.pretty.write("D", "Rerank", "-" * 80)

            # Print each row aligned with header
            for i, d in enumerate(reranked[: mySession.chunks_window]):
                self.pretty.write(
                    "D",
                    "Rerank",
                    row.format(
                        i + 1,
                        d.metadata["rerank_score"],
                        d.page_content[:50],  # truncate text to fit
                    ),
                )

        self.pretty.write(
            "O",
            "Rerank",
            f"Reranking with {self.cross_encoder_model_name} returned {len(reranked)} chunks",
        )
        return reranked

    def _print_chroma_debug(self, docs: Sequence[Any]) -> None:
        """
        Prints a table of chroma stats for each doc when DEBUG is enabled.
        docs: a sequence of objects with doc.metadata containing:
        - 'position'
        - 'chroma_score' (float)
        - 'chroma_sim'   (float)
        - optional 'dist' (float); defaults to chroma_sim
        - 'FileName'     (str)
        """
        # if not self.cfg.get("DEBUG_LEVEL"):
        #    return

        # column formats
        # column formats: add position column first
        header = "{:>6}  {:>12}  {:>9}  {:>8}   {}"
        row = "{:>6}  {:>12.4f}  {:>9.4f}  {:>8.4f}   {}"

        # print header + separator once
        self.pretty.write(
            "D",
            "Chroma",
            header.format("Pos", "ChromaScore", "ChromaSim", "Distance", "File"),
        )
        self.pretty.write("D", "Chroma", "-" * 80)

        # print one row per doc
        for i, doc in enumerate(docs, start=1):
            md: dict[str, Any] = doc.metadata
            score: Any = md.get("chroma_score", 0.0)
            sim: Any = md.get("chroma_sim", 0.0)
            dist: Any = md.get("dist", sim)
            fn: Any = md.get("FileName", "<unknown>")

            self.pretty.write("D", "Chroma", row.format(i, score, sim, dist, fn))

    def retrieve(self, mySession: Session) -> Tuple[str, int]:
        if self.set_vector_store(mySession):
            self.pretty.write(
                "I",
                "Chroma",
                f"Querying Chroma DB on vector store {self.persist_directory}",
            )

            assert (
                self.vector_store is not None
            ), "vector_store not initialized; call set_vector_store first"
            hits: list[Any] = self.vector_store.similarity_search_with_score(
                mySession.query or "", **(mySession.base_kwargs or {})
            )

            # debug & log
            annotated: list[Any] = self.chatContext.annotate_chunks(hits)
            if (mySession.debug_level or 0) >= 1:
                self._print_chroma_debug(annotated)
            self.pretty.write(
                "O",
                "Chroma",
                f"Querying Chroma DB query returned {len(annotated)} chunks",
            )

            # Cap annotated
            capped_docs: list[Any] = annotated[: mySession.chroma_k_value]
            if mySession.use_chat_context is True:
                # merge hits + history in one step
                merged_docs, total_pieces = self.chatContext.merge_with_chunks(
                    capped_docs,
                    mySession,
                )
                if (mySession.debug_level or 0) >= 3:
                    self._print_chroma_debug(merged_docs)
                self.pretty.write(
                    "O",
                    "Chroma",
                    f"Merge Chroma hits and history returned {total_pieces} chunks",
                )
            else:
                merged_docs = capped_docs

            # next—rerank / select if you need:
            if mySession.rerank == 1:
                merged_docs = self._rerank(mySession, merged_docs)

            chosen: list[Any] = cast(list[Any], ChunkSelectionService().select_chunks(merged_docs))  # type: ignore[reportUnknownMemberType]

            # Format the context by combining the formatted adjacent chunks.
            context: str = "\n\n".join(
                self.helperInstance.format_document(doc) for doc in chosen  # type: ignore[reportUnknownMemberType]
            )
            if len(chosen) > 0:
                return context, len(chosen)
            else:
                self.pretty.write("A", "", "No documents found after applying filters")
                self.pretty.write(
                    "W",
                    "Suggested Action",
                    f"Try lowering sensitivity. Current: {mySession.chroma_threshold}",
                )
                return "", 0
        return "", 0
