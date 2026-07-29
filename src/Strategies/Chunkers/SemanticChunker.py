import time
import uuid
from typing import Any

import numpy as np
from langchain_core.documents.base import Document as langchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AI.ModelsCache import ModelsCache
from Config.Config import Config
from Gui.Colors import BRIGHT_BLUE, RESET, VIOLET
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger
from Strategies.Chunkers.ChunkerStrategy import ChunkerStrategy, ChunkResult
from Strategies.Chunkers.SentenceSplitter import SentenceSplitter


class SemanticChunker(ChunkerStrategy):
    """Chunks documents by detecting topic boundaries via cosine similarity
    between consecutive sentence embeddings.

    Sentences whose inter-similarity falls in the bottom
    ``BREAKPOINT_PERCENTILE`` are treated as chunk boundaries.
    Oversized segments are split with RecursiveCharacterTextSplitter
    as a safety cap (``MAX_CHUNK_SIZE``).

    Pipeline flow
    -------------
    1. Split text into sentences              → _split_sentences()
    2. Merge short fragments                  → _consolidate_short()
    3. Embed each sentence into a vector      → _embed_batched()
         e.g. [s0,s1,…s7] → [e0,e1,…e7]
    4. Cosine similarity between consecutive  → _cosine_similarities()
         pairs: cos(e0,e1), cos(e1,e2), …
    5. Find topic-shift breakpoints           → _find_breakpoints()
         (valleys below BREAKPOINT_PERCENTILE)
    6. Group sentences between breakpoints    → _group_ranges()
         e.g. breakpoints=[3,6] → groups (0:3), (3:6), (6:8)
    7. For each group:
       a. If fits in MAX_CHUNK_SIZE           → _weighted_mean_embedding()
            compute chunk embedding as word-count-weighted
            average of its sentence embeddings
       b. If oversized                        → _split_oversized()
            sub-split with RecursiveCharacterTextSplitter;
            embeddings set to None (caller re-embeds these)
    8. Wrap final texts into langchain Docs   → _to_docs()

    Returns (docs, pre_embeddings) so that DocumentIngestionStrategy can
    skip re-embedding chunks that already have a vector.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        helpers: "Helpers | None" = None,
        file_utils: "FileUtils | None" = None,
        embedder: Any | None = None,
        chunker_name: str | None = None,
    ) -> None:
        self._cfg: Config = cfg or Config()
        self._helpers: Helpers = helpers or Helpers()
        self._fileUtils: FileUtils = file_utils or FileUtils()
        self.perf_logger: PerfLogger = PerfLogger()

        chunker_slot: str = (
            f"_CHUNKERS.{chunker_name}"
            if chunker_name
            else self._helpers.get_chunker_config_slot()
        )
        self._max_chunk_size: int = self._cfg.get_int(f"{chunker_slot}.MAX_CHUNK_SIZE")
        self._breakpoint_percentile: int = self._cfg.get_int(
            f"{chunker_slot}.BREAKPOINT_PERCENTILE", 10
        )
        self._embed_batch_size: int = self._cfg.get_int(
            f"{chunker_slot}.EMBED_BATCH_SIZE", 64
        )
        self._min_sentence_words: int = self._cfg.get_int(
            f"{chunker_slot}.MIN_SENTENCE_WORDS", 15
        )
        self._separators: list[Any] = self._cfg.get_list("_SEPARATORS")

        # Embedder for sentence embedding — injectable for tests
        self._embedder: Any = embedder or ModelsCache().get_hf_embeddings()
        self._pretty: PrettyWriter = PrettyWriter()

    # -- ChunkerStrategy interface ------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._max_chunk_size

    def chunk(self, content: str, metadata: dict[str, Any]) -> ChunkResult:
        self.perf_logger.log(
            "SemanticChunker.chunk",
            "chunker",
            f"start content_len={len(content)}",
        )
        _t0 = time.perf_counter()

        sentences: list[str] = self._split_sentences(content)

        if len(sentences) <= 1:
            texts = [content] if content.strip() else []
            if texts and self._fileUtils.count_words(texts[0]) > self._max_chunk_size:
                texts = self._split_oversized(texts[0])
            # No sentence embeddings available → caller must embed these
            result = self._to_docs(texts, metadata), None
            elapsed = time.perf_counter() - _t0
            self.perf_logger.log(
                "SemanticChunker.chunk",
                "chunker",
                f"stop n={len(result[0])} elapsed={elapsed:.3f}s",
            )
            return result

        # Merge consecutive short fragments so they get one embedding
        # instead of noisy individual vectors (common in PDF tables/specs).
        sentences = self._consolidate_short(sentences)

        self._pretty.write(
            "I",
            "SemanticChunker",
            f"{BRIGHT_BLUE}{len(sentences)} sentences to embed{RESET}",
        )

        embeddings: list[list[float]] = self._embed_batched(sentences)
        similarities: list[float] = self._cosine_similarities(embeddings)
        breakpoints: list[int] = self._find_breakpoints(similarities)

        # Build (start, end) ranges for each group so we can map back to
        # the original sentence embeddings when computing chunk vectors.
        group_ranges: list[tuple[int, int]] = self._group_ranges(
            len(sentences), breakpoints
        )

        # Assemble final chunk texts.  For each group that fits within the
        # size budget we compute a word-count-weighted average of its
        # sentence embeddings (a cheap approximation of the full-chunk
        # embedding).  Oversized groups are sub-split and get None — the
        # caller will embed only those.
        final_texts: list[str] = []
        chunk_embeddings: list[list[float] | None] = []

        for start, end in group_ranges:
            group_text = " ".join(sentences[start:end])

            if self._fileUtils.count_words(group_text) > self._max_chunk_size:
                # Oversized → sub-split; can't reuse sentence embeddings
                sub_texts = self._split_oversized(group_text)
                final_texts.extend(sub_texts)
                chunk_embeddings.extend([None] * len(sub_texts))
            else:
                final_texts.append(group_text)
                # Weighted average: longer sentences contribute more to the
                # chunk vector, approximating what the model would produce
                # if it embedded the full concatenated text.
                chunk_embeddings.append(
                    self._weighted_mean_embedding(
                        embeddings[start:end],
                        [sentences[i] for i in range(start, end)],
                    )
                )

        self._pretty.write(
            "I",
            "SemanticChunker",
            f"{BRIGHT_BLUE}{len(final_texts)} chunks created "
            f"({len(breakpoints)} topic boundaries detected){RESET}",
        )

        result = self._to_docs(final_texts, metadata), chunk_embeddings
        elapsed = time.perf_counter() - _t0
        self.perf_logger.log(
            "SemanticChunker.chunk",
            "chunker",
            f"stop n={len(result[0])} elapsed={elapsed:.3f}s",
        )
        return result

    # -- Internal helpers ---------------------------------------------------

    def _embed_batched(self, sentences: list[str]) -> list[list[float]]:
        """Embed sentences in batches to avoid GPU OOM on large documents.

        Long sentences are truncated to ``_max_chunk_size`` words for
        embedding (boundary detection only needs the gist, not the full
        text).  GPU cache is cleared between batches to avoid memory
        fragmentation.
        """
        # Truncate long sentences for embedding — full text stays in the
        # sentence list for final chunk assembly.
        max_words: int = self._max_chunk_size
        truncated: list[str] = []
        for s in sentences:
            words = s.split()
            if len(words) > max_words:
                truncated.append(" ".join(words[:max_words]))
            else:
                truncated.append(s)

        all_embeddings: list[list[float]] = []
        total_batches: int = -(-len(truncated) // self._embed_batch_size)  # ceil div
        bar_width: int = 40
        for batch_num, start in enumerate(
            range(0, len(truncated), self._embed_batch_size), 1
        ):
            batch = truncated[start : start + self._embed_batch_size]
            all_embeddings.extend(self._embedder.embed_documents(batch))
            if total_batches > 1:
                pct = int(batch_num * 100 / total_batches)
                filled = int(bar_width * batch_num / total_batches)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    f"\r   {VIOLET}{'Embeddings (Semantic)':<30} [{bar}] "
                    f"{batch_num}/{total_batches} ({pct}%){RESET}",
                    end="",
                    flush=True,
                )
        #            if torch.cuda.is_available():
        #                torch.cuda.empty_cache()
        if total_batches > 1:
            print()  # newline after progress bar
        return all_embeddings

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using shared boundary detection."""
        return SentenceSplitter.split_sentences(text)

    def _consolidate_short(self, sentences: list[str]) -> list[str]:
        """Merge consecutive short fragments into fewer, denser sentences.

        PDF-extracted text from hardware manuals, spec sheets, and tables
        often produces many tiny "sentences" (e.g. one table row per line).
        Each gets its own embedding vector, but ≤10-word fragments carry
        too little signal for reliable cosine-similarity breakpoint
        detection.

        This pass greedily merges consecutive sentences that are below
        ``_min_sentence_words`` into a single unit, flushing when the
        accumulated text reaches the threshold or a long sentence is
        encountered.
        """
        threshold: int = self._min_sentence_words
        merged: list[str] = []
        buf: list[str] = []
        buf_words: int = 0

        for sent in sentences:
            word_count: int = len(sent.split())
            if word_count >= threshold:
                # Flush any buffered short fragments
                if buf:
                    merged.append(" ".join(buf))
                    buf, buf_words = [], 0
                merged.append(sent)
            else:
                buf.append(sent)
                buf_words += word_count
                if buf_words >= threshold:
                    merged.append(" ".join(buf))
                    buf, buf_words = [], 0

        if buf:
            merged.append(" ".join(buf))

        return merged

    @staticmethod
    def _cosine_similarities(embeddings: list[list[float]]) -> list[float]:
        """Cosine similarity between each consecutive pair of embeddings."""
        vecs = np.array(embeddings)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = vecs / norms
        # dot product between consecutive rows
        sims = np.sum(normed[:-1] * normed[1:], axis=1)
        return sims.tolist()

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """Indices where a new chunk should start (similarity valleys)."""
        if not similarities:
            return []
        threshold: float = float(
            np.percentile(similarities, self._breakpoint_percentile)
        )
        return [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    @staticmethod
    def _group_sentences(sentences: list[str], breakpoints: list[int]) -> list[str]:
        """Merge sentences between breakpoints into chunk texts."""
        groups: list[str] = []
        start = 0
        for bp in breakpoints:
            joined = " ".join(sentences[start:bp])
            if joined.strip():
                groups.append(joined)
            start = bp
        tail = " ".join(sentences[start:])
        if tail.strip():
            groups.append(tail)
        return groups

    @staticmethod
    def _group_ranges(
        n_sentences: int, breakpoints: list[int]
    ) -> list[tuple[int, int]]:
        """Return (start, end) index ranges for sentence groups.

        Each range is a half-open interval [start, end) into the sentence
        list.  Used to map groups back to their original sentence
        embeddings for weighted-average computation.
        """
        ranges: list[tuple[int, int]] = []
        start = 0
        for bp in breakpoints:
            if bp > start:
                ranges.append((start, bp))
            start = bp
        if start < n_sentences:
            ranges.append((start, n_sentences))
        return ranges

    def _weighted_mean_embedding(
        self,
        sentence_embeddings: list[list[float]],
        sentences: list[str],
    ) -> list[float]:
        """Compute a word-count-weighted average of sentence embeddings.

        Longer sentences carry more semantic weight, so their embedding
        vectors contribute proportionally more to the chunk vector.
        The result is L2-normalised to match what the embedding model
        would produce for the concatenated text (most sentence-transformer
        models output unit-norm vectors).
        """
        vecs = np.array(sentence_embeddings)
        weights = np.array([len(s.split()) for s in sentences], dtype=float).reshape(
            -1, 1
        )
        avg = (vecs * weights).sum(axis=0) / weights.sum()
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg.tolist()

    def _split_oversized(self, text: str) -> list[str]:
        """Fall back to RecursiveCharacterTextSplitter for oversized segments."""
        separators_str: list[str] = [str(s) for s in self._separators]
        splitter = RecursiveCharacterTextSplitter(
            separators_str,
            False,
            is_separator_regex=False,
            length_function=self._fileUtils.count_words,
            chunk_size=self._max_chunk_size,
            chunk_overlap=0,
        )
        return [d.page_content for d in splitter.create_documents([text])]

    @staticmethod
    def _to_docs(texts: list[str], metadata: dict[str, Any]) -> list[langchainDoc]:
        """Wrap text segments into langchainDocs with UUIDs and MyChunk index."""
        docs: list[langchainDoc] = []
        for i, text in enumerate(texts):
            meta = dict(metadata)
            meta["MyChunk"] = i
            docs.append(
                langchainDoc(page_content=text, metadata=meta, id=str(uuid.uuid4()))
            )
        return docs
