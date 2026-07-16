# Local module imports
import os
# Standard library includes
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any

from Globals.Session import Session
from Gui.Colors import CYAN
from Gui.FileList import FileList
from Gui.Symbols import Symbols
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import truncate_for_print


class ChunkSelector(ABC):
    """Abstract base for chunk-selection strategies applied after reranking."""

    def __init__(self, session: Session) -> None:
        self.session: Session = session
        self.threshold: float = self.session.chroma_threshold or 0.0
        self.per_file_limit: int = self.session.per_file_limit or 10
        self.pretty: Any = self.session.pretty
        self.cfg: Any = self.session.cfg
        self.fileHist: FileList = FileList()
        web_thr = getattr(self.session, "web_rerank_threshold", None)
        if web_thr is None and self.cfg is not None:
            web_thr = self.cfg.get_float("_WEB_SEARCH.rerank_threshold")
        self.web_rerank_threshold: float = web_thr if web_thr is not None else 0.0
        boost_raw = (
            self.cfg.get_float("SINGLE_CHUNK_SCORE_BOOST", 1.0, silent=True)
            if self.cfg is not None
            else 1.0
        )
        self.single_chunk_boost: float = (
            float(boost_raw) if boost_raw is not None else 1.0
        )

    def _get_score(self, c: Any) -> float:
        if hasattr(c, "score"):
            return float(c.score)
        return float(
            c.metadata.get("rerank_score", c.metadata.get("chroma_score", 0.0))
        )

    def _get_path(self, c: Any) -> str:
        if hasattr(c, "file_path"):
            return c.file_path
        return c.metadata.get("FilePath", c.metadata.get("file_path", ""))

    def _get_filename(self, c: Any) -> str:
        if hasattr(c, "file_name"):
            return c.file_name
        return c.metadata.get("FileName", c.metadata.get("file_name", ""))

    def _print_final_score(
        self,
        misses: list[tuple[Any, float, float, float]],
        hits: list[tuple[Any, float, float, float]],
    ) -> None:
        """
        Prints the post-rerank threshold filtering table.

        NOTE: `threshold` here applies to the *reranker score* (cross-encoder),
        not to Chroma cosine distance.  Web chunks use web_rerank_threshold;
        local chunks use chroma_threshold.  Single-file chunks show
        ``raw→boosted`` when a single-chunk boost was applied.
        """
        header = "{:>2}  {:>8}  {:>8}  {:>9}  {:>17}  {:<40}  {}"
        row = "{:>2}  {:>8.4f}  ({:>6.4f})  {:>+9.4f}  {:>17}  {:<40}  {}"
        row_b = "{:>2}  {:>6.4f}→{:.4f}  ({:>6.4f})  {:>+9.4f}  {:>17}  {:<40}  {}"

        self.pretty.write(
            "A",
            "Rerank select",
            header.format("", "Score", "Thr", "Δ(score)", "Retrievers", "File", "Text"),
            color=CYAN,
        )
        self.pretty.write("A", "Rerank select", "-" * 110, color=CYAN)

        for c, sc, thr, eff in misses:
            dev = eff - thr
            fn = truncate_for_print(self._get_filename(c) or "<unknown>", 40)
            sources: str = str(
                c.metadata.get("retriever_sources", "")
                if hasattr(c, "metadata")
                else ""
            )
            text: str = str(getattr(c, "page_content", "") or "")[:40]
            if eff != sc:
                self.pretty.write(
                    "A",
                    "Rerank select",
                    row_b.format(
                        Symbols.sym_fail(), sc, eff, thr, dev, sources, fn, text
                    ),
                )
            else:
                self.pretty.write(
                    "A",
                    "Rerank select",
                    row.format(Symbols.sym_fail(), sc, thr, dev, sources, fn, text),
                )
            if DebugHelper.check_session(self.session, 32):
                full_text: str = str(getattr(c, "page_content", "") or "")
                self.pretty.write("A", "Chunk content", full_text)

        for c, sc, thr, eff in hits:
            dev = eff - thr
            fn = truncate_for_print(self._get_filename(c) or "<unknown>", 40)
            sources = str(
                c.metadata.get("retriever_sources", "")
                if hasattr(c, "metadata")
                else ""
            )
            text = str(getattr(c, "page_content", "") or "")[:40]
            sym = ("⬆" + Symbols.sym_ok()) if eff != sc else Symbols.sym_ok()
            if eff != sc:
                self.pretty.write(
                    "A",
                    "Rerank select",
                    row_b.format(sym, sc, eff, thr, dev, sources, fn, text),
                )
            else:
                self.pretty.write(
                    "A",
                    "Rerank select",
                    row.format(sym, sc, thr, dev, sources, fn, text),
                )
            if DebugHelper.check_session(self.session, 32):
                full_text = str(getattr(c, "page_content", "") or "")
                self.pretty.write("A", "Chunk content", full_text)

    # When the cross-encoder gives all-negative logits (pool best < threshold),
    # a relative band filter is used instead of the absolute threshold.
    # Accept chunks whose score is within this fraction of the pool's best score.
    _RELATIVE_THRESHOLD_FACTOR: float = 0.75

    def filter_threshold(self, chunks: list[Any]) -> list[Any]:
        # Count how many candidate chunks each file contributes.
        # Files with only one chunk in the pool get a score boost so they
        # are not systematically penalised relative to multi-chunk documents.
        file_counts: Counter[str] = Counter(self._get_path(c) for c in chunks)
        boost = self.single_chunk_boost

        # When the cross-encoder underscores the entire pool (all raw logits
        # negative, common with mmarco-style models on technical content), the
        # absolute threshold rejects every chunk.  Fall back to a relative band:
        # accept chunks whose score is within _RELATIVE_THRESHOLD_FACTOR of the
        # pool's best score instead.
        local_scores = [
            self._get_score(c)
            for c in chunks
            if "Web"
            not in str(
                c.metadata.get("retriever_sources", "")
                if hasattr(c, "metadata")
                else ""
            )
        ]
        max_local_score = max(local_scores, default=0.0)
        use_relative_band = (
            bool(local_scores)
            and max_local_score < self.threshold
            and self.threshold <= 0.35
        )
        if use_relative_band:
            local_threshold = max_local_score * self._RELATIVE_THRESHOLD_FACTOR
            if DebugHelper.check_session(self.session, 10):
                self.pretty.write(
                    "I",
                    "Rerank select",
                    f"Pool max {max_local_score:.4f} < threshold {self.threshold:.4f} — "
                    f"using relative band (×{self._RELATIVE_THRESHOLD_FACTOR}) → "
                    f"effective threshold {local_threshold:.4f}",
                )
        else:
            local_threshold = self.threshold

        hits: list[tuple[Any, float, float, float]] = []
        misses: list[tuple[Any, float, float, float]] = []

        for c in chunks:
            sc = self._get_score(c)
            sources = str(
                c.metadata.get("retriever_sources", "")
                if hasattr(c, "metadata")
                else ""
            )
            thr = self.web_rerank_threshold if "Web" in sources else local_threshold
            path = self._get_path(c)
            eff = sc * boost if (boost > 1.0 and file_counts[path] == 1) else sc
            (hits if eff >= thr else misses).append((c, sc, thr, eff))

        misses.sort(key=lambda x: x[3])
        hits.sort(key=lambda x: x[3], reverse=True)

        if DebugHelper.check_session(self.session, 10):
            self._print_final_score(misses, hits)

        strategy: str = self.session.strategy or "m"
        thr_info = (
            f"local={self.threshold:.4f}  web={self.web_rerank_threshold:.4f}"
            if self.web_rerank_threshold != self.threshold
            else f"{self.threshold:.4f}"
        )
        boost_info = f"  single-chunk boost ×{boost:.2f}" if boost > 1.0 else ""
        self.pretty.write(
            "I",
            f"Strategy: {strategy.lower()}",
            f"{len(hits)} chunks remain after applying threshold {thr_info}{boost_info}",
            color=CYAN,
        )
        return [c for c, _, _, _ in hits]

    def _debugPrint(self, selected: list[Any]):
        if not selected:
            return
        cap = self.session.final_chunks_to_llm or len(selected)
        self.pretty.write(
            "I", "Selected", f"{min(len(selected), cap)} chunks selected", color=CYAN
        )
        header = "{:>2}  {:>8}  {:<40}  {:<70}"
        row = "{:>2}  {:>8.4f}  {:<40}  {:<70}"
        self.pretty.write(
            "D", "Selected", header.format("#", "Score", "File", "Text"), color=CYAN
        )
        self.pretty.write("D", "Selected", "-" * 140, color=CYAN)
        for idx, doc in enumerate(
            selected[: self.session.final_chunks_to_llm], start=1
        ):
            fn = truncate_for_print(self._get_filename(doc) or "<unknown>", 40)
            sc = self._get_score(doc)
            text: str = str(getattr(doc, "page_content", "") or "")[:70]
            self.pretty.write("D", "Selected", row.format(idx, sc, fn, text))

    def _print_chunk_content(self, chunks: list[Any]) -> None:
        """Print full chunk text and metadata for each chunk (debug level 31).

        Emits one block per chunk with:
          - file name, file type, language
          - full file path
          - chunk_id, FileHash
          - all available scores (final, chroma, rerank, rrf) + retriever sources
          - stored metadata (fields not shown via dedicated lines above)
          - the raw page_content text
        """
        _SHOWN_KEYS: frozenset[str] = frozenset(
            {
                "FileName",
                "file_name",
                "FilePath",
                "file_path",
                "chunk_id",
                "id",
                "FileHash",
                "FileType",
                "Language",
                "retriever_sources",
                "chroma_score",
                "chroma_sim",
                "dist",
                "rerank_score",
                "raw_rerank_score",
                "rrf_score",
                "bm25_score",
                "graph_score",
                "position",
                "snippet",  # web docs: printed on its own line below, not in extra_meta
            }
        )
        div = "─" * 90
        if not chunks:
            return
        self.pretty.write("D", "Chunk content", div, color=CYAN)
        for idx, c in enumerate(chunks, start=1):
            meta: dict[str, Any] = getattr(c, "metadata", {}) or {}
            fn = self._get_filename(c) or "<unknown>"
            fp = self._get_path(c) or "<unknown>"
            chunk_id = str(meta.get("chunk_id") or getattr(c, "id", None) or "<none>")
            file_hash = str(meta.get("FileHash") or "<none>")
            file_type = str(meta.get("FileType") or "")
            lang = str(meta.get("Language") or "")
            sources = str(meta.get("retriever_sources") or "")
            content = str(getattr(c, "page_content", "") or "")

            sc = self._get_score(c)
            scores_parts: list[str] = [f"score={sc:.4f}"]
            chroma_sc = meta.get("chroma_score")
            rerank_sc = meta.get("rerank_score")
            rrf_sc = meta.get("rrf_score")
            if chroma_sc is not None:
                scores_parts.append(f"chroma={float(chroma_sc):.4f}")
            if rerank_sc is not None:
                scores_parts.append(f"rerank={float(rerank_sc):.4f}")
            if rrf_sc is not None:
                scores_parts.append(f"rrf={float(rrf_sc):.4f}")

            extra_meta = {k: v for k, v in meta.items() if k not in _SHOWN_KEYS}

            label = f"#{idx}"
            self.pretty.write("D", label, div, color=CYAN)
            self.pretty.write("D", label, f"file={fn}  type={file_type}  lang={lang}")
            self.pretty.write("D", label, f"path={fp}")
            self.pretty.write("D", label, f"chunk_id={chunk_id}  file_hash={file_hash}")
            self.pretty.write(
                "D", label, f"{' '.join(scores_parts)}  sources={sources}"
            )
            if extra_meta:
                self.pretty.write(
                    "D", label, "  ".join(f"{k}={v}" for k, v in extra_meta.items())
                )
            # For web docs: show the snippet that was fed to the cross-encoder
            # for scoring (separate from the full page_content sent to the LLM).
            if meta.get("snippet"):
                self.pretty.write("D", label, f"[scored on snippet] {meta['snippet']}")
            self.pretty.write("D", label, content)
        self.pretty.write("D", "Chunk content", div, color=CYAN)

    @abstractmethod
    def select(self, chunks: list[Any]) -> list[Any]: ...


class ScoreRankedSelector(ChunkSelector):
    """Score-ranked selection. Fills the context window with the highest-scoring
    chunks regardless of source file. Used by DEFAULT, WIDE, and ULTRA_WIDE."""

    def select(self, chunks: list[Any]) -> list[Any]:
        filtered = self.filter_threshold(chunks)
        ordered = sorted(filtered, key=lambda c: self._get_score(c), reverse=True)
        selected = ordered[: self.session.final_chunks_to_llm]
        for c in selected:
            path = self._get_path(c)
            if path:
                self.fileHist.set(
                    f"{self.session.collection_name}_{self.session.chat_name}",
                    "File",
                    os.path.basename(path),
                    path,
                )
        self.pretty.write(
            "I",
            "Chunk selector: Score",
            f"Ranked     Selected {len(selected)} chunks.",
        )
        if DebugHelper.check_session(self.session, 10):
            self._debugPrint(ordered)
        if DebugHelper.check_session(self.session, 32):
            self._print_chunk_content(selected)
        return selected


class PerFileCapSelector(ChunkSelector):
    """Per-file capped selection. Groups chunks by source file, ranks files by
    chunk count, then takes up to ``filelim`` top-scored chunks per file.
    Prevents a single large document from monopolising the context window."""

    def select(self, chunks: list[Any]) -> list[Any]:
        self.per_file_limit = (
            self.session.per_file_limit or self.session.final_chunks_to_llm or 10
        )
        filtered = self.filter_threshold(chunks)
        counts = Counter(self._get_path(c) for c in filtered)
        selected: list[Any] = []
        final_chunks_to_llm: int = self.session.final_chunks_to_llm or 10
        for path, _ in counts.most_common():
            # print(f"Len selected: {len(selected)} vs. window {self.session.final_chunks_to_llm}")
            if len(selected) >= final_chunks_to_llm:
                break

            base = os.path.basename(path)
            self.fileHist.set(
                f"{self.session.collection_name}_{self.session.chat_name}",
                "File",
                base,
                path,
            )
            if self.session.debug_level == 30:
                self.pretty.write(
                    "D", "Per File Cap", f"Promising file path is: {path}"
                )

            group = [c for c in filtered if self._get_path(c) == path]
            group.sort(key=lambda c: self._get_score(c), reverse=True)

            take = min(self.per_file_limit, final_chunks_to_llm - len(selected))
            if self.session.debug_level == 30:
                msg = "Chat Context" if path == "Chat Context" else "Path"
                self.pretty.write(
                    "D",
                    "Chunk selector: Per File Cap",
                    f"Selected {len(group[:take])} chunks from {msg}: {path}",
                )
            selected.extend(group[:take])

        if DebugHelper.check_session(self.session, 10):
            self._debugPrint(selected)
        if DebugHelper.check_session(self.session, 32):
            self._print_chunk_content(selected[: self.session.final_chunks_to_llm])
        return selected[: self.session.final_chunks_to_llm]


class SingleDocumentSelector(ChunkSelector):
    """Single-document focused selection. Compares the file with the highest
    individual chunk score against the file with the most chunks; the winner
    supplies all context. Best for questions targeting a specific document."""

    def select(self, chunks: list[Any]) -> list[Any]:
        filtered = self.filter_threshold(chunks)
        if not filtered:
            return []

        by_path: defaultdict[str, list[Any]] = defaultdict(list)
        for c in filtered:
            by_path[self._get_path(c)].append(c)

        def _best_score(path: str) -> float:
            return max(self._get_score(c) for c in by_path[path])

        # find highest‐score path vs. highest‐count path
        top_path = max(by_path, key=_best_score)
        count_path = max(by_path.items(), key=lambda kv: len(kv[1]))[0]

        if top_path == count_path:
            doc0 = by_path[top_path][0]
            _name = self._get_filename(doc0)
            self.pretty.write(
                "A",
                "Chunk selector: Single Document",
                f"Path with highest score {_best_score(top_path):.4f} "
                f"matches most chunks. Full path: {top_path}",
            )
            chosen = top_path
        else:
            self.pretty.write(
                "I",
                "Chunk selector: Single Document",
                f"Choosing most\u2010chunks path {count_path} "
                f"over highest\u2010score path {top_path}.",
            )
            self.pretty.write(
                "I",
                "Chunk selector: Single Document",
                f"over highest‐score path {top_path}.",
            )
            chosen = count_path

        best_chunks = sorted(
            by_path[chosen], key=lambda c: self._get_score(c), reverse=True
        )
        base = os.path.basename(chosen)
        self.fileHist.set(
            f"{self.session.collection_name}_{self.session.chat_name}",
            "File",
            base,
            chosen,
        )
        self.pretty.write(
            "I",
            "Chunk selector: Single Document",
            f"Final: Selected {len(best_chunks[:self.session.final_chunks_to_llm])} chunks.",
        )
        if DebugHelper.check_session(self.session, 10):
            self._debugPrint(best_chunks)
        if DebugHelper.check_session(self.session, 32):
            self._print_chunk_content(best_chunks[: self.session.final_chunks_to_llm])

        return best_chunks[: self.session.final_chunks_to_llm]


class ChunkSelectionService:
    """Factory that maps ``session.strategy`` to the appropriate selector."""

    def __init__(self, session: Session) -> None:
        self.session: Session = session

    def get_selector(self) -> ChunkSelector:
        strat: str = (self.session.strategy or "default").lower()
        if strat in ("wide", "ultra_wide", "default"):
            selector = ScoreRankedSelector(self.session)
        elif strat == "balanced_file_cap":
            selector = PerFileCapSelector(self.session)
        elif strat == "narrow":
            selector = SingleDocumentSelector(self.session)
        else:
            raise ValueError(f"Unknown strategy {self.session.strategy}")
        selector.pretty.write(
            "I",
            "Chunk selection",
            f"Strategy '{strat.upper()}' → {type(selector).__name__}",
        )
        return selector

    def select_chunks(self, chunks: list[Any]) -> list[Any]:
        return self.get_selector().select(chunks)
