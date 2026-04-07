# Local module imports
import os
# Standard library includes
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any

from Globals.Session import Session
from Gui.FileList import FileList
from Gui.Symbols import Symbols


class ChunkSelector(ABC):
    def __init__(self, session: Session) -> None:
        self.session: Session = session
        self.threshold: float = self.session.chroma_threshold or 0.0
        self.per_file_limit: int = self.session.per_file_limit or 10
        self.pretty: Any = self.session.pretty
        self.cfg: Any = self.session.cfg
        self.fileHist: FileList = FileList()

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
        misses: list[tuple[Any, float]],
        hits: list[tuple[Any, float]],
    ) -> None:
        """
        Prints the post-rerank threshold filtering table.

        NOTE: `threshold` here applies to the *reranker score* (cross-encoder),
        not to Chroma cosine distance.
        """
        # Column formatting (stable alignment)
        header = "{:>2}  {:>8}  {:>8}  {:>9}   {}"
        row = "{:>2}  {:>8.4f}  ({:>6.4f})  {:>+9.4f}   {}"

        self.pretty.write(
            "A",
            "RAG select",
            header.format("", "Score", "Thr", "Δ(score)", "File"),
        )
        self.pretty.write("A", "RAG select", "-" * 80)

        thr = float(self.threshold)

        # Misses: low to high so you see "almost made it" at the bottom of ❌ block
        for c, sc in misses:
            dev = sc - thr
            fn = self._get_filename(c) or "<unknown>"
            self.pretty.write(
                "A", "RAG select", row.format(Symbols.sym_fail(), sc, thr, dev, fn)
            )

        # Hits: high to low so best matches appear first
        for c, sc in hits:
            dev = sc - thr
            fn = self._get_filename(c) or "<unknown>"
            self.pretty.write(
                "A", "RAG select", row.format(Symbols.sym_ok(), sc, thr, dev, fn)
            )

    def filter_threshold(self, chunks: list[Any]) -> list[Any]:
        hits: list[tuple[Any, float]] = []
        misses: list[tuple[Any, float]] = []

        for c in chunks:
            sc = self._get_score(c)
            (hits if sc >= self.threshold else misses).append((c, sc))

        misses.sort(key=lambda x: x[1])
        hits.sort(key=lambda x: x[1], reverse=True)

        if (self.session.debug_level or 0) >= 1:
            self._print_final_score(misses, hits)

        strategy: str = self.session.strategy or "m"
        self.pretty.write(
            "A",
            f"Strategy: {strategy.lower()}",
            f"{len(hits)} chunks remain after applying threshold of {self.threshold:.4f}",
        )
        return [c for c, _ in hits]

    def _debugPrint(self, selected: list[Any]):
        for idx, doc in enumerate(selected[: self.session.chunks_window], start=1):
            file_name = self._get_filename(doc)
            msg = "Chat Context" if file_name == "Chat Context" else "File Name"
            self.pretty.write("D", "Selected", f"{idx}. {msg} = {file_name}")

    @abstractmethod
    def select(self, chunks: list[Any]) -> list[Any]: ...


class WideUltraWideSelector(ChunkSelector):
    def select(self, chunks: list[Any]) -> list[Any]:
        filtered = self.filter_threshold(chunks)
        ordered = sorted(filtered, key=lambda c: self._get_score(c), reverse=True)
        self.pretty.write(
            "I",
            "Chunk selector: Wide / Ultra",
            f"Selected {len(ordered[:self.session.chunks_window])} chunks.",
        )
        if (self.session.debug_level or 0) >= 1:
            self._debugPrint(ordered)
        return ordered[: self.session.chunks_window]


class MediumSelector(ChunkSelector):
    def select(self, chunks: list[Any]) -> list[Any]:
        self.per_file_limit = (
            self.session.per_file_limit or self.session.chunks_window or 10
        )
        filtered = self.filter_threshold(chunks)
        counts = Counter(self._get_path(c) for c in filtered)
        selected: list[Any] = []
        chunks_window: int = self.session.chunks_window or 10
        for path, _ in counts.most_common():
            # print(f"Len selected: {len(selected)} vs. window {self.session.chunks_window}")
            if len(selected) >= chunks_window:
                break

            base = os.path.basename(path)
            self.fileHist.set(
                f"{self.session.collection_name}_{self.session.chat_name}",
                "File",
                base,
                path,
            )
            if self.session.debug_level == 3:
                self.pretty.write("D", "Medium", f"Promising file path is: {path}")

            group = [c for c in filtered if self._get_path(c) == path]
            group.sort(key=lambda c: self._get_score(c), reverse=True)

            take = min(self.per_file_limit, chunks_window - len(selected))
            if self.session.debug_level == 3:
                msg = "Chat Context" if path == "Chat Context" else "Path"
                self.pretty.write(
                    "D",
                    "Chunk selector: Medium",
                    f"Selected {len(group[:take])} chunks from {msg}: {path}",
                )
            selected.extend(group[:take])

        self.pretty.write(
            "I",
            "medium",
            f"Selected {len(selected)} Returning: {len(selected[:self.session.chunks_window])} chunks.",
        )
        if (self.session.debug_level or 0) >= 1:
            self._debugPrint(selected)
        return selected[: self.session.chunks_window]


class NarrowSelector(ChunkSelector):
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
                "Chunk selector: Narrow",
                f"Path with highest score {_best_score(top_path):.4f} "
                f"matches most chunks. Full path: {top_path}",
            )
            chosen = top_path
        else:
            self.pretty.write(
                "I",
                "Chunk selector: Narrow",
                f"Choosing most‐chunks path {count_path} "
                f"over highest‐score path {top_path}.",
            )
            self.pretty.write(
                "I",
                "Chunk selector: Narrow",
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
            "Chunk selector: Narrow",
            f"Final: Selected {len(best_chunks[:self.session.chunks_window])} chunks.",
        )
        if (self.session.debug_level or 0) >= 1:
            self._debugPrint(best_chunks)

        return best_chunks[: self.session.chunks_window]


class ChunkSelectionService:
    def __init__(self, session: Session) -> None:
        self.session: Session = session

    def get_selector(self) -> ChunkSelector:
        strat: str = (self.session.strategy or "medium").lower()
        if strat == "wide" or strat == "ultra_wide":
            return WideUltraWideSelector(self.session)
        if strat == "medium":
            return MediumSelector(self.session)
        if strat == "narrow":
            return NarrowSelector(self.session)
        raise ValueError(f"Unknown strategy {self.session.strategy}")

    def select_chunks(self, chunks: list[Any]) -> list[Any]:
        return self.get_selector().select(chunks)
