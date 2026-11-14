"""Extracted DocumentMarkingService behavior; the coordinating facade owns shared state."""

from __future__ import annotations

import os
import time
from typing import Any

from Chat.RagRetrievalContext import RagRetrievalContext
from Globals.Session import Session
from Gui.Colors import CYAN
from Helpers.DebugHelper import DebugHelper


class DocumentMarkingService(RagRetrievalContext):
    def _mark_sources(
        self, mySession: Session, chosen: list[Any], *, summary: bool = False
    ) -> None:
        """Orchestrate in-memory visual marking of local source documents.

        Yellow highlights = retrieved chunk spans.
        Orange highlights = verbatim chunk fragments that grounded the answer.
        Summaries omit yellow spans when summary_support_only is enabled.
        """
        self.pretty.write(
            "W", "VisualMarker", f"_mark_sources called: {len(chosen)} chunk(s)"
        )

        grouped = self._group_chunks_by_file(chosen)
        if not grouped:
            self.pretty.write(
                "W",
                "VisualMarker",
                "No local PDF paths resolved — nothing to highlight (check FilePath metadata above)",
            )
            mySession.marked_documents = []
            return

        highlight_color, answer_mark_color = self._resolve_mark_colors()
        from Grounding.Grounder import Grounder

        grounded_snippets = Grounder().build_grounded_snippets(
            mySession, chosen, answer_mark_color
        )
        if summary and self.cfg.get_bool(
            "_GROUNDING_MARKING.summary_support_only", True
        ):
            grouped = {path: [] for path in grouped if grounded_snippets.get(path)}
            if not grouped:
                self.pretty.write(
                    "W",
                    "VisualMarker",
                    "No verified source passages for this summary; zero new highlights.",
                )
        started = time.perf_counter()
        produced = self._produce_marked_bytes(
            grouped, grounded_snippets, highlight_color
        )
        if DebugHelper.check_session(mySession, 33):
            self.pretty.write(
                "D",
                "Grounding",
                f"Timing document marking: {(time.perf_counter() - started) * 1000:.1f} ms",
                color=CYAN,
            )

        mySession.marked_documents = produced
        if produced:
            self.pretty.write(
                "I",
                "VisualMarker",
                f"Prepared {len(produced)} highlighted document(s) (in memory)",
                color=CYAN,
            )

    def _group_chunks_by_file(self, chosen: list[Any]) -> "dict[str, list[Any]]":
        """Return a {file_path: [ChunkSnippet, …]} dict for all local chunks."""
        from VisualMarkers import ChunkSnippet

        grouped: dict[str, list[Any]] = {}
        for doc in chosen:
            meta = getattr(doc, "metadata", {}) or {}
            if str(meta.get("Source", "")).lower() == "web":
                continue
            file_path = str(meta.get("FilePath", "")).strip()
            if not file_path or not os.path.isfile(file_path):
                self.pretty.write(
                    "D",
                    "VisualMarker",
                    f"Skipping chunk: FilePath={file_path!r} "
                    f"isfile={os.path.isfile(file_path) if file_path else 'n/a'}",
                )
                continue
            page_number = meta.get("PageNumber")
            try:
                page_int: int | None = (
                    int(page_number) if page_number is not None else None
                )
            except (TypeError, ValueError):
                page_int = None
            grouped.setdefault(file_path, []).append(
                ChunkSnippet(
                    text=getattr(doc, "page_content", "") or "", page_number=page_int
                )
            )
        return grouped

    def _resolve_mark_colors(self) -> "tuple[str, str]":
        """Read marker colors through Config, defaulting to yellow and orange."""
        highlight_color = self.cfg.get_str("_MARKED_DOCS_COLORS.highlight", "yellow")
        answer_mark_color = self.cfg.get_str(
            "_MARKED_DOCS_COLORS.answer_mark", "orange"
        )
        return highlight_color or "yellow", answer_mark_color or "orange"

    def _produce_marked_bytes(
        self,
        grouped: "dict[str, list[Any]]",
        grounded_snippets: "dict[str, list[Any]]",
        highlight_color: str,
    ) -> "list[tuple[str, bytes]]":
        """Combine yellow + orange snippets, run the marker, return (path, bytes) pairs."""
        from pathlib import Path

        from VisualMarkers import VisualMarkerFactory

        grounded_paths = set(grounded_snippets.keys())
        only_grounded = bool(grounded_paths)
        if DebugHelper.check(self.cfg, 33):
            total_orange = sum(len(v) for v in grounded_snippets.values())
            self.pretty.write(
                "D",
                "Grounding",
                f"grounded_paths={len(grounded_paths)}  total_orange_snippets={total_orange}  "
                f"only_grounded={only_grounded}  grouped_files={list(grouped.keys())}",
                color=CYAN,
            )

        jobs: list[tuple[str, Any, list[Any]]] = []
        for src_path, snippets in grouped.items():
            if only_grounded and src_path not in grounded_paths:
                if DebugHelper.check(self.cfg, 33):
                    self.pretty.write(
                        "D",
                        "Grounding",
                        f"skipping (no grounded sentences): {src_path!r}",
                        color=CYAN,
                    )
                continue
            marker = VisualMarkerFactory.for_path(src_path)
            if marker is None:
                continue
            all_snippets = list(snippets) + grounded_snippets.get(src_path, [])
            if DebugHelper.check(self.cfg, 33):
                self.pretty.write(
                    "D",
                    "Grounding",
                    f"marking {os.path.basename(src_path)!r}: "
                    f"yellow={len(snippets)}  orange={len(grounded_snippets.get(src_path, []))}  "
                    f"marker={type(marker).__name__}",
                    color=CYAN,
                )
            jobs.append((src_path, marker, all_snippets))

        results: dict[str, bytes] = self._mark_pdfs_in_parallel(jobs, highlight_color)
        produced: list[tuple[str, bytes]] = []
        for src_path, marker, all_snippets in jobs:
            data = results.get(src_path)
            if data is None:
                try:
                    data = marker.mark_to_bytes(
                        Path(src_path), all_snippets, highlight_color=highlight_color
                    )
                except Exception as exc:
                    self.pretty.write(
                        "W", "VisualMarker", f"Failed to mark {src_path}: {exc}"
                    )
                    continue
            if Path(src_path).suffix.lower() == ".pdf":
                import io

                from pypdf import PdfReader

                metadata = PdfReader(io.BytesIO(data)).metadata
                count = metadata.get("/RAGHighlightCount") if metadata else None
                pages = metadata.get("/RAGHighlightPages") if metadata else None
                if count is None:
                    self.pretty.write(
                        "W",
                        "VisualMarker",
                        f"{Path(src_path).name}: highlight count unavailable.",
                    )
                elif int(count) == 0:
                    self.pretty.write(
                        "W",
                        "VisualMarker",
                        f"{Path(src_path).name}: zero new highlights.",
                    )
                else:
                    self.pretty.write(
                        "I",
                        "VisualMarker",
                        f"{Path(src_path).name}: {count} new highlights on physical pages {pages}.",
                    )
            produced.append((src_path, data))
        return produced

    def _mark_pdfs_in_parallel(
        self,
        jobs: "list[tuple[str, Any, list[Any]]]",
        highlight_color: str,
    ) -> "dict[str, bytes]":
        """Mark PDF jobs in worker processes when worthwhile; return {path: bytes}.

        Files missing from the result (not parallelized or worker failed) are
        marked in-process by the caller.
        """
        from concurrent.futures import ThreadPoolExecutor

        from VisualMarkers.MarkWorker import mark_in_subprocess

        max_workers = self.cfg.get_int("_GROUNDING_MARKING.max_workers")
        min_scan_pages = self.cfg.get_int("_GROUNDING_MARKING.min_scan_pages")
        timeout_s = float(self.cfg.get_int("_GROUNDING_MARKING.worker_timeout_s"))
        pdf_jobs = [job for job in jobs if job[0].lower().endswith(".pdf")]
        if max_workers <= 1 or len(pdf_jobs) < 2:
            return {}
        estimated_pages = 0
        for _path, _marker, snippets in pdf_jobs:
            pages = {getattr(s, "page_number", None) for s in snippets}
            if None in pages:
                # A page-less snippet may scan the whole document.
                estimated_pages = max(estimated_pages, min_scan_pages)
            estimated_pages += len(pages - {None})
        if estimated_pages < min_scan_pages:
            return {}

        def _run(job: "tuple[str, Any, list[Any]]") -> "tuple[str, bytes | None]":
            path, _marker, snippets = job
            try:
                return path, mark_in_subprocess(
                    path, snippets, highlight_color, timeout_s
                )
            except Exception as exc:
                self.pretty.write(
                    "W",
                    "VisualMarker",
                    f"Parallel marking failed for {os.path.basename(path)!r}; "
                    f"marking in-process: {exc}",
                )
                return path, None

        workers = min(max_workers, len(pdf_jobs))
        if DebugHelper.check(self.cfg, 33):
            self.pretty.write(
                "D",
                "Grounding",
                f"parallel PDF marking: files={len(pdf_jobs)}  workers={workers}  "
                f"estimated_pages={estimated_pages}",
                color=CYAN,
            )
        results: dict[str, bytes] = {}
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="pdf-mark"
        ) as pool:
            for path, data in pool.map(_run, pdf_jobs):
                if data is not None:
                    results[path] = data
        return results
