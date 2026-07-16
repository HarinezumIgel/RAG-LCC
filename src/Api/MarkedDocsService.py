"""Helpers that bridge session-level marked PDFs to HTTP-served URLs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from Api.MarkedDocsStore import MarkedDocsStore
from Globals.Session import Session


def register_marked_documents(
    session: Session,
    store: Optional[MarkedDocsStore],
    base_url: str,
) -> str:
    """Register every ``(source_path, pdf_bytes)`` tuple on *session* in
    *store* and return a markdown block of clickable links (when store is available)
    or plain text filenames (when store is None but sources exist).

    Returns an empty string when nothing to do (no marked documents, no sources).

    With store (SERVE_IN_MEMORY_DOCS_HTTP=1):
        ---
        📎 **Marked sources** _(links expire in N min)_
        - [Hedgehogs.pdf (highlighted)](https://host/marked/<token>.pdf)

    Without store (SERVE_IN_MEMORY_DOCS_HTTP=0):
        ---
        📄 **Sources**
        - Hedgehogs.pdf
        - Cats.md
    """
    marked: List[Tuple[str, bytes]] = list(
        getattr(session, "marked_documents", []) or []
    )

    # If store exists and we have marked documents: create HTTP links
    if store is not None and marked:
        base = base_url.rstrip("/") if base_url else ""
        if not base:
            return ""

        lines: list[str] = []
        url_map: dict[str, str] = {}
        for src_path, pdf_bytes in marked:
            try:
                stem = Path(src_path).stem
                src_suffix = Path(src_path).suffix or ".pdf"
                out_suffix = ".txt.md" if src_suffix == ".txt" else src_suffix
                display_name = f"{stem}{src_suffix}"
                stored_name = f"{stem}_marked{out_suffix}"
                token = store.put(pdf_bytes, stored_name)
            except Exception:
                # Cache full / oversized — skip silently rather than break the answer.
                continue
            url = f"{base}/marked/{token}{out_suffix}"
            url_map[src_path] = url
            lines.append(f"- [{display_name} (highlighted)]({url})")

        if url_map:
            session.marked_docs_url_map = url_map  # type: ignore[attr-defined]

        if not lines:
            return ""

        ttl_min = max(1, store.ttl_seconds // 60)
        header = f"\n\n---\n📎 **Marked sources** _(links expire in {ttl_min} min)_\n"
        return header + "\n".join(lines) + "\n"

    # If no store but we have sources: show filenames only (no links)
    # Extract original source paths from session chunks
    chosen = getattr(session, "last_chosen_chunks", [])
    if not chosen:
        return ""

    source_names: set[str] = set()
    for doc in chosen:
        meta = getattr(doc, "metadata", {}) or {}
        if str(meta.get("Source", "")).lower() == "web":
            continue
        file_path = str(meta.get("FilePath", "")).strip()
        if file_path:
            source_names.add(Path(file_path).name)

    if not source_names:
        return ""

    sorted_names = sorted(source_names)
    header = f"\n\n---\n📄 **Sources**\n"
    items = [f"- {name}" for name in sorted_names]
    return header + "\n".join(items) + "\n"
