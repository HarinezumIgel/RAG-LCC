"""Helpers that bridge session-level marked PDFs to HTTP-served URLs."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from urllib.parse import quote

from starlette.responses import JSONResponse, Response

from Api.MarkedDocsStore import MarkedDocsStore
from Gui.GroundedAnswerHtmlRenderer import GroundedAnswerHtmlRenderer

if TYPE_CHECKING:
    from Globals.Session import Session


def marked_document_url(
    base_url: str, store: MarkedDocsStore, document_token: str, suffix: str
) -> str:
    """Build a clickable URL carrying both required capabilities."""
    return (
        f"{base_url.rstrip('/')}/marked/{document_token}{suffix}"
        f"?token={store.session_token}"
    )


def register_grounded_answer(
    session: Session,
    grounder: Any,
    answer_markdown: str,
    store: Optional[MarkedDocsStore],
    base_url: str,
) -> str:
    """Store a grounded answer or unverified answer preview as a short-lived HTML page."""
    if grounder is None or store is None or not base_url or not answer_markdown.strip():
        return ""

    evidence_texts = list(getattr(session, "chunk_texts_for_grounding", []) or [])
    evidence_texts.extend(grounder.collect_web_texts(session))
    if not evidence_texts:
        return ""

    verified = bool(getattr(session, "answer_grounding_verified", False))
    page, content_security_policy = GroundedAnswerHtmlRenderer().render(
        answer_markdown,
        grounder,
        session,
        evidence_texts,
        sentence_translations=getattr(session, "grounding_proxy_sentences", {}),
        verified=verified,
        mark_supported=True,
    )
    partially_grounded = not verified and grounder.has_grounded_parts(
        session, answer_markdown, evidence_texts
    )
    filename = "GroundedAnswer.html" if verified else "AnswerPreview.html"
    try:
        token = store.put(
            page.encode("utf-8"),
            filename,
            response_headers={
                "Content-Security-Policy": content_security_policy,
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )
    except ValueError as exc:
        from Gui.PrettyWriter import PrettyWriter

        PrettyWriter().write(
            "W",
            "GroundedAnswer",
            f"Could not cache formatted answer: {exc}",
        )
        return ""

    url = marked_document_url(base_url, store, token, ".html")
    ttl_minutes = max(1, store.ttl_seconds // 60)
    label = (
        "Grounded answer"
        if verified
        else "Not fully grounded" if partially_grounded else "Answer preview"
    )
    return (
        "\n\n---\n"
        f"🌐 **{label}** *(link expires in {ttl_minutes} min)*\n"
        f"- [Open formatted HTML answer]({url})\n"
    )


def marked_document_response(
    token_filename: str, store: Optional[MarkedDocsStore], session_token: str = ""
) -> Response:
    """Build the HTTP response for a stored marked document or answer page."""
    headers = {
        "Cache-Control": "private, no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    if store is None or not hmac.compare_digest(
        session_token.encode("utf-8"), store.session_token.encode("ascii")
    ):
        return JSONResponse(
            status_code=404, content={"detail": "Not found"}, headers=headers
        )

    token = token_filename.split(".", 1)[0]
    entry = store.get(token)
    if entry is None:
        return JSONResponse(
            status_code=404, content={"detail": "Not found"}, headers=headers
        )

    media_types: dict[str, str] = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".html": "text/html; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }
    safe_name = entry.filename.replace('"', "")
    disposition = (
        f'inline; filename="{safe_name}"'
        if safe_name.isascii()
        else f"inline; filename*=UTF-8''{quote(safe_name, safe='')}"
    )
    suffix = Path(entry.filename).suffix.lower()
    headers["Content-Disposition"] = disposition
    headers.update(entry.response_headers)
    return Response(
        content=entry.data,
        media_type=media_types.get(suffix, "application/octet-stream"),
        headers=headers,
    )


def register_marked_documents(
    session: Session,
    store: Optional[MarkedDocsStore],
    base_url: str,
) -> str:
    """Register every ``(source_path, pdf_bytes)`` tuple on *session* in
    *store* and return a markdown block of clickable links (when store is available)
    or plain text filenames (when store is None but sources exist).

    Returns an empty string when nothing to do (no marked documents, no sources).

    With store (SERVE_RAGCHATSERVICE_GROUNDED_DOCS=1):
        ---
        🔵 **Marked sources** *(links expire in N min)*
        - 🟢 [Hedgehogs.pdf (highlighted)](https://host/marked/<token>.pdf?token=<session-token>)

    Without store (SERVE_RAGCHATSERVICE_GROUNDED_DOCS=0):
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
            url = marked_document_url(base, store, token, out_suffix)
            url_map[src_path] = url
            lines.append(f"- 🟢 [{display_name} (highlighted)]({url})")

        if url_map:
            session.marked_docs_url_map = url_map  # type: ignore[attr-defined]

        if not lines:
            return ""

        ttl_min = max(1, store.ttl_seconds // 60)
        header = f"\n\n---\n🔵 **Marked sources** *(links expire in {ttl_min} min)*\n"
        return header + "\n".join(lines) + "\n"

    # Extract unique original source paths from selected chunks.
    chosen = getattr(session, "last_chosen_chunks", [])
    if not chosen:
        return ""

    source_paths: set[str] = set()
    source_names: set[str] = set()
    for doc in chosen:
        meta = getattr(doc, "metadata", {}) or {}
        if str(meta.get("Source", "")).lower() == "web":
            continue
        file_path = str(meta.get("FilePath", "")).strip()
        if file_path:
            source_paths.add(file_path)
            source_names.add(Path(file_path).name)

    # If visual marking did not produce highlighted documents, still expose
    # the selected local source documents as short-lived green links.
    if store is not None and not marked:
        base = base_url.rstrip("/") if base_url else ""
        if base:
            lines = []
            url_map: dict[str, str] = {}
            for file_path in sorted(source_paths):
                try:
                    source = Path(file_path)
                    if not source.is_file():
                        continue
                    suffix = source.suffix or ".bin"
                    token = store.put(source.read_bytes(), source.name)
                except (OSError, ValueError):
                    continue
                url = marked_document_url(base, store, token, suffix)
                url_map[file_path] = url
                lines.append(f"- 🟢 [{source.name} (source)]({url})")

            if url_map:
                session.marked_docs_url_map = url_map  # type: ignore[attr-defined]
                ttl_min = max(1, store.ttl_seconds // 60)
                header = (
                    f"\n\n---\n🔵 **Marked sources** "
                    f"*(links expire in {ttl_min} min)*\n"
                )
                return header + "\n".join(lines) + "\n"

    if not source_names:
        return ""

    sorted_names = sorted(source_names)
    header = f"\n\n---\n📄 **Sources**\n"
    items = [f"- {name}" for name in sorted_names]
    return header + "\n".join(items) + "\n"
