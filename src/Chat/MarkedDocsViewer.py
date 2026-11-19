"""Marked-document viewer helpers for the RAGChat CLI.

Deliberately kept lightweight (stdlib + Commons.DriveRootGuard, which is also
stdlib-only) so it can be imported and tested without heavy RAG-pipeline
dependencies.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from Commons.DriveRootGuard import resolve_guard_path

if TYPE_CHECKING:
    from Chat.LocalDocsServer import LocalDocsServer
    from Helpers.FileUtils import FileUtils


class _Pretty(Protocol):
    def write(self, *args: Any, **kwargs: Any) -> Any: ...


def supports_osc8() -> bool:
    """Return True when the running terminal is known to support OSC 8 hyperlinks.

    Detection is env-var based (no live probe) to avoid timing issues with
    the terminal's response buffer.  Terminals confirmed to support OSC 8:
    Windows Terminal, VS Code, iTerm2, Kitty, Ghostty, WezTerm, Foot, and
    VTE-based terminals (GNOME Terminal, Tilix, …) since VTE ≥ 0.50.
    """
    env = os.environ
    if env.get("WT_SESSION"):  # Windows Terminal
        return True
    tp = env.get("TERM_PROGRAM", "")
    if tp in ("iTerm.app", "vscode", "WezTerm", "JetBrains-JediTerm"):
        return True
    if env.get("WEZTERM_EXECUTABLE"):  # WezTerm (alt var)
        return True
    if env.get("KITTY_WINDOW_ID"):  # Kitty
        return True
    t = env.get("TERM", "")
    if t in ("xterm-kitty", "xterm-ghostty", "foot"):  # Kitty / Ghostty / Foot
        return True
    if env.get("GHOSTTY_RESOURCES_DIR"):  # Ghostty (alt var)
        return True
    try:  # VTE ≥ 0.50 (≈ 5000)
        if int(env.get("VTE_VERSION", "0")) >= 5000:
            return True
    except ValueError:
        pass
    return False


def open_marked_documents(
    marked: list[tuple[str, bytes]],
    pretty: _Pretty,
    cyan: str = "",
    *,
    project_root: "str | None" = None,
    _register_cleanup: bool = True,
    _file_utils: "FileUtils | None" = None,
    local_server: LocalDocsServer | None = None,
) -> None:
    """Write each ``(source_path, doc_bytes)`` pair to a temp file, then either
    emit OSC 8 clickable hyperlinks (when the terminal supports them) or present
    a numbered picker and open the chosen file with the OS default viewer.

    HTML is offered only through ``local_server``, without writing files or
    opening viewers. Other formats retain their file links or picker.

    Temp files are removed at process exit; the user can "Save As" in the
    viewer to keep a permanent copy.

    Parameters
    ----------
    marked:
        List of ``(original_source_path, output_bytes)`` tuples.
    pretty:
        PrettyWriter (or compatible stub) for structured log lines.
    cyan:
        ANSI colour code to pass to ``pretty.write`` — kept as a parameter
        so the function stays free of Gui imports.
    project_root:
        Absolute path to the project root (``_ABSOLUTE_PATH`` from config).
        When provided, temp files are created inside
        ``<project_root>/tmp/ragchat/`` and ``_cleanup_dir`` refuses to delete
        anything outside that service-specific directory.
        Falls back to the system temp directory when ``None`` or empty.
    _register_cleanup:
        Internal flag; set to False in tests to suppress ``atexit`` side-effects.
    """
    if not marked:
        return

    html_marked: list[tuple[str, bytes]] = []
    file_marked: list[tuple[str, bytes]] = []
    for document in marked:
        if Path(document[0]).suffix.lower() in (".html", ".htm"):
            html_marked.append(document)
        else:
            file_marked.append(document)

    if html_marked and local_server is None:
        pretty.write(
            "W",
            "Marked sources",
            "HTML links require SERVE_RAGCHAT_GROUNDED_DOCS=1 in "
            "Config_Internet_Env.py. Enable it and restart RAGChat.",
        )
    elif html_marked and local_server is not None:
        pretty.write(
            "I",
            "Marked sources",
            f"{len(html_marked)} highlighted HTML document(s) "
            f"(links expire after {local_server.store.ttl_seconds} s or at shutdown):",
            color=cyan,
        )
        for src_path, doc_bytes in html_marked:
            source = Path(src_path)
            try:
                url = local_server.put(
                    doc_bytes, f"{source.stem}_marked{source.suffix}"
                )
            except (OSError, ValueError, RuntimeError) as exc:
                pretty.write(
                    "W", "Marked sources", f"Could not serve {source.name}: {exc}"
                )
                continue
            print(f"   {source.name} (highlighted): {url}")

    marked = file_marked
    if not marked:
        return

    _tmp_base: "Path | None" = (
        Path(project_root) / "tmp" / "ragchat" if project_root else None
    )

    tmp_dir_holder: list[Path] = []  # created on first actual write

    def _get_tmp_dir() -> Path:
        if not tmp_dir_holder:
            if _register_cleanup and (not project_root or _file_utils is None):
                raise RuntimeError(
                    "RAGChat temp files require a project root and delete guard "
                    "when process-exit cleanup is enabled."
                )
            if _tmp_base is not None:
                _tmp_base.mkdir(parents=True, exist_ok=True)
            d = Path(tempfile.mkdtemp(prefix="rag_marked_", dir=_tmp_base))
            if _register_cleanup:
                atexit.register(
                    _cleanup_dir,
                    str(d),
                    project_root or "",
                    str(_tmp_base or tempfile.gettempdir()),
                    _file_utils,
                )
            tmp_dir_holder.append(d)
        return tmp_dir_holder[0]

    def _out_suffix(src_path: str) -> str:
        s = Path(src_path).suffix or ".pdf"
        return ".txt.md" if s == ".txt" else s

    def _write_one(src_path: str, doc_bytes: bytes) -> "Path | None":
        out = _get_tmp_dir() / f"{Path(src_path).stem}_marked{_out_suffix(src_path)}"
        try:
            out.write_bytes(doc_bytes)
            return out
        except Exception as exc:
            pretty.write(
                "W",
                "Marked sources",
                f"Could not write temp file for {src_path}: {exc}",
            )
            return None

    # ── OSC 8 path — write eagerly (all links must exist before user clicks) ──
    if supports_osc8():
        items: list[tuple[str, Path]] = []
        for src_path, doc_bytes in marked:
            out = _write_one(src_path, doc_bytes)
            if out:
                items.append((src_path, out))
        if not items:
            return
        link_green: str = "\033[32m"
        reset: str = "\033[0m"
        pretty.write(
            "I",
            "Marked sources",
            f"{len(items)} highlighted document(s) "
            "(click a link to open; use 'Save As' in the viewer to keep a copy):",
            color=cyan,
        )
        for src_path, out_path in items:
            label = f"{Path(src_path).name} (highlighted)"
            print(f"{link_green}   📎 {label}: {out_path.resolve().as_uri()}{reset}")
        return

    # ── Picker path — write lazily (only the file the user selects) ───────────
    pretty.write(
        "I",
        "Marked sources",
        f"{len(marked)} highlighted document(s) "
        "(use 'Save As' in the viewer to keep a permanent copy):",
        color=cyan,
    )
    for idx, (src_path, _) in enumerate(marked, start=1):
        note = " (opens as .txt.md)" if _out_suffix(src_path) == ".txt.md" else ""
        print(f"   [{idx}] {Path(src_path).name}{note}")

    n = len(marked)
    prompt = (
        f"   Open file [1\u2013{n}, or Enter to skip]: "
        if n > 1
        else "   Open file [1, or Enter to skip]: "
    )
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            break
        try:
            choice = int(raw)
        except ValueError:
            print(
                f"   Please enter a number between 1 and {n}, or press Enter to skip."
            )
            continue
        if not (1 <= choice <= n):
            print(
                f"   Please enter a number between 1 and {n}, or press Enter to skip."
            )
            continue
        src_path, doc_bytes = marked[choice - 1]
        out_path = _write_one(src_path, doc_bytes)
        if out_path is None:
            continue
        try:
            _open_with_os(out_path)
        except Exception as exc:
            pretty.write(
                "W", "Marked sources", f"Could not open {out_path.name}: {exc}"
            )


def open_grounded_answer_html(
    filename: str,
    data: bytes,
    pretty: _Pretty,
    cyan: str = "",
    *,
    local_server: LocalDocsServer | None = None,
    response_headers: Mapping[str, str] | None = None,
) -> str | None:
    """Offer an expiring HTTP link; never write HTML file links or auto-open."""
    grounded = filename == "GroundedAnswer.html"
    label = "Grounded answer" if grounded else "Answer preview"
    if local_server is None:
        pretty.write(
            "W",
            label,
            "Formatted browser page requires SERVE_RAGCHAT_GROUNDED_DOCS=1 in "
            "Config_Internet_Env.py. Enable it and restart RAGChat.",
        )
        return None
    try:
        url = local_server.put(data, filename, response_headers=response_headers)
    except (OSError, ValueError, RuntimeError) as exc:
        pretty.write("W", label, f"Could not serve {filename}: {exc}")
        return None
    pretty.write(
        "I",
        label,
        f"{'Grounded' if grounded else 'Formatted'} browser page (click to open; expires after "
        f"{local_server.store.ttl_seconds} s or at shutdown):",
        color=cyan,
    )
    print(f"   {url}")
    return url


def _open_with_os(path: Path) -> None:
    if sys.platform == "win32":
        # Use `cmd /c start` instead of os.startfile so the spawned process
        # is fully detached from the parent console and its stdout/stderr
        # output (e.g. VS Code startup diagnostics) does not bleed into the
        # terminal.
        subprocess.Popen(
            ["cmd", "/c", "start", "", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _cleanup_dir(
    path: str,
    project_root: str = "",
    store_root: str = "",
    file_utils: "FileUtils | None" = None,
) -> None:
    import shutil

    normalized_path = resolve_guard_path(path)
    if not project_root and not store_root:
        raise RuntimeError(
            f"_cleanup_dir: project_root is not set and no temp store was provided; "
            f"refusing to delete '{normalized_path}'."
        )

    normalized_root = resolve_guard_path(project_root) if project_root else ""
    expected_store_root = (
        resolve_guard_path(os.path.join(normalized_root, "tmp", "ragchat"))
        if normalized_root
        else resolve_guard_path(tempfile.gettempdir())
    )
    normalized_store_root = resolve_guard_path(store_root or expected_store_root)
    if normalized_store_root != expected_store_root:
        raise RuntimeError(
            f"_cleanup_dir: refusing unexpected temp store '{normalized_store_root}'."
        )

    if file_utils is None or not file_utils.is_safe_delete_path(normalized_path):
        raise RuntimeError(
            f"_cleanup_dir: FileUtils.is_safe_delete_path rejected "
            f"'{normalized_path}'; refusing to delete."
        )

    if normalized_path == normalized_store_root:
        raise RuntimeError(
            f"_cleanup_dir: refusing to delete the RAGChat temp store root "
            f"'{normalized_store_root}'."
        )
    if os.path.normcase(os.path.dirname(normalized_path)) != os.path.normcase(
        normalized_store_root
    ):
        raise RuntimeError(
            f"_cleanup_dir: '{normalized_path}' is outside the RAGChat temp store "
            f"'{normalized_store_root}' or is not a direct child; refusing to delete."
        )

    if not os.path.exists(normalized_path):
        raise RuntimeError(
            f"_cleanup_dir: '{normalized_path}' does not exist; refusing to delete."
        )

    shutil.rmtree(normalized_path)
