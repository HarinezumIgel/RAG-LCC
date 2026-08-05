"""Marked-document viewer helpers for the RAGChat CLI.

Deliberately kept stdlib-only so it can be imported and tested without
any of the heavy RAG-pipeline dependencies.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol


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
) -> None:
    """Write each ``(source_path, doc_bytes)`` pair to a temp file, then either
    emit OSC 8 clickable hyperlinks (when the terminal supports them) or present
    a numbered picker and open the chosen file with the OS default viewer.

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
        When provided, temp files are created inside ``<project_root>/tmp/``
        and ``_cleanup_dir`` refuses to delete anything outside that root.
        Falls back to the system temp directory when ``None`` or empty.
    _register_cleanup:
        Internal flag; set to False in tests to suppress ``atexit`` side-effects.
    """
    if not marked:
        return

    _tmp_base: "Path | None" = Path(project_root) / "tmp" if project_root else None

    tmp_dir_holder: list[Path] = []  # created on first actual write

    def _get_tmp_dir() -> Path:
        if not tmp_dir_holder:
            if _tmp_base is not None:
                _tmp_base.mkdir(parents=True, exist_ok=True)
            d = Path(tempfile.mkdtemp(prefix="rag_marked_", dir=_tmp_base))
            if _register_cleanup:
                atexit.register(_cleanup_dir, str(d), project_root or "")
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
        pretty.write(
            "I",
            "Marked sources",
            f"{len(items)} highlighted document(s) "
            "(click a link to open; use 'Save As' in the viewer to keep a copy):",
            color=cyan,
        )
        for src_path, out_path in items:
            label = f"{Path(src_path).name} (highlighted)"
            print(f"   📎 {label}: {out_path.resolve().as_uri()}")
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


def _cleanup_dir(path: str, project_root: str = "") -> None:
    import shutil

    abs_path = os.path.normpath(os.path.abspath(path))
    if not project_root:
        raise RuntimeError(
            f"_cleanup_dir: project_root is not set; refusing to delete '{abs_path}'."
        )
    abs_root = os.path.normpath(os.path.abspath(project_root))
    # Refuse drive/filesystem roots ("C:\" or "/"): rmtree there is catastrophic
    # and the abs_path == abs_root branch below would otherwise pass the guard.
    for candidate in (abs_root, abs_path):
        _, tail = os.path.splitdrive(candidate)
        if len(tail) <= 2:
            raise RuntimeError(
                f"_cleanup_dir: refusing to operate on drive/filesystem root "
                f"'{candidate}'."
            )
    # Jailbreak guard: refuse to delete anything outside the project root.
    if not (abs_path.startswith(abs_root + os.sep) or abs_path == abs_root):
        raise RuntimeError(
            f"_cleanup_dir: '{abs_path}' is outside the project root "
            f"'{abs_root}'; refusing to delete."
        )
    if not os.path.exists(abs_path):
        raise RuntimeError(
            f"_cleanup_dir: '{abs_path}' does not exist; refusing to delete."
        )
    shutil.rmtree(abs_path, ignore_errors=True)
