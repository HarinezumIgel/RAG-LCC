"""Subprocess entry point for marking one document outside the main process.

Run as ``python -m VisualMarkers.MarkWorker`` with ``src`` on ``PYTHONPATH``.
Reads one JSON job from stdin and writes the marked document bytes to stdout::

    {"path": "...", "highlight_color": "yellow",
     "snippets": [["text", 3, "orange"], ["text", null, null]]}

A separate lightweight module is used instead of ``ProcessPoolExecutor`` so the
worker does not re-import the application's ``__main__`` on Windows (spawn).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from VisualMarkers.VisualMarker import ChunkSnippet
from VisualMarkers.VisualMarkerFactory import VisualMarkerFactory

_SRC_DIR = str(Path(__file__).resolve().parent.parent)


def encode_job(
    path: str, snippets: Sequence[ChunkSnippet], highlight_color: str
) -> bytes:
    """Serialize one marking job for :func:`main`."""
    return json.dumps(
        {
            "path": path,
            "highlight_color": highlight_color,
            "snippets": [[s.text, s.page_number, s.color] for s in snippets],
        }
    ).encode("utf-8")


def run_job(job: dict[str, Any]) -> bytes:
    """Mark the document described by *job* and return its bytes."""
    path = str(job["path"])
    marker = VisualMarkerFactory.for_path(path)
    if marker is None:
        raise ValueError(f"No visual marker for {path!r}")
    snippets = [
        ChunkSnippet(text=str(text), page_number=page, color=color)
        for text, page, color in job["snippets"]
    ]
    return marker.mark_to_bytes(
        Path(path), snippets, highlight_color=str(job["highlight_color"])
    )


def mark_in_subprocess(
    path: str,
    snippets: Sequence[ChunkSnippet],
    highlight_color: str,
    timeout_s: float,
) -> bytes:
    """Mark *path* in a fresh worker process; raise ``RuntimeError`` on failure."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_DIR + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [sys.executable, "-m", "VisualMarkers.MarkWorker"],
            input=encode_job(path, snippets, highlight_color),
            capture_output=True,
            timeout=timeout_s,
            env=env,
            creationflags=creationflags,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"mark worker timed out after {timeout_s:g}s") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"mark worker exited with code {result.returncode}: {detail or 'no output'}"
        )
    return result.stdout


def main() -> int:
    try:
        job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        data = run_job(job)
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
