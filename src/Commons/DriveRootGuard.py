"""Early drive-root execution guard for the standalone CLI scripts in
``src/Scripts``.

Deliberately stdlib-only (no project imports, no ANSI helpers) so it can run
*before* any heavy imports and refuse execution instantly when the project
resolves to a drive/filesystem root — where the deletion path-guards would be
disabled. Mirrors ``Helpers.is_in_drive_root`` but exits cleanly via
``sys.exit(1)`` instead of raising, so scripts abort without a traceback.
"""

from __future__ import annotations

import os
import sys

_BRIGHT_RED = "\033[91m"
_RESET = "\033[0m"


def is_drive_root(project_root: str) -> bool:
    """Return True when *project_root* is empty/unresolved or resolves to a
    drive/filesystem root.

    The check inspects only the tail returned by ``os.path.splitdrive``
    (the part after the drive, e.g. ``"\\"`` for ``"C:\\"`` or ``"/"``
    for POSIX root). Root tails are at most 2 characters long.
    """
    if not project_root:
        return True
    _, tail = os.path.splitdrive(project_root)
    return len(tail) <= 2


def drive_root_message(project_root: str) -> str:
    """Build the uppercase drive-root refusal message for *project_root*."""
    shown = project_root or "<UNRESOLVED>"
    return (
        "EXECUTION BLOCKED: RAG-LCC MUST NOT RUN FROM A DRIVE OR "
        f"FILESYSTEM ROOT (RESOLVED PROJECT ROOT: '{shown}'). "
        "INSTALL RAG-LCC INSIDE A NAMED SUBDIRECTORY "
        "(E.G. 'D:\\RAG-LCC' OR '/HOME/USER/RAG-LCC') AND RE-RUN."
    )


def assert_not_drive_root(script_file: str) -> None:
    """Exit(1) when the project root resolves to a drive/filesystem root.

    The project root is taken as two levels above ``script_file`` — every
    script lives at ``src/Scripts/X.py``, so ``..\\..`` is the project root.
    A resolved splitdrive tail of <= 2 chars (e.g. ``"\\"`` from
    ``"C:\\"`` or ``"/"``), or an unresolved path, is treated as a
    drive/filesystem root.
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))

    if is_drive_root(project_root):
        msg = drive_root_message(project_root)
        print(f"{_BRIGHT_RED}{msg}{_RESET}", file=sys.stderr)
        sys.exit(1)
