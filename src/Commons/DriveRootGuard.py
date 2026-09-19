"""Early drive-root execution guards for app and script entrypoints.

Deliberately stdlib-only (no project imports, no ANSI helpers) so it can run
*before* any heavy imports and refuse execution instantly when the project
resolves to a drive/filesystem root — where the deletion path-guards would be
disabled. Mirrors ``Helpers.is_in_drive_root`` but exits cleanly via
``sys.exit(1)`` instead of raising, so scripts abort without a traceback.
"""

from __future__ import annotations

import ntpath
import os
import sys
from typing import Any

_BRIGHT_RED = "\033[91m"
_RESET = "\033[0m"

_PATH_SLOT_HINT_TOKENS: set[str] = {
    "PATH",
    "PATHS",
    "DIR",
    "DIRS",
    "DIRECTORY",
    "DIRECTORIES",
    "FOLDER",
    "FOLDERS",
    "ROOT",
    "ROOTS",
    "HOME",
    "CACHE",
    "FILE",
    "FILES",
    "LOG",
    "LOGS",
}


def _is_root_tail(tail: str) -> bool:
    normalized = tail.strip()
    return normalized in {"", "/", "\\", ":", ":/", ":\\"}


def _guard_fail(message: str) -> None:
    """Print a bright-red guard failure and terminate the process."""
    print(f"{_BRIGHT_RED}{message}{_RESET}", file=sys.stderr)
    sys.exit(1)


def _is_within(path_value: str, root_value: str) -> bool:
    """Return True when *path_value* is equal to or nested under *root_value*."""
    try:
        common = os.path.normcase(
            os.path.normpath(os.path.commonpath([path_value, root_value]))
        )
        root = os.path.normcase(os.path.normpath(root_value))
        return common == root
    except ValueError:
        return False


def _validate_configured_root_alignment(
    *,
    configured_project_root: str | None,
    derived_project_root: str,
) -> None:
    """Validate optional configured root against a derived project root."""
    if configured_project_root is None:
        return

    configured_raw = str(configured_project_root).strip()
    # Guard raw root-like values (for example "\\" on POSIX) before
    # resolve_guard_path can reinterpret them relative to derived_project_root.
    if is_drive_root(configured_raw, resolve_traversal=False):
        shown = configured_raw or "<UNRESOLVED>"
        _guard_fail(
            "EXECUTION BLOCKED: CONFIGURED _ABSOLUTE_PATH RESOLVES TO A DRIVE OR "
            f"FILESYSTEM ROOT ('{shown}'). FIX _ABSOLUTE_PATH IN "
            "src/Configuration/Config_Global.py."
        )

    configured_resolved = resolve_guard_path(
        configured_raw,
        base_dir=derived_project_root,
    )
    if is_drive_root(configured_resolved):
        _guard_fail(
            "EXECUTION BLOCKED: CONFIGURED _ABSOLUTE_PATH RESOLVES TO A DRIVE OR "
            f"FILESYSTEM ROOT ('{configured_resolved}'). FIX _ABSOLUTE_PATH IN "
            "src/Configuration/Config_Global.py."
        )

    if not (
        _is_within(derived_project_root, configured_resolved)
        or _is_within(configured_resolved, derived_project_root)
    ):
        _guard_fail(
            "EXECUTION BLOCKED: CONFIGURED _ABSOLUTE_PATH IS NOT PATH-RELATED TO "
            "THE ENTRYPOINT-DERIVED PROJECT ROOT. "
            f"CONFIGURED='{configured_resolved}' DERIVED='{derived_project_root}'."
        )


def _looks_like_drive_root(path_value: str) -> bool:
    _, nt_tail = ntpath.splitdrive(path_value)
    if _is_root_tail(nt_tail):
        return True

    _, os_tail = os.path.splitdrive(path_value)
    if _is_root_tail(os_tail):
        return True

    return False


def is_drive_root(
    project_root: str,
    *,
    resolve_traversal: bool = True,
    base_dir: str | None = None,
) -> bool:
    """Return True when *project_root* is empty/unresolved or resolves to a
    drive/filesystem root.

    The check accepts both POSIX and Windows-style roots on every platform,
    including malformed Windows root-like strings such as ``"C::/"``.

    When *resolve_traversal* is True (default), relative paths are normalized
    via :func:`resolve_guard_path` first so traversal inputs like ``"../../.."``
    are blocked if they collapse to a drive/filesystem root.
    """
    if not project_root:
        return True

    if _looks_like_drive_root(project_root):
        return True

    if not resolve_traversal:
        return False

    resolved_path = resolve_guard_path(project_root, base_dir=base_dir)
    if not resolved_path:
        return True

    if resolved_path == project_root:
        return False

    return _looks_like_drive_root(resolved_path)


def split_path_candidates(path_value: str) -> list[str]:
    """Split a path slot value into one or more candidate path strings.

    Supports OS-pair values separated by ``|``.
    """
    if "|" not in path_value:
        value = path_value.strip()
        return [value] if value else []
    return [part.strip() for part in path_value.split("|") if part.strip()]


def _looks_like_filesystem_path(value: str) -> bool:
    """Return True when *value* resembles a filesystem path string."""
    raw = value.strip()
    if not raw:
        return False

    if raw in {".", ".."}:
        return True

    # Recognize Windows drive-prefixed values (for example "C:") on all OSes.
    nt_drive, _ = ntpath.splitdrive(raw)
    if nt_drive:
        return True

    drive, _ = os.path.splitdrive(raw)
    if drive:
        return True

    if raw.startswith(("~", "./", "../", "/", "\\")):
        return True

    if "/" in raw or "\\" in raw:
        return True

    basename = os.path.basename(raw)
    _, ext = os.path.splitext(basename)
    if ext:
        suffix = ext[1:]
        if suffix.isalnum() and 1 <= len(suffix) <= 10:
            return True

    return False


def is_path_like_slot(slot_name: str, value: Any) -> bool:
    """Return True when a config slot likely contains a filesystem path."""
    if not isinstance(value, str):
        return False

    raw = value.strip()
    if not raw:
        return False

    # Exclude obvious non-filesystem URLs first.
    lowered = raw.lower()
    if "://" in lowered and not lowered.startswith("file://"):
        return False

    slot_upper = slot_name.upper()
    normalized_slot = slot_upper
    for separator in (".", "[", "]", "-", " "):
        normalized_slot = normalized_slot.replace(separator, "_")
    tokens = {t for t in normalized_slot.split("_") if t}
    if not (tokens & _PATH_SLOT_HINT_TOKENS):
        return False

    # URL-shaped slots are network endpoints, not local filesystem paths.
    if "URL" in tokens or "URI" in tokens:
        return False

    if not _looks_like_filesystem_path(raw):
        return False

    return True


def resolve_guard_path(path_value: str, *, base_dir: str | None = None) -> str:
    """Return normalized absolute path for guard checks.

    Relative values are resolved against *base_dir* when provided; otherwise the
    current working directory is used by ``os.path.abspath``.

    Windows-drive values are normalized with ``ntpath`` and preserved even on
    POSIX so guard checks can still classify drive-root-shaped inputs.
    """
    raw = path_value.strip()
    if not raw:
        return ""

    expanded = os.path.expanduser(raw)
    nt_drive, _ = ntpath.splitdrive(expanded)
    if nt_drive:
        return ntpath.normpath(expanded)

    drive, _ = os.path.splitdrive(expanded)
    if base_dir and not drive and not os.path.isabs(expanded):
        expanded = os.path.join(base_dir, expanded)

    return os.path.normpath(os.path.abspath(expanded))


def drive_root_message(project_root: str) -> str:
    """Build the uppercase drive-root refusal message for *project_root*."""
    shown = project_root or "<UNRESOLVED>"
    return (
        "EXECUTION BLOCKED: RAG-LCC MUST NOT RUN FROM A DRIVE OR "
        f"FILESYSTEM ROOT (RESOLVED PROJECT ROOT: '{shown}'). "
        "INSTALL RAG-LCC INSIDE A NAMED SUBDIRECTORY "
        "(E.G. 'D:\\RAG-LCC' OR '/HOME/USER/RAG-LCC') AND RE-RUN."
    )


def _resolve_expected_start_root(
    *,
    configured_project_root: str | None,
    derived_project_root: str,
) -> str:
    """Return the project root that startup cwd must match."""
    if configured_project_root is None:
        return os.path.normpath(os.path.abspath(derived_project_root))

    resolved = resolve_guard_path(
        configured_project_root,
        base_dir=derived_project_root,
    )
    return resolved or os.path.normpath(os.path.abspath(derived_project_root))


def _assert_cwd_matches_project_root(expected_project_root: str) -> None:
    """Abort when cwd does not equal *expected_project_root*."""
    cwd = os.path.normpath(os.path.abspath(os.getcwd()))
    expected = os.path.normpath(os.path.abspath(expected_project_root))

    if os.path.normcase(cwd) != os.path.normcase(expected):
        _guard_fail(
            "Start this app from project root: "
            f"{expected} (current working directory: {cwd})"
        )


def assert_script_project_root(
    script_file: str,
    *,
    configured_project_root: str | None = None,
    require_cwd_match: bool = False,
) -> str:
    """Validate script location and optional config root alignment.

    The script must live under ``<project_root>/src/Scripts``. The derived
    ``project_root`` is always checked against drive/filesystem root. When
    ``configured_project_root`` is provided, it must not resolve to a root and
    must be path-related to the script-derived project root (one contains the
    other). When ``require_cwd_match`` is True, startup is refused unless the
    current working directory equals the configured project root (when present)
    or the derived project root.

    Returns the script-derived project root on success; exits(1) on violation.
    """
    script_path = os.path.abspath(script_file)
    script_dir = os.path.dirname(script_path)
    src_dir = os.path.normpath(os.path.join(script_dir, ".."))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))

    if (
        os.path.basename(script_dir).lower() != "scripts"
        or os.path.basename(src_dir).lower() != "src"
    ):
        _guard_fail(
            "EXECUTION BLOCKED: SCRIPT LOCATION MUST BE '<PROJECT>/src/Scripts'. "
            f"RECEIVED: '{script_path}'."
        )

    if is_drive_root(project_root):
        _guard_fail(drive_root_message(project_root))

    _validate_configured_root_alignment(
        configured_project_root=configured_project_root,
        derived_project_root=project_root,
    )

    if require_cwd_match:
        expected_project_root = _resolve_expected_start_root(
            configured_project_root=configured_project_root,
            derived_project_root=project_root,
        )
        _assert_cwd_matches_project_root(expected_project_root)

    return project_root


def assert_app_project_root(
    app_file: str,
    *,
    configured_project_root: str | None = None,
) -> str:
    """Validate app location and optional config root alignment.

    The app entrypoint must live under ``<project_root>/src/Apps``. The derived
    ``project_root`` is always checked against drive/filesystem root. When
    ``configured_project_root`` is provided, it must not resolve to a root and
    must be path-related to the app-derived project root (one contains the
    other).

    Returns the app-derived project root on success; exits(1) on violation.
    """
    app_path = os.path.abspath(app_file)
    app_dir = os.path.dirname(app_path)
    src_dir = os.path.normpath(os.path.join(app_dir, ".."))
    project_root = os.path.normpath(os.path.join(app_dir, "..", ".."))

    if (
        os.path.basename(app_dir).lower() != "apps"
        or os.path.basename(src_dir).lower() != "src"
    ):
        _guard_fail(
            "EXECUTION BLOCKED: APP LOCATION MUST BE '<PROJECT>/src/Apps'. "
            f"RECEIVED: '{app_path}'."
        )

    if is_drive_root(project_root):
        _guard_fail(drive_root_message(project_root))

    _validate_configured_root_alignment(
        configured_project_root=configured_project_root,
        derived_project_root=project_root,
    )

    return project_root


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
