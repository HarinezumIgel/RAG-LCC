"""
Cursor-displacement regression test.

Interactive manual test
-----------------------
Run from the project root in a **fresh** VS Code terminal:

    python tests/test_cursor.py prompt_toolkit   # should work correctly
    python tests/test_cursor.py pyreadline3       # will show cursor displacement
    python tests/test_cursor.py bare              # baseline without any readline

Type text at each prompt and verify the cursor stays aligned.

Automated pytest checks
-----------------------
``pytest tests/test_cursor.py`` runs the non-interactive assertions below.
"""

import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Non-interactive pytest tests
# ---------------------------------------------------------------------------


def test_prompt_toolkit_session_creation() -> None:
    """prompt_toolkit PromptSession can be instantiated (no pyreadline3 side-effects)."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.input import DummyInput

    session: PromptSession[str] = PromptSession(
        input=DummyInput(), output=DummyOutput()
    )
    # Verify ANSI prompt wrapping works without error
    _ = ANSI("\033[35m 🛠️  >\033[0m")
    assert session is not None


def test_pyreadline3_not_in_history_manager() -> None:
    """HistoryManager must not reference pyreadline3 (causes cursor displacement in VS Code)."""
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    hist_path = os.path.join(src, "Gui", "HistoryManager.py")
    content = open(hist_path, encoding="utf-8").read()
    assert (
        "pyreadline3" not in content
    ), "HistoryManager should no longer import pyreadline3"


def test_history_manager_uses_prompt_toolkit() -> None:
    """HistoryManager should import prompt_toolkit, not pyreadline3."""
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    hist_path = os.path.join(src, "Gui", "HistoryManager.py")
    content = open(hist_path, encoding="utf-8").read()
    assert "prompt_toolkit" in content, "HistoryManager should use prompt_toolkit"
    assert (
        "pyreadline3" not in content
    ), "HistoryManager should no longer import pyreadline3"


# ---------------------------------------------------------------------------
# Interactive manual mode (run directly: python tests/test_cursor.py <mode>)
# ---------------------------------------------------------------------------


def _interactive() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "prompt_toolkit"

    if mode == "bare":
        print("=== Running with bare input() (no readline) ===\n")

        print("--- Test 1: ASCII ---")
        val = input(" settings >")
        print(f"  got: {val!r}\n")

        print("--- Test 2: emoji ---")
        val = input(" 🛠️  >")
        print(f"  got: {val!r}\n")

        print("--- Test 3: ANSI color + emoji ---")
        val = input("\033[35m 🛠️  >\033[0m")
        print(f"  got: {val!r}\n")

    elif mode == "pyreadline3":
        print("=== Running with pyreadline3 ===\n")
        try:
            import readline
        except ImportError:
            import pyreadline3 as readline  # type: ignore

        print("--- Test 1: ASCII ---")
        val = input(" settings >")
        print(f"  got: {val!r}\n")

        print("--- Test 2: emoji ---")
        val = input(" 🛠️  >")
        print(f"  got: {val!r}\n")

        print("--- Test 3: ANSI color + emoji ---")
        val = input("\033[35m 🛠️  >\033[0m")
        print(f"  got: {val!r}\n")

    else:
        print("=== Running with prompt_toolkit ===\n")
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import ANSI

        session: PromptSession[str] = PromptSession()

        print("--- Test 1: ASCII ---")
        val = session.prompt(" settings >")
        print(f"  got: {val!r}\n")

        print("--- Test 2: emoji ---")
        val = session.prompt(" 🛠️  >")
        print(f"  got: {val!r}\n")

        print("--- Test 3: ANSI color + emoji ---")
        val = session.prompt(ANSI("\033[35m 🛠️  >\033[0m"))
        print(f"  got: {val!r}\n")

    print("--- Done ---")


if __name__ == "__main__":
    _interactive()
