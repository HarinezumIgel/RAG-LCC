#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RunTests.py - Test suite runner for RAG-LCC

Runs all tests in the tests/ directory using pytest.
Usage:
    python tests/RunTests.py              # Run all tests
    python tests/RunTests.py -v           # Verbose output
    python tests/RunTests.py -k pattern   # Run tests matching pattern
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run the test suite with pytest."""
    # Get the project root (parent of tests/)
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent

    # Build pytest command
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "--tb=short",
    ]

    # Forward any command-line arguments to pytest
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])
    else:
        # Default: quiet mode with summary
        pytest_args.append("-q")

    print(f"Running tests from: {tests_dir}")
    print(f"Command: {' '.join(pytest_args)}")
    print("-" * 70)

    # Run pytest
    try:
        result = subprocess.run(
            pytest_args,
            cwd=str(project_root),
            check=False,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTest run cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError running tests: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
