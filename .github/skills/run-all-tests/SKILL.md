---
name: run-all-tests
description: "Run the complete RAG-LCC test suite. Use when asked to run all tests, full suite, or release validation. Executes tests/RunTests.py with the project virtual environment and reports totals."
argument-hint: "[optional pytest args passed through RunTests.py]"
user-invocable: true
---

# Run All Tests (RAG-LCC)

## When to Use
- User asks to run all tests or full test suite.
- Pre-release validation is requested.
- Need a complete pass/fail baseline before deploy or commit.

## Procedure
1. Read the newest section in `CHANGELOG.md` first to understand what changed.
2. Open the repository root.
3. Run the canonical full-suite command with the repo virtual environment.

Windows (PowerShell):

```powershell
Set-Location D:\RAG-LCC
.\.venv\Scripts\python.exe .\tests\RunTests.py
```

Linux/macOS (bash):

```bash
cd /workspaces/RAG-LCC
./.venv/bin/python ./tests/RunTests.py
```

4. For verbose or filtered execution, pass arguments through to pytest via `RunTests.py`.

```powershell
.\.venv\Scripts\python.exe .\tests\RunTests.py -v
.\.venv\Scripts\python.exe .\tests\RunTests.py -k rewrite
```

5. Fallback only if needed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --tb=short
```

## Reporting Requirements
- Report passed, failed, and skipped counts.
- Report runtime.
- If failures exist, include failing test names and first traceback block.
- Mention the changelog date/version reviewed before the run.

## Changelog-Aware Baseline
- Recent full-suite baseline recorded in release docs:
	- `2026-09-25`: `1540 passed in 44.37s`
- Current workspace baseline (latest full run with this skill command):
	- `1559 passed in 45.64s`
- If totals differ from baseline, report the delta and call out that test count drift can be expected after new tests are added.

## Notes
- Always use the repository virtual-environment interpreter.
- Avoid root-level `pytest -q` without the explicit `tests` target.
