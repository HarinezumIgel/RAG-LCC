# Completion and verification

- Inspect the existing implementation and trace the actual user entry point before editing. Scaffolding, imports, and help text do not constitute a working feature.
- Wire the requested behavior end to end. Reuse existing helpers and verify that unrelated behavior is preserved.
- Validate the exact requirements with focused tests, including failure paths. Mocked tests do not establish that a live model or external service works.
- For Python changes, run Pyright and inspect available VS Code diagnostics. Compilation alone does not check unused imports or type errors.
- Do not claim completion based on attempted edits or successful tool execution. Confirm the changes were applied and the requested behavior was verified.
- Report what changed, what validation passed or failed, and what remains incomplete or untested. Distinguish pre-existing diagnostics from new ones; never describe a failing check as clean.
