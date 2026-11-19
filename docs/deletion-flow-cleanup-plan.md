# Deletion Flow Cleanup Plan

## Status and scope

This is a proposed plan only. No deletion code has been changed or tested.
Downloader flows are explicitly out of scope.

## Required safety invariants

- Preserve all checks in [`FileUtils.is_safe_delete_path`](../src/Helpers/FileUtils.py):
  jailbreak avoidance and project-root containment are intentional requirements.
- The retrieval-store configuration requires the Chroma, BM25, Graph, and Regex
  store paths to remain inside the project root
  ([Config_Load_Retrievers.py](../src/Configuration/Config_Load_Retrievers.py)).
- [`FileUtils.delete_file_or_dir`](../src/Helpers/FileUtils.py) invokes
  `is_safe_delete_path` before checking whether the target exists. A missing
  safe target is treated as success; an unsafe target must still be rejected.
- Callers should invoke `delete_file_or_dir` directly. Calling
  `is_safe_delete_path` first would run the same guard twice.
- Any direct filesystem deletion must retain the shared safety guard
  immediately before deleting. Never fall back to an unguarded deletion.
- Treat root-path and symlink/junction behavior as separate hardening concerns;
  they are not reasons to weaken or bypass current checks.

## Proposed changes

1. **Remove redundant caller-side existence checks for retrieval-store
   directories.** In [ChromaDBHelper.py](../src/Helpers/ChromaDBHelper.py) and
   [Informer.py](../src/Gui/Informer.py), call `delete_file_or_dir` without an
   outer `os.path.exists` check. Keep handling its boolean result. Adjust
   messages that say “Deleted” because the helper also succeeds for an absent
   path.

2. **Preserve marked-document cleanup protections.** The cleanup in
   [MarkedDocsViewer.py](../src/Chat/MarkedDocsViewer.py) calls `shutil.rmtree`
   directly, so keep its explicit shared guard and service-root/direct-child
   restrictions. Preserve its missing-directory error behavior.

3. **Keep Office-conversion intermediates inside project-owned storage.**
   [OfficeDocConverter.py](../src/Helpers/OfficeDocConverter.py) directly
   deletes generated temporary outputs, some currently placed in the system
   temporary directory. Move these generated intermediates to a dedicated
   project-owned directory, then use `delete_file_or_dir` or the shared guard
   immediately before the file-only deletion. Preserve the original input
   documents; do not make an external-temp exception to project containment.

4. **Guard Exclusions temporary-file cleanup.** The `.tmp` file handled by
   [Exclusions.py](../src/Compliance/Exclusions.py) is managed application
   storage. Prefer `delete_file_or_dir` without a caller-side existence check,
   or retain direct deletion with the shared guard immediately beforehand.
   Choose one approach, not both, and preserve atomic-write and error behavior.

5. **Review other non-downloader direct deletion scripts.** The output roots in
   `deploy/scripts/sign_files.py`, `CreateClassGraphs.py`, and
   `show_3rd_party_licenses.py` are caller-selected. Decide their containment
   contract before adding application-configured guards; do not silently
   bypass the project-root requirement.

6. **Keep logical record removals distinct from filesystem deletion.**
   Chroma `collection.delete` and the retrievers’ `remove_by_filepath` methods
   remove vector/index records, not source files or index directories. Do not
   replace those operations with `delete_file_or_dir` or add filesystem
   existence checks. Preserve the retrievers’ current guards in this cleanup.

## Focused validation plan

- Keep the existing path-guard tests and ensure an unsafe, nonexistent target
  is rejected before any existence-based success path.
- Test that ChromaDBHelper and Informer delegate absent targets to the guarded
  helper, preserve rejection handling, and avoid success messages on rejection.
- Preserve marked-document tests for guard rejection, ownership boundaries,
  missing paths, and process-exit cleanup.
- Test Office cleanup stays within project-owned storage, refuses rejected
  targets, and never deletes the original document.
- Test Exclusions temporary-file cleanup on failure and ensure rejected paths
  remain untouched.

## Separate hardening follow-up

The current guard appears to allow the project root itself, while
`delete_file_or_dir` can recursively remove a directory. The path normalization
is lexical and does not resolve symlinks or junctions. Review those behaviors
as separate safety issues before making deletion-flow changes; do not weaken
the current containment checks.
