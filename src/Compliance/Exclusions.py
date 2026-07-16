import os
from typing import List, Optional, Set

from Commons.Exceptions import ExclusionsError
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class Exclusions(SingletonMixin):

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.globals: Globals = Globals()
        self.pretty: PrettyWriter = pretty or PrettyWriter(always_on=True)
        self.helpers: Helpers = helpers or Helpers()

        self.collection: str = self.cfg.get_str("COLLECTION", "")

        # lines preserves file order (comments and entries). Each element is the original line string.
        self.lines: List[str] = []
        # seen set stores normalized paths that have already been kept (first-seen)
        self.seen: Set[str] = set()

        # header date used for filename and for appended headers
        self.header_date = self.globals.get_date()

        # counter: how many times contains() returned True
        self.applied_counter: int = 0

        self.filepath: str = self._build_filepath()
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # load file into _lines and _seen, dropping later duplicates (keep first-seen)
        self._load_file()

    # -------------------------
    # Public API
    # -------------------------
    def add(self, path: str) -> bool:
        """
        Append path to the file (in-memory and persisted) only if not already present.
        Returns True if added, False if already present or invalid.
        """
        norm = self._normalize(path)
        if not norm:
            return False

        if norm in self.seen:
            return False
        # append original normalized representation (use normalized form for storage)
        line = norm
        self.lines.append(line)
        self.seen.add(norm)
        self._flush_to_disk()
        self.pretty.write(
            "W",
            "Exclusions",
            f"File {path} is added to Exclusions file {self.filepath}",
        )
        return True

    def remove(self, path: str) -> bool:
        """
        Remove all occurrences of the path from the file and persist.
        Returns True if any removal happened, False otherwise.
        """
        norm = self._normalize(path)
        if not norm:
            return False

        if norm not in self.seen:
            return False

        # Rebuild lines preserving comments and other entries, dropping any line that normalizes to norm
        new_lines: List[str] = []
        new_seen: Set[str] = set()
        removed = False
        for ln in self.lines:
            if ln.startswith("#"):
                new_lines.append(ln)
                continue
            ln_norm = self._normalize(ln)
            if ln_norm == norm:
                removed = True
                continue
            # keep only first-seen occurrences of other paths
            if ln_norm and ln_norm not in new_seen:
                new_seen.add(ln_norm)
                new_lines.append(ln)
        if removed:
            self.lines = new_lines
            self.seen = new_seen
            self._flush_to_disk()
        return removed

    def contains(self, path: str) -> bool:
        """
        Check whether the normalized path is present (first-seen).
        Increments an internal counter each time this returns True.
        """
        norm: str = self._normalize(path)
        if not norm:
            return False

        found: bool = norm in self.seen
        if found:
            # increment applied counter for each successful contains check
            self.applied_counter += 1
        return found

    # -------------------------
    # New helpers requested
    # -------------------------
    def get_exclusions_filename(self) -> str:
        """
        Return the full path to the exclusions file (as used by this instance).
        """
        return self.filepath

    def get_applied_exclusions(self) -> int:
        """
        Return how many times contains() returned True since this instance was created.
        """
        return int(self.applied_counter)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _build_filepath(self) -> str:
        """
        Construct filename like: <FRIENDLY_NAM>_HUMAN_REVIEW_YYYYMMDD_HHMMSS.csv
        Directory from config.get("_EXCLUSIONS_DIR") or cwd.
        """
        friendly: str = self.cfg.get_str("_FRIENDLY_NAME")
        doc_dir: str = self.cfg.get_str("DOC_DIR")
        doc_dir_hash: str = self.helpers.short_hash(doc_dir, length=8)
        if self.collection != "":
            filename = f"{friendly}_{doc_dir_hash}_{self.collection}_EXCLUSIONS.csv"
        else:
            filename = f"{friendly}_{doc_dir_hash}_EXCLUSIONS.csv"
        try:
            dirpath = self.cfg.get_str("_EXCLUSIONS_DIR")
        except Exception:
            # If config access fails, write a message and exit
            msg = "Exclusions directory does not exist or cannot be read. Please adjust in config or create the directory."
            self.pretty.write("E", "Exclusions", msg, color=RED)
            raise ExclusionsError(msg)

        return os.path.join(dirpath, filename)

    def _normalize(self, path: Optional[str]) -> str:
        """
        Normalize path for comparison and storage:
        - convert backslashes to forward slashes
        - collapse multiple slashes
        - os.path.normpath to collapse '.' and '..'
        - convert to forward slashes and apply os.path.normcase
        Returns empty string for invalid input.
        """
        if not path:
            return ""
        p = str(path).strip()
        if not p:
            return ""
        p = p.replace("\\", "/")
        while "//" in p:
            p = p.replace("//", "/")
        os_sep_path = p.replace("/", os.sep)
        normed = os.path.normpath(os_sep_path)
        normed = normed.replace(os.sep, "/")
        normed = os.path.normcase(normed)
        return normed

    def _load_file(self) -> None:
        """
        Load the single file into memory, preserving header/comment lines and the first-seen
        occurrence of each path. Later duplicates are dropped (so the in-memory representation
        contains only the first occurrence of each path).
        If file does not exist, create it with an initial header line.
        """
        self.lines.clear()
        self.seen.clear()

        if not os.path.exists(self.filepath):
            # create initial header and persist
            header_line = f"# Date: {self.header_date}"
            self.lines.append(header_line)
            self._flush_to_disk()
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith("#"):
                        # keep header/comment lines as-is
                        self.lines.append(line)
                        # try to parse header date if present (optional)
                        parts = line.lstrip("#").strip().split()
                        if "Date:" in parts:
                            try:
                                idx = parts.index("Date:")
                                date_val = parts[idx + 1]
                                self.header_date = date_val
                            except Exception:
                                pass
                        continue
                    # non-comment line: treat as path
                    norm = self._normalize(line)
                    if not norm:
                        continue
                    if norm in self.seen:
                        # duplicate later occurrence -> skip (do not append to _lines)
                        continue
                    # keep first-seen occurrence
                    self.seen.add(norm)
                    # store the normalized form as the canonical stored line
                    self.lines.append(norm)
        except Exception:
            # on read error, initialize file
            header_line = f"# Date: {self.header_date}"
            self.lines = [header_line]
            self.seen.clear()
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """
        Atomically write the current _lines to disk.
        """
        tmp_path: str = self.filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for ln in self.lines:
                    f.write(ln + "\n")
            os.replace(tmp_path, self.filepath)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
