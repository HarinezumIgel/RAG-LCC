"""Select the document for ``summarize:`` when the user did not name one.

Resolution order:

1. ``summarize: cancel`` stops without summarizing.
2. An explicit ``summarize: <file name or path>`` target.
3. The pinned ``file``/``path`` selection.
4. RAGChat opens the existing path picker when no document is selected.
   RAGChatService uses source documents of earlier answers; a single candidate
   is used automatically, several candidates require a choice.

RAGChat uses the same picker as ``path!`` in session settings.
RAGChatService replies with a numbered list because OpenWebUI cannot show a
picker. The service keeps no server-side state; the list is read back from the
conversation history that OpenWebUI sends with every request.
"""

from __future__ import annotations

import ntpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Sequence

from Globals.Session import Session

if TYPE_CHECKING:
    from Chat.RAGChatImpl import RAGChatImpl

SelectionStatus = Literal[
    "explicit", "pinned", "auto", "picked", "cancelled", "choose", "none"
]

_CANCEL_WORDS: frozenset[str] = frozenset({"cancel", "c", "0"})
_CHAT_CONTEXT = "Chat Context"
SELECTION_HEADER = "**Select a document to summarize**"
CONFIRMATION_PREFIX = "Summarizing document:"
CANCELLED_MESSAGE = "Summarization cancelled. Nothing was summarized."

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKED_RE = re.compile(r"^\s*-\s*🟢\s*\[(?P<name>.+?) \(highlighted\)\]\(")
_FILEPATH_RE = re.compile(r"^\s*-\s*FilePath:\s*(?P<path>.+?)\s*$")
_CONFIRMATION_RE = re.compile(rf"^{re.escape(CONFIRMATION_PREFIX)}\s*`(?P<path>[^`]+)`")
_CHOICE_RE = re.compile(r"^\s*(?P<number>\d+)\.\s+(?:👉\s+)?`(?P<path>[^`]+)`\s*$")


@dataclass(frozen=True)
class FileSelection:
    """Outcome of a selection.

    ``target`` is set for ``explicit``, ``auto`` and ``picked``; ``pinned``
    leaves it empty so the summary keeps using the session pin. ``message``
    holds the user-facing text for ``cancelled`` and ``choose``.
    """

    status: SelectionStatus
    target: str = ""
    candidates: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""

    @property
    def stops_summary(self) -> bool:
        return self.status in ("cancelled", "choose")

    def summary_query(self, fallback: str) -> str:
        """Return the summarize command to run for this selection."""
        if self.status in ("auto", "picked") and self.target:
            return f"summarize: {self.target}"
        return fallback


class FileSelector:
    """Shared resolution; subclasses provide candidates and the choice UI."""

    @staticmethod
    def is_cancel(text: str) -> bool:
        return text.strip().lower() in _CANCEL_WORDS

    @staticmethod
    def has_pin(session: Session) -> bool:
        pins = (session.file_name, session.file_path)
        return any(pins) and _CHAT_CONTEXT not in pins

    @staticmethod
    def path_identity(path: str) -> str:
        normalized = ntpath.normpath(path)
        # Windows drive and UNC paths are case-insensitive; preserve POSIX case.
        return ntpath.normcase(normalized) if ntpath.splitdrive(path)[0] else normalized

    @classmethod
    def unique_paths(cls, paths: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            path = path.strip()
            if not path or path == _CHAT_CONTEXT:
                continue
            identity = cls.path_identity(path)
            if identity not in seen:
                seen.add(identity)
                unique.append(path)
        return unique

    @staticmethod
    def confirmation(document: str) -> str:
        """Notice shown before every summary, whatever selected the document."""
        return f"{CONFIRMATION_PREFIX} `{document}`\n\n"

    def select(self, session: Session, target: str) -> FileSelection:
        target = target.strip()
        if self.is_cancel(target):
            return FileSelection("cancelled", message=CANCELLED_MESSAGE)
        if target:
            return FileSelection("explicit", target=target)
        if self.has_pin(session):
            return FileSelection("pinned")
        candidates = tuple(self.unique_paths(self.candidates(session)))
        if not candidates:
            # The summary pipeline reports how to select a document.
            return FileSelection("none")
        if len(candidates) == 1:
            return FileSelection("auto", target=candidates[0], candidates=candidates)
        return self.choose(session, candidates)

    def candidates(self, session: Session) -> list[str]:
        raise NotImplementedError

    def choose(self, session: Session, candidates: tuple[str, ...]) -> FileSelection:
        raise NotImplementedError


class CliFileSelector(FileSelector):
    """RAGChat: open path! directly when no target or pin is available."""

    def __init__(self, path_picker: Callable[[Session], bool]) -> None:
        self.path_picker = path_picker

    def select(self, session: Session, target: str) -> FileSelection:
        selection = super().select(session, target)
        if selection.status != "none":
            return selection
        if self.path_picker(session) and self.has_pin(session):
            return FileSelection("pinned")
        return FileSelection("cancelled", message=CANCELLED_MESSAGE)

    def candidates(self, session: Session) -> list[str]:
        return []


class ServiceFileSelector(FileSelector):
    """RAGChatService: candidates come from this conversation's history.

    Earlier answers list their grounded sources (file names, sometimes paths);
    names are resolved to indexed paths so paths are always preferred.
    """

    def __init__(self, rag: "RAGChatImpl", messages: Sequence[tuple[str, str]]) -> None:
        self.rag = rag
        self.messages = list(messages)

    def _history(self) -> list[tuple[str, str]]:
        """Messages before the current (last) user message."""
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index][0] == "user":
                return self.messages[:index]
        return list(self.messages)

    def _last_assistant_text(self) -> str:
        for role, text in reversed(self._history()):
            if role == "assistant":
                return text
        return ""

    def resolve_reply(self, user_text: str) -> FileSelection | None:
        """Handle an answer to a selection list; ``None`` if it is not one.

        Only the reply directly after the list counts, so any other message
        leaves the question and continues the chat normally.
        """
        previous = self._last_assistant_text()
        if SELECTION_HEADER not in previous:
            return None
        reply = user_text.strip()
        if self.is_cancel(reply):
            return FileSelection("cancelled", message=CANCELLED_MESSAGE)
        if not reply.isdigit():
            return None
        choices = {
            int(match.group("number")): match.group("path")
            for match in map(_CHOICE_RE.match, previous.splitlines())
            if match
        }
        candidates = tuple(choices[number] for number in sorted(choices))
        picked = choices.get(int(reply))
        if picked is None:
            return FileSelection(
                "choose",
                candidates=candidates,
                message=f"{reply} is not in the list.\n\n"
                + self.format_choices(candidates),
            )
        return FileSelection("picked", target=picked, candidates=candidates)

    def candidates(self, session: Session) -> list[str]:
        references = self.extract_document_references(self._last_assistant_text())
        return self._resolve_indexed(session, references) if references else []

    def _resolve_indexed(self, session: Session, references: list[str]) -> list[str]:
        names = list(dict.fromkeys(ntpath.basename(ref) for ref in references))
        metadata = self.rag.indexed_file_metadata(session, names)
        indexed: dict[str, list[str]] = {}
        for item in metadata:
            name = str(item.get("FileName", "")).strip()
            path = str(item.get("FilePath", "")).strip()
            if name and path:
                indexed.setdefault(name, []).append(path)
        resolved: list[str] = []
        for ref in references:
            paths = indexed.get(ntpath.basename(ref), [])
            if ntpath.dirname(ref):
                wanted = self.path_identity(ref)
                paths = [path for path in paths if self.path_identity(path) == wanted]
            resolved.extend(paths)
        return resolved

    def choose(self, session: Session, candidates: tuple[str, ...]) -> FileSelection:
        return FileSelection(
            "choose", candidates=candidates, message=self.format_choices(candidates)
        )

    @staticmethod
    def format_choices(candidates: Sequence[str]) -> str:
        lines = [
            f"{SELECTION_HEADER} — no document is pinned and earlier answers "
            "used several documents:",
            "",
        ]
        lines += [f"{number}. 👉 `{path}`" for number, path in enumerate(candidates, 1)]
        lines += [
            "",
            "Reply with the number to summarize that document, or `cancel` "
            "(or `0`) to skip summarizing. Any other message continues the chat "
            "without summarizing.",
        ]
        return "\n".join(lines)

    @staticmethod
    def extract_document_references(text: str) -> list[str]:
        """Return file names/paths listed in an answer's source blocks."""
        references: list[str] = []
        in_sources = False
        for raw_line in text.splitlines():
            line = _MD_LINK_RE.sub(r"\1", raw_line).replace("`", "").rstrip()
            if "📄 **Sources**" in line:
                in_sources = True
                continue
            if in_sources:
                if line.lstrip().startswith("- "):
                    references.append(line.lstrip()[2:].strip())
                    continue
                in_sources = False
            for pattern, source in (
                (_MARKED_RE, raw_line),
                (_FILEPATH_RE, line),
                (_CONFIRMATION_RE, raw_line),
            ):
                match = pattern.match(source)
                if match:
                    references.append(match.group(1).strip())
                    break
        return [
            ref for ref in dict.fromkeys(references) if ref and ref != _CHAT_CONTEXT
        ]
