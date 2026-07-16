import re
import unicodedata

from Gui.PrettyWriter import PrettyWriter


class UnicodeNormalizer:
    """
    Utility providing robust Unicode normalization for compliance and
    text-processing pipelines. Ensures that text is converted into a
    canonical, security-safe representation before further analysis.

    Features:
    - NFKC normalization
    - Case-folding
    - Whitespace normalization
    """

    def __init__(self) -> None:
        self.pretty: PrettyWriter = PrettyWriter()

    def normalize(self, text: str, *, preserve_newlines: bool = False) -> str:
        """
        Apply the full normalization pipeline:
        1. Unicode NFKC normalization
        2. Case-folding
        3. Whitespace normalization
        """
        if not text:
            return ""

        self.pretty.write(
            "I",
            "UnicodeNormalizer",
            "NFKC: Compatibility Decomposition + Canonical Composition",
        )
        out: str = unicodedata.normalize("NFKC", text)
        out = out.casefold()
        out = self._normalize_whitespace(out, preserve_newlines=preserve_newlines)
        return out

    @staticmethod
    def _normalize_whitespace(text: str, *, preserve_newlines: bool = False) -> str:
        """
        Replace whitespace sequences with a single space and trim edges.

        When *preserve_newlines* is ``True``, newlines are kept so that
        structure-aware chunkers (e.g. HeadingChunker) can still detect
        line-based markers such as Markdown ``#`` headings.
        """
        if preserve_newlines:
            # Normalize whitespace within each line, keep line breaks
            lines = text.split("\n")
            lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in lines]
            return "\n".join(lines).strip()
        return re.sub(r"\s+", " ", text).strip()
