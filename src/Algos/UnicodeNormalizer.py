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

    def normalize(self, text: str) -> str:
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
        out = self._normalize_whitespace(out)
        return out

    def _normalize_whitespace(self, text: str) -> str:
        """
        Replace all whitespace sequences with a single space and trim edges.
        """
        return re.sub(r"\s+", " ", text).strip()
