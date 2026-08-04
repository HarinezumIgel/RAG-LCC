"""Shared sentence-splitting logic for chunker strategies.

Handles clean prose (.!? boundaries) as well as PDF-extracted text with
bullet markers and paragraph breaks.  Deliberately conservative: avoids
splitting on semicolons, colons, single newlines (often just PDF line
wraps) and double-space runs so that chunks retain enough semantic
context for retrieval.

Special care for PDF-extracted numbered lists:  ``6. Some text`` is NOT
split between ``6.`` and ``Some text`` — the look-behind requires at
least two word characters before the period to avoid orphan-number
chunks.
"""

import re


class SentenceSplitter:
    """Conservative sentence splitter for RAG chunk preparation.

    Splits on real sentence boundaries (.!?), bullet markers, paragraph
    breaks, and newlines before uppercase/digit.  Preserves semicolons,
    colons, double-spaces, PDF line-wraps, and numbered-list labels to
    keep chunks coherent.
    """

    # ── Boundary patterns (order matters — first match wins) ──────────
    #
    # 1. Sentence ending after ≥2 word-chars:  word.  word!  word?
    #    The 2-char look-behind avoids splitting on numbered list
    #    labels like "6." or "10." which are common in PDF text.
    # 2. Bullet / list markers at start of line:  •  -  *  ●  ◦  ▪
    # 3. Two or more consecutive newlines (paragraph break)
    # 4. Single newline before uppercase letter (new sentence)
    #
    # NOT split (preserves semantic coherence in chunks):
    #   - Semicolons  (keeps related clauses together)
    #   - Colons      (keeps label: value pairs intact)
    #   - Double spaces / tabs  (keeps PDF table rows intact)
    #   - Single newline before lowercase/digit (PDF line-wrap / list continuation)
    #   - Period after 1 digit  (list label like "6." or "10.")

    _SENTENCE_RE = re.compile(
        r"(?<=\w{2}[.!?])\s+"  # .!? after ≥2 word chars + whitespace
        r"|(?:^|\n)\s*[•\-\*●◦▪]\s+"  # bullet markers at line start
        r"|\n{2,}"  # paragraph breaks (2+ newlines)
        r"|\n(?=[A-Z])"  # newline before uppercase only
    )

    def split_sentences(self, text: str) -> list[str]:
        """Split *text* into sentence-like segments.

        Works for both clean prose and messy PDF-extracted text.
        Returns a list of non-empty, stripped strings.
        """
        parts: list[str] = self._SENTENCE_RE.split(text)
        return [s.strip() for s in parts if s.strip()]
