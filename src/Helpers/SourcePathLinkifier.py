from __future__ import annotations

import os
import re
from pathlib import Path


class SourcePathLinkifier:
    """Convert FilePath entries in markdown text into clickable links."""

    _FILEPATH_LINE_RE = re.compile(r"(^\s*[-*]?\s*FilePath:\s*)(.+)$", re.MULTILINE)
    _MD_LINK_RE = re.compile(r"^\[(?P<label>.+?)\]\((?P<url>[^)]+)\)$")

    # Matches inline markdown links in body text (not image links preceded by !).
    _INLINE_BODY_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
    # Detects a local filesystem path embedded anywhere in a URL:
    #   - Windows drive letter  e.g. "D:/" or "D:\"  (negative lookbehind ensures
    #     "p://" in "http://" and "s://" in "https://" are NOT matched)
    #   - file:// scheme
    _LOCAL_PATH_IN_URL_RE = re.compile(r"(?i)file://|(?<![A-Za-z])[A-Za-z]:[/\\]")

    @staticmethod
    def _to_uri(path_text: str, *, allow_local_file_uri: bool) -> str | None:
        """Convert a path string to a URI, respecting the allow_local_file_uri flag.

        Args:
            path_text: The path string to convert.
            allow_local_file_uri: Whether to allow local file:// URIs.

        Returns:
            A URI string, or None if the path cannot be converted or is not allowed.
        """
        raw = path_text.strip().strip("`\"'")
        if not raw:
            return None

        lowered = raw.lower()
        if lowered.startswith(("http://", "https://")):
            return raw
        if lowered.startswith("file://"):
            return raw if allow_local_file_uri else None

        if not allow_local_file_uri:
            return None

        try:
            normalized = os.path.normpath(raw)
            path_obj = Path(normalized)
            if not path_obj.is_absolute():
                return None
            return path_obj.resolve(strict=False).as_uri()
        except Exception:
            return None

    @staticmethod
    def linkify_source_paths_md(
        text: str,
        *,
        allow_local_file_uri: bool,
        strip_local_open_link_tail: bool = False,
    ) -> str:
        """Normalize/convert FilePath lines into clickable markdown links.

        - CLI mode: allow local ``file://`` links.
        - API mode: disallow local ``file://`` links; keep only HTTP(S) links.

        Args:
            text: The markdown text to process.
            allow_local_file_uri: Whether to allow local file:// URIs.
            strip_local_open_link_tail: Whether to strip existing [open](file://...) suffixes.

        Returns:
            The processed markdown text with linkified FilePath entries.
        """
        if not text or "FilePath:" not in text:
            return text

        def _replace(match: re.Match[str]) -> str:
            prefix = match.group(1)
            value = match.group(2).strip()

            if strip_local_open_link_tail:
                value = re.sub(
                    r"\s*\(\s*\[open\]\(\s*file://[^)]+\)\s*\)\s*$",
                    "",
                    value,
                    flags=re.IGNORECASE,
                ).strip()

            md_match = SourcePathLinkifier._MD_LINK_RE.match(value)
            if md_match:
                url = md_match.group("url").strip()
                if url.lower().startswith(("http://", "https://")):
                    return f"{prefix}{value}"
                if allow_local_file_uri and url.lower().startswith("file://"):
                    return f"{prefix}{value}"
                # Drop non-permitted markdown link and keep plain label.
                return f"{prefix}{md_match.group('label')}"

            uri = SourcePathLinkifier._to_uri(
                value, allow_local_file_uri=allow_local_file_uri
            )
            if not uri:
                return f"{prefix}{value}"
            return f"{prefix}{value} ([open]({uri}))"

        return SourcePathLinkifier._FILEPATH_LINE_RE.sub(_replace, text)

    @staticmethod
    def strip_inline_file_citation_links(
        text: str,
        *,
        allowed_url_prefixes: tuple[str, ...] = (),
    ) -> str:
        """Reduce inline markdown links that wrap a local file path to plain label text.

        The LLM sometimes embeds local FilePath values (e.g. ``D:/path/to/file.pdf``)
        into external file-serving URLs (e.g. ``filepicker.io``).  This function strips
        such links, leaving only the visible label — the safest representation in
        API / service mode where local paths must not be exposed as clickable links.

        Links whose URL starts with any entry in *allowed_url_prefixes* are always kept
        intact regardless of content — pass the in-memory document server base URL here
        so ``/marked/<token>`` links are never stripped.

        Only links whose URL contains a Windows drive-letter path (``D:/``) or a
        ``file://`` scheme are affected after the whitelist check.  Pure HTTP/HTTPS
        web links are kept intact.  Image links (``![...](url)``) are never touched.

        Args:
            text: Markdown text, typically the assembled LLM answer body.
            allowed_url_prefixes: URL prefixes that must never be stripped.

        Returns:
            Text with hallucinated file-citation links reduced to plain labels.
        """
        if not text:
            return text

        def _replace(m: re.Match[str]) -> str:
            label = m.group(1)
            url = m.group(2).strip()
            if allowed_url_prefixes and any(
                url.startswith(p) for p in allowed_url_prefixes if p
            ):
                return m.group(0)
            if SourcePathLinkifier._LOCAL_PATH_IN_URL_RE.search(url):
                return label
            return m.group(0)

        return SourcePathLinkifier._INLINE_BODY_LINK_RE.sub(_replace, text)
