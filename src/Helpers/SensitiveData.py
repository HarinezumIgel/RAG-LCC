import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(
    r"(\b(?:proxy-authorization|authorization)\b[\"']?\s*[:=]\s*)"
    r"(?:(Bearer|Basic)\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|"
    r"secret|password|passwd|client[-_ ]?secret|credential)\b"
    r"[\"']?(?:\s*[:=]\s*|\s+))"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_KEY_RE = re.compile(
    r"(?:api[-_]?key|access[-_]?token|refresh[-_]?token|token|secret|"
    r"password|passwd|auth|authorization|credential|signature|client[-_]?secret|^key$)",
    re.IGNORECASE,
)


def _mask_url(match: re.Match[str]) -> str:
    raw_url = match.group(0)
    suffix = ""
    while raw_url and raw_url[-1] in ".,);]}":
        suffix = raw_url[-1] + suffix
        raw_url = raw_url[:-1]

    try:
        parsed = urlsplit(raw_url)
        netloc = parsed.netloc
        has_userinfo = "@" in netloc
        if has_userinfo:
            netloc = "***@" + netloc.rsplit("@", 1)[1]

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        fragment_pairs = parse_qsl(parsed.fragment, keep_blank_values=True)
        if not has_userinfo and not any(
            _SENSITIVE_QUERY_KEY_RE.search(key)
            for key, _value in (*query_pairs, *fragment_pairs)
        ):
            return raw_url + suffix

        query = [
            (key, "******" if _SENSITIVE_QUERY_KEY_RE.search(key) else value)
            for key, value in query_pairs
        ]
        fragment = [
            (key, "******" if _SENSITIVE_QUERY_KEY_RE.search(key) else value)
            for key, value in fragment_pairs
        ]
        safe_url = urlunsplit(
            (parsed.scheme, netloc, parsed.path, urlencode(query), urlencode(fragment))
        )
    except ValueError:
        return "[URL masked]" + suffix

    return safe_url + suffix


def mask_sensitive_data(value: object) -> str:
    """Mask credential-like values in text before it is displayed or logged."""
    text = str(value)
    text = _URL_RE.sub(_mask_url, text)

    def mask_auth(match: re.Match[str]) -> str:
        scheme = match.group(2)
        return f"{match.group(1)}{scheme + ' ' if scheme else ''}******"

    text = _AUTH_HEADER_RE.sub(mask_auth, text)

    def mask_assignment(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quoted = match.group(0)[len(prefix) :]
        quote = quoted[0] if quoted.startswith(("'", '"')) else ""
        return f"{prefix}{quote}******{quote}"

    # Masking applies to displayed copies; callers keep original values for use.
    return _SECRET_ASSIGNMENT_RE.sub(mask_assignment, text)
