"""Render grounded Markdown answers as self-contained browser pages."""

from __future__ import annotations

import base64
import re
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

from markdown_it import MarkdownIt

_FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*$")
_MERMAID_FENCE = re.compile(r"(?im)^[ \t]{0,3}(`{3,}|~{3,})[ \t]*mermaid(?:[ \t].*)?$")
_FLOWCHART_START = re.compile(r"^\s*(?:flowchart|graph)[ \t]+(?:TB|TD|BT|RL|LR)\b")
_FLOWCHART_LABEL = re.compile(
    r'"[^"\n]*"|\|[^|\n]*\||%%[^\n]*'
    r"|(?P<node>\b[A-Za-z_][A-Za-z0-9_-]*[ \t]*)"
    r'\[(?P<label>[^\[\]"\n]*[()][^\[\]"\n]*)\]'
)
_FAVICON_PATH = (
    Path(__file__).resolve().parents[1] / "Apps" / "resources" / "favicon.png"
)
_FAVICON_DATA_URI = (
    "data:image/png;base64,"
    + base64.b64encode(_FAVICON_PATH.read_bytes()).decode("ascii")
    if _FAVICON_PATH.is_file()
    else ""
)


class GroundedAnswerHtmlRenderer:
    """Render a grounded answer without trusting model-produced HTML."""

    def render(
        self,
        answer_markdown: str,
        grounder: Any,
        session: Any,
        evidence_texts: Sequence[str],
        sentence_translations: Mapping[str, str] | None = None,
        *,
        verified: bool = True,
        mark_supported: bool = True,
    ) -> tuple[str, str]:
        answer_markdown, confidence_separator, confidence_payload = (
            answer_markdown.partition("\n\n---\n\n**Answer confidence:**\n")
        )
        safe_markdown = self._escape_raw_html(answer_markdown)
        marked_markdown = (
            grounder.mark_answer_html(
                session,
                safe_markdown,
                evidence_texts,
                sentence_translations=sentence_translations,
            )
            if mark_supported
            else safe_markdown
        )
        if verified:
            title = "Grounded answer"
            note = (
                "Orange highlights mark claims supported by retrieved evidence. "
                "Orange diagram nodes and edges have supporting evidence; "
                "unmarked claims are not verified."
            )
        elif mark_supported and evidence_texts:
            title = "Not fully grounded"
            note = (
                "Not fully grounded: this generated answer is unverified and may "
                "contain unsupported claims. Any orange highlights mark only spans "
                "or diagram elements matched to retrieved evidence. Unmarked claims "
                "and source references are not verified."
            )
        else:
            title = "Unverified answer preview"
            note = (
                "Grounding failed, was skipped, or was disabled. This generated answer "
                "is unverified and may contain unsupported claims."
            )
        markdown = MarkdownIt("default", {"html": True}).enable(
            ["table", "strikethrough"]
        )
        tokens = markdown.parse(marked_markdown)
        for token in tokens:
            if (
                token.type == "fence"
                and token.info.strip().lower() == "mermaid"
                and _FLOWCHART_START.match(token.content)
            ):
                token.content = _FLOWCHART_LABEL.sub(
                    lambda match: (
                        f'{match["node"]}["{match["label"]}"]'
                        if match["node"] is not None
                        else match[0]
                    ),
                    token.content,
                )
        rendered_body = markdown.renderer.render(tokens, markdown.options, {})
        rendered_confidence = ""
        if confidence_separator:
            confidence_html = MarkdownIt("default", {"html": False}).render(
                "**Answer confidence:**\n" + confidence_payload
            )
            rendered_confidence = (
                '<section class="answer-confidence">\n<hr>\n'
                f"{confidence_html}</section>"
            )
        nonce = secrets.token_urlsafe(18)
        csp = (
            "default-src 'none'; "
            f"script-src 'nonce-{nonce}' https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline'; img-src data:; font-src data:; "
            "connect-src 'none'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'"
        )
        mermaid_scripts = ""
        if _MERMAID_FENCE.search(answer_markdown):
            mermaid_scripts = f"""
<script nonce="{nonce}" defer src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<script nonce="{nonce}">
document.addEventListener("DOMContentLoaded", async function () {{
  if (window.mermaid) window.mermaid.initialize({{
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    flowchart: {{ htmlLabels: false }},
    themeVariables: {{
      primaryColor: "#f5f5f5",
      primaryBorderColor: "#777777",
      lineColor: "#777777",
      secondaryColor: "#f5f5f5",
      tertiaryColor: "#ffffff"
    }}
  }});
  for (const code of document.querySelectorAll("pre > code.language-mermaid")) {{
    const source = (code.textContent || "").split(/\\r?\\n/)
      .filter(function (line) {{ return !line.trim().startsWith("%%{{"); }})
      .join("\\n");
    const diagram = document.createElement("pre");
    diagram.className = "mermaid";
    diagram.textContent = source;
    const pre = code.parentElement;
    pre.replaceWith(diagram);
    try {{
      if (!window.mermaid) throw new Error("Mermaid library could not be loaded.");
      await window.mermaid.run({{ nodes: [diagram] }});
    }} catch (error) {{
      diagram.replaceWith(pre);
      const message = document.createElement("p");
      message.className = "mermaid-error";
      message.setAttribute("role", "alert");
      message.textContent = "Diagram could not be rendered. Showing Mermaid source. "
        + (error.message || String(error));
      pre.before(message);
      console.error("Mermaid rendering failed", error);
    }}
  }}
}});
</script>"""

        page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
{f'<link rel="icon" type="image/png" href="{_FAVICON_DATA_URI}">' if _FAVICON_DATA_URI else ""}
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1000px; padding: 2rem; color: #242424; line-height: 1.6; }}
h1, h2, h3 {{ line-height: 1.25; }}
a {{ color: #075e9b; }}
mark.grounded {{ background: #ffe0b2; color: inherit; border-radius: .2em; padding: 0 .12em; }}
.grounding-note {{ color: #555; font-size: .9rem; }}
table {{ border-collapse: collapse; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid #bbb; padding: .4rem .7rem; }}
th {{ background: #f3f3f3; }}
pre {{ overflow-x: auto; padding: 1rem; background: #f5f5f5; border-radius: .4rem; }}
pre.mermaid {{ background: transparent; padding: .5rem 0; text-align: center; }}
blockquote {{ border-left: 3px solid #ccc; color: #555; margin-left: 0; padding-left: 1rem; }}
img {{ max-width: 100%; height: auto; }}
</style>
{mermaid_scripts}
</head>
<body>
<h1>{title}</h1>
<p class="grounding-note">{note}</p>
{rendered_body}
{rendered_confidence}
</body>
</html>
"""
        return page, csp

    @staticmethod
    def _escape_raw_html(markdown: str) -> str:
        lines: list[str] = []
        fence_char = ""
        fence_size = 0
        for line in markdown.splitlines(keepends=True):
            content = line.rstrip("\r\n")
            opening = _FENCE_OPEN.match(content)
            closing = _FENCE_CLOSE.match(content)
            if opening is not None and not fence_char:
                marker = opening.group(1)
                fence_char = marker[0]
                fence_size = len(marker)
                lines.append(line)
            elif fence_char:
                lines.append(line)
                if (
                    closing is not None
                    and closing.group(1)[0] == fence_char
                    and len(closing.group(1)) >= fence_size
                ):
                    fence_char = ""
                    fence_size = 0
            else:
                lines.append(line.replace("<", "&lt;"))
        return "".join(lines)
