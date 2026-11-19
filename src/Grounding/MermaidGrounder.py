"""Extract factual claims from the flowchart subset of Mermaid."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*$")
_NODE = r"[A-Za-z_][A-Za-z0-9_-]*(?:[ \t]*(?:\[[^\]\n]*\]|\([^()\n]*\)|\{[^{}\n]*\}))?"
_NODE_TOKEN = re.compile(
    r"^(?P<id>[A-Za-z_][A-Za-z0-9_-]*)"
    r"(?:[ \t]*(?:\[(?P<square>[^\]\n]*)\]"
    r"|\((?P<round>[^()\n]*)\)|\{(?P<brace>[^{}\n]*)\}))?$"
)
_EDGE = re.compile(
    rf"^(?P<source>{_NODE})[ \t]*(?P<operator>-->|---|-.->|==>|--o|--x|o--o|x--x)"
    rf"[ \t]*(?:\|(?P<label>[^|\n]+)\|[ \t]*)?(?P<target>{_NODE})[ \t]*$"
)
_LABELLED_EDGE = re.compile(
    rf"^(?P<source>{_NODE})[ \t]+--[ \t]*(?P<label>.+?)[ \t]*-->[ \t]*(?P<target>{_NODE})[ \t]*$"
)
_GRAPH_START = re.compile(
    r"^(?:graph|flowchart)[ \t]+(?:TB|TD|BT|RL|LR)\b", re.IGNORECASE
)
_NON_FACTUAL_DIRECTIVE = re.compile(
    r"^(?:classDef|class|style|linkStyle|click|direction)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class MermaidBlock:
    """A Mermaid fence and the claims extracted from its supported syntax."""

    start: int
    end: int
    source_start: int
    source_end: int
    source: str
    claims: tuple[str, ...]
    parseable: bool
    # (source line index, claim) pairs so callers can highlight individual
    # diagram lines whose claim is evidence-supported.
    claim_lines: tuple[tuple[int, str], ...] = ()


class MermaidGrounder:
    """Parse Mermaid flowcharts into ordered, evidence-checkable claims."""

    def find_blocks(self, markdown: str) -> list[MermaidBlock]:
        """Find Mermaid fences, including malformed unclosed Mermaid fences."""
        lines = markdown.splitlines(keepends=True)
        blocks: list[MermaidBlock] = []
        offset = 0
        fence_char = ""
        fence_size = 0
        fence_start = 0
        source_start = 0
        source_lines: list[str] = []
        mermaid = False

        for line in lines:
            line_without_newline = line.rstrip("\r\n")
            if not fence_char:
                opening = _FENCE_OPEN.match(line_without_newline)
                if opening:
                    marker, info = opening.groups()
                    fence_char = marker[0]
                    fence_size = len(marker)
                    fence_start = offset
                    source_start = offset + len(line)
                    source_lines = []
                    language = info.strip().split(maxsplit=1)[0:1]
                    mermaid = bool(language and language[0].lower() == "mermaid")
                offset += len(line)
                continue

            closing = _FENCE_CLOSE.match(line_without_newline)
            if (
                closing
                and closing.group(1)[0] == fence_char
                and len(closing.group(1)) >= fence_size
            ):
                if mermaid:
                    source = "".join(source_lines)
                    extracted = self._extract_claims_with_lines(source)
                    blocks.append(
                        MermaidBlock(
                            start=fence_start,
                            end=offset + len(line),
                            source_start=source_start,
                            source_end=offset,
                            source=source,
                            claims=extracted[0] if extracted else (),
                            parseable=extracted is not None,
                            claim_lines=extracted[1] if extracted else (),
                        )
                    )
                fence_char = ""
                fence_size = 0
                mermaid = False
                source_lines = []
            elif mermaid:
                source_lines.append(line)
            offset += len(line)

        if fence_char and mermaid:
            # An unterminated Mermaid fence must not bypass validation.
            source = "".join(source_lines)
            extracted = self._extract_claims_with_lines(source)
            blocks.append(
                MermaidBlock(
                    start=fence_start,
                    end=len(markdown),
                    source_start=source_start,
                    source_end=len(markdown),
                    source=source,
                    claims=extracted[0] if extracted else (),
                    parseable=False,
                    claim_lines=extracted[1] if extracted else (),
                )
            )
        return blocks

    def without_blocks(self, markdown: str) -> str:
        """Remove Mermaid blocks so prose grounding ignores diagram syntax."""
        parts: list[str] = []
        cursor = 0
        for block in self.find_blocks(markdown):
            parts.append(markdown[cursor : block.start])
            cursor = block.end
        parts.append(markdown[cursor:])
        return "".join(parts)

    def extract_claims(self, source: str) -> tuple[str, ...] | None:
        """Return flowchart node/edge claims, or None for unsupported syntax."""
        extracted = self._extract_claims_with_lines(source)
        return None if extracted is None else extracted[0]

    def with_grounding_styles(
        self, markdown: str, is_grounded: Callable[[str], bool]
    ) -> str:
        """Colour only supported flowchart nodes and edges in browser output."""
        result = markdown
        for block in reversed(self.find_blocks(markdown)):
            if not block.parseable:
                continue
            supported_lines = {
                index for index, claim in block.claim_lines if is_grounded(claim)
            }
            nodes: set[str] = set()
            edges: list[int] = []
            lines: list[str] = []
            edge_index = 0
            for index, original in enumerate(block.source.splitlines()):
                line = original.split("%%", 1)[0].strip().rstrip(";").strip()
                # Model-provided colours must not masquerade as evidence marks.
                if re.match(
                    r"^(?:classDef|class|style|linkStyle)\b", line, re.IGNORECASE
                ):
                    continue
                lines.append(original)
                edge = _LABELLED_EDGE.fullmatch(line) or _EDGE.fullmatch(line)
                if edge is not None:
                    if index in supported_lines:
                        for key in ("source", "target"):
                            node = self._parse_node(edge.group(key))
                            if node is not None:
                                nodes.add(node[0])
                        edges.append(edge_index)
                    edge_index += 1
                elif index in supported_lines:
                    node = self._parse_node(line)
                    if node is not None:
                        nodes.add(node[0])
            if nodes:
                lines.extend(
                    (
                        "classDef groundedEvidence fill:#ffe0b2,stroke:#ef6c00,color:#242424;",
                        f"class {','.join(sorted(nodes))} groundedEvidence;",
                    )
                )
            if edges:
                lines.append(
                    f"linkStyle {','.join(map(str, edges))} stroke:#ef6c00,stroke-width:3px;"
                )
            result = (
                result[: block.source_start]
                + "\n".join(lines)
                + "\n"
                + result[block.source_end :]
            )
        return result

    def _extract_claims_with_lines(
        self, source: str
    ) -> tuple[tuple[str, ...], tuple[tuple[int, str], ...]] | None:
        """Extract claims plus the source line index each claim came from."""
        nodes: dict[str, tuple[str, bool]] = {}
        node_first_line: dict[str, int] = {}
        edges: list[tuple[str, str, str, str, int]] = []
        connected: set[str] = set()
        saw_graph = False

        for line_index, original_line in enumerate(source.splitlines()):
            line = original_line.split("%%", 1)[0].strip().rstrip(";").strip()
            if not line:
                continue
            if not saw_graph and _GRAPH_START.match(line):
                saw_graph = True
                continue
            if line.lower() == "end" or _NON_FACTUAL_DIRECTIVE.match(line):
                continue
            if line.startswith("%%{"):
                continue

            edge_match = _LABELLED_EDGE.fullmatch(line)
            edge_label = ""
            if edge_match is None:
                edge_match = _EDGE.fullmatch(line)
                if edge_match is not None:
                    edge_label = (edge_match.group("label") or "").strip()
            else:
                edge_label = edge_match.group("label").strip()

            if edge_match is not None:
                source_node = self._parse_node(edge_match.group("source"))
                target_node = self._parse_node(edge_match.group("target"))
                if source_node is None or target_node is None:
                    return None
                self._add_node(nodes, *source_node)
                self._add_node(nodes, *target_node)
                node_first_line.setdefault(source_node[0], line_index)
                node_first_line.setdefault(target_node[0], line_index)
                operator = edge_match.groupdict().get("operator", "-->")
                edges.append(
                    (source_node[0], target_node[0], edge_label, operator, line_index)
                )
                connected.update((source_node[0], target_node[0]))
                continue

            node = self._parse_node(line)
            if node is not None:
                self._add_node(nodes, *node)
                node_first_line.setdefault(node[0], line_index)
                continue
            return None

        if not saw_graph or not nodes or any(not label for label, _ in nodes.values()):
            return None

        claims: list[str] = []
        claim_lines: list[tuple[int, str]] = []
        for source_id, target_id, edge_label, operator, line_index in edges:
            source_label = nodes[source_id][0]
            target_label = nodes[target_id][0]
            relation = edge_label or (
                "connected to" if operator in {"---", "o--o", "x--x"} else "leads to"
            )
            claim = f"{source_label} {relation} {target_label}".strip()
            claims.append(claim)
            claim_lines.append((line_index, claim))

        for node_id, (label, _) in nodes.items():
            if node_id not in connected:
                claims.append(label)
                claim_lines.append((node_first_line[node_id], label))

        if not claims:
            return None
        return tuple(dict.fromkeys(claims)), tuple(claim_lines)

    @staticmethod
    def _parse_node(value: str) -> tuple[str, str, bool] | None:
        match = _NODE_TOKEN.fullmatch(value.strip())
        if match is None:
            return None
        node_id = match.group("id")
        label = match.group("square") or match.group("round") or match.group("brace")
        explicit_label = label is not None
        if label is None:
            label = node_id.replace("_", " ").replace("-", " ")
        label = label.strip().strip("\"'`")
        return node_id, label, explicit_label

    @staticmethod
    def _add_node(
        nodes: dict[str, tuple[str, bool]],
        node_id: str,
        label: str,
        explicit_label: bool,
    ) -> None:
        existing = nodes.get(node_id)
        if existing is None:
            nodes[node_id] = (label, explicit_label)
        elif not existing[0]:
            return
        elif label == existing[0]:
            nodes[node_id] = (label, existing[1] or explicit_label)
        elif explicit_label and not existing[1]:
            nodes[node_id] = (label, True)
        elif explicit_label and existing[1]:
            nodes[node_id] = ("", True)
