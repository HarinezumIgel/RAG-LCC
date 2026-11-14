"""Structured summary statements and their candidate grounding context."""

import json
from dataclasses import dataclass
from typing import Any, Sequence

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "context_chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                    },
                },
                "required": ["text", "context_chunk_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["statements"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SummaryStatement:
    text: str
    context_chunk_ids: tuple[int, ...]


def summary_payload(statements: Sequence[SummaryStatement]) -> str:
    return json.dumps(
        {
            "statements": [
                {
                    "text": statement.text,
                    "context_chunk_ids": list(statement.context_chunk_ids),
                }
                for statement in statements
            ]
        },
        ensure_ascii=False,
    )


def parse_summary(content: str) -> list[SummaryStatement]:
    """Reject malformed output rather than guessing references from prose."""
    data = json.loads(content)
    if not isinstance(data, dict) or set(data) != {"statements"}:
        raise ValueError("Expected an object containing only 'statements'.")
    items = data["statements"]
    if not isinstance(items, list) or not items:
        raise ValueError("'statements' must be a non-empty list.")
    result: list[SummaryStatement] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"text", "context_chunk_ids"}:
            raise ValueError(
                "Each statement must contain 'text' and 'context_chunk_ids'."
            )
        text, ids = item["text"], item["context_chunk_ids"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Statement text must be non-empty.")
        if not isinstance(ids, list) or any(
            type(value) is not int or value < 1 for value in ids
        ):
            raise ValueError("'context_chunk_ids' must be a list of positive integers.")
        result.append(SummaryStatement(text.strip(), tuple(dict.fromkeys(ids))))
    return result
