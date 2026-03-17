from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers

# ============================================================
# Unified Output Object
# ============================================================


@dataclass
class ModelOutput:
    content: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    raw: Optional[str] = None
    is_json: bool = False


# ============================================================
# Normalization Helpers (model-agnostic)
# ============================================================

LLAMA_GUARD_S_LOOKUP = {
    "S0": ("allow", "LlamaGuard: safe / no violation"),
    "S1": ("block", "LlamaGuard: hate speech"),
    "S2": ("block", "LlamaGuard: harassment"),
    "S3": ("block", "LlamaGuard: violence"),
    "S4": ("block", "LlamaGuard: sexual content"),
    "S5": ("block", "LlamaGuard: criminal activity"),
    "S6": ("block", "LlamaGuard: regulated advice"),
    "S7": ("block", "LlamaGuard: self-harm"),
    "S8": ("block", "LlamaGuard: weapons / extremism"),
    "S9": ("block", "LlamaGuard: other safety risk"),
}

SAFE_KEYWORDS = {"safe", "allow", "allowed", "benign", "compliant", "ok"}
UNSAFE_KEYWORDS = {
    "unsafe",
    "block",
    "blocked",
    "violation",
    "not allowed",
    "not_allowed",
    "forbidden",
    "harmful",
    "dangerous",
    "illegal",
    "noncompliant",
}

# ============================================================
# ModelOutputAdapter
# ============================================================


class ModelOutputAdapter:
    """
    Registry for model-specific output adapters.
    Normalizes raw model output into a unified ModelOutput object.
    Provides interpretation and normalization for compliance and LLM outputs.
    """

    def __init__(
        self, pretty: "PrettyWriter | None" = None, config: "Config | None" = None
    ) -> None:
        self.helpers: Helpers = Helpers()
        self.fileUtils: FileUtils = FileUtils()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.cfg: Config = config or Config()
        # Registry of model-specific adapters
        self._adapters: Dict[str, Callable[[str, bool], ModelOutput]] = {
            "mistral": self._adapter_json_llm_adapter(fallback_to_generic=True),
            "llama-guard": self._adapter_guard,
            "promptguard": self._adapter_guard,
            "guard": self._adapter_guard,
            "llama": self._adapter_json_llm_adapter(fallback_to_generic=False),
        }

    def _normalize_compliance_label(self, text: str) -> Tuple[Optional[str], str]:
        """
        Normalize compliance label text to decision and reason.
        """
        if not text:
            return None, "Empty compliance output"
        t = text.strip()
        t_low = t.lower()
        m = re.search(r"\b(s\s*\d{1})\b", t_low)
        if m:
            key = m.group(1).upper()
            if key in LLAMA_GUARD_S_LOOKUP:
                decision, reason = LLAMA_GUARD_S_LOOKUP[key]
                return decision, reason
            return "error", f"Unknown S-token [{t}]"
        if t_low in SAFE_KEYWORDS:
            return "allow", "Prompt classified as safe"
        if t_low in UNSAFE_KEYWORDS:
            return "block", "Prompt classified as unsafe"
        if any(x in t_low for x in UNSAFE_KEYWORDS):
            return "block", "Detected unsafe keywords"
        if any(x in t_low for x in SAFE_KEYWORDS):
            return "allow", "Detected safe keywords"
        return None, f"Unknown compliance label [{t}]"

    def _try_extract_json(self, text: str) -> Tuple[bool, dict[str, Any]]:
        """
        Try to extract a JSON block (fenced or whole text) and parse it.
        Returns (success, parsed_dict). Uses fileUtils.try_parse_json which
        should handle nested JSON strings if available.
        Fallback: merge multiple JSON objects emitted on separate lines.
        """
        block = (
            self._carve_out(text, r"```{", r"}```", include_fence=False) or text
        ).strip()
        ok, parsed = self.fileUtils.try_parse_json(block)
        if ok and isinstance(parsed, dict):
            return True, parsed  # type: ignore[reportUnnecessaryIsInstance]
        # Fix missing closing braces/brackets (common LLM truncation issue)
        repaired = self._try_fix_json(block)
        if repaired is not None:
            ok2, parsed2 = self.fileUtils.try_parse_json(repaired)
            if ok2 and isinstance(parsed2, dict):
                return True, parsed2  # type: ignore[reportUnnecessaryIsInstance]
        # Fallback: model emitted multiple JSON objects on separate lines
        # (e.g. {"allowed": "allowed"}\n{"explanation": "..."}) — merge dicts.
        merged: dict[str, Any] = {}
        found_any = False
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            line_ok, obj = self.fileUtils.try_parse_json(line)
            if line_ok and isinstance(obj, dict):
                merged.update(obj)  # type: ignore[reportArgumentType]
                found_any = True
        if found_any:
            return True, merged
        return False, {}

    def _flatten_set_literals(self, text: str) -> str:
        """
        LLMs sometimes emit set-like values instead of strings, e.g.
          "key": {"str1", "str2"}
        which is not valid JSON.  Detect such patterns and flatten them
        to a comma-separated string:
          "key": "str1, str2"
        """

        # Pattern: { "...", "..." } where entries have NO colon (so not a real dict)
        def _flatten_match(m: re.Match[str]) -> str:
            inner = m.group(1)
            # If any entry contains a colon between a key and value (i.e. "k": "v")
            # it's likely a real JSON object — leave it alone.
            if re.search(r'"[^"]*"\s*:', inner):
                return m.group(0)
            # Extract all quoted strings and join them
            strings = re.findall(r'"([^"]*)"', inner)
            return '"' + ", ".join(strings) + '"'

        return re.sub(
            r'\{\s*((?:"[^"]*"(?:\s*,\s*"[^"]*")*)\s*)\}', _flatten_match, text
        )

    def _try_fix_json(self, text: str) -> str | None:
        """
        If text has more opening braces/brackets than closing ones,
        append the missing closers and return the patched string.
        Returns None if no repair was needed or the patch still isn't valid JSON.
        When TRY_FIX_JSON_LLM_REPLY is disabled, still detects fixable issues
        and hints the user to enable the switch, but returns None without patching.
        """
        enabled = self.cfg.get("TRY_FIX_JSON_LLM_REPLY") == True
        # First flatten any set-like {"str", "str"} patterns to plain strings
        stripped = self._flatten_set_literals(text.strip())
        open_braces = stripped.count("{") - stripped.count("}")
        open_brackets = stripped.count("[") - stripped.count("]")
        if open_braces <= 0 and open_brackets <= 0:
            # Even if no braces to add, flattening alone might have fixed it
            try:
                json.loads(stripped)
                if stripped != text.strip():
                    if enabled:
                        self.pretty.write(
                            "W",
                            "JSON Repair",
                            "Flattened set-like values to strings",
                            color=ORANGE,
                        )
                        return stripped
                    else:
                        self.pretty.write(
                            "W",
                            "JSON Repair",
                            "Malformed LLM JSON detected (set-like values). "
                            "Enable TRY_FIX_JSON_LLM_REPLY in config to auto-fix.",
                            color=ORANGE,
                        )
                        return None
            except Exception:
                pass
            return None
        patched = stripped + ("}" * max(open_braces, 0)) + ("]" * max(open_brackets, 0))
        # Quick sanity check before returning
        try:
            json.loads(patched)
        except Exception:
            return None
        fixes: list[str] = []
        if stripped != text.strip():
            fixes.append("flattened set-like values")
        if open_braces > 0:
            fixes.append(f"{open_braces} missing '}}' added")
        if open_brackets > 0:
            fixes.append(f"{open_brackets} missing ']' added")
        if enabled:
            self.pretty.write(
                "W",
                "JSON Repair",
                f"Fixed malformed LLM JSON: {', '.join(fixes)}",
                color=ORANGE,
            )
            return patched
        else:
            self.pretty.write(
                "W",
                "JSON Repair",
                f"Malformed LLM JSON detected ({', '.join(fixes)}). "
                "Enable TRY_FIX_JSON_LLM_REPLY in config to auto-fix.",
                color=ORANGE,
            )
            return None

    # -------------------------
    # Public API
    # -------------------------

    def interpret(
        self,
        raw_output: Any,
        model_name: str,
        is_compliance: bool = False,
        is_streaming: bool = False,
    ) -> ModelOutput:
        """
        Interpret model output into a unified ModelOutput object.

        Args:
            raw_output:    The model's raw output (dict, string, or other).
            model_name:    Name of the model — selects the appropriate adapter.
            is_compliance: True when calling a guard / compliance model.
            is_streaming:  True when output is NDJSON fragments; they are
                           assembled before the adapter runs.

        Returns:
            ModelOutput with normalised content, decision, reason, raw, is_json.
        """
        raw_text = self._extract_raw_text(raw_output, is_compliance)
        original_raw = raw_text  # preserve before any assembly overwrites it
        model_name_l = str(model_name).lower()

        # Streaming mode: assemble fragments into a single JSON response
        if is_streaming:
            raw_text = self._assemble_stream(raw_text)

        # Compliance mode → use compliance adapters
        if is_compliance:
            for key, adapter in self._adapters.items():
                if key in model_name_l:
                    result = adapter(raw_text, is_streaming)
                    result.raw = original_raw
                    return result
            result = self._adapter_generic_compliance(raw_text)
            result.raw = original_raw
            return result

        # Normal LLM mode
        for key, adapter in self._adapters.items():
            if key in model_name_l:
                result = adapter(raw_text, is_streaming)
                result.raw = original_raw
                return result

        result = self._adapter_generic_llm(raw_text)
        result.raw = original_raw
        return result

    # -------------------------
    # Raw text extraction
    # -------------------------

    def _extract_raw_text(self, raw_output: Any, is_compliance: bool) -> str:
        """
        Extract raw text from model output (dict or string).
        """
        if isinstance(raw_output, dict):
            d: dict[str, Any] = cast(dict[str, Any], raw_output)
            if is_compliance:
                return (
                    d.get("raw")
                    or d.get("content")
                    or d.get("response")  # Ollama non-streaming guard response
                    or json.dumps(d, ensure_ascii=False)
                )
            else:
                return (
                    d.get("content")
                    or d.get("raw")
                    or json.dumps(d, ensure_ascii=False)
                )
        return str(raw_output)

    # -------------------------
    # Streaming assembly (only called when is_streaming=True)
    # -------------------------

    def _assemble_stream(self, text: str) -> str:
        """
        Assemble response fragments from NDJSON streaming output.
        Each line is a JSON object with a 'response', 'content', or 'delta' key.
        Returns the concatenated string, or the original text if parsing fails.
        """
        parsed = self._parse_json_lines(text)
        if not parsed:
            return text

        fragments: List[str] = []
        keys = ("response", "content", "delta")
        for obj in parsed:
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    fragments.append(v)
                    break

        return "".join(fragments) if fragments else text

    # -------------------------
    # Generic LLM adapter (tries to extract JSON block and normalize)
    # -------------------------

    def _adapter_generic_llm(self, text: str) -> ModelOutput:
        """
        Generic LLM adapter: tries to extract JSON block and normalize.
        """
        try:
            ok, parsed = self._try_extract_json(text)
            if ok:
                return ModelOutput(
                    content=json.dumps(parsed, ensure_ascii=False),
                    raw=text,
                    is_json=True,
                )
        except Exception:
            pass
        return ModelOutput(content=text, raw=text, is_json=False)

    # -------------------------
    # Generic compliance adapter (fallback)
    # -------------------------

    def _adapter_generic_compliance(self, text: str) -> ModelOutput:
        """
        Generic compliance adapter: fallback for compliance output normalization.
        """
        decision, reason = self._normalize_compliance_label(text)
        return ModelOutput(
            decision=decision or "error", reason=reason, raw=text, is_json=False
        )

    # -------------------------
    # Shared helpers: compliance key detection + two-level JSON parse
    # -------------------------

    def _compliance_output_from_dict(
        self, d: dict[str, Any], raw_text: str
    ) -> Optional[ModelOutput]:
        """
        Return a compliance ModelOutput if d contains answer/allowed keys, else None.
        Handles both boolean and string forms of the allowed value.
        Maps answer/reason/explanation to the canonical decision/reason fields.
        """
        if "answer" in d and isinstance(d["answer"], str):
            answer_val = d["answer"].strip().lower()
            is_allowed = answer_val in (
                "allow",
                "allowed",
                "safe",
                "compliant",
                "yes",
                "true",
            )
            return ModelOutput(
                content=d["answer"],
                decision="allow" if is_allowed else "block",
                reason=d.get("reason") or d.get("explanation"),
                raw=raw_text,
                is_json=True,
            )
        if "allowed" in d:
            allowed_val = d["allowed"]
            if isinstance(allowed_val, bool):
                is_allowed = allowed_val
            else:
                is_allowed = str(allowed_val).strip().lower() in (
                    "allowed",
                    "allow",
                    "true",
                    "yes",
                    "compliant",
                )
            return ModelOutput(
                decision="allow" if is_allowed else "block",
                reason=d.get("explanation") or d.get("reason"),
                raw=raw_text,
                is_json=True,
            )
        return None

    def _adapter_json_llm(
        self, text: str, fallback_to_generic: bool = False
    ) -> ModelOutput:
        """
        Shared JSON-aware adapter used by both Mistral and Llama.

        Parse flow:
          1. Parse text as JSON (top level).
          2. Check for compliance keys (answer/allowed) at top level
             — handles already-assembled streaming output.
          3. If a text field in the top dict is itself a JSON string
             (Ollama non-streaming wrapper), re-parse and repeat compliance check.
          4. Inner plain text → return as content.
          5. Top-level JSON with no text wrapper → return stringified as content.

        fallback_to_generic: if True, total parse failure delegates to
          _adapter_generic_llm (Mistral behaviour); otherwise returns
          ModelOutput(content=text) directly (Llama behaviour).
        """
        ok, parsed = self._try_extract_json(text)
        if ok:
            # Step 1: compliance keys at top level (streaming / assembled case)
            result = self._compliance_output_from_dict(parsed, text)
            if result:
                return result

            # Step 2: outer dict may wrap a text field that is itself JSON
            # (non-streaming Ollama: {"model":..., "response":"{\"answer\":...}"})
            # Only check known wrapper keys (response/content/text/…) to avoid
            # mistaking classification output values for an inner wrapper.
            inner_text = self._first_text_field(parsed, priority_only=True)
            if inner_text:
                inner_ok, inner_parsed = self._try_extract_json(inner_text)
                if inner_ok:
                    result = self._compliance_output_from_dict(inner_parsed, text)
                    if result:
                        return result
                    return ModelOutput(
                        content=json.dumps(inner_parsed, ensure_ascii=False),
                        raw=text,
                        is_json=True,
                    )
                # inner is plain text
                return ModelOutput(content=inner_text, raw=text, is_json=True)

            # No text wrapper — return top-level JSON as content
            return ModelOutput(
                content=json.dumps(parsed, ensure_ascii=False),
                raw=text,
                is_json=True,
            )
        return (
            self._adapter_generic_llm(text)
            if fallback_to_generic
            else ModelOutput(content=text, raw=text, is_json=False)
        )

    # -------------------------
    # JSON LLM adapter factory
    # -------------------------

    def _adapter_json_llm_adapter(
        self, fallback_to_generic: bool
    ) -> Callable[[str, bool], ModelOutput]:
        """Return an adapter that delegates to _adapter_json_llm with the given fallback flag."""

        def _adapter(text: str, is_streaming: bool) -> ModelOutput:
            return self._adapter_json_llm(text, fallback_to_generic=fallback_to_generic)

        return _adapter

    def _first_text_field(
        self, obj: dict[str, Any], priority_only: bool = False
    ) -> Optional[str]:
        """
        Return the first plausible text field from a parsed JSON object.
        Checks priority keys first (text/content/response/answer/message/body),
        then falls back to a shallow recursive scan of nested dicts and lists.

        Args:
            priority_only: If True, only check the priority keys (known model
                wrapper keys) and skip the shallow scan.  Use this when the
                caller needs to distinguish wrapper dicts from real LLM output.
        """
        if not isinstance(obj, dict):  # type: ignore[reportUnnecessaryIsInstance]
            return None

        # Priority keys commonly used by models
        priority = ("text", "content", "response", "answer", "message", "body")
        for k in priority:
            root = obj.get(k)
            if isinstance(root, str) and root.strip():
                return root.strip()

        if priority_only:
            return None

        # Shallow scan: top-level string values
        for root in obj.values():
            if isinstance(root, str) and root.strip():
                return root.strip()

        # Shallow recursive scan for nested dicts/lists (one level)
        for root in obj.values():
            if isinstance(root, dict):
                nested_dict: dict[str, Any] = cast(dict[str, Any], root)
                for nested in nested_dict.values():
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
            elif isinstance(root, list):
                items_list: list[Any] = cast(list[Any], root)
                for item in items_list:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        item_dict: dict[str, Any] = cast(dict[str, Any], item)
                        for nested in item_dict.values():
                            if isinstance(nested, str) and nested.strip():
                                return nested.strip()

        return None

    # -------------------------
    # Guard-style adapter (Llama-Guard / PromptGuard / generic guard)
    # Streaming mode: output is JSONL with response fragments
    # Non-streaming mode: output is compliance label text (S0-S9, safe, unsafe, etc.)
    # -------------------------

    def _adapter_guard(self, text: str, is_streaming: bool) -> ModelOutput:
        """
        Guard-style adapter (Llama-Guard / PromptGuard / generic guard).
        Streaming mode: output is JSONL with response fragments.
        Non-streaming mode: output is compliance label text.
        """
        if is_streaming:
            objs = self._parse_json_lines(text)
            if objs:
                return self._adapter_guard_ndjson(text, objs)

        # Non-streaming: text may already be a label string, or may be a full
        # Ollama JSON object (e.g. {"response": "unsafe\nS2", "done": true, ...}).
        # Try to parse as single JSON object and pull the "response" field so that
        # the S-token regex in _normalize_compliance_label gets a clean target.
        label: str = text
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                obj_d: dict[str, Any] = cast(dict[str, Any], obj)
                label = str(obj_d.get("response") or text)
        except (json.JSONDecodeError, ValueError):
            pass

        # _normalize_compliance_label checks S-token regex FIRST, so
        # "unsafe\nS2" → S2 → LLAMA_GUARD_S_LOOKUP wins over keyword fallback.
        decision, reason = self._normalize_compliance_label(label)
        return ModelOutput(
            decision=decision or "error", reason=reason, raw=text, is_json=False
        )

    # -------------------------
    # Newline delimited json
    # -------------------------
    def _adapter_guard_ndjson(
        self, text: str, objs: List[dict[str, Any]]
    ) -> ModelOutput:
        """
        Adapter for newline-delimited JSON guard output.
        Reuses _assemble_stream for fragment extraction, then matches
        S-tokens or semantic labels.
        """
        merged = self._assemble_stream(text).strip()

        # Try S-token match first
        m = re.search(r"s\s*\d", merged, flags=re.IGNORECASE)
        if m:
            s_token = m.group(0).replace(" ", "").upper()
            decision, reason = self._normalize_compliance_label(s_token)
            return ModelOutput(
                decision=decision or "error",
                reason=reason,
                content=merged,
                raw=text,
                is_json=False,
            )

        # Fall back to full merged text as a semantic label
        if merged:
            decision, reason = self._normalize_compliance_label(merged)
            return ModelOutput(
                decision=decision or "error",
                reason=reason,
                content=merged,
                raw=text,
                is_json=False,
            )

        return ModelOutput(
            decision="error", reason="No guard decision found", raw=text, is_json=False
        )

    def _parse_json_lines(self, text: str) -> Optional[List[dict[str, Any]]]:
        """
        Parse text as JSONL (newline-delimited JSON).
        Returns a list of dicts on success, or None if any line fails to parse
        or if the input is empty / contains no JSON objects.
        """
        if not text:
            return None

        objs: List[dict[str, Any]] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                return None
            if not isinstance(obj, dict):
                return None
            objs.append(cast(dict[str, Any], obj))

        return objs if objs else None

    def _carve_out(
        self, text: str, start: str, end: str, include_fence: bool = False
    ) -> Optional[str]:
        """
        Extract a fenced block between `start` and `end`.
        If include_fence is False, return the inner content (between fences).
        If include_fence is True, return the block including the fences.
        """
        if (
            not isinstance(text, str)  # type: ignore[reportUnnecessaryIsInstance]
            or not isinstance(start, str)  # type: ignore[reportUnnecessaryIsInstance]
            or not isinstance(end, str)  # type: ignore[reportUnnecessaryIsInstance]
        ):
            raise TypeError("text, start and end must be strings")
        s_idx = text.find(start)
        if s_idx == -1:
            return None
        e_idx = text.find(end, s_idx + len(start))
        if e_idx == -1:
            return None
        inner_start = s_idx if include_fence else s_idx + len(start)
        inner_end = e_idx + len(end) if include_fence else e_idx
        return text[inner_start:inner_end]
