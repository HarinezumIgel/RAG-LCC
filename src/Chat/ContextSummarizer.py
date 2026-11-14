"""Context-budgeted map/reduce summarization of indexed document chunks."""

import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, cast

from langchain_core.documents import Document
from tqdm import tqdm

from AI.LLMCaller import LLMCaller
from AI.TokenBudget import TokenBudget
from Commons.Exceptions import LLMResultError
from Config.Config import Config
from Globals.Session import Session
from Grounding.SourceReferences import (SUMMARY_SCHEMA, SummaryStatement,
                                        parse_summary, summary_payload)
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers


class ContextSummarizer:
    LENGTH_INSTRUCTIONS = {
        "short": "Be concise: retain only the main points and conclusions.",
        "medium": "Provide a balanced summary of key points, facts, and important qualifications.",
        "long": "Provide a detailed section-by-section summary, retaining facts, examples, and qualifications.",
        "very_long": (
            "Provide comprehensive section-by-section coverage, retaining supporting details, "
            "examples, exceptions, and qualifications. Do not pad or invent details."
        ),
    }

    @classmethod
    def validate_setting(cls, name: str, value: str | float | int) -> str | float | int:
        if name == "summary_length":
            normalized = str(value).strip().lower().replace(" ", "_")
            if normalized not in cls.LENGTH_INSTRUCTIONS:
                raise ValueError(
                    "summary_length must be short, medium, long, or very_long."
                )
            return normalized
        if name == "summary_temperature":
            try:
                temperature = float(value)
            except ValueError as exc:
                raise ValueError(
                    "summary_temperature must be between 0 and 1."
                ) from exc
            if not math.isfinite(temperature) or not 0 <= temperature <= 1:
                raise ValueError("summary_temperature must be between 0 and 1.")
            return temperature
        if name == "chunk_overlap":
            try:
                overlap = int(value)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    "chunk_overlap must be a non-negative integer."
                ) from exc
            if overlap < 0 or float(value) != overlap:
                raise ValueError("chunk_overlap must be a non-negative integer.")
            return overlap
        raise ValueError(f"Unknown summarize setting: {name}")

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        llm_caller: LLMCaller | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.helpers = Helpers()
        self.llm_caller = llm_caller or LLMCaller()
        self.token_budget = token_budget or TokenBudget(cfg=self.cfg)
        self.pretty = PrettyWriter()
        self.debug_helper = DebugHelper(self.cfg)
        self.chunk_overlap = self.cfg.get_int("CHUNK_OVERLAP")
        self.summary_prompt_template = self.cfg.get_str("SUMMARY_PROMPT_TEMPLATE")
        self.final_prompt_template = self.cfg.get_str(
            "_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE"
        )
        self.summary_temperature = self.cfg.get_float("SUMMARY_TEMPERATURE")
        if self.chunk_overlap < 0:
            raise ValueError("Summarize: CHUNK_OVERLAP must be non-negative.")
        self.chunk_summaries: list[list[SummaryStatement]] = []
        self.reduction_summaries: list[list[list[SummaryStatement]]] = []
        self.statements: list[SummaryStatement] = []
        self.summary = ""
        self._summary_calls = 0

    @staticmethod
    def _format_prompt(instruction: str, parts: Sequence[SummaryStatement]) -> str:
        return f"{instruction}\n\n" + summary_payload(parts)

    def _summary_word_limits(self, length: str) -> tuple[int, int]:
        raw_limits: object = self.cfg.get("SUMMARY_WORD_LIMITS")
        if not isinstance(raw_limits, dict):
            raise ValueError(
                "Summarize: SUMMARY_WORD_LIMITS must be a preset-to-budget mapping."
            )
        limits = cast(dict[str, object], raw_limits)
        raw_preset = limits.get(length)
        if not isinstance(raw_preset, dict):
            raise ValueError(
                f"Summarize: SUMMARY_WORD_LIMITS.{length} must define intermediate and final budgets."
            )
        preset = cast(dict[str, object], raw_preset)
        values: list[int] = []
        for stage in ("intermediate", "final"):
            value = preset.get(stage)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Summarize: SUMMARY_WORD_LIMITS.{length}.{stage} must be a positive integer."
                )
            values.append(value)
        return values[0], values[1]

    @staticmethod
    def _summary_instruction(template: str, detail: str, max_words: int) -> str:
        if "{max_words}" in template:
            template = template.replace("{max_words}", str(max_words))
        else:
            # Keep existing custom templates working while still requesting a limit.
            template = f"{template}\nUse at most {max_words} words. Do not pad."
        return (
            f"{template}\n{detail}\n"
            'Return ONLY JSON: {"statements":[{"text":"summary statement",'
            '"context_chunk_ids":[1]}]}. Keep IDs from the input; never invent IDs. '
            "Each text must contain plain summary prose, not embedded chunk labels. "
            "Preserve grounding IDs when combining statements."
        )

    def _count_tokens(self, text: str) -> int:
        # A word estimate undercounts code, hex dumps and multilingual PDF text.
        # UTF-8 bytes bound byte-fallback tokenizers without loading model weights.
        return max(
            self.token_budget.count_tokens_approx(text), len(text.encode("utf-8"))
        )

    def _fits(self, parts: Sequence[SummaryStatement], instruction: str) -> bool:
        return (
            self._count_tokens(self._format_prompt(instruction, parts))
            <= self.prompt_limit
        )

    def _batches(
        self, parts: Sequence[SummaryStatement], instruction: str, overlap: int
    ) -> list[list[SummaryStatement]]:
        batches: list[list[SummaryStatement]] = []
        start = 0
        previous: list[SummaryStatement] = []
        while start < len(parts):
            carry = previous[-overlap:] if overlap else []
            # Keep as many complete previous chunks as fit, but always make
            # room for at least one new chunk so even large overlaps progress.
            while carry and not self._fits([*carry, parts[start]], instruction):
                carry = carry[1:]
            batch = list(carry)
            end = start
            while end < len(parts) and self._fits([*batch, parts[end]], instruction):
                batch.append(parts[end])
                end += 1
            if end == start:
                raise ValueError("Summarize: a chunk exceeds the usable model context.")
            batches.append(batch)
            previous = batch
            start = end
        return batches

    def _call_summary(
        self, parts: Sequence[SummaryStatement], instruction: str, *, final: bool
    ) -> list[SummaryStatement]:
        quiet = self._summary_calls > 0
        self._summary_calls += 1
        with PrettyWriter.suppress_info() if quiet else nullcontext():
            return self._generate_summary(parts, instruction, final=final)

    def _generate_summary(
        self, parts: Sequence[SummaryStatement], instruction: str, *, final: bool
    ) -> list[SummaryStatement]:
        prompt = self._format_prompt(instruction, parts)
        if not self._fits(parts, instruction):
            raise ValueError("Summarize: prompt exceeds the usable model context.")
        output_tokens = min(
            self.reserved_output,
            self.token_budget.compute_dynamic_max_tokens(
                prompt, self.model, silent=True, model_role="_ACTIVE_LLM"
            ),
            self.context_limit - self.reserved_system - self._count_tokens(prompt),
        )
        if output_tokens <= 0:
            raise ValueError(
                "Summarize: no output tokens fit within the model context."
            )
        options: dict[str, Any] = {
            "temperature": self.summary_temperature,
            "num_ctx": self.context_limit,
            "num_predict": output_tokens,
        }
        endpoint = self.helpers.get_active_endpoint_args("_ACTIVE_LLM")
        if not endpoint.get("USE_GPU", True):
            options["num_gpu"] = 0
        if self.debug_helper.on(30):
            self.pretty.write(
                "D",
                "Summarize",
                f"Calling LLM for {'final document' if final else 'document chunk'} "
                f"summarization: model={self.model}, chunks={len(parts)}.",
            )
        response = self.llm_caller.call_llm(
            model=self.model,
            prompt=prompt,
            ollama_options=options,
            answer_is_json=True,
            top_level_params=(
                {"format": SUMMARY_SCHEMA}
                if self.cfg.get_str("_ACTIVE_ENDPOINT").lower() == "ollama"
                else {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "document_summary",
                            "strict": True,
                            "schema": SUMMARY_SCHEMA,
                        },
                    }
                }
            ),
            model_role="_ACTIVE_LLM",
            streaming=False,
            stage="Summarize final document" if final else "Summarize document chunks",
            purpose="Document summarization",
        )
        if response.get("error"):
            raise LLMResultError(response["error"])
        content = response.get("content", "").strip()
        if not content:
            raise LLMResultError("The LLM returned an empty document summary.")
        try:
            statements = parse_summary(content)
        except ValueError as exc:
            raise LLMResultError(f"Invalid structured document summary: {exc}") from exc
        allowed: set[int] = set()
        for part in parts:
            for number in part.context_chunk_ids:
                allowed.add(number)

        invalid: set[int] = set()
        for item in statements:
            for number in item.context_chunk_ids:
                if number not in allowed:
                    invalid.add(number)
        if invalid:
            input_description = "input summaries" if final else "this input batch"
            self.pretty.write(
                "W",
                "Summarize",
                f"Discarding context chunk IDs absent from {input_description}: {sorted(invalid)}",
            )
            statements = [
                SummaryStatement(
                    item.text, tuple(i for i in item.context_chunk_ids if i in allowed)
                )
                for item in statements
            ]
        return statements

    def _split_statement(
        self, statement: SummaryStatement, instruction: str
    ) -> list[SummaryStatement]:
        if self._fits([statement], instruction):
            return [statement]
        text = statement.text
        if len(text) < 2:
            raise ValueError(
                "Summarize: the model context cannot fit structured summary content."
            )
        middle = len(text) // 2
        boundary = (
            max(text.rfind(char, 0, middle + 1) for char in (" ", "\n", "\t")) + 1
        )
        if 0 < boundary < len(text):
            middle = boundary
        return self._split_statement(
            SummaryStatement(text[:middle], statement.context_chunk_ids), instruction
        ) + self._split_statement(
            SummaryStatement(text[middle:], statement.context_chunk_ids), instruction
        )

    def summarize_documents(
        self, documents: Sequence[Document], *, session: Session | None = None
    ) -> str:
        """Summarize every chunk, then reduce all intermediate summaries."""
        self.chunk_summaries = []
        self.reduction_summaries = []
        self.summary = ""
        self.statements = []
        if session is not None:
            session.summary_statements = []
        self._summary_calls = 0
        length = str(
            self.validate_setting(
                "summary_length",
                (
                    session.summary_length
                    if session is not None and session.summary_length is not None
                    else self.cfg.get_str("SUMMARY_LENGTH")
                ),
            )
        )
        self.summary_temperature = float(
            self.validate_setting(
                "summary_temperature",
                (
                    session.summary_temperature
                    if session is not None and session.summary_temperature is not None
                    else self.cfg.get_float("SUMMARY_TEMPERATURE")
                ),
            )
        )
        self.chunk_overlap = int(
            self.validate_setting(
                "chunk_overlap",
                (
                    session.chunk_overlap
                    if session is not None and session.chunk_overlap is not None
                    else self.cfg.get_int("CHUNK_OVERLAP")
                ),
            )
        )
        detail_instruction = self.LENGTH_INSTRUCTIONS[length]
        intermediate_words, final_words = self._summary_word_limits(length)
        self.summary_prompt_template = self._summary_instruction(
            self.cfg.get_str("SUMMARY_PROMPT_TEMPLATE"),
            detail_instruction,
            intermediate_words,
        )
        reduction_prompt_template = self._summary_instruction(
            self.cfg.get_str("_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE"),
            detail_instruction,
            intermediate_words,
        )
        self.final_prompt_template = self._summary_instruction(
            self.cfg.get_str("_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE"),
            detail_instruction,
            final_words,
        )
        model_args = self.helpers.get_model_args("_ACTIVE_LLM")
        self.model = str(model_args["MODEL"])
        self.context_limit = (
            self.token_budget.get_effective_context_limit(self.model, session)
            if session is not None
            else self.token_budget.get_context_limit(self.model)
        )
        self.context_limit = min(
            self.context_limit,
            int(model_args.get("TOKEN_BUDGET_CONTEXT_CAP", self.context_limit)),
        )
        self.reserved_system = int(
            model_args.get(
                "TOKEN_BUDGET_RESERVED_SYSTEM", self.token_budget.reserved_system
            )
        )
        self.reserved_output = int(
            model_args.get(
                "TOKEN_BUDGET_RESERVED_OUTPUT", self.token_budget.reserved_output
            )
        )
        if self.reserved_system < 0 or self.reserved_output <= 0:
            raise ValueError(
                "Summarize: model token reserves must allow a positive output budget."
            )
        self.prompt_limit = (
            self.context_limit - self.reserved_system - self.reserved_output
        )
        if self.prompt_limit <= 0:
            raise ValueError(
                "Summarize: model token reserves leave no usable input context."
            )
        if any(
            not self._fits([SummaryStatement("x", (1,))], instruction)
            for instruction in (
                self.summary_prompt_template,
                reduction_prompt_template,
                self.final_prompt_template,
            )
        ):
            raise ValueError(
                "Summarize: the model context cannot fit the summary prompt and content."
            )
        parts: list[SummaryStatement] = []
        for chunk_id, document in enumerate(documents, 1):
            document.metadata["SummaryChunkId"] = chunk_id
            if document.page_content.strip():
                parts.extend(
                    self._split_statement(
                        SummaryStatement(document.page_content, (chunk_id,)),
                        self.summary_prompt_template,
                    )
                )
        if not parts:
            raise ValueError("Summarize: the selected document has no indexed text.")

        batches = self._batches(parts, self.summary_prompt_template, self.chunk_overlap)
        # Show first-call diagnostics before drawing the continuously updated bar.
        self.chunk_summaries.append(
            self._call_summary(batches[0], self.summary_prompt_template, final=False)
        )
        with tqdm(
            total=len(batches), initial=1, desc="Summarize batches", unit="batch"
        ) as progress:
            for batch in batches[1:]:
                self.chunk_summaries.append(
                    self._call_summary(batch, self.summary_prompt_template, final=False)
                )
                progress.update(1)

        results = [item for batch in self.chunk_summaries for item in batch]
        while not self._fits(results, self.final_prompt_template):
            reduction_parts: list[SummaryStatement] = []
            for result in results:
                reduction_parts.extend(
                    self._split_statement(result, reduction_prompt_template)
                )
            reduction_batches = self._batches(
                reduction_parts, reduction_prompt_template, 0
            )
            reduced: list[list[SummaryStatement]] = []
            with tqdm(
                total=len(reduction_batches),
                desc=f"Reduce summaries (pass {len(self.reduction_summaries) + 1})",
                unit="batch",
            ) as progress:
                for batch in reduction_batches:
                    reduced.append(
                        self._call_summary(batch, reduction_prompt_template, final=True)
                    )
                    progress.update(1)
            flattened = [item for batch in reduced for item in batch]
            if self._count_tokens(summary_payload(flattened)) >= (
                self._count_tokens(summary_payload(results))
            ):
                raise LLMResultError(
                    "Intermediate summaries did not shrink enough to fit the model context."
                )
            self.reduction_summaries.append(reduced)
            results = flattened
        with tqdm(total=1, desc="Final summary", unit="merge") as progress:
            self.statements = self._call_summary(
                results, self.final_prompt_template, final=True
            )
            self.summary = "\n\n".join(item.text for item in self.statements)
            if session is not None:
                session.summary_statements = list(self.statements)
            progress.update(1)
        if not any(item.context_chunk_ids for item in self.statements):
            self.pretty.write(
                "W",
                "Summarize",
                "Summary contains no context chunk IDs; grounding will search all chunks.",
            )
        return self.summary
