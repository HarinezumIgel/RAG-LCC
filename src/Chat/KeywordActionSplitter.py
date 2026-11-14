"""Split local information needs from the complete answer-generation instruction."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, cast

from AI.LLMCaller import LLMCaller
from AI.TokenBudget import TokenBudget
from Commons.Exceptions import BackendUnavailableError, LLMResultError
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import BROWN
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


@dataclass(frozen=True)
class KeywordActionSplit:
    retrieval_queries: list[str]
    action_part: str | None
    answer_instruction: str
    entities: list[str]
    key_words: list[str]


class KeywordActionSplitter:
    def __init__(
        self,
        *,
        cfg: Config | None = None,
        pretty: PrettyWriter | None = None,
        helpers: Helpers | None = None,
        llm_caller: LLMCaller | None = None,
    ) -> None:
        self.cfg = cfg or Config()
        self.pretty = pretty or PrettyWriter()
        self.pretty_always = PrettyWriter(
            logger=self.pretty.logger, cfg=self.cfg, always_on=True
        )
        self.helpers = helpers or Helpers()
        self.llm_caller = llm_caller or LLMCaller()
        self.file_utils = FileUtils(cfg=self.cfg, pretty=self.pretty)
        self.token_budget = TokenBudget()
        self.perf_logger = PerfLogger()

    def _string_list(self, data: dict[str, Any], key: str) -> list[str]:
        value: object = data[key]
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array of strings")
        result: list[str] = []
        for item in cast(list[object], value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{key} must contain nonempty strings")
            if item.strip().casefold() not in {text.casefold() for text in result}:
                result.append(item.strip())
        return result

    def parse(self, text: str) -> KeywordActionSplit:
        """Reuse the shared JSON parser, then enforce the splitter schema."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
        ok, parsed = self.file_utils.try_parse_json(stripped)
        if not ok or not isinstance(parsed, dict):
            raise ValueError("Splitter response must be a JSON object")
        data = cast(dict[str, Any], parsed)
        required = {
            "retrieval_queries",
            "action_part",
            "answer_instruction",
            "entities",
            "key_words",
        }
        if set(data) != required:
            raise ValueError(f"Splitter fields must be exactly {sorted(required)}")
        queries = self._string_list(data, "retrieval_queries")
        keywords = self._string_list(data, "key_words")
        entities = self._string_list(data, "entities")
        generic_verbs = {
            "create",
            "summarize",
            "compare",
            "list",
            "explain",
            "generate",
        }
        invalid_keywords = [
            keyword for keyword in keywords if keyword.casefold() in generic_verbs
        ]
        keywords = [
            keyword for keyword in keywords if keyword.casefold() not in generic_verbs
        ]
        max_queries = self.cfg.get_int(
            "_KEYWORD_ACTION_SPLITTER.max_retrieval_queries", 8
        )
        min_keywords = self.cfg.get_int("_KEYWORD_ACTION_SPLITTER.min_keywords", 3)
        max_keywords = self.cfg.get_int("_KEYWORD_ACTION_SPLITTER.max_keywords", 8)
        if len(queries) > max_queries:
            raise ValueError(f"More than {max_queries} retrieval queries returned")
        if queries and not min_keywords <= len(keywords) <= max_keywords:
            removed = (
                f" after removing generic action keywords {invalid_keywords}"
                if invalid_keywords
                else ""
            )
            raise ValueError(
                f"Expected {min_keywords}-{max_keywords} retrieval keywords{removed}; got {len(keywords)}"
            )
        if not queries and keywords:
            raise ValueError("Keywords require retrieval queries")
        action: object = data["action_part"]
        if action is not None and (not isinstance(action, str) or not action.strip()):
            raise ValueError("action_part must be a nonempty string or null")
        instruction: object = data["answer_instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("answer_instruction must be a nonempty string")
        if not instruction.strip().startswith("Using only the retrieved context,"):
            raise ValueError(
                "answer_instruction must be a context-only task, not an answer"
            )
        if invalid_keywords:
            self.pretty.write(
                "W",
                "KeywordActionSplitter",
                f"Removed generic action keywords from decomposition: {invalid_keywords}",
            )
        return KeywordActionSplit(
            queries,
            action.strip() if isinstance(action, str) else None,
            instruction.strip(),
            entities,
            keywords,
        )

    def split(self, session: Session) -> KeywordActionSplit | None:
        """Return validated decomposition, or warn explicitly before legacy fallback."""
        model_args = self.helpers.get_model_args("_ACTIVE_LLM_REWRITE_PROMPT")
        model: str = model_args["MODEL"]
        template, prompt_name = self.cfg.indirect_get("_PROMPT_KEYWORD_ACTION_SPLITTER")
        language = session.retrieval_language or "english"
        inputs = {
            "original_request": session.user_query_original or session.query or "",
            "translated_request": session.orig_translated_query_en
            or session.query
            or "",
            "accepted_rewrite": session.query or "",
        }
        formatted = template.format(
            input_json=json.dumps(inputs, ensure_ascii=False),
            retrieval_language=language,
            pronoun_substitution=session.enable_pronoun_substitution is not False,
            max_retrieval_queries=self.cfg.get_int(
                "_KEYWORD_ACTION_SPLITTER.max_retrieval_queries", 8
            ),
            min_keywords=self.cfg.get_int("_KEYWORD_ACTION_SPLITTER.min_keywords", 3),
            max_keywords=self.cfg.get_int("_KEYWORD_ACTION_SPLITTER.max_keywords", 8),
        )
        options: dict[str, Any] = dict(
            self.cfg.get_dict("_KEYWORD_ACTION_SPLITTER.LLM_PARAM")
        )
        use_gpu = bool(options.pop("use_ollama_gpu", True))
        if not use_gpu:
            options["num_gpu"] = 0
        options["num_ctx"] = self.token_budget.get_effective_context_limit(
            model, session
        )
        self.pretty_always.write(
            "I",
            "KeywordActionSplitter",
            f"model={model!r} language={language!r} inputs={inputs!r}",
            color=BROWN,
        )
        started = time.perf_counter()
        try:
            result = self.llm_caller.call_llm(
                model=model,
                prompt=formatted,
                ollama_options=options,
                answer_is_json=True,
                template_name=prompt_name,
                streaming=False,
                stage="Keyword/action splitting",
                purpose="Local retrieval decomposition",
                model_role="_ACTIVE_LLM_REWRITE_PROMPT",
            )
            if result.get("error"):
                raise LLMResultError(result["error"])
            split = self.parse(result.get("content", ""))
            supplied_text = " ".join(inputs.values()).casefold()
            unsupported_entities = [
                entity
                for entity in split.entities
                if entity.casefold() not in supplied_text
            ]
            if unsupported_entities:
                raise ValueError(
                    f"Entities not grounded in the supplied request: {unsupported_entities}"
                )
        except (ValueError, LLMResultError, BackendUnavailableError) as exc:
            self.pretty.write(
                "W",
                "KeywordActionSplitter",
                f"Decomposition failed; using unchanged query pipeline: {exc}",
            )
            return None
        finally:
            elapsed = time.perf_counter() - started
            self.perf_logger.log(
                "KeywordActionSplitter.split", "chat", f"elapsed={elapsed:.3f}s"
            )
            self.pretty_always.write(
                "I",
                "KeywordActionSplitter",
                f"LLM call + validation: {elapsed:.3f} s",
                color=BROWN,
            )
        details = [
            "Validated decomposition:",
            f"Retrieval queries ({len(split.retrieval_queries)}):",
        ]
        details.extend(
            f"[query_id={query_id}] {query}"
            for query_id, query in enumerate(split.retrieval_queries)
        )
        if not split.retrieval_queries:
            details.append("(none)")
        details.extend(
            [
                f"Action: {split.action_part or '(none - no transformation requested)'}",
                f"Answer instruction: {split.answer_instruction}",
                f"Entities: {', '.join(split.entities) or '(none)'}",
                f"Keywords: {', '.join(split.key_words) or '(none)'}",
            ]
        )
        self.pretty_always.write(
            "OK", "KeywordActionSplitter", "\n".join(details), color=BROWN
        )
        return split
