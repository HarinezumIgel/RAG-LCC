"""Presentation helpers for current query and session settings."""

from typing import Any, Callable

from AI.TokenBudget import TokenBudget
from Config.Config import Config
from Globals.Session import Session
from Gui.Colors import CYAN, GREEN, RESET, YELLOW
from Helpers.QualityDimensionEvaluator import QualityDimensionEvaluator


class QuerySettingsPresenter:
    cfg: Config
    tokenBudget: TokenBudget
    session: Session
    prompt_map: dict[str, dict[str, str]]
    COMMAND_SPECS: dict[str, dict[str, Any]]
    _SETTINGS_LAYOUT: dict[str, dict[str, tuple[str, ...]]]
    _FRIENDLY_TOKEN_MAP: dict[str, str]

    def _format_settings(
        self, content: dict[str, Any], *, descriptions: bool = False
    ) -> str:
        lines: list[str] = []
        layout = {
            section: dict(groups) for section, groups in self._SETTINGS_LAYOUT.items()
        }
        assigned = {
            key
            for groups in layout.values()
            for keys in groups.values()
            for key in keys
        }
        extra = tuple(
            key for key in self.prompt_map if key in content and key not in assigned
        )
        if extra:
            layout["Chat settings"]["Other commands"] = extra
        width = max(len(group) for groups in layout.values() for group in groups)
        seen: set[str] = set()
        display_names = {
            "strategy": "retrieval strategy",
            "extra_ollama_options": "extraOllamaOptions",
            "ollama_top_level_params": "ollamaTopLevelParams",
        }
        for section, groups in layout.items():
            lines.append(f"\n  ▶ {GREEN}{section:<{width}}{RESET}")
            for group, keys in groups.items():
                visible = [
                    key
                    for key in keys
                    if key in content and (not descriptions or key not in seen)
                ]
                if not visible and (keys or descriptions):
                    continue
                color = (
                    YELLOW
                    if group in ("Retrieval strategy", "Orchestration strategy")
                    else CYAN
                )
                header = f"    ▶ {color}{group:<{width}}{RESET}: "
                if descriptions:
                    lines.append(header.rstrip())
                    for key in visible:
                        lines.append(f"      ▶ {key}: {content[key]}")
                        seen.add(key)
                    if group == "Summary":
                        lines.append(
                            "      ▶ summarize: summarize the selected indexed file; "
                            "summarize: name or full path selects a file for this turn. "
                            "summary: is an alias. Without a pin, CLI opens path!; "
                            "OpenWebUI offers recent grounded sources."
                        )
                else:
                    line = header
                    continuation = " " * (width + 8)
                    for key in visible:
                        if (
                            self.COMMAND_SPECS.get(key, {}).get("section_break")
                            == "before"
                            and line != header
                        ):
                            lines.append(line.rstrip())
                            line = continuation
                        line += f"{display_names.get(key, key)}={content[key]!r}  "
                    lines.append(line.rstrip())
        return "\n".join(lines)

    def session_values(self, session: Session | None = None) -> dict[str, Any]:
        """Return the explicit allowlist of user-configurable session settings."""
        s = session or self.session

        # Build values dict based on known attributes
        values: dict[str, Any] = {
            "summary_length": (
                s.summary_length
                if s.summary_length is not None
                else self.cfg.get_str("SUMMARY_LENGTH")
            ),
            "summary_temperature": (
                s.summary_temperature
                if s.summary_temperature is not None
                else self.cfg.get_float("SUMMARY_TEMPERATURE")
            ),
            "chunk_overlap": (
                s.chunk_overlap
                if s.chunk_overlap is not None
                else self.cfg.get_int("CHUNK_OVERLAP")
            ),
            "collection": getattr(s, "collection_name", None),
            "chat_name": getattr(s, "chat_name", None),
            "strategy": getattr(s, "strategy", None),
            "orchestrator_flow": getattr(s, "orchestrator_flow", None),
            "confidence_evaluation": QualityDimensionEvaluator.resolve_mode(
                self.cfg, s
            ),
            "force_retrieve_mode": getattr(s, "force_retrieve_mode", None),
            "main_query_source": getattr(s, "main_query_source", None),
            "secondary_query_source": getattr(s, "secondary_query_source", None),
            "rerank_query_source": getattr(s, "rerank_query_source", None),
            "retrieval_language": getattr(s, "retrieval_language_override", None),
            "fetch_k": getattr(s, "retriever_k", None),
            "context_chunks": getattr(s, "final_chunks_to_llm", None),
            "threshold": getattr(s, "chroma_threshold", None),
            "max_output_tokens": (
                f"{getattr(s, 'max_output_tokens', None)}"
                + (
                    f"  [override(api: max_tokens)={s.max_output_tokens_override}]"
                    if getattr(s, "max_output_tokens_override", None) is not None
                    else ""
                )
            ),
            "context_size": (
                f"{self.tokenBudget.get_context_limit()}"
                if getattr(s, "context_size_override", None) is None
                else f"override(api: num_ctx)={s.context_size_override}"
            ),
            "terminal_line_size": self._resolve_terminal_line_size(s),
            "preferred_response_language": s.preferred_response_language,
            "temperature": getattr(s, "temperature", None),
            "top_p": getattr(s, "top_p", None),
            "top_k": getattr(s, "top_k", None),
            "rerank": getattr(s, "rerank", None),
            "vector_weight": getattr(s, "vector_weight", None),
            "bm25_weight": getattr(s, "bm25_weight", None),
            "graph_weight": getattr(s, "graph_weight", None),
            "regex_weight": getattr(s, "regex_weight", None),
            "web_search": (
                "web_only"
                if getattr(s, "retrieve_mode", None) == "WEB"
                else (
                    "local_and_web" if getattr(s, "web_search", False) else "local_only"
                )
            ),
            "web_weight": getattr(s, "web_weight", None),
            "web_fetch_k": getattr(s, "web_fetch_k", None),
            "fetch_page_content": (
                "fetch pages"
                if getattr(s, "fetch_page_content", False)
                else "snippets only"
            ),
            "file": getattr(s, "file_name", None),
            "path": getattr(s, "file_path", None),
            "metadata": (
                "  ".join(
                    f"{k}={v}"
                    for k, v in (getattr(s, "metadata_filters", {}) or {}).items()
                )
                or None
            ),
            "file_cap": getattr(s, "per_file_limit", None),
            "use_chat_context": getattr(s, "use_chat_context", None),
            "history_keep": getattr(s, "turns", None),
            "history_prune": getattr(s, "prune_batch", None),
            "rewrite_context": getattr(s, "max_history_turns", None),
            "topic_summary": getattr(s, "topic_summary_mode", None),
            "debug_level": getattr(s, "debug_level", None),
            "mark_text": getattr(s, "mark_text", False),
            "extra_ollama_options": getattr(s, "extraOllamaOptions", None),
            "ollama_top_level_params": getattr(s, "ollamaTopLevelParams", None),
            "enable_query_rewrite": getattr(s, "enable_query_rewrite", None),
            "enable_keyword_action_splitter": getattr(
                s, "enable_keyword_action_splitter", None
            ),
            "enable_pronoun_substitution": getattr(
                s,
                "enable_pronoun_substitution",
                None,
            ),
            "enable_guardrail_leg": getattr(s, "enable_guardrail_leg", None),
            "enable_vector_alternate_queries": getattr(
                s,
                "enable_vector_alternate_queries",
                None,
            ),
            "enable_rerank": getattr(s, "enable_rerank", None),
            "enable_low_score_fallback": getattr(
                s,
                "enable_low_score_fallback",
                None,
            ),
            "enable_low_recall_rescue": getattr(
                s,
                "enable_low_recall_rescue",
                None,
            ),
            "enable_grounding": getattr(s, "enable_grounding", None),
            "enable_indexed_query_shaping": getattr(
                s,
                "enable_indexed_query_shaping",
                None,
            ),
            "enable_indexed_query_translation": getattr(
                s,
                "enable_indexed_query_translation",
                None,
            ),
            "enable_indexed_query_wordnet_expansion": getattr(
                s,
                "enable_indexed_query_wordnet_expansion",
                None,
            ),
            "enable_original_language_vector_leg": getattr(
                s,
                "enable_original_language_vector_leg",
                None,
            ),
            "enable_vector_retriever": getattr(s, "enable_vector_retriever", None),
            "enable_bm25_retriever": getattr(s, "enable_bm25_retriever", None),
            "enable_graph_retriever": getattr(s, "enable_graph_retriever", None),
            "enable_regex_retriever": getattr(s, "enable_regex_retriever", None),
        }

        values["debug_mode"] = getattr(s, "debug_mode", "le") or "le"
        values["bm25_pre_filter"] = (
            self.cfg.get_float("_WEB_SEARCH.bm25_pre_filter") or 0.0
        )
        values["cosine_pre_filter"] = (
            self.cfg.get_float("_WEB_SEARCH.cosine_pre_filter") or 0.0
        )
        if values["web_fetch_k"] is None:
            values["web_fetch_k"] = self.cfg.get_int("_WEB_SEARCH.max_results") or 5
        values["web_rerank_threshold"] = getattr(s, "web_rerank_threshold", None)
        if values["web_rerank_threshold"] is None:
            values["web_rerank_threshold"] = (
                self.cfg.get_float("_WEB_SEARCH.rerank_threshold") or 0.0
            )
        for alias, canonical in self._FRIENDLY_TOKEN_MAP.items():
            if canonical in values:
                values[alias] = values[canonical]
        return values

    def _resolve_terminal_line_size(self, session: Session | None = None) -> int:
        """Return the active terminal line size, resolving the debug/no_debug dict if needed.

        When TERMINAL_LINE_SIZE is a dict, the result is driven by the session's
        debug_level so that the width tracks the live session state rather than
        the config string (which may lag after a mid-session change).
        """
        raw = self.cfg.get("TERMINAL_LINE_SIZE")
        if isinstance(raw, dict):
            if int(getattr(session or self.session, "debug_level", 0) or 0) != 0:
                return int(raw.get("debug", 180))  # type: ignore[reportUnknownArgumentType]
            return int(raw.get("no_debug", 100))  # type: ignore[reportUnknownArgumentType]
        return int(str(raw)) if raw is not None else 160

    def _web_search_display(self) -> str:
        s = self.session
        return (
            "web_only"
            if getattr(s, "retrieve_mode", None) == "WEB"
            else ("local_and_web" if s.web_search else "local_only")
        )

    def _build_show_handlers(self) -> dict[str, Callable[[], None]]:
        s = self.session

        def _show_max_output_tokens() -> None:
            override = s.max_output_tokens_override
            used = s.max_output_tokens
            if override is not None:
                print(f"[max_output_tokens] override={override}  last_used={used}")
            else:
                print(f"[max_output_tokens] {used}  (dynamic, no override set)")

        def _show_context_size() -> None:
            override = s.context_size_override
            if override is not None:
                print(
                    f"[context_size] override={override}  (auto={self.tokenBudget.get_context_limit()})"
                )
            else:
                print(
                    f"[context_size] {self.tokenBudget.get_context_limit()}  (auto, no override set)"
                )

        def _show_web_rerank_threshold() -> None:
            val = getattr(s, "web_rerank_threshold", None)
            if val is None:
                val = self.cfg.get_float("_WEB_SEARCH.rerank_threshold") or 0.0
            print(f"[web_rerank_threshold] {val}")

        def _show_fetch_page_content() -> None:
            mode = "fetch pages" if s.fetch_page_content else "snippets only"
            print(f"[fetch_page_content] {mode}")

        def _show_web_fetch_k() -> None:
            val = getattr(s, "web_fetch_k", None)
            if val is None:
                val = self.cfg.get_int("_WEB_SEARCH.max_results") or 5
            print(f"[web_fetch_k] {val}")

        def _show_debug_level() -> None:
            mode = getattr(s, "debug_mode", "le") or "le"
            print(f"[debug_level] {int(s.debug_level or 0)}  mode={mode!r}")

        def _show_debug_mode() -> None:
            mode = getattr(s, "debug_mode", "le") or "le"
            print(f"[debug_mode] {mode!r}")

        handlers: dict[str, Callable[[], None]] = {
            "confidence_evaluation": lambda: print(
                "[confidence_evaluation] "
                + QualityDimensionEvaluator.resolve_mode(self.cfg, s)
            ),
            "strategy": lambda: print(f"[strategy] {s.strategy}"),
            "orchestrator_flow": lambda: print(
                f"[orchestrator_flow] {s.orchestrator_flow}"
            ),
            "force_retrieve_mode": lambda: print(
                f"[force_retrieve_mode] {getattr(s, 'force_retrieve_mode', None)}"
            ),
            "main_query_source": lambda: print(
                f"[main_query_source] {getattr(s, 'main_query_source', None)}"
            ),
            "secondary_query_source": lambda: print(
                f"[secondary_query_source] {getattr(s, 'secondary_query_source', None)}"
            ),
            "rerank_query_source": lambda: print(
                f"[rerank_query_source] {getattr(s, 'rerank_query_source', None)}"
            ),
            "retrieval_language": lambda: print(
                f"[retrieval_language] {getattr(s, 'retrieval_language_override', None)}"
            ),
            "chat_name": lambda: print(f"[chat_name] {s.chat_name}"),
            "context_chunks": lambda: print(
                f"[context_chunks] {s.final_chunks_to_llm}"
            ),
            "fetch_k": lambda: print(f"[fetch_k] {s.retriever_k}"),
            "threshold": lambda: print(f"[threshold] {s.chroma_threshold}"),
            "max_output_tokens": _show_max_output_tokens,
            "context_size": _show_context_size,
            "temperature": lambda: print(f"[temperature] {s.temperature}"),
            "top_p": lambda: print(f"[top_p] {s.top_p}"),
            "top_k": lambda: print(f"[top_k] {s.top_k}"),
            "file_cap": lambda: print(f"[file_cap] {s.per_file_limit}"),
            "vector_weight": lambda: print(f"[vector_weight] {s.vector_weight}"),
            "bm25_weight": lambda: print(f"[bm25_weight] {s.bm25_weight}"),
            "graph_weight": lambda: print(f"[graph_weight] {s.graph_weight}"),
            "regex_weight": lambda: print(f"[regex_weight] {s.regex_weight}"),
            "web_search": lambda: print(f"[web_search] {self._web_search_display()}"),
            "web_weight": lambda: print(f"[web_weight] {s.web_weight}"),
            "web_fetch_k": _show_web_fetch_k,
            "web_rerank_threshold": _show_web_rerank_threshold,
            "fetch_page_content": _show_fetch_page_content,
            "rerank": lambda: print(f"[rerank] {s.rerank}"),
            "collection": lambda: print(f"[collection] {s.collection_name}"),
            "use_chat_context": lambda: print(
                f"[use_chat_context] {s.use_chat_context}"
            ),
            "history_keep": lambda: print(f"[history_keep] {s.turns}"),
            "history_prune": lambda: print(f"[history_prune] {s.prune_batch}"),
            "rewrite_context": lambda: print(
                f"[rewrite_context] {s.max_history_turns}"
            ),
            "topic_summary": lambda: print(f"[topic_summary] {s.topic_summary_mode}"),
            "terminal_line_size": lambda: print(
                f"[terminal_line_size] {self._resolve_terminal_line_size()}"
            ),
            "debug_level": _show_debug_level,
            "debug_mode": _show_debug_mode,
            "mark_text": lambda: print(
                f"[mark_text] {bool(getattr(s, 'mark_text', False))}"
            ),
        }
        return handlers
