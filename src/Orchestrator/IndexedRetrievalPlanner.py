"""Extracted IndexedRetrievalPlanner behavior; the coordinating facade owns shared state."""

from __future__ import annotations

from typing import Any, Literal, cast

from Commons.Exceptions import ConfigurationError, QueryTranslationError
from Globals.Session import Session
from Helpers.LanguageConfig import get_active_language_codes
from Orchestrator.OrchestrationStageContext import OrchestrationStageContext
from Orchestrator.OrchestratorTypes import GraphRegexStageIteration


class IndexedRetrievalPlanner(OrchestrationStageContext):
    def _normalize_language_bucket(self, language: Any) -> str | None:
        """Normalize a language label into a lower-case bucket code."""
        return self._rag_chat_impl._normalize_language_bucket(language)

    def _language_bucket_from_filter(
        self,
        file_filter: dict[str, Any] | None,
    ) -> str | None:
        """Extract an explicit Language filter bucket if one is set."""
        if not isinstance(file_filter, dict) or "Language" not in file_filter:
            return None
        language_value = file_filter.get("Language")
        if isinstance(language_value, dict):
            language_dict = cast(dict[str, Any], language_value)
            eq_value = language_dict.get("$eq")
            return self._normalize_language_bucket(eq_value)
        return self._normalize_language_bucket(language_value)

    @staticmethod
    def _remove_language_from_filter(
        file_filter: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Drop Language from a flat filter so discovery sees all languages."""
        if not isinstance(file_filter, dict):
            return None
        without_language = {k: v for k, v in file_filter.items() if k != "Language"}
        return without_language or None

    def _indexed_query_settings(self) -> dict[str, Any]:
        settings = self.orchestratorOverrides.cfg_get_dict("_INDEXED_QUERY", {})
        policy = settings.get("translation_failure_policy", "skip_language")
        if policy not in {"stop", "skip_language"}:
            raise ConfigurationError(
                "_INDEXED_QUERY.translation_failure_policy must be stop or skip_language"
            )
        limit = settings.get("language_detection_limit", 0)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ConfigurationError(
                "_INDEXED_QUERY.language_detection_limit must be a nonnegative integer (0 = all)"
            )
        if not isinstance(settings.get("use_alternates", True), bool):
            raise ConfigurationError("_INDEXED_QUERY.use_alternates must be a boolean")
        return settings

    def _discover_graph_regex_languages(
        self,
        file_filter: dict[str, Any] | None,
    ) -> list[str]:
        """Discover normalized language values from collection metadata."""
        collection = self._rag_chat_impl.collection
        if collection is None:
            return []

        where_filter = self._remove_language_from_filter(file_filter)
        limit = self._indexed_query_settings().get("language_detection_limit", 0)
        query_options: dict[str, Any] = {}
        if limit:
            query_options["limit"] = limit
        try:
            result = collection.get(
                include=["metadatas"],
                where=where_filter,
                **query_options,
            )
        except Exception as exc:
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="E",
                message=f"Document-language discovery failed: {exc}",
            )
            raise

        result_dict: dict[str, Any] | None = (
            cast(dict[str, Any], result) if isinstance(result, dict) else None
        )
        metadatas_obj = (
            result_dict.get("metadatas") if result_dict is not None else None
        )
        if not isinstance(metadatas_obj, list):
            return []
        metadatas = cast(list[Any], metadatas_obj)

        languages: set[str] = set()
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            meta_dict = cast(dict[str, Any], meta)
            normalized = self._normalize_language_bucket(meta_dict.get("Language"))
            if normalized:
                languages.add(normalized)

        return sorted(languages)

    def _active_configured_language_codes(self) -> set[str]:
        """Return active language codes from shared ACTIVE_LANGUAGES config."""
        configured_codes = get_active_language_codes(self._rag_chat_impl.cfg)
        active_codes: set[str] = set()
        for code in configured_codes:
            normalized = self._normalize_language_bucket(code)
            if normalized:
                active_codes.add(normalized)

        return active_codes or {"en"}

    def _relevant_graph_regex_languages(
        self,
        mySession: Session,
        available_languages: list[str],
    ) -> list[str]:
        """Select query-relevant languages from the discovered set."""
        available = {lang for lang in available_languages if lang}
        if not available:
            return []

        current_query_lang = cast(
            str | None, getattr(mySession, "current_query_lang", None)
        )
        user_language = cast(str | None, getattr(mySession, "user_language", None))
        retrieval_language = cast(
            str | None, getattr(mySession, "retrieval_language", None)
        )
        candidates: list[str | None] = [
            current_query_lang,
            user_language,
            retrieval_language,
            "english",
            "en",
        ]

        selected: list[str] = []
        for candidate in candidates:
            normalized = self._normalize_language_bucket(candidate)
            if normalized and normalized in available and normalized not in selected:
                selected.append(normalized)
        return selected

    def _plan_graph_regex_language_buckets(
        self,
        mySession: Session,
        file_filter: dict[str, Any] | None,
    ) -> list[str]:
        """Plan language buckets for Graph/Regex stage iterations."""
        vector_store_languages = self._discover_graph_regex_languages(None)
        active_languages: set[str] | None = None
        if vector_store_languages:
            active_languages = self._active_configured_language_codes()
            self.orchestratorMessages.emit_vector_store_language_status(
                mySession,
                vector_store_languages=vector_store_languages,
                active_codes=sorted(active_languages),
            )

        explicit_language = self._language_bucket_from_filter(file_filter)
        # A language-only filter is removed during discovery, so its scan is
        # identical to the unfiltered corpus scan already performed above.
        available_languages = (
            vector_store_languages
            if self._remove_language_from_filter(file_filter) is None
            else self._discover_graph_regex_languages(file_filter)
        )
        if active_languages is None:
            active_languages = self._active_configured_language_codes()
        allowed_languages = sorted(
            lang for lang in available_languages if lang in active_languages
        )
        limit = self._indexed_query_settings().get("language_detection_limit", 0)
        self.orchestratorMessages.debug_query_step(
            mySession,
            f"Step 4: document languages={available_languages!r}; "
            f"active/present={allowed_languages!r}; "
            f"metadata limit={limit or 'whole collection'}",
        )

        if explicit_language:
            return [explicit_language] if explicit_language in allowed_languages else []

        relevant_languages = self._relevant_graph_regex_languages(
            mySession,
            allowed_languages,
        )
        if relevant_languages:
            # Keep query-relevant buckets first, then run remaining active/present
            # buckets so multilingual corpora are still fully covered.
            relevant = [
                lang for lang in relevant_languages if lang in allowed_languages
            ]
            relevant_set = set(relevant)
            remaining = [lang for lang in allowed_languages if lang not in relevant_set]
            return relevant + remaining

        # Loop only over languages active in config and present in the vector store.
        return [lang for lang in allowed_languages]

    @staticmethod
    def _build_graph_regex_stage_filter(
        base_file_filter: dict[str, Any] | None,
        language_bucket: str | None,
    ) -> dict[str, Any] | None:
        """Build the file filter for one Graph/Regex language bucket."""
        staged_filter = dict(base_file_filter or {})
        if language_bucket:
            staged_filter["Language"] = language_bucket
        return staged_filter or None

    def _graph_regex_stage_filters(
        self,
        mySession: Session,
        file_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any] | None]:
        """Return Graph/Regex stage filters for each execution iteration."""
        stage_filters: list[dict[str, Any] | None] = []
        for language_bucket in self._plan_graph_regex_language_buckets(
            mySession,
            file_filter,
        ):
            stage_filters.append(
                self._build_graph_regex_stage_filter(file_filter, language_bucket)
            )
        return stage_filters

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
    ) -> tuple[str, str]:
        """Return per-language query forms for one Graph/Regex iteration.

        Delegates query-shaping policy to RAGChatImpl.
        This keeps translation/backend policy centralized in RAGChatImpl.
        """
        target_language = self._normalize_language_bucket(language_bucket)
        try:
            shaped = self._rag_chat_impl._shape_graph_regex_stage_queries(
                mySession,
                language_bucket=target_language,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                enable_translation=self._flow_enable_indexed_query_translation,
                enable_synonym_expansion=self._flow_enable_indexed_query_wordnet_expansion,
            )
            if not isinstance(shaped, tuple) or len(shaped) != 2:
                raise QueryTranslationError(
                    f"Indexed query shaping for {target_language} returned an invalid query pair"
                )
            primary, guardrail = shaped
            if (
                not isinstance(primary, str)
                or not isinstance(guardrail, str)
                or (primary_query.strip() and not primary.strip())
            ):
                raise QueryTranslationError(
                    f"Indexed query shaping for {target_language} returned invalid query text"
                )
            return primary.strip(), guardrail.strip()
        except QueryTranslationError:
            raise
        except Exception as exc:
            self.orchestratorMessages.write_retrieval_orchestration(
                severity="E",
                message=f"Indexed query shaping failed for {target_language}: {exc}",
            )
            raise

    def _shape_primary_indexed_query(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        query: str,
        failure_policy: str,
    ) -> str | None:
        """Shape a required query; report a skipped language or stop on failure."""
        try:
            shaped_query, _ = self._shape_graph_regex_stage_queries(
                mySession,
                language_bucket=language_bucket,
                primary_query=query,
                guardrail_query="",
            )
            return shaped_query
        except QueryTranslationError as exc:
            self.orchestratorMessages.warn_indexed_translation_once(
                mySession,
                language=language_bucket,
                category="primary",
                query=query,
                severity="E" if failure_policy == "stop" else "W",
                message=f"Translation failed for {language_bucket}: {exc}. "
                + (
                    "Stopping before retrieval."
                    if failure_policy == "stop"
                    else "Skipping this language's BM25/Graph/Regex searches; coverage is incomplete."
                ),
            )
            if failure_policy == "stop":
                raise QueryTranslationError(
                    f"Cannot translate the query for document language '{language_bucket}'."
                ) from exc
            self.orchestratorMessages.record_indexed_query_notice(
                mySession,
                f"Translation unavailable for document language '{language_bucket}'; "
                "its BM25/Graph/Regex searches were skipped. Retrieval coverage is incomplete.",
            )
            return None

    def _shape_optional_indexed_query(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        query: str,
        category: Literal["guardrail", "alternative"],
    ) -> str | None:
        """Report optional-query failures without aborting the required query."""
        try:
            shaped_query, _ = self._shape_graph_regex_stage_queries(
                mySession,
                language_bucket=language_bucket,
                primary_query=query,
                guardrail_query="",
            )
            return shaped_query
        except QueryTranslationError as exc:
            self.orchestratorMessages.warn_indexed_translation_once(
                mySession,
                language=language_bucket,
                category=category,
                query=query,
                message=f"Skipping {category} for {language_bucket}: {exc}",
            )
            notice = (
                f"The optional guardrail query was skipped for document language '{language_bucket}'."
                if category == "guardrail"
                else f"Some query alternatives were skipped for document language '{language_bucket}'."
            )
            self.orchestratorMessages.record_indexed_query_notice(mySession, notice)
            return None

    def _shape_indexed_query_alternatives(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        alternatives: list[str],
    ) -> list[tuple[str, str]]:
        """Keep successful original/shaped pairs for the per-language report."""
        shaped_alternatives: list[tuple[str, str]] = []
        for alternative in alternatives:
            if alternative.strip().casefold() == primary_query.strip().casefold():
                continue
            shaped = self._shape_optional_indexed_query(
                mySession,
                language_bucket=language_bucket,
                query=alternative,
                category="alternative",
            )
            if shaped is not None:
                shaped_alternatives.append((alternative, shaped))
        return shaped_alternatives

    def _graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[GraphRegexStageIteration]:
        """Plan Graph/Regex iterations with staged filters and shaped queries."""
        iterations: list[GraphRegexStageIteration] = []
        shape_queries = bool(self._indexed_stage_shape_queries)
        settings = self._indexed_query_settings()
        alternatives = (
            self._indexed_alternate_queries
            if shape_queries and settings.get("use_alternates", True)
            else []
        )
        for stage_file_filter in self._graph_regex_stage_filters(
            mySession, file_filter
        ):
            stage_language_bucket = self._language_bucket_from_filter(stage_file_filter)
            if shape_queries:
                shaped = self._shape_stage_iteration_queries(
                    mySession,
                    language_bucket=stage_language_bucket,
                    primary_query=primary_query,
                    guardrail_query=guardrail_query,
                    alternatives=alternatives,
                    failure_policy=settings.get(
                        "translation_failure_policy", "skip_language"
                    ),
                )
                if shaped is None:
                    continue
                stage_primary_query, stage_guardrail_query, stage_alternatives = shaped
            else:
                stage_primary_query = primary_query
                stage_guardrail_query = guardrail_query
                stage_alternatives = []
            iterations.extend(
                self._expand_stage_iterations(
                    language_bucket=stage_language_bucket,
                    file_filter=stage_file_filter,
                    primary_query=stage_primary_query,
                    guardrail_query=stage_guardrail_query,
                    stage_alternatives=stage_alternatives,
                )
            )
            was_translated = (
                stage_primary_query != primary_query
                or stage_guardrail_query != guardrail_query
                or any(shaped != original for original, shaped in stage_alternatives)
            )
            self.orchestratorMessages.write_indexed_stage_query_status(
                flow_name=self._indexed_flow_name,
                target_language=self._normalize_language_bucket(stage_language_bucket),
                primary_query_source=self._indexed_primary_query_source,
                primary_source_language=self.orchestratorMessages.indexed_query_source_language(
                    mySession,
                    self._indexed_primary_query_source,
                ),
                was_translated=was_translated,
                primary_query=primary_query,
                stage_primary_query=stage_primary_query,
                guardrail_query_source=self._indexed_guardrail_query_source,
                guardrail_source_language=self.orchestratorMessages.indexed_query_source_language(
                    mySession,
                    self._indexed_guardrail_query_source,
                ),
                guardrail_query=guardrail_query,
                stage_guardrail_query=stage_guardrail_query,
                guardrail_disabled_reason=self._indexed_guardrail_disabled_reason(
                    primary_query=primary_query,
                    guardrail_query=guardrail_query,
                ),
                indexed_shaping_enabled=shape_queries,
                stage_alternatives=stage_alternatives,
            )
        return iterations

    def _shape_stage_iteration_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        alternatives: list[str],
        failure_policy: str,
    ) -> tuple[str, str, list[tuple[str, str]]] | None:
        """Shape primary, guardrail and alternate queries for one language bucket.

        Returns ``None`` when the primary query cannot be shaped and the language
        must be skipped; guardrail/alternate failures only drop that query.
        """
        stage_primary_query = self._shape_primary_indexed_query(
            mySession,
            language_bucket=language_bucket,
            query=primary_query,
            failure_policy=failure_policy,
        )
        if stage_primary_query is None:
            return None
        if not guardrail_query:
            stage_guardrail_query = ""
        elif guardrail_query == primary_query:
            stage_guardrail_query = stage_primary_query
        else:
            stage_guardrail_query = (
                self._shape_optional_indexed_query(
                    mySession,
                    language_bucket=language_bucket,
                    query=guardrail_query,
                    category="guardrail",
                )
                or ""
            )
        stage_alternatives = self._shape_indexed_query_alternatives(
            mySession,
            language_bucket=language_bucket,
            primary_query=primary_query,
            alternatives=alternatives,
        )
        return stage_primary_query, stage_guardrail_query, stage_alternatives

    @staticmethod
    def _expand_stage_iterations(
        *,
        language_bucket: str | None,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
        stage_alternatives: list[tuple[str, str]],
    ) -> list[GraphRegexStageIteration]:
        """Build the primary iteration plus one per distinct shaped alternate query."""
        iterations = [
            GraphRegexStageIteration(
                language_bucket=language_bucket,
                file_filter=file_filter,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
            )
        ]
        seen = {primary_query.casefold()}
        for _, shaped_alternative in stage_alternatives:
            if shaped_alternative.casefold() in seen:
                continue
            seen.add(shaped_alternative.casefold())
            iterations.append(
                GraphRegexStageIteration(
                    language_bucket=language_bucket,
                    file_filter=file_filter,
                    primary_query=shaped_alternative,
                    guardrail_query="",
                    is_alternate=True,
                )
            )
        return iterations

    def _indexed_guardrail_disabled_reason(
        self,
        *,
        primary_query: str,
        guardrail_query: str,
    ) -> str | None:
        """Explain why the indexed guardrail query is not used, or ``None`` if it is."""
        if not self._indexed_guardrail_enabled:
            return "disabled_by_flow"
        if guardrail_query and (
            guardrail_query.strip().casefold() == primary_query.strip().casefold()
        ):
            return "same as primary"
        return None

    def _shared_graph_regex_stage_iterations(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None,
        primary_query: str,
        guardrail_query: str,
    ) -> list[GraphRegexStageIteration]:
        """Return precomputed language-stage iterations for indexed retrievers."""
        active_iterations = self._active_indexed_stage_iterations
        if active_iterations is not None:
            return active_iterations
        return self._graph_regex_stage_iterations(
            mySession,
            file_filter=file_filter,
            primary_query=primary_query,
            guardrail_query=guardrail_query,
        )
