"""Extracted LocalRetrieverExecutor behavior; the coordinating facade owns shared state."""

from __future__ import annotations

import time
from typing import Any, cast

from langchain_chroma import Chroma

from Algos.Synonyms import Synonyms
from Chat.RagRetrievalContext import RagRetrievalContext
from Commons.Exceptions import QueryTranslationError
from Globals.Session import Session
from Helpers.DebugHelper import DebugHelper
from Orchestrator import FlowConfig
from Orchestrator.Orchestrator import Orchestrator
from Retrievers.BM25Retriever import BM25Retriever


class LocalRetrieverExecutor(RagRetrievalContext):
    @staticmethod
    def _chroma_where(flt: "dict[str, Any] | None") -> "dict[str, Any] | None":
        """Normalise a flat multi-field filter into ChromaDB ``where`` form.

        ChromaDB requires an explicit ``$and`` when more than one field is
        present. BM25/graph matchers AND flat multi-key dicts directly, so the
        shared ``base_kwargs`` filter stays flat and is converted only here for
        the vector query.
        """
        if not flt:
            return flt
        # Single field or an existing logical operator ($and/$or) — pass through.
        if len(flt) <= 1:
            return flt
        return {"$and": [{k: v} for k, v in flt.items()]}

    def _vector_kwargs(self, mySession: Session) -> "dict[str, Any]":
        """Return base_kwargs with the filter normalised for the Chroma query."""
        kwargs: dict[str, Any] = dict(mySession.base_kwargs or {})
        if "filter" in kwargs:
            kwargs["filter"] = self._chroma_where(kwargs["filter"])
        return kwargs

    @staticmethod
    def _resolve_file_filter(mySession: Session) -> dict[str, Any] | None:
        """Extract the shared metadata file/path filter from base_kwargs."""
        if mySession.base_kwargs and "filter" in mySession.base_kwargs:
            return mySession.base_kwargs["filter"]
        return None

    def _query_language_label(
        self,
        mySession: Session,
        *,
        file_filter: dict[str, Any] | None = None,
    ) -> str:
        """Resolve a compact language label for retrieval query logs."""
        if isinstance(file_filter, dict) and "Language" in file_filter:
            language_value = file_filter.get("Language")
            if isinstance(language_value, dict):
                language_dict = cast(dict[str, Any], language_value)
                language_value = language_dict.get("$eq")
            normalized = self._normalize_language_bucket(language_value)
            if normalized:
                return normalized

        for candidate in (
            getattr(mySession, "retrieval_language", None),
            getattr(mySession, "current_query_lang", None),
            getattr(mySession, "user_language", None),
        ):
            normalized = self._normalize_language_bucket(candidate)
            if normalized:
                return normalized

        return "default"

    def _normalize_language_bucket(self, language: Any) -> str | None:
        """Normalize language names/codes to lower-case ISO-ish bucket code."""
        text = str(language or "").strip().lower()
        if not text:
            return None

        mapping_obj = getattr(self._shared, "lang_name_to_code", None)
        if isinstance(mapping_obj, dict):
            mapping = cast(dict[str, Any], mapping_obj)
            mapped = mapping.get(text)
            if mapped is not None:
                mapped_text = str(mapped).strip().lower()
                if mapped_text:
                    return mapped_text

        return text

    def _shape_graph_regex_stage_queries(
        self,
        mySession: Session,
        *,
        language_bucket: str | None,
        primary_query: str,
        guardrail_query: str,
        enable_translation: bool = True,
        enable_synonym_expansion: bool = True,
    ) -> tuple[str, str]:
        """Shape Graph/Regex stage queries for one language bucket.

        Retrieval queries are normalized to English upstream, or kept in the
        user's language when the flow uses NATIVE retrieval. Translation runs
        source->bucket language; WordNet expansion only applies to English
        queries. Both are independently controlled by orchestration-flow toggles.
        """
        target_language = self._normalize_language_bucket(language_bucket)
        source_language = "en"
        if self._is_native_retrieval(mySession):
            source_language = (
                self._normalize_language_bucket(
                    getattr(mySession, "retrieval_language", None)
                )
                or "en"
            )
        source_is_english = source_language == "en" or source_language.startswith("en-")
        if not source_is_english:
            enable_synonym_expansion = False

        should_translate = bool(enable_translation)
        if not target_language or target_language == source_language:
            should_translate = False
        elif source_is_english and (
            target_language == "en" or target_language.startswith("en-")
        ):
            should_translate = False

        if should_translate:
            backend = str(
                getattr(mySession, "translation_backend", None)
                or getattr(self, "_translation_backend", "off")
                or "off"
            )
            if backend.strip().lower() != "argos":
                raise QueryTranslationError(
                    f"Indexed translation {source_language}->{target_language} requires the Argos backend"
                )

        def _shape_one_query(query: str) -> str:
            stage_query = str(query or "").strip()
            if not stage_query:
                return stage_query

            if enable_synonym_expansion:
                expanded = self._expand_indexed_query_with_synonyms(stage_query).strip()
                stage_query = expanded or stage_query

            if should_translate and target_language is not None:
                pair_key = f"{source_language}->{target_language or '?'}"
                try:
                    translated_obj = self._shared.translate_query_text(
                        stage_query, target_language, source_language
                    )
                except QueryTranslationError:
                    raise
                except Exception as exc:
                    raise QueryTranslationError(
                        f"Indexed translation {pair_key} failed: {exc}"
                    ) from exc
                translated = str(translated_obj or "").strip()
                if not translated:
                    raise QueryTranslationError(
                        f"Indexed translation {pair_key} returned empty text"
                    )
                return translated

            return stage_query

        primary = _shape_one_query(primary_query)
        guardrail = (
            primary
            if guardrail_query == primary_query
            else _shape_one_query(guardrail_query)
        )
        return primary, guardrail

    def _expand_indexed_query_with_synonyms(self, query: str) -> str:
        """Append WordNet synonym terms to an indexed-retriever query string."""
        query_text = str(query or "").strip()
        if not query_text:
            return query_text

        tokens = [
            token.strip().lower()
            for token in self._shared.tokenize(query_text)
            if token.strip()
        ]
        if not tokens:
            tokens = [
                token.strip().lower() for token in query_text.split() if token.strip()
            ]
        if not tokens:
            return query_text

        try:
            expanded_terms = Synonyms().expand(tokens)
        except Exception:
            return query_text

        base_terms = {token.lower() for token in tokens}
        extras: list[str] = []
        seen_extras: set[str] = set()
        for term in expanded_terms:
            term_text = str(term or "").strip()
            if not term_text:
                continue
            term_key = term_text.lower()
            if term_key in base_terms or term_key in seen_extras:
                continue
            seen_extras.add(term_key)
            extras.append(term_text)

        if not extras:
            return query_text

        return f"{query_text} {' '.join(extras)}"

    def _run_indexed_retriever_with_guardrail(
        self,
        *,
        mySession: Session,
        retriever: Any,
        label: str,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[list[Any], int, int]:
        """Run one indexed retriever on final and optional guardrail query.

        The guardrail query is the pre-rewrite English seed retrieval query
        (``orig_translated_query_en``), used as a second retrieval leg when it
        differs from the final rewritten query.

        Returns:
            (docs, top_k_orig_query_en, top_k_post_rewrite_query_en)
        """
        assert self.collection is not None

        label_lower = label.lower()
        query_language = self._query_language_label(
            mySession,
            file_filter=file_filter,
        )
        self.pretty.write(
            "I",
            label,
            f"Querying {label_lower} index on collection {self.collection_name} "
            f"for language {query_language}",
        )
        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"start {label_lower} block collection={self.collection_name}",
        )
        _t_block = time.perf_counter()

        index_dir = retriever.get_index_dir(self.collection_name)
        retriever.load_or_rebuild(
            index_dir,
            self.collection_name,
            self.collection,
        )

        docs_final = retriever.query(
            primary_query,
            k=mySession.retriever_k or 100,
            file_filter=file_filter,
        )
        for doc in docs_final:
            doc.metadata["retriever_sources"] = label

        docs = docs_final
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = len(docs_final)

        if use_guardrail:
            # Guardrail leg: query again with the pre-rewrite English seed
            # retrieval query to recover when rewrite/post-rewrite translation
            # drifts away from the user intent.
            docs_seed = retriever.query(
                guardrail_query,
                k=mySession.retriever_k or 100,
                file_filter=file_filter,
            )
            for doc in docs_seed:
                doc.metadata["retriever_sources"] = label
            top_k_orig_query_en = len(docs_seed)
            if docs_seed:
                docs = BM25Retriever.reciprocal_rank_fusion(
                    docs_final,
                    docs_seed,
                    k=self.bm25_retriever.rrf_k,
                    labels=[label, label],
                    weights=[1.0, 1.0],
                )

        self.pretty.write(
            "O",
            label,
            f"{label} retrieval returned {len(docs)} chunks",
        )
        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  {label_lower} block n={len(docs)} elapsed={time.perf_counter() - _t_block:.3f}s",
        )

        return docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _run_vector_retriever_with_guardrail(
        self,
        *,
        mySession: Session,
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        alternate_queries: list[str],
    ) -> tuple[list[Any], int, int]:
        """Run vector retrieval on final and optional guardrail query.

        The guardrail query is the pre-rewrite English seed retrieval query
        (``orig_translated_query_en``), used as a second retrieval leg when it
        differs from the final rewritten query.

        Also runs alternate-query fanout and de-duplicates added chunks by id.

        Returns:
            (docs, top_k_orig_query_en, top_k_post_rewrite_query_en)
        """
        query_language = self._query_language_label(mySession)
        self.pretty.write(
            "I",
            "Chroma",
            f"Querying Chroma DB on vector store {self.persist_directory} "
            f"for language {query_language}",
        )
        assert (
            self.vector_store is not None
        ), "vector_store not initialized; call set_vector_store first"

        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            "start vector similarity_search",
        )
        _t_vec = time.perf_counter()
        vector_kwargs = self._vector_kwargs(mySession)

        hits_final: list[Any] = self.vector_store.similarity_search_with_score(
            primary_query,
            **vector_kwargs,
        )
        vector_docs_final = self.chatContext.annotate_chunks(hits_final)
        for doc in vector_docs_final:
            doc.metadata["retriever_sources"] = "Vector"

        vector_docs = vector_docs_final
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = len(vector_docs_final)

        if use_guardrail:
            try:
                # Guardrail leg mirrors indexed retrievers: fuse final-query and
                # pre-rewrite English seed-retrieval-query hits so retrieval
                # remains resilient to rewrite drift.
                hits_seed: list[Any] = self.vector_store.similarity_search_with_score(
                    guardrail_query,
                    **vector_kwargs,
                )
                vector_docs_seed = self.chatContext.annotate_chunks(hits_seed)
                for doc in vector_docs_seed:
                    doc.metadata["retriever_sources"] = "Vector"
                top_k_orig_query_en = len(vector_docs_seed)
                if vector_docs_seed:
                    vector_docs = BM25Retriever.reciprocal_rank_fusion(
                        vector_docs_final,
                        vector_docs_seed,
                        k=self.bm25_retriever.rrf_k,
                        labels=["Vector", "Vector"],
                        weights=[1.0, 1.0],
                    )
            except Exception as seed_exc:
                # Vector guardrail failures should not abort the whole turn.
                # We keep the primary-query result set and continue.
                self.pretty.write(
                    "W",
                    "MultiQuery",
                    f"Seed-query vector search failed: {seed_exc}",
                )

        self.perf_logger.log(
            "RAGChatImpl._retrieve",
            "chat",
            f"stop  vector similarity_search n={len(vector_docs)} elapsed={time.perf_counter() - _t_vec:.3f}s",
        )

        if alternate_queries:
            existing_ids: set[str] = {
                str(doc.metadata.get("id", doc.page_content)) for doc in vector_docs
            }
            for query_index, alt_query in enumerate(alternate_queries, start=1):
                try:
                    alt_hits: list[Any] = (
                        self.vector_store.similarity_search_with_score(
                            alt_query,
                            **vector_kwargs,
                        )
                    )
                    alt_docs = self.chatContext.annotate_chunks(alt_hits)
                    for doc in alt_docs:
                        doc_id = str(doc.metadata.get("id", doc.page_content))
                        if doc_id not in existing_ids:
                            doc.metadata["retriever_sources"] = "Vector"
                            vector_docs.append(doc)
                            existing_ids.add(doc_id)
                except Exception as alt_exc:
                    self.pretty.write(
                        "W",
                        "MultiQuery",
                        f"Alternate query {query_index} vector search failed: {alt_exc}",
                    )

        if DebugHelper.check_session(mySession, 10):
            self._print_chroma_debug(vector_docs)
        self.pretty.write(
            "O",
            "Chroma",
            f"Querying Chroma DB query returned {len(vector_docs)} chunks",
        )

        return vector_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    @staticmethod
    def _mode_flags_for_local_retrievers(
        retrieve_mode: str,
    ) -> tuple[bool, bool, bool, bool]:
        """Return enabled flags for Vector/BM25/Graph/Regex local retrievers."""
        return FlowConfig.mode_flags_for_local_retrievers(retrieve_mode)

    @staticmethod
    def _is_guardrail_query_active(primary_query: str, guardrail_query: str) -> bool:
        """Return True when the pre-rewrite English seed query should run.

        The second guardrail leg runs only when this seed retrieval query is
        distinct from the final rewritten retrieval query.
        """
        return bool(
            guardrail_query and guardrail_query.lower() != primary_query.strip().lower()
        )

    def _resolve_guardrail_queries(
        self,
        *,
        bm25_query: str,
        orig_translated_query_en: str,
    ) -> tuple[str, str, bool]:
        """Resolve primary and guardrail retrieval query forms.

        ``primary_query`` is the final rewritten retrieval query.
        ``guardrail_query`` is the pre-rewrite English seed retrieval query.
        The guardrail leg is active only when both forms differ.
        """
        primary_query: str = bm25_query or ""
        guardrail_query: str = (orig_translated_query_en or "").strip()
        use_guardrail: bool = self._is_guardrail_query_active(
            primary_query,
            guardrail_query,
        )
        return primary_query, guardrail_query, use_guardrail

    @staticmethod
    def _accumulate_topk_counts(
        current_orig_query_en: int,
        current_post_rewrite_query_en: int,
        add_orig_query_en: int,
        add_post_rewrite_query_en: int,
    ) -> tuple[int, int]:
        """Return updated guardrail counters after adding one retriever leg."""
        return (
            current_orig_query_en + add_orig_query_en,
            current_post_rewrite_query_en + add_post_rewrite_query_en,
        )

    def _indexed_retriever_specs(
        self,
        mySession: Session,
        *,
        bm25_enabled: bool,
        graph_enabled: bool,
        regex_enabled: bool,
    ) -> list[tuple[str, bool, float, Any, int, Any]]:
        """Build retrieval specs for indexed local retrievers."""
        bm25_weight = float(
            mySession.bm25_weight if mySession.bm25_weight is not None else 1.0
        )
        graph_weight = float(
            mySession.graph_weight if mySession.graph_weight is not None else 1.0
        )
        regex_weight = float(
            mySession.regex_weight if mySession.regex_weight is not None else 1.0
        )
        return [
            (
                "BM25",
                bm25_enabled,
                bm25_weight,
                self.bm25_retriever,
                10,
                self._print_bm25_debug,
            ),
            (
                "Graph",
                graph_enabled,
                graph_weight,
                self.graph_retriever,
                30,
                self._print_graph_debug,
            ),
            (
                "Regex",
                regex_enabled,
                regex_weight,
                self.regex_retriever,
                30,
                self._print_regex_debug,
            ),
        ]

    def _run_indexed_local_retrievers(
        self,
        mySession: Session,
        *,
        indexed_specs: list[tuple[str, bool, float, Any, int, Any]],
        primary_query: str,
        guardrail_query: str,
        use_guardrail: bool,
        file_filter: dict[str, Any] | None,
    ) -> tuple[dict[str, list[Any]], int, int]:
        """Run BM25/Graph/Regex retrieval specs and collect guardrail counters.

        Guardrail counters compare hits from the pre-rewrite English seed
        retrieval query versus the final rewritten retrieval query.
        """
        indexed_docs: dict[str, list[Any]] = {
            "BM25": [],
            "Graph": [],
            "Regex": [],
        }
        top_k_orig_query_en: int = 0
        top_k_post_rewrite_query_en: int = 0

        for (
            label,
            enabled,
            weight,
            retriever,
            debug_level,
            debug_printer,
        ) in indexed_specs:
            if not enabled or weight == 0.0:
                continue
            if label == "BM25":
                assert self.persist_directory is not None

            (
                docs,
                top_k_orig,
                top_k_post,
            ) = self._run_indexed_retriever_with_guardrail(
                mySession=mySession,
                retriever=retriever,
                label=label,
                primary_query=primary_query,
                guardrail_query=guardrail_query,
                use_guardrail=use_guardrail,
                file_filter=file_filter,
            )
            indexed_docs[label] = docs
            top_k_orig_query_en += top_k_orig
            top_k_post_rewrite_query_en += top_k_post
            if DebugHelper.check_session(mySession, debug_level):
                debug_printer(docs)

        return indexed_docs, top_k_orig_query_en, top_k_post_rewrite_query_en

    def _store_guardrail_retrieval_topk(
        self,
        mySession: Session,
        *,
        top_k_orig_query_en: int,
        top_k_post_rewrite_query_en: int,
    ) -> None:
        """Store guardrail dual-query retrieval counts on the session.

        ``*_orig_query_en`` fields track the pre-rewrite English seed retrieval
        query. ``*_post_rewrite_query_en`` fields track the final rewritten
        retrieval query.
        """
        # These counters are only meaningful when the guardrail second leg ran.
        mySession.retrieval_top_k_orig_query_en = top_k_orig_query_en
        mySession.retrieval_top_k_post_rewrite_query_en = top_k_post_rewrite_query_en
        mySession.retrieval_top_k_seed_query = top_k_orig_query_en
        mySession.retrieval_top_k_final_query = top_k_post_rewrite_query_en
        mySession.retrieval_top_k_before_t2 = top_k_orig_query_en
        mySession.retrieval_top_k_after_t2 = top_k_post_rewrite_query_en
        self.pretty.write(
            "I",
            "LangDrift",
            f"retrieval_top_k_orig_query_en={top_k_orig_query_en}  "
            f"retrieval_top_k_post_rewrite_query_en={top_k_post_rewrite_query_en}",
        )

    def _fetch_local_docs(
        self,
        mySession: Session,
        retrieve_mode: str,
        bm25_query: str,
        alternate_queries: list[str],
        orig_translated_query_en: str = "",
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Delegate local retrieval orchestration to Orchestrator."""
        return self.orchestrator.run_local_retrievers(
            mySession,
            retrieve_mode,
            bm25_query,
            alternate_queries,
            orig_translated_query_en,
        )
