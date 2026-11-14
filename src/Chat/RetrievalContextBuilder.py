"""Extracted RetrievalContextBuilder behavior; the coordinating facade owns shared state."""

from __future__ import annotations

from typing import Any, Tuple, cast

from Chat.RagRetrievalContext import RagRetrievalContext
from Globals.Session import Session
from Gui.Colors import CYAN
from Helpers.DebugHelper import DebugHelper
from Orchestrator.OrchestratorHelper import OrchestratorHelper
from Retrievers.BM25Retriever import BM25Retriever
from Strategies.HomeBrewChunkSelector import ChunkSelectionService


class RetrievalContextBuilder(RagRetrievalContext):
    def _merge_local_docs_by_language(
        self,
        local_active: list[list[Any]],
        labels: list[str],
        weights: list[float],
        *,
        rrf_k: int,
        limit: int | None,
    ) -> list[Any] | None:
        """Rank-fuse each language's results independently, then interleave."""
        language_queries = getattr(self.orchestrator, "_rerank_language_queries", [])
        if not isinstance(language_queries, list) or not language_queries:
            return None
        if not any(
            isinstance(item, dict)
            and self._normalize_language_bucket(item.get("language"))
            and str(item.get("query", "") or "").strip()
            for item in language_queries
        ):
            return None
        if limit is not None and limit <= 0:
            return []

        grouped_sources: dict[str, list[list[Any]]] = {}
        for source_index, docs in enumerate(local_active):
            for doc in docs:
                metadata = getattr(doc, "metadata", {}) or {}
                language = (
                    self._normalize_language_bucket(metadata.get("Language"))
                    or "unknown"
                )
                if language not in grouped_sources:
                    grouped_sources[language] = [[] for _ in range(len(local_active))]
                grouped_sources[language][source_index].append(doc)

        fused_by_language: dict[str, list[Any]] = {}
        for language, source_groups in grouped_sources.items():
            active_indexes = [index for index, docs in enumerate(source_groups) if docs]
            if not active_indexes:
                continue
            language_sources = [source_groups[index] for index in active_indexes]
            language_labels = [labels[index] for index in active_indexes]
            language_weights = [weights[index] for index in active_indexes]
            if len(language_sources) > 1:
                fused = BM25Retriever.reciprocal_rank_fusion(
                    *language_sources,
                    k=rrf_k,
                    labels=language_labels,
                    weights=language_weights,
                )
            else:
                # Even one source needs rank-based scores here: different
                # languages may have single-source results from different
                # retrievers, whose raw scores are not on a shared scale.
                fused = BM25Retriever.reciprocal_rank_fusion(
                    language_sources[0],
                    k=rrf_k,
                    labels=language_labels,
                    weights=language_weights,
                )
            fused_by_language[language] = fused

        interleaved: list[Any] = []
        max_language_results = max(
            (len(docs) for docs in fused_by_language.values()),
            default=0,
        )
        for rank in range(max_language_results):
            for docs in fused_by_language.values():
                if rank < len(docs):
                    interleaved.append(docs[rank])
                    if limit is not None and len(interleaved) >= limit:
                        break
            if limit is not None and len(interleaved) >= limit:
                break

        self.pretty.write(
            "I",
            "Merge",
            "Language-scoped RRF candidate pool "
            f"languages={len(fused_by_language)} "
            f"per_language={ {language: len(docs) for language, docs in fused_by_language.items()} } "
            f"interleaved={len(interleaved)}/{limit}",
            color=CYAN,
        )
        return interleaved

    def _merge_and_select(
        self,
        mySession: Session,
        vector_docs: list[Any],
        bm25_docs: list[Any],
        graph_docs: list[Any],
        regex_docs: list[Any],
        web_docs: list[Any],
    ) -> list[Any]:
        """RRF-fuse local docs, cap, append web docs, dedup, rerank, and select.

        Returns the final chosen list.
        """
        # Local retrievers (Vector, BM25, Graph, Regex) are fused via RRF and then
        # capped to retriever_k.  Web docs are appended AFTER the cap so they
        # always reach the reranker.  Including web in the same RRF pool caused
        # them to be pushed off the list: with weight=0.5 and only 5 results
        # their best RRF score (≈0.008) falls below the 100th local slot (≈0.011).
        local_sources = [
            (vector_docs, "Vector"),
            (bm25_docs, "BM25"),
            (graph_docs, "Graph"),
            (regex_docs, "Regex"),
        ]
        local_active_labeled = [(d, lbl) for d, lbl in local_sources if d]
        local_active = [d for d, _ in local_active_labeled]
        local_active_labels = [lbl for _, lbl in local_active_labeled]
        split_enabled = getattr(mySession, "keyword_action_split", None) is not None
        local_weight_map = {
            "Vector": float(
                mySession.vector_weight if mySession.vector_weight is not None else 1.0
            ),
            "BM25": float(
                mySession.bm25_weight if mySession.bm25_weight is not None else 1.0
            ),
            "Graph": float(
                mySession.graph_weight if mySession.graph_weight is not None else 1.0
            ),
            "Regex": float(
                mySession.regex_weight if mySession.regex_weight is not None else 1.0
            ),
        }
        local_weights = [local_weight_map.get(lbl, 1.0) for lbl in local_active_labels]
        rerank_flow_control = getattr(mySession, "enable_rerank", None)
        rerank_flow_enabled = (
            True if rerank_flow_control is None else bool(rerank_flow_control)
        )
        language_scoped_docs = (
            self._merge_local_docs_by_language(
                local_active,
                local_active_labels,
                local_weights,
                rrf_k=self.bm25_retriever.rrf_k,
                limit=mySession.retriever_k,
            )
            if mySession.rerank == 1 and rerank_flow_enabled and local_active
            else None
        )
        if language_scoped_docs is not None:
            capped_docs = language_scoped_docs
            if DebugHelper.check_session(mySession, 10):
                self._print_merged_debug(capped_docs)
        elif len(local_active) > 1:
            capped_docs = BM25Retriever.reciprocal_rank_fusion(
                *local_active,
                k=self.bm25_retriever.rrf_k,
                labels=local_active_labels,
                weights=local_weights,
            )
            capped_docs = capped_docs[: mySession.retriever_k]
            self.pretty.write(
                "O",
                "Merge",
                f"Reciprocal Rank Fusion (RRF) produced {len(capped_docs)} local chunks",
            )
            if DebugHelper.check_session(mySession, 10):
                self._print_merged_debug(capped_docs)
        elif local_active:
            capped_docs = local_active[0][: mySession.retriever_k]
            # Stamp retriever_sources so rerank/selection debug prints show the origin.
            src = local_active_labels[0]
            for doc in capped_docs:
                doc.metadata["retriever_sources"] = src
        else:
            capped_docs = []

        if web_docs:
            capped_docs = capped_docs + web_docs
            self.pretty.write(
                "O",
                "Merge",
                f"Appended {len(web_docs)} web result(s) → reranker pool: {len(capped_docs)} chunks",
            )

        dedup_enabled: bool = self.cfg.get_bool("_CHUNK_DEDUP.enabled")
        dedup_threshold: float = self.cfg.get_float("_CHUNK_DEDUP.threshold") or 0.85
        dedup_include_web: bool = self.cfg.get_bool(
            "_CHUNK_DEDUP.include_web_chunks",
            True,
        )
        if dedup_enabled and capped_docs:
            before_dedup = len(capped_docs)
            if dedup_include_web:
                capped_docs = self._remove_similar_chunks(capped_docs, dedup_threshold)
            else:
                local_docs: list[Any] = []
                web_docs_only: list[Any] = []
                for doc in capped_docs:
                    meta = getattr(doc, "metadata", {}) or {}
                    source = str(meta.get("Source", "") or "").strip().lower()
                    retrievers = str(meta.get("retriever_sources", "") or "").lower()
                    if source == "web" or "web" in retrievers:
                        web_docs_only.append(doc)
                    else:
                        local_docs.append(doc)
                deduped_local = self._remove_similar_chunks(local_docs, dedup_threshold)
                capped_docs = deduped_local + web_docs_only

            dropped = before_dedup - len(capped_docs)
            if dropped > 0 and DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "I",
                    "ChunkDedup",
                    f"Removed {dropped} near-duplicate chunk(s) "
                    f"(threshold={dedup_threshold:.2f}, include_web={'on' if dedup_include_web else 'off'}, kept {len(capped_docs)})",
                    color=CYAN,
                )

        if mySession.rerank == 1 and rerank_flow_enabled:
            web_count = sum(
                1
                for doc in capped_docs
                if str((getattr(doc, "metadata", {}) or {}).get("Source", "")).lower()
                == "web"
            )
            local_count = len(capped_docs) - web_count
            original_leg_enabled = bool(
                getattr(mySession, "original_query_leg_enabled", False)
            )
            original_leg_lang = str(
                getattr(mySession, "original_query_leg_language", None) or "n/a"
            )
            original_leg_vector_added = int(
                getattr(mySession, "original_query_leg_vector_added", 0) or 0
            )
            original_leg_bm25_added = int(
                getattr(mySession, "original_query_leg_bm25_added", 0) or 0
            )
            original_leg_graph_added = int(
                getattr(mySession, "original_query_leg_graph_added", 0) or 0
            )
            original_leg_regex_added = int(
                getattr(mySession, "original_query_leg_regex_added", 0) or 0
            )
            self.pretty.write(
                "I",
                "Rerank",
                "Pre-rerank pool "
                f"local={local_count} web={web_count} total={len(capped_docs)} "
                f"original_leg={'on' if original_leg_enabled else 'off'} "
                f"lang={original_leg_lang} "
                f"added[V/B/G/R]={original_leg_vector_added}/"
                f"{original_leg_bm25_added}/{original_leg_graph_added}/"
                f"{original_leg_regex_added}",
                color=CYAN,
            )
            capped_docs = self._rerank(mySession, capped_docs)

        chosen: list[Any] = cast(list[Any], ChunkSelectionService(mySession).select_chunks(capped_docs))  # type: ignore[reportUnknownMemberType]

        if split_enabled and DebugHelper.check_session(mySession, 30):
            helper = OrchestratorHelper()
            split = mySession.keyword_action_split
            if split is not None:
                for query_id, query in enumerate(split.retrieval_queries):
                    count = sum(
                        query_id
                        in mySession.keyword_action_query_ids.get(
                            helper.doc_identity(doc), []
                        )
                        for doc in chosen
                    )
                    self.pretty.write(
                        "D",
                        "KeywordActionSplitter",
                        f"Final context query_id={query_id}: {count} associated chunks for {query!r} (association is not proof of answer support)",
                    )
        if DebugHelper.check_session(mySession, 30):
            self.pretty.write(
                "D",
                "ChunkSelect",
                f"After chunk selection: {len(chosen)}/{len(capped_docs)} kept",
                color=CYAN,
            )

        return chosen

    def _build_context(self, mySession: Session, chosen: list[Any]) -> Tuple[str, int]:
        """Populate session grounding fields and format the LLM context string.

        Returns (context, len(chosen)), or ("", 0) when chosen is empty.
        """
        mySession.chunk_texts_for_grounding = (
            [
                getattr(doc, "page_content", "") or ""
                for doc in chosen
                if str((getattr(doc, "metadata", {}) or {}).get("Source", "")).lower()
                != "web"
            ]
            if chosen
            else []
        )

        mySession.last_chosen_chunks = chosen
        self.confidenceEvaluator.store_answer_confidence(mySession, chosen)

        # Pre-compute distinct FileNames/URLs so weak LLMs can't skip sources.
        seen_local: set[str] = set()
        distinct_local: list[str] = []
        seen_web: set[str] = set()
        distinct_web: list[str] = []
        for d in chosen:
            if d.metadata.get("Source") == "Web":
                fp = str(d.metadata.get("FilePath", "")).strip()
                if fp and fp not in seen_web:
                    seen_web.add(fp)
                    distinct_web.append(fp)
            else:
                fn = str(d.metadata.get("FileName", "")).strip()
                if fn and fn not in seen_local:
                    seen_local.add(fn)
                    distinct_local.append(fn)

        header_parts: list[str] = []
        if distinct_local:
            header_parts.append(
                f"LOCAL SOURCE FILES ({len(distinct_local)} files \u2014 you MUST consider every one of them and MUST include any that contain relevant information in your Sources section):\n"
                + "\n".join(f"  - {fn}" for fn in distinct_local)
            )
        if distinct_web:
            header_parts.append(
                f"WEB SOURCES ({len(distinct_web)} URLs \u2014 treat as supplementary internet context and MUST include any that contain relevant information in your Sources section):\n"
                + "\n".join(f"  - {url}" for url in distinct_web)
            )
        header: str = "\n\n".join(header_parts) + "\n\n" if header_parts else ""

        body: str = "\n\n".join(
            self.helperInstance.format_document(doc) for doc in chosen  # type: ignore[reportUnknownMemberType]
        )
        context: str = header + body
        if chosen:
            self.perf_logger.log(
                "RAGChatImpl._retrieve", "chat", f"stop  retrieve n={len(chosen)}"
            )
            return context, len(chosen)
        return "", 0

    def _remove_similar_chunks(self, docs: list[Any], threshold: float) -> list[Any]:
        """Remove near-duplicate chunks using Jaccard similarity on word tokens.

        Iterates *docs* in ranked order (highest rank first).  For each
        candidate, compares it against all already-kept chunks; if the Jaccard
        similarity of their lowercased word tokens meets *threshold*, the
        candidate is discarded.  The kept list preserves the original ranking.
        """
        kept: list[Any] = []
        kept_tokens: list[list[str]] = []

        for doc in docs:
            tokens: list[str] = doc.page_content.lower().split()
            is_dup = any(
                self._shared.jaccard(tokens, kt) >= threshold for kt in kept_tokens
            )
            if not is_dup:
                kept.append(doc)
                kept_tokens.append(tokens)

        return kept
