"""Extracted RetrievalReranker behavior; the coordinating facade owns shared state."""

from __future__ import annotations

import time
from typing import Any

from Chat.RagRetrievalContext import RagRetrievalContext
from Commons.Exceptions import RerankError
from Globals.Session import Session
from Gui.Colors import CYAN, GREEN, RED, YELLOW
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import (align_retriever_sources_for_print,
                             truncate_for_print)
from Orchestrator.OrchestratorHelper import OrchestratorHelper


class RetrievalReranker(RagRetrievalContext):
    def _get_rerank_language_queries(self) -> dict[str, str]:
        raw_queries = getattr(self.orchestrator, "_rerank_language_queries", [])
        queries_by_language: dict[str, str] = {}
        if isinstance(raw_queries, list):
            for item in raw_queries:
                if not isinstance(item, dict):
                    continue
                language = self._normalize_language_bucket(item.get("language"))
                query = str(item.get("query", "") or "").strip()
                if language and query and language not in queries_by_language:
                    queries_by_language[language] = query
        return queries_by_language

    def _predict_rerank_scores(
        self,
        candidates: list[Any],
        pairs: list[tuple[str, str]],
        queries_by_language: dict[str, str] | None = None,
    ) -> list[float]:
        """Score language-matched groups; log yellow inputs and raw-score summaries."""
        if queries_by_language is None:
            queries_by_language = self._get_rerank_language_queries()

        if not queries_by_language:
            scores = self.x_encoder.predict(pairs, show_progress_bar=False)
            if len(scores) != len(candidates):
                raise ValueError(
                    f"Reranker returned {len(scores)} scores for "
                    f"{len(candidates)} candidates"
                )
            return [float(score) for score in scores]

        query_instruction = str(
            getattr(self, "cross_encoder_query_instruction", "") or ""
        )
        groups: dict[tuple[str, str], list[int]] = {}
        for index, doc in enumerate(candidates):
            metadata = getattr(doc, "metadata", {}) or {}
            if str(metadata.get("Source", "")).strip().lower() == "web":
                groups.setdefault(("web", pairs[index][0]), []).append(index)
                continue
            language = self._normalize_language_bucket(metadata.get("Language"))
            language_query = queries_by_language.get(language or "")
            if language_query:
                query_text = query_instruction + language_query
                group_language = language or "unknown"
            else:
                query_text = pairs[index][0]
                group_language = "legacy-fallback"
            groups.setdefault((group_language, query_text), []).append(index)

        scores_by_index = [0.0] * len(candidates)
        fallback_count = 0
        summary_rows: list[str] = []
        for (language, query_text), indexes in groups.items():
            if language == "legacy-fallback":
                fallback_count += len(indexes)
            self.pretty.write(
                "I",
                "Rerank",
                f"Passing language={language} chunks={len(indexes)} "
                f"to reranker; query={query_text!r}",
                color=YELLOW,
            )
            grouped_pairs = [(query_text, pairs[index][1]) for index in indexes]
            grouped_scores = self.x_encoder.predict(
                grouped_pairs,
                show_progress_bar=False,
            )
            if len(grouped_scores) != len(indexes):
                raise ValueError(
                    f"Reranker for language '{language}' returned "
                    f"{len(grouped_scores)} scores for {len(indexes)} candidates"
                )
            for index, score in zip(indexes, grouped_scores):
                scores_by_index[index] = float(score)

            ranking = sorted(
                indexes, key=lambda index: scores_by_index[index], reverse=True
            )
            sorted_scores = [scores_by_index[index] for index in ranking]
            top_three = sorted_scores[:3]
            summary_rows.append(
                f"{language:<18} {len(indexes):>6} "
                f"{sorted_scores[0]:>10.4f} "
                f"{sum(top_three) / len(top_three):>10.4f} "
                f"{sum(sorted_scores) / len(sorted_scores):>10.4f}"
            )
            top_results = ", ".join(
                f"{rank}:{candidates[index].metadata.get('FileName', '<unknown>')} "
                f"(raw={scores_by_index[index]:.4f})"
                for rank, index in enumerate(ranking[:5], start=1)
            )
            self.pretty.write(
                "I",
                "Rerank",
                f"language={language} top5=[{top_results}]",
                color=CYAN,
            )

        self.pretty.write(
            "I",
            "Rerank summary",
            f"{'Language':<18} {'Chunks':>6} {'Top raw':>10} "
            f"{'Mean top3':>10} {'Mean raw':>10}",
            color=CYAN,
        )
        for row in summary_rows:
            self.pretty.write("I", "Rerank summary", row, color=CYAN)

        if fallback_count:
            self.pretty.write(
                "W",
                "Rerank",
                f"{fallback_count} candidate(s) had no "
                "matching language query; used the existing rerank query.",
                color=CYAN,
            )
        return scores_by_index

    def _select_best_rerank_language(
        self,
        candidates: list[Any],
        raw_scores: list[float],
    ) -> tuple[list[Any], list[float], str | None]:
        """Keep the best local language group plus independently scored web results."""
        if len(candidates) != len(raw_scores):
            raise ValueError(
                f"Cannot select a rerank language from {len(candidates)} candidates "
                f"and {len(raw_scores)} scores"
            )
        query_languages = self._get_rerank_language_queries()
        if not query_languages:
            return candidates, raw_scores, None

        best_by_language: dict[str, tuple[float, int]] = {}
        for index, (doc, score) in enumerate(zip(candidates, raw_scores)):
            metadata = getattr(doc, "metadata", {}) or {}
            if str(metadata.get("Source", "")).strip().lower() == "web":
                continue
            language = self._normalize_language_bucket(metadata.get("Language"))
            if language not in query_languages:
                continue
            current_best = best_by_language.get(language)
            if current_best is None or float(score) > current_best[0]:
                best_by_language[language] = (float(score), index)

        if not best_by_language:
            if all(
                str((getattr(doc, "metadata", {}) or {}).get("Source", ""))
                .strip()
                .lower()
                == "web"
                for doc in candidates
            ):
                return candidates, raw_scores, None
            self.pretty.write(
                "W",
                "Rerank",
                "No candidates matched the available language queries; keeping "
                "the legacy reranker pool.",
                color=CYAN,
            )
            return candidates, raw_scores, None

        selected_language, (top_score, top_index) = max(
            best_by_language.items(),
            key=lambda item: item[1][0],
        )
        selected_indexes = [
            index
            for index, doc in enumerate(candidates)
            if str((getattr(doc, "metadata", {}) or {}).get("Source", ""))
            .strip()
            .lower()
            == "web"
            or self._normalize_language_bucket(
                (getattr(doc, "metadata", {}) or {}).get("Language")
            )
            == selected_language
        ]
        selected_docs = [candidates[index] for index in selected_indexes]
        selected_scores = [raw_scores[index] for index in selected_indexes]
        web_count = sum(
            str((getattr(doc, "metadata", {}) or {}).get("Source", "")).strip().lower()
            == "web"
            for doc in selected_docs
        )
        top_doc = candidates[top_index]
        filename = str(
            (getattr(top_doc, "metadata", {}) or {}).get("FileName", "<unknown>")
        )
        self.pretty.write(
            "I",
            "Rerank",
            f"Selected language={selected_language} by highest top raw reranker "
            f"score={top_score:.4f}; top chunk={filename!r}; "
            f"passing {len(selected_docs) - web_count} matching-language local "
            f"candidates + {web_count} web candidates to chunk selection.",
            color=GREEN,
        )
        return selected_docs, selected_scores, selected_language

    def _predict_keyword_action_scores(
        self, mySession: Session, candidates: list[Any], pairs: list[tuple[str, str]]
    ) -> list[float]:
        split = mySession.keyword_action_split
        if split is None:
            raise ValueError("Decomposed reranking requires a split")
        helper = OrchestratorHelper()
        # Keep retrieval-time associations fixed while scores add diagnostic
        # associations, so a keyword-only hit is still scored for every need.
        original_query_ids = {
            identity: list(ids)
            for identity, ids in mySession.keyword_action_query_ids.items()
        }
        best_scores = [float("-inf")] * len(candidates)
        for query_id, query in enumerate(split.retrieval_queries):
            # Score associated local chunks against their information need.
            # Unassociated keyword candidates are evaluated against every need.
            indexes = [
                index
                for index, doc in enumerate(candidates)
                if str(doc.metadata.get("Source", "")).lower() != "web"
                and (
                    not original_query_ids.get(helper.doc_identity(doc))
                    or query_id in original_query_ids[helper.doc_identity(doc)]
                )
            ]
            if not indexes:
                continue
            query_text = self.cross_encoder_query_instruction + query
            scores = self._predict_rerank_scores(
                [candidates[index] for index in indexes],
                [(query_text, pairs[index][1]) for index in indexes],
                queries_by_language=mySession.keyword_action_language_queries.get(
                    query_id, {}
                ),
            )
            for index, score in zip(indexes, scores):
                identity = helper.doc_identity(candidates[index])
                mySession.keyword_action_scores.setdefault(identity, {})[
                    query_id
                ] = score
                ids = mySession.keyword_action_query_ids.setdefault(identity, [])
                if query_id not in ids:
                    ids.append(query_id)
                # Existing thresholding expects one raw score per document.
                # Keep its best relevance score, retaining per-need scores
                # separately for query-specific diagnostics.
                best_scores[index] = max(best_scores[index], score)
        web_indexes = [
            index
            for index, doc in enumerate(candidates)
            if str(doc.metadata.get("Source", "")).lower() == "web"
        ]
        if web_indexes:
            # Web retrieval and web reranking retain the existing full query.
            scores = self._predict_rerank_scores(
                [candidates[index] for index in web_indexes],
                [pairs[index] for index in web_indexes],
                queries_by_language={},
            )
            for index, score in zip(web_indexes, scores):
                best_scores[index] = score
        return best_scores

    def _rerank(self, mySession: Session, all_docs: list[Any]) -> list[Any]:
        self.perf_logger.log(
            "RAGChatImpl._rerank",
            "chat",
            f"start rerank pairs={len(all_docs)} model={self.cross_encoder_model_name!r}",
        )
        _t_rerank = time.perf_counter()
        # Prepare query-document pairs for re-ranking.
        # For web documents, use the original search-engine snippet rather than
        # doc.page_content for the cross-encoder input.  When fetch_page_content
        # is active, page_content is the full fetched page — often thousands of
        # tokens — and the cross-encoder's 512-token truncation window is filled
        # by navigation headers and boilerplate before the relevant content is
        # reached.  The snippet (stored in metadata["snippet"]) is the concise
        # search-engine description that scores reliably for relevance.
        # The full page_content is untouched and still goes to the LLM prompt.
        candidates: list[Any] = all_docs
        pairs: list[tuple[str, str]] = []
        # sentence_transformers CrossEncoder always expects (str_A, str_B) pairs;
        # the tokenizer handles [CLS] query [SEP] passage [SEP] formatting.
        # Rerankers that document a query prefix can set QUERY_INSTRUCTION in
        # the _CROSS model config entry.  Standard cross-encoders leave it empty.
        query_text = getattr(mySession, "rerank_query", None) or mySession.query or ""
        if self.cross_encoder_query_instruction and query_text:
            query_text = self.cross_encoder_query_instruction + query_text
        self.pretty.write("I", "Rerank", f"Rerank query: {query_text!r}")
        for doc in candidates:
            chunk_text = (
                doc.metadata["snippet"]
                if doc.metadata.get("Source") == "Web" and doc.metadata.get("snippet")
                else doc.page_content
            )
            pairs.append((query_text, chunk_text))

        if not pairs:
            self.pretty.write(
                "W",
                "Rerank",
                f"Reranking with {self.cross_encoder_model_name} returned {len(pairs)} chunks",
            )
            return pairs

        # Get re-ranking scores from cross-encoder model
        try:
            raw_rerank = (
                self._predict_keyword_action_scores(mySession, candidates, pairs)
                if getattr(mySession, "keyword_action_split", None) is not None
                else self._predict_rerank_scores(candidates, pairs)
            )
        except RuntimeError as e:
            self.pretty.write(
                "E",
                "Rerank",
                f"Reranking failed due to a model input mismatch. "
                f"Did you run Load.py with a different _CHROMA_EMBED_PARAMS.CHUNK_SIZE than RAGChat.py is using now? "
                f"Details: {e}",
                color=RED,
            )
            raise RerankError

        # Credits: Fix according to input from Don Karter (u/donk8r on Reddit).
        # Before fix: We only hire people if they are among the best applicants in today's candidate pool.
        # So an identical candidate may pass on Tuesday and fail on Wednesday.
        # Fix: We hire anyone scoring above 80. Then rank hired candidates afterward.
        #
        # Score normalization — plain min-max over the active pool. In the
        # language-aware path, this is only the selected language group.
        #
        #   rerank_score = (raw − pool_lo) / (pool_hi − pool_lo)
        #
        # rerank_score is used ONLY for intra-query ordering.  It is deliberately
        # query-relative: the best chunk in any pool approaches 1.0, so it must
        # not be used for absolute keep/drop decisions.
        #
        # Threshold decisions use raw_rerank_score — the cross-encoder logit —
        # while language selection compares the top raw logit in each language.
        # See HomeBrewChunkSelector.filter_threshold.
        #
        # Web docs are multiplied by web_wt for ordering so local knowledge
        # retains a natural edge.  In web-only mode the multiplier is skipped
        # (it would uniformly lower all scores without changing their order).

        # Resolve the web weight from the session override or the global config default.
        web_wt: float = float(
            mySession.web_weight
            if mySession.web_weight is not None
            else (self.cfg.get_float("_WEB_SEARCH.default_web_weight") or 0.5)
        )

        # Store raw cross-encoder logits; compute pool min/max for ordering.
        for i, doc in enumerate(candidates):
            doc.metadata["raw_rerank_score"] = float(raw_rerank[i])

        candidates, raw_rerank, _ = self._select_best_rerank_language(
            candidates,
            [float(score) for score in raw_rerank],
        )

        all_raw_scores = [float(raw_rerank[i]) for i in range(len(candidates))]
        lo = min(all_raw_scores) if all_raw_scores else 0.0
        hi = max(all_raw_scores) if all_raw_scores else 1.0
        full_range = hi - lo if hi != lo else 1.0

        # Assign rerank_score: plain min-max for relative ordering only.
        # Web docs are additionally scaled by web_wt to let local knowledge retain
        # a natural edge when both sources are present.  In web-only mode (all docs
        # are web) the multiplier is skipped — there is no local content to protect
        # and applying it would only uniformly lower all scores without changing order.
        all_web: bool = bool(candidates) and all(
            d.metadata.get("Source") == "Web" for d in candidates
        )
        for i, doc in enumerate(candidates):
            raw = float(raw_rerank[i])
            normalized = (raw - lo) / full_range
            if doc.metadata.get("Source") == "Web" and not all_web:
                doc.metadata["rerank_score"] = web_wt * normalized
            else:
                doc.metadata["rerank_score"] = normalized

        # Sort documents by combined score in descending order
        reranked: list[Any] = sorted(
            candidates, key=lambda d: d.metadata["rerank_score"], reverse=True  # type: ignore[reportUnknownLambdaType, reportUnknownMemberType]
        )

        # debug print
        if DebugHelper.check_session(mySession, 10):
            header = "{:>6}  {:>10}  {:>10}  {:>24}  {:<40}  {}"
            row = "{:>6}  {:>10.4f}  {:>10.4f}  {:>24}  {:<40}  {}"

            # Print header row — RawScore is the cross-encoder output; AdjScore is
            # the pool-normalized value (web docs also scaled by web_weight).
            self.pretty.write(
                "D",
                "Rerank",
                header.format(
                    "Pos", "RawScore", "AdjScore", "Retrievers", "File", "Text"
                ),
                color=CYAN,
            )
            self.pretty.write("D", "Rerank", "-" * 110, color=CYAN)

            # Print each row aligned with header
            for i, d in enumerate(reranked[: mySession.final_chunks_to_llm]):
                file_name: str = truncate_for_print(
                    str(d.metadata.get("FileName", d.metadata.get("source", ""))), 40
                )
                sources: str = align_retriever_sources_for_print(
                    str(d.metadata.get("retriever_sources", "")), 24
                )
                # Show the text actually fed to the cross-encoder: snippet for
                # web docs (prefixed [S]), page_content for all others.
                scoring_text: str = (
                    "[S] " + d.metadata["snippet"]
                    if d.metadata.get("Source") == "Web" and d.metadata.get("snippet")
                    else d.page_content
                )
                self.pretty.write(
                    "D",
                    "Rerank",
                    row.format(
                        i + 1,
                        d.metadata.get("raw_rerank_score", 0.0),
                        d.metadata["rerank_score"],
                        sources,
                        file_name,
                        scoring_text[:40],  # truncate text to fit
                    ),
                )

        self.pretty.write(
            "I",
            "Rerank",
            f"Reranking with {self.cross_encoder_model_name} returned {len(reranked)} chunks",
            color=CYAN,
        )
        self.perf_logger.log(
            "RAGChatImpl._rerank",
            "chat",
            f"stop  rerank n={len(reranked)} elapsed={time.perf_counter() - _t_rerank:.3f}s",
        )
        return reranked
