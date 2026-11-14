"""Extracted QueryNormalizer behavior; the coordinating facade owns shared state."""

from __future__ import annotations

from typing import Any

from Chat.RagRetrievalContext import RagRetrievalContext
from Globals.Session import Session
from Gui.Colors import CYAN
from Helpers.DebugHelper import DebugHelper


class QueryNormalizer(RagRetrievalContext):
    def _get_translator(self, backend: str) -> Any:
        """Return translator for *backend*.

        Supported values:
        - ``"argos"``
        - ``"off"``

        Unknown or ``"off"`` values return ``None`` so callers can skip
        translation with a simple ``if translator is not None`` guard.
        """
        b = (backend or "off").lower()
        if b == "argos":
            return self._shared
        return None

    def _drain_lang_detection_confidence_events(self, mySession: Session) -> None:
        """Forward buffered language-detection confidence events to Session telemetry."""
        events = self._fileUtils.pop_lang_detection_events()
        for event in events:
            self.confidence_logger.log_language_detection(mySession, event)

    def _resolve_turn_translation_backend(self, mySession: Session) -> tuple[str, Any]:
        """Resolve the per-turn translation backend and translator instance."""
        backend: str = (
            getattr(mySession, "translation_backend", None)
            or self._translation_backend
            or "off"
        ).lower()
        return backend, self._get_translator(backend)

    def _detect_raw_query_language(self, mySession: Session, raw_query: str) -> str:
        """Detect and persist the user query language for this turn."""
        raw_query_language: str = (
            self._fileUtils.get_user_text_language(
                raw_query,
                output="nltk",
                native_lang=mySession.preferred_response_language,
                stage_label="raw_user_query",
            )
            if raw_query
            else "english"
        )
        self._drain_lang_detection_confidence_events(mySession)
        # Keep chat-history language filtering on the user language, not the
        # retrieval language, so language-specific contexts remain isolated.
        mySession.user_language = raw_query_language
        mySession.current_query_lang = raw_query_language
        mySession.retrieval_language = (
            raw_query_language if self._is_native_retrieval(mySession) else "english"
        )
        return raw_query_language

    @staticmethod
    def _is_native_retrieval(mySession: Session) -> bool:
        """True when the flow keeps the query in the user's language."""
        mode = str(getattr(mySession, "retrieval_language_mode", "") or "").upper()
        return mode == "NATIVE"

    def _translate_query_to_english(
        self,
        query_text: str,
        source_language: str,
        translator: Any,
        backend: str,
        log_prefix: str,
    ) -> tuple[str, bool]:
        """Translate query_text to English when required; return (text, translated?)."""
        if translator is None or not query_text or source_language == "english":
            return query_text, False

        translated: str = translator.translate_text(
            query_text,
            target_lang="en",
            source_lang=source_language,
        )
        if translated and translated != query_text:
            self.pretty.write(
                "I",
                "QueryNorm",
                f"{log_prefix} [{source_language}→english, "
                f"backend={backend}]: "
                f"{query_text!r} → {translated!r}",
            )
            return translated, True
        return query_text, False

    def _rewrite_query_with_strict_retry(
        self,
        mySession: Session,
    ) -> tuple[str, bool]:
        """Rewrite the query and retry once with strict-English instructions."""
        pre_rewrite_query: str = mySession.query or ""
        if mySession.use_chat_context:
            mySession.query = self.promptRewrite.rewrite(mySession)

        rewritten_query: str = mySession.query or ""
        was_rewritten: bool = rewritten_query != pre_rewrite_query

        rewrite_language: str = (
            self._fileUtils.get_user_text_language(
                rewritten_query,
                output="nltk",
                native_lang=None,
                stage_label="rewrite_query",
            )
            if rewritten_query
            else "english"
        )
        self._drain_lang_detection_confidence_events(mySession)
        mySession.rewritten_query = rewritten_query
        mySession.rewrite_language = rewrite_language

        native_mode: bool = self._is_native_retrieval(mySession)
        target_language: str = (
            str(getattr(mySession, "retrieval_language", "") or "english")
            if native_mode
            else "english"
        )

        # If rewrite drifted out of the retrieval language, retry with stricter
        # instructions before falling back to translation.
        if (
            rewrite_language != target_language
            and mySession.use_chat_context
            and not mySession.force_skip_rewrite
        ):
            self.confidence_logger.log_step(
                mySession,
                step_name="strict rewrite retry",
                status="executing",
                confidence_level="PENDING",
                detail=(
                    f"rewrite output is not {target_language}; retrying with strict prompt"
                ),
                source="query_rewrite",
            )
            strict_candidate: str = (
                (
                    self.promptRewrite.rewrite(
                        mySession, strict_language=target_language
                    )
                    if native_mode
                    else self.promptRewrite.rewrite(mySession, strict_english=True)
                )
                or ""
            ).strip()
            if strict_candidate:
                strict_lang: str = self._fileUtils.get_user_text_language(
                    strict_candidate,
                    output="nltk",
                    native_lang=None,
                    stage_label="strict_rewrite_candidate",
                )
                self._drain_lang_detection_confidence_events(mySession)
                if strict_lang == target_language:
                    mySession.query = strict_candidate
                    rewritten_query = strict_candidate
                    rewrite_language = strict_lang
                    mySession.rewritten_query = rewritten_query
                    mySession.rewrite_language = rewrite_language
                    was_rewritten = rewritten_query != pre_rewrite_query
                    self.pretty.write(
                        "I",
                        "QueryNorm",
                        f"Strict {target_language} rewrite retry succeeded.",
                    )
                    self.confidence_logger.log_step(
                        mySession,
                        step_name="strict rewrite retry",
                        status="executed",
                        confidence_level="HIGH",
                        confidence_score=1.0,
                        detail=f"strict retry produced {target_language} rewrite",
                        source="query_rewrite",
                    )
                else:
                    self.confidence_logger.log_step(
                        mySession,
                        step_name="strict rewrite retry",
                        status="executed",
                        confidence_level="LOW",
                        confidence_score=0.0,
                        detail=f"strict retry did not produce {target_language} rewrite",
                        source="query_rewrite",
                    )
            else:
                self.confidence_logger.log_step(
                    mySession,
                    step_name="strict rewrite retry",
                    status="skipped",
                    confidence_level="N/A",
                    detail="strict retry returned empty candidate",
                    source="query_rewrite",
                )
        else:
            self.confidence_logger.log_step(
                mySession,
                step_name="strict rewrite retry",
                status="skipped",
                confidence_level="N/A",
                detail="conditions not met (rewrite already English or retry disabled)",
                source="query_rewrite",
            )

        return rewrite_language, was_rewritten

    def _apply_post_rewrite_translation(
        self,
        mySession: Session,
        rewrite_language: str,
        translator: Any,
        backend: str,
    ) -> bool:
        """Enforce English after rewrite; return whether a translation happened."""
        if rewrite_language == "english":
            return False

        if translator is not None and mySession.query:
            translated_after, translated_flag = self._translate_query_to_english(
                mySession.query,
                rewrite_language,
                translator,
                backend,
                "Normalized rewritten query",
            )
            if translated_flag:
                mySession.query = translated_after
            return translated_flag

        self.pretty.write(
            "W",
            "QueryNorm",
            "Rewrite language is non-English but no translator is active; "
            "retrieval query may stay cross-lingual.",
        )
        return False

    def _log_query_language_drift(
        self,
        mySession: Session,
        raw_query_language: str,
        orig_translated_query_en: str,
    ) -> None:
        """Emit standardized language-drift debug lines for retrieval stages."""
        self.pretty.write(
            "I",
            "LangDrift",
            f"raw_query_language={raw_query_language}  "
            f"orig_translated_query_en={orig_translated_query_en!r}",
        )
        self.pretty.write(
            "I",
            "LangDrift",
            f"rewrite_language={mySession.rewrite_language}  "
            f"rewritten_query={mySession.rewritten_query!r}",
        )
        self.pretty.write(
            "I",
            "LangDrift",
            f"post_rewrite_query_en={mySession.post_rewrite_query_en!r}",
        )

    def _set_effective_query_metadata(
        self,
        mySession: Session,
        final_query: str,
        user_query_original: str,
        was_translated: bool,
        was_rewritten: bool,
    ) -> None:
        """Persist effective-query fields and emit the FinalQuery log line."""
        if final_query != user_query_original:
            mySession.effective_query = final_query
            if was_translated and was_rewritten:
                mySession.effective_query_reason = "translated+rewritten"
            elif was_translated:
                mySession.effective_query_reason = "translated"
            elif was_rewritten:
                mySession.effective_query_reason = "rewritten"
            else:
                mySession.effective_query_reason = "changed"
            self.pretty.write(
                "I",
                "FinalQuery",
                f"Final query for retrieval: {final_query!r} "
                f"(was: {user_query_original!r})",
                color=CYAN,
            )
            return

        self.pretty.write(
            "I",
            "FinalQuery",
            f"Final query for retrieval: {final_query!r} (unchanged)",
            color=CYAN,
        )

    def _normalize_query(
        self, mySession: Session, user_query_original: str
    ) -> tuple[str, list[str]]:
        """Apply the retrieval-language contract and produce alternate queries.

        Pipeline:
          1) Detect raw user language (user_language)
          2) Build orig_translated_query_en in retrieval_language (English)
          3) Rewrite in retrieval language
          4) Enforce retrieval language after rewrite
          5) Generate alternate retrieval queries

        Returns (final_query, alternate_queries).
        """
        backend, translator = self._resolve_turn_translation_backend(mySession)

        raw_query: str = mySession.query or ""
        raw_query_language: str = self._detect_raw_query_language(
            mySession,
            raw_query,
        )
        if DebugHelper.check_session(mySession, 30):
            self.pretty.write(
                "D",
                "QueryPlan",
                f"Step 1: detected query language={raw_query_language}; query={raw_query!r}",
                color=CYAN,
            )
        native_mode: bool = self._is_native_retrieval(mySession)
        if native_mode:
            # Multilingual embedder/reranker: no translation before or after rewrite.
            translator = None

        # Build the original translated retrieval query before rewrite.
        orig_translated_query_en, translated_orig_query = (
            self._translate_query_to_english(
                raw_query,
                raw_query_language,
                translator,
                backend,
                "Normalized user query",
            )
        )
        was_translated: bool = translated_orig_query

        mySession.orig_translated_query_en = orig_translated_query_en
        # Seed retrieval query = the original user query normalized to English
        # before rewrite. Guardrail retrieval can reuse this form when the
        # post-rewrite query drifts and misses relevant chunks.
        mySession.seed_retrieval_query = orig_translated_query_en
        mySession.t1_query = orig_translated_query_en
        mySession.query = orig_translated_query_en

        rewrite_language, was_rewritten = self._rewrite_query_with_strict_retry(
            mySession
        )
        if DebugHelper.check_session(mySession, 30):
            self.pretty.write(
                "D",
                "QueryPlan",
                f"Step 2: rewrite completed; language={rewrite_language}; query={mySession.query!r}",
                color=CYAN,
            )

        # Build the post-rewrite English retrieval query.
        translated_post_rewrite = (
            False
            if native_mode
            else self._apply_post_rewrite_translation(
                mySession,
                rewrite_language,
                translator,
                backend,
            )
        )

        was_translated = translated_post_rewrite or was_translated

        final_query: str = mySession.query or ""
        mySession.post_rewrite_query_en = final_query
        mySession.final_retrieval_query = final_query
        mySession.t2_query = final_query

        if DebugHelper.check_session(mySession, 29):
            self._log_query_language_drift(
                mySession,
                raw_query_language,
                orig_translated_query_en,
            )

        self._set_effective_query_metadata(
            mySession,
            final_query,
            user_query_original,
            was_translated,
            was_rewritten,
        )

        alternate_queries: list[str] = (
            []
            if getattr(mySession, "enable_keyword_action_splitter", False)
            else self._generate_alternate_queries(final_query, mySession)
        )
        if DebugHelper.check_session(mySession, 30):
            self.pretty.write(
                "D",
                "QueryPlan",
                f"Step 3: generated {len(alternate_queries)} alternatives after rewrite: {alternate_queries!r}",
                color=CYAN,
            )
        if alternate_queries and DebugHelper.check_session(mySession, 29):
            query_lines = [f"{i+1}: {q!r}" for i, q in enumerate(alternate_queries)]
            self.pretty.write(
                "D",
                "MultiQuery",
                f"Alternate queries ({len(alternate_queries)})"
                + "\n"
                + "\n".join(query_lines),
                color=CYAN,
            )

        return final_query, alternate_queries

    def _check_gates(self, mySession: Session, final_query: str) -> bool:
        """Run the retrieval eligibility and intent classifier gates.

        Returns True if retrieval should be aborted (caller returns "", 0).
        """
        if self.retrievalGate.check_pronouns_resolved(mySession):
            return True

        score, outcome, reasons = self.intent_filter.score_query(
            final_query, path="local"
        )
        if outcome == "REFUSE":
            reason_str = ", ".join(reasons) if reasons else "intent score"
            mySession.clarification_response = (
                f"Your query was blocked by the content policy "
                f"(intent score {score} — {reason_str})."
            )
            if DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "W",
                    "IntentFilter",
                    f"Query blocked \u2014 score={score}, reasons={reasons}",
                )
            return True
        if outcome == "ALLOW_WITH_SAFETY_FRAMING":
            if DebugHelper.check_session(mySession, 30):
                self.pretty.write(
                    "W",
                    "IntentFilter",
                    f"Dual-use query detected \u2014 score={score}, reasons={reasons}. "
                    "Proceeding with retrieval.",
                )
        return False

    def _generate_alternate_queries(
        self, query: str, mySession: "Session"
    ) -> list[str]:
        """Generate alternate phrasings of *query* using the rewrite LLM.

        Returns a list of up to ``_MULTI_QUERY.num_variants`` strings on
        success, or an empty list on any error or when the feature is disabled.
        """
        import json

        enabled: bool = self.cfg.get_bool("_MULTI_QUERY.enabled")
        if not enabled or not query:
            return []

        num_variants: int = self.cfg.get_int("_MULTI_QUERY.num_variants") or 3
        temperature: float = (
            self.cfg.get_float("_MULTI_QUERY.LLM_PARAM.temperature") or 0.5
        )
        top_k: int = self.cfg.get_int("_MULTI_QUERY.LLM_PARAM.top_k") or 40
        top_p: float = self.cfg.get_float("_MULTI_QUERY.LLM_PARAM.top_p") or 0.95
        num_predict: int = self.cfg.get_int("_MULTI_QUERY.LLM_PARAM.num_predict") or 256
        use_gpu: bool = self.cfg.get_bool("_MULTI_QUERY.LLM_PARAM.use_ollama_gpu")

        # Reuse the same model already loaded for topic detection / query rewrite.
        llm_args: dict[str, Any] = self.helperInstance.get_model_args(
            "_ACTIVE_LLM_REWRITE_PROMPT"
        )
        llm_model: str = llm_args["MODEL"]
        expand_prompt_var: str = llm_args.get(
            "PROMPT_QUERY_EXPAND", "_PROMPT_QUERY_EXPAND"
        )
        prompt_template: str
        prompt_name: str | None
        prompt_template, prompt_name = self.cfg.indirect_get(expand_prompt_var)

        native_mode = (
            str(getattr(mySession, "retrieval_language_mode", "") or "").upper()
            == "NATIVE"
        )
        retrieval_language: str = (
            str(getattr(mySession, "retrieval_language", "") or "english")
            if native_mode
            else "english"
        )
        formatted: str = prompt_template.format(
            num_variants=num_variants,
            query=query,
            output_language=retrieval_language,
        )

        effective_ctx: int = self.tokenBudget.get_effective_context_limit(
            llm_model, mySession
        )

        ollama_options: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": effective_ctx,
        }
        if not use_gpu:
            ollama_options["num_gpu"] = 0

        try:
            from AI.LLMCaller import LLMCaller

            self.pretty.write(
                "I",
                "LLM Plan",
                f"Alternate queries: generating up to {num_variants} retrieval variants from the normalized query.",
            )

            result: dict[str, str] = LLMCaller().call_llm(
                model=llm_model,
                prompt=formatted,
                ollama_options=ollama_options,
                answer_is_json=True,
                template_name=prompt_name,
                streaming=False,
                stage="Multi-query expansion",
                purpose="Query alternates",
                model_role="_ACTIVE_LLM_REWRITE_PROMPT",
            )
        except Exception as exc:
            self.pretty.write(
                "W",
                "MultiQuery",
                f"LLM call failed — skipping expansion: {exc}",
            )
            return []

        raw: str = (result.get("content") or "").strip()
        if not raw:
            self.pretty.write(
                "W", "MultiQuery", "Empty LLM response - using the original query only."
            )
            return []

        try:
            variants: Any = json.loads(raw)
            if not isinstance(variants, list) or not all(
                isinstance(v, str) for v in variants
            ):
                self.pretty.write(
                    "W",
                    "MultiQuery",
                    "Expected a JSON array of strings - using the original query only.",
                )
                return []
            cleaned: list[str] = []
            seen = {query.strip().casefold()}
            for variant in variants:
                text = str(variant).strip()
                if text and text.casefold() not in seen:
                    cleaned.append(text)
                    seen.add(text.casefold())
            if len(cleaned) < num_variants:
                self.pretty.write(
                    "W",
                    "MultiQuery",
                    f"Only {len(cleaned)}/{num_variants} distinct alternatives available; retaining the original query.",
                )
            return cleaned[:num_variants]
        except Exception:
            self.pretty.write(
                "W",
                "MultiQuery",
                f"JSON parse failed — skipping expansion. Raw: {raw[:120]}",
            )
            return []
