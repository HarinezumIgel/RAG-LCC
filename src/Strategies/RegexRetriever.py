"""Regex-style retrieval index focused on content verbs and nouns.

Builds a persisted per-chunk lemma index over content verbs (excluding
auxiliaries by default) and noun tokens. The index can be:
  - Persisted to disk during RAGLoad (``build_and_persist``)
  - Loaded from disk during RAGChat  (``load_or_rebuild``)
  - Updated incrementally per file   (``remove_by_filepath`` / ``add_chunks``)

Scoring at query time uses soft coverage over query verbs and nouns with a
strict gate and an optional fallback-to-verb-only pass.
"""

import gzip
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from langchain_core.documents.base import Document as LangchainDocument

from Commons.Exceptions import ModelLoadError
from Commons.SingletonMixin import SingletonMixin
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.FileUtils import FileUtils
from Helpers.PerfLogger import PerfLogger


class _RegexIndexData:
    """Serializable container for the regex retriever index state."""

    __slots__ = (
        "chunk_verbs",
        "chunk_nouns",
        "chunk_metas",
        "chunk_texts",
        "collection_name",
        "doc_count_at_build",
    )

    def __init__(self) -> None:
        self.chunk_verbs: Dict[str, Set[str]] = {}
        self.chunk_nouns: Dict[str, Set[str]] = {}
        self.chunk_metas: Dict[str, Dict[str, Any]] = {}
        self.chunk_texts: Dict[str, str] = {}
        self.collection_name: str = ""
        self.doc_count_at_build: int = 0


class RegexRetriever(SingletonMixin):
    """Singleton that manages a per-collection verb/noun regex-style index."""

    INDEX_FILENAME = "regex_index.pkl.gz"

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        nlp: Any = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.perf_logger: PerfLogger = PerfLogger()
        self._file_utils: FileUtils = FileUtils(cfg=self.cfg, pretty=self.pretty)
        self._data: _RegexIndexData = _RegexIndexData()

        self._regex_index_dir: str = self.cfg.get_str(
            "_REGEX_INDEX.REGEX_INDEX_DIR",
            os.path.join(self.cfg.get_str("_ABSOLUTE_PATH"), "chromadb", "regex"),
        )
        self._max_candidates: int = self.cfg.get_int("_REGEX_INDEX.max_candidates", 50)
        self._min_token_chars: int = self.cfg.get_int("_REGEX_INDEX.min_token_chars", 3)
        self._exclude_auxiliaries: bool = self.cfg.get_bool(
            "_REGEX_INDEX.exclude_auxiliaries",
            True,
        )
        noun_pos = self.cfg.get_list("_REGEX_INDEX.noun_pos_tags", [], silent=True)
        self._noun_pos_tags: Set[str] = {str(tag).upper() for tag in noun_pos}

        aux_lemmas_cfg = self.cfg.get_list("_REGEX_INDEX.aux_lemmas", [], silent=True)
        self._aux_lemmas: Set[str] = {str(x).strip().lower() for x in aux_lemmas_cfg}
        self._shared: SharedHelpers = SharedHelpers(cfg=self.cfg, pretty=self.pretty)
        # Per-language auxiliary lemma cache keyed by ISO/NLTK-ish code.
        self._aux_lemmas_by_lang: Dict[str, Set[str]] = {"en": set(self._aux_lemmas)}

        self._require_both_when_available: bool = self.cfg.get_bool(
            "_REGEX_INDEX.require_both_when_available",
            True,
        )
        self._min_verb_hits: int = max(
            1,
            self.cfg.get_int("_REGEX_INDEX.min_verb_hits", 1),
        )
        self._min_noun_hits: int = max(
            1,
            self.cfg.get_int("_REGEX_INDEX.min_noun_hits", 1),
        )

        self._fallback_to_verb_only: bool = self.cfg.get_bool(
            "_REGEX_INDEX.fallback_to_verb_only",
            True,
        )
        self._fallback_min_verb_hits: int = max(
            1,
            self.cfg.get_int("_REGEX_INDEX.fallback_min_verb_hits", 1),
        )

        self._verb_weight: float = self.cfg.get_float("_REGEX_INDEX.verb_weight", 2.0)
        self._noun_weight: float = self.cfg.get_float("_REGEX_INDEX.noun_weight", 1.0)
        self._both_match_bonus: float = self.cfg.get_float(
            "_REGEX_INDEX.both_match_bonus",
            0.25,
        )
        self._fallback_score_scale: float = self.cfg.get_float(
            "_REGEX_INDEX.fallback_score_scale",
            0.6,
        )

        self._spacy_model: str = self.cfg.get_str("_REGEX_INDEX.spacy_model")

        if nlp is not None:
            self._nlp = nlp
        else:
            try:
                import spacy  # type: ignore[import-untyped]

                self._nlp = spacy.load(self._spacy_model)
            except OSError as exc:
                raise ModelLoadError(
                    f"spaCy model '{self._spacy_model}' not found. "
                    f"Run: python -m spacy download {self._spacy_model}"
                ) from exc

        if self._exclude_auxiliaries:
            self._prime_active_aux_lemmas_cache()

    # ------------------------------------------------------------------
    # Public API -- directory helpers
    # ------------------------------------------------------------------

    def get_regex_dir(self, collection_name: str) -> str:
        """Return the regex index directory for *collection_name*."""
        return os.path.join(self._regex_index_dir, collection_name)

    def get_index_dir(self, collection_name: str) -> str:
        """Protocol alias for ``get_regex_dir``."""
        return self.get_regex_dir(collection_name)

    # ------------------------------------------------------------------
    # Public API -- index state queries
    # ------------------------------------------------------------------

    def is_loaded_for(self, collection_name: str) -> bool:
        """True if the in-memory index belongs to *collection_name*."""
        return self._data.collection_name == collection_name and bool(
            self._data.chunk_texts
        )

    # ------------------------------------------------------------------
    # Public API -- index lifecycle
    # ------------------------------------------------------------------

    def load_or_rebuild(
        self,
        regex_directory: str,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Load a persisted index or rebuild from the ChromaDB collection."""
        if (
            self._data.collection_name == collection_name
            and bool(self._data.chunk_texts)
            and self._data.doc_count_at_build == collection.count()
        ):
            return

        idx_path = self._index_path(regex_directory)

        if os.path.isfile(idx_path):
            self._load(idx_path)
            if (
                self._data.collection_name == collection_name
                and self._data.doc_count_at_build == collection.count()
            ):
                unique_verbs, unique_nouns = self._unique_term_counts()
                self.pretty.write(
                    "O",
                    "Regex",
                    f"Loaded persisted regex index ({len(self._data.chunk_texts)} chunks, "
                    f"{unique_verbs} verbs, {unique_nouns} nouns)",
                )
                return
            self.pretty.write(
                "I",
                "Regex",
                "Persisted regex index is stale -- rebuilding from collection",
            )

        self._rebuild_from_collection(collection_name, collection)
        self._persist(idx_path)

    def build_and_persist(
        self,
        regex_directory: str,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Full rebuild from collection + write to disk. Called from RAGLoad."""
        self._rebuild_from_collection(collection_name, collection)
        self._persist(self._index_path(regex_directory))

    def _can_remove_filepath(self, file_path: str) -> bool:
        """Guard remove_by_filepath with the shared jailbreak-safe path check."""
        if not hasattr(self, "_file_utils"):
            self._file_utils = FileUtils(cfg=self.cfg, pretty=self.pretty)
        return self._file_utils.is_safe_delete_path(file_path)

    def remove_by_filepath(self, file_path: str) -> None:
        """Remove all chunks belonging to *file_path*."""
        if not self._can_remove_filepath(file_path):
            self.pretty.write(
                "W",
                "Regex",
                f"Skipped remove_by_filepath for unsafe path: {file_path!r}",
            )
            return

        chunk_ids_to_remove = [
            cid
            for cid, meta in self._data.chunk_metas.items()
            if meta.get("FilePath") == file_path
        ]
        if not chunk_ids_to_remove:
            return

        for cid in chunk_ids_to_remove:
            self._data.chunk_verbs.pop(cid, None)
            self._data.chunk_nouns.pop(cid, None)
            self._data.chunk_metas.pop(cid, None)
            self._data.chunk_texts.pop(cid, None)

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Extract per-chunk terms and add to index state."""
        for chunk_id, text, meta in zip(ids, texts, metas):
            lang = str((meta or {}).get("Language", "en"))
            verbs, nouns = self._extract_terms(text, lang)
            self._data.chunk_verbs[chunk_id] = verbs
            self._data.chunk_nouns[chunk_id] = nouns
            self._data.chunk_metas[chunk_id] = dict(meta)
            self._data.chunk_texts[chunk_id] = text

    def persist(self, regex_directory: str) -> None:
        """Write current index state to disk."""
        self._persist(self._index_path(regex_directory))

    def ingest_file(
        self,
        file_path: str,
        collection_name: str,
        collection: Any,
        ids: List[str],
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Incrementally update the regex index for a single file."""
        regex_dir = self.get_regex_dir(collection_name)
        if not self.is_loaded_for(collection_name):
            self.load_or_rebuild(regex_dir, collection_name, collection)
        self.remove_by_filepath(file_path)
        if ids:
            self.add_chunks(ids, texts, metas)
        self.persist(regex_dir)

    # ------------------------------------------------------------------
    # Public API -- query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 100,
        file_filter: Optional[Dict[str, Any]] = None,
    ) -> List[LangchainDocument]:
        """Score *query_text* against indexed verb/noun terms and return top-*k*."""
        if not self._data.chunk_texts:
            return []

        query_verbs, query_nouns = self._extract_terms(query_text, "en")
        if not query_verbs and not query_nouns:
            return []

        self.perf_logger.log(
            "RegexRetriever.query",
            "retriever",
            f"start regex query q={query_text[:60]!r}",
        )
        _t0 = time.perf_counter()

        scored: List[Tuple[str, float, int, int, List[str], List[str], str]] = []
        q_has_verbs = bool(query_verbs)
        q_has_nouns = bool(query_nouns)

        for cid in self._data.chunk_texts:
            if file_filter and not self._matches_filter(
                self._data.chunk_metas.get(cid, {}),
                file_filter,
            ):
                continue

            c_verbs = self._data.chunk_verbs.get(cid, set())
            c_nouns = self._data.chunk_nouns.get(cid, set())
            v_matches_set = query_verbs.intersection(c_verbs)
            n_matches_set = query_nouns.intersection(c_nouns)
            verb_hits = len(v_matches_set)
            noun_hits = len(n_matches_set)

            if not self._passes_strict_gate(
                q_has_verbs=q_has_verbs,
                q_has_nouns=q_has_nouns,
                verb_hits=verb_hits,
                noun_hits=noun_hits,
            ):
                continue

            score = self._score(
                verb_hits, noun_hits, len(query_verbs), len(query_nouns)
            )
            if score <= 0.0:
                continue

            scored.append(
                (
                    cid,
                    score,
                    verb_hits,
                    noun_hits,
                    sorted(v_matches_set),
                    sorted(n_matches_set),
                    "strict",
                )
            )

        if not scored and self._fallback_to_verb_only and q_has_verbs:
            for cid in self._data.chunk_texts:
                if file_filter and not self._matches_filter(
                    self._data.chunk_metas.get(cid, {}),
                    file_filter,
                ):
                    continue

                c_verbs = self._data.chunk_verbs.get(cid, set())
                c_nouns = self._data.chunk_nouns.get(cid, set())
                v_matches_set = query_verbs.intersection(c_verbs)
                n_matches_set = query_nouns.intersection(c_nouns)
                verb_hits = len(v_matches_set)
                noun_hits = len(n_matches_set)
                if verb_hits < self._fallback_min_verb_hits:
                    continue

                score = self._score(
                    verb_hits,
                    noun_hits,
                    len(query_verbs),
                    len(query_nouns),
                )
                score *= self._fallback_score_scale
                if score <= 0.0:
                    continue

                scored.append(
                    (
                        cid,
                        score,
                        verb_hits,
                        noun_hits,
                        sorted(v_matches_set),
                        sorted(n_matches_set),
                        "fallback_verb_only",
                    )
                )

        scored.sort(
            key=lambda item: (item[1], item[2] + item[3], item[2], item[3]),
            reverse=True,
        )
        top = scored[: self._max_candidates][:k]

        docs: List[LangchainDocument] = []
        for cid, score, verb_hits, noun_hits, v_matches, n_matches, mode in top:
            meta = dict(self._data.chunk_metas.get(cid, {}))
            meta["regex_score"] = score
            meta["regex_verb_hits"] = verb_hits
            meta["regex_noun_hits"] = noun_hits
            meta["regex_match_mode"] = mode
            meta["regex_verb_matches"] = ", ".join(v_matches)
            meta["regex_noun_matches"] = ", ".join(n_matches)
            meta["bm25_score"] = 0.0
            meta["graph_score"] = 0.0
            meta["chroma_score"] = score
            meta["chroma_sim"] = 1.0
            docs.append(
                LangchainDocument(
                    page_content=self._data.chunk_texts.get(cid, ""),
                    metadata=meta,
                    id=cid,
                )
            )

        self.perf_logger.log(
            "RegexRetriever.query",
            "retriever",
            f"stop  regex query n={len(docs)} elapsed={time.perf_counter() - _t0:.3f}s",
        )
        return docs

    # ------------------------------------------------------------------
    # Internal -- term extraction and scoring
    # ------------------------------------------------------------------

    def _normalize_lang_code(self, language: str | None) -> str:
        """Normalize language label to lower-case code used for cache keys."""
        lang = str(language or "en").strip().lower()
        if not lang:
            return "en"
        shared = getattr(self, "_shared", None)
        if shared is None:
            return lang
        return shared.lang_name_to_code.get(lang, lang)

    def _active_aux_languages(self) -> List[str]:
        """Return active language set from configured Argos translation pairs."""
        langs: Set[str] = {"en"}
        pairs: list[Any] = self.cfg.get_list(
            "_ARGOS_DEFINITIONS.ARGOS_LANGUAGES",
            [],
            silent=True,
        )
        for pair in pairs:
            if not isinstance(pair, (list, tuple)):
                continue
            pair_items = cast(list[Any] | tuple[Any, ...], pair)
            if len(pair_items) != 2:
                continue
            src = self._normalize_lang_code(str(pair_items[0]))
            dst = self._normalize_lang_code(str(pair_items[1]))
            if src:
                langs.add(src)
            if dst:
                langs.add(dst)
        return sorted(langs)

    def _aux_lemmas_for_lang(self, language: str | None) -> Set[str]:
        """Return a cached auxiliary-lemma set translated for *language*."""
        if not hasattr(self, "_aux_lemmas_by_lang"):
            self._aux_lemmas_by_lang = {"en": set(self._aux_lemmas)}

        lang = self._normalize_lang_code(language)
        cached = self._aux_lemmas_by_lang.get(lang)
        if cached is not None:
            return cached

        shared = getattr(self, "_shared", None)
        if shared is None:
            fallback = set(self._aux_lemmas)
            self._aux_lemmas_by_lang[lang] = fallback
            return fallback

        translated = shared.get_translated_wordlist(
            sorted(self._aux_lemmas),
            language=lang,
            algo="Regex Aux",
        )
        expanded: Set[str] = set(self._aux_lemmas)
        for phrase in translated:
            tokens = shared.tokenize(str(phrase))
            if tokens:
                expanded.update(tokens)
            else:
                normalized = str(phrase).strip().lower()
                if normalized:
                    expanded.add(normalized)

        self._aux_lemmas_by_lang[lang] = expanded
        return expanded

    def _prime_active_aux_lemmas_cache(self) -> None:
        """Pre-build aux-lemma sets for active languages at startup."""
        active_langs = self._active_aux_languages()
        fallback_count = 0
        for lang in active_langs:
            try:
                self._aux_lemmas_for_lang(lang)
            except Exception:
                # Keep retrieval functional even with partial translation setup.
                self._aux_lemmas_by_lang.setdefault(lang, set(self._aux_lemmas))
                fallback_count += 1

        msg = f"Aux-lemma cache built for {len(active_langs)} active language(s)"
        if fallback_count > 0:
            msg += f" ({fallback_count} fallback-to-English set(s))"
        self.pretty.write("O", "Regex", msg)

    def _extract_terms(
        self,
        text: str,
        language: str | None = None,
    ) -> Tuple[Set[str], Set[str]]:
        """Extract content-verb and noun lemmas from *text*."""
        doc = self._nlp(text or "")
        verbs: Set[str] = set()
        nouns: Set[str] = set()
        aux_lemmas: Set[str] = set()
        if self._exclude_auxiliaries:
            aux_lemmas = self._aux_lemmas_for_lang(language)

        for tok in doc:
            pos = str(getattr(tok, "pos_", "")).upper()
            lemma = str(getattr(tok, "lemma_", "")).strip().lower()
            if not lemma:
                continue
            if self._min_token_chars > 0 and len(lemma) < self._min_token_chars:
                continue
            if not any(ch.isalpha() for ch in lemma):
                continue

            if pos == "AUX":
                if not self._exclude_auxiliaries:
                    verbs.add(lemma)
                continue

            if pos == "VERB":
                if self._exclude_auxiliaries and lemma in aux_lemmas:
                    continue
                verbs.add(lemma)
                continue

            if pos in self._noun_pos_tags:
                nouns.add(lemma)

        return verbs, nouns

    def _passes_strict_gate(
        self,
        *,
        q_has_verbs: bool,
        q_has_nouns: bool,
        verb_hits: int,
        noun_hits: int,
    ) -> bool:
        """Check strict pass criteria for a candidate chunk."""
        if q_has_verbs and q_has_nouns:
            if self._require_both_when_available:
                return (
                    verb_hits >= self._min_verb_hits
                    and noun_hits >= self._min_noun_hits
                )
            return verb_hits >= self._min_verb_hits or noun_hits >= self._min_noun_hits
        if q_has_verbs:
            return verb_hits >= self._min_verb_hits
        if q_has_nouns:
            return noun_hits >= self._min_noun_hits
        return False

    def _score(
        self,
        verb_hits: int,
        noun_hits: int,
        query_verb_count: int,
        query_noun_count: int,
    ) -> float:
        """Coverage-based score with optional both-match bonus."""
        verb_cov = (
            float(verb_hits) / float(query_verb_count) if query_verb_count > 0 else 0.0
        )
        noun_cov = (
            float(noun_hits) / float(query_noun_count) if query_noun_count > 0 else 0.0
        )
        score = (self._verb_weight * verb_cov) + (self._noun_weight * noun_cov)
        if verb_hits > 0 and noun_hits > 0:
            score += self._both_match_bonus
        return max(0.0, float(score))

    def _unique_term_counts(self) -> Tuple[int, int]:
        """Return unique (verb_count, noun_count) for the current index."""
        all_verbs: Set[str] = set()
        all_nouns: Set[str] = set()
        for terms in self._data.chunk_verbs.values():
            all_verbs.update(terms)
        for terms in self._data.chunk_nouns.values():
            all_nouns.update(terms)
        return len(all_verbs), len(all_nouns)

    # ------------------------------------------------------------------
    # Internal -- rebuild from collection
    # ------------------------------------------------------------------

    def _rebuild_from_collection(
        self,
        collection_name: str,
        collection: Any,
    ) -> None:
        """Fetch all chunks from ChromaDB and build the regex index."""
        self.pretty.write(
            "I",
            "Regex",
            f"Building regex index from collection '{collection_name}'...",
        )

        result = collection.get(include=["documents", "metadatas"])
        ids: List[str] = result.get("ids", []) or []
        documents: List[str] = result.get("documents", []) or []
        metadatas: List[Dict[str, Any]] = result.get("metadatas", []) or []

        data = _RegexIndexData()
        data.collection_name = collection_name
        data.doc_count_at_build = collection.count()

        for chunk_id, text, meta in zip(ids, documents, metadatas):
            text = text or ""
            lang = str((meta or {}).get("Language", "en"))
            verbs, nouns = self._extract_terms(text, lang)
            data.chunk_verbs[chunk_id] = verbs
            data.chunk_nouns[chunk_id] = nouns
            data.chunk_metas[chunk_id] = dict(meta) if meta else {}
            data.chunk_texts[chunk_id] = text

        self._data = data
        unique_verbs, unique_nouns = self._unique_term_counts()
        self.pretty.write(
            "O",
            "Regex",
            f"Built regex index: {len(self._data.chunk_texts)} chunks, "
            f"{unique_verbs} verbs, {unique_nouns} nouns",
        )

    # ------------------------------------------------------------------
    # Internal -- persistence
    # ------------------------------------------------------------------

    def _index_path(self, regex_directory: str) -> str:
        return os.path.join(regex_directory, self.INDEX_FILENAME)

    def _persist(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.pretty.write(
            "O",
            "Regex",
            f"Persisted regex index to {path} ({len(self._data.chunk_texts)} chunks)",
        )

    def _load(self, path: str) -> None:
        with gzip.open(path, "rb") as f:
            self._data = pickle.load(f)  # noqa: S301

    # ------------------------------------------------------------------
    # Internal -- filter matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filter(meta: Dict[str, Any], filt: Dict[str, Any]) -> bool:
        """Check if a metadata dict matches a ChromaDB-style where filter."""
        for key, condition in filt.items():
            if isinstance(condition, dict):
                for op, val in cast(Dict[str, Any], condition).items():
                    if op == "$eq" and meta.get(key) != val:
                        return False
            else:
                if meta.get(key) != condition:
                    return False
        return True
