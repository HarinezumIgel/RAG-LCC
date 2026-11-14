from __future__ import annotations

from typing import Any, cast

from Orchestrator.OrchestratorTypes import INDEXED_RETRIEVER_LABELS


class OrchestratorHelper:
    """Candidate identity and merging policies, independent of pipeline state."""

    @staticmethod
    def doc_identity(doc: Any) -> str:
        """Return a stable best-effort identity key for deduping documents."""
        metadata_obj = getattr(doc, "metadata", None)
        metadata = (
            cast(dict[str, Any], metadata_obj) if isinstance(metadata_obj, dict) else {}
        )
        doc_id = (
            metadata.get("chunk_id") or getattr(doc, "id", None) or metadata.get("id")
        )
        if doc_id is not None:
            doc_id_text = str(doc_id).strip()
            if doc_id_text:
                return f"id:{doc_id_text}"

        page_content = getattr(doc, "page_content", None)
        if isinstance(page_content, str) and page_content:
            return f"text:{page_content}"

        return f"obj:{id(doc)}"

    @staticmethod
    def append_retriever_source_marker(doc: Any, marker: str) -> None:
        """Append a source marker without duplicating an existing marker."""
        if not marker:
            return
        metadata_obj = getattr(doc, "metadata", None)
        if not isinstance(metadata_obj, dict):
            return

        metadata = cast(dict[str, Any], metadata_obj)
        raw_sources = str(metadata.get("retriever_sources", "") or "").strip()
        if not raw_sources:
            metadata["retriever_sources"] = marker
            return

        source_parts = [part.strip() for part in raw_sources.split(",") if part.strip()]
        if marker in source_parts:
            return
        metadata["retriever_sources"] = ",".join(source_parts + [marker])

    def merge_docs_with_native_leg(
        self,
        *,
        base_docs: list[Any],
        native_docs: list[Any],
        source_marker: str,
    ) -> tuple[list[Any], int, int]:
        """Preserve base order and return merged docs, additions, and overlaps."""
        merged_docs = list(base_docs)
        existing_by_identity: dict[str, Any] = {
            self.doc_identity(doc): doc for doc in merged_docs
        }
        native_added = 0
        native_overlap = 0
        for doc in native_docs:
            identity = self.doc_identity(doc)
            existing_doc = existing_by_identity.get(identity)
            if existing_doc is not None:
                native_overlap += 1
                self.append_retriever_source_marker(existing_doc, source_marker)
                continue

            self.append_retriever_source_marker(doc, source_marker)
            merged_docs.append(doc)
            existing_by_identity[identity] = doc
            native_added += 1

        return merged_docs, native_added, native_overlap

    @staticmethod
    def merge_indexed_stage_docs(
        first_stage_docs: dict[str, list[Any]],
        second_stage_docs: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """Merge two indexed-doc maps with stable BM25/Graph/Regex keys."""
        return {
            label: list(first_stage_docs.get(label, []))
            + list(second_stage_docs.get(label, []))
            for label in INDEXED_RETRIEVER_LABELS
        }

    def merge_ranked_variants(
        self,
        ranked_lists: list[tuple[list[Any], float]],
        *,
        rrf_k: float,
    ) -> list[Any]:
        """Keep each candidate's best weighted reciprocal rank across variants."""
        best: dict[str, tuple[float, Any]] = {}
        for docs, weight in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                identity = self.doc_identity(doc)
                score = weight / (rrf_k + rank)
                # Do not sum variant hits: repeated alternates must not outvote
                # a stronger original-query result. Equal scores keep the first.
                if identity not in best or score > best[identity][0]:
                    best[identity] = (score, doc)
        return [
            doc
            for _, doc in sorted(best.values(), key=lambda item: item[0], reverse=True)
        ]
