"""Chunker routing, chunker parameters, and metadata extraction settings."""

from typing import Any

# Active chunker configuration profile -- pick one of the keys in _CHUNK_STRATEGY.
# Select DETAILED only if you have a strong CPU or better GPU. SEMANTIC chunking takes time
_ACTIVE_CHUNKER_CONFIG = "DETAILED"

# Two-level chunker routing: each profile maps file extensions to a chunker.
# Keys are lowercase file extensions (from ValidExtensions.getFileType).
# "DEFAULT" is the fallback for any extension not explicitly mapped.
#
# Keep global file-type handling in Config_Global._CONSIDER_AS_TEXT_FILE so
# DocClassify and RAGLoad continue sharing one source of truth.
_CHUNK_STRATEGY: dict[str, dict[str, str]] = {
    "DETAILED": {
        "pdf": "PDF_PAGE",
        "doc": "HEADING",
        "docx": "HEADING",
        "pptx": "SLIDE",
        "ppt": "SLIDE",
        "xlsx": "RECURSIVE",
        "xls": "RECURSIVE",
        "csv": "RECURSIVE",
        "txt": "SLIDING_WINDOW",
        "md": "HEADING",
        "py": "RECURSIVE",
        "c": "RECURSIVE",
        "h": "RECURSIVE",
        "cpp": "RECURSIVE",
        "log": "RECURSIVE",
        "png": "RECURSIVE",
        "jpg": "RECURSIVE",
        "jpeg": "RECURSIVE",
        "gif": "RECURSIVE",
        "bmp": "RECURSIVE",
        "tiff": "RECURSIVE",
        "webp": "RECURSIVE",
        "DEFAULT": "SEMANTIC",
    },
    "FAST": {
        "DEFAULT": "RECURSIVE",
    },
}

_CHUNKERS: dict[str, dict[str, float | int | bool | str]] = {
    "RECURSIVE": {
        "CHUNK_SIZE": 256,  # Number of words per chunk
        "CHUNK_OVERLAP": 32,  # Overlap between consecutive chunks (10-30% of CHUNK_SIZE)
        "PRESERVE_NEWLINES": False,
    },
    "SEMANTIC": {
        "MAX_CHUNK_SIZE": 256,  # Safety cap -- split semantic segments exceeding this
        # Higher = fewer breaks = larger chunks (diffuse embeddings).
        # Lower  = more breaks  = smaller, tighter chunks (sharper retrieval).
        # 50-70 is a good starting range; 10 was far too conservative.
        "BREAKPOINT_PERCENTILE": 15,  # Bottom N% cosine similarity = chunk boundary
        "EMBED_BATCH_SIZE": 32,  # Sentences per embedding batch (controls GPU memory)
        "MIN_SENTENCE_WORDS": 15,  # Merge consecutive fragments shorter than this before
        # embedding -- prevents noisy vectors from PDF table rows
        "PRESERVE_NEWLINES": False,
    },
    "SENTENCE_WINDOW": {
        "MAX_CHUNK_SIZE": 256,  # Pack sentences up to this many words per chunk
        "PRESERVE_NEWLINES": False,
    },
    "SLIDING_WINDOW": {
        "MAX_CHUNK_SIZE": 256,  # Pack sentences up to this many words per chunk
        "OVERLAP_SENTENCES": 3,  # Re-include last N sentences of previous chunk in next
        "PRESERVE_NEWLINES": False,
    },
    "HEADING": {
        "MAX_CHUNK_SIZE": 256,  # Max words per heading-section chunk
        "PRESERVE_NEWLINES": True,  # Newlines needed for heading detection
        # Where to place the heading breadcrumb ("H1 > H2 > H3") inside each chunk.
        #   "prefix"    -- prepend to chunk text (legacy; pollutes leading tokens
        #                 when many chunks share the same breadcrumb, which can
        #                 confuse small LLMs and weight all embeddings toward
        #                 the shared prefix).
        #   "suffix"    -- append after the body text (recommended default: the
        #                 chunk's leading tokens are the actual content, while
        #                 the breadcrumb is still embedded for section context).
        #   "off"       -- omit from chunk text entirely. The breadcrumb is still
        #                 preserved in metadata["HeadingPath"] for filtering /
        #                 display / reranking.
        "BREADCRUMB_MODE": "suffix",
    },
    "SLIDE": {
        "MAX_CHUNK_SIZE": 256,  # Max words per slide chunk
        "PRESERVE_NEWLINES": False,
    },
    "PDF_PAGE": {
        "MAX_CHUNK_SIZE": 200,  # Max words per page chunk; dense pages are split
        "PRESERVE_NEWLINES": False,
        # Best-effort: recover the printed page number (e.g. roman "iii") from
        # each page's footer/header text when the PDF's /PageLabels metadata
        # does not declare it. Heuristic -- set False to trust /PageLabels only.
        "DETECT_PRINTED_LABEL": True,
    },
}

# =============================================================================
# Document metadata extraction
# =============================================================================
# Extra metadata harvested from source files at load time and attached to
# EVERY chunk of that file (all chunkers benefit -- extraction happens once in
# the ingestion pipeline via DocumentMetadataExtractor). Formats with readable
# document properties (PDF, docx, pptx, xlsx) yield the rich DOC_INFO_FIELDS;
# every other type (images, text, csv, code, legacy Office, ...) falls back to
# the generic filesystem GENERIC_FIELDS. Chunk metadata is baked into the
# ChromaDB collection, so changing anything here requires a reload
# (RETRIEVAL_STORES_KEEP = False).
#
# ChromaDB only accepts scalar metadata (str/int/float/bool) -- every value is
# coerced to a string; missing/empty fields are skipped.
# =============================================================================
_METADATA_EXTRACTION: dict[str, Any] = {
    # Master switch. False = attach nothing extra (legacy behaviour).
    "ENABLED": True,
    # Canonical chunk-metadata field name -> ordered list of raw source-property
    # synonyms searched across formats (case-insensitive; first non-empty wins).
    # Different formats name the same concept differently -- e.g. PDFs expose a
    # "creation_date" while Office core-properties use "created". Format quirks
    # that are truly ambiguous are pre-normalised inside DocumentMetadataExtractor
    # (e.g. the xlsx "creator" property, which actually means *author*, is
    # emitted as "author"), so the synonyms below stay simple.
    "DOC_INFO_FIELDS": {
        "Author": ["author"],
        "DocTitle": ["title"],
        "Subject": ["subject"],
        "Creator": ["creator"],  # authoring application (PDF)
        "Producer": ["producer"],  # producing application (PDF)
        "DocCreated": ["creation_date", "created"],
        "DocModified": ["modification_date", "modified"],
        "LastModifiedBy": ["last_modified_by"],
        "Keywords": ["keywords"],
    },
    # Generic fallback fields for every OTHER file type (images, text, csv,
    # code, legacy Office, ...) that has no readable document properties. Sourced
    # from the filesystem: "size" (bytes) and "modified" (mtime).
    "GENERIC_FIELDS": {
        "FileSizeBytes": ["size"],
        "FileModified": ["modified"],
    },
    # Printed per-page page label (e.g. "i", "ii", "1", "2") captured by the
    # PDF page chunker and written under this field. Empty string = disabled.
    # The physical 1-based page index always stays in "PageNumber".
    "PDF_PAGE_LABEL_FIELD": "PageLabel",
    # Append a "Document metadata" section (author/dates/pages per source
    # file) to CLI and RAGChatService answers, built from the harvested fields.
    "SHOW_IN_ANSWER": True,
}
