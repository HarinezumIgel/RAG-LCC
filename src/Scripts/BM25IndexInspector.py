#!/usr/bin/env python3
"""
Inspect a persisted BM25 index file (bm25_index.pkl.gz).

Displays index-level statistics and per-chunk details (id, metadata,
tokens, text preview) for the first N chunks.

Usage (run from project root):
    python src/Scripts/BM25IndexInspector.py -path chromadb/bm25/Test -chunks 3
    python src/Scripts/BM25IndexInspector.py -path ./chromadb/bm25/Test
    python src/Scripts/BM25IndexInspector.py -path chromadb/bm25/Test -chunks 10

Arguments:
    -path    Relative path from the project root to the collection directory
             that contains bm25_index.pkl.gz  (required)
    -chunks  Number of chunks to display (default: 5)
"""

import argparse
import gzip
import os
import pickle
import sys
import types
from collections import Counter
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this script)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# Register a lightweight stub so pickle can deserialise _BM25IndexData
# without importing the full application (avoids heavy dependency chain).
# ---------------------------------------------------------------------------
_strategies_mod = types.ModuleType("Strategies")
_bm25_mod = types.ModuleType("Strategies.BM25Retriever")


class _BM25IndexData:
    """Minimal stand-in matching the real class's __slots__ layout."""

    chunk_ids: List[str]
    chunk_tokens: List[List[str]]
    chunk_metas: List[Dict[str, Any]]
    chunk_texts: List[str]
    df: "Counter[str]"
    N: int  # pyright: ignore[reportConstantRedefinition]
    avg_dl: float
    idf: Dict[str, float]
    collection_name: str
    doc_count_at_build: int


_bm25_mod._BM25IndexData = _BM25IndexData  # type: ignore[attr-defined]
sys.modules.setdefault("Strategies", _strategies_mod)
sys.modules.setdefault("Strategies.BM25Retriever", _bm25_mod)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_FILENAME = "bm25_index.pkl.gz"


def _load_index(index_path: str) -> _BM25IndexData:
    """Decompress and unpickle the BM25 index file."""
    with gzip.open(index_path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local file


def _print_summary(data: _BM25IndexData) -> None:
    """Print high-level index statistics."""
    print("=" * 72)
    print("  BM25 Index Summary")
    print("=" * 72)
    print(f"  Collection name       : {data.collection_name}")
    print(f"  Total chunks  (N)     : {data.N}")
    print(f"  Avg doc length        : {data.avg_dl:.1f} tokens")
    print(f"  Doc count at build    : {data.doc_count_at_build}")
    print(f"  Unique terms  (vocab) : {len(data.idf)}")
    print()

    # Top-10 terms by document frequency
    print("  Top-10 terms by document frequency:")
    for term, cnt in data.df.most_common(10):
        print(f"    {term:20s}  appears in {cnt} chunks")
    print()

    # 5 sample IDF values
    sample_terms = list(data.idf.keys())[:5]
    if sample_terms:
        print("  Sample IDF values:")
        for term in sample_terms:
            print(f"    {term:20s}  idf = {data.idf[term]:.4f}")
    print()


def _print_chunks(data: _BM25IndexData, n: int) -> None:
    """Print detailed info for the first *n* chunks."""
    total = len(data.chunk_ids)
    show = min(n, total)
    print(f"  Showing {show} of {total} chunks")
    print("-" * 72)

    for i in range(show):
        print(f"\n  [{i}]  chunk_id : {data.chunk_ids[i]}")

        # Metadata
        meta = data.chunk_metas[i] if i < len(data.chunk_metas) else {}
        for k, v in meta.items():
            print(f"        {k:16s}: {v}")

        # Tokens (first 20)
        tokens = data.chunk_tokens[i] if i < len(data.chunk_tokens) else []
        preview_tokens = tokens[:20]
        suffix = " ..." if len(tokens) > 20 else ""
        print(f"        tokens ({len(tokens):>4d})  : {preview_tokens}{suffix}")

        # Text preview (first 150 chars)
        text = data.chunk_texts[i] if i < len(data.chunk_texts) else ""
        preview = text[:150].replace("\n", " ")
        ellipsis = "..." if len(text) > 150 else ""
        print(f"        text preview   : {preview}{ellipsis}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a persisted BM25 index (bm25_index.pkl.gz)."
    )
    parser.add_argument(
        "-path",
        required=True,
        help=(
            "Relative path from the project root to the collection directory "
            "containing bm25_index.pkl.gz, e.g. chromadb/docs/Test"
        ),
    )
    parser.add_argument(
        "-chunks",
        type=int,
        default=5,
        help="Number of chunks to display (default: 5)",
    )
    args = parser.parse_args()

    # Build absolute path, resolving ./ prefixes and normalising separators
    resolved = os.path.normpath(os.path.join(_PROJECT_ROOT, args.path.lstrip("./\\")))
    # Accept both a bare collection directory and a direct path to the .pkl.gz file
    if resolved.endswith(INDEX_FILENAME):
        index_path = resolved
    else:
        index_path = os.path.join(resolved, INDEX_FILENAME)

    if not os.path.isfile(index_path):
        print(f"Error: index file not found at {index_path}")
        sys.exit(1)

    # Load and display
    data = _load_index(index_path)
    _print_summary(data)
    _print_chunks(data, args.chunks)


if __name__ == "__main__":
    main()
