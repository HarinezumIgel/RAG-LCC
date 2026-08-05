#!/usr/bin/env python3
"""
Inspect a persisted graph index file (graph_index.pkl.gz).

Displays index-level statistics, per-chunk entity lists, and a sample
of adjacency edges (highest co-occurrence weight first).

Usage (run from project root):
    python src/Scripts/GraphIndexInspector.py -path chromadb/graph/Test
    python src/Scripts/GraphIndexInspector.py -path chromadb/graph/Test -chunks 3
    python src/Scripts/GraphIndexInspector.py -path chromadb/graph/Test -chunks 10 -edges 20

Arguments:
    -path    Relative path from the project root to the collection directory
             that contains graph_index.pkl.gz  (required)
    -chunks  Number of chunks to display (default: 5)
    -edges   Number of top adjacency edges to display (default: 10)
"""

import argparse
import gzip
import os
import pickle
import sys
import types
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this script)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Refuse to run from a drive/filesystem root before doing anything
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
from Commons.DriveRootGuard import assert_not_drive_root  # noqa: E402

assert_not_drive_root(__file__)

# ---------------------------------------------------------------------------
# Register a lightweight stub so pickle can deserialise _GraphIndexData
# without importing the full application (avoids heavy dependency chain).
# ---------------------------------------------------------------------------
_strategies_mod = types.ModuleType("Strategies")
_graph_mod = types.ModuleType("Strategies.GraphRetriever")


class _GraphIndexData:
    """Minimal stand-in matching the real class's __slots__ layout."""

    chunk_entities: Dict[str, List[str]]
    chunk_metas: Dict[str, Dict[str, Any]]
    chunk_texts: Dict[str, str]
    adjacency: Dict[str, Dict[str, int]]
    entity_to_chunks: Dict[str, List[str]]
    collection_name: str
    doc_count_at_build: int


_graph_mod._GraphIndexData = _GraphIndexData  # type: ignore[attr-defined]
sys.modules.setdefault("Strategies", _strategies_mod)
sys.modules.setdefault("Strategies.GraphRetriever", _graph_mod)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_FILENAME = "graph_index.pkl.gz"


def _load_index(index_path: str) -> _GraphIndexData:
    """Decompress and unpickle the graph index file."""
    with gzip.open(index_path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local file


def _print_summary(data: _GraphIndexData, top_edges: int) -> None:
    """Print high-level index statistics and top adjacency edges."""
    print("=" * 72)
    print("  Graph Index Summary")
    print("=" * 72)
    print(f"  Collection name       : {data.collection_name}")
    print(f"  Chunks indexed        : {len(data.chunk_entities)}")
    print(f"  Doc count at build    : {data.doc_count_at_build}")
    print(f"  Graph nodes (entities): {len(data.adjacency)}")

    # Total edges (each undirected edge stored twice)
    total_directed = sum(len(v) for v in data.adjacency.values())
    print(
        f"  Graph edges (directed): {total_directed}  ({total_directed // 2} undirected)"
    )
    print()

    # Top N edges by co-occurrence weight
    all_edges: List[tuple] = []
    seen_pairs: set[frozenset] = set()
    for node_a, neighbours in data.adjacency.items():
        for node_b, weight in neighbours.items():
            pair = frozenset((node_a, node_b))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                all_edges.append((weight, node_a, node_b))
    all_edges.sort(reverse=True)

    show = min(top_edges, len(all_edges))
    print(f"  Top-{show} co-occurrence edges:")
    for weight, a, b in all_edges[:show]:
        print(f"    w={weight:>3}  {a!r:35}  <->  {b!r}")
    print()

    # Top N entities by degree (number of distinct neighbours)
    degrees = [(len(nbrs), node) for node, nbrs in data.adjacency.items()]
    degrees.sort(reverse=True)
    print(f"  Top-10 entities by degree:")
    for deg, node in degrees[:10]:
        print(f"    degree={deg:>4}  {node!r}")
    print()


def _print_chunks(data: _GraphIndexData, n: int) -> None:
    """Print detailed entity info for the first *n* chunks."""
    chunk_ids = list(data.chunk_entities.keys())
    total = len(chunk_ids)
    show = min(n, total)
    print(f"  Showing {show} of {total} chunks")
    print("-" * 72)

    for cid in chunk_ids[:show]:
        ents = data.chunk_entities[cid]
        meta = data.chunk_metas.get(cid, {})
        text = data.chunk_texts.get(cid, "")
        preview = text[:150].replace("\n", " ")
        ellipsis = "..." if len(text) > 150 else ""

        print(f"\n  chunk_id : {cid}")
        for k, v in meta.items():
            print(f"    {k:16s}: {v}")
        print(
            f"    entities ({len(ents):>3d}) : {ents[:10]}"
            + (" ..." if len(ents) > 10 else "")
        )
        print(f"    text preview   : {preview}{ellipsis}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a persisted graph index (graph_index.pkl.gz)."
    )
    parser.add_argument(
        "-path",
        required=True,
        help=(
            "Relative path from the project root to the collection directory "
            "containing graph_index.pkl.gz, e.g. chromadb/graph/Test"
        ),
    )
    parser.add_argument(
        "-chunks",
        type=int,
        default=5,
        help="Number of chunks to display (default: 5)",
    )
    parser.add_argument(
        "-edges",
        type=int,
        default=10,
        help="Number of top co-occurrence edges to display (default: 10)",
    )
    args = parser.parse_args()

    resolved = os.path.normpath(os.path.join(_PROJECT_ROOT, args.path.lstrip("./\\")))
    if resolved.endswith(INDEX_FILENAME):
        index_path = resolved
    else:
        index_path = os.path.join(resolved, INDEX_FILENAME)

    if not os.path.isfile(index_path):
        print(f"Error: index file not found at {index_path}")
        sys.exit(1)

    data = _load_index(index_path)
    _print_summary(data, args.edges)
    _print_chunks(data, args.chunks)


if __name__ == "__main__":
    main()
