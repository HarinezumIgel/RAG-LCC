#!/usr/bin/env python3

"""
Search or inspect chunks in a ChromaDB collection.

Examples:

    # Search specific terms in a specific document
    python diagnose_chunk.py --collection Test --document p620 --terms pcie card

    # Search across all documents
    python diagnose_chunk.py --collection Test --terms pcie card

    # Dump all chunks from matching document(s)
    python diagnose_chunk.py --collection Test --document p620 --all
"""

import argparse
import os
import pprint
import sys
from collections.abc import Mapping
from typing import Any

# Add project src directory to path (script is in scripts_posh/private/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from chromadb.config import Settings

import chromadb


def print_metadata(meta: Mapping[str, Any]) -> None:
    print("METADATA:")
    pprint.pprint(meta, sort_dicts=True)


def main():
    parser = argparse.ArgumentParser(description="Search a ChromaDB collection.")

    parser.add_argument(
        "--collection",
        default="Test",
        help="Collection name (default: Test)",
    )

    parser.add_argument(
        "--document",
        help="Filter by source document name (case-insensitive substring match)",
    )

    parser.add_argument(
        "--terms",
        nargs="+",
        help="One or more search terms (case-insensitive)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all chunks for matching document(s)",
    )

    args = parser.parse_args()

    if not args.all and not args.terms:
        parser.error("either --terms or --all must be specified")

    root = os.getcwd()

    chroma_db_dir = os.path.join(
        root,
        "chromadb",
        "docs",
        args.collection,
    )

    client = chromadb.PersistentClient(
        path=chroma_db_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    print(f"\nAvailable collections: " f"{[c.name for c in client.list_collections()]}")

    try:
        col = client.get_collection(args.collection)
    except Exception as exc:
        print(f"ERROR: could not open collection " f"'{args.collection}': {exc}")
        sys.exit(1)

    print(f"\nCollection '{args.collection}'" f" - {col.count()} chunks total\n")

    results = col.get(include=["documents", "metadatas"])

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    rows = list(zip(documents, metadatas))

    #
    # document filter
    #
    if args.document:
        rows = [
            (doc, meta)
            for doc, meta in rows
            if args.document.lower() in str(meta.get("FileName", "")).lower()
        ]

    print(f"Loaded {len(rows)} matching chunk(s)\n")

    #
    # dump all chunks
    #
    if args.all:
        for idx, (doc, meta) in enumerate(rows, start=1):
            print("=" * 120)
            print(f"CHUNK {idx}")
            print("=" * 120)

            print_metadata(meta)

            print("\nDOCUMENT:")
            print("-" * 120)
            print(doc)
            print()

        print("Done.")
        return

    #
    # term search
    #
    for term in args.terms:
        matches = [(doc, meta) for doc, meta in rows if term.lower() in doc.lower()]

        print("=" * 120)
        print(f"TERM: {term}")
        print(f"MATCHES: {len(matches)}")
        print()

        if not matches:
            continue

        for idx, (doc, meta) in enumerate(matches, start=1):
            print(f"[{idx}]")

            print_metadata(meta)

            print("\nPREVIEW:")
            print("-" * 120)
            print(doc[:1000])
            print()

    print("Done.")


if __name__ == "__main__":
    main()
