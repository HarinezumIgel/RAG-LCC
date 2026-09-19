---
layout: default
title: RAG-LCC
description: Experimental RAG under constraints
---

<p align="center">
  <img src="assets/AI_Igel.png" alt="RAG-LCC hedgehog icon" width="260" />
</p>

# RAG-LCC

Experimental Retrieval-Augmented Generation under constraints.

RAG-LCC is a practical lab for understanding where RAG systems fail and how to fix them: retrieval misses, context poisoning, multilingual drift, and compliance edge cases.

Current release: v0.5.1/1480 (2026-09-19)

## What you get

- Hybrid retrieval pipeline with Vector, BM25, Graph, and Regex retrieval
- Multi-turn chat support with query rewriting and topic switching
- Multi-query expansion for better recall
- Cross-encoder reranking and strategy-based context assembly
- Compliance filtering during ingestion, query, and answer phases
- Grounding markers so answer text can be traced to evidence chunks

## Application suite

| Application | Purpose | Typical output |
| --- | --- | --- |
| DocClassify | Classify and compress corpus content before indexing | CSV metadata used for selective loading |
| RAGLoad | Ingest and index local documents into multiple retrievers | ChromaDB + BM25 + Graph + Regex indexes |
| RAGChat | Interactive CLI RAG with grounding and safety checks | Grounded multi-turn answers |
| RAGChatService | OpenAI-compatible API wrapper for RAGChat | API endpoint for OpenWebUI and clients |

## Why this project is different

Most RAG stacks optimize only retrieval score. RAG-LCC also optimizes context quality.

- Detects contradictory or noisy context before generation
- Handles multilingual retrieval routing by language buckets
- Makes retrieval and grounding behavior observable with debug levels
- Exposes most architecture decisions through configuration

## Quick start

1. Read installation guide
   https://github.com/HarinezumIgel/RAG-LCC/blob/main/INSTALL.md
2. Load docs
   python ./src/Apps/RAGLoad.py --doc-dir TestDocs
3. Start chat
   python ./src/Apps/RAGChat.py --doc-dir TestDocs

## Documentation hub

- Project overview
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/README.md
- Installation
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/INSTALL.md
- Full configuration reference
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/CONFIGURATION_REFERENCE.md
- Architecture deep dive
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/ARCHITECTURE.md
- End-to-end examples
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/EXAMPLES.md
- Hands-on tour
  https://github.com/HarinezumIgel/RAG-LCC/blob/main/HANDS_ON_TOUR.md

## Source repository

https://github.com/HarinezumIgel/RAG-LCC
