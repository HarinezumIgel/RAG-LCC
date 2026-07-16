# RAGChatService — Simplified Flow

```mermaid
flowchart LR
    A["**Startup**\nFastAPI, Models, Concurrency"]
    B["**HTTP Request**\nPOST /v1/chat/completions"]
    C["**Auth & Routing**\nBearer token validation\nHousekeeping vs real query"]
    D["**Prompt Filter Chain**\nEnsemble scorers + LLM Guard"]
    E{"**Prompt OK?**"}
    F["**Multi-Store Retrieval**\nVector (ChromaDB) · BM25 · Graph\nRRF fusion (multi-store modes)"]
    G["**Rerank & Select Chunks**\nCross-encoder, threshold filter"]
    GM["**Visual Marking** *(optional)*\nSource docs highlighted in memory\nLink block appended to response"]
    H["**Prompt Assembly & LLM Call**\nToken budgeting, Ollama invoke"]
    I["**Answer Filter Chain**\nEnsemble check on response"]
    J{"**Answer OK?**"}
    K["**Response**\nStreaming (SSE) or JSON\nMarked-doc links *(mark_text)*"]
    BLOCK(["**Blocked**\nFilter Chain rejection"])

    A --> B --> C --> D --> E
    E -- "Fail" --> BLOCK
    E -- "Pass" --> F --> G --> GM --> H --> I --> J
    J -- "Fail" --> BLOCK
    J -- "Pass" --> K

    style A fill:#4a86c8,color:#fff
    style B fill:#1abc9c,color:#fff
    style C fill:#1abc9c,color:#fff
    style D fill:#e74c3c,color:#fff
    style F fill:#e8a838,color:#fff
    style G fill:#e8a838,color:#fff
    style H fill:#d4ac0d,color:#fff
    style I fill:#e74c3c,color:#fff
    style K fill:#2ecc71,color:#fff
    style BLOCK fill:#c0392b,color:#fff
    style GM fill:#16a085,color:#fff
```
