# RAGChat Pipeline Flow

The interactive entry point is [`RAGChat.py`](../../src/Apps/RAGChat.py).
Retrieval is orchestrated by [`Orchestrator.py`](../../src/Orchestrator/Orchestrator.py);
answer generation and post-processing are handled by
[`Chatter.py`](../../src/Chat/Chatter.py). Steps vary by session and flow.

```mermaid
flowchart TD
    A["Initialize models, session, strategy and flow"] --> B["Settings and query input"]
    B --> C{"Summarize command?"}
    C -- Yes --> SUM["Select document and run summary path<br/>answer checks still apply"]
    C -- No --> D["Apply file and metadata filters"]
    D --> E{"Prompt compliance and LLM guard pass?"}
    E -- No --> REJECT["Log rejection and return to prompt"]
    E -- Yes --> F["Chatter invokes retrieval orchestrator"]
    F --> G["Resolve flow, vector store and session context"]
    G --> H["Normalize query: language, optional translation,<br/>history rewrite and alternate queries"]
    H --> I{"Retrieval and intent gates pass?"}
    I -- No --> NORESULT["Clarification or no-result response"]
    I -- Yes --> J["Retrieve enabled local candidates:<br/>vector, BM25, graph and regex"]
    I -- Yes --> K["Optionally retrieve web candidates"]
    J --> L["Merge candidates, remove duplicates<br/>and optionally rerank"]
    K --> L
    L --> M["Select chunks; apply configured<br/>low-score fallback and low-recall rescue"]
    M --> N{"Context available?"}
    N -- No --> NORESULT
    N -- Yes --> O["Build prompt and token budget; call LLM"]
    O --> P{"Answer compliance passes?"}
    P -- No --> BLOCK["Block answer"]
    P -- Yes --> Q["Optional grounding and marked sources;<br/>confidence, masking, history and display"]
    SUM --> B
    REJECT --> B
    NORESULT --> B
    BLOCK --> B
    Q --> B
    classDef check fill:#fff3cd,stroke:#8a6d3b,color:#111
    classDef process fill:#e6f1ff,stroke:#2870a8,color:#111
    classDef result fill:#e2f5e6,stroke:#33834a,color:#111
    class C,E,I,N,P check
    class A,B,D,F,G,H,J,K,L,M,O process
    class SUM,REJECT,NORESULT,BLOCK,Q result
```
