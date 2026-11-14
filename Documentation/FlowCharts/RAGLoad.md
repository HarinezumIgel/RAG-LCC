# RAGLoad Pipeline Flow

The processing path depends on configuration and file type. Entry point:
[`RAGLoad.py`](../../src/Apps/RAGLoad.py); per-file processing:
[`LoadAndClassifyProcessor.py`](../../src/Pipeline/LoadAndClassifyProcessor.py) and
[`DocumentIngestionStrategy.py`](../../src/Strategies/DocumentIngestionStrategy.py).

```mermaid
flowchart TD
    A["Startup, configuration and embedding model"] --> B["Create ingestion strategy and processor"]
    B --> C{"Classification CSV configured?"}
    C -- Yes --> D["Read allowed paths using optional SQL query"]
    C -- No --> E["Use configured exclusions"]
    D --> F["Walk document directory"]
    E --> F
    F --> G{"Allowed path and valid extension?"}
    G -- No --> SKIP["Skip file"]
    G -- Yes --> H{"File changed?"}
    H -- No --> SKIP
    H -- Yes --> I["Extract text and optional structured blocks<br/>according to file type"]
    I --> J{"Extraction succeeded and content nonempty?"}
    J -- No --> SKIP
    J -- Yes --> K["Normalize Unicode and mask text<br/>build document metadata"]
    K --> L["Open collection; enrich document metadata"]
    L --> M{"Language supported?"}
    M -- No --> NOTOK["Write NOT_OK; skip document"]
    M -- Yes --> N["Choose chunker by file type<br/>prepare chunks and embeddings"]
    N --> O["Remove prior chunks for this file"]
    O --> P["Run compliance checks on full chunk text<br/>retain or discard individual chunks"]
    P --> Q["Upsert retained chunks into ChromaDB"]
    Q --> R["Refresh BM25, graph and regex indexes<br/>with retained chunks"]
    R --> S["Write status / human-review output as applicable"]
    S --> T["Report processing results"]
    SKIP --> T
    NOTOK --> T
    classDef check fill:#fff3cd,stroke:#8a6d3b,color:#111
    classDef process fill:#e6f1ff,stroke:#2870a8,color:#111
    classDef result fill:#e2f5e6,stroke:#33834a,color:#111
    class G,H,J,M check
    class A,B,D,E,F,I,K,L,N,O,P,Q,R process
    class SKIP,NOTOK,S,T result
```
