# DocClassify Pipeline Flow

Entry point: [`DocClassify.py`](../../src/Apps/DocClassify.py). The shared
[`LoadAndClassifyProcessor`](../../src/Pipeline/LoadAndClassifyProcessor.py)
extracts each document before [`ClassifyStrategy`](../../src/Strategies/ClassifyStrategy.py)
processes it.

```mermaid
flowchart TD
    A["Startup, configuration and embedding model"] --> B["Create classification strategy and processor"]
    B --> C["Walk document directory"]
    C --> D{"Allowed by path filters<br/>and valid extension?"}
    D -- No --> SKIP["Skip file"]
    D -- Yes --> E["Extract text and optional structured blocks"]
    E --> F{"Extraction succeeded and content nonempty?"}
    F -- No --> SKIP
    F -- Yes --> G["Normalize Unicode, mask text<br/>and build document metadata"]
    G --> H["Clean text and detect language"]
    H --> I{"Language supported?"}
    I -- No --> NOTOK["Write NOT_OK"]
    I -- Yes --> J["Embed document and run two-pass KeyBERT"]
    J --> K["Merge keyword and cosine weights<br/>stem, sort and format keywords"]
    K --> L["Run document compliance checks<br/>record human-review flag"]
    L --> M["Build prompt and call classification LLM"]
    M --> N{"Valid JSON object?"}
    N -- No --> BAD["Write NOT_OK"]
    N -- Yes --> O["Parse classification and optionally<br/>restore keyword surface forms"]
    O --> P["Add metadata and write OK result"]
    P --> Q{"Human-review flag set?"}
    Q -- Yes --> R["Also write HUMAN_REVIEW result"]
    Q -- No --> DONE["Report processing results"]
    R --> DONE
    BAD --> DONE
    NOTOK --> DONE
    SKIP --> DONE
    classDef check fill:#fff3cd,stroke:#8a6d3b,color:#111
    classDef process fill:#e6f1ff,stroke:#2870a8,color:#111
    classDef result fill:#e2f5e6,stroke:#33834a,color:#111
    class D,F,I,N,Q check
    class A,B,C,E,G,H,J,K,L,M,O process
    class SKIP,NOTOK,BAD,P,R,DONE result
```
