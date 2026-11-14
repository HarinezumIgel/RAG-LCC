# RAGChatService Pipeline Flow

The service entry point is [`RAGChatService.py`](../../src/Apps/RAGChatService.py).
Request handling and the streaming/buffered branches are in
[`ChatCompletionHandler.py`](../../src/Api/ChatCompletionHandler.py); the
shared answer path is [`Chatter.py`](../../src/Chat/Chatter.py).

```mermaid
flowchart TD
    A["Initialize FastAPI, models, executor and lock"] --> B["POST /v1/chat/completions"]
    B --> C{"Bearer authentication and request validation pass?"}
    C -- No --> ERR["Return authentication or validation response"]
    C -- Yes --> D{"OpenWebUI housekeeping request?"}
    D -- Yes --> FAST["Return housekeeping response without retrieval"]
    D -- No --> E["Acquire lock; configure request session<br/>and determine effective streaming"]
    E --> F{"Summarize command?"}
    F -- Yes --> SUMMARY["Select document; skip user-prompt checks"]
    F -- No --> G{"Prompt filter, LLM guard<br/>and intent filter pass?"}
    G -- No --> REJECT["Return compliance response"]
    G -- Yes --> H["Invoke shared Chatter retrieval and answer path"]
    SUMMARY --> H
    H --> I["Flow-controlled retrieval, selection and LLM call<br/>or clarification / no-results response"]
    I --> J{"Answer passes compliance?"}
    J -- No --> BLOCK["Return blocked response"]
    J -- Yes --> K["Optional grounding, confidence,<br/>filter results and marked-document links"]
    K --> L{"Effective streaming?"}
    L -- Yes --> M["Return SSE; generator releases lock"]
    L -- No --> N["Return buffered JSON; release lock"]
    REJECT --> RELEASE["Release lock"]
    BLOCK --> RELEASE
    ERR --> END["Done"]
    FAST --> END
    RELEASE --> END
    M --> END
    N --> END
    classDef check fill:#fff3cd,stroke:#8a6d3b,color:#111
    classDef process fill:#e6f1ff,stroke:#2870a8,color:#111
    classDef result fill:#e2f5e6,stroke:#33834a,color:#111
    class C,D,F,G,J,L check
    class A,B,E,H,I,K process
    class ERR,FAST,SUMMARY,REJECT,BLOCK,RELEASE,M,N,END result
```

When `mark_text` is enabled, requested streaming is downgraded to a buffered
reply. Otherwise, streaming callbacks can send tokens during the LLM call,
before answer compliance has completed. The `/v1/models` and optional
`/marked/{token_filename}` endpoints are separate from the chat-completions
path.
