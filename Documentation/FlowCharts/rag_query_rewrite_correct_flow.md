# Query Rewrite - Coreference Resolution Before Retrieval

The flow varies with the selected orchestration profile and session settings.
See [`RAGChatImpl.py`](../../src/Chat/RAGChatImpl.py),
[`PromptRewrite.py`](../../src/Chat/PromptRewrite.py), and
[`RetrievalGate.py`](../../src/Chat/RetrievalGate.py).

```mermaid
flowchart TD
    A["User query and session"] --> B{"New-topic command?"}
    B -- Yes --> C["Strip command, reset conversation<br/>and skip rewrite for this turn"]
    B -- No --> D["Prepare current turn"]
    C --> E["Detect language and translate if the<br/>flow and translation backend require it"]
    D --> E
    E --> F{"Chat context and rewrite enabled;<br/>history present; no forced skip?"}
    F -- No --> ORIGINAL["Keep input to rewriter unchanged"]
    F -- Yes --> G["Build topic summary from saved referents<br/>or recent assistant history"]
    G --> H["Call rewrite LLM with previous and current<br/>utterances and topic summary"]
    H --> I{"LLM response and JSON valid?"}
    I -- No --> ORIGINAL
    I -- Yes --> J["Read dependency, confidence,<br/>contextual and standalone rewrites, referents"]
    J --> K["Apply pronoun-substitution setting and<br/>check standalone rewrite against history"]
    K --> L{"Unresolved or invented referent?"}
    L -- Yes --> M["Mark underspecified; preserve old referents;<br/>sanitize when needed"]
    L -- No --> N["Keep valid rewrite / referents"]
    M --> O["Choose contextual rewrite if dependency,<br/>confidence and contextual text qualify;<br/>otherwise choose standalone"]
    N --> O
    ORIGINAL --> P["Detect rewritten-query language"]
    O --> P
    P --> Q["Optional strict-language retry and<br/>post-rewrite translation"]
    Q --> R{"RetrievalGate / intent gate blocks?"}
    R -- Yes --> T["Return clarification or refusal;<br/>do not retrieve"]
    R -- No --> U["Run enabled retrieval stages"]
```
