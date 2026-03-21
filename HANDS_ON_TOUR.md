# Example Usages

> **⚠️ Experimental Walkthrough**
>
> This document demonstrates example interactions and observed behaviors in a
> controlled laboratory environment.
>
> All outputs shown are illustrative. Actual results may vary depending on
> configuration, model behavior, data, and runtime conditions.
>
> Nothing in this walkthrough should be interpreted as a guarantee of behavior,
> correctness, safety, or non‑hallucination.

## 💻 View the CLI parameters and defaults

```Windows
python ./src/Apps/RAGLoad.py -h
python ./src/Apps/RAGChat.py -h
python ./src/Apps/DocClassify.py -h
```

## 📥 Load the documents in the Test folder

Default collection name is from COLLECTION key in Config_Global.py.
If you work with many collection, you must specify the collection you want to use: `--collection some_collection`.

```Windows
python .\src\Apps\RAGLoad.py --doc-dir TestDocs --collection MyTestDocs
```

## 💬 Chat with the documents in the Test folder

```Windows
python .\src\Apps\RAGChat.py --collection MyTestDocs
```

```text
🛠️  > collection! choose MyTestDocs or collection=MyTestDocs or pass on startup --collection MyTestDocs
press enter
```

Query:

```text
💬 Your actual query> where do hedgehogs live?
press enter
```

Change debug level:

```text
🛠️  > debug! and choose "Ollama response" (or debug=8)
press enter
```

Query: (You can enter it or toggle with shif up/down. This works for chat and settings)

```text
💬 Your actual query> what do elephants eat?
press enter
```

Reset debug level:

```text
🛠️  > debug! and choose "Standard" (or debug=3)
press enter
```

Set output tokens to low value:

```text
🛠️  >   max_output_token=10
press enter
```

Query:

```text
💬 Your actual query> what do elephants eat?
press enter
```

## 🧠 Demonstrating how insufficient context can increase hallucination risk

LLM hallucination cannot be reliably prevented and may occur even under ideal conditions.

Reset max_output_tokens and set context_size to 10. In this configuration, the LLM may only receive a truncated portion of the prompt, which increases the likelihood of ignoring grounding instructions. You get a warning:
⚠ context_size=10 is dangerously low. Ollama uses this as the total KV-cache for input + output. The full prompt (instructions, retrieved context, and query) will be physically truncated — the model may not see the grounding instructions or the retrieved documents and will hallucinate from its training data instead.

```text
🛠️  >  max_output_tokens-
🛠️  >  context_size=10
press enter
```

Query:

```text
💬 Your actual query> what do elephants eat?
press enter
```

Reset context_size:

```text
🛠️  >  context_size-
press enter
```

```text
💬 Your actual query> what do elephants eat?
press enter
```

```text
💬 Your actual query> tell me about dogs
press enter
```

List the available files and select Dogs.png and the narrow strategy since we work upon one file now.
There are 4 predefined strategies: ultra_wide, wide, medium and narrow. The predefined settings can be found in `Configuration/Config_RAGChat.py`. The wider strategies are configured to retrieve more chunks and use higher thresholds. The narrow strategies are configured to retrieve fewer chunks and use lower thresholds.
**See the session parameters change in the ▶ output**

```text
🛠️  > file! select Dogs.png from the list
press enter
🛠️  > strategy! and choose "NARROW" from the list (or strategy=narrow)
press enter
```

Now you are "fixed" upon the Dogs.png file: ▶ File Input: file='Dogs.png' path='.....'

Query

```text
💬 Your actual query> What animals are described in the documents?
press enter
```

Clear the file filter so all documents are searched again and choose MEDIUM strategy:

```text
🛠️  > file-
press enter
🛠️  > strategy=medium
```

Set threshold to 0.9:

```text
🛠️  > threshold=0.9
press enter
```

Query again. In this example, no chunks were returned because the threshold was set very high.
You may observe output similar to:
🟡 Suggested Action               Try lowering sensitivity. Current: 0.90

```text
💬 Your actual query> What animals are discussed in the documents?
```

Lower the threshold too much.  All animals in the RAG context will be listed:

```text
🛠️  > threshold=0.2
press enter
```

Query again:

```text
💬 Your actual query> Which animals are discussed in the documents?
press enter
```

Switch to ULTRA_WIDE selection strategy.

```text
🛠️  > strategy! or strategy=medium
press enter
```

Query:

```text
💬 Your actual query> give me a short summary about each discussed animal
press enter
```

Enable chat context. Watch the debug output and look for a message like this:
🔵 Add chat context               Upserted turn 1 for chat_name ... message

```text
🛠️  > use_chat_context=True
press enter
🛠️  > debug=3
press enter
```

```text
💬 Your actual query> Tell me for each discussed animal whether they are mammals or not
```

Now the content of the discussion since use_chat_context was enabled is also available for the LLM:
Look for output like this:

```text
⚪ Selected                       1. Chat Context = Chat Context
⚪ Selected                       2. File Name = Cats.md
⚪ Selected                       3. Chat Context = Chat Context
```

Query:

```text
💬 Your actual query> give me a short summary about each discussed animal
press enter
```

Deactivate chat context. Chat context is no longer included in subsequent queries

```text
🛠️  > use_chat_context=False
press enter
```

Query:

```text
💬 Your actual query> Tell me for each discussed animal whether they are mammals or not
```

Press twice enter to exit.

---

### 🗃️ Work with a second collection

The next example shows how you can switch between different collections you created. One could be about animals and another about plants.

Place some documents in a folder `yourpath` or use the TestDocs documents and load them

```Windows
python .\src\Apps\RAGLoad.py  --doc-dir `yourpath|TestDocs` --collection My2ndCollection
```

Start RAGChat.py:

```Windows
python .\src\Apps\RAGChat.py --collection My2ndCollection
```

Select the collection (if not passed as `--collection` parameter). We use the ULTRA_WIDE strategy for exploring.

```text
🛠️  > collection! and choose My2ndCollection
🛠️  > strategy! or strategy=ultra_wide
press enter
```

When `FileHash: [SECRET]` appears in the query output, the Masker has matched configured patterns and replaced the corresponding text in this example.

```text
💬 Your actual query> Tell me what is in the provided context
press enter
```

Start a new chat (so far you are in the MyFirstChat chat, defined in Config_RAGChat.py `_DEFAULT_CHAT_NAME` )

```text
🛠️  > chat_name=newchat
press enter
```

```text
💬 Your actual query> Tell me about hedgehogs
press enter
```

Switch back to previous chat:

```text
🛠️  > chat_name! and choose "MyFirstChat" or chat_name=MyFirstChat
```

```text
💬 Your actual query> use shift up/down key and you are in the history (query and settings) of the previous chat. If you enable use_chat_context=True the chat context is filtered by chat_name. So each chat_name has its own chat_context
press enter
press enter again to exit.

---

## 🏷️ Classify the documentation

This demonstrates `DocClassify.py`.

```Windows
python .\src\Apps\DocClassify.py --doc-dir yourpath
```

When the program is done, you see a message where the output `.csv` / `.xlsx` files can be viewed (in the logs directory).
Open the one labeled `DocClassify_OK<date>.csv` / `.xlsx`.

### 🔑 Define your classification keys

The file `Config_DocClassify.py` contains detailed instructions how you can change classification keys.

### ✏️ Change provided example prompt

The goal is to extract information from the *context* only about the animals discussed.

| Animal | Habitat |
| --- | --- |
| cats | Dont know |
| dogs | Dont know |
| elephant | elephant: forest |
| fish, shark, ray, amphibians | fish: reef, shark: ocean, ray: ocean, amphibian: Dont know |
| hedgehogs | Dont know |
| Dont know | Dont know |
| horse | Dont know |

Try this example prompt. "Dont know" reflects the model not finding explicit information in the provided keywords and therefore not inferring additional details in this example.

```Text
_PROMPT_CLASSIFY_MISTRAL = (
    "You are given a dictionary of keywords extracted from a document, "
    "each key is a keyword and its value is a relevance weight. "
    "You must analyze ONLY these keywords. Do NOT use any external knowledge.\n\n"

    "Determine the following fields:\n"
    "1. Classification: category labels (up to {CLASSIFICATION_WORD_CNT} words).\n"
    "2. Purpose: brief summary (up to {SUMMARY_SENTENCE_CNT} sentences).\n"
    "3. Language: detected document language.\n"
    "4. Topic: short topic phrase.\n"
    "5. Animal: animals explicitly mentioned in the keywords.\n"

    # ✅ CHANGED FIELD 6
    "6. Habitat: For EACH animal from field 5, determine where it lives.\n"
    "   RULES FOR HABITAT:\n"
    "   - You MUST use ONLY the provided keywords.\n"
    "   - Do NOT use biological or real-world knowledge.\n"
    "   - A habitat may be given ONLY if a keyword explicitly states or clearly\n"
    "     contains habitat information (e.g. a keyword phrase like "
    "     \"reef fish\", \"river salmon\", \"forest deer\").\n"
    "   - If habitat information is NOT explicitly present in the keywords,\n"
    "     you MUST answer \"Dont know\" for that animal.\n"
    "   - Do NOT infer habitats from animal names alone.\n"
    "   - Do NOT invent or generalize habitats.\n\n"

    "Habitat output format (single string):\n"
    "animal: habitat_or_Dont know, animal2: habitat_or_Dont know\n\n"

    "IMPORTANT:\n"
    "- Return ONLY ONE valid JSON object.\n"
    "- No explanations, no comments, no markdown.\n"
    "- The JSON object must contain exactly these keys:\n"
    "\"Classification\", \"Purpose\", \"Language\", \"Topic\", \"Animal\", \"Habitat\".\n"
    "- Every value MUST be a plain string (no arrays, no nested objects).\n"
    "- Only mention animals that actually appear in the keywords.\n\n"

    "Example output:\n"
    "{{"
    "\"Classification\": \"Science\", "
    "\"Purpose\": \"A document summary\", "
    "\"Language\": \"English\", "
    "\"Topic\": \"Marine biology\", "
    "\"Animal\": \"fish\", "
    "\"Habitat\": \"fish: reef\""
    "}}"

    "Weighted Keywords:\n"
)
```

**Don't forget to adjust the classification key and change it from "Mammal" to "Habitat":**

```Text
_YOUR_CLASSIFICATION_KEYS = [
    "Classification",
    "Purpose",
    "Topic",
    "Animal",
    "Habitat", # was: "Mammal"
    "Language",
]  # For user‑defined keys beyond the core set
```

### 🏚️ Extraction & KeyBERT variant tuning

`Config_DocClassify.py` ships three named presets — `STRICT`, `BALANCED`,
and `RECALL` — for both the extraction LLM parameters and the KeyBERT
keyword-extraction passes.  Two independent selectors (`_ACTIVE_EXTRACTION_CONFIG`
and `_ACTIVE_KEYBERT_CONFIG`) let you mix and match.

For the full variant matrix and consumer mapping see
[Extraction & KeyBERT Variant Configuration in ARCHITECTURE.md](ARCHITECTURE.md#extraction--keybert-variant-configuration).

### 🚫 Add banned word

Requirement: Argostranslate and language package en → de must be installed. See [5. Install argos translate in README.md](README.md#5-install-argos-translate).

Edit `Config_Banned.py` and add `Pferd` (german for horse) to the `_STRICT_BANNED` wordlist. Start RAGLoad.py:

```Windows
python .\src\Apps\RAGLoad.py  --doc-dir `TestDocs` --collection BannedHorseCollection
```

You may observe that the `Pferde.pdf` triggers a message similar to:

```Text
🟡 KeyWrdChk Depth                3 algos passed threshold vs. required 3
🟢 KeyWrdChk Breadth              3 algos had a score vs. required 4
🟡 Vector store                   No chunks inserted  to Test_ChatContext. All 1 chunk(s) were skipped.
```

---

## 🧪 Further Experiments

Below are ideas for configuration changes you can try. Each explains **what** to change and **what behaviour** you may observe.

### 🔓 1. Loosen the consensus rules to block more documents

In `Config_Banned.py`, the RAGLoad pipeline requires **3** algorithms above threshold (depth) **and** **4** algorithms with any score (breadth) before a chunk is blocked. Try lowering both values:

```python
"REQUIRED_ALGOS_ABOVE_THRESHOLD": 1,
"REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 2,
```

**What changes:** In this configuration, more chunks are likely to be flagged and rejected during loading. Even a single algorithm scoring above its threshold is enough to block a chunk. This is useful when you want a very strict ingestion policy, but expect more false positives — harmless chunks may be rejected because one algorithm happened to score high.

After editing, update `_BANNED_CONFIG_HASH` in `Config_Global.py` (the required hash is printed at startup).

### 🔒 2. Tighten the consensus rules to let more documents through

Raise the consensus values in the RAGLoad pipeline:

```python
"REQUIRED_ALGOS_ABOVE_THRESHOLD": 4,
"REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE": 4,
```

**What changes:** All four algorithms must agree before a chunk is blocked. This makes the filter very permissive — only clearly problematic content gets rejected. Fewer false positives, but some borderline content may slip through.

### 🔧 3. Disable fuzzy regex matching

In the `Regex` section of any pipeline in `Config_Banned.py`, set:

```python
"FUZZY_REGEX_EVAL_AFTER_HARD": False,
```

**What changes:** The regex algorithm will only perform strict word-boundary matching. Fuzzy anchored matches (e.g. "bic" matching "bicycle") will no longer be found. This reduces false positives from the regex algorithm but may miss content that uses abbreviations, misspellings, or partial word overlaps.

### 🧩 4. Change chunk size and observe retrieval differences

In `Config_Global.py`, modify the chunking parameters:

```python
_CHROMA_EMBED_AND_RETRIEVE_PARAMS = {
    "CHUNK_SIZE": 128,       # was 256
    "CHUNK_OVERLAP": 16,     # was 32
    ...
}
```

**What changes:** Documents are split into smaller pieces. RAGChat will return more, shorter chunks for each query. This can improve precision (each chunk is more focused) but may lose context that spans across chunk boundaries. Existing embeddings in a collection were created with the old chunk size, so after changing chunk size you **must** re-embed your documents. You have two options:

- **Re-create the existing collection:** set `CHROMA_COLLECTION_KEEP = False` in `Config_Global.py` (or pass `--chroma_collection_keep False` on the CLI) and run RAGLoad again.
- **Use a new collection name:** e.g. `--collection Experiment_SmallChunks` so you can compare results side-by-side with the old collection.

Try the opposite too — set `CHUNK_SIZE` to `512` with `CHUNK_OVERLAP` of `64` for longer chunks that preserve more context but may include irrelevant surrounding text.

### 🔍 5. Tune HNSW neighbor exploration

The same `_CHROMA_EMBED_AND_RETRIEVE_PARAMS` dict in `Config_Global.py` contains two parameters that control how thoroughly ChromaDB's HNSW index searches for similar vectors:

```python
_CHROMA_EMBED_AND_RETRIEVE_PARAMS = {
    ...
    "NEIGHBORS_ON_LOAD": 512,    # neighbours explored when building the index (RAGLoad)
    "NEIGHBORS_RETRIEVE": 512,   # neighbours explored when querying the index (RAGChat)
}
```

- **`NEIGHBORS_ON_LOAD`** affects index build quality. A higher value makes RAGLoad slower but produces a more accurate index. Try lowering it to `64` — loading will be faster, but the index may miss some connections between similar chunks, which can slightly reduce retrieval quality later.
- **`NEIGHBORS_RETRIEVE`** affects query-time search quality. A higher value makes each RAGChat query explore more of the index graph, returning better results at the cost of speed. Try lowering it to `64` and compare the chunks returned for the same query — you may notice that some relevant chunks are no longer found.

**What changes:** Lower values speed up loading and querying but may reduce recall (fewer relevant chunks found). Higher values improve accuracy but increase computation time. The effect is most noticeable with large collections; for the small `TestDocs` set the difference may be subtle. Unlike chunk size changes, you do **not** need to re-create the collection when changing `NEIGHBORS_RETRIEVE` — it takes effect immediately on the next query. However, changing `NEIGHBORS_ON_LOAD` only affects newly indexed chunks, so for a fair comparison you should reload the collection.

### 🎯 6. Experiment with RAGChat retrieval strategies

Start RAGChat and switch strategies interactively to see how retrieval quality changes:

```text
🛠️  > strategy=narrow
💬 Your actual query> What do elephants eat?

🛠️  > strategy=ultra_wide
💬 Your actual query> What do elephants eat?
```

**What changes:** `NARROW` fetches fewer chunks with a high relevance bar (threshold 0.75) — answers are precise but may miss relevant context spread across documents. `ULTRA_WIDE` fetches up to 3000 vector candidates with a low threshold (0.20) — you get comprehensive answers but the LLM receives much more context, which can dilute precision or slow down responses.

### 📊 7. Lower individual algorithm thresholds

Pick one algorithm — for example Jaccard in the RAGLoad pipeline — and lower its threshold:

```python
"Jaccard": {
    "CHAR_NGRAM_RANGE": (4, 6),
    "THRESHOLD": 0.40,        # was 0.75
    "THRESHOLD_MIN": 0.2,     # was 0.5
},
```

**What changes:** Jaccard will flag more content as matching banned words because even moderate character overlap is now enough to exceed the threshold. Combined with the consensus rules, this means Jaccard will contribute a "yes" vote more often, making the overall filter stricter for Jaccard-sensitive content (short words, partial overlaps).

### 🎭 8. Add your own masking rule

In `Config_Banned.py`, find the `_STRICT_MASKING_REGEXES` dictionary and add a new rule:

```python
{
    "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
    "mask": "[MASKED-ID]",
    "enabled": True,
    "priority": 100,
    "desc": "Custom ID pattern",
},
```

**What changes:** Any text matching the pattern (e.g. `123-45-6789`) will be replaced with `[MASKED-ID]` before the content is stored or returned. You can verify this by placing a document containing such a pattern in your document folder and loading it — the masker output will show `[MASKED-ID]` instead of the original value.

### 🔄 9. Switch the compliance-check LLM

In `Config_Models.py`, change the compliance checker from Llama Guard to the same model used for generation:

```python
_LLM_CHK = "mistral"   # was "llama_guard"
```

**What changes:** Prompt compliance checks are now performed by Mistral instead of the dedicated Llama Guard safety model. Llama Guard is specifically trained for safety classification and returns structured safe/unsafe labels. Mistral will still perform the check using the configured prompt template, but its judgements may differ — it could be more lenient or flag different content. This is useful for comparing how different models evaluate the same prompts.

After editing, update `_MODELS_CONFIG_HASH` in `Config_Global.py`.

### ⚙️ 10. Try different DocClassify extraction presets

In `Config_DocClassify.py`, switch the extraction and KeyBERT presets:

```python
_ACTIVE_EXTRACTION_CONFIG = "RECALL"     # was "STRICT"
_ACTIVE_KEYBERT_CONFIG = "BALANCED"      # was "STRICT"
```

**What changes:** `RECALL` uses a slightly higher temperature and wider top-k sampling, so the LLM extracts more classification labels (higher recall, possibly more noise). `BALANCED` KeyBERT extracts more keyword candidates (80 phrases vs. 60). Together this produces richer but potentially noisier classification output. Compare the resulting CSV files side-by-side to see which preset works better for your documents.

### 🧲 11. Enable the Cosine algorithm

In `Config_Banned.py`, uncomment the Cosine entries in the pipeline and in `ALGOS_TO_PROCESS`:

```python
"Cosine": {"THRESHOLD": 0.45, "THRESHOLD_MIN": 0.2},
```

```python
"ALGOS_TO_PROCESS": {
    "Regex": True,
    "Jaccard": True,
    "BM25": True,
    "Cosine": True,
    "Keybert": True,
},
```

**What changes:** A fifth algorithm votes in the consensus. Since Cosine and KeyBERT both use embedding-based similarity, they tend to produce correlated scores. This effectively gives embedding similarity more weight in the consensus decision. If you also raise `REQUIRED_DIFFERENT_ALGOS_HAVE_A_SCORE` to `5`, all five algorithms must participate — making the breadth check very strict.

### 💡 General tips

- When experimenting, changing **one thing at a time** can make it easier to observe effects.
- Use `DEBUG_LEVEL = 4` in `Config_Global.py` to see per-algorithm scores and understand why a chunk was accepted or rejected.
- After editing `Config_Banned.py` or `Config_Models.py`, remember to update the corresponding hash in `Config_Global.py` — the new hash is printed when you start the application.
- When experimenting with RAGLoad settings, use a **new collection name** (e.g. `--collection Experiment1`) so you can compare results without overwriting your previous collection.
