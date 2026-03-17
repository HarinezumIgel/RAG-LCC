# Acknowledgments

## Third-Party Libraries and Models

This project builds on excellent open-source software and pre-trained models. RAG‑LCC does not distribute these packages; users install them independently from upstream sources. For comprehensive licensing information and attribution for all third-party dependencies, see [3rdPartyLicenses/Licenses.md](3rdPartyLicenses/Licenses.md).

### Key Dependencies

- **ChromaDB** – Vector database for embedding storage and retrieval
- **Ollama** – Local LLM inference engine
- **HuggingFace Transformers** – Pre-trained NLP models and sentence transformers
- **NLTK** – Natural Language Toolkit for tokenization and stemming
- **PyTorch** – Deep learning framework
- **Argos Translate** – Language translation library
- **pywin32** – Python extensions for Windows COM automation (used for optional Microsoft Office document extraction)

### Pre-Trained Models

Model selections, quantization options, and embedding/cross-encoder configurations are managed in [Configuration/Config_Models.py](src/Configuration/Config_Models.py). Models referenced in this repository:

- **Embedding**: Snowflake Arctic-embed (or configurable alternative)
- **Cross-Encoder**: Built from HuggingFace model hub
- **LLM**: Ollama-compatible endpoint (commonly used with local Ollama deployments)

Model downloads and licensing are user-controlled; see the setup instructions in [README.md](README.md) and model disclaimers.

**Attribution:** Llama 3.1 and Llama Guard 3 — Built with Meta Llama 3. Licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved. By downloading and using these models, operators are bound by the model license terms.

### External Applications

- **Microsoft Office** – Optional. Required only for extracting text from Office formats (.doc(x), .ppt(x), .xls(x)) via COM automation. Microsoft Office is **not** included with or distributed by RAG‑LCC; users must obtain and license it independently.

## License

RAG-LCC itself is licensed under [LICENSE](LICENSE). All third-party software retains its original license; see [3rdPartyLicenses/Licenses.md](3rdPartyLicenses/Licenses.md).
