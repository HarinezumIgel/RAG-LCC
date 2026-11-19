# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned_Detection.py,
#     Config_Banned_Content.py, Config_Banned_Prompts.py,
#     Config_Models.py, Config_Global.py
# - Entries starting with $ are indirect lookups
# - Top-level settings must be uppercase
# -------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Model Configuration & Governance Notes
# -----------------------------------------------------------------------------
#
# This file declares the model configurations used by RAG‑LCC, including:
#   - Embedding models
#   - Cross‑encoder models
#   - Local LLMs (via Ollama)
#   - Safety / guard models
#
# The entries in this file describe which models may be referenced at runtime.
# Model availability, installation, and execution depend on external tooling,
# local environment state, and user configuration.
#
# -----------------------------------------------------------------------------
# License Consent Handling
# -----------------------------------------------------------------------------
#
# RAG‑LCC tracks license consent for the models selected by the _ACTIVE_*
# slots below (only those whose USED_BY includes the running app).
# Alternative models in _MODELS that are not selected are not checked.
# Consent handling is applied uniformly, independent of how or where a model
# runtime is installed.
#
# Consent is requested when:
#   - an _ACTIVE_* slot points to a model without recorded consent
#     (switching back to a previously consented model does not prompt)
#   - the _MODELS entry of an active model changes (config hash differs)
#
# When consent is required, the system may:
#   - Present associated license text for review
#   - Request an explicit acknowledgment from the user
#   - Record consent metadata in ModelGovernance/licenses/
#
# Recorded consent is used as a gating signal for model usage within RAG‑LCC.
#
# -----------------------------------------------------------------------------
# Model Retrieval and Installation Scope
# -----------------------------------------------------------------------------
#
# RAG‑LCC itself does not install or manage local model runtimes.
#
# - Ollama models are expected to be managed externally (e.g. via `ollama pull`)
# - Any license terms associated with third‑party tooling are handled outside
#   the scope of RAG‑LCC
#
# For models sourced from Hugging Face:
#   - If internet access is enabled (see Config_Internet_Env.py), and
#   - If license consent has been recorded,
#   the system may retrieve required model artifacts automatically.
#
# Download activity, when it occurs, is recorded in the ModelGovernance
# directory together with relevant metadata (e.g. source reference, revision).
#
# If required models are unavailable or consent is not recorded, the associated
# pipeline components may be disabled at runtime with an explanatory message.
#
# -----------------------------------------------------------------------------
# Configuration Hash
# -----------------------------------------------------------------------------
#
# Changes to the model definitions below affect the configuration fingerprint.
# The active configuration hash is compared against _MODELS_CONFIG_HASH in
# Config_Global.py and displayed at runtime.
#
# This mechanism supports reproducibility and audit‑oriented workflows.
#
# -----------------------------------------------------------------------------
# Implementation Selectors
# -----------------------------------------------------------------------------
#
# Each _ACTIVE_<ROLE> variable below selects an implementation identifier
# (impl) for one model role. The value must match a top-level key in the
# _MODELS dictionary defined later in this file. The role is the selector
# name without "_ACTIVE" (Helpers/ModelSelection.py):
#
#     _MODELS[<impl>][_<ROLE>]
#
# For example, if _ACTIVE_LLM = "mistral", the LLM configuration is read from:
#
#     _MODELS["mistral"]["_LLM"]
#
# Exception: _ACTIVE_ENDPOINT ("ollama" or "vllm") selects the provider impl;
# its single role is named after the provider (_OLLAMA or _VLLM). It holds the
# generic server settings for every model (see "Provider slots" below).
#
# To change a model, update the corresponding impl value to another key that
# defines the required role.
#
# Provider slots: model settings that differ between Ollama and vLLM go into
# nested "OLLAMA" / "VLLM" dicts of a role entry, e.g.
#
#     "OLLAMA": {"MODEL": "mistral:7b"},
#     "VLLM":   {"MODEL": "mistral_7b", "REQUEST_TIMEOUT": 300.0},
#
# Model-specific values always supersede the global/generic ones. Precedence,
# most specific wins:
#   1. the model's slot matching _ACTIVE_ENDPOINT (the other slot is ignored)
#   2. the model's role-level keys
#   3. the generic endpoint entry (_MODELS.ollama._OLLAMA / _MODELS.vllm._VLLM)
#      and global settings such as _HF_API_KEY
# Examples:
#   - "VLLM": {"REQUEST_TIMEOUT": 300.0} in mistral._LLM raises the timeout for
#     that model only; all other models keep _VLLM.REQUEST_TIMEOUT.
#   - "OLLAMA": {"HF_API_KEY": "hf_..."} in llama._LLM_REWRITE_PROMPT uses that
#     token for this model when _ACTIVE_ENDPOINT = "ollama", instead of the
#     role-level HF_API_KEY and the global _HF_API_KEY.
# Any endpoint setting (BASE_URL, API_KEY, REQUEST_TIMEOUT, BATCH_SIZE,
# API_STYLE, ...) can be overridden this way.
#
# Embedders and cross-encoders have no OLLAMA slot (they run in-process with
# Ollama). Their role-level MODEL is the Hugging Face id; VLLM.MODEL is the
# alias served by vLLM/LiteLLM. Without an own VLLM.BASE_URL they use the
# _VLLM server root (its BASE_URL without "/chat/completions"); the client
# appends /embeddings, /rerank or /score. Chroma collections store the effective
# MODEL, the stable _ACTIVE_EMBED alias and EMBEDDER_BITS. Matching alias and
# bits allow reuse across endpoints/hosts despite different MODEL names.
# Keep an alias tied to the same underlying embedder; use a different key
# when changing the model.
#
# USED_BY lists the apps (_FRIENDLY_NAME: RAGChat, RAGLoad, RAGChatService,
# DocClassify) that use an entry. License consent is only requested for active
# entries whose USED_BY includes the running app.
#
# -----------------------------------------------------------------------------

from typing import Any

_ACTIVE_LLM_CHK = "llama_guard"  # mistral llama_guard
_ACTIVE_LLM = "qwen2_5_7b_instruct"  # mistral llama qwen2_5_7b_instruct
_ACTIVE_LLM_REWRITE_PROMPT = "qwen2_5_7b_instruct"  # mistral llama qwen2_5_7b_instruct
_ACTIVE_LLM_JUDGE = "qwen2_5_14b_instruct"
_ACTIVE_EMBED = "bge_m3"  # bge_m3 snowflake
_ACTIVE_CROSS = "bge_reranker"  # bge_reranker mmarco
# For configuration-driven orchestration of models, LLMs, embedders, and
# rerankers, see the companion project:
# https://github.com/HarinezumIgel/harinezumigel-llm-stack
# LLM provider: "vllm" or "ollama". Also selects where the embedder and
# cross-encoder run: vllm -> remote vLLM/LiteLLM server (_MODELS.vllm._VLLM,
# overridable per model in its "VLLM" slot); ollama -> in-process HF /
# sentence-transformers.
_ACTIVE_ENDPOINT = "ollama"  # ollama vllm

# Service entries (_MODELS.openwebui._OPENWEBUI,
# _MODELS.ragchatservice._RAGCHATSERVICE) have no selector: they are fixed and
# always in use (license consent still follows their USED_BY).

# Optional Hugging Face access token used for gated/private model downloads.
# Leave empty for public models and anonymous access.
# Global fallback: a model's own HF_API_KEY (role-level or in its "OLLAMA" /
# "VLLM" slot) supersedes this value.
_HF_API_KEY = ""

# _MODELS hierarchy: _MODELS[impl][role] -> config dict
_MODELS: dict[str, dict[str, dict[str, Any]]] = {
    "snowflake": {
        "_EMBED": {
            "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
            "VLLM": {"MODEL": "snowflake-embedder"},
            "FRIENDLY_NAME": "Snowflake Arctic Embed L v2.0",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "Embedder: Snowflake arctic-embed-l-v2.0 is the newest addition to the suite of embedding models Snowflake has released optimizing for retrieval performance and inference efficiency",
            "MODEL_CARD": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
            "USED_BY": ["RAGChat", "RAGLoad", "RAGChatService", "DocClassify"],
        },
    },
    # BGE-M3 embedder (dense mode only; 1024 dims, no query/passage prefix).
    # Switching _ACTIVE_EMBED to "bge_m3" requires re-running RAGLoad and
    # DocClassify: existing Chroma collections are stamped with the old model.
    "bge_m3": {
        "_EMBED": {
            "MODEL": "BAAI/bge-m3",
            "VLLM": {"MODEL": "bge-m3-embedder"},
            "FRIENDLY_NAME": "BAAI BGE-M3",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/BAAI/bge-m3",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/LICENSE",
            "COMPLIANCE_MSG": "Embedder: BAAI BGE-M3 is a multilingual (100+ languages) embedding model supporting inputs up to 8192 tokens. Only its dense mode is used.",
            "MODEL_CARD": "https://huggingface.co/BAAI/bge-m3",
            "USED_BY": ["RAGChat", "RAGLoad", "RAGChatService", "DocClassify"],
        },
    },
    # BGE multilingual reranker (cross-encoder, single relevance logit).
    "bge_reranker": {
        "_CROSS": {
            "MODEL": "BAAI/bge-reranker-v2-m3",
            "VLLM": {
                "MODEL": "bge-reranker",
                # "rerank": Cohere-style /rerank (LiteLLM proxy or vLLM)
                # "score":  vLLM-native /score (direct vLLM server only)
                "API_STYLE": "rerank",
            },
            "FRIENDLY_NAME": "BAAI BGE Reranker v2 M3",
            "REVISION": "",
            # No query prefix: the model scores plain (query, passage) pairs.
            "QUERY_INSTRUCTION": "",
            "SOURCE": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "Cross Encoder: BAAI bge-reranker-v2-m3 is a lightweight multilingual reranker built on bge-m3.",
            "MODEL_CARD": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
    },
    "mmarco": {
        "_CROSS": {
            "MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "VLLM": {
                "MODEL": "mmarco-reranker",
                # "rerank": Cohere-style /rerank (LiteLLM proxy or vLLM)
                # "score":  vLLM-native /score (direct vLLM server only)
                "API_STYLE": "rerank",
            },
            "FRIENDLY_NAME": "Cross Encoder MMARCO MiniLM v2 L12 H384",
            "REVISION": "",
            # QUERY_INSTRUCTION: optional prefix prepended to the query before
            # scoring.  Standard BERT cross-encoders (including mmarco) do not
            # need one — leave empty.  Only instruction-tuned rerankers that
            # document a query prefix need it (bge-reranker-v2-m3 does not).
            "QUERY_INSTRUCTION": "",
            "SOURCE": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "Cross Encoder: This model was trained on the MMARCO dataset. It is a machine translated version of MS MARCO using Google Translate.",
            "MODEL_CARD": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
    },
    "mistral": {
        "_LLM": {
            "OLLAMA": {"MODEL": "mistral:7b"},
            "VLLM": {"MODEL": "mistral_7b"},
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "LLM: Mistral 7B is a 7.3B parameter model",
            "TAG": "LLM: Mistral 7B is a 7.3B parameter model",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_MISTRAL",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
        "_LLM_REWRITE_PROMPT": {
            "OLLAMA": {"MODEL": "mistral:7b"},
            "VLLM": {"MODEL": "mistral_7b"},
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "LLM: Mistral 7B is a 7.3B parameter model",
            "TAG": "LLM: Mistral 7B is a 7.3B parameter model",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_TOPIC_DETECT": "_PROMPT_TOPIC_DETECT",
            "PROMPT_QUERY_EXPAND": "_PROMPT_QUERY_EXPAND",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
        "_LLM_CHK": {
            "OLLAMA": {"MODEL": "mistral:7b"},
            "VLLM": {"MODEL": "mistral_7b"},
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "LLM: Mistral 7B is a 7.3B parameter model",
            "TAG": "LLM: Mistral 7B is a 7.3B parameter model",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },
    "qwen2_5_7b_instruct": {
        "_LLM": {
            "OLLAMA": {"MODEL": "qwen2.5:7b"},
            "VLLM": {"MODEL": "openai/qwen2_5_7b_instruct"},
            "FRIENDLY_NAME": "Qwen2.5 7B Instruct",
            "SOURCE": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "LLM: Qwen2.5-7B-Instruct is licensed under the Apache License, Version 2.0.",
            "TAG": "LLM: Qwen2.5 7B Instruct",
            "MODEL_CARD": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_MISTRAL",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
        "_LLM_REWRITE_PROMPT": {
            "OLLAMA": {"MODEL": "qwen2.5:7b"},
            "VLLM": {"MODEL": "openai/qwen2_5_7b_instruct"},
            "FRIENDLY_NAME": "Qwen2.5 7B Instruct",
            "SOURCE": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "LLM: Qwen2.5-7B-Instruct is licensed under the Apache License, Version 2.0.",
            "TAG": "LLM: Qwen2.5 7B Instruct",
            "MODEL_CARD": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "PROMPT_TOPIC_DETECT": "_PROMPT_TOPIC_DETECT",
            "PROMPT_QUERY_EXPAND": "_PROMPT_QUERY_EXPAND",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
        "_LLM_CHK": {
            "OLLAMA": {"MODEL": "qwen2.5:7b"},
            "VLLM": {"MODEL": "openai/qwen2_5_7b_instruct"},
            "FRIENDLY_NAME": "Qwen2.5 7B Instruct",
            "SOURCE": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "LLM: Qwen2.5-7B-Instruct is licensed under the Apache License, Version 2.0.",
            "TAG": "LLM: Qwen2.5 7B Instruct",
            "MODEL_CARD": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },
    "llama": {
        "_LLM": {
            "OLLAMA": {"MODEL": "llama3.1:8b"},
            "VLLM": {"MODEL": "llama3_1_8b"},
            "FRIENDLY_NAME": "Llama 3.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "SOURCE": "https://github.com/meta-llama/llama-models/tree/main/models/llama3_1",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
            "COMPLIANCE_MSG": "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_LLAMA",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
        "_LLM_REWRITE_PROMPT": {
            "OLLAMA": {"MODEL": "llama3.1:8b"},
            "VLLM": {"MODEL": "llama3_1_8b"},
            "FRIENDLY_NAME": "Llama 3.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "SOURCE": "https://github.com/meta-llama/llama-models/tree/main/models/llama3_1",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
            "COMPLIANCE_MSG": "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright \u00a9 Meta Platforms, Inc. All Rights Reserved",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
            "PROMPT_TOPIC_DETECT": "_PROMPT_TOPIC_DETECT",
            "PROMPT_QUERY_EXPAND": "_PROMPT_QUERY_EXPAND",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
        "_LLM_CHK": {
            # This configuration uses llama3.1 and the mistral prompt check prompt
            "OLLAMA": {"MODEL": "llama3.1:8b"},
            "VLLM": {"MODEL": "llama3_1_8b"},
            "FRIENDLY_NAME": "Llama 3.1",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "SOURCE": "https://github.com/meta-llama/llama-models/tree/main/models/llama3_1",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
            "COMPLIANCE_MSG": "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
            # Token budget — hardware/policy cap for this model
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },
    "llama_guard": {
        "_LLM_CHK": {
            "OLLAMA": {"MODEL": "llama-guard3:8b"},
            "VLLM": {"MODEL": "llama_guard3_8b"},
            "FRIENDLY_NAME": "Llama Guard 3",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "SOURCE": "https://ollama.com/library/llama-guard3",
            "UPSTREAM_SOURCE": "https://huggingface.co/meta-llama/Llama-Guard-3-8B",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://huggingface.co/meta-llama/Llama-Guard-3-8B/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "Llama Guard 3 is licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://ollama.com/library/llama-guard3",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_LLAMA_GUARD",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_LLAMA_GUARD",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,  # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 64,  # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 64,  # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },
    "qwen2_5_14b_instruct": {
        "_LLM_JUDGE": {
            "OLLAMA": {"MODEL": "qwen2.5:14b"},
            "VLLM": {"MODEL": "openai/qwen2_5_14b_instruct"},
            "FRIENDLY_NAME": "Qwen2.5 14B Instruct",
            "SOURCE": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "Judge: Qwen2.5-14B-Instruct is licensed under the Apache License, Version 2.0.",
            "MODEL_CARD": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 256,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
    },
    # judge_small
    "qwen2_5_1_5b_instruct": {
        "_LLM_JUDGE": {
            "OLLAMA": {"MODEL": "qwen2.5:1.5b"},
            "VLLM": {"MODEL": "openai/qwen2_5_1_5b_instruct"},
            "FRIENDLY_NAME": "Qwen2.5 1.5B Instruct (Small Judge)",
            "SOURCE": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
            "HF_API_KEY": "",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "Judge small: Qwen2.5-1.5B-Instruct is licensed under the Apache License, Version 2.0.",
            "MODEL_CARD": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 256,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
    },
    "ollama": {
        "_OLLAMA": {
            "PROVIDER": "ollama",
            "FRIENDLY_NAME": "Ollama Local LLM Provider",
            "SOURCE": "https://github.com/ollama/ollama",
            "BASE_URL": "http://<ollama host>:11434/api/generate",  # server IP, localhost or Docker host.docker.internal
            "API_KEY": "",
            "STREAMING_REQ": False,
            "TRY_FALLBACK_URLS": True,  # True: try localhost/docker fallbacks on failure  False: only probe BASE_URL
            "USE_GPU": True,
            "REQUEST_TIMEOUT": 120.0,
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/ollama/ollama/main/LICENSE",
            "MODEL_LICENSES_NOTE": "Each model has its own license—see `ollama list`",
            "COMPLIANCE_MSG": "OLLAMA Local LLM Provider",
            "MODEL_CARD": "https://ollama.com",
            "USED_BY": ["RAGChat", "RAGLoad", "RAGChatService", "DocClassify"],
        },
    },
    "vllm": {
        "_VLLM": {
            "PROVIDER": "vllm",
            "FRIENDLY_NAME": "vLLM OpenAI-compatible Provider",
            "SOURCE": "https://github.com/vllm-project/vllm",
            # Chat endpoint. Embedders/cross-encoders use the same server root
            # (BASE_URL without "/chat/completions") unless their "VLLM" slot
            # sets an own BASE_URL.
            "BASE_URL": "http://<vllm host>:4000/v1/chat/completions",  # server IP, localhost or Docker host.docker.internal
            # Bearer token for the vLLM / LiteLLM server (might need sk- prefix).
            "API_KEY": "",
            "STREAMING_REQ": False,
            "TRY_FALLBACK_URLS": True,  # True: try localhost/docker fallbacks on failure  False: only probe BASE_URL
            "USE_GPU": True,
            "REQUEST_TIMEOUT": 600.0,
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://raw.githubusercontent.com/vllm-project/vllm/main/LICENSE",
            "MODEL_LICENSES_NOTE": "Model licenses depend on the served checkpoint.",
            "COMPLIANCE_MSG": "vLLM OpenAI-compatible backend",
            "MODEL_CARD": "https://github.com/vllm-project/vllm",
            "USED_BY": ["RAGChat", "RAGLoad", "RAGChatService", "DocClassify"],
        },
    },
    "openwebui": {
        "_OPENWEBUI": {
            "FRIENDLY_NAME": "OpenWebUI (OpenAI-compatible UI)",
            "SOURCE": "https://github.com/open-webui/open-webui",
            "BASE_URL": "http://<openwebui host>:3000",  # server IP, localhost or Docker host.docker.internalc
            "LICENSE": "Modified BSD-3-Clause (with branding clause)",
            "LICENSE_URL": "https://raw.githubusercontent.com/open-webui/open-webui/main/LICENSE",
            "COMPLIANCE_MSG": "OpenWebUI is licensed under a modified BSD 3-Clause license with a branding-preservation clause. See LICENSE and LICENSE_HISTORY in the Open WebUI repository.",
            "MODEL_CARD": "https://github.com/open-webui/open-webui#readme",
            "USED_BY": ["RAGChatService"],
        },
    },
    "ragchatservice": {
        "_RAGCHATSERVICE": {
            "FRIENDLY_NAME": "RAGChatService (OpenAI-compatible REST API)",
            "SOURCE": "https://github.com/HarinezumIgel/RAG-LCC",
            "HOST": "<RAGChatService host>",  # Safest: loopback; opt in to remote access explicitly
            "PORT": 11435,  # Service port
            "BASE_URL": "http://localhost:11435",  # Full service URL
            # Bearer token that OpenWebUI must send (not a Hugging Face key).
            "API_KEY": "",  # Bearer token for authenticating incoming requests from OpenWebUI
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/HarinezumIgel/RAG-LCC/main/LICENSE",
            "COMPLIANCE_MSG": "RAG-LCC is licensed under the MIT License. Copyright (c) 2026 @HarinezumIgel",
            "USED_BY": ["RAGChatService"],
        },
    },
}
