# -------------------------------------------------------------------------
# - Lookup order (highest priority first):
#     Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#     Config_WebSearch.py, Config_Banned.py, Config_Models.py, Config_Global.py
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
# RAG‑LCC tracks license consent for models referenced in this configuration.
# Consent handling is applied uniformly, independent of how or where a model
# runtime is installed.
#
# When consent checks are enabled and the configuration changes, the system
# may:
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
# Each variable below selects an implementation identifier (impl) for a given
# model role. The value must match a top‑level key in the _MODELS dictionary
# defined later in this file.
#
# At runtime, model resolution follows:
#
#     _MODELS[<impl>][<role>]
#
# For example, if _LLM = "mistral", the LLM configuration is read from:
#
#     _MODELS["mistral"]["_LLM"]
#
# To change a model, update the corresponding impl value to another key that
# defines the required role.
#
# -----------------------------------------------------------------------------

from typing import Any

_ACTIVE_LLM_CHK    = "llama_guard"
_ACTIVE_LLM        = "mistral"
_ACTIVE_LLM_REWRITE_PROMPT = "mistral"
_ACTIVE_EMBED      = "snowflake"
_ACTIVE_CROSS      = "mmarco"
_ACTIVE_ENDPOINT   = "ollama"
_ACTIVE_OPENWEBUI  = "openwebui"
_ACTIVE_RAGCHATSERVICE = "ragchatservice"
_ACTIVE_TRANSLATION = "m2m100"

# Optional Hugging Face access token used for gated/private model downloads.
# Leave empty for public models and anonymous access.
# This one overrides the model specific _HF_API_KEY definitions.
_HF_API_KEY = ""

# _MODELS hierarchy: _MODELS[impl][role] -> config dict
_MODELS: dict[str, dict[str, dict[str, Any]]] = {
    "snowflake": {
        "_EMBED": {
            "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
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

    "mmarco": {
        "_CROSS": {
            "MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "FRIENDLY_NAME": "Cross Encoder MMARCO MiniLM v2 L12 H384",
            "REVISION": "",
            # QUERY_INSTRUCTION: optional prefix prepended to the query before
            # scoring.  Standard BERT cross-encoders (including mmarco) do not
            # need one — leave empty.  Instruction-tuned models such as
            # BAAI/bge-reranker-v2-m3 benefit from a prefix like
            # "Represent this sentence for searching relevant passages: ".
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

    "m2m100": {
        "_TRANSLATION": {
            # Facebook/Meta's M2M-100 1.2B (MIT license) — many-to-many
            # translator covering 100 languages. We previously tried
            # MADLAD-400-3B but the T5X-converted HF checkpoint produced
            # garbage output on CPU with current transformers (model emits
            # constant-class tokens regardless of input). The 418M variant
            # of M2M-100 worked but mistranslated several common German
            # nouns (e.g. "Säugetiere" -> "seed animals"); the 1.2B variant
            # fixes those at the cost of ~5 GB on disk and ~5 GB resident.
            # M2M-100 is HF-native, well tested and MIT-licensed
            # (commercial-safe).
            "MODEL": "facebook/m2m100_1.2B",
            "FRIENDLY_NAME": "M2M-100 1.2B Multilingual Translation",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/facebook/m2m100_1.2B",
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "HF_API_KEY": "",
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt",
            "COMPLIANCE_MSG": "M2M-100 is a many-to-many multilingual translation model released by Facebook/Meta under the MIT licence, supporting 100 languages. Used to translate non-English user queries into English before retrieval.",
            "MODEL_CARD": "https://huggingface.co/facebook/m2m100_1.2B",
            # When True, load on CUDA in FP16; otherwise CPU in FP32.
            # Default False: keeps the GPU available for the retrieval
            # stack (embedder + reranker + KeyBERT). The 1.2B variant is
            # ~5 GB on disk and ~5 GB resident in fp32 on CPU; first-token
            # latency on CPU is a few seconds for typical chat queries,
            # which is acceptable for a once-per-turn translation. Switch
            # to True only when the GPU has clear headroom (>= 8 GB free).
            "USE_GPU": False,
            "USED_BY": ["RAGChat", "RAGChatService"],
        },
    },

    "mistral": {
        "_LLM": {
            "MODEL_OLLAMA": "mistral:7b",
            "MODEL_VLLM": "mistral_7b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },

        "_LLM_REWRITE_PROMPT": {
            "MODEL_OLLAMA": "mistral:7b",
            "MODEL_VLLM": "mistral_7b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService"],
        },

        "_LLM_CHK": {
            "MODEL_OLLAMA": "mistral:7b",
            "MODEL_VLLM": "mistral_7b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },

    "llama": {
        "_LLM": {
            "MODEL_OLLAMA": "llama3.1:8b",
            "MODEL_VLLM": "llama3_1_8b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },

        "_LLM_REWRITE_PROMPT": {
            "MODEL_OLLAMA": "llama3.1:8b",
            "MODEL_VLLM": "llama3_1_8b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService"],
        },

        "_LLM_CHK": {
            # This configuration uses llama3.1 and the mistral prompt check prompt
            "MODEL_OLLAMA": "llama3.1:8b",
            "MODEL_VLLM": "llama3_1_8b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },

    "llama_guard": {
        "_LLM_CHK": {
            "MODEL_OLLAMA": "llama-guard3:8b",
            "MODEL_VLLM": "llama_guard3_8b",
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
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,      # cap Ollama-reported context window
            "TOKEN_BUDGET_RESERVED_OUTPUT": 64,   # tokens reserved for model reply
            "TOKEN_BUDGET_RESERVED_SYSTEM": 64,   # tokens reserved for system preamble
            "USED_BY": ["RAGChat", "RAGChatService", "DocClassify"],
        },
    },

    "ollama": {
        "_OLLAMA": {
            "PROVIDER": "ollama",
            "FRIENDLY_NAME": "Ollama Local LLM Provider",
            "SOURCE": "https://github.com/ollama/ollama",
            "BASE_URL": "http://<ollama host>:11434/api/generate", # server IP, localhost or Docker host.docker.internal
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
            "BASE_URL": "http://<vllm host>:8000/v1/chat/completions", # server IP, localhost or Docker host.docker.internal
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "API_KEY": "", # might need sk- prefix
            "STREAMING_REQ": False,
            "TRY_FALLBACK_URLS": True,  # True: try localhost/docker fallbacks on failure  False: only probe BASE_URL
            "USE_GPU": True,
            "REQUEST_TIMEOUT": 120.0,
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
            "BASE_URL": "http://<openwebui host>:3000", # server IP, localhost or Docker host.docker.internalc
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
            "HOST": "<RAGChatService host>",  # Listener address (0.0.0.0 for Docker/external access)
            "PORT": 11435,        # Service port
            "BASE_URL": "http://localhost:11435",  # Full service URL
            # Optional per-model HF key. If empty, _HF_API_KEY is used.
            "API_KEY": "",        # Bearer token for authenticating incoming requests from OpenWebUI
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/HarinezumIgel/RAG-LCC/main/LICENSE",
            "COMPLIANCE_MSG": "RAG-LCC is licensed under the MIT License. Copyright (c) 2026 @HarinezumIgel",
            "USED_BY": ["RAGChatService"],
        },
    },
}