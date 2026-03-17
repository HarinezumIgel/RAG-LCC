# -------------------------------------------------------------------------
# - Lookup order: Config_<RAGChat.py|Config_RAGLoad.py|Config_DocClassify.py>,
#   Config_Banned, Config_Models.py, Config_Globals.py
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

_LLM_CHK = "llama_guard"  # impl for _LLM_CHK role. llama_guard, llama, mistral
_LLM     = "mistral"      # impl for _LLM role. mistral, llama
_EMBED   = "snowflake"    # impl for _EMBED role
_CROSS   = "mmarco"       # impl for _CROSS role
_OLLAMA  = "ollama"       # impl for _OLLAMA role

# _MODELS hierarchy: _MODELS[impl][role] -> config dict
_MODELS = {
    "snowflake": {
        "_EMBED": {
            "MODEL": "snowflake/snowflake-arctic-embed-l-v2.0",
            "FRIENDLY_NAME": "Snowflake Arctic Embed L v2.0",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "COMPLIANCE_MSG": "Embedder: Snowflake arctic-embed-l-v2.0 is the newest addition to the suite of embedding models Snowflake has released optimizing for retrieval performance and inference efficiency",
            "MODEL_CARD": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
        },
    },

    "mmarco": {
        "_CROSS": {
            "MODEL": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "FRIENDLY_NAME": "Cross Encoder MMARCO MiniLM v2 L12 H384",
            "REVISION": "",
            "SOURCE": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md",
            "COMPLIANCE_MSG": "Cross Encoder: This model was trained on the MMARCO dataset. It is a machine translated version of MS MARCO using Google Translate.",
            "MODEL_CARD": "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        },
    },

    "mistral": {
        "_LLM": {
            "MODEL": "mistral:7b",
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/mistral-7b",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md",
            "COMPLIANCE_MSG": "LLM: Mistral 7B is a 7.3B parameter model",
            "TAG": "LLM: Mistral 7B is a 7.3B parameter model",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_MISTRAL",
        },

        "_LLM_CHK": {
            "MODEL": "mistral:7b",
            "FRIENDLY_NAME": "Mistral 7B",
            "SOURCE": "https://huggingface.co/mistralai/mistral-7b",
            "LICENSE": "Apache-2.0",
            "LICENSE_URL": "https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md",
            "COMPLIANCE_MSG": "LLM: Mistral 7B is a 7.3B parameter model",
            "TAG": "LLM: Mistral 7B is a 7.3B parameter model",
            "MODEL_CARD": "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
        },
    },

    "llama": {
        "_LLM": {
            "MODEL": "llama3.1:8b",
            "FRIENDLY_NAME": "Llama 3.1",
            "SOURCE": "https://github.com/meta-llama/llama-models/tree/main/models/llama3_1",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
            "COMPLIANCE_MSG": "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
            "PROMPT_CHAT": "_PROMPT_CHAT",
            "PROMPT_CLASSIFY": "_PROMPT_CLASSIFY_LLAMA",
        },

        "_LLM_CHK": {
            # This configuration uses llama3.1 and the mistral prompt check prompt
            "MODEL": "llama3.1:8b",
            "FRIENDLY_NAME": "Llama 3.1",
            "SOURCE": "https://github.com/meta-llama/llama-models/tree/main/models/llama3_1",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
            "COMPLIANCE_MSG": "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_MISTRAL",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_MISTRAL",
        },
    },

    "llama_guard": {
        "_LLM_CHK": {
            "MODEL": "llama-guard3:8b",
            "FRIENDLY_NAME": "Llama Guard 3",
            "SOURCE": "https://ollama.com/library/llama-guard3",
            "UPSTREAM_SOURCE": "https://huggingface.co/meta-llama/Llama-Guard-3-8B",
            "LICENSE": "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT",
            "LICENSE_URL": "https://huggingface.co/meta-llama/Llama-Guard-3-8B/resolve/main/LICENSE",
            "COMPLIANCE_MSG": "Llama Guard 3 is licensed under the Llama 3.1 Community License Agreement, Copyright © Meta Platforms, Inc. All Rights Reserved.",
            "TAG": "Built with Meta Llama 3.",
            "MODEL_CARD": "https://ollama.com/library/llama-guard3",
            "PROMPT_CHAT": "_PROMPT_CHECK_CHAT_LLAMA_GUARD",
            "PROMPT_CLASSIFY": "_PROMPT_CHECK_CLASSIFY_LLAMA_GUARD",
        },
    },

    "ollama": {
        "_OLLAMA": {
            "PROVIDER": "ollama",
            "FRIENDLY_NAME": "Ollama Local LLM Provider",
            "SOURCE": "https://github.com/ollama/ollama",
            "BASE_URL": "http://localhost:11434",
            "LICENSE": "MIT",
            "LICENSE_URL": "https://raw.githubusercontent.com/ollama/ollama/main/LICENSE",
            "MODEL_LICENSES_NOTE": "Each model has its own license—see `ollama list`",
            "COMPLIANCE_MSG": "OLLAMA Local LLM Provider",
            "MODEL_CARD": "https://ollama.com",
        },
    },
}