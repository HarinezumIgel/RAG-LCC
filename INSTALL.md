<!-- markdownlint-disable MD033 -->
# 🚀 RAG‑LCC — Installation & First Run

← Back to [README](README.md) · See also: [EXAMPLES.md](EXAMPLES.md) · [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) · [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md)

This guide helps you set up RAG-LCC, an **experimental lab environment**.

> **Recommended path:** Use `python ./src/Scripts/Setup.py`.
> The setup script  guides the operator through installation, consent steps,
> runtime configuration, and final hash updates. The sections below remain as
> reference material and for manual/deep-dive workflows.

## 🧭 0. Guided setup first

After cloning and activating your environment, start with:

```bash
python ./src/Scripts/Setup.py
```

The script guides you through:

- System package/license consent flow (for OCR dependencies)
- Python dependency license review and install
- Example config copy confirmation
- Optional NLTK and Argos download/consent steps
- Runtime config questions (endpoint, internet toggles, API keys)
- Final config hash recalculation

Before performing an action, the script requires 3rd party license consents.

Use the other docs for detail after (or during) setup:

- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for all config keys and tuning
- [EXAMPLES.md](EXAMPLES.md) for end-to-end usage examples
- [HANDS_ON_TOUR.md](HANDS_ON_TOUR.md) for guided walkthrough scenarios

## 📋 Prerequisites

- **Python 3.13** (developed and tested with 3.13; minimum 3.10)
- Memory and disk resources appropriate for the example models
- Optional GPU for faster inference (see [GPU Setup](#-gpu-setup) below)
- Optional: Visual Studio Code (used during development)
- Tested and developed on Windows 11

## ✅ Pre-install Checklist

Complete these steps before running any install command.

### ⚖️ 1. Review third-party licenses

Inspect `./3rdPartyLicenses` and obtain any required approvals before installing dependencies.

## 🛠️ Installation

### 📋 Install Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)

⚠️ **Note:** `Setup.py` can run on both Windows and Unix-based systems, either inside a Docker container or directly within a host `.venv` environment. The prerequisites listed below are required **only for Docker-based installations**.

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- VS Code extension: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## 🤖 Install Ollama (manual/reference)

RAG-LCC supports local LLMs via Ollama. Install Ollama from: <https://ollama.com>
Ollama is licensed under the [MIT License](https://github.com/ollama/ollama/blob/main/LICENSE).

## 📥 Install the models in Ollama

Pull the models referenced in your configuration (example). By downloading and using these models, you are bound by the model owner's license terms.

```shell
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull llama-guard3:8b
```

## 🔀 Optional: use vLLM instead of Ollama

RAG-LCC supports both `ollama` and `vllm` backends for inference.

```shell
pip install vllm
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1
```

Then set endpoint config in `src/Configuration/Config_Models.py`:

- `_ACTIVE_ENDPOINT = "vllm"`
- `_MODELS["vllm"]["_VLLM"]["BASE_URL"]` to your server URL

And ensure selected models are available for the active endpoint:

- `_ACTIVE_LLM` (RAG inference)
- `_ACTIVE_LLM_CHK` (prompt/compliance checks)

If you changed model config values, recalculate hashes:

```shell
python ./src/Scripts/RecalcConfigHashes.py
```

## 🌐 Install Open WebUI (Optional)

[Open WebUI](https://docs.openwebui.com/) is a feature-rich, self-hosted web interface for LLMs that provides a ChatGPT-like experience in your browser. When connected to RAG-LCC's `RAGChatService.py` REST API, it serves as a user-friendly chat interface that allows you to:

- Query your RAG-LCC document collections through a modern web UI
- Adjust retrieval parameters via the Controls sidebar (search mode, filters, KeyBERT settings)
- View algorithm results and ensemble scores (when `SHOW_CLI_LIKE_ALGO_RESULTS` is enabled)
- Maintain conversation history across sessions
- Access highlighted PDF documents through the `/marked` endpoint (when enabled)

The simplest installation method is via Python pip:

```shell
pip install open-webui
```

Then start it with:

```shell
open-webui serve
```

Open WebUI will be available at <http://localhost:8080>.
Point its **OpenAI API** connection to the `RAGChatService` endpoint (`http://localhost:11435/v1`, see [Config_RAGChatService.py — HTTP Listener for OpenWebUI](#-config_ragchatservicepy--http-listener-for-openwebui) for endpoint definition).

- Documentation: <https://docs.openwebui.com/>
- GitHub: <https://github.com/open-webui/open-webui>
- License: **Open WebUI License** — see <https://github.com/open-webui/open-webui/blob/main/LICENSE> (includes a branding-preservation clause; prior contributions retain their original licenses — see `LICENSE_HISTORY`).

## � Performance Optimization: HF Transfer (xet Protocol)

For faster HuggingFace model downloads, enable high performance transfer with Xet:

```bash
export HF_XET_HIGH_PERFORMANCE=1  # Linux/macOS
set HF_XET_HIGH_PERFORMANCE=1     # Windows CMD
$env:HF_XET_HIGH_PERFORMANCE="1"  # Windows PowerShell
```

This requires the `xet` package to be installed. The environment variable is automatically enabled in deployed environments via `Config_Internet_Env.py`.

**Benefits:**
- Significantly faster downloads for large models
- Uses the xet protocol for optimized data transfer
- Particularly useful for models with multiple files (e.g., sharded checkpoints)

> **Note:** This is automatically configured during deployment. For manual installations, set the environment variable before pulling models.

## �📦 2. Clone and Setup

> **Important:** RAG-LCC must be installed in its own dedicated directory. Do not clone it into an existing project folder or a shared location. The application uses its installation directory as a trust boundary — file and directory deletions (e.g. ChromaDB collection removal) are restricted to paths inside the project root. Placing RAG-LCC inside another project's tree may cause unintended interactions.

```bash
# Clone repository (replace with your repository URL)
git clone https://github.com/HarinezumIgel/RAG-LCC.git
cd RAG-LCC
```

After cloning, choose one of the following deployment paths.

2. **Open in container:**
   - Open the `RAG-LCC` folder in VS Code
   - Press `F1` (or `Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Select **"Dev Containers: Reopen in Container"**
   - VS Code will build the container and reopen the workspace inside it

### 🐳 Option A: Docker deployment (supported)

Docker-based deployment is supported via the repository's container definition.

#### Recommended: VS Code Dev Containers

The easiest way to work with the containerized environment is using VS Code:

3. **Inside the container:**
   - Open a terminal in VS Code (`` Ctrl+` `` or **Terminal → New Terminal**)

```bash
   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate
   - Run the guided setup script:
      # Create and activate a virtual environment
   python ./src/Scripts/Setup.py
```

> **Note:** The container includes a `.venv` virtual environment. In case it is missing, the script will create a `.venv`.

### 🧰 Option B: Host virtual-environment deployment

```bash
   # Create and activate a virtual environment
   python -m venv .venv
   # Unix
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Then continue with the guided setup script:
   python ./src/Scripts/Setup.py
```

#### Alternative: Manual docker container setup commands

For advanced users who prefer manual container management:

```bash
# Build image
docker build -f .devcontainer/Dockerfile -t rag-lcc:latest .

# Run container (CPU mode)
# Note: A port (default 11435 is will be opened.
# The port will be served if RAGChatService.py is stared. Requires
# SERVE_OPENWEBUI_CHAT="1"
docker run --rm -it -p 11435:11435 rag-lcc:latest bash

# Optional: try GPU passthrough (only if your Docker/NVIDIA setup supports it)
docker run --rm -it --gpus all -p 11435:11435 rag-lcc:latest bash
```

> **Tip — HF model cache:** Mount the host Hugging Face cache into the container to avoid re-downloading models on every restart:
>
> ```bash
> # Linux / macOS host
> docker run --rm -it -p 11435:11435 \
>   -v "$HOME/.cache/huggingface:/home/vscode/.cache/huggingface" \
>   rag-lcc:latest bash
>
> # Windows host (PowerShell)
> docker run --rm -it -p 11435:11435 `
>   -v "${env:USERPROFILE}\.cache\huggingface:/home/vscode/.cache/huggingface" `
>   rag-lcc:latest bash
> ```
>
> With VS Code Dev Containers, add a volume mount to `.devcontainer/devcontainer.json`:
>
> ```json
> "mounts": [
>   "source=${localEnv:USERPROFILE}/.cache/huggingface,target=/home/vscode/.cache/huggingface,type=bind,consistency=cached"
> ]
> ```
>
> The environment variable `HF_HOME` can be set to a custom path if your cache lives elsewhere.

Inside the container, continue with the guided setup script:

```bash
python ./src/Scripts/Setup.py
```

## Explanation to the steps performed in `Setup.py`

> **⚠️ Specify where Ollama/vLLM and Open WebUI are running:** During setup, you will be asked to enter the BASE_URL where your LLM backend (Ollama or vLLM) and Open WebUI (optional) are listening. This is critical for connectivity. Common scenarios:
>
> - **No Docker: Backends on same machine as RAG-LCC:**
>   - Ollama:     `http://localhost:11434/api/generate`
>   - vLLM:       `http://localhost:4000/v1/chat/completions`
>   - Open WebUI: `http://localhost:8080`
> - **RAG-LCC in Docker, backends on host:**
>   - Ollama:     `http://host.docker.internal:11434/api/generate`
>   - vLLM:       `http://host.docker.internal:4000/v1/chat/completions`
>   - Open WebUI: `http://host.docker.internal:8080`
> - **Docker or no Docker: Backends on different machine than RAG-LCC:**
>   - Ollama:     `http://<your host>:11434/api/generate`
>   - vLLM:       `http://<your host>:4000/v1/chat/completions`
>   - Open WebUI: `http://<your host>:8080`
>
> **Container → Host connectivity is NOT blocked by default.** TCP access from container to host works natively. The key is using the correct hostname:
>
> - `localhost` inside a container refers to the container itself, not the host
> - Use `host.docker.internal` (Docker Desktop) or `172.17.0.1` (Linux) to reach the host
>
> **Test connectivity before setup** (run from inside the container):
>
> ```bash
> Unix:
> `curl` -v <one of above URLs>
> Windows:
> `Invoke-WebRequest` <one of above URLs>
> ```
>
> If curl succeeds and returns JSON with your models, connectivity is working - just use that URL in the setup script.
>
[!IMPORTANT]
> If RAG-LCC can't connect to the LLM provider or Open WebUI (optional), it tries a 6-step fallback procedure described in [the endpoint fallback mechanism](ARCHITECTURE.md#endpoint-fallback-mechanism).
>
> **GPU note:** GPU support may not be available inside your Docker container, depending on host drivers, runtime, and container toolkit configuration. If GPU is not available in the container, use the host virtual-environment installation path below.
>
> **RAGChatService port forwarding:** **Port forwarding is automatically configured** in `.devcontainer/devcontainer.json`. This is required for `RAGChatService` since it recieves queries from Open WebUI.
> Disable these lines in `devcontainer.devcontainer.json` and do "Reopen in Container" in VS Code.

``` json
  "forwardPorts": [
    11435
  ],
```
>
> - In Docker set `_MODELS.ragchatservice._RAGCHATSERVICE.HOST` to `"0.0.0.0"` to listen on all container interfaces
> - OpenWebUI running on your host machine can connect to `http://localhost:<configured-port>`
> - OpenWebUI on another machine can connect to `http://<your-host-ip>:<configured-port>`

### 📁 Directory structure (quick orientation)

Core folders used by setup and runtime:

- `src/` - application and framework source code
- `src/Configuration/` - all `Config_*.py` files
- `TestDocs/` - sample documents for `RAGLoad` and `DocClassify`
- `chromadb/` - Chroma/BM25/graph persistent stores
- `logs/` - runtime logs (per app)
- `ModelGovernance/` - model and package consent/license metadata
- `3rdPartyLicenses/` - generated and bundled third-party license summaries
- `deploy/` - deployment/packaging scripts
- `tests/` - automated test suite

### ✅ Verify file signatures

Run the signature verification script to confirm that shipped files have not been tampered with. The public key is in `verify_sign/`.

```python
.\src\Scripts\VerifySignatures.py -InputDir .
```

## 📥 3. Install Dependencies (manual/reference)

The `./requirements` folder contains `requirements_final.txt` and a list of required modules. Consult [3rdPartyLicenses](3rdPartyLicenses/Licenses.md) for an overview of development environment licenses.

```bash
# Install dependencies only after you have reviewed the license files
pip install -r requirements/requirements_final.txt
```

If pip reports a **dependency conflict** (`ResolutionImpossible`), the most common cause is a package pinned to an exact version whose dependencies clash with another pinned package. Open `requirements/requirements_final.txt`, find the offending pin(s) and either loosen them to a minimum range (e.g. `langchain-huggingface>=1.0.1`) or remove the line(s) entirely so pip can resolve compatible versions automatically. Then re-run the install command above.

## ⚖️ 4. Post-install license review (manual/reference)

To generate a new third-party license report based upon your **actual** `.venv` run in powershell:

```powershell
.\scripts_posh\Show3rdPartyLicenses.ps1 -ProjectPath . -VenvName <your venv> -ContainingLicenseDirectoryName <directory for license summary files>
```

You may need to install pip-licenses first.

View the generated Licenses.* file, available as .md, .json, .spdx.

## 📋 Model permission requirement

Some models are distributed under licenses that require you to accept the
license terms before use (for example, Meta's Llama Community License for the
Llama LLM and Llama Guard model).
Ollama does **not** enforce a gating step, so `ollama pull` will succeed
without prior approval; however, **by downloading and using the model you are
still bound by the model owner's license terms**.

## ✅ Required: model license consent

RAG-LCC does not download LLMs. On startup `RAGLoad.py`, `RAGChat.py`, `RAGChatService.py` and `DocClassify.py` check whether the licenses belonging to the models used in `./Configuration/Config_Models.py` have been accepted.
RAG-LCC asks you whether it may temporarily allow internet access to fetch the licenses. Then you are guided through the license consent loop. The fetched licenses and operator consent are stored in `./ModelGovernance/licenses`. Operators can monitor network activity using the built-in [Socket-Level Network Tracing](#-socket-level-network-tracing).

Hugging Face models (embedder, cross-encoder, and translation model) follow a separate consent-based download flow. See [Hugging Face Models (Embedder, Cross-Encoder + Translation)](#-hugging-face-models-embedder--cross-encoder--translation) for details on when and how these models are downloaded.

## 🌍 7. Install Argos Translate

RAG-LCC supports optional local translation of banned phrases from English to the detected document language using Argos Translate.

- Argos License is here: <https://github.com/argosopentech/argos-translate?tab=MIT-1-ov-file>
- Stanza License is here: <https://github.com/stanfordnlp/stanza?tab=Apache-2.0-1-ov-file>
- To enable this feature please refer to: <https://www.argosopentech.com>

```shell
pip install argostranslate
```

**Important:** When `ARGOS_STANZA_DOWNLOAD` is `"0"` (default used in this repository), the Argos Translate language packages for the languages expected in your documents must be **pre-installed** before processing. If a document's language is not installed, translation is skipped, a warning is issued, and the compliance pipeline falls back to English-normalized patterns. When `ARGOS_STANZA_DOWNLOAD` is `"1"`, stanza may download missing tokenizer models at runtime, so pre-installation is not strictly required — but a warning is still issued if no matching translation pair is found.

### Controlling behaviour for unsupported languages

The config key **`UNSUPPORTED_LANGUAGE_ACTION`** (in `Config_Global.py`) determines
what happens when a document’s detected language is not installed:

| Value           | Behaviour                                                                              |
|-----------------|----------------------------------------------------------------------------------------|
| `FALLBACK_EN`   | Process silently with English-only banlists                                            |
| `NOT_OK`        | *(default)* Reject the document -- write to NOT_OK CSV, skip all further processing    |

Install language packages using the provided helper script:

```powershell
# Install Argos Translate language packages.
# Languages to install are defined in Config_Global.py ARGOS_LANGUAGES slot
# Each package bundles the required stanza tokenizer models.
# The script shows what will be installed and asks for confirmation before proceeding.
python src\Scripts\ArgosTranslatePackages.py install

# Remove all installed Argos Translate language packages.
# The script shows what will be removed and asks for confirmation before proceeding.
python src\Scripts\ArgosTranslatePackages.py remove
```

Enable the languages you need, see [Translation configuration (Argos)](CONFIGURATION_REFERENCE.md#-translation-configuration-argos).

For license consent details, see [Argos](#-argos).

Each Argos Translate language package bundles the stanza tokenizer models it needs (stored inside the package directory, e.g. `~/.local/share/argos-translate/packages/<pkg>/stanza/`). Without the correct language packages installed, translation is skipped at runtime and non-English documents fall back to English-normalized patterns (e.g. German "Pferde" would not match the banned word "horse"). The remove command uninstalls all Argos packages and their bundled stanza models.

The setting in ./Configuration/Config_Internet_Env.py forces local translation:

```Python
# ARGOS_MODEL_PROVIDER: Force Argos to use local package-based translation
# and avoid remote provider paths (LibreTranslate/OpenAI).
os.environ["ARGOS_MODEL_PROVIDER"] = "OPENNMT"
```

```Python
# ARGOS_CHUNK_TYPE: Select the sentence boundary detection (SBD) backend
# used by Argos Translate before translating.
# "SPACY"  = use SpaCy sentencizer (works offline, default used in this repository).
# "STANZA" = use stanza Pipeline (broken offline in argos-translate ≥1.11,
#            see argosopentech/argos-translate#385 / #512).
# "MINISBD" / "ARGOSTRANSLATE" / "DEFAULT" = other options.
os.environ["ARGOS_CHUNK_TYPE"] = "SPACY"
```

> **Note:** You may see the warning `Language en package default expects mwt, which has been added` during translation. This is a harmless informational message from **Stanza** (the NLP tokenizer used internally by Argos Translate). It means the English language model expects a Multi-Word Token (MWT) processor and Stanza added it automatically. No action is required.

## 📝 8. NLTK Stopwords (Text Preprocessing)

Install NLTK:
NLTK license is here: <https://raw.githubusercontent.com/nltk/nltk/v3.10.0/LICENSE.txt>

```shell
pip install nltk
```

Download the stopwords corpus manually:

```python
import nltk
nltk.download("stopwords")
```

If `NLTK_STOPWORDS_DOWNLOAD` in `Configuration/Config_Internet_Env.py` is set to `"1"` and a stopword list for a newly detected language is missing, RAG‑LCC will download the required NLTK stopword corpus.
If downloads are disabled (default used in this repository `"0"`), the system falls back to an empty stopword list.

Adjust the NLTK data path in `Configuration/Config_Global.py` if needed:

```python
_CUSTOM_NLTK_DATA_DIRECTORY = (
    _ABSOLUTE_PATH + r"\AppData\Roaming\nltk_data\corpora\stopwords"
)
```

## 📖 8a. NLTK WordNet Synonyms (Optional — Banned‑Word Expansion)

RAG‑LCC can optionally expand the English banned‑word list with synonyms from
[WordNet](https://wordnet.princeton.edu/) before detection and translation.
This strengthens the lexical scorers (Regex, Jaccard, BM25) by giving them
more surface forms to match against, while KeyBERT (embedding‑based) is left
unchanged since it already captures semantic neighbours.

NLTK is already installed if you followed step 8 above. You only need to
download the WordNet corpus:

```python
import nltk
nltk.download("wordnet")
```

If the WordNet corpus is **not** installed, RAG‑LCC prints an orange warning
at startup and proceeds with the original (unexpanded) banned‑word list — no
functionality is lost.

> **License:** The WordNet corpus is developed and maintained by Princeton
> University. **WordNet 3.0** — Copyright © 2006 by Princeton University.
> Licensed under the [WordNet 3.0 License](https://wordnet.princeton.edu/license-and-commercial-use).
> The corpus is **not** distributed with RAG‑LCC; operators download it
> independently from NLTK's data servers and are bound by its license terms.

**Configuration** in `Configuration/Config_Global.py`:

```python
_WORDNET = {
    "ENABLED": True,             # Toggle synonym expansion on / off
    "DEPTH": 1,                  # 1 = direct synonyms only, 2 = synonyms of synonyms
    "MAX_SYNONYMS_PER_PHRASE": 1,# Cap per banned phrase to prevent list explosion
    "POS_FILTER": ["n", "v"],    # WordNet POS tags: n(oun), v(erb), a(dj), r(adv), s(at‑adj)
    "STOPLIST": ["word", "number", ...],  # Overly generic terms to suppress
}
```

| Key | Default | Description |
| ----- | ------- | ----------- |
| `ENABLED` | `True` | Master switch. Set to `False` to skip expansion entirely. |
| `DEPTH` | `1` | Synonym hop depth. `1` = direct synonyms only (recommended). `2` adds synonyms‑of‑synonyms. |
| `MAX_SYNONYMS_PER_PHRASE` | `1` | Maximum synonyms added per original banned phrase. Prevents list explosion. |
| `POS_FILTER` | `["n", "v"]` | Restrict to these WordNet parts of speech. Empty list = accept all. |
| `STOPLIST` | *(~20 generic words)* | Words excluded from expansion even if they appear as WordNet synonyms. |

## 👁️ 9. Installing OCR Support (Tesseract)

RAG‑LCC uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) ([Apache-2.0 License](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)) to extract text from non plain text files. The Python wrapper [pytesseract](https://github.com/madmaze/pytesseract) is also licensed under [Apache-2.0](3rdPartyLicenses/Licenses.md#pytesseract-0313).

> **Note:** Tesseract OCR is **not** included with or distributed by RAG‑LCC. Operators must obtain and install the Tesseract engine independently. By downloading and using Tesseract, operators are bound by its license terms.

### Installation

**Windows:**

Download and install the official installer:
<https://github.com/UB-Mannheim/tesseract/wiki> ([Apache-2.0 License](https://github.com/UB-Mannheim/tesseract/blob/main/LICENSE))

Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**Linux:**

Install via package manager:

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# Fedora/RHEL
sudo dnf install tesseract

# Arch Linux
sudo pacman -S tesseract
```

Default installation path: `/usr/bin/tesseract`

### Configuration

The framework uses an OS-aware path format in `Configuration/Config_Internet_Env.py`:

```python
# Format: "windows_path|linux_path"
os.environ.setdefault(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe|/usr/bin/tesseract"
)
```

The system automatically selects the correct path based on the current platform. If you installed Tesseract in a custom location, update both paths accordingly:

```python
# Custom installation paths
os.environ.setdefault(
    "TESSERACT_PATH",
    r"C:\Custom\Path\tesseract.exe|/opt/tesseract/bin/tesseract"
)
```

## 🕸️ 10. Installing Graph Retrieval Support (spaCy)

RAG‑LCC uses [spaCy](https://spacy.io/) ([MIT License](https://github.com/explosion/spaCy/blob/master/LICENSE), © Explosion AI) for named-entity recognition (NER) and noun-phrase extraction when any `*_GRAPH` or `GRAPH` search mode is active. The `en_core_web_sm` language model is downloaded separately and is also released under the [MIT License](https://github.com/explosion/spacy-models/blob/master/LICENSE).

> **Note:** spaCy and its models are **not** bundled with RAG‑LCC. `RAGLoad` always builds all three retrieval stores (ChromaDB, BM25, and graph index) unconditionally, so spaCy **must** be installed before running `RAGLoad`. During `RAGChat`, spaCy is loaded on demand only when a graph search mode is active. If the model is missing, a descriptive error with the exact `python -m spacy download …` command is raised.

Install spaCy (already listed in `requirements.txt`) and download the English model:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Entity types, BFS depth, and noise-filter thresholds are configured in the `_GRAPH_INDEX` slot in `Config_Global.py`. See [Retrieval Stores & Search Modes](ARCHITECTURE.md#-retrieval-stores) for details.

## 🎮 GPU Setup

If your machine has an NVIDIA GPU and you want to use it for inference:

1. Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) ([EULA](https://docs.nvidia.com/cuda/eula/index.html)) matching your GPU driver.
2. Install the CUDA-enabled PyTorch wheels **inside your virtual environment**:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   Adjust the `cu121` suffix to match your installed CUDA version.
3. Set `USE_CPU = False` and `EMBEDDER_BITS = 16` or `EMBEDDER_BITS = 32` in your configuration.

> **Note:** The NVIDIA CUDA Toolkit is **not** included with or distributed by RAG‑LCC. Operators must obtain and install it independently. By downloading and using the CUDA Toolkit, operators accept NVIDIA's [EULA](https://docs.nvidia.com/cuda/eula/index.html). PyTorch is licensed under the [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE).

If `USE_CPU` is `False` but no functional GPU is detected (possible drivers/runtime not installed in the venv), the application will automatically fall back to CPU with 32-bit precision and display a warning.

## 👀 Review the example config files

Please read `Configuration/Config_Models.py` before enabling internet access or accepting model licenses.

## 📋 If ok, copy example configs into place

Manually copy the example configuration files into the `Configuration/` folder.

```powershell
copy ./Examples/Example_Config_Banned.py    ./src/Configuration/Config_Banned.py
copy ./Examples/Example_Config_Models.py   ./src/Configuration/Config_Models.py
copy ./Examples/Example_Config_WebSearch.py ./src/Configuration/Config_WebSearch.py
```

Open `Configuration/Config_Banned.py`, `Configuration/Config_Models.py`, and `Configuration/Config_WebSearch.py` and configure the settings according to your needs.

## 🧪 11. Run the tests

Note: Running the tests will take some time.

`pytest` is listed in `tests/requirements-dev.txt`. Install it once if needed:

```bash
pip install pytest
```

**Activate the virtual environment first**, then run from the project root:

```powershell
# PowerShell (Windows)
python -m pytest tests -q --tb=short
```

```bash
# Bash / Linux / macOS
python -m pytest tests -q --tb=short
```

Without an activated venv, use the interpreter directly:

```powershell
# PowerShell — no activated venv
.venv\Scripts\python.exe -m pytest tests -q --tb=short *>&1 | Out-String
```

```bash
# Bash — no activated venv
.venv/bin/python -m pytest tests -q --tb=short
```

Alternatively, use the bundled runner (picks up `sys.executable` automatically):

```bash
python tests/RunTests.py           # all tests, quiet
python tests/RunTests.py -v        # verbose
python tests/RunTests.py -k name   # filter by test name
```

## ⚙️ 12. Adjust Configuration

## 🤖 LLM, Embedder and Cross Encoder

For details on how model implementations are selected, see [Model Implementation Selectors in ARCHITECTURE.md](ARCHITECTURE.md#model-implementation-selectors).

## 🔌 Define Ollama endpoint

The Ollama endpoint, streaming mode, and GPU flag are defined inside the
`_MODELS["ollama"]["_OLLAMA"]` dictionary in `Config_Models.py`:

```python
"ollama": {
    "_OLLAMA": {
    "BASE_URL": "http://127.0.0.1:11434/api/generate",
        "STREAMING_REQ": False,
        "USE_GPU": True,
        ...
    },
},
```

## 🔌 Define vLLM endpoint

The vLLM endpoint, streaming mode, and GPU flag are defined inside the
`_MODELS["vllm"]["_VLLM"]` dictionary in `Config_Models.py`:

```python
"vllm": {
    "_VLLM": {
        "PROVIDER": "vllm",
        "FRIENDLY_NAME": "vLLM OpenAI-compatible Provider",
        "BASE_URL": "http://localhost:4000/v1/chat/completions",
        "API_KEY": "",
        "STREAMING_REQ": False,
        "USE_GPU": True,
        ...
    },
},
```

### 🌐 Specifying where endpoints (Ollam/vLLM and Open WebUI) are running

During setup, you will be prompted for the OpenWebUI BASE_URL. This is used by the Informer to verify OpenWebUI is reachable at startup. **Enter the actual URL where OpenWebUI is accessible from RAGChatService:**

RAG-LCC automatically tries up to **6 fallback candidates** when a configured endpoint is unavailable (configured host, localhost, 127.0.0.1, host.docker.internal with configured port, plus localhost and host.docker.internal with the default port if it differs from the configured port). T See [ARCHITECTURE.md § Endpoint Fallback Mechanism](ARCHITECTURE.md#endpoint-fallback-mechanism) for the complete probe sequence and behavior.

**Note:** The OpenWebUI URL is bidirectional:

- **RAGChatService → OpenWebUI:** RAGChatService checks if OpenWebUI is reachable (uses `_MODELS.openwebui._OPENWEBUI.BASE_URL`)
- **OpenWebUI → RAGChatService:** OpenWebUI connects to RAGChatService (uses `_MODELS.ragchatservice._RAGCHATSERVICE.HOST` and `_MODELS.ragchatservice._RAGCHATSERVICE.PORT`)
- **Browser → `/marked` endpoint:** For in-memory document serving, the browser fetches from RAGChatService's `/marked` endpoint (same host:port as the RAGChatService API)

**Container → Host connectivity:** TCP access from Docker container to host is **not blocked by default**—it works natively. The issue is hostname resolution: `localhost` inside a container refers to the container itself, not the host. Use `host.docker.internal` (Docker Desktop) or `172.17.0.1` (Linux bridge gateway) to reach the host.

After setup, if you need to change the endpoint, edit `Config_Models.py` and recalculate hashes:

```shell
python ./src/Scripts/RecalcConfigHashes.py
```

## 🌐 Config_RAGChatService.py — HTTP Listener for OpenWebUI

`RAGChatService` exposes `RAGChat` over an OpenAI-compatible REST API
(`POST /v1/chat/completions`).
Its configuration file `Config_RAGChatService.py` **re-exports the entire
`Config_RAGChat.py`** — every retrieval strategy, prompt template, KeyBERT
setting, and history parameter is inherited as-is. Only the following
settings are added specifically for the HTTP listener:

| Key | Default used in this repository | Purpose |
| --- | --- | --- |
| `OPENWEBUI_THREAD_POOL_WORKERS` | `2` | `ThreadPoolExecutor` max workers for `chatter.run()`. |
| `SHOW_CLI_LIKE_ALGO_RESULTS` | `True` | When enabled, filter chain algo results (depth/breadth table and ensemble summary) are appended to the LLM answer in Markdown format, mirroring the terminal output of the CLI version. |
| `_MARKED_DOCS["enabled"]` | `False` | **Disabled by default in deployed builds** (enforced by `Deploy.ps1`). Set to `True` in `Config_RAGChatService.py` to enable the highlighted-document service. When enabled, RAG-LCC produces in-memory highlighted copies of retrieved PDFs and serves them at short-lived `GET /marked/<token>.pdf` URLs that are injected into the LLM answer. This also starts an in-process HTTP token store; configure `ttl_seconds`, `max_total_mb`, and `cors_origins` in the same block. See [SECURITY.md](SECURITY.md) for the token security model. |

**RAGChatService Listener Configuration:**

The listener host, port, and API key are configured in `Config_Models.py` under `_MODELS.ragchatservice._RAGCHATSERVICE`:

| Key | Default | Purpose |
| --- | --- | --- |
| `HOST` | `"127.0.0.1"` | Bind address for the uvicorn server. **Configurable during setup** with the actual hostname/IP where RAGChatService should listen. |
| `PORT` | `11435` | Port for the uvicorn server. **Configurable during setup** to avoid port conflicts. |
| `API_KEY` | `""` | Shared secret for Bearer token authentication. Must match the API key configured in OpenWebUI. |

> **VS Code Dev Containers:** The configured port is **automatically added** to `.devcontainer/devcontainer.json` for port forwarding. Set `HOST = "0.0.0.0"` to make RAGChatService accessible from the host.

When configuring where RAGChatService listens, set the appropriate hostname or IP address based on your deployment:

| Scenario | Use as `HOST` |
| --- | --- |
| **RAGChatService on host, OpenWebUI on same machine** | `localhost` or `127.0.0.1` |
| **RAGChatService in Docker/Dev Container, OpenWebUI on host** | `0.0.0.0` (listens on all interfaces) |
| **OpenWebUI on another machine** | `0.0.0.0` or specific network interface IP |

**Legacy scenarios** (when OpenWebUI runs in Docker and RAGChatService on host):

| Scenario | Use as `HOST` |
| --- | --- |
| **OpenWebUI in Docker (Docker Desktop for Windows/Mac)** | `host.docker.internal` |
| **OpenWebUI in Docker (Linux)** | `172.17.0.1` (bridge gateway) or host's network IP |

**Note:** When RAGChatService runs in a Docker container and needs to be accessible from the host or other machines, bind to `0.0.0.0` to listen on all interfaces. When running on the host machine and only local access is needed, use `127.0.0.1`.

`_FRIENDLY_NAME` is set to `"RAGChatService"` so that compliance lookups
resolve to the correct detection profile in `Config_Banned.py`.

The chat prompt (`_PROMPT_CHAT`) is overridden with an OpenWebUI-specific
variant that instructs the LLM to suggest adjusting retrieval parameters
(available in the OpenWebUI Controls sidebar) when context is insufficient.

When `SHOW_CLI_LIKE_ALGO_RESULTS` is `True` (default), the filter chain
algo results — including the per-algorithm depth/breadth table and the
ensemble summary — are appended to the LLM answer in Markdown format.
This mirrors the diagnostic output normally shown in the RAGChat terminal,
making it visible directly inside the OpenWebUI chat window. Set to `False`
to return only the LLM answer.

> **Limitations.** `RAGChatService` deliberately exposes only a narrow
> subset of the OpenAI REST surface — tool calling, vision, audio,
> embeddings, image generation, file uploads, structured outputs, and the
> OpenWebUI Knowledge passthrough are not supported. See
> [ARCHITECTURE.md § RAGChatService — OpenAI Surface](ARCHITECTURE.md#-ragchatservice--openai-surface-intentional-restrictions-vs-known-gaps)
> for the full list and the rationale, and
> [LEGAL.md § No OpenAI / OpenWebUI Feature Parity](LEGAL.md#-no-openai--openwebui-feature-parity)
> for the disclaimer.

## 🔗 Connecting OpenWebUI to RAGChatService

> **Note:** Port `11435` is forwarded in docker. If `SERVE_OPENWEBUI_CHAT="1"` in `Configuration/Config_Internet_Env.py` requests from Open WebUI are served by `RAGChatService`.

1. **Enable the service** — set `SERVE_OPENWEBUI_CHAT` to `"1"` in
   `Configuration/Config_Internet_Env.py` (default used in this repository is `"1"`):

   ```python
   os.environ["SERVE_OPENWEBUI_CHAT"] = "1"
   ```

2. **Start RAGChatService**:

   ```powershell
   python .\src\Apps\RAGChatService.py
   ```

   The service listens on the configured host and port
   (default `127.0.0.1:11435`, configured via `_MODELS.ragchatservice._RAGCHATSERVICE.HOST` and `_MODELS.ragchatservice._RAGCHATSERVICE.PORT`)

   On first launch, `RAGChatService` runs through the same
   [Initial consent workflow](#-initial-consent-workflow) as `RAGLoad`,
   `RAGChat`, and `DocClassify` (hash confirmation and model license consent).

3. **Add the connection in OpenWebUI** — open
   `<your-openwebui-url>/admin/settings/connections` (or navigate to
   **Admin Panel → Settings → Connections**) and add a new **OpenAI API**
   connection (not under the Ollama API section below it):

   | Field | Value |
   | --- | --- |
   | URL | `http://127.0.0.1:11435/v1` |

   Click the **⚙️ gear icon** on the right-hand side of the new connection
   entry to open its detail settings and enter the **API Key** (Bearer
   token):

   | Field | Value |
   | --- | --- |
   | API Key | The value of `_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY` (configured in `Config_Models.py`) |

4. **Select a collection** — after saving the connection, the Chroma DB
   collections loaded by `RAGLoad` appear as selectable models in the
   OpenWebUI model dropdown (`GET /v1/models`). Pick a collection and
   start chatting.

5. **Disable follow-up prompt generation** — by default OpenWebUI
   automatically generates follow-up question suggestions after each
   model response. The follow-ups impose load for the LLM. So if you
   don't need the follow-ups, you should disable them in **one** of the
   following ways:

   - **Per-user (UI):** go to **Settings → Interface** and turn off
     **Follow-Up Auto-Generation** in the *Chat* section.
   - **Globally (Admin UI):** go to **Admin Panel → Settings → Interface** and turn off
     **Follow-Up Auto-Generation**.
   - **Environment variable:** set `ENABLE_FOLLOW_UP_GENERATION=False`
     before starting OpenWebUI.

   For details see the
   [Open WebUI Follow-Up Prompts documentation](https://docs.openwebui.com/features/chat-conversations/chat-features/follow-up-prompts)
   and the
   [`ENABLE_FOLLOW_UP_GENERATION` environment variable reference](https://docs.openwebui.com/reference/env-configuration#enable_follow_up_generation).

> **Note:** All `/v1/` endpoints require a valid `Authorization: Bearer <key>`
> header. The key must match `_MODELS.ragchatservice._RAGCHATSERVICE.API_KEY` in `Config_Models.py`.
> Change the default value before exposing the service beyond localhost.

## 🌍 About internet

See [Internet Access](#-internet-access) for details on how internet connectivity is configured and controlled.

## 🔑 Initial consent workflow

You will be guided through a two-step process which ensures that you:

1. Confirm changes to Config_Models.py, Config_Banned.py, and Config_WebSearch.py. See [Update the hashes](#-update-the-hashes).
2. Consent to the licenses belonging to the models used in Models.py, see [License consent](#-license-consent).
Both are recorded so in future runs these steps are skipped unless you make changes

## ▶️ Start RAGLoad.py

```python
./src/apps/RAGLoad.py
```

The expected configuration hashes are displayed:

## 🔒 Update the hashes

- `_MODELS_CONFIG_HASH = "<new_hash>"`      — update after editing Configuration/Config_Models.py
- `_BANNED_CONFIG_HASH = "<new_hash>"`      — update after editing Configuration/Config_Banned.py
- `_WEB_SEARCH_CONFIG_HASH = "<new_hash>"` — update after editing Configuration/Config_WebSearch.py

After any change in these 3 files the new required hash is displayed at startup of RAGLoad, RAGChat or DocClassify.

You can either copy the expected value from the startup message into
`Config_Global.py` manually, or run the helper script which recomputes all
three hashes and rewrites the slots in `Config_Global.py` in-place:

```powershell
python src\Scripts\RecalcConfigHashes.py
```

The script prints the old and new hash for each of the three pinned config
files and updates `_MODELS_CONFIG_HASH`, `_BANNED_CONFIG_HASH`, and
`_WEB_SEARCH_CONFIG_HASH` accordingly.

## ▶️ Start RAGLoad.py again

```python
./src/apps/RAGLoad.py
```

If you see a `RequestsDependencyWarning` see [Troubleshooting](CONFIGURATION_REFERENCE.md#-troubleshooting).

## 📝 License consent

You are asked to consent to the licenses for the models defined in `Config_Models.py`. With the default used in this repository  `LICENSE_DOWNLOAD = "0"`, RAG‑LCC prompts you with `[y/N]` on each individual license download. You can also set `LICENSE_DOWNLOAD` to `"1"` in `Config_Internet_Env.py` to skip the per-fetch prompt if this is acceptable for your environment and policies. In this case licenses will be fetched online on every run.

This step is repeated for each model that the started application actually requires (for example, `RAGChat` needs the LLM, compliance-check LLM, embedder, cross-encoder, translation model, and Ollama provider — but not the OpenWebUI entry, which is only used by `RAGChatService`). Only the licenses relevant to the launched `.py` file are checked; models not used by that application are skipped. Once consented, license consent is only re-requested if a local license file is missing, the config hash changes, or — when `LICENSE_DOWNLOAD` is enabled — a changed remote license text or TLS certificate is detected.

```text
🟡 License                        mistral._LLM: missing license or metadata
🟡 License                        License consent required
Press Enter to accept as-is or type your email/ID to override: y
🔵 License                        Mistral 7B: fetching LICENSE from https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md
🔵 License                        >>>> Internet connection is set to 'None'
🔵 License                        >>>> Do you allow to online fetch license for  [Mistral 7B] from URL:
↳                                 https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md

>>>>  [y/N] y
🔵 License                        Fetching license online from: https://huggingface.co/datasets/choosealicense/licenses/resolve/main/markdown/apache-2.0.md

🔵 License                        First time online fetch: [mistral._LLM] [Mistral 7B]

=​=====================================================================
  Model License Consent  (first online fetch)
  Section: mistral._LLM  /  Mistral 7B
=​=====================================================================

Press Enter to review the license ...
```

## 🌐 Internet Access

Internet access is configured in `Configuration/Config_Internet_Env.py`.

| Environment Variable | default used in this repository | Purpose |
| --- | --- | --- |
| `LICENSE_DOWNLOAD` | `"0"` | Allow online fetch of model license files defined in `Config_Models.py`. When `"0"`, the Compliance module prompts for per-fetch consent. |
| `NLTK_STOPWORDS_DOWNLOAD` | `"0"` | Allow download of missing NLTK stopwords corpus. When `"0"`, the system falls back to an empty stopword list. |
| `RAG_LCC_NW_TRACE` | `"0"` | Socket-level network tracing (debug). |
| `RAG_LCC_STACK_TRACE` | `"0"` | Stack traces on errors. |
| `HF_HUB_OFFLINE` | `"0"` | Set to `"1"` to disable Hugging Face Hub downloads. Default `"0"` allows model downloads. |
| `TRANSFORMERS_OFFLINE` | `"1"` | Disable transformers library hub access when `"1"`. |
| `HF_DATASETS_OFFLINE` | `"1"` | Disable HF datasets hub access when `"1"`. |
| `ARGOS_STANZA_DOWNLOAD` | `"0"` | Control stanza network access for Argos Translate. When `"0"`, stanza is blocked from downloading — only pre-installed packages are used. When `"1"`, stanza may download missing tokenizer models at runtime. Requires prior license acceptance via `python src\Scripts\ArgosTranslatePackages.py install`. |
| `ARGOS_MODEL_PROVIDER` | `"OPENNMT"` | Force Argos Translate to use local packages only. |
| `ARGOS_CHUNK_TYPE` | "SPACY" | ARGOS_CHUNK_TYPE: Select the sentence boundary detection (SBD) backend |
| `SERVE_OPENWEBUI_CHAT` | `"1"` | Enable the `RAGChatService` HTTP listener. When `"1"`, `RAGChatService.py` starts a uvicorn server that accepts incoming connections on the configured host/port (`ragchatservice` slot in `Config_Models.py`). **This opens a network listener.** Change the default `API_KEY` in `_MODELS.ragchatservice._RAGCHATSERVICE` before exposing the service beyond localhost. |

For convenience, these values are displayed at startup.

```text
🔵 Environment variable           HF_HUB_OFFLINE=1
🔵 Environment variable           HF_DATASETS_OFFLINE=1
...
```

## � Enable Internet (Web) Search (Optional)

Web search is **disabled by default**. When enabled, `RAGChat` / `RAGChatService`
may issue outbound HTTPS queries to a public search backend and merge the
results into the retrieval context alongside local ChromaDB chunks. Read
[LEGAL.md § Web Search](LEGAL.md#-web-search) and [SECURITY.md](SECURITY.md)
before turning the master switch to "1".

### 1. Flip the master switch

Edit `src/Configuration/Config_Internet_Env.py`:

```python
os.environ["WEB_SEARCH_MODE"] = "1"   # one of: "0", "1"
```

| Value | Behaviour |
| --- | --- |
| `"0"` | No internet leg. Web queries are blocked. |
| `"1"` | Full production path. Outbound HTTPS calls are issued. |

The master switch overrides every other web-search setting, including
`_OPENWEB_UI_WEBSEARCH` (see step 4).

### 2. Recalculate the config hash

`Config_Internet_Env.py` is hash-pinned. After editing it, refresh
`_CRITICAL_CONFIG_HASHES["Config_Internet_Env"]` in `Config_Global.py`:

```powershell
python src\Scripts\RecalcConfigHashes.py
```

(or copy the expected hash from the startup error message into
`_WEB_SEARCH_CONFIG_HASH` manually — see [Update the hashes](#-update-the-hashes)).

### 3. Pick a backend (optional)

The active backend is configured inside the `_WEB_SEARCH` dict in
`Config_WebSearch.py`:

```python
_WEB_SEARCH = {
    "backend": "duckduckgo",   # duckduckgo | brave | tavily | bing
    "max_results": 10,
    "max_query_length": 500,
    "block_on_injection": True,
    "default_web_weight": 0.5,
    ...
}
```

- `duckduckgo` is the default and requires **no API key**.
- `brave`, `tavily`, `bing` are recognised names but currently raise
  `NotImplementedError`. Adding a key for one of these is not enough — the
  backend implementation must be supplied first.

### 4. (OpenWebUI only) Auto-enable per request

By default, OpenWebUI requests must pass `web_search=True` explicitly as an
Advanced Parameter. To enable web search automatically for **every** OpenWebUI
request, set in `src/Configuration/Config_WebSearch.py`:

```python
_OPENWEB_UI_WEBSEARCH = True
```

This flag has **no effect** when `WEB_SEARCH_MODE` is `"0"`.
If you set it to `True` while the master switch is not `"1"`, a startup
warning is printed.

### 5. Per-session toggles

Once the master switch is `"1"`, end users can still control web search per
chat / per request:

- **RAGChat CLI:** type `web_search=local_and_web` (or `web_search=local_only`) at the
  prompt. `web_search=web_only` skips local indexes entirely. Boolean `True`/`False` is also accepted as shorthand for `local_and_web`/`local_only`. Also `web_weight=0.7` (RRF weight applied to web results) and
  `fetch_page_content=True` (download page bodies, not just snippets; the original snippet is always retained in metadata for cross-encoder scoring).
- **OpenWebUI Controls sidebar:** the same three knobs appear as Advanced
  Parameters.

### 6. Audit log

Every web search attempt — including ones blocked by the intent filter —
is appended to `_QUERY_LOG` (default `logs/RAGChat/queries.log`). Set
`_QUERY_LOG = ""` in `Config_RAGChat.py` to disable logging.

### 7. Extend the intent filter (optional)

The baseline intent classifier (which decides whether a query *looks like* a
web-search question) lives in `Config_WebSearch.py`. Operators can extend it
without editing that file by populating `WEB_SEARCH_INTENT_EXTENSIONS` in
`Config_Banned.py` (`entity_extensions`, `entity_categories_extra`,
`threshold_overrides`). After editing, rerun
`python src\Scripts\RecalcConfigHashes.py` to refresh `_BANNED_CONFIG_HASH`.

See also: [Web Search — Admin Knobs in CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md#-web-search--admin-knobs)
and [Web-Search Intent Filter Extensions in CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md#-web-search-intent-filter-extensions).

## �🔌 Socket-Level Network Tracing

When `RAG_LCC_NW_TRACE` is set to `"1"`, RAG-LCC monkey-patches Python's `socket.connect` and `socket.getaddrinfo` at startup via `NetworkTracer`. Every DNS resolution and outgoing TCP connection is logged to the console with a timestamp, the destination host/port, resolved IP addresses (forward DNS) or hostname (reverse DNS), and a filtered stack trace showing only project frames (site-packages are excluded). This may assist operators in observing certain Python‑level network activity and associated code paths, but does not guarantee completeness or accuracy. Set `RAG_LCC_STACK_TRACE` to `"1"` alongside it to also get full Python stack traces on errors.

## 🤗 Hugging Face Models (Embedder + Cross-Encoder + Translation)

If you enable HF_HUB_OFFLINE="0" and accept the model licenses, RAG-LCC will download the required Hugging Face models (embedder, cross-encoder, and translation model). In this case, a download consent is requested. Consent is recorded in the `ModelGovernance/consents` directory.

The translation model (`facebook/m2m100_1.2B`, MIT, ~5 GB) is downloaded lazily on the first query that requires translation. Its license (MIT) and download are gated through the same `HFDownloader` consent flow as the embedder — you will be prompted once and consent is recorded under `ModelGovernance/consents`.

If internet is not reachable or HF_HUB_OFFLINE="1" you must install the HF models manually in the local HF cache.

In both cases, make sure the `_HF_HOME` and `_HF_HUB_CACHE` are set correctly:

```python
_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
_HF_HUB_CACHE = os.path.join(_HF_HOME, "hub")
```

- If HF_HUB_OFFLINE="0", you are asked to give consent for the model download. Here the embedding model download.

```text
>>>> Missing model [Snowflake Arctic Embed L v2.0]

>>>> MODEL: snowflake/snowflake-arctic-embed-l-v2.0
>>>> REVISION: None
>>>> SOURCE: https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0

>>>> Do you accept downloading model [Snowflake Arctic Embed L v2.0]?

Proceed? [y/N]
```

Same for remaining models.

- If internet access is disabled, you get:

🔴 INTERNET ACCESS DISABLED       Model 'snowflake/snowflake-arctic-embed-l-v2.0' (revision 'None') was not found in the local cache. Searched cache:
↳                                 C:\Users\pfm\.cache\huggingface\hub Internet access is disabled (HF_HUB_OFFLINE="1"). Enable internet or place the model in
↳                                 the cache directory. (Probably change internet access flags in Configuration/Config_Internet_Env.py)

> **Note:** After completing these steps, internet access is typically no longer required,
> subject to configuration and third‑party behavior.

## 🌍 Argos

When language packages are pre-installed with the install script (`python src\Scripts\ArgosTranslatePackages.py install`), the script requires license consent for Argos Translate before downloading any packages. The consent is recorded in `ModelGovernance/consents/argos_translate/`.

When `ARGOS_STANZA_DOWNLOAD` is set to `"1"` in `Config_Internet_Env.py`, the Argos Translate license consent is also verified at runtime — similar to the LLM and HuggingFace model consent flow. If the license has not been accepted, execution is stopped and the operator is prompted to run the install script to complete the consent.

## 🏃 First run with data

## 📥 Load documents

Use the provided test documents in the `./TestDocs` directory. Load them into the Test Chroma DB collection.

```Windows
python .\src\Apps\RAGLoad.py --doc-dir TestDocs --collection Test
or, since `Config_Global.py` defines  `DOC_DIR` as "TestDocs" and `COLLECTION` as "Test"
python .\src\Apps\RAGLoad.py
```

## 💬 Chat with the documents in the Test Collection

```Windows
python .\src\Apps\RAGChat.py --collection Test
or, since `Config_Global.py` defines `COLLECTION` as "Test"
python .\src\Apps\RAGChat.py
```

## 🏷️ Classify the documents in the TestDocs folder

```Windows
python .\src\Apps\DocClassify.py --doc-dir TestDocs
or, since `Config_Global.py` defines `DOC_DIR` as "TestDocs"
python .\src\Apps\DocClassify.py
```

The classification results can be viewed in the file ./logs/DocClassify_OK*
See also the hints that are displayed by `DocClassify.py` on completion.

For hands-on examples, see [Change provided example prompt in HANDS_ON_TOUR.md](HANDS_ON_TOUR.md#change-provided-example-prompt).

## 📂 Load classified documents into the vector database

After classifying documents with `DocClassify`, you can feed only the approved files
into `RAGLoad` by pointing it at a classification CSV. Pass
`--load-from-classify-csv` with the CSV filename (resolved relative to the `logs/`
directory) or an absolute path:

```Windows
python .\src\Apps\RAGLoad.py --load-from-classify-csv DocClassify_OK_20260325_141005.csv
```

To further narrow ingestion to only rows whose classification columns match a
condition, add `--classify-csv-query` with a SQL WHERE clause. The CSV is loaded
into an in-memory SQLite table, so standard SQLite syntax is supported (`LIKE`,
`AND`, `OR`, `NOT LIKE`, `=`, `!=`, `IN`, etc.):

```Windows
python .\src\Apps\RAGLoad.py --load-from-classify-csv DocClassify_OK_20260325_141005.csv --classify-csv-query "Mammal LIKE '%%Yes%%'"
```

```Windows
python .\src\Apps\RAGLoad.py --load-from-classify-csv DocClassify_OK_20260325_141005.csv --classify-csv-query "Mammal LIKE '%%Yes%%' AND Language = 'English'"
```

The same settings can be made permanent in `Config_RAGLoad.py`:

```python
LOAD_FROM_CLASSIFY_CSV = "DocClassify_OK_20260325_141005.csv"
CLASSIFY_CSV_QUERY = "Mammal LIKE '%Yes%'"  # optional
```

When the classify‑then‑load filter is active, only file paths present in the CSV
are ingested; all other files in `DOC_DIR` are skipped. When a
`CLASSIFY_CSV_QUERY` is set, rows that do not satisfy the SQL condition are excluded
before the allow-set is built. Exclusion checks (`USE_EXCLUSIONS`) are bypassed
automatically because `DocClassify` already evaluated them during its run.

If the CSV file is not found, `RAGLoad` raises a `ClassifyCSVNotFoundError` and
stops immediately.
