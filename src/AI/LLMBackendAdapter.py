from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Iterator, cast

import requests

from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class LLMBackendAdapter(ABC):
    """Abstract transport adapter for LLM backends.

    The RAG stack should keep using the same high-level calling flow; only
    the transport provider needs to vary.
    """

    name: str = "base"

    def __init__(
        self,
        helpers: Helpers | None = None,
        cfg: Config | None = None,
        pretty: PrettyWriter | None = None,
    ) -> None:
        self.helpers = helpers or Helpers()
        self.cfg = cfg or Config()
        self.pretty = pretty or PrettyWriter()

    @property
    def provider_args(self) -> dict[str, Any]:
        """Return provider settings for the active endpoint.

        The adapter keeps working with both the new helper contract
        (get_active_endpoint_args) and older lightweight test doubles that
        only expose get_model_args().
        """
        if hasattr(self.helpers, "get_active_endpoint_args"):
            try:
                return self.helpers.get_active_endpoint_args()
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                pass

        endpoint = str(self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()
        try:
            return self.helpers.get_model_args(endpoint, role=f"_{endpoint.upper()}")
        except TypeError:
            if endpoint == "vllm":
                return self.helpers.get_model_args("_ACTIVE_VLLM")
            return self.helpers.get_model_args("_ACTIVE_OLLAMA")

    @property
    def base_url(self) -> str:
        return str(self.provider_args.get("BASE_URL", ""))

    def headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        api_key = str(self.provider_args.get("API_KEY", "")).strip()
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    @abstractmethod
    def build_payload(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any],
        *,
        streaming: bool,
        top_level_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create the provider-specific request payload."""

    @abstractmethod
    def iter_chunks(self, response: requests.Response) -> Iterator[Any]:
        """Yield one chunk at a time from the backend response."""

    @abstractmethod
    def extract_text(self, chunk: Any) -> str:
        """Extract a content fragment from one backend chunk."""

    def get_context_limit(self, model_name: str) -> int | None:
        """Return provider-specific context length when supported."""
        return None


class OllamaBackendAdapter(LLMBackendAdapter):
    """Adapter for Ollama /api/generate requests."""

    name = "ollama"

    def build_payload(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any],
        *,
        streaming: bool,
        top_level_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": streaming,
            "options": options,
        }
        if top_level_params:
            payload.update(top_level_params)
        return payload

    def iter_chunks(self, response: requests.Response) -> Iterator[Any]:
        if not response.headers.get("content-type", "").startswith("application/json"):
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                try:
                    yield raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    yield raw_line.decode("latin-1", errors="ignore")
            return

        request = getattr(response, "request", None)
        if request is not None and request.method == "POST":
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, (dict, list)):
                yield data
                return

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            try:
                yield raw_line.decode("utf-8")
            except UnicodeDecodeError:
                yield raw_line.decode("latin-1", errors="ignore")

    def extract_text(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                return chunk
            chunk = parsed

        if isinstance(chunk, dict):
            typed_chunk = cast(dict[str, Any], chunk)
            return str(
                typed_chunk.get("response", "")
                or cast(dict[str, Any], typed_chunk.get("message", {})).get(
                    "content", ""
                )
                or ""
            )
        return str(chunk)

    def get_context_limit(self, model_name: str) -> int | None:
        base_url = self.provider_args.get("BASE_URL", "")
        show_url = base_url.replace("/api/generate", "/api/show")
        try:
            response = requests.post(
                show_url,
                json={"name": model_name},
                headers=self.headers(),
                timeout=10,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            # Primary: model_info field (Ollama >= 0.2)
            model_info: dict[str, Any] = data.get("model_info", {})
            arch = str(model_info.get("general.architecture", ""))
            if arch:
                ctx_raw = model_info.get(f"{arch}.context_length")
                if ctx_raw is not None:
                    return int(ctx_raw)

            # Fallback 1: parameters string (e.g. "num_ctx 32768\nstop ...")
            params_str = data.get("parameters", "")
            if isinstance(params_str, str):
                for line in params_str.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[0].lower() == "num_ctx":
                        try:
                            return int(parts[1])
                        except ValueError:
                            pass

            # Fallback 2: modelfile PARAMETER directive
            modelfile = data.get("modelfile", "")
            if isinstance(modelfile, str):
                for line in modelfile.splitlines():
                    parts = line.strip().split()
                    if (
                        len(parts) >= 3
                        and parts[0].upper() == "PARAMETER"
                        and parts[1].lower() == "num_ctx"
                    ):
                        try:
                            return int(parts[2])
                        except ValueError:
                            pass
        except Exception:
            return None
        return None


class VllmBackendAdapter(LLMBackendAdapter):
    """Skeleton for an OpenAI-compatible vLLM backend.

    This keeps the provider-specific transport isolated so the rest of the
    pipeline can reuse the same streaming/non-streaming flow later.
    """

    name = "vllm"

    def build_payload(
        self,
        model: str,
        prompt: str,
        options: dict[str, Any],
        *,
        streaming: bool,
        top_level_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": streaming,
            "temperature": options.get("temperature"),
            "top_p": options.get("top_p"),
            "top_k": options.get("top_k"),
            "max_tokens": options.get("num_predict"),
            "seed": options.get("seed"),
        }
        if top_level_params:
            payload.update(top_level_params)
        return {k: v for k, v in payload.items() if v is not None}

    def iter_chunks(self, response: requests.Response) -> Iterator[Any]:
        request = getattr(response, "request", None)
        if request is not None and request.method == "POST":
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, (dict, list)):
                yield data
                return

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore")
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    def extract_text(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            typed_chunk = cast(dict[str, Any], chunk)
            choices = cast(list[Any], typed_chunk.get("choices") or [])
            if choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    choice = cast(dict[str, Any], first_choice)
                    delta_value = choice.get("delta", None)
                    if isinstance(delta_value, dict):
                        delta = cast(dict[str, Any], delta_value)
                        content = delta.get("content", "")
                        if isinstance(content, str) and content:
                            return content
                    message_value = choice.get("message", None)
                    if isinstance(message_value, dict):
                        message = cast(dict[str, Any], message_value)
                        content = message.get("content", "")
                        if isinstance(content, str):
                            return content
        return ""

    @staticmethod
    def _normalize_model_id(name: str) -> str:
        """Strip HF org prefix, version/variant tags, and punctuation for fuzzy matching.

        Strips the organisation prefix from HuggingFace-style paths, removes common
        version and variant suffixes, then collapses all remaining non-alphanumeric
        characters so that config aliases can be matched against whatever name the
        vLLM server happens to report.

        Examples::

            "meta-llama/Llama-Guard-3-8B"        → "llamaguard38b"
            "llama_guard3_8b"                     → "llamaguard38b"
            "mistralai/Mistral-7B-v0.1"           → "mistral7b"
            "mistral_7b"                          → "mistral7b"
            "meta-llama/Llama-3.1-8B-Instruct"   → "llama318b"
            "llama3_1_8b"                         → "llama318b"
        """
        if "/" in name:
            name = name.split("/")[-1]
        # Strip trailing version / variant tags and everything after them.
        # Covers patterns such as: -v0.1  -Instruct  -Chat  -GGUF  -AWQ  -fp16
        name = re.sub(
            r"[-_](v\d[\d.]*|instruct|chat|hf|gguf|awq|gptq|fp16|int8|int4)\b.*$",
            "",
            name,
            flags=re.IGNORECASE,
        )
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def get_context_limit(self, model_name: str) -> int | None:
        base_url = str(self.provider_args.get("BASE_URL", ""))
        # Derive /v1/models base from /v1/chat/completions
        models_base = base_url.replace("/chat/completions", "/models")
        headers = self.headers()
        try:
            # Primary: individual model endpoint (/v1/models/{model_id})
            resp = requests.get(
                f"{models_base}/{model_name}", headers=headers, timeout=10
            )
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                max_len = data.get("max_model_len")
                if max_len is not None:
                    return int(max_len)
            # Fallback: list endpoint (/v1/models) — handles names with slashes that
            # may be mis-routed as path segments by some HTTP stacks
            resp = requests.get(models_base, headers=headers, timeout=10)
            resp.raise_for_status()
            listing: dict[str, Any] = resp.json()
            entries: list[Any] = listing.get("data", [])

            # First pass: exact ID match
            for entry in entries:
                if isinstance(entry, dict):
                    typed: dict[str, Any] = cast(dict[str, Any], entry)
                    if typed.get("id") == model_name:
                        max_len = typed.get("max_model_len")
                        if max_len is not None:
                            return int(max_len)

            # Second pass: normalised match — handles HF org-prefix and version-suffix
            # differences, e.g. "mistralai/Mistral-7B-v0.1" → "mistral7b" == "mistral_7b"
            norm_target = self._normalize_model_id(model_name)
            for entry in entries:
                if isinstance(entry, dict):
                    typed = cast(dict[str, Any], entry)
                    entry_id = str(typed.get("id", ""))
                    if self._normalize_model_id(entry_id) == norm_target:
                        max_len = typed.get("max_model_len")
                        if max_len is not None:
                            return int(max_len)

            # Third pass: LiteLLM proxy — /model/info endpoint
            # LiteLLM's /v1/models listing omits max_model_len, but its own
            # /model/info endpoint exposes context_length per model under
            # data[*].model_info.context_length.
            model_info_url = base_url.replace("/v1/chat/completions", "/model/info")
            self.pretty.write(
                "I",
                "TokenBudget",
                f"/v1/models did not yield max_model_len for {model_name!r}; "
                f"retrying via LiteLLM /model/info endpoint",
            )
            info_resp = requests.get(model_info_url, headers=headers, timeout=10)
            if info_resp.status_code == 200:
                info_data: dict[str, Any] = info_resp.json()
                for item in info_data.get("data", []):
                    if not isinstance(item, dict):
                        continue
                    typed_item = cast(dict[str, Any], item)
                    item_name = str(typed_item.get("model_name", ""))
                    if (
                        item_name == model_name
                        or self._normalize_model_id(item_name) == norm_target
                    ):
                        mi = typed_item.get("model_info")
                        if isinstance(mi, dict):
                            ctx = cast(dict[str, Any], mi).get("context_length")
                            if ctx is not None:
                                return int(ctx)

            # Fourth pass: single-model deployment safety net — when only one model is
            # registered and neither exact nor normalised names matched (e.g. the vLLM
            # instance was started with a long HF path that survives normalisation
            # differently), the single registered model must be the one we want.
            if len(entries) == 1:
                typed = cast(dict[str, Any], entries[0])
                max_len = typed.get("max_model_len")
                if max_len is not None:
                    return int(max_len)
        except Exception:
            return None
        return None
