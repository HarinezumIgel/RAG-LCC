import json
import re
import time
from typing import Any, Callable, Dict, Optional, cast

import requests

from AI.LLMBackendAdapter import (LLMBackendAdapter, OllamaBackendAdapter,
                                  VllmBackendAdapter)
from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers
from Helpers.PerfLogger import PerfLogger


class LLMCaller:
    """
    Wrapper for calling an Ollama-compatible LLM endpoint with streaming.
    Provides:
    - call_llm(): main streaming call
    - make_on_chunk(): callback factory that binds temperature + max_output_tokens
    Handles request/response, streaming, and callback logic for LLMs.
    """

    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.pretty: PrettyWriter = PrettyWriter()
        self.helpers: Helpers = Helpers()

    def redact(
        self, obj: Any, keys: tuple[str, ...] = ("authorization", "api_key", "token")
    ) -> str:
        """
        Redact sensitive keys from a dict for logging/debugging.
        """
        if isinstance(obj, dict):
            data = cast(dict[str, Any], obj)
            redacted: dict[str, Any] = {}
            for key, value in data.items():
                redacted["<REDACTED>" if key in keys else key] = (
                    "<REDACTED>" if key in keys else value
                )
            return json.dumps(redacted, default=str)
        return json.dumps(obj, default=str)

    def make_on_chunk(
        self, ollama_options: dict[str, Any]
    ) -> Callable[[Dict[str, str]], None]:
        """
        Create a callback that receives each parsed chunk and has access
        to the Ollama options dict (temperature, num_predict, top_k, top_p, etc.).
        This factory allows binding request parameters into the callback closure.
        """

        def _on_chunk(chunk: Dict[str, str]) -> None:
            # Reference parameters in closure to prevent Python optimization
            _ = (chunk, ollama_options)
            # Users can override with custom callback

        return _on_chunk

    def _resolve_token_budget(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        context_size: int | None,
        model_role: str | None = None,
    ) -> tuple[int, int]:
        """Resolve the authoritative output-token budget and context window.

        Computes the token budget from the actual full prompt using the
        word-count approximation (words × 1.3).  This runs inside
        ``call_llm`` — the last point before the request — so the budget
        is based on the real prompt, not an early estimate.

        Parameters
        ----------
        model:
            Ollama model tag.
        prompt:
            The fully-assembled prompt that will be sent to the model.
        max_output_tokens:
            The caller's *estimated* output-token cap (planning value).
        context_size:
            Caller-supplied context window hint.  ``None`` means "let the
            budget system figure it out from the model metadata".

        Returns
        -------
        (resolved_max_output_tokens, resolved_context_size)
        """
        # Lazy import to avoid circular dependency (TokenBudget imports LLMCaller).
        from AI.TokenBudget import TokenBudget

        token_budget: TokenBudget = TokenBudget()

        # --- context window (num_ctx) ---
        resolved_ctx: int = (
            context_size
            if context_size is not None
            else token_budget.get_context_limit(model)
        )
        if context_size is None:
            self.pretty.write(
                "I",
                "TokenBudget",
                f"Caller did not supply context_size; resolved num_ctx={resolved_ctx} for {model}",
            )

        # --- output-token budget (num_predict) ---
        resolved_output: int = token_budget.compute_dynamic_max_tokens(
            prompt, model, silent=True, model_role=model_role
        )
        if resolved_output != max_output_tokens:
            self.pretty.write(
                "I",
                "TokenBudget",
                f"Authoritative budget={resolved_output} vs caller estimate={max_output_tokens}; "
                f"using authoritative value",
                color=ORANGE,
            )

        # Always log the resolved values so the user sees the token budget
        # for every LLM call — not just the first time a model is detected.
        prompt_tokens: int = token_budget.count_tokens_approx(prompt)
        self.pretty.write(
            "I",
            "TokenBudget",
            f"[{model}] num_ctx={resolved_ctx} prompt≈{prompt_tokens} "
            f"→ num_predict={resolved_output}",
        )

        return resolved_output, resolved_ctx

    def _create_backend_adapter(self) -> LLMBackendAdapter:
        """Select the transport adapter for the active provider."""
        provider = str(self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()
        if provider == "vllm":
            return VllmBackendAdapter(self.helpers, self.cfg, self.pretty)
        return OllamaBackendAdapter(self.helpers, self.cfg, self.pretty)

    def call_llm(
        # Main method to call LLM endpoint with streaming and callback support.
        self,
        model: str,
        prompt: str,
        ollama_options: dict[str, Any],
        *,
        answer_is_json: bool = True,
        template_name: str | None = None,
        on_chunk: Optional[Callable[[Dict[str, str]], None]] = None,
        timeout: Optional[float] = None,
        streaming: bool | None = None,
        stage: str = "",
        top_level_params: dict[str, Any] | None = None,
        model_role: str | None = None,
    ) -> Dict[str, str]:

        # Use provided timeout, otherwise get from provider configuration
        if timeout is None:
            adapter = self._create_backend_adapter()
            timeout = adapter.provider_args.get("REQUEST_TIMEOUT", 120.0)
        else:
            timeout = timeout
        adapter = self._create_backend_adapter()
        provider_label: str = adapter.name.upper()
        provider_args = adapter.provider_args
        base_url: str = adapter.base_url
        streaming = bool(streaming or provider_args.get("STREAMING_REQ", False))
        headers: dict[str, str] = adapter.headers()

        # Extract token-budget inputs from the options dict; _resolve_token_budget
        # will rewrite them with authoritative values.
        max_output_tokens: int = int(ollama_options.pop("num_predict", 0))
        context_size: int | None = ollama_options.pop("num_ctx", None)
        if context_size is not None:
            context_size = int(context_size)

        # Authoritative token-budget & context-size resolution
        max_output_tokens, resolved_ctx = self._resolve_token_budget(
            model,
            prompt,
            max_output_tokens,
            context_size,
            model_role,
        )

        # Write resolved values back into options.
        # Ollama expects sampling knobs inside "options", not at top level.
        # num_predict = output tokens; num_ctx = KV-cache / context window size.
        # Without num_ctx Ollama defaults to 2048 regardless of model capacity.
        # All other keys (temperature, top_k, top_p, seed, mirostat, …) pass through.
        ollama_options["num_predict"] = max_output_tokens
        ollama_options["num_ctx"] = resolved_ctx

        options: dict[str, Any] = ollama_options

        payload: dict[str, Any] = adapter.build_payload(
            model,
            prompt,
            options,
            streaming=streaming,
            top_level_params=top_level_params,
        )

        thinking_acc: str = ""
        content_acc: str = ""
        raw_acc: str = ""

        try:
            # Main request/streaming logic
            self.pretty.write(
                "I",
                "Call LLM",
                f"Model: {model} prompt template: {template_name} stage: {stage}",
            )
            self.pretty.write(
                "I",
                "Call LLM",
                f"options: {options} streaming: {streaming}",
            )

            start: float = time.perf_counter()
            PerfLogger().log(
                "LLMCaller.call_llm", f"start inference model={model!r} stage={stage!r}"
            )
            # Log request details for debugging
            if DebugHelper.check(self.cfg, 80):
                self.pretty.write("D", f"{provider_label} REQUEST URL", f"{base_url}")
                self.pretty.write(
                    "D", f"{provider_label} REQUEST HEADERS", f"{self.redact(headers)}"
                )
                self.pretty.write(
                    "D", f"{provider_label} REQUEST PAYLOAD", f"{self.redact(payload)}"
                )
                # Include prompt preview for verification
                self.pretty.write(
                    "D", "PROMPT PREVIEW", f"{prompt[:500].replace(chr(10), '\\n')}"
                )

            resp: requests.Response = requests.post(
                base_url,
                json=payload,
                headers=headers,
                stream=streaming,
                timeout=timeout,
            )
            if DebugHelper.check(self.cfg, 80):
                self.pretty.write("D", "OLLAMA RESPONSE STATUS", f"{resp.status_code}")
                self.pretty.write(
                    "D", "OLLAMA RESPONSE HEADERS", f"{dict(resp.headers)}"
                )
            resp.raise_for_status()

            # Streaming loop with robust accumulation and label detection
            first_lines: list[str] = []
            MAX_DEBUG_LINES: int = 10
            label_re: re.Pattern[str] = re.compile(r"^S\s*\d$")  # matches "S6" or "S 6"
            candidate_label: str | None = None

            for chunk in adapter.iter_chunks(resp):
                # Process each streamed chunk from the backend response.
                # Parse JSON strings into dicts when possible so done/label markers
                # and provider-specific content extraction behave consistently.
                parsed_chunk: Any = chunk
                if isinstance(chunk, str):
                    try:
                        parsed_chunk = json.loads(chunk)
                    except json.JSONDecodeError:
                        parsed_chunk = chunk

                raw_line = (
                    chunk
                    if isinstance(chunk, str)
                    else json.dumps(chunk, ensure_ascii=False)
                )
                raw_acc += raw_line + "\n"
                if len(first_lines) < MAX_DEBUG_LINES:
                    first_lines.append(raw_line)
                if DebugHelper.check(self.cfg, 100):
                    if isinstance(chunk, str):
                        self.pretty.write(
                            "D", f"STREAM RAW LINE {len(raw_line)} bytes", f"{raw_line}"
                        )
                    else:
                        self.pretty.write(
                            "D",
                            "STREAM JSON CHUNK",
                            json.dumps(chunk, ensure_ascii=False),
                        )

                resp_text = adapter.extract_text(parsed_chunk)
                if resp_text:
                    content_acc += resp_text
                    if on_chunk:
                        on_chunk(
                            {"thinking": "", "content": resp_text, "raw": raw_line}
                        )

                # Check last non-empty token sequence for a complete label.
                candidate = (
                    content_acc.strip().splitlines()[-1].strip()
                    if content_acc.strip()
                    else ""
                )
                candidate_tail = (
                    candidate[-3:].strip() if len(candidate) > 3 else candidate
                )

                if isinstance(parsed_chunk, dict):
                    chunk_dict = cast(dict[str, Any], parsed_chunk)
                    if label_re.match(candidate) or label_re.match(candidate_tail):
                        label = (
                            candidate if label_re.match(candidate) else candidate_tail
                        )
                        label_normalized = label.replace(" ", "")
                        candidate_label = label_normalized
                        if DebugHelper.check(self.cfg, 80):
                            self.pretty.write(
                                "D", "Detected classifier label", f"{candidate_label}"
                            )
                        content_acc = candidate_label
                        break

                    if chunk_dict.get("done") is True:
                        if DebugHelper.check(self.cfg, 80):
                            self.pretty.write(
                                "D", "Stream signaled done", f"{repr(content_acc)}"
                            )
                        break

            # Log first few raw lines for debugging
            if DebugHelper.check(self.cfg, 80):
                self.pretty.write(
                    "D",
                    "First few lines",
                    f"{len(first_lines)}\n{'\n'.join(first_lines)}",
                )
                self.pretty.write(
                    "D", "Full raw stream length", f"{len(raw_acc)} bytes"
                )

            end: float = time.perf_counter()
            elapsed: float = end - start
            minutes: int = int(elapsed // 60)
            seconds: int = int(elapsed % 60)

            self.pretty.write(
                "I",
                "Call LLM",
                f"Elapsed time calling: {model} took {minutes:02d}:{seconds:02d}",
            )
            PerfLogger().log(
                "LLMCaller.call_llm",
                f"stop  inference model={model!r} stage={stage!r} elapsed={elapsed:.3f}s",
            )

            return {"content": content_acc, "thinking": thinking_acc, "raw": raw_acc}

        except requests.exceptions.RequestException as e:
            self.pretty.write("E", f"{provider_label} request", f"Failed {base_url}")
            return {
                "content": content_acc,
                "thinking": thinking_acc,
                "raw": raw_acc,
                "error": str(e),
            }

    def get_model_context_limit(self, model_name: str) -> int | None:
        """
        Query Ollama /api/show to read the model's declared context length.

        The key lives under model_info as "{architecture}.context_length"
        (e.g. "llama.context_length" for Mistral/LLaMA families).
        Returns None on any error so callers can fall back gracefully.
        """
        adapter = self._create_backend_adapter()
        try:
            ctx_value = adapter.get_context_limit(model_name)
            if ctx_value is not None:
                self.pretty.write(
                    "I",
                    "TokenBudget",
                    f"Detected context_length={ctx_value} for {model_name} via {adapter.name} adapter",
                )
                return ctx_value
        except Exception as e:
            self.pretty.write(
                "W",
                "TokenBudget",
                f"Could not query context limit for {model_name}: {e}",
            )

        return None
