import json
import re
import time
from typing import Any, Callable, Dict, Optional, cast

import requests

from Config.Config import Config
from Gui.Colors import ORANGE
from Gui.PrettyWriter import PrettyWriter


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
        from AI.AIHelpers import AIHelpers

        self.aiHelpers: AIHelpers = AIHelpers()

    def redact(
        self, obj: Any, keys: tuple[str, ...] = ("authorization", "api_key", "token")
    ) -> str:
        """
        Redact sensitive keys from a dict for logging/debugging.
        """
        s: str = json.dumps(obj, default=str)
        for k in keys:
            s = s.replace(k, "<REDACTED>")
        return s

    def make_on_chunk(
        self, temperature: float, max_output_tokens: int, top_k: float, top_p: float
    ) -> Callable[[Dict[str, str]], None]:
        """
        Create a callback that receives each parsed chunk and has access
        to the dynamic request parameters (temperature, max_output_tokens, top_k, top_p).
        This factory allows binding request parameters into the callback closure.
        """

        def _on_chunk(chunk: Dict[str, str]) -> None:
            _thinking = chunk.get("thinking", "")
            _content = chunk.get("response", "")
            _raw = chunk.get("raw", "")
            # Reference parameters in closure to prevent Python optimization
            _ = (temperature, max_output_tokens, top_k, top_p)
            # Users can override with custom callback

        return _on_chunk

    def _resolve_token_budget(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        context_size: int | None,
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
            prompt, model, silent=True
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

    def call_llm(
        # Main method to call LLM endpoint with streaming and callback support.
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_k: float,
        top_p: float,
        max_output_tokens: int,
        answer_is_json: bool = True,
        use_ollama_gpu: bool = True,
        template_name: str | None = None,
        stream_url: Optional[str] = None,
        on_chunk: Optional[Callable[[Dict[str, str]], None]] = None,
        timeout: Optional[float] = None,
        streaming: bool | None = None,
        stage: str = "",
        context_size: int | None = None,
    ) -> Dict[str, str]:

        timeout = timeout or self.cfg.get_int("REQUEST_TIMEOUT")
        base_url: str = stream_url or self.cfg.get_str("_OLLAMA_BASE_URL")
        streaming = streaming or self.cfg.get_bool("OLLAMA_STREAMING_REQ")
        headers: dict[str, str] = {"Content-Type": "application/json"}

        # Authoritative token-budget & context-size resolution
        max_output_tokens, resolved_ctx = self._resolve_token_budget(
            model,
            prompt,
            max_output_tokens,
            context_size,
        )

        # Build request payload with sampling parameters
        # Ollama expects sampling knobs inside "options", not at top level.
        # num_predict = output tokens; num_ctx = KV-cache / context window size.
        # Without num_ctx Ollama defaults to 2048 regardless of model capacity.
        options: dict[str, Any] = {
            "num_predict": max_output_tokens,
            "num_ctx": resolved_ctx,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        # Force CPU mode if requested (overrides Modelfile defaults)
        if use_ollama_gpu is False:
            options["num_gpu"] = 0

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": streaming,
            "options": options,
        }

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
                f"max_output_tokens: {max_output_tokens} num_ctx: {resolved_ctx} temperature: {temperature} top_k_ {top_k} top_p_ {top_p} use_ollama_gpu: {use_ollama_gpu} streaming: {streaming}",
            )

            start: float = time.perf_counter()
            # Log request details for debugging
            if self.cfg.get_int("DEBUG_LEVEL") >= 80:
                self.pretty.write("D", "OLLAMA REQUEST URL", f"{base_url}")
                self.pretty.write(
                    "D", "OLLAMA REQUEST HEADERS", f"{self.redact(headers)}"
                )
                self.pretty.write(
                    "D", "OLLAMA REQUEST PAYLOAD", f"{self.redact(payload)}"
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
            if self.cfg.get_int("DEBUG_LEVEL") >= 80:
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

            for raw_line in resp.iter_lines(decode_unicode=False):
                # Process each streamed line from LLM response
                if not raw_line:
                    continue

                try:
                    line: str = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("latin-1", errors="ignore")

                raw_acc += line + "\n"
                if len(first_lines) < MAX_DEBUG_LINES:
                    first_lines.append(line)
                if self.cfg.get_int("DEBUG_LEVEL") >= 100:
                    self.pretty.write(
                        "D", f"STREAM RAW LINE {len(line)} bytes", f"{line}"
                    )

                # Try to parse JSON chunk
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON chunk: append raw text and emit via callback
                    content_acc += line
                    if on_chunk:
                        on_chunk({"thinking": "", "content": line, "raw": line})
                    continue

                # Extract streamed text (Ollama uses "response" or nested "message"->"content")
                resp_text: str = ""
                if isinstance(chunk, dict):
                    typed_chunk: dict[str, Any] = cast(dict[str, Any], chunk)
                    resp_text = str(
                        typed_chunk.get("response", "")
                        or typed_chunk.get("message", {}).get("content", "")
                        or ""
                    )

                if resp_text:
                    content_acc += resp_text
                    if on_chunk:
                        on_chunk({"thinking": "", "content": resp_text, "raw": line})

                # Check last non-empty token sequence for a complete label
                candidate = (
                    content_acc.strip().splitlines()[-1].strip()
                    if content_acc.strip()
                    else ""
                )
                candidate_tail = (
                    candidate[-3:].strip() if len(candidate) > 3 else candidate
                )

                if label_re.match(candidate) or label_re.match(candidate_tail):
                    label = candidate if label_re.match(candidate) else candidate_tail
                    label_normalized = label.replace(" ", "")
                    candidate_label = label_normalized
                    if self.cfg.get_int("DEBUG_LEVEL") >= 80:
                        self.pretty.write(
                            "D", "Detected classifier label", f"{candidate_label}"
                        )
                    # set content_acc to normalized label and break
                    content_acc = candidate_label
                    break

                # If the chunk indicates done, stop reading
                if isinstance(chunk, dict) and chunk.get("done") is True:  # type: ignore[reportUnknownMemberType]
                    if self.cfg.get_int("DEBUG_LEVEL") >= 80:
                        self.pretty.write(
                            "D", "Stream signaled done", f"{repr(content_acc)}"
                        )
                    break

            # Log first few raw lines for debugging
            if self.cfg.get_int("DEBUG_LEVEL") >= 80:
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

            return {"content": content_acc, "thinking": thinking_acc, "raw": raw_acc}

        except requests.exceptions.RequestException as e:
            self.pretty.write("E", "OLLAMA request", f"Failed {base_url}")
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
        base_url: str = self.cfg.get_str("_OLLAMA_BASE_URL")
        show_url: str = base_url.replace("/api/generate", "/api/show")
        try:
            resp = requests.post(
                show_url,
                json={"name": model_name},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            model_info: dict[str, Any] = data.get("model_info", {})
            arch: str = str(model_info.get("general.architecture", ""))
            if not arch:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"No architecture key in /api/show for {model_name}",
                )
                return None
            ctx_key: str = f"{arch}.context_length"
            ctx_raw: Any = model_info.get(ctx_key)
            if ctx_raw is None:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"Key '{ctx_key}' not found in model_info for {model_name}",
                )
                return None
            ctx: int = int(ctx_raw)
            self.pretty.write(
                "I",
                "TokenBudget",
                f"Detected context_length={ctx} for {model_name} via /api/show",
            )
            return ctx
        except Exception as e:
            self.pretty.write(
                "W", "TokenBudget", f"Could not query /api/show for {model_name}: {e}"
            )
            return None
