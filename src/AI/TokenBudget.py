from AI.LLMCaller import LLMCaller
from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import GREEN, ORANGE
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class TokenBudget(SingletonMixin):
    """
    Computes the dynamic output-token budget for any LLM call.

    Context limits are resolved per-model: on first access for a given model
    name, the active backend's context metadata is queried and the result is
    cached.  This means
    the main inference model and the compliance-check model (which may differ)
    each get their own accurate limit without repeated network calls.

    The operator cap TOKEN_BUDGET_CONTEXT_CAP always wins when Ollama reports
    a larger window — this protects weak CPUs / GPUs.

    Formula applied before each call::

        available = context_limit(model) - RESERVED_SYSTEM - prompt_tokens
        max_output_tokens = clamp(available, 1, RESERVED_OUTPUT)

    Token counting uses a word-count approximation (words × 1.3) — no extra
    tokenizer dependency is required.
    """

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        llm_caller: "LLMCaller | None" = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()

        self.llmCaller: LLMCaller = llm_caller or LLMCaller()
        self.helpers: Helpers = Helpers()

        # Token budget constants — read from the active LLM model definition
        # so each model entry in Config_Models.py can declare its own cap and reserves.
        _llm_args: dict = self.helpers.get_model_args("_ACTIVE_LLM")

        def _read_budget_value(key: str, default: int) -> int:
            value = _llm_args.get(key)
            if value is None:
                return self.cfg.get_int(key, default, silent=True)
            return int(value)

        self.context_cap: int = _read_budget_value("TOKEN_BUDGET_CONTEXT_CAP", 32768)
        self.reserved_output: int = _read_budget_value(
            "TOKEN_BUDGET_RESERVED_OUTPUT", 2048
        )
        self.reserved_system: int = _read_budget_value(
            "TOKEN_BUDGET_RESERVED_SYSTEM", 1024
        )

        # Per-model cache: model_name -> effective context limit
        self.cache: dict[str, int] = {}

        # Eagerly warm cache for the main inference model
        self.main_model: str = self.helpers.get_model_args("_ACTIVE_LLM")["MODEL"]
        self._load_context_limit(self.main_model)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_context_limit(self, model_name: str) -> int:
        """
        Return the effective context limit for *model_name*, fetching the
        provider context metadata and caching the result on first call.
        Subsequent calls are
        served from the in-memory cache.
        """
        # Input validation
        if not isinstance(model_name, str) or not model_name.strip():
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid model_name provided: {model_name!r}. Must be a non-empty string.",
                color="RED",
            )
            return self.context_cap

        if model_name in self.cache:
            return self.cache[model_name]

        detected: int | None = self.llmCaller.get_model_context_limit(model_name)

        provider = str(self.cfg.get_str("_ACTIVE_ENDPOINT", "ollama")).lower()
        backend_label = "vLLM" if provider == "vllm" else "Ollama"

        if detected is None:
            limit = self.context_cap
            self.pretty.write(
                "W",
                "TokenBudget",
                f"{backend_label} context-length metadata unavailable for {model_name!r}; "
                f"using config cap: {self.context_cap}",
                color=ORANGE,
            )
        elif detected > self.context_cap:
            limit = self.context_cap
            self.pretty.write(
                "I",
                "TokenBudget",
                f"Model {model_name!r} reports {detected} tokens; "
                f"capped to {self.context_cap} (TOKEN_BUDGET_CONTEXT_CAP)",
                color=ORANGE,
            )
        else:
            limit = detected
            self.pretty.write(
                "I",
                "TokenBudget",
                f"Model {model_name!r}: context_limit={detected} "
                f"(below cap {self.context_cap} — using detected value)",
                color=GREEN,
            )

        self.cache[model_name] = limit
        return limit

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_context_limit(self, model_name: str | None = None) -> int:
        """
        Return the cached effective context limit for *model_name*.

        Triggers a lazy context-length metadata lookup on first call for that
        model,
        then serves from cache.  Pass ``None`` to get the main inference model
        limit (``_MODELS.LLM.MODEL``).

        Use this to supply ``num_ctx`` to Ollama so it allocates the full KV-
        cache rather than the Ollama default of 2048.
        """
        # Input validation
        if model_name is not None and (
            not isinstance(model_name, str) or not model_name.strip()
        ):
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid model_name provided: {model_name!r}. Must be a non-empty string or None.",
                color="RED",
            )
            return self.context_cap

        return self._load_context_limit(model_name or self.main_model)

    def get_effective_context_limit(
        self, model_name: str | None, session: object
    ) -> int:
        """Return the context limit for *model_name*, honouring any per-session override.

        Combines the two-step pattern that appears at every call site:
            ctx_limit = self.tokenBudget.get_context_limit(model)
            ctx_override = getattr(session, "context_size_override", None)
            effective_ctx = min(ctx_override, ctx_limit) if ctx_override is not None else ctx_limit
        """
        ctx_limit: int = self.get_context_limit(model_name)
        ctx_override: int | None = getattr(session, "context_size_override", None)
        return min(ctx_override, ctx_limit) if ctx_override is not None else ctx_limit

    def count_tokens_approx(self, text: str) -> int:
        """
        Approximate token count using word count × 1.3.
        No tokenizer dependency; suitable for budget estimation.
        """
        # Input validation
        if not isinstance(text, str):
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid text provided for token counting: must be a string, got {type(text).__name__}",
                color="RED",
            )
            return 0

        if not text:
            return 0

        return int(len(text.split()) * 1.3)

    def compute_dynamic_max_tokens(
        self,
        prompt: str,
        model_name: str | None = None,
        silent: bool = False,
        model_role: str | None = None,
    ) -> int:
        """
        Compute the output-token budget for a given prompt.

        Parameters
        ----------
        prompt:
            The fully-formatted prompt that will be sent to the model.
        model_name:
            The Ollama model tag to budget for.  Defaults to the main
            inference model (_MODELS.LLM.MODEL).  Pass the compliance-check
            model name here when sizing compliance LLM calls.
        silent:
            When True, suppress the log message.  Pass ``True`` from call
            sites that already emit their own budget summary (e.g.
            ``LLMCaller._resolve_token_budget``).
        model_role:
            The active model selector (e.g., "_ACTIVE_LLM", "_ACTIVE_LLM_CHK").
            When provided, reads TOKEN_BUDGET_* values from that specific model's
            config instead of using the defaults from _ACTIVE_LLM.

        Returns the number of output tokens the model may generate, clamped
        between 1 and RESERVED_OUTPUT.
        """
        # Input validation
        if not isinstance(prompt, str):
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid prompt provided: must be a string, got {type(prompt).__name__}",
                color="RED",
            )
            return self.reserved_output

        if not prompt:
            self.pretty.write(
                "W",
                "TokenBudget",
                "Empty prompt provided. Using reserved_output.",
                color=ORANGE,
            )
            return self.reserved_output

        if model_name is not None and (
            not isinstance(model_name, str) or not model_name.strip()
        ):
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid model_name provided: {model_name!r}. Must be a non-empty string or None.",
                color="RED",
            )
            return self.reserved_output

        if not isinstance(silent, bool):
            self.pretty.write(
                "E",
                "TokenBudget",
                f"Invalid silent parameter: must be a boolean, got {type(silent).__name__}",
                color="RED",
            )
            silent = False

        resolved_model: str = model_name or self.main_model
        context_limit: int = self._load_context_limit(resolved_model)
        prompt_tokens: int = self.count_tokens_approx(prompt)

        # Read model-specific budget values if model_role is provided
        if model_role:
            _role_args: dict = self.helpers.get_model_args(model_role)
            reserved_system = int(
                _role_args.get("TOKEN_BUDGET_RESERVED_SYSTEM", self.reserved_system)
            )
            reserved_output = int(
                _role_args.get("TOKEN_BUDGET_RESERVED_OUTPUT", self.reserved_output)
            )
        else:
            reserved_system = self.reserved_system
            reserved_output = self.reserved_output

        available: int = context_limit - reserved_system - prompt_tokens

        if available <= 0:
            result = reserved_output
            if not silent:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"[{resolved_model}] Prompt ({prompt_tokens} tokens) exceeds context window "
                    f"({context_limit}) minus system reserve ({reserved_system}) by "
                    f"{-available} tokens. Granting full reserved_output={result} — "
                    f"output quality may be degraded. "
                    f"Consider raising TOKEN_BUDGET_CONTEXT_CAP (currently {self.context_cap}), "
                    f"increasing TOKEN_BUDGET_RESERVED_OUTPUT (currently {reserved_output}), "
                    f"or reducing prompt size. (Config_Globa.py)"
                    f"Note: the active backend defaults to only 2048 tokens if num_ctx is not set explicitly.",
                    color=ORANGE,
                )
            return result

        result: int = min(available, reserved_output)
        if not silent:
            self.pretty.write(
                "I",
                "TokenBudget",
                f"[{resolved_model}] context={context_limit} reserved_sys={reserved_system} "
                f"prompt≈{prompt_tokens} → max_output_tokens={result}",
            )
        return result
