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
    name, Ollama /api/show is queried and the result is cached.  This means
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

        # Reserved constants from config (policy, not per-model)
        self.context_cap: int = self.cfg.get_int("TOKEN_BUDGET_CONTEXT_CAP")
        self.reserved_output: int = self.cfg.get_int("TOKEN_BUDGET_RESERVED_OUTPUT")
        self.reserved_system: int = self.cfg.get_int("TOKEN_BUDGET_RESERVED_SYSTEM")

        # Per-model cache: model_name -> effective context limit
        self.cache: dict[str, int] = {}

        # Eagerly warm cache for the main inference model
        self.main_model: str = self.helpers.get_model_args("_LLM")["MODEL"]
        self._load_context_limit(self.main_model)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_context_limit(self, model_name: str) -> int:
        """
        Return the effective context limit for *model_name*, fetching from
        Ollama and caching the result on first call.  Subsequent calls are
        served from the in-memory cache.
        """
        if model_name in self.cache:
            return self.cache[model_name]

        detected: int | None = self.llmCaller.get_model_context_limit(model_name)

        if detected is None:
            limit = self.context_cap
            self.pretty.write(
                "W",
                "TokenBudget",
                f"Ollama unreachable for {model_name!r} — using config cap: {self.context_cap}",
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

        Triggers a lazy Ollama /api/show fetch on first call for that model,
        then serves from cache.  Pass ``None`` to get the main inference model
        limit (``_MODELS.LLM.MODEL``).

        Use this to supply ``num_ctx`` to Ollama so it allocates the full KV-
        cache rather than the Ollama default of 2048.
        """
        return self._load_context_limit(model_name or self.main_model)

    def count_tokens_approx(self, text: str) -> int:
        """
        Approximate token count using word count × 1.3.
        No tokenizer dependency; suitable for budget estimation.
        """
        return int(len(text.split()) * 1.3)

    def compute_dynamic_max_tokens(
        self, prompt: str, model_name: str | None = None, silent: bool = False
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

        Returns the number of output tokens the model may generate, clamped
        between 1 and RESERVED_OUTPUT.
        """
        resolved_model: str = model_name or self.main_model
        context_limit: int = self._load_context_limit(resolved_model)
        prompt_tokens: int = self.count_tokens_approx(prompt)
        available: int = context_limit - self.reserved_system - prompt_tokens

        if available <= 0:
            result = self.reserved_output
            if not silent:
                self.pretty.write(
                    "W",
                    "TokenBudget",
                    f"[{resolved_model}] Prompt ({prompt_tokens} tokens) exceeds context window "
                    f"({context_limit}) minus system reserve ({self.reserved_system}) by "
                    f"{-available} tokens. Granting full reserved_output={result} — "
                    f"output quality may be degraded. "
                    f"Consider raising TOKEN_BUDGET_CONTEXT_CAP (currently {self.context_cap}), "
                    f"increasing TOKEN_BUDGET_RESERVED_OUTPUT (currently {self.reserved_output}), "
                    f"or reducing prompt size. (Config_Globa.py)"
                    f"Note: Ollama defaults to only 2048 tokens if num_ctx is not set explicitly.",
                    color=ORANGE,
                )
            return result

        result: int = min(available, self.reserved_output)
        if not silent:
            self.pretty.write(
                "I",
                "TokenBudget",
                f"[{resolved_model}] context={context_limit} reserved_sys={self.reserved_system} "
                f"prompt≈{prompt_tokens} → max_output_tokens={result}",
            )
        return result
