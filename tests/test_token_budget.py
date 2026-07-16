# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeArgument=false
# pyright: reportReturnType=false, reportUnknownLambdaType=false
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class StubPrettyWriter:
    def write(self, *args, **kwargs):
        return None


class StubLLMCaller:
    def get_model_context_limit(self, model_name):
        return None


class GuardConfig:
    def get_int(self, key, default=0, silent=False):
        raise AssertionError(f"unexpected lookup for {key}")

    def get(self, key, default=None, silent=False):
        return default

    def get_str(self, key, default="", silent=False):
        return default


def test_token_budget_uses_model_config_without_config_fallback(monkeypatch):
    from AI.TokenBudget import TokenBudget
    from Helpers.Helpers import Helpers

    TokenBudget._reset()

    monkeypatch.setattr(
        Helpers,
        "get_model_args",
        lambda self, selector: {
            "MODEL": "mistral:7b",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
        },
    )

    try:
        budget = TokenBudget(
            cfg=GuardConfig(),
            pretty=StubPrettyWriter(),
            llm_caller=StubLLMCaller(),
        )
    finally:
        TokenBudget._reset()

    assert budget.context_cap == 32768
    assert budget.reserved_output == 2048
    assert budget.reserved_system == 1024


def test_token_budget_message_uses_active_backend_name(monkeypatch):
    from AI.LLMCaller import LLMCaller
    from AI.TokenBudget import TokenBudget
    from Helpers.Helpers import Helpers

    TokenBudget._reset()

    class CapturePrettyWriter:
        def __init__(self):
            self.messages = []

        def write(self, level, tag, message, color=None):
            self.messages.append(message)

    class StubConfig:
        def get_int(self, key, default=0, silent=False):
            return default

        def get_str(self, key, default="", silent=False):
            return {"_ACTIVE_ENDPOINT": "vllm"}.get(key, default)

    pretty = CapturePrettyWriter()

    monkeypatch.setattr(
        Helpers,
        "get_model_args",
        lambda self, selector: {
            "MODEL": "llama3.1:8b",
            "TOKEN_BUDGET_CONTEXT_CAP": 32768,
            "TOKEN_BUDGET_RESERVED_OUTPUT": 2048,
            "TOKEN_BUDGET_RESERVED_SYSTEM": 1024,
        },
    )
    monkeypatch.setattr(
        LLMCaller, "get_model_context_limit", lambda self, model_name: None
    )

    try:
        TokenBudget(cfg=StubConfig(), pretty=pretty, llm_caller=LLMCaller())
    finally:
        TokenBudget._reset()

    assert any(
        "vLLM" in message and "context-length" in message for message in pretty.messages
    )
