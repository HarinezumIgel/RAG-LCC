import re
from typing import Any, Dict, List, Pattern, Tuple

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.PrettyWriter import PrettyWriter
from Helpers.DebugHelper import DebugHelper
from Helpers.Helpers import Helpers

MaskValue = str
RawSpecDict = Dict[str, Dict[str, Any]]
CompiledSpec = Tuple[Pattern[str], MaskValue, str]  # (compiled_pattern, mask, name)


class Masker(SingletonMixin):
    """
    Notes:
    - Masks are simple replacement strings only (no callables).
    - Rules are applied in ascending priority order, then by name.
    - If no config is provided, no rules are loaded (empty spec list).
    - Every debug pretty call is guarded by `if DebugHelper.check(self.cfg, 40):`.
    """

    def __init__(
        self, cfg: "Config | None" = None, *, pretty: "PrettyWriter | None" = None
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = Helpers()
        # Users can set this to control verbose debug output; default 0 (quiet).

        self.apply_masking: bool = self.cfg.get_bool(
            self.helpers.get_compliance_config_slot("MASKING") + ".APPLY_MASKING"
        )
        # Read config from the requested variable name
        regex_definitions: Any = self.helpers.get_masking_regexes_config_slot()
        if regex_definitions is None:
            self.pretty.write(
                "W", "Masker", "No _MASKING_REGEXES found in cfg; no mask rules loaded"
            )
            self.specs: List[CompiledSpec] = []
        else:
            self.pretty.write(
                "I", "Masker", "Loaded _MASKING_REGEXES from configuration"
            )
            self.specs = self._normalize_and_compile(regex_definitions)

        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D", "Masker", f"Compiled {len(self.specs)} mask patterns"
            )

    def mask(self, text: str) -> str:
        """
        Apply all configured masks in priority order and return masked text.
        Rules are applied in ascending priority order, then by name.
        Masks are simple replacement strings applied with re.sub().
        """
        if self.apply_masking is False or not text:
            return text

        if DebugHelper.check(self.cfg, 40):
            self.pretty.write(
                "D", "Masker", f"Masking text with {len(self.specs)} rules"
            )

        out: str = text
        matches: int = 0
        for pattern, mask, name in self.specs:
            before: str = out
            try:
                out = pattern.sub(mask, out)
            except Exception as e:
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D", "Masker", f"Error applying spec '{name}': {e}"
                    )
                out = before
            if out != before:
                matches += 1
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write("I", "Masker", f"Applied '{name}' using {mask}")
        self.pretty.write(
            "I",
            "Masker",
            f"{matches} of {len(self.specs)} rules produced matches and were replaced",
        )
        return out

    def _normalize_and_compile(self, raw: RawSpecDict) -> List[CompiledSpec]:
        """
        Normalize dict-based config into ordered compiled specs.
        Accepts dict: NAME -> { pattern, mask, enabled, priority, desc }.
        """
        specs: List[CompiledSpec] = []

        if not raw:
            raise TypeError("_MASKING_REGEXES must be a dict of named specs")

        items: list[tuple[int, str, Any, str]] = []
        for name, cfg in raw.items():
            # Skip disabled specs
            if not cfg.get("enabled", True):
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D", "Masker", f"Spec '{name}' disabled; skipping"
                    )
                continue

            # Validate required fields
            if "pattern" not in cfg or "mask" not in cfg:
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D",
                        "Masker",
                        f"Spec '{name}' missing required 'pattern' or 'mask'; skipping",
                    )
                continue

            priority: int = int(cfg.get("priority", 100))
            pattern: Any = cfg["pattern"]
            mask: str = cfg["mask"]

            # Mask must be a string for simple replacement
            if not mask:
                if DebugHelper.check(self.cfg, 40):
                    self.pretty.write(
                        "D", "Masker", f"Spec '{name}' mask must be a string; skipping"
                    )
                continue

            items.append((priority, name, pattern, mask))

        # Sort by priority then name for deterministic order
        items.sort(key=lambda t: (t[0], t[1]))

        for priority, name, pattern, mask in items:
            # Compile pattern if not already compiled
            if isinstance(pattern, re.Pattern):
                compiled: re.Pattern[str] = pattern  # type: ignore[reportUnknownVariableType]
            else:
                compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            specs.append((compiled, mask, name))
            if DebugHelper.check(self.cfg, 40):
                self.pretty.write(
                    "D", "Masker", f"Loaded spec '{name}' (priority={priority})"
                )

        return specs
