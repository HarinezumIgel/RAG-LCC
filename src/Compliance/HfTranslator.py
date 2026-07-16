"""HfTranslator — Hugging Face based offline translation backend.

Provides a `translate_text(text, target_lang, source_lang)` API that mirrors
the signature of :pymeth:`Compliance.SharedHelpers.SharedHelpers.translate_text`
so it can be slotted in as an alternative backend for the user-query
normalisation path in :pymod:`Chat.RAGChatImpl`.

The default model is ``facebook/m2m100_418M`` (MIT-licensed), a many-to-many
multilingual translation model covering 100 languages. It handles short /
colloquial chat queries substantially better than the OPUS-MT based Argos
packages and is small enough (~1.7 GB) to run comfortably on CPU.

Loading is lazy: the model is only materialised on the first call to
``translate_text``. Consent + download are routed through the existing
:pyclass:`Compliance.HFDownloader.HFDownloader` flow exactly the same way the
embedder does it, so users get the standard prompt and audit trail.

Argos Translate is intentionally still used by the Compliance pipeline for
banlist / synonym translation — single-token vocabulary translation where its
simpler model is adequate.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from Commons.SingletonMixin import SingletonMixin
from Config.Config import Config
from Gui.Colors import ORANGE, RED
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers


class HfTranslator(SingletonMixin):
    """Singleton wrapper around a HF seq2seq translation model (M2M-100 by default)."""

    def __init__(
        self,
        *,
        cfg: "Config | None" = None,
        pretty: "PrettyWriter | None" = None,
        helpers: "Helpers | None" = None,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        self.cfg: Config = cfg or Config()
        self.pretty: PrettyWriter = pretty or PrettyWriter()
        self.helpers: Helpers = helpers or Helpers()

        # NLTK-name → ISO 639-1 code map (reuse Argos definitions).
        code_to_name: Dict[str, Any] = self.cfg.get_dict(
            "_ARGOS_DEFINITIONS.LANG_CODE_TO_NAME"
        )
        self._lang_name_to_code: Dict[str, str] = {
            str(name).lower(): str(code).lower() for code, name in code_to_name.items()
        }

        # Cache keyed by (text, target_iso, source_iso).
        self._cache: Dict[Tuple[str, str, str], str] = {}
        # Lazy-loaded model + tokenizer.
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._load_lock = threading.Lock()
        # Track once-per-process warnings.
        self._warned_targets: set[str] = set()

    # ------------------------------------------------------------------ utils
    def _to_iso(self, lang: str) -> str:
        """Normalise an NLTK lang name (``"german"``) or ISO code to ISO."""
        if not lang:
            return ""
        low = lang.lower()
        return self._lang_name_to_code.get(low, low)

    # --------------------------------------------------------------- loading
    def _ensure_loaded(self) -> bool:
        """Lazy-load the HF translation model. Returns True on success."""
        if self._model is not None and self._tokenizer is not None:
            return True

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return True

            # Heavy imports kept local so import-time cost is paid only by
            # callers that actually use the HF translation backend.
            try:
                import torch  # type: ignore[import-not-found]
                from transformers import (  # type: ignore[import-not-found]
                    M2M100ForConditionalGeneration, M2M100Tokenizer)
            except Exception as exc:  # pragma: no cover - env-dependent
                self.pretty.write(
                    "E",
                    "HfTranslator",
                    f"transformers/torch import failed: {exc!r}",
                    color=RED,
                )
                return False

            # Resolve model config and trigger consent / download if needed.
            model_args: Dict[str, Any] = self.helpers.get_model_args(
                "_ACTIVE_TRANSLATION"
            )
            model_name: str = str(
                model_args.get("model_name") or model_args.get("MODEL") or ""
            )
            revision: Optional[str] = model_args.get("revision")
            use_gpu: bool = bool(model_args.get("USE_GPU", True))

            # Trigger consent + ensure cached. Imported lazily so the module
            # has no hard dependency on HFDownloader at import-time (keeps
            # tests light).
            from Commons.Exceptions import (InternetConnectionDisabledError,
                                            UserNoDownLoadAccept)
            from Compliance.HFDownloader import HFDownloader

            try:
                HFDownloader().download("_MODELS._TRANSLATION")
            except (InternetConnectionDisabledError, UserNoDownLoadAccept):
                # These are user-actionable conditions: the program must
                # stop so the user can either enable HF_HUB_OFFLINE="0" /
                # pre-stage the model, or accept the consent prompt.
                # Propagate to the global handler in StartupCommons.
                raise
            except Exception as exc:
                self.pretty.write(
                    "E",
                    "HfTranslator",
                    f"Consent/download for {model_name} failed: {exc!r}",
                    color=RED,
                )
                return False

            # Re-read args in case download() resolved REVISION.
            model_args = self.helpers.get_model_args("_ACTIVE_TRANSLATION")
            revision = model_args.get("revision")

            # Pick device + dtype.
            #
            # Defaulting USE_GPU to False keeps the GPU available for the
            # retrieval stack (embedder + reranker + KeyBERT). M2M-100 418M
            # is small enough that CPU latency is acceptable for one short
            # query per turn. Operators with spare GPU headroom can flip
            # USE_GPU=True in Config_Models.py.
            try:
                cuda_ok = bool(use_gpu and torch.cuda.is_available())
            except Exception:
                cuda_ok = False
            device = torch.device("cuda" if cuda_ok else "cpu")
            dtype = torch.float16 if cuda_ok else torch.float32

            self.pretty.write(
                "I",
                "HfTranslator",
                f"Loading {model_name} (rev={revision or 'main'}) "
                f"on {device} dtype={dtype}",
            )

            # M2M-100's HF checkpoint stores `model.shared.weight`,
            # `encoder.embed_tokens.weight`, `decoder.embed_tokens.weight`
            # and `lm_head.weight` separately even though its config
            # requests them tied. Transformers logs three loud warnings on
            # every load asking the user to set `tie_word_embeddings=False`.
            # The warnings are not actionable for end users, so silence
            # them for the duration of `from_pretrained()`.
            try:
                from transformers.utils import \
                    logging as hf_logging  # type: ignore[import-not-found]

                prev_verbosity = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
            except Exception:
                hf_logging = None  # type: ignore[assignment]
                prev_verbosity = None

            try:
                # M2M-100 ships a SentencePiece tokenizer; ``M2M100Tokenizer``
                # is the slow Python implementation (no fast variant exists).
                # Requires the ``sentencepiece`` package.
                self._tokenizer = M2M100Tokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    model_name,
                    revision=revision or "main",
                    local_files_only=True,
                )
                self._model = M2M100ForConditionalGeneration.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    model_name,
                    revision=revision or "main",
                    local_files_only=True,
                    torch_dtype=dtype,
                )
                self._model.to(device).eval()
                self._device = device
                # The shipped GenerationConfig pre-sets ``max_length=200``
                # and ``early_stopping=True``. The first conflicts with our
                # per-call ``max_new_tokens`` (transformers prints a warning
                # on every generate); the second is invalid for greedy /
                # single-beam search. Clear both so generate() stays quiet.
                try:
                    gen_cfg = getattr(self._model, "generation_config", None)
                    if gen_cfg is not None:
                        gen_cfg.max_length = None  # type: ignore[assignment]
                        gen_cfg.early_stopping = False
                except Exception:
                    pass
            except Exception as exc:
                self.pretty.write(
                    "E",
                    "HfTranslator",
                    f"Failed to load {model_name}: {exc!r}",
                    color=RED,
                )
                self._model = None
                self._tokenizer = None
                return False
            finally:
                if hf_logging is not None and prev_verbosity is not None:
                    try:
                        hf_logging.set_verbosity(prev_verbosity)
                    except Exception:
                        pass

            return True

    # ------------------------------------------------------------- translate
    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
    ) -> str:
        """Translate *text* into *target_lang*.

        Both *target_lang* and *source_lang* may be NLTK language names
        (``"german"``) or ISO 639-1 codes (``"de"``). M2M-100 selects the
        source via ``tokenizer.src_lang``; when *source_lang* is missing or
        ``"auto"`` we default to ``"en"`` (callers in RAG-LCC always know
        the detected source language and SHOULD pass it).
        """
        if not text or not target_lang:
            return text

        tgt_iso = self._to_iso(target_lang)
        src_iso = self._to_iso(source_lang) if source_lang else "auto"
        if not tgt_iso:
            return text

        # No-op translations (e.g. en→en) just round-trip.
        if src_iso and src_iso != "auto" and src_iso == tgt_iso:
            return text

        cache_key = (text, tgt_iso, src_iso)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not self._ensure_loaded():
            # Loader emitted its own diagnostic — return unchanged so the
            # caller's downstream behaviour is identical to the Argos
            # "no pair installed" path.
            return text

        try:
            import torch  # type: ignore[import-not-found]

            # M2M-100 selects the source language via ``tokenizer.src_lang``
            # and the target via ``forced_bos_token_id``. When the caller
            # didn't provide a source we fall back to ``"en"``; the model
            # tolerates a wrong hint reasonably well for short text but a
            # correct one improves quality, so callers SHOULD pass it.
            src_for_tok = src_iso if src_iso and src_iso != "auto" else "en"
            self._tokenizer.src_lang = src_for_tok
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            input_len = int(inputs["input_ids"].shape[1])
            cap = max(32, min(256, input_len * 3 + 16))
            forced_bos = self._tokenizer.get_lang_id(tgt_iso)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=cap,
                    num_beams=1,
                    do_sample=False,
                )
            translated = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            translated = (translated or "").strip()
            if not translated:
                if tgt_iso not in self._warned_targets:
                    self._warned_targets.add(tgt_iso)
                    self.pretty.write(
                        "W",
                        "HfTranslator",
                        f"Empty translation for target '{tgt_iso}' \u2014 "
                        f"returning original text.",
                        color=ORANGE,
                    )
                translated = text
        except Exception as exc:
            self.pretty.write(
                "W",
                "HfTranslator",
                f"Translation failed ({exc!r}) \u2014 returning original text.",
                color=ORANGE,
            )
            translated = text

        self._cache[cache_key] = translated
        return translated
