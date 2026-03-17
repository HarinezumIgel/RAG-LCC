import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

from huggingface_hub import \
    snapshot_download  # type: ignore[reportUnknownVariableType]

from Commons.Exceptions import (HFDownloaderError, HfHubHTTPError,
                                InternetConnectionDisabledError,
                                ModelLoadError, UserNoDownLoadAccept)
from Compliance.SharedHelpers import SharedHelpers
from Config.Config import Config
from Globals.Globals import Globals
from Gui.Colors import BRIGHT_BLUE, ORANGE, RED, RESET, YELLOW
from Gui.PrettyWriter import PrettyWriter
from Helpers.Helpers import Helpers

# Try to import the HF-specific HTTP exception; fall back gracefully if not present


class HFDownloader:
    def __init__(self) -> None:
        self.cfg: Config = Config()
        self.globalsInstance: Globals = Globals()
        self.helpers: Helpers = Helpers()
        # NOTE: AIHelpers is intentionally NOT instantiated here.
        # AIHelpers.__init__ eagerly loads the embedder model via get_hf_embeddings(),
        # which would bypass the consent-based download flow managed by this class.
        # Model args are read directly from Config via _get_model_args().
        self.sharedHelpers: SharedHelpers = SharedHelpers()
        self.logger: Any = self.helpers.setup_logger("Compliance")
        self.pretty: PrettyWriter = PrettyWriter()

        self.base_dir: Path = Path("ModelGovernance/consents")
        self.base_dir.mkdir(exist_ok=True)
        self._identity_cache: dict[str, Any] | None = None
        self.hf_hub_offline: str | None = os.environ.get("HF_HUB_OFFLINE", "1")

    # -------------------------
    # Deterministic hash helpers
    # -------------------------
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _compute_cfg_hash(self, cfg_obj: Dict[str, Any]) -> str:
        return self._hash_text(json.dumps(cfg_obj, sort_keys=True))

    # -------------------------
    # Metadata IO
    # -------------------------
    def _write_meta(self, path: Path, meta: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _load_meta(self, path: Path) -> Dict[str, Any]:
        if not path.is_file():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # HF cache detection helper
    # -------------------------
    # Filenames that indicate a valid HF model snapshot (at least one must exist)
    _MODEL_MARKER_FILES = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }

    @staticmethod
    def _snapshot_has_content(snapshot_dir: Path) -> bool:
        """Return True if snapshot_dir contains at least one recognised model file.

        An empty directory, a directory with only sub-dirs, or a directory
        that happens to have unrelated files (e.g. the hub root) is
        treated as an incomplete / invalid cache entry.
        """
        try:
            if not snapshot_dir.is_dir():
                return False
            for f in snapshot_dir.rglob("*"):
                if f.is_file() and f.name in HFDownloader._MODEL_MARKER_FILES:
                    return True
            return False
        except Exception:
            return False

    def _find_cached_snapshot(
        self, hf_hub_cache: Optional[str], model_id: str, revision: Any
    ) -> tuple[Optional[Path], str]:
        """
        Inspect the HF hub cache directory for a snapshot matching model_id and revision.
        Returns (Path to snapshot directory, resolved_revision_hash) or (None, "").
        Only returns a snapshot that actually contains files (not an empty stub).
        """
        if not hf_hub_cache:
            return None, ""

        try:
            cache_root = Path(str(hf_hub_cache)).expanduser().resolve()
        except Exception:
            return None, ""

        if not cache_root.exists():
            return None, ""

        # Common HF cache layout: models--owner--repo/snapshots/<revision>
        sanitized = model_id.replace("/", "--")
        model_dir = cache_root / f"models--{sanitized}"

        # When revision is specified, look for that exact snapshot
        if revision:
            candidate = model_dir / "snapshots" / str(revision)
            if candidate.exists() and self._snapshot_has_content(candidate):
                return candidate, str(revision)

        # When revision is empty/None, or exact match failed, pick any snapshot
        snap_dir = model_dir / "snapshots"
        if snap_dir.is_dir():
            snapshots = sorted(
                [
                    d
                    for d in snap_dir.iterdir()
                    if d.is_dir() and self._snapshot_has_content(d)
                ],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                chosen = snapshots[0]
                if not revision:
                    self.pretty.write(
                        "I",
                        "HF Cache",
                        f"No revision pinned for '{model_id}'; "
                        f"using latest cached snapshot: {chosen.name}"
                        + (
                            f" ({len(snapshots)} snapshots available)"
                            if len(snapshots) > 1
                            else ""
                        ),
                    )
                return chosen, chosen.name

        # Looser search: glob for variants (handles different HF versions/layouts)
        try:
            for p in cache_root.glob(f"models--{sanitized}*"):
                snap = p / "snapshots"
                if snap.is_dir():
                    children = [
                        d
                        for d in snap.iterdir()
                        if d.is_dir() and self._snapshot_has_content(d)
                    ]
                    if revision:
                        exact = snap / str(revision)
                        if exact.exists() and self._snapshot_has_content(exact):
                            return exact, str(revision)
                    if children:
                        return children[0], children[0].name
        except Exception:
            # Non-fatal: return None if inspection fails
            self.logger.debug(
                f"HF cache inspection failed for {hf_hub_cache}", exc_info=True
            )
            return None, ""

        return None, ""

    # -------------------------
    # Main download function
    # -------------------------
    def download(self, key: str) -> Dict[str, Any]:
        # Extract model type from key (e.g. "_MODELS._CROSS" -> "_CROSS")
        model_type = key.rsplit(".", 1)[-1] if "." in key else key
        model_args = self.helpers.get_model_args(model_type)
        model_name = model_args["model_name"]
        revision = model_args.get("revision")
        friendly = model_args.get("friendly_name")
        source = model_args.get("source")

        # Resolve the impl to build the full config path
        impl: str = self.cfg.get_str(model_type)
        resolved_key = f"_MODELS.{impl}.{model_type}"

        # Directory for this model
        model_dir = self.base_dir / resolved_key
        model_dir.mkdir(parents=True, exist_ok=True)

        meta_path = model_dir / "download_meta.json"

        # Compute config hash for comparison
        model_cfg = self.cfg.get(resolved_key)
        if model_cfg is None:
            raise ValueError(f"No configuration found for key: {resolved_key}")

        cfg_hash = self._compute_cfg_hash(cast(Dict[str, Any], model_cfg))

        # Load existing metadata if present
        existing_meta = self._load_meta(meta_path)
        local_path = existing_meta.get("local_path")

        # If metadata indicates a local copy that exists and matches revision/config -> skip
        if existing_meta:
            recorded_revision = existing_meta.get("revision")
            recorded_cfg_hash = existing_meta.get("config_hash")
            # When config REVISION is empty, any recorded revision is acceptable
            # (it was resolved from the cache on a prior run).
            revision_ok = (
                not revision  # empty / None → accept whatever was recorded
                or recorded_revision == revision
            )
            local_path_obj = Path(local_path) if local_path else None
            if (
                revision_ok
                and recorded_cfg_hash == cfg_hash
                and local_path_obj is not None
                and local_path_obj.exists()
                and self._snapshot_has_content(local_path_obj)
            ):
                info = f"{key}: model already downloaded with matching revision and config hash. Skipping download."
                self.logger.info(info)
                self.pretty.write("O", "HF Download", info)
                # Propagate the previously-resolved revision to runtime config
                if recorded_revision and recorded_revision != revision:
                    self.cfg.set(f"{key}.REVISION", recorded_revision, force=True)
                return existing_meta

        # Determine HF hub cache path from config (allow both exact key and trimmed key)
        hf_hub_cache: str | None = cast(
            Optional[str],
            self.cfg.get("_HF_HUB_CACHE") or self.cfg.get("_HF_HUB_CACHE ", None),
        )
        if not hf_hub_cache:
            info = f"{key}: No HF Hub cache configured (no _HF_HUB_CACHE setting in Configuration/Config_Global.py)"
            self.logger.error(f"{info}")
            self.pretty.write("E", "HF Download", f"{info}", color=RED)
            raise HFDownloaderError(info)
        # If HF cache configured, try to detect an existing snapshot there first
        cached_snapshot, resolved_rev = self._find_cached_snapshot(
            hf_hub_cache, model_name, revision
        )
        if cached_snapshot:
            # Use the resolved snapshot hash as the effective revision
            effective_revision: str = resolved_rev or revision or ""
            info = f"{key}: model found in HF cache at: {cached_snapshot} (revision: {effective_revision})"
            self.logger.info(info)
            self.pretty.write("O", "HF Cache", info)

            # If consent metadata was deleted, require re-consent before
            # silently re-creating it from the cached snapshot.
            if not existing_meta or not existing_meta.get("accepted_by"):
                print(f"{BRIGHT_BLUE}\n{'=' * 70}")
                print(f"  Model Consent Required  (cached model)")
                print(f"  Model: {friendly}  ({model_name})")
                print(f"  Revision: {effective_revision}")
                print(f"{'=' * 70}{RESET}\n")
                print(
                    f"{ORANGE}>>>> Consent metadata missing. Accept this cached model?{RESET}"
                )
                ans = input("Accept? [y/N] ").strip().lower()
                if ans != "y":
                    msg = f"User declined re-consent for cached model {key}"
                    self.pretty.write("E", "HF Download", msg, color=RED)
                    self.logger.info(msg)
                    raise UserNoDownLoadAccept(msg)
                identity = self.sharedHelpers.capture_acceptance_identity_once()
            else:
                identity: dict[str, Any] = {
                    "accepted_by": existing_meta.get("accepted_by"),
                    "accepted_by_source": existing_meta.get("accepted_by_source"),
                    "accepted_by_verified": existing_meta.get(
                        "accepted_by_verified", False
                    ),
                    "host": existing_meta.get("host"),
                    "pid": existing_meta.get("pid"),
                }

            # Persist resolved revision into runtime config so downstream
            # cache keys (get_hf_embeddings, load_quantized_model) are
            # deterministic and match the actual snapshot hash.
            if effective_revision and effective_revision != revision:
                self.cfg.set(f"{key}.REVISION", effective_revision, force=True)
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            meta: dict[str, Any] = {
                "model_id": model_name,
                "friendly_name": friendly,
                "revision": effective_revision,
                "source": source,
                "downloaded_at": now,
                "config_hash": cfg_hash,
                "accepted_by": identity.get("accepted_by"),
                "accepted_by_source": identity.get("accepted_by_source"),
                "accepted_by_verified": identity.get("accepted_by_verified", False),
                "host": identity.get("host"),
                "pid": identity.get("pid"),
                "local_path": str(cached_snapshot),
                "config": model_cfg,
            }
            # Persist metadata to model_dir for auditability (overwrite or create)
            self._write_meta(meta_path, meta)
            return meta

        # If we reach here, either no cache was configured or model not present in cache.
        # If existing meta differs, handle mismatch first
        if existing_meta:
            recorded_revision = existing_meta.get("revision")
            recorded_cfg_hash = existing_meta.get("config_hash")
            info1 = f"{key}: existing download metadata differs.\n"
            self.pretty.write("W", "HF Download", info1)
            info2 = f"Recorded revision: {recorded_revision} / Requested revision: {revision}\n"
            self.pretty.write("W", "HF Download", info2)
            info3 = f"Recorded config_hash: {recorded_cfg_hash} / Requested config_hash: {cfg_hash}"
            self.pretty.write("W", "HF Download", info3)
            mismatch_info = info1 + info2 + info3
            self.logger.info(mismatch_info)
            self._check_no_internet_allowed(model_name, revision, hf_hub_cache)

            prompt = (
                f"{YELLOW}\n\n>>>> Existing model metadata differs for [{friendly}].\n"
                f">>>> Re-download and replace local copy? [y/N] {RESET}"
            )
            ans = input(prompt).strip().lower()
            if ans != "y":
                msg = f"User declined re-download for {key}"
                self.pretty.write("E", "HF Download", msg, color=RED)
                raise UserNoDownLoadAccept(msg)
            # else proceed to download and overwrite

        # -------------------------
        # Internet access check (if no existing meta or user agreed to re-download)
        # -------------------------
        self._check_no_internet_allowed(model_name, revision, hf_hub_cache)

        # -------------------------
        # Prompt user and get consent
        # -------------------------
        print(f"{BRIGHT_BLUE}\n\n>>>> Missing model [{friendly}]\n{RESET}")

        print(
            f">>>> MODEL: {model_name}\n"
            f">>>> REVISION: {revision}\n"
            f">>>> SOURCE: {source}\n"
        )
        print(
            f"{ORANGE}\n\n>>>> Do you accept downloading model [{friendly}]?\n{RESET}"
        )
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != "y":
            msg = f"User declined download for model {key}"
            self.pretty.write("E", "HF Download", msg, color=RED)
            self.logger.info(msg)
            raise UserNoDownLoadAccept(msg)

        identity = self.sharedHelpers.capture_acceptance_identity_once()
        msg = f"Downloading model: {model_name} to: {hf_hub_cache}"
        self.pretty.write("I", "HF Download", msg)
        self.logger.info(
            f"User accepted download for model {model_name} to {hf_hub_cache}"
        )
        # -------------------------
        # Perform deterministic HF download (prefer cache_dir if configured)
        # -------------------------
        downloaded_path: str = ""
        try:
            if hf_hub_cache:

                Path(hf_hub_cache).mkdir(parents=True, exist_ok=True)
                downloaded_path = snapshot_download(  # type: ignore[reportCallIssue]
                    repo_id=model_name,
                    revision=revision or None,
                    cache_dir=hf_hub_cache,
                )
                self.pretty.write("N", "", "")
        except HfHubHTTPError as e:
            msg = f"Failed to download model:{model_name} user: {identity.get('accepted_by')}"
            self.pretty.write("E", "HF Download", msg, color=RED)
            self.logger.error(msg)
            raise ModelLoadError(msg) from e
        except Exception as e:
            msg = f"Failed to download model: {model_name} user: {identity.get('accepted_by')} error: {e}"
            self.pretty.write("E", "HF Download", msg, color=RED)
            self.logger.error(msg)
            raise

        # -------------------------
        # Resolve effective revision from downloaded snapshot path
        # -------------------------
        dl_effective_rev = revision or ""
        if downloaded_path:
            dl_path = Path(downloaded_path)
            # snapshot_download returns .../snapshots/<hash> — extract the hash
            if dl_path.parent.name == "snapshots":
                dl_effective_rev = dl_path.name
            elif dl_path.name == "snapshots":
                # Shouldn't happen, but guard
                pass
            # Write resolved revision back to runtime config
            if dl_effective_rev and dl_effective_rev != revision:
                self.cfg.set(f"{key}.REVISION", dl_effective_rev, force=True)

        # -------------------------
        # Build and persist metadata
        # -------------------------
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        meta: dict[str, Any] = {
            "model_id": model_name,
            "friendly_name": friendly,
            "revision": dl_effective_rev,
            "source": source,
            "downloaded_at": now,
            "config_hash": cfg_hash,
            "accepted_by": identity["accepted_by"],
            "accepted_by_source": identity["accepted_by_source"],
            "accepted_by_verified": identity["accepted_by_verified"],
            "host": identity["host"],
            "pid": identity["pid"],
            "local_path": str(downloaded_path),  # type: ignore[reportUnknownArgumentType]
            "config": model_cfg,
        }

        self._write_meta(meta_path, meta)
        msg = f"Model: {key} downloaded to {downloaded_path}"
        self.pretty.write("O", "HF Download", msg)
        self.logger.info(f"Model: {key} downloaded → {downloaded_path}")

        return meta

    def _check_no_internet_allowed(
        self,
        model_name: str = "",
        revision: str | None = None,
        cache_path: str = "",
    ) -> None:
        # If internet disabled, instruct user to change config and abort
        if self.hf_hub_offline == "1":
            parts: list[str] = []
            if model_name:
                parts.append(
                    f"Model '{model_name}' (revision '{revision}') was not found in the local cache."
                )
            if cache_path:
                parts.append(f"Searched cache: {cache_path}")
            parts.append(
                'Internet access is disabled (HF_HUB_OFFLINE="1"). '
                "Enable internet or place the model in the cache directory."
            )
            msg = " ".join(parts)
            self.logger.error(msg)
            raise InternetConnectionDisabledError(msg)
