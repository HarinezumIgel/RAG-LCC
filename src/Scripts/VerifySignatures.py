#!/usr/bin/env python3
"""
VerifySignatures.py — Verify RSA signatures for shipped files.

Verifies that files have not been tampered with by checking RSA signatures
against the public key in verify_sign/RAG_LCC_public.pem.

Signature sources (in order of preference):
  1. signed.txt manifest (if present and its signature is valid)
  2. Individual .sig files alongside each file

Usage:
    python src/Scripts/VerifySignatures.py
    python src/Scripts/VerifySignatures.py --input-dir /path/to/project
    python src/Scripts/VerifySignatures.py --include-dirs requirements src/Scripts
"""

from __future__ import annotations

import argparse
import base64
# Refuse to run from a drive/filesystem root before doing anything
import os
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Commons.DriveRootGuard import assert_not_drive_root  # noqa: E402

assert_not_drive_root(__file__)

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
except ImportError:
    print("Error: cryptography package not found.")
    print("Install it with: pip install cryptography")
    sys.exit(1)


class VerificationResult(NamedTuple):
    relative_path: str
    status: str  # "OK", "INVALID", "MISSING_FILE", "MISSING_SIG", "ERROR"
    message: str


def _load_public_key(key_path: Path) -> rsa.RSAPublicKey:
    """Load RSA public key from PEM file."""
    try:
        with key_path.open("rb") as f:
            pem_data = f.read()
        public_key = serialization.load_pem_public_key(pem_data)
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError("Key is not an RSA public key")
        return public_key
    except Exception as e:
        print(f"Error: Failed to load public key from {key_path}: {e}")
        sys.exit(1)


def _verify_signature(
    public_key: rsa.RSAPublicKey, data: bytes, signature_b64: str
) -> bool:
    """Verify RSA-SHA256 signature."""
    try:
        signature_bytes = base64.b64decode(signature_b64.strip())
        public_key.verify(
            signature_bytes,
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def _load_manifest(
    manifest_path: Path, manifest_sig_path: Path, public_key: rsa.RSAPublicKey
) -> list[tuple[str, str]] | None:
    """
    Load and verify the manifest file.

    Returns:
        List of (relative_path, signature_b64) tuples if manifest is valid,
        None otherwise.
    """
    if not manifest_path.exists():
        return None

    if not manifest_sig_path.exists():
        print(f"Warning: Manifest signature file not found: {manifest_sig_path}")
        print("         Manifest is not trusted; falling back to .sig files.")
        return None

    try:
        # Verify manifest signature
        manifest_data = manifest_path.read_bytes()
        manifest_sig_b64 = manifest_sig_path.read_text(encoding="utf-8")

        if not _verify_signature(public_key, manifest_data, manifest_sig_b64):
            print(f"Warning: Manifest signature INVALID: {manifest_path}")
            print("         Will not trust manifest; falling back to .sig files.")
            return None

        print("Manifest signature valid. Using manifest entries.")

        # Parse manifest
        entries = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed manifest line: {line}")
                continue
            rel_path, sig_b64 = parts
            entries.append((rel_path, sig_b64))

        return entries

    except Exception as e:
        print(f"Error: Failed to load manifest: {e}")
        print("       Falling back to .sig files.")
        return None


def _find_sig_files(
    input_dir: Path,
    include_dirs: list[Path] | None,
    manifest_sig_path: Path,
    exclude_dirs: list[Path] | None = None,
) -> list[Path]:
    """Find all .sig files (excluding manifest signature and excluded directories)."""
    sig_files = []

    if include_dirs:
        for dir_path in include_dirs:
            if dir_path.exists() and dir_path.is_dir():
                sig_files.extend(dir_path.rglob("*.sig"))
            elif dir_path.exists() and dir_path.is_file() and dir_path.suffix == ".sig":
                sig_files.append(dir_path)
    else:
        sig_files = list(input_dir.rglob("*.sig"))

    # Exclude manifest signature
    sig_files = [f for f in sig_files if f != manifest_sig_path]

    # Exclude files in excluded directories
    if exclude_dirs:
        sig_files = [
            f
            for f in sig_files
            if not any(f.is_relative_to(excluded) for excluded in exclude_dirs)
        ]

    return sig_files


def _verify_from_manifest(
    manifest_entries: list[tuple[str, str]],
    input_dir: Path,
    public_key: rsa.RSAPublicKey,
    include_dirs: list[Path] | None,
    exclude_dirs: list[Path] | None = None,
) -> list[VerificationResult]:
    """Verify files using manifest entries."""
    results = []

    for rel_path, sig_b64 in manifest_entries:
        file_path = input_dir / rel_path

        # Check if file is in excluded directories
        if exclude_dirs and any(
            file_path.is_relative_to(excluded) for excluded in exclude_dirs
        ):
            continue

        # Check if file is in included directories
        if include_dirs:
            in_scope = any(
                file_path.is_relative_to(d) or file_path == d for d in include_dirs
            )
            if not in_scope:
                continue

        if not file_path.exists():
            results.append(
                VerificationResult(
                    rel_path, "MISSING_FILE", f"Referenced file not found"
                )
            )
            continue

        try:
            file_data = file_path.read_bytes()
            if _verify_signature(public_key, file_data, sig_b64):
                results.append(VerificationResult(rel_path, "OK", "Valid"))
            else:
                results.append(
                    VerificationResult(rel_path, "INVALID", "Signature mismatch")
                )
        except Exception as e:
            results.append(
                VerificationResult(rel_path, "ERROR", f"Verification error: {e}")
            )

    return results


def _verify_from_sig_files(
    sig_files: list[Path], input_dir: Path, public_key: rsa.RSAPublicKey
) -> list[VerificationResult]:
    """Verify files using individual .sig files."""
    results = []

    for sig_file in sig_files:
        # Determine original file path
        orig_file = sig_file.with_suffix("")
        rel_path = str(orig_file.relative_to(input_dir))

        if not orig_file.exists():
            results.append(
                VerificationResult(rel_path, "MISSING_FILE", "Original file not found")
            )
            continue

        try:
            file_data = orig_file.read_bytes()
            sig_b64 = sig_file.read_text(encoding="utf-8")

            if _verify_signature(public_key, file_data, sig_b64):
                results.append(VerificationResult(rel_path, "OK", "Valid"))
            else:
                results.append(
                    VerificationResult(rel_path, "INVALID", "Signature mismatch")
                )
        except Exception as e:
            results.append(
                VerificationResult(rel_path, "ERROR", f"Verification error: {e}")
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify RSA signatures for shipped files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--include-dirs",
        nargs="+",
        type=Path,
        help="Optional list of directories/files to include (relative to input-dir or absolute)",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="+",
        type=Path,
        help="Optional list of directories to exclude (relative to input-dir or absolute)",
    )
    parser.add_argument(
        "--key-dir",
        type=Path,
        default=None,
        help="Directory where keys are stored (default: verify_sign under input-dir)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    key_dir = args.key_dir.resolve() if args.key_dir else input_dir / "verify_sign"
    if not key_dir.exists():
        print(f"Error: Key directory not found: {key_dir}")
        sys.exit(1)

    public_key_path = key_dir / "RAG_LCC_public.pem"
    if not public_key_path.exists():
        print(f"Error: Public key not found: {public_key_path}")
        sys.exit(1)

    # Resolve include directories
    include_dirs = None
    if args.include_dirs:
        include_dirs = []
        for d in args.include_dirs:
            if d.is_absolute():
                resolved = d.resolve()
            else:
                resolved = (input_dir / d).resolve()

            if not resolved.exists():
                print(f"Warning: Include path not found: {d} — skipping")
                continue

            include_dirs.append(resolved)

        if not include_dirs:
            print("Error: No valid include directories found")
            sys.exit(1)

    # Resolve exclude directories
    exclude_dirs = None
    if args.exclude_dirs:
        exclude_dirs = []
        for d in args.exclude_dirs:
            if d.is_absolute():
                resolved = d.resolve()
            else:
                resolved = (input_dir / d).resolve()

            if not resolved.exists():
                print(f"Warning: Exclude path not found: {d} — skipping")
                continue

            exclude_dirs.append(resolved)

    manifest_path = input_dir / "signed.txt"
    manifest_sig_path = input_dir / "signed.txt.sig"

    print(f"Input directory: {input_dir}")
    print(f"Public key: {public_key_path}")
    if include_dirs:
        print(
            f"Include paths: {', '.join(str(d.relative_to(input_dir) if d.is_relative_to(input_dir) else d) for d in include_dirs)}"
        )
    if exclude_dirs:
        print(
            f"Exclude paths: {', '.join(str(d.relative_to(input_dir) if d.is_relative_to(input_dir) else d) for d in exclude_dirs)}"
        )
    print()

    # Load public key
    public_key = _load_public_key(public_key_path)

    # Try to load and verify manifest
    manifest_entries = _load_manifest(manifest_path, manifest_sig_path, public_key)

    results = []

    if manifest_entries:
        results = _verify_from_manifest(
            manifest_entries, input_dir, public_key, include_dirs, exclude_dirs
        )
    else:
        # Fall back to .sig files
        print("Attempting verification using .sig files on disk.")
        sig_files = _find_sig_files(
            input_dir, include_dirs, manifest_sig_path, exclude_dirs
        )

        if not sig_files:
            print("Error: No manifest and no .sig files found to verify.")
            sys.exit(1)

        results = _verify_from_sig_files(sig_files, input_dir, public_key)

    # Print results
    if not results:
        print("No files to verify in the selected scope.")
        sys.exit(0)

    ok_count = 0
    fail_count = 0

    for result in results:
        if result.status == "OK":
            print(f"OK: {result.relative_path}")
            ok_count += 1
        else:
            print(f"{result.status}: {result.relative_path} — {result.message}")
            fail_count += 1

    print()
    if fail_count == 0:
        print(f"All signatures valid. ({ok_count} file(s) verified)")
        sys.exit(0)
    else:
        print(f"Verification failed for {fail_count} item(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
