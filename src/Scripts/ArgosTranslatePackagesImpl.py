"""Shared Argos language-package operations used by compliance and CLI wrappers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Sequence

from Compliance.ArgosSpacySentencizerPatch import \
    prepare_argos_runtime_for_patch
from Gui.Colors import ORANGE, RESET

prepare_argos_runtime_for_patch()

import argostranslate.package  # type: ignore[reportMissingTypeStubs]
import argostranslate.translate  # type: ignore[reportMissingTypeStubs]

_LICENSE_HINT_PATTERNS: list[tuple[str, str]] = [
    ("MIT", r"\bmit\b"),
    ("Apache-2.0", r"apache(?:\s+license)?[\s-]*2(?:\.0)?"),
    ("CC-BY 4.0", r"cc[\s-]*by[\s-]*4\.0"),
    ("CC-BY-SA 3.0", r"cc[\s-]*by[\s-]*sa[\s-]*3\.0"),
]
_STANZA_LICENSE_URL = "https://github.com/stanfordnlp/stanza/blob/master/LICENSE"


def print_install_scope_note() -> None:
    """Explain what Argos package install does and does not include."""
    print("This command installs Argos language packages only.")
    print("Argos Translate itself is MIT licensed.")
    print(
        f"{ORANGE}Argos Translate itself is MIT licensed."
        f"Installed language packages may contain additional model, "
        "tokenizer, vocabulary, or data artifacts under their own licenses."
        f"{RESET}"
    )
    print(
        f"{ORANGE}This command does not download or use the spaCy model "
        f"xx_sent_ud_sm.{RESET}"
    )


def installed_packages() -> list[Any]:
    """Return installed Argos package objects."""
    try:
        return list(argostranslate.package.get_installed_packages())
    except Exception:
        return []


def installed_language_pairs() -> set[tuple[str, str]]:
    """Return installed language pairs as (from_code, to_code)."""
    result: set[tuple[str, str]] = set()
    for pkg in installed_packages():
        result.add(
            (
                str(getattr(pkg, "from_code", "")),
                str(getattr(pkg, "to_code", "")),
            )
        )
    return result


def missing_language_pairs(
    languages: Sequence[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Return configured language pairs that are not installed yet."""
    configured_pairs = {(str(a), str(b)) for a, b in languages}
    return configured_pairs - installed_language_pairs()


def install_language_pairs(
    languages: Sequence[tuple[str, str]],
    *,
    on_missing_pair: Callable[[str, str], None] | None = None,
) -> None:
    """Install configured language pairs from the Argos package index."""
    argostranslate.package.update_package_index()
    available = list(argostranslate.package.get_available_packages())

    for from_code, to_code in languages:
        try:
            pkg = next(
                p
                for p in available
                if p.from_code == from_code and p.to_code == to_code
            )
            print(f"  Installing {from_code} -> {to_code} ...")
            argostranslate.package.install_from_path(pkg.download())
        except StopIteration:
            if on_missing_pair is None:
                print(f"No package found for {from_code} -> {to_code}")
            else:
                on_missing_pair(from_code, to_code)


def remove_packages(
    packages: Sequence[Any],
    *,
    on_uninstall_error: Callable[[str, str, Exception], None] | None = None,
) -> int:
    """Uninstall provided Argos package objects and return removed count."""
    removed = 0
    for pkg in packages:
        from_code = str(getattr(pkg, "from_code", ""))
        to_code = str(getattr(pkg, "to_code", ""))
        try:
            argostranslate.package.uninstall(pkg)
            removed += 1
        except Exception as exc:
            if on_uninstall_error is not None:
                on_uninstall_error(from_code, to_code, exc)
            continue
    return removed


def show_status(*, tokenizer_resources_hint: str | None = None) -> None:
    """Print installed Argos languages and packages."""
    try:
        langs = list(argostranslate.translate.get_installed_languages())
    except Exception:
        langs = []
    print(f"\nInstalled languages ({len(langs)}):")
    for lang in langs:
        print(f"  {lang.name} ({lang.code})")

    installed = installed_packages()
    print(f"\nInstalled packages ({len(installed)}):")
    for pkg in installed:
        print(f"  {pkg.from_code} -> {pkg.to_code}")

    if tokenizer_resources_hint:
        print(f"\n{tokenizer_resources_hint}")


def report_xx_sent_ud_sm_state() -> None:
    """Show whether xx_sent_ud_sm artifacts exist in common Argos paths."""
    hits = _find_xx_sent_ud_sm_artifacts()
    print()
    if hits:
        print(f"Detected {len(hits)} xx_sent_ud_sm artifact(s):")
        for path in hits:
            print(f"  {path}")
    else:
        print("No xx_sent_ud_sm artifacts detected in Argos cache/package paths.")


def report_argos_package_license_hints() -> None:
    """Print per-package license hints from local Argos package files."""
    installed = installed_packages()
    if not installed:
        return

    print("\nArgos package license hints (local package files):")
    for pkg in sorted(
        installed,
        key=lambda p: (
            str(getattr(p, "from_code", "")),
            str(getattr(p, "to_code", "")),
        ),
    ):
        from_code = str(getattr(pkg, "from_code", "")).strip()
        to_code = str(getattr(pkg, "to_code", "")).strip()
        package_path = str(
            getattr(pkg, "package_path", "") or getattr(pkg, "path", "")
        ).strip()

        hints: list[str]
        if package_path:
            hints = _argos_package_license_hints(Path(package_path))
        else:
            hints = ["Unknown (license file(s) may be in subdirectories)"]

        print(f"  {from_code} -> {to_code}: {', '.join(hints)}")


def _find_xx_sent_ud_sm_artifacts() -> list[Path]:
    """Return filesystem paths indicating xx_sent_ud_sm artifacts."""
    roots = [
        Path.home() / ".local" / "cache" / "argos-translate",
        Path.home() / ".local" / "share" / "argos-translate" / "packages",
        Path.home() / ".local" / "share" / "resources",
    ]
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*xx_sent_ud_sm*"):
                hits.append(path)
        except Exception:
            continue
    return hits


def _license_hints_from_text(text: str) -> list[str]:
    """Return recognized license labels from raw text snippets."""
    labels: list[str] = []
    for label, pattern in _LICENSE_HINT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            labels.append(label)
    return list(dict.fromkeys(labels))


def _argos_package_license_hints(package_dir: Path) -> list[str]:
    """Extract local license hints from one Argos package directory."""
    text_chunks: list[str] = []
    for glob_pattern in ("LICENSE*", "COPYING*", "NOTICE*"):
        for candidate in package_dir.glob(glob_pattern):
            if not candidate.is_file():
                continue
            try:
                text_chunks.append(
                    candidate.read_text(encoding="utf-8", errors="ignore")
                )
            except Exception:
                continue

    readme_path = package_dir / "README.md"
    if readme_path.is_file():
        try:
            text_chunks.append(readme_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    all_text = "\n".join(text_chunks)
    hints = _license_hints_from_text(all_text)

    lowered = all_text.lower()
    if "stanza" in lowered and "license" in lowered:
        hints.append(f"Stanza license: {_STANZA_LICENSE_URL}")

    if not hints:
        hints.append("Unknown (license file(s) may be in subdirectories)")
    return list(dict.fromkeys(hints))
