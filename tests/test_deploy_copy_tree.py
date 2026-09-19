from pathlib import Path

import pytest

pytest.importorskip("deploy")
import deploy.scripts.deploy as deploy


def test_copy_tree_copies_files_without_treating_directories_as_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"

    nested_file = source / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("template: true\n", encoding="utf-8")

    binary_file = source / "Documentation" / "Pics" / "image.png"
    binary_file.parent.mkdir(parents=True)
    binary_file.write_bytes(b"PNGDATA")

    deploy.copy_tree(source, destination)

    assert (destination / ".github" / "ISSUE_TEMPLATE" / "config.yml").exists()
    assert (destination / "Documentation" / "Pics" / "image.png").exists()
    assert (destination / ".github" / "ISSUE_TEMPLATE").is_dir()


def test_copy_tree_excludes_directory_prefixes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dest"

    skipped_file = source / "ModelGovernance" / "licenses" / "license_meta.json"
    skipped_file.parent.mkdir(parents=True)
    skipped_file.write_text("{}\n", encoding="utf-8")

    kept_file = source / "src" / "App.py"
    kept_file.parent.mkdir(parents=True)
    kept_file.write_text("print('ok')\n", encoding="utf-8")

    deploy.copy_tree(source, destination, exclude_rel_paths={"ModelGovernance"})

    assert not (
        destination / "ModelGovernance" / "licenses" / "license_meta.json"
    ).exists()
    assert (destination / "src" / "App.py").exists()


def test_host_mirror_cleanup_and_copy_omit_model_governance(tmp_path: Path) -> None:
    source = tmp_path / "deploy_target"
    destination = tmp_path / "host_target"

    model_governance_file = (
        source / "ModelGovernance" / "consents" / "download_meta.json"
    )
    model_governance_file.parent.mkdir(parents=True)
    model_governance_file.write_text("{}\n", encoding="utf-8")

    stale_host_file = destination / "ModelGovernance" / "stale.txt"
    stale_host_file.parent.mkdir(parents=True)
    stale_host_file.write_text("stale\n", encoding="utf-8")

    readme_file = source / "README.md"
    readme_file.write_text("deploy\n", encoding="utf-8")

    deploy.remove_safe(destination / "ModelGovernance")
    deploy.copy_tree(
        source,
        destination,
        exclude_rel_paths={"ModelGovernance"},
    )

    assert not (destination / "ModelGovernance").exists()
    assert (destination / "README.md").exists()
