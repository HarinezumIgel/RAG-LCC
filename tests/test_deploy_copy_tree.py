from pathlib import Path

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
