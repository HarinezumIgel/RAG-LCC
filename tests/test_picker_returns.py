from pathlib import Path
from unittest.mock import MagicMock

import pytest
from InquirerPy import inquirer  # type: ignore[attr-defined]
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from Gui.FileList import FileEntry, FileList
from Gui.MetadataPicker import MetadataPicker
from Gui.Picker import PickerCancelled, RETURN_LABEL, pick_value


@pytest.mark.parametrize("value", [False, 0, True, 30, "path"])
def test_picker_keeps_falsy_values_selectable(value: object) -> None:
    prompt = MagicMock()
    prompt.select.return_value.execute.return_value = value
    assert pick_value(prompt, choices=[value], message="Choose") == value
    choice = prompt.select.call_args.kwargs["choices"][-1]
    assert choice.name == RETURN_LABEL
    assert choice.value is None


def test_picker_return_is_distinct_from_false_and_zero() -> None:
    prompt = MagicMock()
    prompt.select.return_value.execute.return_value = None
    with pytest.raises(PickerCancelled):
        pick_value(prompt, choices=[False, 0], message="Choose")


def test_real_picker_return_with_keyboard_input() -> None:
    with create_pipe_input() as keyboard:
        with create_app_session(input=keyboard, output=DummyOutput()):
            keyboard.send_text("\x1b[B\r")
            with pytest.raises(PickerCancelled):
                pick_value(inquirer, choices=[True], message="Choose", default=True)


def test_file_history_return_preserves_current_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import Gui.FileList as module

    files = object.__new__(FileList)
    current = FileEntry("old", "old.pdf", r"D:\old.pdf", "File")
    files.entries = [current, FileEntry("new", "new.pdf", r"D:\new.pdf", "File")]
    files.cur = current
    files.current_store = "old"
    files.usage = "File"
    prompt = MagicMock()
    prompt.select.return_value.execute.return_value = None
    monkeypatch.setattr(module, "inquirer", prompt)

    assert files.select("new", "File") is None

    assert files.cur is current
    assert files.current_store == "old"
    assert files.usage == "File"
    choice = prompt.select.call_args.kwargs["choices"][-1]
    assert choice.name == RETURN_LABEL


@pytest.mark.parametrize("return_stage", ["fields", "first_value", "second_value"])
def test_metadata_return_preserves_all_existing_filters(
    monkeypatch: pytest.MonkeyPatch,
    return_stage: str,
) -> None:
    import Gui.MetadataPicker as module

    picker = object.__new__(MetadataPicker)
    monkeypatch.setattr(
        MetadataPicker,
        "_collect_fields",
        lambda self, collection: {"Author": {"A", "B"}, "Language": {"en", "de"}},
    )
    prompt = MagicMock()
    prompt.checkbox.return_value.execute.return_value = (
        ["Author", None] if return_stage == "fields" else ["Author", "Language"]
    )
    prompt.select.return_value.execute.side_effect = (
        ["B", None] if return_stage == "second_value" else [None]
    )
    monkeypatch.setattr(module, "inquirer", prompt)
    current = {"Author": "A", "Language": "en"}

    assert picker.pick("docs", current) == current
    assert current == {"Author": "A", "Language": "en"}
    assert prompt.checkbox.call_args.kwargs["choices"][-1].name == RETURN_LABEL


@pytest.mark.parametrize("return_stage", ["action", "files", "directory"])
def test_browser_return_does_not_commit_pending_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    return_stage: str,
) -> None:
    import Gui.FileList as module

    (tmp_path / "source.txt").write_text("source", encoding="ascii")
    (tmp_path / "child").mkdir()
    files = object.__new__(FileList)
    files.entries = []
    files.cur = None
    files.current_store = "old"
    files.usage = "File"
    prompt = MagicMock()
    prompt.select.return_value.execute.side_effect = {
        "action": ["select", None],
        "files": ["select"],
        "directory": ["select", "down", None],
    }[return_stage]
    prompt.checkbox.return_value.execute.return_value = (
        ["source.txt", None] if return_stage == "files" else ["source.txt"]
    )
    monkeypatch.setattr(module, "inquirer", prompt)

    assert files.select("new", "DocSelect", str(tmp_path)) == []

    assert files.entries == []
    assert files.current_store == "old"
    assert files.usage == "File"


def test_browser_finish_commits_selected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import Gui.FileList as module

    (tmp_path / "source.txt").write_text("source", encoding="ascii")
    files = object.__new__(FileList)
    files.entries = []
    files.cur = None
    files.current_store = "old"
    files.usage = "File"
    prompt = MagicMock()
    prompt.select.return_value.execute.side_effect = ["select", "done"]
    prompt.checkbox.return_value.execute.return_value = ["source.txt"]
    monkeypatch.setattr(module, "inquirer", prompt)

    selected = files.select("new", "DocSelect", str(tmp_path))

    assert selected == [
        FileEntry("new", "source.txt", str(tmp_path / "source.txt"), "DocSelect")
    ]
    assert files.entries == selected
    assert files.current_store == "new"
    assert files.usage == "DocSelect"
    actions = prompt.select.call_args_list[0].kwargs["choices"]
    assert actions[0].value == "select"
    assert actions[0].name == "Select files here"
