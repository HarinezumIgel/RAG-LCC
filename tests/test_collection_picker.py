from unittest.mock import MagicMock, patch

from Gui.CollectionPicker import CollectionPicker
from Gui.Picker import RETURN_LABEL


def test_picker_labels_embedder_without_changing_collection_value():
    picker = object.__new__(CollectionPicker)
    config = MagicMock()
    config.get_str.return_value = "bge_m3"
    picker.cfg = config
    collections = ["Docs_embedder_bge_m3", "Docs_snowflake_embedder_bge_m3"]
    with (
        patch.object(CollectionPicker, "_list_collections", return_value=collections),
        patch("Gui.CollectionPicker.inquirer") as prompt,
    ):
        prompt.select.return_value.execute.return_value = collections[1]
        assert picker.pick_collection() == collections[1]
    choices = prompt.select.call_args.kwargs["choices"]
    assert choices[:-1] == [
        {"name": f"{collection} (embedder: bge_m3)", "value": collection}
        for collection in collections
    ]
    assert choices[-1].name == RETURN_LABEL
    assert choices[-1].value is None


def test_return_keeps_collection_unchanged():
    picker = object.__new__(CollectionPicker)
    picker.cfg = MagicMock()
    picker.cfg.get_str.return_value = "bge_m3"
    with (
        patch.object(CollectionPicker, "_list_collections", return_value=["Docs"]),
        patch("Gui.CollectionPicker.inquirer") as prompt,
    ):
        prompt.select.return_value.execute.return_value = None
        assert picker.pick_collection() is None
