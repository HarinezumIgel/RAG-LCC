from typing import Any

from InquirerPy.base.control import Choice  # type: ignore[attr-defined]

RETURN_LABEL = "Return - keep current setting"


class PickerCancelled(Exception):
    """The user returned from a picker without applying a change."""


def pick_value(inquirer: Any, *, choices: list[Any], **kwargs: Any) -> Any:
    value = inquirer.select(
        choices=[*choices, Choice(None, name=RETURN_LABEL)], **kwargs
    ).execute()
    if value is None:
        raise PickerCancelled
    return value
