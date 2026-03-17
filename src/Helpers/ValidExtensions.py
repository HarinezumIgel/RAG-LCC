# System imports
import os

from Config.Config import Config


class ValidExtensions:

    def __init__(self, extensions: list[str] | None = None) -> None:
        """
        Initialize with a list of allowed extensions.
        If none is provided, a default set is used.
        """
        self.cfg: Config = Config()
        if extensions is None:
            self._valid_extensions = [
                "pdf",
                "doc",
                "docx",
                "ppt",
                "pptx",
                "xls",
                "xlsx",
                "png",
                "jpg",
                "gif",
                "jpeg",
                "bmp",
                "tiff",
                "webp",
            ]
            self._valid_extensions.extend(self.cfg.get_list("_CONSIDER_AS_TEXT_FILE"))
            # After extending, if you want unique entries while preserving order:
            seen: set[str] = set()
            unique: list[str] = []
            for ext in self._valid_extensions:
                if ext not in seen:
                    seen.add(ext)
                    unique.append(ext)
            self._valid_extensions = unique
        else:
            self.set(extensions)

    def set(self, extensions: list[str]) -> None:
        """
        Set the allowed extensions list. Each extension is converted
        to lowercase and stripped of any leading period.

        Parameters:
            extensions (list): A list of extension strings.
        """
        self._valid_extensions = [ext.lower().lstrip(".") for ext in extensions]

    def get(self) -> list[str]:
        """
        Get the list of allowed extensions.

        Returns:
            list: The current allowed extensions in normalized form.
        """
        return self._valid_extensions

    def check(self, path: str, extensions: str | list[str]) -> bool:
        """
        Check if the file at the given path has any of the specified extension(s)
        that are also part of the allowed extensions list.

        Parameters:
            path (str): The file path (e.g., "document.pdf").
            extensions (str or list): A single extension (e.g., "pdf")
                                      or a list of extensions (e.g., ["pdf", "doc"]).

        Returns:
            bool: True if the file's extension (extracted from the path) is found
                  among the provided extension(s) and is allowed; otherwise, False.
        """
        # Ensure extensions is a list
        if not isinstance(extensions, list):
            extensions = [extensions]

        # Normalize the provided extensions (lowercase and remove any leading dot)
        normalized_extensions: list[str] = [
            ext.lower().lstrip(".") for ext in extensions
        ]

        # Filter out any provided extensions not present in the allowed list.
        valid_extensions_to_check: list[str] = [
            ext for ext in normalized_extensions if ext in self._valid_extensions
        ]

        if not valid_extensions_to_check:
            # None of the provided extensions is allowed.
            return False

        # Extract the file extension from the path and normalize it.
        _: str
        ext_in_path: str
        _, ext_in_path = os.path.splitext(path)
        ext_in_path = ext_in_path.lower().lstrip(".")

        # Return True if the file's extension is among the valid ones.
        return self.getFileType(path) in valid_extensions_to_check

    def getFileType(self, path: str) -> str:
        # Extract the file extension from the path and normalize it.
        _, ext_in_path = os.path.splitext(path)
        ext_in_path = ext_in_path.lower().lstrip(".")
        return ext_in_path
