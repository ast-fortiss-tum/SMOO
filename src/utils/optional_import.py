import warnings
from functools import partial
from typing import Any


def optional_import(module_path: str, symbol_name: str) -> object:
    """
    Handles optional imports if dependencies are not installed.

    :param module_path: The module path to use for import.
    :param symbol_name: The symbol name to import.
    :returns: The imported symbol or a dummy object if an ImportError is raised.
    """
    try:
        module = __import__(module_path, fromlist=[symbol_name])
    except ModuleNotFoundError as import_error:
        warnings.warn(
            f"Could not import {symbol_name} from {module_path} due to {import_error}. Will use dummy object instead.",
            ImportWarning,
        )

        class Dummy:
            """A dummy class for imports that fail."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """
                Initialize the dummy class.

                :param args: The args to pass to the class.
                :param kwargs: The kwargs to pass to the class.
                :raises ImportError: Always raised.
                """
                raise ImportError(
                    f"Could not import {kwargs['symbol_name']}, it requires {kwargs['import_error'].name} to be installed."
                )

        return partial(Dummy, symbol_name=symbol_name, import_error=import_error)
    return getattr(module, symbol_name)
