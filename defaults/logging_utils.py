import inspect
import logging
import sys


def enable_logging_notebook() -> None:
    """Allow logging outputs in notebooks."""
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.handlers[0].stream = sys.stdout


class ContextFilter(logging.Filter):
    """Includes context of where functions are called."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log for relevant context.

        :param record: Log record.
        :returns: True.
        """
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            module = frame.f_globals.get("__name__", "")
            if not module.startswith("logging"):
                break

        cls_name = ""
        if "self" in frame.f_locals:
            cls_name = frame.f_locals["self"].__class__.__name__

        record.context = (
            f"{cls_name}.{frame.f_code.co_name}:{frame.f_lineno}"
            if cls_name
            else f"{module}.{frame.f_code.co_name}:{frame.f_lineno}"
        )
        return True


def get_default_handler() -> logging.Handler:
    """
    Get a default handler.

    :returns: Default handler.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "\033[94m[%(asctime)s]\033[0m - %(levelname)s - \033[90m%(context)s\033[0m - %(message)s"
        )
    )
    return handler
