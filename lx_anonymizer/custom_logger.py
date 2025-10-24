import logging
import sys

# Define a custom verbose log level
VERBOSE_LOG_LEVEL = 15
logging.addLevelName(VERBOSE_LOG_LEVEL, "VERBOSE")


def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LOG_LEVEL):
        self.log(VERBOSE_LOG_LEVEL, message, *args, **kwargs)


def configure_global_logger(verbose=False):
    """
    Configure the global logger with optional verbose logging.

    Args:
        verbose (bool): Enable verbose logging if True.
    """
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_logger(name, verbose=False):
    """
    Create a logger with the given name.

    Args:
        name (str): The name of the logger.
        verbose (bool): Whether to enable verbose logging

    Returns:
        logging.Logger: A logger object.
    """
    logger = logging.getLogger(name)

    # Configure the logger if it hasn't been configured yet
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    return logger


# Create a default logger for imports
logger = get_logger("lx_anonymizer")
