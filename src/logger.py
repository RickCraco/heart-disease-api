import logging
from pathlib import Path

def setup_logger(file_path: Path, logger_name: str) -> logging.Logger:
    """
    Configure and return a logger that writes to both console and file.
    """
    # create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # handler for console and file
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    # formatter for logs format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger