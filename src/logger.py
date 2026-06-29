import logging
from pathlib import Path

def setup_logger(file_path: Path, logger_name: str) -> logging.Logger:
    """
    Configure and return a logger that writes to console and file.

    Creates the parent directory of ``file_path`` if it does not exist, sets
    the log level to INFO, and attaches stream and file handlers with a
    timestamped message format.

    Args:
        file_path (Path): Path to the log file.
        logger_name (str): Name used to retrieve or create the logger instance.

    Returns:
        logging.Logger: The configured logger with console and file handlers.
    """
    # create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # check if the logs folder exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

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