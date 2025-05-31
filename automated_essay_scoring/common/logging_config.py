"""
Logging configuration utilities.

Provides a function to configure the root logger with a console handler,
a rotating file handler, and to suppress verbose logs from common libraries.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


def configure_logging(level: str = "INFO", filename: str = "train.log") -> None:
    """Configure the root logger with console and rotating file handlers.

    Sets up:
      * Console handler with timestamped formatting.
      * Rotating file handler under `LOG_DIR` (10 MB per file, 3 backups).
      * Suppression of verbose output from PyTorch Lightning, Transformers, MLflow, etc.
      * Elevates MLflow “local version label” warnings to ERROR level.

    Args:
        level (str): Logging level for the root logger (e.g., "INFO", "DEBUG").
        filename (str): Name of the log file to create under `LOG_DIR`.

    Returns:
        None
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers (e.g., from Lightning / Transformers)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console_handler)

    # Rotating file handler: 10 MB per file, keep 3 backups
    file_handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_handler)

    # Suppress verbose logs from common libraries
    # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    # logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
    # logging.getLogger("transformers").setLevel(logging.WARNING)
    # logging.getLogger("fsspec").setLevel(logging.WARNING)
    # logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)

    # # Silence MLflow “local version label” warnings
    # logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
