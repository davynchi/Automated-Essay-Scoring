# common/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)


def configure_logging(level: str = "INFO", filename: str = "train.log") -> None:
    """
    Настраивает корневой логгер:
      * Консоль + вращающийся file‑handler.
      * Подавляет подробный вывод Lightning/Transformers/MLflow.
      * Перенаправляет предупреждения о «local version label» на ERROR.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # clean root handlers created by Lightning/Transformers, if any
    for h in list(root.handlers):
        root.removeHandler(h)

    # console
    con = logging.StreamHandler()
    con.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(con)

    # rotating file (10 MB × 3 backups)
    fh = RotatingFileHandler(
        LOG_DIR / filename, maxBytes=10_000_000, backupCount=3, encoding="utf‑8"
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    # 2) raise Lightning’s log level to WARNING (so you only get warnings/errors)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
    # if you still see stuff from transformers or fsspec, you can also:
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)

    # 3) silence MLflow’s “local version label” warnings
    logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
