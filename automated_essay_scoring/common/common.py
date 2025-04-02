import os
import random
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import numpy as np
import torch

from .constants import OUTPUT_DIR_TRAIN, SEED


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def setup_logger(filename=OUTPUT_DIR_TRAIN / "train.log"):
    OUTPUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = setup_logger()


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=SEED)
