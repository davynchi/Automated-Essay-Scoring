import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from .constants import OUTPUT_DIR, OUTPUT_DIR_INFERENCE, SEED


def ignore_warnings():
    warnings.filterwarnings("ignore")


ignore_warnings()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(OUTPUT_DIR_INFERENCE):
    os.makedirs(OUTPUT_DIR_INFERENCE)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_logger(filename=OUTPUT_DIR + "train"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=SEED)
