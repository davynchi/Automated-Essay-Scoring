from pathlib import Path

import torch


# Fundamental defaults
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Technical paths (stable defaults)
BASE_PATH_TO_SAVE_FINETUNED = Path("./pretrained_models")
CACHED_DATA_DIR = Path("./cached_data")
TRAIN_DATA_PATH = "./learning-agency-lab-automated-essay-scoring-2/train.csv"
TRAIN_TEXT_PATH = CACHED_DATA_DIR / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_DIR / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_DIR / "train.pkl"
INPUT_DIR = "./kaggle/input/aes2-train-data/"
OOF_DIR = ""
OUTPUT_DIR = "./trained_models"
INPUT_DIR_INFERENCE = "./learning-agency-lab-automated-essay-scoring-2/"
OUTPUT_DIR_INFERENCE = "./"
SUBMISSION_FILE_NAME = "submission.csv"

NAMES_OF_MODELS = {
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}
PATH_TO_TOKENIZER = "microsoft/deberta-base"
PICKLE_NAME = "oof_df.pkl"
MODEL_UNIT_CONFIG_NAME = "config.pth"

# File naming convention
CHECKPOINT_POSTFIX = "_chk"
