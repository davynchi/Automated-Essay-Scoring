from pathlib import Path

import torch


# Fundamental defaults
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR_FINETUNED = Path("./pretrained_models")
CACHED_DATA_PATH = Path("./cached_data")
OUTPUT_DIR_TRAIN = Path("./trained_models")

DATA_PATH = Path("./learning-agency-lab-automated-essay-scoring-2")
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"

TRAIN_TEXT_PATH = CACHED_DATA_PATH / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_PATH / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_PATH / "train.pkl"

OOF_DIR = Path("./")
SUBMISSION_PATH = Path("./")
SUBMISSION_FILENAME = "submission.csv"

NAMES_OF_MODELS = {
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}
PATH_TO_TOKENIZER = "microsoft/deberta-base"
PICKLE_NAME = "oof_df.pkl"
MODEL_UNIT_CONFIG_NAME = "config.pth"

# File naming convention
CHECKPOINT_POSTFIX = "_chk"
