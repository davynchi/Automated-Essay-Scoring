from pathlib import Path

import torch


SEED = 42
BASE_PATH_TO_SAVE_FINETUNED = Path("./pretrained_models")
NAMES_OF_MODEL_TO_FINETUNE = {
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}
CHECKPOINTS_NAMES = {}
CHECKPOINT_POSTFIX = "_chk"
CACHED_DATA_DIR = Path("./cached_data")
PATH_TO_TOKENIZER = (
    "microsoft/deberta-base"  # "./kaggle/input/lal-deberta-base-v018/" -- так было
)
TRAIN_DATA_PATH = "./learning-agency-lab-automated-essay-scoring-2/train.csv"
TRAIN_TEXT_PATH = CACHED_DATA_DIR / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_DIR / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_DIR / "train.pkl"
INPUT_DIR = "./kaggle/input/aes2-train-data/"
OOF_DIR = ""
OUTPUT_DIR = "./trained_models"  # "./kaggle/working/"
INPUT_DIR_INFERENCE = "./learning-agency-lab-automated-essay-scoring-2/"
OUTPUT_DIR_INFERENCE = "./"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_SIZE = 64  # 512 in the original code
N_FOLDS = 2
