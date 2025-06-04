from pathlib import Path


OUTPUT_DIR_FINETUNED = Path("./data/models/pretrained_models")
CACHED_DATA_PATH = Path("./data/cached_data")
OUTPUT_DIR_TRAIN = Path("./data/models/trained_models")
RAW_DATA_PATH = Path("./data/raw")
TRITON_MODELS_PATH = Path("./triton_repo")
HYDRA_CONFIG_PATH = Path("./../../configs")

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
PROMPTED_DATA_FILENAME = "persuade_2.0_human_scores_demo_id_github.csv"
ALL_DATA_FILENAMES = [TRAIN_FILENAME, TEST_FILENAME, PROMPTED_DATA_FILENAME]

TRAIN_TEXT_PATH = CACHED_DATA_PATH / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_PATH / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_PATH / "train.pkl"

BEST_ENSEMBLE_WEIGHTS_DIR = Path("./data/models/best_ensemble_weights")
BEST_ENSEMBLE_WEIGHTS_FILENAME = "best_ensemble_weights.npy"
BEST_ENSEMBLE_WEIGHTS_PATH = BEST_ENSEMBLE_WEIGHTS_DIR / BEST_ENSEMBLE_WEIGHTS_FILENAME

SUBMISSION_DIR = Path("./data/submission")
SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_PATH = SUBMISSION_DIR / SUBMISSION_FILENAME

NAMES_OF_MODELS = {
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}
PATH_TO_TOKENIZER = "microsoft/deberta-base"
MODEL_UNIT_CONFIG_NAME = "config.pth"
HYDRA_CONFIG_NAME = "defaults"

# File naming convention
CHECKPOINT_POSTFIX = "_chk"
BEST_CHECKPOINT_POSTFIX = "_final"

STAGES_NUM = 2
