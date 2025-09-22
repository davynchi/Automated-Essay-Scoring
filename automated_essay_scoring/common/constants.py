from pathlib import Path


OUTPUT_DIR_FINETUNED = Path("./data/models/pretrained_models")
CACHED_DATA_DIR = Path("./data/cached_data")
OUTPUT_DIR_TRAIN = Path("./data/models/trained_models")
RAW_DATA_PATH = Path("./data/raw")
TRITON_MODELS_PATH = Path("./triton_repo")
HYDRA_CONFIG_PATH = Path("./../../configs")

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
TEST_SCORE_FILENAME = "test_score.csv"
PROMPTED_DATA_FILENAME = "persuade_2.0_human_scores_demo_id_github.csv"

TRAIN_DATA_FILENAMES = [TRAIN_FILENAME, PROMPTED_DATA_FILENAME]
TEST_DATA_FILENAMES = [TEST_FILENAME, TEST_SCORE_FILENAME]

TRAIN_TEXT_PATH = CACHED_DATA_DIR / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_DIR / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_DIR / "train.pkl"

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
MAX_SPLIT_SIZE_MB = 128
