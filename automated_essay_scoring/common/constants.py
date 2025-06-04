from pathlib import Path


OUTPUT_DIR_FINETUNED = Path("./pretrained_models")
CACHED_DATA_PATH = Path("./cached_data")
OUTPUT_DIR_TRAIN = Path("./trained_models")
BEST_ENSEMBLE_WEIGHTS_PATH = Path("./best_ensemble_weights")
RAW_DATA_PATH = Path("./data/raw")
TRITON_MODELS_PATH = Path("./triton_repo")
HYDRA_CONFIG_PATH = Path("./../../configs")

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
PROMPTED_DATA_FILENAME = "persuade_2.0_human_scores_demo_id_github.csv"
ALL_DATA_FILENAMES = [TRAIN_FILENAME, TEST_FILENAME, PROMPTED_DATA_FILENAME]

BEST_ENSEMBLE_WEIGHTS_FILENAME = "best_ensemble_weights.npy"

TRAIN_TEXT_PATH = CACHED_DATA_PATH / "train_text.txt"
VAL_TEXT_PATH = CACHED_DATA_PATH / "val_text.txt"
TRAIN_PICKLE_PATH = CACHED_DATA_PATH / "train.pkl"

SUBMISSION_PATH = Path("./data/submission/submission.csv")

NAMES_OF_MODELS = {
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}
PATH_TO_TOKENIZER = "microsoft/deberta-base"
PICKLE_NAME = "oof_df.pkl"
MODEL_UNIT_CONFIG_NAME = "config.pth"
HYDRA_CONFIG_NAME = "defaults"

# File naming convention
CHECKPOINT_POSTFIX = "_chk"
BEST_CHECKPOINT_POSTFIX = "_final"

STAGES_NUM = 2
