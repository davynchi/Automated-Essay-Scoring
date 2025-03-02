import torch


SEED = 42
PATH_TO_SAVE_FINETUNED = "./deberta_v3_base"
PATH_TO_TOKENIZER = "./kaggle/input/lal-deberta-base-v018/"
TRAIN_DATA_PATH = "./learning-agency-lab-automated-essay-scoring-2/train.csv"
TRAIN_TEXT_PATH = "train_text.txt"
VAL_TEXT_PATH = "val_text.txt"
NAME_OF_MODEL_TO_FINETUNE = "microsoft/deberta-v3-base"
INPUT_DIR = "./kaggle/input/aes2-train-data/"
OOF_DIR = ""
MLM_PATH = "./kaggle/input/lal-deberta-base-mlm/deberta_v3_base_chk/checkpoint-57824/"
OUTPUT_DIR = "./kaggle/working/"
INPUT_DIR_INFERENCE = "./kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
OUTPUT_DIR_INFERENCE = "./"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
