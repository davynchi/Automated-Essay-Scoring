from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from transformers import DebertaTokenizer

from .constants import (
    INPUT_DIR_INFERENCE,
    PATH_TO_TOKENIZER,
    TRAIN_DATA_PATH,
    TRAIN_PICKLE_PATH,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from .utils import modify_texts


def read_train_dataset():
    train = pd.read_csv(TRAIN_DATA_PATH)
    # train = train[["essay_id", "full_text"]]
    train.columns = ["id", "text", "score"]
    return train


def load_test_submission_data():
    test = pd.read_csv(f"{INPUT_DIR_INFERENCE}test.csv")
    submission = pd.read_csv(f"{INPUT_DIR_INFERENCE}sample_submission.csv")
    modify_texts(test["full_text"])
    return test, submission


def divide_train_into_folds(train, n_splits):  # n_splits=20): Так было, поправь потом
    gkf = GroupKFold(n_splits=n_splits)
    train["fold"] = -1

    for fold, (_, val_) in enumerate(gkf.split(train, train, train["id"])):
        train.loc[val_, "fold"] = fold


def divide_train_into_train_and_two_validations(train):
    train["flag"] = -1

    for fold_value in train["fold"].unique():
        fold_indices = train.index[train["fold"] == fold_value]
        flag_0, flag_1 = train_test_split(fold_indices, test_size=0.25)
        train.loc[flag_0, "flag"] = 0
        train.loc[flag_1, "flag"] = 1


def write_data_into_pickle(data, path):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data_back = data.rename(columns={"text": "full_text", "id": "essay_id"})
    data_back.to_pickle(path)


def create_tokenizer(path):
    tokenizer = DebertaTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[BR]"]})
    return tokenizer


def tokenize_text(data, path_to_tokenizer=PATH_TO_TOKENIZER):
    tokenizer = create_tokenizer(path=path_to_tokenizer)

    def text_encode(text):
        return len(tokenizer.encode(text))

    data["length"] = data["full_text"].map(text_encode)
    data = data.sort_values("length", ascending=True).reset_index(drop=True)
    return tokenizer


def divide_train_into_train_and_val_by_fold(train):
    train_text = "\n".join(train.loc[train["fold"] != 0, "text"].tolist())
    val_text = "\n".join(train.loc[train["fold"] == 0, "text"].tolist())
    return train_text, val_text


def write_train_and_val(train_text, val_text):
    with open(TRAIN_TEXT_PATH, "w") as f:
        f.write(train_text)
    with open(VAL_TEXT_PATH, "w") as f:
        f.write(val_text)


def modify_train_data(cfg):
    train = read_train_dataset()
    train = train[:24]
    modify_texts(train["text"])
    divide_train_into_folds(train, n_splits=cfg.n_folds)
    divide_train_into_train_and_two_validations(train)
    write_data_into_pickle(train, TRAIN_PICKLE_PATH)
    train_text, val_text = divide_train_into_train_and_val_by_fold(train)
    write_train_and_val(train_text, val_text)
