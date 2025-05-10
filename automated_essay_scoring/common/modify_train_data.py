from glob import glob
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import DebertaTokenizer

from .constants import (
    DATA_PATH,
    PATH_TO_TOKENIZER,
    PROMPTED_DATA_FILENAME,
    TRAIN_FILENAME,
    TRAIN_PICKLE_PATH,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from .utils import modify_texts


# def load_pickle_data(cfg, load_from_existed_pickle):
#     train = pd.read_pickle(TRAIN_PICKLE_PATH)
#     if load_from_existed_pickle:
#         oof = pd.read_pickle(Path(cfg.path) / PICKLE_NAME)
#         train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
#         train[cfg.base.modif_target_cols[0]] = (
#             (train[cfg.base.target_cols[0]].values / 5) * (1 - cfg.base.sl_rate)
#         ) + (train["pred"].values * cfg.base.sl_rate)
#     else:
#         train[cfg.base.modif_target_cols[0]] = train[cfg.base.target_cols[0]].values / 5

#     return train


def load_pickle_data(cfg_unit, load_from_existed_pickle: bool) -> pd.DataFrame:
    """
    Return the training dataframe, optionally blended with stage-1 OOF predictions.

    Parameters
    ----------
    cfg_unit : DictConfig for the current ensemble member
    load_from_existed_pickle : bool
        False → stage-1, no OOF yet
        True  → stage-2, blend with OOF from each fold if they exist
    """
    train = pd.read_pickle(TRAIN_PICKLE_PATH)

    if load_from_existed_pickle:
        # ── gather all oof_fold*.pkl written in stage-1 ───────────────────── #
        oof_paths = sorted(glob(str(Path(cfg_unit.path) / "oof_fold*.pkl")))
        if not oof_paths:  # safety: if stage-1 never wrote the files yet
            print(
                f"[WARN] No OOF files found in {cfg_unit.path}; "
                "continuing without self-learning blend."
            )
            train[cfg_unit.base.modif_target_cols[0]] = (
                train[cfg_unit.base.target_cols[0]].values / 5
            )
            return train

        oof_list = [pd.read_pickle(p) for p in oof_paths]
        oof = pd.concat(oof_list, ignore_index=True)

        # ── merge preds & blend target -------------------------------------- #
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[cfg_unit.base.modif_target_cols[0]] = (
            (train[cfg_unit.base.target_cols[0]].values / 5) * (1 - cfg_unit.base.sl_rate)
        ) + (train["pred"].fillna(0).values * cfg_unit.base.sl_rate)
    else:
        train[cfg_unit.base.modif_target_cols[0]] = (
            train[cfg_unit.base.target_cols[0]].values / 5
        )

    return train


def read_train_dataset():
    train = pd.read_csv(DATA_PATH / TRAIN_FILENAME)
    train.columns = ["id", "text", "score"]
    return train


def divide_train_into_folds(train, n_splits):
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train["fold"] = -1
    train.loc[train["prompt_name"].isna(), "prompt_name"] = "Unknown prompt name"

    for fold, (_, val_) in enumerate(mskf.split(train, train[["prompt_name", "score"]])):
        train.loc[val_, "fold"] = fold


def set_flag_using_prompted_data(train):
    prompted_data = pd.read_csv(DATA_PATH / PROMPTED_DATA_FILENAME)
    merged_data = pd.merge(
        train, prompted_data, left_on="text", right_on="full_text", how="left"
    )

    merged_data["flag"] = 0
    merged_data.loc[merged_data["prompt_name"].isna(), "flag"] = 1
    return merged_data


def write_data_into_pickle(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data_back = data.rename(columns={"text": "full_text", "id": "essay_id"})
    data_back.to_pickle(file_path)


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
    train = train[:72]
    modify_texts(train["text"])
    train = set_flag_using_prompted_data(train)
    divide_train_into_folds(train, n_splits=cfg.n_folds)
    train = train[["id", "text", "score", "flag", "fold"]]
    write_data_into_pickle(train, TRAIN_PICKLE_PATH)
    train_text, val_text = divide_train_into_train_and_val_by_fold(train)
    write_train_and_val(train_text, val_text)
