from pathlib import Path

import pandas as pd

from ..common.constants import PICKLE_NAME, TRAIN_PICKLE_PATH
from ..common.modify_train_data import tokenize_text
from ..common.utils import LOGGER, get_result
from .train_loop import train_loop


def load_pickle_data(cfg, load_from_existed_pickle):
    train = pd.read_pickle(TRAIN_PICKLE_PATH)
    if load_from_existed_pickle:
        oof = pd.read_pickle(Path(cfg.path) / PICKLE_NAME)
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[cfg.base.target_cols3[0]] = (
            (train[cfg.base.target_cols2[0]].values / 5) * (1 - cfg.base.sl_rate)
        ) + (train["pred"].values * cfg.base.sl_rate)
    else:
        train[cfg.base.target_cols3[0]] = train[cfg.base.target_cols2[0]].values / 5

    return train


def train_one_stage_of_model(
    cfg, cfg_unit, checkpoints_names, tokenizer, will_train_again
):
    train = load_pickle_data(cfg_unit, load_from_existed_pickle=will_train_again)
    tokenize_text(train)
    oof_df = pd.DataFrame()
    for fold in range(cfg.n_folds):
        _oof_df = train_loop(
            train,
            fold,
            cfg_unit,
            checkpoints_names,
            tokenizer,
            will_train_again=will_train_again,
        )
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(cfg.base.target_cols2, _oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info("========== CV ==========")
    get_result(cfg.base.target_cols2, oof_df)
    oof_df.to_pickle(Path(cfg_unit.path) / PICKLE_NAME)


def train_and_save_main_model(cfg, checkpoints_names, tokenizer):
    for cfg_unit in cfg.ensemble.values():
        train_one_stage_of_model(cfg, cfg_unit, checkpoints_names, tokenizer, False)
        train_one_stage_of_model(cfg, cfg_unit, checkpoints_names, tokenizer, True)
