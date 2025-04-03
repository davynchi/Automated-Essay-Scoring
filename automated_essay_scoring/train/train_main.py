from pathlib import Path

import pandas as pd

from ..common.constants import OOF_DIR, PICKLE_NAME, TRAIN_PICKLE_PATH
from ..common.modify_train_data import tokenize_text
from ..common.utils import LOGGER, get_score
from .train_loop import train_loop


def get_result(cfg, oof_df):
    labels = oof_df[cfg.base.target_cols2].values
    preds = oof_df["pred"].values
    score = get_score(labels, preds)
    labels = oof_df.loc[oof_df.flag == 1, cfg.base.target_cols2].values
    preds = oof_df.loc[oof_df.flag == 1, "pred"].values
    score2 = get_score(labels, preds)
    LOGGER.info(f"Score: {score:<.4f} Score2: {score2:<.4f}")


def load_pickle_data(cfg, load_from_existed_pickle):
    train = pd.read_pickle(TRAIN_PICKLE_PATH)
    if load_from_existed_pickle:
        oof = pd.read_pickle(OOF_DIR / PICKLE_NAME)
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[cfg.base.target_cols3[0]] = (
            (train[cfg.base.target_cols2[0]].values / 5) * (1 - cfg.base.sl_rate)
        ) + (train["pred"].values * cfg.base.sl_rate)
    else:
        train[cfg.base.target_cols3[0]] = train[cfg.base.target_cols2[0]].values / 5

    return train


def train_and_save_main_model(cfg, checkpoints_names, tokenizer):
    for cfg_unit in cfg.ensemble.values():
        if cfg_unit.base.train:
            train = load_pickle_data(cfg, False)
            tokenize_text(train)
            oof_df = pd.DataFrame()
            for fold in range(cfg.n_folds):
                if fold in cfg_unit.base.trn_fold:
                    _oof_df = train_loop(
                        train, fold, cfg_unit, checkpoints_names, tokenizer
                    )
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(cfg, _oof_df)
            oof_df = oof_df.reset_index(drop=True)
            LOGGER.info("========== CV ==========")
            get_result(cfg, oof_df)
            oof_df.to_pickle(Path(cfg_unit.path) / PICKLE_NAME)
