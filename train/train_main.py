import pandas as pd

from ..common.cfg import CFG, CFG_LIST
from ..common.common import LOGGER
from ..common.constants import N_FOLDS, OOF_DIR, TRAIN_PICKLE_PATH
from ..common.modify_train_data import tokenize_text
from ..common.score import get_score
from .train_loop import train_loop


def get_result(oof_df):
    labels = oof_df[CFG.target_cols2].values
    preds = oof_df["pred"].values
    score = get_score(labels, preds)
    labels = oof_df.loc[oof_df.flag == 1, CFG.target_cols2].values
    preds = oof_df.loc[oof_df.flag == 1, "pred"].values
    score2 = get_score(labels, preds)
    LOGGER.info(f"Score: {score:<.4f} Score2: {score2:<.4f}")


def load_pickle_data():
    train = pd.read_pickle(TRAIN_PICKLE_PATH)
    if CFG.sl:
        oof = pd.read_pickle(f"{OOF_DIR}oof_df.pkl")
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[CFG.target_cols3[0]] = (
            (train[CFG.target_cols2[0]].values / 5) * (1 - CFG.sl_rate)
        ) + (train["pred"].values * CFG.sl_rate)
    else:
        train[CFG.target_cols3[0]] = train[CFG.target_cols2[0]].values / 5

    return train


def train_and_save_main_model():
    for cfg in CFG_LIST:
        if CFG.train:
            train = load_pickle_data()
            tokenize_text(train)
            oof_df = pd.DataFrame()
            for fold in range(N_FOLDS):
                if fold in CFG.trn_fold:
                    _oof_df = train_loop(train, fold, cfg)
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(_oof_df)
            oof_df = oof_df.reset_index(drop=True)
            LOGGER.info("========== CV ==========")
            get_result(oof_df)
            oof_df.to_pickle(cfg.path / cfg.pickle_name)
