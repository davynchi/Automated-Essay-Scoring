from pathlib import Path

import mlflow
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
        train[cfg.base.modif_target_cols[0]] = (
            (train[cfg.base.target_cols[0]].values / 5) * (1 - cfg.base.sl_rate)
        ) + (train["pred"].values * cfg.base.sl_rate)
    else:
        train[cfg.base.modif_target_cols[0]] = train[cfg.base.target_cols[0]].values / 5

    return train


def train_one_stage_of_model(
    cfg, cfg_unit, model_idx, checkpoints_names, tokenizer, will_train_again
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
        score_all_texts, score_unprompted_texts = get_result(
            cfg.base.target_cols, _oof_df
        )
        mlflow.log_metric(
            f"model_{model_idx}_fold_{fold}_score_all_texts", score_all_texts
        )
        mlflow.log_metric(
            f"model_{model_idx}_fold_{fold}_score_unprompted_texts",
            score_unprompted_texts,
        )
    oof_df = oof_df.reset_index(drop=True)

    LOGGER.info("========== CV ==========")
    score_all_texts, score_unprompted_texts = get_result(cfg.base.target_cols, oof_df)
    mlflow.log_metric(f"CV_model_{model_idx}_score_all_texts", score_all_texts)
    mlflow.log_metric(
        f"CV_model_{model_idx}_score_unprompted_texts", score_unprompted_texts
    )

    oof_df.to_pickle(Path(cfg_unit.path) / PICKLE_NAME)
    mlflow.log_artifact(str(Path(cfg_unit.path) / PICKLE_NAME))


def train_model(cfg, checkpoints_names, tokenizer):
    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        train_one_stage_of_model(
            cfg, cfg_unit, model_idx, checkpoints_names, tokenizer, False
        )
        train_one_stage_of_model(
            cfg, cfg_unit, model_idx, checkpoints_names, tokenizer, True
        )
