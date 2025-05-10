import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils import get_result, get_score


log = logging.getLogger(__name__)


def load_all_folds(model_dir: Path) -> pd.DataFrame:
    """
    Concatenate every `oof_fold*.pkl` saved for this model directory.
    """
    pkls = sorted(model_dir.glob("oof_fold*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No OOF pickle found in {model_dir}")
    dfs = [pd.read_pickle(p) for p in pkls]
    return pd.concat(dfs, ignore_index=True)


def get_oof_preds(cfg):
    for i, cfg_unit in enumerate(cfg.ensemble.values()):
        oof_df = load_all_folds(Path(cfg_unit.path))

        log.info(f"pred_{i} {cfg_unit.path}")
        score_all_texts, score_unprompted_texts = get_result(
            cfg_unit.base.target_cols, oof_df
        )
        mlflow.log_metric(f"model_{i}_score_all_texts", score_all_texts)
        mlflow.log_metric(f"model{i}_score_unprompted_texts", score_unprompted_texts)

        if i == 0:
            df_oof = oof_df[
                ["essay_id", "fold", "flag"] + cfg_unit.base.target_cols + ["pred"]
            ].rename(columns={"pred": f"pred_{i}"})
        else:
            oof_df = oof_df[["essay_id", "pred"]].rename(columns={"pred": f"pred_{i}"})
            df_oof = pd.merge(df_oof, oof_df, on="essay_id", how="left").reset_index(
                drop=True
            )

    df_oof = df_oof.dropna().reset_index(drop=True)
    return df_oof


def calc_best_weights_for_ensemble(cfg):
    df_oof = get_oof_preds(cfg)
    y_values = df_oof[cfg.base.target_cols].values

    predictions = []
    lls = []
    wghts = []

    def loss_func(weights, train_idx, predictions):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions, strict=True):
            final_prediction += weight * prediction[train_idx]

        score = get_score(y_values[train_idx], final_prediction)
        return -score

    num_models_in_ensemble = len(cfg.ensemble)
    for i in range(num_models_in_ensemble):
        predictions.append(df_oof[f"pred_{i}"].values)

    for fold in range(cfg.n_folds):
        starting_values = [1 / num_models_in_ensemble] * num_models_in_ensemble
        res = minimize(
            loss_func,
            starting_values,
            args=(
                df_oof.loc[df_oof.fold == fold].index,
                predictions,
            ),
            method="Nelder-Mead",
            tol=1e-6,
        )

        lls.append(res["fun"])
        wghts.append(res["x"])

    bestSC = np.mean(lls)
    bestWght = np.mean(wghts, axis=0)
    bestWght = bestWght / bestWght.sum()
    log.info("\n Ensemble Score: {best_score:.7f}".format(best_score=-bestSC))
    mlflow.log_metric("ensemble_score", -bestSC)
    log.info("\n Best Weights: {weights:}".format(weights=bestWght))
    mlflow.log_param("best_weights", bestWght)

    df_oof["blending"] = np.sum(
        bestWght * df_oof[[f"pred_{j}" for j in range(num_models_in_ensemble)]],
        axis=1,
    )

    blending_score_all_texts, blending_score_unprompted_texts = get_result(
        cfg.base.target_cols, df_oof, "blending"
    )
    mlflow.log_metric("blending_score_all_texts", blending_score_all_texts)
    mlflow.log_metric("blending_score_unprompted_texts", blending_score_unprompted_texts)

    # # 1. Диапазон и уникальность прогнозов
    # print(df_oof[[f"pred_{i}" for i in range(4)]].describe())
    # print(df_oof[[f"pred_{i}" for i in range(4)]].corr())

    # # 2. Попробовать другой старт
    # for _ in range(3):
    #     start = np.random.dirichlet(np.ones(len(cfg.ensemble)))
    #     res = minimize(
    #         loss_func,
    #         start,
    #         args=(
    #             df_oof.loc[df_oof.fold == fold].index,
    #             predictions,
    #         ),
    #         method="Nelder-Mead",
    #         tol=1e-6,
    #     )
    #     print(res.x, -res.fun)

    return bestWght
