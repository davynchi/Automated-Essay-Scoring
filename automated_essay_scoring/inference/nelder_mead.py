import mlflow
import numpy as np
from scipy.optimize import minimize

from ..common.utils import LOGGER, get_result, get_score
from .oof import get_oof_preds


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
    LOGGER.info("\n Ensemble Score: {best_score:.7f}".format(best_score=-bestSC))
    mlflow.log_metric("ensemble_score", -bestSC)
    LOGGER.info("\n Best Weights: {weights:}".format(weights=bestWght))
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

    return bestWght
