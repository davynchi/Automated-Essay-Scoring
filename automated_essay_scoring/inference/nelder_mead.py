import gc

import numpy as np
from scipy.optimize import minimize

from ..common.utils import LOGGER, get_score
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
        predictions.append(df_oof[f"pred_{i + 1}"].values)

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
    LOGGER.info("\n Best Weights: {weights:}".format(weights=bestWght))

    df_oof["blending"] = np.sum(
        bestWght * df_oof[[f"pred_{j + 1}" for j in range(num_models_in_ensemble)]],
        axis=1,
    )
    LOGGER.info(f"Blending score: {get_score(y_values, df_oof['blending'].values)}")
    LOGGER.info(
        f"Score on non prompted data: {get_score(df_oof.loc[df_oof.flag == 1, 'score'].values, df_oof.loc[df_oof.flag == 1, 'blending'].values)}"
    )

    del predictions, lls, wghts, starting_values, res, bestSC
    gc.collect()

    return bestWght
