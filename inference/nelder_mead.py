import gc

import numpy as np
from scipy.optimize import minimize

from ..common.cfg import CFG, CFG_LIST
from ..common.model_utils import get_score
from .oof import get_oof_preds


def calc_best_weights():
    df_oof = get_oof_preds()
    y_values = df_oof[CFG.target_cols].values

    predictions = []
    lls = []
    wghts = []

    def loss_func(weights, train_idx, predictions):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions, strict=True):
            final_prediction += weight * prediction[train_idx]

        score = get_score(y_values[train_idx], final_prediction)
        return -score

    for i in range(len(CFG_LIST)):
        predictions.append(df_oof[f"pred_{i + 1}"].values)

    for fold in range(CFG.n_fold):
        starting_values = [1 / len(CFG_LIST)] * len(CFG_LIST)
        res = minimize(
            loss_func,
            starting_values,
            args=(
                df_oof.loc[(df_oof.flag != 2) & (df_oof.fold == fold)].index,
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
    print("\n Ensemble Score: {best_score:.7f}".format(best_score=-bestSC))
    print("\n Best Weights: {weights:}".format(weights=bestWght))

    df_oof["blending"] = np.sum(
        bestWght * df_oof[[f"pred_{j + 1}" for j in range(len(CFG_LIST))]], axis=1
    )
    print(get_score(y_values, df_oof["blending"].values))
    print(
        get_score(
            df_oof.loc[df_oof.flag == 1, "score"].values,
            df_oof.loc[df_oof.flag == 1, "blending"].values,
        )
    )

    del predictions, lls, wghts, starting_values, res, bestSC
    gc.collect()

    return bestWght
