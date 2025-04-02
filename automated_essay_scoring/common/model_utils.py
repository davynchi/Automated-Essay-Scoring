import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def get_essay_score(y_preds):
    return pd.cut(
        y_preds.reshape(-1) * 5,
        [-np.inf, 0.83333333, 1.66666667, 2.5, 3.33333333, 4.16666667, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)


def get_score(y_trues, y_preds):
    y_preds = get_essay_score(y_preds)
    score = cohen_kappa_score(y_trues, y_preds, weights="quadratic")
    return score
