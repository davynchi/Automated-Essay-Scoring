import logging
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..common.constants import BEST_ENSEMBLE_WEIGHTS_PATH
from ..common.utils import get_score


log = logging.getLogger(__name__)


def load_all_folds(model_dir: Path) -> pd.DataFrame:
    """Собирает все OOF-pickle файлы в один DataFrame.

    Args:
        model_dir (Path): Путь к директории с файлами вида `oof_fold*.pkl`.

    Returns:
        pd.DataFrame: Объединённый DataFrame по всем фолдам.

    Raises:
        FileNotFoundError: Если в директории `model_dir` не найдено ни одного файла.
    """
    pkls = sorted(model_dir.glob("oof_fold*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No OOF pickle found in {model_dir}")
    dfs = [pd.read_pickle(p) for p in pkls]
    return pd.concat(dfs, ignore_index=True)


def get_result(
    target_cols: list[str], oof_df: pd.DataFrame, pred_col: str = "pred"
) -> Tuple[float, float]:
    """Compute overall and unprompted QWK scores for OOF predictions.

    Logs and returns:
      - score on all examples
      - score on subset where `flag == 1` (unprompted)

    Args:
        target_cols (list[str]): Column names of true labels in `oof_df`.
        oof_df (pd.DataFrame): DataFrame with true labels, predictions, and `flag`.
        pred_col (str): Column name of predictions in `oof_df`.

    Returns:
        Tuple[float, float]: (all_texts_score, unprompted_texts_score).
    """
    y_true_all = oof_df[target_cols].values
    y_pred_all = oof_df[pred_col].values
    score_all = get_score(y_true_all, y_pred_all)

    mask = oof_df["flag"] == 1
    y_true_up = oof_df.loc[mask, target_cols].values
    y_pred_up = oof_df.loc[mask, pred_col].values
    score_up = get_score(y_true_up, y_pred_up)

    logging.getLogger(__name__).info(
        f"Score: {score_all:.4f}, Unprompted Score: {score_up:.4f}"
    )
    return score_all, score_up


def get_oof_preds(cfg) -> pd.DataFrame:
    """Формирует общую таблицу OOF-предсказаний для всех моделей ансамбля.

    Для каждой модели в `cfg.ensemble`:
      - загружает предсказания всех фолдов через `load_all_folds`
      - вычисляет метрики качества `get_result` и логирует их в MLflow
      - объединяет столбцы предсказаний `pred_{i}` для каждого `essay_id`

    Args:
        cfg: Конфигурация, полученная от Hydra.
            Должна содержать атрибут `ensemble` (mapping)
            и у каждого элемента — `path` и `base.target_cols`.

    Returns:
        pd.DataFrame:
            DataFrame со столбцами:
                - essay_id
                - fold
                - flag
                - целевые колонки из `cfg.base.target_cols`
                - pred_0, pred_1, ..., pred_{n_models-1}

    Raises:
        FileNotFoundError: Если для какой-то модели не найдено OOF-файлов.
    """
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
            oof_small = oof_df[["essay_id", "pred"]].rename(columns={"pred": f"pred_{i}"})
            df_oof = pd.merge(df_oof, oof_small, on="essay_id", how="left").reset_index(
                drop=True
            )

    df_oof = df_oof.dropna().reset_index(drop=True)
    return df_oof


def calc_best_weights_for_ensemble(cfg) -> np.ndarray:
    """Подбирает и сохраняет оптимальные веса для ансамбля моделей.

    Используется алгоритм Nelder–Mead для максимизации QWK на каждом фолде:
      1. Получает OOF-предсказания через `get_oof_preds`
      2. Для каждого фолда минимизирует отрицательное значение `get_score`
      3. Усредняет найденные вектора весов, нормирует их и сохраняет в файл
      4. Логирует финальный скор ансамбля и веса в MLflow
      5. Вычисляет и логирует скор финального смешанного предсказания

    В результате сохраняется файл
    `<BEST_ENSEMBLE_WEIGHTS_DIR>/best_ensemble_weights.npy`.

    Args:
        cfg: Конфигурация, полученная от Hydra.
            Должна содержать атрибуты:
            - `ensemble` (mapping моделей)
            - `base.target_cols` (список целевых колонок)
            - `n_folds` (число фолдов)

    Returns:
        np.ndarray:
            Нормированный вектор весов ансамбля формы `(n_models,)`.
    """
    df_oof = get_oof_preds(cfg)
    y_values = df_oof[cfg.base.target_cols].values

    predictions = []
    losses_per_fold = []
    weights_per_fold = []

    def loss_func(weights, train_idx, preds):
        """Вспомогательная функция для оптимизации QWK."""
        final_pred = np.sum(
            [w * preds[j][train_idx] for j, w in enumerate(weights)], axis=0
        )
        score = get_score(y_values[train_idx], final_pred)
        return -score

    num_models = len(cfg.ensemble)
    predictions = [df_oof[f"pred_{i}"].values for i in range(num_models)]

    for fold in range(cfg.n_folds):
        start = [1 / num_models] * num_models
        res = minimize(
            loss_func,
            start,
            args=(df_oof[df_oof.fold == fold].index, predictions),
            method="Nelder-Mead",
            tol=1e-6,
        )
        losses_per_fold.append(res.fun)
        weights_per_fold.append(res.x)

    # Средний скор (отрицательный в оптимизации) и нормировка весов
    avg_loss = np.mean(losses_per_fold)
    best_weights = np.mean(weights_per_fold, axis=0)
    best_weights /= best_weights.sum()

    log.info(f"Ensemble Score: {-avg_loss:.7f}")
    mlflow.log_metric("ensemble_score", -avg_loss)
    log.info(f"Best Weights: {best_weights}")
    mlflow.log_param("best_weights", best_weights)

    # Смешанное предсказание
    df_oof["blending"] = np.sum(
        best_weights * df_oof[[f"pred_{j}" for j in range(num_models)]],
        axis=1,
    )
    blend_all, blend_unprompt = get_result(cfg.base.target_cols, df_oof, "blending")
    mlflow.log_metric("blending_score_all_texts", blend_all)
    mlflow.log_metric("blending_score_unprompted_texts", blend_unprompt)

    # Сохранение весов на диск
    BEST_ENSEMBLE_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(BEST_ENSEMBLE_WEIGHTS_PATH, best_weights)

    return best_weights
