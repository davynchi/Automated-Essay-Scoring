# inference_lightning.py
import gc
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .constants import (
    BEST_ENSEMBLE_WEIGHTS_FILENAME,
    BEST_ENSEMBLE_WEIGHTS_PATH,
    DATA_PATH,
    PATH_TO_TOKENIZER,
    SUBMISSION_FILENAME,
    SUBMISSION_PATH,
    TEST_FILENAME,
)
from .dataset import LALDataset, collate
from .lightning_modules import EssayScoringPL
from .modify_train_data import create_tokenizer, modify_texts
from .utils import get_essay_score


log = logging.getLogger(__name__)


# ───────────────── helpers ───────────────────────────────────────────── #
def load_test() -> pd.DataFrame:
    """Читает и нормализует тестовый набор"""
    test = pd.read_csv(DATA_PATH / TEST_FILENAME)
    modify_texts(test["full_text"])
    return test[:128]


def fold_stage2_ckpt(model_dir: Path, model_idx: int, fold: int) -> str:
    """
    Находит новейший чек‑пойнт stage‑2 для пары (model_idx, fold).
    """
    pattern = f"model{model_idx}_fold{fold}_stage2.ckpt"
    matches = list(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No ckpt matches {pattern}")
    # usually only one file, but take the newest just in case
    return max(matches, key=lambda p: p.stat().st_mtime).as_posix()


# ---------- collate for inference ------------------------------------ #
def collate_infer(batch):
    """Collate для инференса — только обрезка до max_len без меток."""
    inputs = default_collate(batch)  # step-1
    return collate(inputs)  # step-2 (trim)


def write_submission(test_df: pd.DataFrame) -> None:
    """Пишет ``submission.csv`` и логирует его в MLflow."""
    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    test_df[["essay_id", "score"]].to_csv(
        SUBMISSION_PATH / SUBMISSION_FILENAME, index=False
    )
    log.info(f"Saved {SUBMISSION_PATH / SUBMISSION_FILENAME}")
    mlflow.log_artifact(str(SUBMISSION_PATH / SUBMISSION_FILENAME))


def add_pred_and_score_columns(
    test_df: pd.DataFrame, predictions_list: list[np.ndarray]
) -> None:
    """
    Add the prediction and score columns to the test DataFrame.
    """
    best_ensemble_weights = np.load(
        BEST_ENSEMBLE_WEIGHTS_PATH / BEST_ENSEMBLE_WEIGHTS_FILENAME
    )
    test_df[[f"pred_{i}" for i in range(len(predictions_list))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(predictions_list))
    test_df["pred"] = np.sum(
        best_ensemble_weights
        * test_df[[f"pred_{i}" for i in range(len(predictions_list))]],
        axis=1,
    )
    test_df["score"] = get_essay_score(test_df["pred"].values) + 1


# ───────────────── main API ──────────────────────────────────────────── #
def make_submission_lightning(cfg) -> None:
    """
    Выполняет инференс всех моделей ансамбля на тесте и
    формирует финальный файл отправки.
    """
    # ---------- data and tokenizer ------------------------------------- #
    test_df = load_test()
    tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)

    # ---------- ensemble predictions ----------------------------------- #
    trainer = Trainer(devices="auto", accelerator="auto", logger=False)
    predictions_list = []

    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        test_ds = LALDataset(cfg_unit, test_df, tokenizer, is_train=False)
        test_dl = DataLoader(
            test_ds,
            batch_size=cfg.base.batch_size,
            shuffle=False,
            collate_fn=collate_infer,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )

        preds_folds = []
        for fold in range(cfg.n_folds):
            ckpt_path = fold_stage2_ckpt(Path(cfg_unit.path), model_idx, fold)
            model = EssayScoringPL.load_from_checkpoint(
                ckpt_path,
                cfg=cfg_unit,
                model_key=cfg_unit.model_key,
                load_from_existed=True,
            )
            preds = trainer.predict(model, test_dl)
            preds_folds.append(torch.cat(preds).numpy())
            del model
            torch.cuda.empty_cache()
            gc.collect()
        predictions_list.append(np.mean(preds_folds, axis=0))

    # ---------- blend weights (Nelder–Mead) ---------------------------- #
    add_pred_and_score_columns(test_df, predictions_list)

    # ---------- write submission --------------------------------------- #
    write_submission(test_df)
