# inference_lightning.py
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .calc_ensemble_weights import calc_best_weights_for_ensemble
from .constants import DATA_PATH, SUBMISSION_FILENAME, SUBMISSION_PATH, TEST_FILENAME
from .dataset import LALDataset, collate
from .lightning_modules import EssayScoringPL
from .modify_train_data import modify_texts
from .utils import get_essay_score


log = logging.getLogger(__name__)


# ───────────────── helpers ───────────────────────────────────────────── #
def load_test():
    test = pd.read_csv(DATA_PATH / TEST_FILENAME)
    modify_texts(test["full_text"])
    return test[:128]


def fold_stage2_ckpt(model_dir: Path, model_idx: int, fold: int) -> str:
    pattern = f"model{model_idx}_fold{fold}_stage2.ckpt"
    matches = list(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No ckpt matches {pattern}")
    # usually only one file, but take the newest just in case
    return max(matches, key=lambda p: p.stat().st_mtime).as_posix()


# ---------- collate for inference ------------------------------------ #
def collate_infer(batch):
    """
    `batch` is list[dict[str, Tensor]] coming from LALDataset (is_train=False)
    1. default_collate merges the list into a dict[str, Tensor] with batch dim
    2. legacy `collate()` trims to max real sequence length in the batch
    """
    inputs = default_collate(batch)  # step-1
    return collate(inputs)  # step-2 (trim)


# ───────────────── main API ──────────────────────────────────────────── #
def make_submission_lightning(cfg, tokenizer):
    # ---------- data --------------------------------------------------- #
    test_df = load_test()

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
            )
            preds = trainer.predict(model, test_dl)
            preds_folds.append(torch.cat(preds).numpy())
            del model
            torch.cuda.empty_cache()
            gc.collect()
        predictions_list.append(np.mean(preds_folds, axis=0))

    # ---------- blend weights (Nelder–Mead) ---------------------------- #
    bestW = calc_best_weights_for_ensemble(cfg)
    test_df[[f"pred_{i}" for i in range(len(predictions_list))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(predictions_list))
    test_df["pred"] = np.sum(
        bestW * test_df[[f"pred_{i}" for i in range(len(predictions_list))]], axis=1
    )
    test_df["score"] = get_essay_score(test_df["pred"].values) + 1

    # ---------- write submission --------------------------------------- #
    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    test_df[["essay_id", "score"]].to_csv(
        SUBMISSION_PATH / SUBMISSION_FILENAME, index=False
    )
    log.info(f"Saved {SUBMISSION_PATH / SUBMISSION_FILENAME}")
