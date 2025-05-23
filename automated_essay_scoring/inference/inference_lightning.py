import gc
import logging
from pathlib import Path

import dvc.api
import mlflow
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ..common.constants import (
    BEST_ENSEMBLE_WEIGHTS_FILENAME,
    BEST_ENSEMBLE_WEIGHTS_PATH,
    PATH_TO_TOKENIZER,
    RAW_DATA_PATH,
    SUBMISSION_FILENAME,
    SUBMISSION_PATH,
    TEST_FILENAME,
)
from ..common.utils import (
    create_tokenizer,
    find_latest_file,
    get_essay_score,
    modify_texts,
)
from ..dataset.dataset import LALDataset, collate
from ..model.lightning_modules import EssayScoringPL


log = logging.getLogger(__name__)


def load_test() -> pd.DataFrame:
    """Load and normalize the test dataset.

    Reads the test CSV file, applies
    text modifications in place.

    Returns:
        pd.DataFrame: Normalized test DataFrame.
    """
    with dvc.api.open(
        path=str(RAW_DATA_PATH / TEST_FILENAME),
        repo=".",
        rev="HEAD",
        mode="r",
    ) as fd:
        test = pd.read_csv(fd)
        modify_texts(test, "full_text")
        return test[:128]


def collate_infer(batch) -> dict[str, torch.Tensor]:
    """Collate function for inference: pad-trim sequences without labels.

    Performs a standard `default_collate` on the raw batch, then trims
    all tensors to the maximum actual sequence length via `collate`.

    Args:
        batch (Sequence[Any]): Raw batch as emitted by the Dataset.

    Returns:
        Dict[str, torch.Tensor]: Collated and length-trimmed input tensors.
    """
    raw_inputs = default_collate(batch)
    return collate(raw_inputs)


def write_submission(test_df: pd.DataFrame) -> None:
    """Write the submission CSV and log it to MLflow.

    Creates `SUBMISSION_PATH` if needed, writes `test_df[['essay_id','score']]`
    to `SUBMISSION_FILENAME` without index, and logs the file as an MLflow artifact.

    Args:
        test_df (pd.DataFrame): DataFrame containing 'essay_id' and 'score' columns.
    """
    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    output_file = SUBMISSION_PATH / SUBMISSION_FILENAME
    test_df[["essay_id", "score"]].to_csv(output_file, index=False)
    log.info(f"Saved submission to {output_file}")
    mlflow.log_artifact(str(output_file))


def add_pred_and_score_columns(
    test_df: pd.DataFrame, predictions_list: list[np.ndarray]
) -> None:
    """Add model predictions and final score columns to the test DataFrame.

    Loads the best ensemble weights from disk, constructs `pred_i` columns
    for each model in `predictions_list`, computes the weighted sum as `pred`,
    then applies `get_essay_score` and adds 1 to form the final `score`.

    Args:
        test_df (pd.DataFrame): Test DataFrame to augment with predictions.
        predictions_list (list[np.ndarray]): List of per-model prediction arrays.
    """
    best_weights = np.load(BEST_ENSEMBLE_WEIGHTS_PATH / BEST_ENSEMBLE_WEIGHTS_FILENAME)
    log.info(f"{best_weights}")
    # Create pred_i columns
    test_df[[f"pred_{i}" for i in range(len(predictions_list))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(predictions_list))
    test_df["pred"] = np.sum(
        best_weights * test_df[[f"pred_{i}" for i in range(len(predictions_list))]],
        axis=1,
    )
    test_df["score"] = get_essay_score(test_df["pred"].values) + 1


def make_submission_lightning(cfg) -> None:
    """Run inference on all ensemble models and produce the final submission.

    1. Load and normalize the test DataFrame via `load_test`.
    2. Instantiate the tokenizer from `PATH_TO_TOKENIZER`.
    3. For each model in `cfg.ensemble`:
       a. Create `LALDataset` and `DataLoader` with `collate_infer`.
       b. For each fold, load the corresponding stage-2 checkpoint and
          run `Trainer.predict` to collect fold predictions.
       c. Average fold predictions for each model.
    4. Blend all model predictions using precomputed ensemble weights.
    5. Write and log the submission CSV.

    Args:
        cfg: Configuration object with attributes:
            - `ensemble` (mapping of model configs, each with `path` and `model_key`)
            - `base.batch_size` (int) and `base.num_workers` (int)
            - `n_folds` (int) number of folds in the ensemble.
    """
    # Step 1: data and tokenizer
    test_df = load_test()
    tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)

    # Step 2: prepare Trainer and collect per-model predictions
    trainer = Trainer(devices="auto", accelerator="auto", logger=False)
    all_model_preds: list[np.ndarray] = []

    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        ds = LALDataset(cfg_unit, test_df, tokenizer, is_train=False)
        dl = DataLoader(
            ds,
            batch_size=cfg.base.batch_size,
            shuffle=False,
            collate_fn=collate_infer,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )

        fold_preds: list[np.ndarray] = []
        for fold in range(cfg.n_folds):
            ckpt = find_latest_file(
                Path(cfg_unit.path), pattern=f"model{model_idx}_fold{fold}_stage2.ckpt"
            )
            pl_module = EssayScoringPL.load_from_checkpoint(
                ckpt,
                cfg=cfg_unit,
                model_key=cfg_unit.model_key,
                load_from_existed=True,
            )
            preds = trainer.predict(pl_module, dl)
            fold_preds.append(torch.cat(preds).cpu().numpy())
            del pl_module
            torch.cuda.empty_cache()
            gc.collect()

        all_model_preds.append(np.mean(fold_preds, axis=0))

    # Step 3: blend and write submission
    add_pred_and_score_columns(test_df, all_model_preds)
    write_submission(test_df)
