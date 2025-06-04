import logging

import mlflow
import numpy as np
import pandas as pd
import torch
import tritonclient.grpc as grpcclient
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ..common.constants import (
    BEST_ENSEMBLE_WEIGHTS_PATH,
    PATH_TO_TOKENIZER,
    RAW_DATA_PATH,
    SUBMISSION_PATH,
    TEST_FILENAME,
    TEST_SCORE_FILENAME,
)
from ..common.utils import create_tokenizer, get_essay_score, modify_texts, read_dataset
from ..dataset.dataset import LALDataset, collate


log = logging.getLogger(__name__)


def triton_infer_batch(
    model_name: str,
    client: grpcclient.InferenceServerClient,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """
    Send one batch of (input_ids, attention_mask) to Triton and return logits.
    """

    # 1) Build Triton Tensors for this batch
    in_ids = grpcclient.InferInput("input_ids", input_ids.shape, "INT64")
    in_mask = grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    in_ids.set_data_from_numpy(input_ids)
    in_mask.set_data_from_numpy(attention_mask)

    # 2) Request the “logits” output
    out_logits = grpcclient.InferRequestedOutput("logits")

    # 3) Call Triton
    response = client.infer(
        model_name=model_name,
        inputs=[in_ids, in_mask],
        outputs=[out_logits],
    )

    # 4) Extract logits as a NumPy array
    logits = response.as_numpy("logits")  # shape = (batch_size, 1) or (batch_size,)
    return logits.squeeze()


def collate_infer(batch, seq_len: int) -> dict[str, torch.Tensor]:
    """
    Collate a batch and then force‐pad/truncate everything to exactly `seq_len`.
    """
    raw = default_collate(batch)
    out = collate(raw)  # out["input_ids"].shape == (B, L) with L = max_len_in_this_batch

    B, L = out["input_ids"].shape
    if L < seq_len:
        pad_amt = seq_len - L
        pad_ids = torch.zeros(
            B, pad_amt, dtype=torch.int64, device=out["input_ids"].device
        )
        pad_mask = torch.zeros(
            B, pad_amt, dtype=torch.int64, device=out["attention_mask"].device
        )
        out["input_ids"] = torch.cat([out["input_ids"], pad_ids], dim=1)
        out["attention_mask"] = torch.cat([out["attention_mask"], pad_mask], dim=1)
    elif L > seq_len:
        out["input_ids"] = out["input_ids"][:, :seq_len]
        out["attention_mask"] = out["attention_mask"][:, :seq_len]

    return out


def write_submission(test_df: pd.DataFrame) -> None:
    """
    Exactly the same as inference_lightning:
    Write test_df[['essay_id','score']] → CSV + MLflow artifact.
    """
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df[["essay_id", "score"]].to_csv(SUBMISSION_PATH, index=False)
    log.info(f"Saved submission to {SUBMISSION_PATH}")
    mlflow.log_artifact(str(SUBMISSION_PATH))


def add_pred_and_score_columns(
    test_df: pd.DataFrame, predictions_list: list[np.ndarray]
) -> None:
    """
    Exactly the same as inference_lightning:
    Load best_ensemble_weights.npy, build pred_i columns, blend → final score.
    """
    best_weights = np.load(BEST_ENSEMBLE_WEIGHTS_PATH)
    # Create pred_i columns
    test_df[[f"pred_{i}" for i in range(len(predictions_list))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(predictions_list))
    test_df["pred"] = np.sum(
        best_weights * test_df[[f"pred_{i}" for i in range(len(predictions_list))]],
        axis=1,
    )
    test_df["score"] = get_essay_score(test_df["pred"].values) + 1


def get_score_on_test_data(test_df: pd.DataFrame) -> float:
    """
    Объединяет переданный test_df (essay_id, score_pred) с истинными оценками
    из RAW_DATA_PATH/"test"/TEST_SCORE_FILENAME, вычисляет Quadratic Weighted Kappa
    через get_score и логирует результат.

    Args:
        test_df (pd.DataFrame): DataFrame с колонками ['essay_id', 'score']
                                где 'score' — это предсказание модели.

    Returns:
        float: Значение QWK (квадратично-взвешенный каппа).
    """
    true_df: pd.DataFrame = read_dataset(RAW_DATA_PATH / "test" / TEST_SCORE_FILENAME)[
        :256
    ]
    true_df.rename(columns={"score": "score_true"}, inplace=True)

    pred_df = test_df.rename(columns={"score": "score_pred"})

    merged_df = pd.merge(
        true_df, pred_df, on="essay_id", how="inner", validate="one_to_one"
    )

    # Проверим, не потерялось ли что-либо при merge
    if merged_df.shape[0] != test_df.shape[0]:
        missing_ids = set(test_df["essay_id"]) - set(merged_df["essay_id"])
        logging.warning("При объединении пропущены строки с essay_id: %s", missing_ids)

    y_trues: np.ndarray = merged_df["score_true"].to_numpy(dtype=int)
    y_preds: np.ndarray = merged_df["score_pred"].to_numpy(dtype=float)
    final_score: float = cohen_kappa_score(y_trues, y_preds, weights="quadratic")

    logging.info(f"The final score is {final_score:.6f}")
    mlflow.log_metric("final_score", final_score)

    return final_score


def make_submission_triton(cfg, triton_url: str = "localhost:8001") -> None:
    """
    Mirror make_submission_lightning, but call Triton instead of ONNXSession:

    1) load_test()
    2) create tokenizer
    3) for each model_idx, cfg_unit:
       a) build LALDataset & DataLoader with collate_infer
       b) for each fold: call Triton sub‐model (model{idx}_fold{fold}_stage2)
          on every batch → collect logits → average folds
    4) blend model‐level outputs → final score
    5) write_submission()
    """
    # Step 1: data + tokenizer
    test_df = read_dataset(RAW_DATA_PATH / "test" / TEST_FILENAME)[:256]
    modify_texts(test_df, "full_text")
    tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)

    # Triton gRPC client
    client = grpcclient.InferenceServerClient(triton_url)

    all_model_preds: list[np.ndarray] = []

    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        seq_len = cfg_unit.max_len
        # a) Build a test‐set DataLoader just like before
        ds = LALDataset(cfg_unit, test_df, tokenizer, is_train=False)
        dl = DataLoader(
            ds,
            batch_size=cfg.base.infer_batch_size,
            shuffle=False,
            collate_fn=lambda batch, seq_len=seq_len: collate_infer(
                batch, seq_len=seq_len
            ),
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )

        fold_preds: list[np.ndarray] = []

        for fold in range(cfg.n_folds):
            # Triton model name for this fold
            triton_model_name = f"model{model_idx}_fold{fold}_stage2"

            # Run every batch through Triton
            batch_logits: list[np.ndarray] = []
            for batch in tqdm(dl, desc=f"Model {model_idx}, Fold {fold}"):
                # `batch` is a dict: { "input_ids": Tensor, "attention_mask": Tensor }
                input_ids = batch["input_ids"].numpy()
                attention_mask = batch["attention_mask"].numpy()

                logits = triton_infer_batch(
                    triton_model_name, client, input_ids, attention_mask
                )
                batch_logits.append(logits)

            # Stack all batches → shape (num_examples, )
            fold_pred = np.vstack(batch_logits).squeeze()
            fold_preds.append(fold_pred)

        # b) Average over folds → shape (num_examples, )
        all_model_preds.append(np.mean(fold_preds, axis=0))

    # Step 4: Blend & write
    add_pred_and_score_columns(test_df, all_model_preds)
    write_submission(test_df)
    get_score_on_test_data(test_df)
