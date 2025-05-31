import io
import logging
from pathlib import Path

import dvc.api
import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

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


log = logging.getLogger(__name__)


def load_test() -> pd.DataFrame:
    with dvc.api.open(
        path=str(RAW_DATA_PATH / TEST_FILENAME),
        repo=".",
        rev="HEAD",
        mode="rb",  # открыть бинарно
    ) as raw_fd:
        # оборачиваем в TextIOWrapper, чтобы ставить encoding и errors
        text_fd = io.TextIOWrapper(raw_fd, encoding="utf-8", errors="replace")
        df = pd.read_csv(text_fd, engine="python")
    modify_texts(df, "full_text")
    return df[:256]


def load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """
    Создаёт onnxruntime.InferenceSession с GPU (если доступно) и fallback на CPU.
    """
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def run_with_iobinding(
    session: ort.InferenceSession,
    io_binding,
    batch: dict[str, torch.Tensor],
    output_name,
    valid_inputs,
) -> np.ndarray:
    """Bind inputs on GPU and run without extra Python<>C++ copies."""

    # 1) bind all inputs directly from the batch
    for name, tensor in batch.items():
        if name not in valid_inputs:
            continue

        # make sure it lives on the GPU you’ll run on
        if tensor.device.type == "cpu":
            tensor = tensor.cuda(non_blocking=True)

        io_binding.bind_input(
            name,
            device_type="cuda",
            device_id=tensor.device.index if tensor.device.index is not None else 0,
            element_type=tensor.cpu().numpy().dtype,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    # 2) bind output; this one you can leave bound to CPU by default
    io_binding.bind_output(output_name)

    # 3) run
    session.run_with_iobinding(io_binding)

    # 4) pull the results back to host
    return io_binding.copy_outputs_to_cpu()[0]


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

    all_model_preds: list[np.ndarray] = []

    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        ds = LALDataset(cfg_unit, test_df, tokenizer, is_train=False)
        dl = DataLoader(
            ds,
            batch_size=cfg.base.infer_batch_size,
            shuffle=False,
            collate_fn=collate_infer,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )

        # — ONNX inference вместо LightningModule —
        fold_preds: list[np.ndarray] = []
        for fold in range(cfg.n_folds):
            # найдём onnx-файл, который вы экспортировали ранее
            onnx_file = find_latest_file(
                Path(cfg_unit.path), pattern=f"model{model_idx}_fold{fold}_*.onnx"
            )
            session = load_onnx_session(str(onnx_file))

            # 4. Берём имя выходного тензора на всякий случай
            io_binding = session.io_binding()
            output_name = session.get_outputs()[0].name
            valid_inputs = {i.name for i in session.get_inputs()}

            # 5. Прогоняем батчи через session.run
            preds_batches = []
            for batch in tqdm(dl):
                logits = run_with_iobinding(
                    session, io_binding, batch, output_name, valid_inputs
                )
                preds_batches.append(logits)

            # 6. Собираем fold_preds
            #      предположим, что logits.shape == (batch_size, 1) или (batch_size,)
            fold_pred = np.vstack(preds_batches).squeeze()
            fold_preds.append(fold_pred)

        all_model_preds.append(np.mean(fold_preds, axis=0))

    # Step 3: blend and write submission
    add_pred_and_score_columns(test_df, all_model_preds)
    write_submission(test_df)
