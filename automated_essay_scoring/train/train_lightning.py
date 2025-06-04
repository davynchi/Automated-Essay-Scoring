import gc
import shutil
from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
import torch.onnx
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import DebertaTokenizer

from ..common.constants import (
    CACHED_DATA_DIR,
    OUTPUT_DIR_FINETUNED,
    OUTPUT_DIR_TRAIN,
    PATH_TO_TOKENIZER,
    TRAIN_PICKLE_PATH,
    TRITON_MODELS_PATH,
)
from ..common.utils import create_tokenizer, get_checkpoint_name, remove_files
from ..dataset.data_modules import EssayDataModule
from ..dataset.dataset import collate
from ..model.lightning_modules import EssayScoringPL
from .prepare_to_infer import add_model_to_triton, convert_to_tensorrt, export_to_onnx


def tokenize_text(
    data: pd.DataFrame, path_to_tokenizer: str = PATH_TO_TOKENIZER
) -> DebertaTokenizer:
    """Compute token lengths and sort DataFrame by length for efficient padding.

    Adds a `length` column and sorts `data` in-place by ascending token count.

    Args:
        data (pd.DataFrame): DataFrame with `full_text` column.
        path_to_tokenizer (str): Tokenizer path or identifier.

    Returns:
        DebertaTokenizer: Tokenizer used for encoding.
    """
    tokenizer = create_tokenizer(path=path_to_tokenizer)

    def _encode(text: str) -> int:
        return len(tokenizer.encode(text))

    data["length"] = data["full_text"].map(_encode)
    data.sort_values("length", ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return tokenizer


def load_pickle_data(cfg_unit, load_from_existed_pickle: bool) -> pd.DataFrame:
    """Load or blend training DataFrame for a given ensemble member.

    In Stage-1 (load_from_existed_pickle=False), adds column `score_s` as `score / 5`.
    In Stage-2 (load_from_existed_pickle=True), merges model's OOF predictions and
    blends with the true scores using `cfg_unit.base.sl_rate` for self-learning.

    Args:
        cfg_unit: Configuration object for the ensemble member, must have `path`,
            `base.target_cols`, `base.modif_target_cols`, and `base.sl_rate`.
        load_from_existed_pickle (bool): Whether to perform Stage-2 blending.

    Returns:
        pd.DataFrame: DataFrame containing columns `essay_id`, `text`, `score`,
            `flag`, `fold`, and `score_s` (modified target column).

    Raises:
        FileNotFoundError: If expected OOF pickle files are not found in Stage-2.
    """
    train = pd.read_pickle(TRAIN_PICKLE_PATH)

    if load_from_existed_pickle:
        oof_paths = sorted(glob(str(Path(cfg_unit.path) / "oof_fold*.pkl")))
        if not oof_paths:
            # No OOF files: fallback to Stage-1 behavior
            train[cfg_unit.base.modif_target_cols[0]] = (
                train[cfg_unit.base.target_cols[0]].values / 5
            )
            return train

        oof_list = [pd.read_pickle(p) for p in oof_paths]
        oof = pd.concat(oof_list, ignore_index=True)

        # Merge and blend
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[cfg_unit.base.modif_target_cols[0]] = (
            (train[cfg_unit.base.target_cols[0]].values / 5) * (1 - cfg_unit.base.sl_rate)
        ) + (train["pred"].fillna(0).values * cfg_unit.base.sl_rate)
    else:
        train[cfg_unit.base.modif_target_cols[0]] = (
            train[cfg_unit.base.target_cols[0]].values / 5
        )

    return train


def build_pred_dl(dataset, cfg) -> DataLoader:
    """Create a DataLoader for inference that returns only input dicts.

    This DataLoader drops labels and applies the `collate` function to pad/trim inputs.

    Args:
        dataset: Dataset yielding tuples `(inputs, label?, label2?)`.
        cfg: Configuration object with attributes `base.batch_size` and `base.num_workers`.

    Returns:
        DataLoader: Iterable over batches of input dictionaries.
    """

    def collate_infer(batch):
        inputs_list = [b[0] for b in batch]
        batch_inputs = default_collate(inputs_list)
        return collate(batch_inputs)

    return DataLoader(
        dataset,
        batch_size=cfg.base.batch_size,
        shuffle=False,
        collate_fn=collate_infer,
        num_workers=cfg.base.num_workers,
        pin_memory=True,
    )


def free_trainer(trainer: L.Trainer, *objs) -> None:
    """Clean up trainer and free GPU memory after training.

    Calls teardown hooks, clears optimizer state, deletes objects,
    empties CUDA cache, and runs garbage collection.

    Args:
        trainer (L.Trainer): The PyTorch Lightning Trainer to clean up.
        *objs: Additional objects to delete (e.g., model, dataloader).
    """
    # Teardown Lightning internals
    if hasattr(trainer, "_teardown"):
        trainer._teardown()
    elif hasattr(trainer, "_call_teardown_hook"):
        trainer._call_teardown_hook()

    # Clear optimizer states
    for optimizer in trainer.optimizers:
        optimizer.state.clear()

    # Delete references
    del trainer
    for obj in objs:
        del obj

    # Free GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def delete_cached_model_data():
    for path in [OUTPUT_DIR_FINETUNED, CACHED_DATA_DIR, OUTPUT_DIR_TRAIN]:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def train_one_stage(
    cfg,
    cfg_unit,
    model_idx: int,
    fold: int,
    train_df: pd.DataFrame,
    eval_on_prompted: bool,
    stage_idx: int,
    init_ckpt: str | None = None,
    path_to_finetuned_models: str | None = None,
    prepare_to_infer: bool = False,
    skip_converting_to_tensorrt: bool = False,
) -> None:
    """Train a single fold for one stage (stage-1 or stage-2).

    Args:
        cfg: Global configuration object.
        cfg_unit: Ensemble-member configuration.
        model_idx (int): Index of the model in the ensemble.
        fold (int): Fold number for cross-validation.
        train_df (pd.DataFrame): DataFrame of training examples for this fold.
        eval_on_prompted (bool): If False, train on all examples and validate on flagged set;
            if True, train on flagged only and validate on unflagged.
        stage_tag (str): Identifier tag for this stage, e.g., "stage1" or "stage2".
        init_ckpt (str | None): Path to an existing checkpoint to initialize training.
        path_to_finetuned_models (str | None): Directory of pretrained weights for core model.
    """
    # Prepare DataModule
    dm = EssayDataModule(cfg_unit, train_df, fold, eval_on_prompted)
    dm.setup()

    # Callbacks and Logger
    run_tag = get_checkpoint_name(model_idx, fold, stage_idx)
    ckpt_callback = ModelCheckpoint(
        dirpath=cfg_unit.path,
        filename=run_tag,
        monitor="val2_qwk",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    mlflow_logger = MLFlowLogger(
        experiment_name="essay-scoring-pipeline",
        run_name=run_tag,
    )

    # Instantiate Trainer
    trainer = L.Trainer(
        max_epochs=cfg.base.epochs,
        precision="16-mixed" if cfg.base.apex else 32,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=cfg.base.gradient_accumulation_steps,
        deterministic=False,
        gradient_clip_val=cfg.base.max_grad_norm,
        callbacks=[ckpt_callback],
        logger=mlflow_logger,
    )

    # Initialize or load model
    ckpt_path = None
    if isinstance(init_ckpt, str):
        ckpt_path = init_ckpt

    if ckpt_path:
        model = EssayScoringPL.load_from_checkpoint(
            ckpt_path,
            cfg=cfg_unit,
            model_key=cfg_unit.model_key,
            load_from_existed=True,
        )
    else:
        model = EssayScoringPL(
            cfg=cfg_unit,
            model_key=cfg_unit.model_key,
            load_from_existed=False,
            path_to_finetuned_models=path_to_finetuned_models,
        )

    # Fit model
    trainer.fit(model=model, datamodule=dm)

    # Generate OOF predictions and save pickle
    pred_dl = build_pred_dl(dm.val_ds_1, cfg)
    pred_batches = trainer.predict(model, dataloaders=pred_dl)
    preds = torch.cat(pred_batches).squeeze().cpu().numpy()
    oof_df = dm.val_ds_1.df.reset_index(drop=True).copy()
    oof_df["pred"] = preds
    oof_df.to_pickle(Path(cfg_unit.path) / f"oof_fold{fold}.pkl")

    if prepare_to_infer:
        onnx_path = export_to_onnx(cfg_unit, model, pred_dl, onnx_name=run_tag)

        if not skip_converting_to_tensorrt:
            plan_path = convert_to_tensorrt(
                onnx_path,
                batch_max=cfg.base.infer_batch_size,
                seq_len=cfg_unit.max_len,
                fp16=True,
                int8=False,
                workspace_mb=4096,
            )
            artefact_path = Path(plan_path)
        else:
            artefact_path = onnx_path

        add_model_to_triton(
            artefact_path,
            seq_len=cfg_unit.max_len,
            mbatch=cfg.base.infer_batch_size,
            model_idx=model_idx,
            fold=fold,
        )

    # Cleanup
    free_trainer(trainer, model, pred_dl, preds, oof_df)


def train_model_lightning(cfg, skip_converting_to_tensorrt) -> None:
    """Run full training pipeline over all ensemble members and folds.

    For each model index and fold:
      1. Stage-1: train on all data + monitor on flagged subset.
      2. Stage-2: train on flagged only + monitor on unflagged, initializing from stage-1.
      3. Purge stage-1 checkpoints after stage-2 completes.

    Args:
        cfg: Global configuration object, containing `ensemble`, `n_folds`, and `base` settings.
        path_to_finetuned_models (str | None): Directory of pretrained core-model weights.
    """
    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        for fold in range(cfg.n_folds):
            # Stage-1
            df_s1 = load_pickle_data(cfg_unit, load_from_existed_pickle=False)
            tokenize_text(df_s1)
            train_one_stage(
                cfg,
                cfg_unit,
                model_idx,
                fold,
                df_s1,
                eval_on_prompted=False,
                stage_idx=1,
                init_ckpt=None,
                path_to_finetuned_models=OUTPUT_DIR_FINETUNED,
                prepare_to_infer=False,
            )

            # Locate best stage-1 checkpoint
            s1_ckpt = (
                (Path(cfg_unit.path) / get_checkpoint_name(model_idx, fold, 1))
                .with_suffix(".ckpt")
                .as_posix()
            )

            # Stage-2
            df_s2 = load_pickle_data(cfg_unit, load_from_existed_pickle=True)
            tokenize_text(df_s2)
            TRITON_MODELS_PATH.mkdir(parents=True, exist_ok=True)
            train_one_stage(
                cfg,
                cfg_unit,
                model_idx,
                fold,
                df_s2,
                eval_on_prompted=True,
                stage_idx=2,
                init_ckpt=s1_ckpt,
                path_to_finetuned_models=OUTPUT_DIR_FINETUNED,
                prepare_to_infer=True,
                skip_converting_to_tensorrt=skip_converting_to_tensorrt,
            )

            # Purge stage ckpts
            remove_files(Path(cfg_unit.path), pattern="*.ckpt")
