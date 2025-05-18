import gc
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .data_modules import EssayDataModule
from .dataset import collate
from .lightning_modules import EssayScoringPL
from .modify_train_data import load_pickle_data, tokenize_text


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


def purge_stage1_checkpoints(model_dir: Path) -> None:
    """Remove all stage-1 checkpoint files from a directory.

    Args:
        model_dir (Path): Path to the directory containing checkpoints.
    """
    for ckpt in model_dir.glob("*_stage1.ckpt"):
        try:
            ckpt.unlink()
        except FileNotFoundError:
            pass


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


def train_one_stage(
    cfg,
    cfg_unit,
    model_idx: int,
    fold: int,
    train_df: pd.DataFrame,
    eval_on_prompted: bool,
    stage_tag: str,
    init_ckpt: str | None = None,
    path_to_finetuned_models: str | None = None,
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
    run_tag = f"model{model_idx}_fold{fold}_{stage_tag}"
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
    elif callable(init_ckpt):
        ckpt_path = init_ckpt(fold)

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

    # Cleanup
    free_trainer(trainer, model, pred_dl, preds, oof_df)


def train_model_lightning(cfg, path_to_finetuned_models: str | None = None) -> None:
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
                stage_tag="stage1",
                init_ckpt=None,
                path_to_finetuned_models=path_to_finetuned_models,
            )

            # Locate best stage-1 checkpoint
            pattern = f"model{model_idx}_fold{fold}_stage1.ckpt"
            s1_ckpt = max(
                Path(cfg_unit.path).glob(pattern),
                key=lambda p: p.stat().st_mtime,
            ).as_posix()

            # Stage-2
            df_s2 = load_pickle_data(cfg_unit, load_from_existed_pickle=True)
            tokenize_text(df_s2)
            train_one_stage(
                cfg,
                cfg_unit,
                model_idx,
                fold,
                df_s2,
                eval_on_prompted=True,
                stage_tag="stage2",
                init_ckpt=s1_ckpt,
                path_to_finetuned_models=path_to_finetuned_models,
            )

            # Purge stage-1 ckpts
            purge_stage1_checkpoints(Path(cfg_unit.path))
