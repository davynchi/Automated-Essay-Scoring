# train_lightning.py
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


# ---------------------------------------------------------------------------- #
def build_pred_dl(dataset, cfg) -> DataLoader:
    """
    Создаёт DataLoader, который возвращает **только** словари ``inputs`` —
    без меток — для этапа *predict()*.
    """

    def collate_infer(batch):
        inputs_dicts = [b[0] for b in batch]  # drop labels
        return collate(default_collate(inputs_dicts))

    return DataLoader(
        dataset,
        batch_size=cfg.base.batch_size,
        shuffle=False,
        collate_fn=collate_infer,
        num_workers=cfg.base.num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------- #
def purge_stage1_checkpoints(model_dir: Path) -> None:
    """Удаляет все чек‑пойнты вида ``*_stage1.ckpt`` в указанной папке."""
    for p in model_dir.glob("*_stage1.ckpt"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------- #
def free_trainer(trainer: L.Trainer, *objs) -> None:
    """
    Аккуратно освобождает память после завершения обучения:
    вызов teardown, очистка state оптимизаторов, ``torch.cuda.empty_cache`` и т.д.
    """
    # teardown (PL 2.4+) or fallback
    if hasattr(trainer, "_teardown"):
        trainer._teardown()
    elif hasattr(trainer, "_call_teardown_hook"):
        trainer._call_teardown_hook()

    for opt in trainer.optimizers:
        opt.state.clear()

    del trainer
    for o in objs:
        del o

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


# ---------------------------------------------------------------------------- #
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
    """
    Обучает **один** фолд на одной стадии (stage‑1 или stage‑2).

    Параметры
    ---------
    eval_on_prompted : bool
        * False — тренируемся на A+B, валидируемся на B (flag 1).
        * True  — тренируемся только на B, валидируемся на A (flag 0).
    init_ckpt : str | None
        Путь к чек‑пойнту stage‑1, с которого продолжаем stage‑2.
    path_to_finetuned_models : str | None
        Если задано, берём дообученные веса DeBERTa из этой папки,
        иначе из ``OUTPUT_DIR_FINETUNED``.
    """
    # -------- datamodule -------------------------------------------------- #
    dm = EssayDataModule(cfg_unit, train_df, fold, eval_on_prompted)
    dm.setup()

    # -------- callbacks / logger ------------------------------------------ #
    tag = f"model{model_idx}_fold{fold}_{stage_tag}"
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg_unit.path,
        filename=f"{tag}",
        monitor="val2_qwk",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    logger = MLFlowLogger(experiment_name="essay-scoring-pipeline", run_name=tag)

    # -------- trainer ------------------------------------------------------ #
    trainer = L.Trainer(
        max_epochs=cfg.base.epochs,
        precision=16 if cfg.base.apex else 32,
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=cfg.base.gradient_accumulation_steps,
        deterministic=False,
        gradient_clip_val=cfg.base.max_grad_norm,
        callbacks=[ckpt_cb],
        logger=logger,
    )

    # -------- model -------------------------------------------------------- #
    ckpt_path = None
    if isinstance(init_ckpt, str):
        ckpt_path = init_ckpt
    elif callable(init_ckpt):
        ckpt_path = init_ckpt(fold)  # fold-specific path

    if ckpt_path:
        # weights-only: load them into the module, but DON'T give ckpt_path to fit()
        model = EssayScoringPL.load_from_checkpoint(
            ckpt_path, cfg=cfg_unit, model_key=cfg_unit.model_key, load_from_existed=True
        )
    else:
        model = EssayScoringPL(
            cfg=cfg_unit,
            model_key=cfg_unit.model_key,
            load_from_existed=False,
            path_to_finetuned_models=path_to_finetuned_models,
        )

    trainer.fit(model=model, datamodule=dm)

    # -------- OOF predictions & pickle ------------------------------------ #
    pred_dl = build_pred_dl(dm.val_ds_1, cfg)  # same split we monitor
    pred_chunks = trainer.predict(model, dataloaders=pred_dl)  # list[Tensor]
    preds = torch.cat(pred_chunks).squeeze().cpu().numpy()

    oof_df = dm.val_ds_1.df.copy().reset_index(drop=True)
    oof_df["pred"] = preds
    oof_df.to_pickle(Path(cfg_unit.path) / f"oof_fold{fold}.pkl")

    # -------------- MEMORY CLEANUP  ---------------------------------------- #
    free_trainer(trainer, model, pred_dl, preds, oof_df)


# ---------------------------------------------------------------------------- #
def train_model_lightning(cfg, path_to_finetuned_models: str | None = None) -> None:
    """
    Верхнеуровневый цикл по *моделям* и *фолдам*:
    для каждого (model_i, fold_j) выполняет подряд stage‑1 → stage‑2,
    собирает OOF‑предсказания и чистит stage‑1 чек‑пойнты.
    """
    for model_idx, cfg_unit in enumerate(cfg.ensemble.values()):
        for fold in range(cfg.n_folds):
            # ---------- Stage-1  (A+B, monitor B) --------------------------------- #
            train_df_stage1 = load_pickle_data(cfg_unit, load_from_existed_pickle=False)
            tokenize_text(train_df_stage1)  # populate 'length'
            train_one_stage(
                cfg,
                cfg_unit,
                model_idx,
                fold,
                train_df_stage1,
                eval_on_prompted=False,
                stage_tag="stage1",
                init_ckpt=None,
                path_to_finetuned_models=path_to_finetuned_models,
            )

            # ---------- locate best ckpt per fold --------------------------------- #
            s1_ckpt_pattern = f"model{model_idx}_fold{fold}_stage1.ckpt"
            s1_ckpt = max(
                Path(cfg_unit.path).glob(s1_ckpt_pattern),
                key=lambda p: p.stat().st_mtime,
            ).as_posix()

            # ---------- Stage-2  (B-only, monitor A) ------------------------------ #
            train_df_stage2 = load_pickle_data(cfg_unit, load_from_existed_pickle=True)
            tokenize_text(train_df_stage2)
            train_one_stage(
                cfg,
                cfg_unit,
                model_idx,
                fold,
                train_df_stage2,
                eval_on_prompted=True,
                stage_tag="stage2",
                init_ckpt=s1_ckpt,
                path_to_finetuned_models=path_to_finetuned_models,
            )

            # ------------- purge Stage‑1 ckpts (they’re no longer needed) ----------
            purge_stage1_checkpoints(Path(cfg_unit.path))
