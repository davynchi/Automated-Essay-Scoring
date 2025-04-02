import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ..common.common import LOGGER
from ..common.constants import DEVICE, NAMES_OF_MODELS
from ..common.dataset import LALDataset
from ..common.model import create_model
from .funcs_for_training_and_validating import train_fn


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_folds(folds, fold, cfg):
    if cfg.base.flag == 0:
        train_folds = folds[(folds["fold"] != fold) & ((folds["flag"] != 2))].reset_index(
            drop=True
        )
        valid_folds = folds[(folds["fold"] == fold) & (folds["flag"] != 2)].reset_index(
            drop=True
        )
        valid_folds2 = folds[(folds["fold"] == fold) & (folds["flag"] == 1)].reset_index(
            drop=True
        )
    else:
        train_folds = folds[(folds["fold"] != fold) & ((folds["flag"] == 1))].reset_index(
            drop=True
        )
        valid_folds = folds[(folds["fold"] == fold) & (folds["flag"] != 2)].reset_index(
            drop=True
        )
        valid_folds2 = folds[(folds["fold"] == fold) & (folds["flag"] == 0)].reset_index(
            drop=True
        )

    valid_folds = valid_folds.sort_values(["length", "essay_id"]).reset_index(drop=True)
    valid_folds2 = valid_folds2.sort_values(["length", "essay_id"]).reset_index(drop=True)

    return train_folds, valid_folds, valid_folds2


def create_dataloaders(train_folds, valid_folds, valid_folds2, cfg, tokenizer):
    train_dataset = LALDataset(cfg, train_folds, tokenizer, is_train=True)
    valid_dataset = LALDataset(cfg, valid_folds, tokenizer, is_train=True)
    valid_dataset2 = LALDataset(cfg, valid_folds2, tokenizer, is_train=True)

    dataloaders = []
    for i, dataset in enumerate([train_dataset, valid_dataset, valid_dataset2]):
        dataloaders.append(
            DataLoader(
                dataset,
                batch_size=cfg.base.batch_size,
                shuffle=True if i == 0 else False,
                num_workers=0,  # CFG.num_workers, -- Так было, потом поправь
                pin_memory=True,
                drop_last=True if i == 0 else False,
            )
        )

    return dataloaders[0], dataloaders[1], dataloaders[2]


def create_optimizer(cfg, model):
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=cfg.base.encoder_lr,
        decoder_lr=cfg.base.decoder_lr,
        weight_decay=cfg.base.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters,
        lr=cfg.base.encoder_lr,
        eps=cfg.base.eps,
        betas=cfg.base.betas,
    )

    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.base.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.base.num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif cfg.base.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.base.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=cfg.base.num_cycles,
        )
    return scheduler


def train_loop(folds, fold, cfg, checkpoints_names, tokenizer):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds, valid_folds, valid_folds2 = get_folds(folds, fold, cfg)
    # print(len(train_folds), len(valid_folds), len(valid_folds2))

    train_loader, valid_loader, valid_loader2 = create_dataloaders(
        train_folds, valid_folds, valid_folds2, cfg, tokenizer
    )

    assert checkpoints_names is not None
    model = create_model(cfg, fold, checkpoints_names)

    optimizer = create_optimizer(cfg, model)

    num_train_steps = int(len(train_folds) / cfg.base.batch_size * cfg.base.epochs)

    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = -np.inf

    valid_labels = valid_folds[cfg.base.target_cols2].values
    valid_labels2 = valid_folds2[cfg.base.target_cols2].values

    for epoch in range(cfg.base.epochs):
        if epoch < 3:
            best_score = train_fn(
                fold,
                train_loader,
                valid_loader,
                valid_labels,
                valid_loader2,
                valid_labels2,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                DEVICE,
                best_score,
                cfg,
            )

    predictions = torch.load(
        Path(cfg.path)
        / f"{NAMES_OF_MODELS[cfg.model_key].replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )["predictions"]
    valid_folds["pred"] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
