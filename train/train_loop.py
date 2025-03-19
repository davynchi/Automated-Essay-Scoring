import gc

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ..common.common import LOGGER
from ..common.constants import DEVICE
from ..common.dataset import LALDataset
from ..common.model import (
    CustomModel_attention,
    CustomModel_lstm,
    CustomModel_mean_pooling,
)
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
    if cfg.flag == 0:
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


def create_dataloaders(train_folds, valid_folds, valid_folds2, cfg):
    train_dataset = LALDataset(cfg, train_folds, is_train=True)
    valid_dataset = LALDataset(cfg, valid_folds, is_train=True)
    valid_dataset2 = LALDataset(cfg, valid_folds2, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # CFG.num_workers, -- Так было, потом поправь
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,  # CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader2 = DataLoader(
        valid_dataset2,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,  # CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader, valid_loader2


def create_model(fold, cfg):
    if cfg.sl:
        if cfg.head == "mean_pooling":
            model = CustomModel_mean_pooling(
                cfg, config_path=cfg.config_path, pretrained=False
            )
        elif cfg.head == "attention":
            model = CustomModel_attention(
                cfg, config_path=cfg.config_path, pretrained=False
            )
        elif cfg.head == "lstm":
            model = CustomModel_lstm(cfg, config_path=cfg.config_path, pretrained=False)
        state = torch.load(
            cfg.path + f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth",
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model.load_state_dict(state["model"])
    else:
        if cfg.head == "mean_pooling":
            model = CustomModel_mean_pooling(cfg, config_path=None, pretrained=True)
        elif cfg.head == "attention":
            model = CustomModel_attention(cfg, config_path=None, pretrained=True)
        elif cfg.head == "lstm":
            model = CustomModel_lstm(cfg, config_path=None, pretrained=True)
    torch.save(model.config, cfg.path / "config.pth")
    model.to(DEVICE)

    return model


def create_optimizer(model, cfg):
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=cfg.encoder_lr,
        decoder_lr=cfg.decoder_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas
    )

    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles,
        )
    return scheduler


def train_loop(folds, fold, cfg):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds, valid_folds, valid_folds2 = get_folds(folds, fold, cfg)
    print(len(train_folds), len(valid_folds), len(valid_folds2))

    train_loader, valid_loader, valid_loader2 = create_dataloaders(
        train_folds, valid_folds, valid_folds2, cfg
    )

    model = create_model(fold, cfg)

    optimizer = create_optimizer(model, cfg)

    num_train_steps = int(len(train_folds) / cfg.batch_size * cfg.epochs)

    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = -np.inf

    valid_labels = valid_folds[cfg.target_cols2].values
    valid_labels2 = valid_folds2[cfg.target_cols2].values

    for epoch in range(cfg.epochs):
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
        cfg.path / f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )["predictions"]
    valid_folds["pred"] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
