import gc

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ..common.cfg import CFG
from ..common.common import LOGGER
from ..common.constants import DEVICE, OUTPUT_DIR
from .dataset import TrainDataset
from .funcs_for_training_and_validating import train_fn
from .model import CustomModel_attention, CustomModel_lstm, CustomModel_mean_pooling


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


def get_folds(folds, fold):
    if CFG.flag == 0:
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


def create_dataloaders(train_folds, valid_folds, valid_folds2):
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)
    valid_dataset2 = TrainDataset(CFG, valid_folds2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader2 = DataLoader(
        valid_dataset2,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader, valid_loader2


def create_model(fold):
    if CFG.sl:
        if CFG.head == "mean_pooling":
            model = CustomModel_mean_pooling(CFG, config_path=None, pretrained=False)
        elif CFG.head == "attention":
            model = CustomModel_attention(CFG, config_path=None, pretrained=False)
        elif CFG.head == "lstm":
            model = CustomModel_lstm(CFG, config_path=None, pretrained=False)
        state = torch.load(
            "/content/" + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            map_location=torch.DEVICE("cpu"),
        )
        model.load_state_dict(state["model"])
    else:
        if CFG.head == "mean_pooling":
            model = CustomModel_mean_pooling(CFG, config_path=None, pretrained=True)
        elif CFG.head == "attention":
            model = CustomModel_attention(CFG, config_path=None, pretrained=True)
        elif CFG.head == "lstm":
            model = CustomModel_lstm(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR + "config.pth")
    model.to(DEVICE)

    return model


def create_optimizer(model):
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.decoder_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
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


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds, valid_folds, valid_folds2 = get_folds(folds, fold)

    train_loader, valid_loader, valid_loader2 = create_dataloaders(
        train_folds, valid_folds, valid_folds2
    )

    model = create_model(fold)

    optimizer = create_optimizer(model)

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)

    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = -np.inf

    valid_labels = valid_folds[CFG.target_cols2].values
    valid_labels2 = valid_folds2[CFG.target_cols2].values

    for epoch in range(CFG.epochs):
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
            )

    predictions = torch.load(
        OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.DEVICE("cpu"),
    )["predictions"]
    valid_folds["pred"] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
