import gc
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ..common.constants import NAMES_OF_MODELS
from ..common.dataset import LALDataset
from ..common.model import create_model
from ..common.utils import LOGGER
from .funcs_for_training_and_validating import train_fn


def get_optimizer_params(
    model, encoder_lr: float, decoder_lr: float, weight_decay: float = 0.0
):
    """
    Returns optimizer parameter groups with separate learning rates for encoder and decoder parts.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Parameters for the encoder (inside model.model)
    encoder_params = [
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
    ]

    # Parameters for the decoder or additional parts (outside model.model)
    decoder_params = [
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": decoder_lr,
            "weight_decay": 0.0,
        }
    ]

    return encoder_params + decoder_params


def get_folds(folds, fold: int, cfg) -> Tuple:
    """
    Splits the dataset into training and two validation folds based on configuration flags.

    Args:
        folds: DataFrame containing fold and flag information.
        fold: The current fold number.
        cfg: Configuration object with base settings.

    Returns:
        Tuple of (train_folds, valid_folds, valid_folds2) DataFrames.
    """
    if cfg.base.flag == 0:
        train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
        valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
        valid_folds2 = folds[(folds["fold"] == fold) & (folds["flag"] == 1)].reset_index(
            drop=True
        )
    else:
        train_folds = folds[(folds["fold"] != fold) & (folds["flag"] == 1)].reset_index(
            drop=True
        )
        valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
        valid_folds2 = folds[(folds["fold"] == fold) & (folds["flag"] == 0)].reset_index(
            drop=True
        )

    valid_folds = valid_folds.sort_values(["length", "essay_id"]).reset_index(drop=True)
    valid_folds2 = valid_folds2.sort_values(["length", "essay_id"]).reset_index(drop=True)

    return train_folds, valid_folds, valid_folds2


def create_dataloaders(train_folds, valid_folds, valid_folds2, cfg, tokenizer):
    """
    Creates DataLoader objects for training and validation datasets.

    Args:
        train_folds, valid_folds, valid_folds2: DataFrames for respective datasets.
        cfg: Configuration object.
        tokenizer: Tokenizer for preprocessing text data.

    Returns:
        Tuple of DataLoaders: (train_loader, valid_loader, valid_loader2).
    """
    train_dataset = LALDataset(cfg, train_folds, tokenizer, is_train=True)
    valid_dataset = LALDataset(cfg, valid_folds, tokenizer, is_train=True)
    valid_dataset2 = LALDataset(cfg, valid_folds2, tokenizer, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.base.batch_size,
        shuffle=True,
        num_workers=0,  # Adjust to cfg.num_workers if needed
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.base.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader2 = DataLoader(
        valid_dataset2,
        batch_size=cfg.base.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader, valid_loader2


def create_optimizer(cfg, model):
    """
    Creates an AdamW optimizer with parameter groups for the encoder and decoder.

    Args:
        cfg: Configuration object.
        model: The model to be optimized.

    Returns:
        An AdamW optimizer instance.
    """
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


def get_scheduler(cfg, optimizer, num_train_steps: int):
    """
    Creates a learning rate scheduler based on the configuration.

    Args:
        cfg: Configuration object.
        optimizer: Optimizer instance.
        num_train_steps: Total number of training steps.

    Returns:
        A learning rate scheduler.
    """
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
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg.base.scheduler}")
    return scheduler


def train_loop(folds, fold: int, cfg, checkpoints_names, tokenizer):
    """
    Executes the main training loop for a single fold.

    Args:
        folds: DataFrame containing the dataset folds.
        fold: Current fold number.
        cfg: Configuration object.
        checkpoints_names: Model checkpoint names (must be provided).
        tokenizer: Tokenizer for data processing.

    Returns:
        DataFrame with validation predictions.
    """
    LOGGER.info(f"========== fold: {fold} training ==========")

    # Prepare folds and DataLoaders
    train_folds, valid_folds, valid_folds2 = get_folds(folds, fold, cfg)
    train_loader, valid_loader, valid_loader2 = create_dataloaders(
        train_folds, valid_folds, valid_folds2, cfg, tokenizer
    )

    if checkpoints_names is None:
        raise ValueError("checkpoints_names must be provided")
    model = create_model(cfg, fold, checkpoints_names)
    optimizer = create_optimizer(cfg, model)

    # Setup scheduler and loss function
    num_train_steps = int(len(train_folds) / cfg.base.batch_size * cfg.base.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    best_score = -np.inf

    valid_labels = valid_folds[cfg.base.target_cols2].values
    valid_labels2 = valid_folds2[cfg.base.target_cols2].values

    # Run training for each epoch (only training for first 3 epochs as in original logic)
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
                best_score,
                cfg,
            )

    # Load best model predictions
    model_file = (
        Path(cfg.path)
        / f"{NAMES_OF_MODELS[cfg.model_key].replace('/', '-')}_fold{fold}_best.pth"
    )
    predictions = torch.load(
        model_file, map_location=torch.device("cpu"), weights_only=False
    )["predictions"]
    valid_folds["pred"] = predictions

    # Cleanup GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
