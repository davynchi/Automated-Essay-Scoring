import time
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .model import HEAD_CLASS_MAPPING
from .utils import get_score


class EssayScoringPL(L.LightningModule):
    """PyTorch LightningModule for training and inference of essay scoring.

    Wraps a core Transformer-based model head and handles training, validation,
    prediction steps, optimizer and scheduler configuration, and logging of metrics.

    Attributes:
        cfg: Configuration object with training hyperparameters.
        model_key: Identifier key for selecting the model head.
        core: The underlying model head instance from `HEAD_CLASS_MAPPING`.
        criterion: Loss function (`BCEWithLogitsLoss`).
        train_preds, train_trues: Lists accumulating predictions and labels during training.
        val1_preds, val1_trues: Lists for fold-A validation predictions and labels.
        val2_preds, val2_trues: Lists for fold-B validation predictions and labels.
    """

    def __init__(
        self,
        cfg,
        model_key: str,
        path_to_finetuned_models: Path | None = None,
        load_from_existed: bool = False,
    ):
        """Initialize the LightningModule.

        Args:
            cfg: Configuration object with attributes:
                - `base.encoder_lr`, `base.weight_decay`, `base.betas`, `base.eps`
                - `base.scheduler`, `base.num_warmup_steps`, `base.num_cycles`
            model_key (str): Key to select the appropriate head class.
            path_to_finetuned_models (Path | None): Path to pretrained weights, if loading.
            load_from_existed (bool): If True, load the core model from existing checkpoint.
        """
        super().__init__()
        self.cfg = cfg
        self.model_key = model_key
        self.save_hyperparameters(
            "cfg", "model_key", "path_to_finetuned_models", "load_from_existed"
        )

        # Instantiate core model head
        model_cls = HEAD_CLASS_MAPPING[cfg.head]
        self.core = model_cls(
            cfg,
            load_from_existed=load_from_existed,
            path_to_finetuned_models=path_to_finetuned_models,
            config_path=Path(cfg.path) / "config.pth",
            pretrained=not load_from_existed,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.train_preds, self.train_trues = [], []
        self.val1_preds, self.val1_trues = [], []
        self.val2_preds, self.val2_trues = [], []

    def forward(self, inputs):
        """Forward pass to compute raw model logits.

        Args:
            inputs: Dictionary or tensor batch of model inputs.

        Returns:
            torch.Tensor: Raw logits output by the core model.
        """
        return self.core(inputs)

    def configure_optimizers(self):
        """Configure optimizer and learning-rate scheduler.

        Creates separate parameter groups for encoder vs. decoder (no weight_decay on norms/bias),
        sets up AdamW, and attaches either a linear or cosine warmup scheduler.

        Returns:
            dict: Dictionary with `"optimizer"` and `"lr_scheduler"` for Lightning.

        Raises:
            ValueError: If `cfg.base.scheduler` is not one of `"linear"` or `"cosine"`.
        """
        encoder_params, decoder_params = [], []
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        for name, param in self.core.named_parameters():
            group = encoder_params if "model" in name else decoder_params
            wd = (
                self.cfg.base.weight_decay
                if not any(nd in name for nd in no_decay)
                else 0.0
            )
            group.append({"params": param, "weight_decay": wd})

        optimizer = AdamW(
            encoder_params + decoder_params,
            lr=self.cfg.base.encoder_lr,
            betas=tuple(self.cfg.base.betas),
            eps=self.cfg.base.eps,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler_type = self.cfg.base.scheduler.lower()
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.base.num_warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.base.num_warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.cfg.base.num_cycles,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.cfg.base.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self):
        """Hook: record start time of the training epoch."""
        self._epoch_start = time.time()

    def on_validation_epoch_start(self):
        """Hook: ensure `_epoch_start` exists for timing validation."""
        if not hasattr(self, "_epoch_start"):
            self._epoch_start = time.time()

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: Tuple of (inputs, labels, labels2).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value to backpropagate.
        """
        inputs, labels, labels2 = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.train_preds.append(logits.sigmoid().float().detach().cpu().numpy())
        self.train_trues.append(labels.cpu().numpy())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Hook: log the current learning rate after each training batch."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        """Hook: compute and log training QWK at epoch end, then clear buffers."""
        preds = np.concatenate(self.train_preds, axis=0)
        trues = np.concatenate(self.train_trues, axis=0)
        self.log("train_qwk", get_score(trues, preds), prog_bar=True)
        self.train_preds.clear()
        self.train_trues.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single validation step for one of the two validation loaders.

        Args:
            batch: Tuple of (inputs, labels, labels2).
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): 0 for first validation set, 1 for second.
        """
        inputs, labels, labels2 = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels2)
        self.log(f"val{dataloader_idx}_loss", loss, prog_bar=True)

        preds = logits.sigmoid().float().detach().cpu().numpy()
        if dataloader_idx == 0:
            self.val1_preds.append(preds)
            self.val1_trues.append(labels.cpu().numpy())
        else:
            self.val2_preds.append(preds)
            self.val2_trues.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        """Hook: compute and log QWK for both validation sets and epoch duration."""
        if self.val1_preds:
            p1 = np.concatenate(self.val1_preds, axis=0)
            t1 = np.concatenate(self.val1_trues, axis=0)
            self.log("val1_qwk", get_score(t1, p1), prog_bar=True)
        if self.val2_preds:
            p2 = np.concatenate(self.val2_preds, axis=0)
            t2 = np.concatenate(self.val2_trues, axis=0)
            self.log("val2_qwk", get_score(t2, p2), prog_bar=True)

        # Clear buffers
        self.val1_preds.clear()
        self.val1_trues.clear()
        self.val2_preds.clear()
        self.val2_trues.clear()

        # Log epoch time
        epoch_time = time.time() - self._epoch_start
        self.log("epoch_time", epoch_time, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform inference step, returning sigmoid probabilities on CPU.

        Accepts either a batch of inputs or a tuple (inputs, labels, labels2).

        Args:
            batch: Inputs or tuple with labels.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): (unused) index of the dataloader.

        Returns:
            torch.Tensor: Sigmoid-activated logits on CPU.
        """
        # Drop labels if present
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        preds = self(inputs).sigmoid().float().cpu()
        return preds
