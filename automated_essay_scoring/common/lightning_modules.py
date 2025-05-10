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
    def __init__(self, cfg, model_key, checkpoints_names=None, load_from_existed=False):
        super().__init__()
        self.cfg = cfg
        self.model_key = model_key
        self.save_hyperparameters()  # stores cfg in checkpoint

        model_cls = HEAD_CLASS_MAPPING[cfg.head]
        self.core = model_cls(
            cfg,
            load_from_existed=load_from_existed,
            checkpoints_names=checkpoints_names,
            config_path=Path(cfg.path) / "config.pth",
            pretrained=not load_from_existed,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.train_preds, self.train_trues = [], []
        self.val1_preds, self.val1_trues = [], []
        self.val2_preds, self.val2_trues = [], []

    # ── forward ────────────────────────────────────────────────────────────────── #
    def forward(self, inputs):
        return self.core(inputs)

    # ── optimizer & scheduler ──────────────────────────────────────────────────── #
    def configure_optimizers(self):
        encoder_params, decoder_params = [], []
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        for n, p in self.core.named_parameters():
            group = encoder_params if "model" in n else decoder_params
            wd = (
                self.cfg.base.weight_decay if not any(nd in n for nd in no_decay) else 0.0
            )
            group.append({"params": p, "weight_decay": wd})

        optimizer = AdamW(
            encoder_params + decoder_params,
            lr=self.cfg.base.encoder_lr,
            betas=tuple(self.cfg.base.betas),
            eps=self.cfg.base.eps,
        )
        total = self.trainer.estimated_stepping_batches
        if self.cfg.base.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.base.num_warmup_steps,
                num_training_steps=total,
            )
        elif self.cfg.base.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.base.num_warmup_steps,
                num_training_steps=total,
                num_cycles=self.cfg.base.num_cycles,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.cfg.base.scheduler}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── training ───────────────────────────────────────────────────────────────── #
    def on_train_epoch_start(self):
        self._epoch_start = time.time()

    def on_validation_epoch_start(self):
        # guarantees attribute exists for the sanity check
        if not hasattr(self, "_epoch_start"):
            self._epoch_start = time.time()

    def training_step(self, batch, batch_idx):
        inputs, labels, labels2 = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels2)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_preds.append(outputs.sigmoid().float().detach().cpu().numpy())
        self.train_trues.append(labels.cpu().numpy())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        p = np.concatenate(self.train_preds)
        t = np.concatenate(self.train_trues)
        score = get_score(t, p)
        self.log("train_qwk", score, prog_bar=True)
        self.train_preds.clear()
        self.train_trues.clear()

    # ── validation (loader 0 = A, loader 1 = B) ────────────────────────────────── #
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels, labels2 = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels2)
        self.log(f"val{dataloader_idx}_loss", loss, prog_bar=True)
        preds = outputs.sigmoid().float().detach().cpu().numpy()
        if dataloader_idx == 0:
            self.val1_preds.append(preds)
            self.val1_trues.append(labels.cpu().numpy())
        else:
            self.val2_preds.append(preds)
            self.val2_trues.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        p1 = np.concatenate(self.val1_preds) if self.val1_preds else np.empty((0, 1))
        t1 = np.concatenate(self.val1_trues) if self.val1_trues else np.empty((0, 1))
        p2 = np.concatenate(self.val2_preds) if self.val2_preds else np.empty((0, 1))
        t2 = np.concatenate(self.val2_trues) if self.val2_trues else np.empty((0, 1))
        if p1.size:
            self.log("val1_qwk", get_score(t1, p1), prog_bar=True)
        if p2.size:
            self.log("val2_qwk", get_score(t2, p2), prog_bar=True)
        self.val1_preds.clear()
        self.val1_trues.clear()
        self.val2_preds.clear()
        self.val2_trues.clear()
        self.log("epoch_time", time.time() - self._epoch_start, logger=True)

    # ── prediction ─────────────────────────────────────────────────── #
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Called by `Trainer.predict()`.
        Accept both (inputs, label, label2) *or* inputs-only.
        Returns sigmoid logits on CPU so they can be concatenated.
        """
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]  # drop labels
        else:
            inputs = batch
        preds = self(inputs).sigmoid().float().cpu()
        return preds
