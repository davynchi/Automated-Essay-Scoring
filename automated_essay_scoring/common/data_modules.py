import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .constants import PATH_TO_TOKENIZER
from .dataset import LALDataset
from .dataset import collate as clip_to_max_len
from .modify_train_data import create_tokenizer


def lightning_collate(batch):
    """Collate‑функция для PL DataLoader: паддинг + конвертация меток."""
    inputs, y, y2 = zip(*batch, strict=True)  # un-zip list of tuples
    inputs = default_collate(inputs)
    inputs = clip_to_max_len(inputs)
    return inputs, torch.stack(y), torch.stack(y2)


def get_folds(
    folds: pd.DataFrame, fold: int, will_eval_prompted_set: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and two validation folds based on configuration flags.

    Args:
        folds: DataFrame containing fold and flag information.
        fold: The current fold number.
        cfg: Configuration object with base settings.

    Returns:
        Tuple of (train_folds, valid_folds, valid_folds2) DataFrames.
    """
    if not will_eval_prompted_set:
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


class EssayDataModule(pl.LightningDataModule):
    """
    LightningDataModule, инкапсулирующий логику split‑ов и DataLoader‑ов
    для каждого фолда и стадии.
    """

    def __init__(self, cfg, df: pd.DataFrame, fold: int, eval_on_prompted: bool):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        self.df = df
        self.fold = fold
        self.eval_on_prompted = eval_on_prompted

    def setup(self, stage=None):
        tr, v1, v2 = get_folds(self.df, self.fold, self.eval_on_prompted)
        self.train_ds = LALDataset(self.cfg, tr, self.tokenizer, is_train=True)
        self.val_ds_1 = LALDataset(self.cfg, v1, self.tokenizer, is_train=True)
        self.val_ds_2 = LALDataset(self.cfg, v2, self.tokenizer, is_train=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.base.batch_size,
            shuffle=True,
            num_workers=self.cfg.base.num_workers,
            pin_memory=True,
            collate_fn=lightning_collate,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_ds_1,
                batch_size=self.cfg.base.batch_size,
                shuffle=False,
                num_workers=self.cfg.base.num_workers,
                pin_memory=True,
                collate_fn=lightning_collate,
            ),
            DataLoader(
                self.val_ds_2,
                batch_size=self.cfg.base.batch_size,
                shuffle=False,
                num_workers=self.cfg.base.num_workers,
                pin_memory=True,
                collate_fn=lightning_collate,
            ),
        ]
