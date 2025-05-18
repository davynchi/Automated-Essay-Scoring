import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ..common.constants import PATH_TO_TOKENIZER
from ..common.utils import create_tokenizer
from .dataset import LALDataset
from .dataset import collate as clip_to_max_len


def lightning_collate(batch):
    """Обрабатывает пачку данных для PyTorch Lightning DataLoader.

    Выполняет стандартный collate, паддинг до максимальной длины и преобразование меток в тензоры.

    Args:
        batch (Sequence[Tuple[Any, torch.Tensor, torch.Tensor]]):
            Список кортежей (inputs, y, y2), где
            inputs — данные (например, токены),
            y, y2 — два набора меток.

    Returns:
        Tuple[Any, torch.Tensor, torch.Tensor]:
            - inputs_processed: результат `default_collate` + `clip_to_max_len`
            - y_stack: тензор меток y формы (batch_size, ...)
            - y2_stack: тензор меток y2 формы (batch_size, ...)
    """
    inputs, y, y2 = zip(*batch, strict=True)
    inputs = default_collate(inputs)
    inputs = clip_to_max_len(inputs)
    return inputs, torch.stack(y), torch.stack(y2)


def get_folds(
    folds: pd.DataFrame, fold: int, will_eval_prompted_set: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбивает DataFrame с фолдами и флагами на обучающую и две валидационные выборки.

    В зависимости от `will_eval_prompted_set` выбирается, какие примеры попадут
    в обучающую и валидационные части.

    Args:
        folds (pd.DataFrame):
            Таблица с колонками "fold", "flag" и дополнительными полями (например, "length", "essay_id").
        fold (int):
            Номер текущего фолда для выделения валидационных данных.
        will_eval_prompted_set (bool):
            Флаг, указывающий, какой валидационный набор считать "prompted":
            - False: valid_folds2 = те же фолды & flag == 1
            - True:  valid_folds2 = те же фолды & flag == 0

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train_folds: объединённые фолды, не равные `fold` (и, при will_eval, ещё flag == 1)
            - valid_folds: фолды, равные `fold`
            - valid_folds2: subset valid_folds с учётом флага
    """
    if not will_eval_prompted_set:
        train = folds[folds["fold"] != fold].reset_index(drop=True)
        valid = folds[folds["fold"] == fold].reset_index(drop=True)
        valid2 = folds[(folds["fold"] == fold) & (folds["flag"] == 1)].reset_index(
            drop=True
        )
    else:
        train = folds[(folds["fold"] != fold) & (folds["flag"] == 1)].reset_index(
            drop=True
        )
        valid = folds[folds["fold"] == fold].reset_index(drop=True)
        valid2 = folds[(folds["fold"] == fold) & (folds["flag"] == 0)].reset_index(
            drop=True
        )

    valid = valid.sort_values(["length", "essay_id"]).reset_index(drop=True)
    valid2 = valid2.sort_values(["length", "essay_id"]).reset_index(drop=True)

    return train, valid, valid2


class EssayDataModule(pl.LightningDataModule):
    """LightningDataModule для обучения и валидации LAL-модели.

    Инкапсулирует логику разбиения на фолды, создание датасетов и DataLoader-ов.

    Attributes:
        cfg: Объект конфигурации (например, от Hydra), содержащий
             параметры batch_size, num_workers и т. д.
        df (pd.DataFrame): Таблица всех данных с колонками фолдов и флагов.
        fold (int): Номер текущего фолда для разделения выборок.
        eval_on_prompted (bool): Флаг, выбирающий стратегию формирования valid_folds2.
        tokenizer: Токенизатор, созданный через `create_tokenizer`.
        train_ds: Датасет для обучения (LALDataset).
        val_ds_1: Первый валидационный датасет (LALDataset).
        val_ds_2: Второй валидационный датасет (LALDataset).
    """

    def __init__(self, cfg, df: pd.DataFrame, fold: int, eval_on_prompted: bool):
        """Инициализирует модуль данными конфигурации и таблицей.

        Args:
            cfg: Конфигурация, должна содержать атрибуты `base.batch_size` и `base.num_workers`.
            df (pd.DataFrame): Исходный DataFrame с данными.
            fold (int): Индекс фолда для разделения (например, 0, 1, ..., n_folds-1).
            eval_on_prompted (bool): Стратегия выбора вторичного валидационного набора.
        """
        super().__init__()
        self.cfg = cfg
        self.tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        self.df = df
        self.fold = fold
        self.eval_on_prompted = eval_on_prompted

    def setup(self, stage: str | None = None):
        """Готовит датасеты для обучения и валидации.

        Осуществляет разбиение через `get_folds` и создаёт объекты LALDataset.

        Args:
            stage (str | None): Этап ("fit", "test" и т.п.), по умолчанию None.
        """
        tr, v1, v2 = get_folds(self.df, self.fold, self.eval_on_prompted)
        self.train_ds = LALDataset(self.cfg, tr, self.tokenizer, is_train=True)
        self.val_ds_1 = LALDataset(self.cfg, v1, self.tokenizer, is_train=True)
        self.val_ds_2 = LALDataset(self.cfg, v2, self.tokenizer, is_train=True)

    def train_dataloader(self) -> DataLoader:
        """Создаёт DataLoader для обучения.

        Returns:
            DataLoader: с `shuffle=True`, `drop_last=True` и коллэйтом `lightning_collate`.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.base.batch_size,
            shuffle=True,
            num_workers=self.cfg.base.num_workers,
            pin_memory=True,
            collate_fn=lightning_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        """Создаёт DataLoader-ы для валидации.

        Returns:
            list[DataLoader]: два DataLoader-а для `val_ds_1` и `val_ds_2`
            без перемешивания и без усечения.
        """
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
