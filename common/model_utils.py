import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

from ..common.constants import DEVICE
from .model import CustomModel_attention, CustomModel_lstm, CustomModel_mean_pooling


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


def get_essay_score(y_preds):
    return pd.cut(
        y_preds.reshape(-1) * 5,
        [-np.inf, 0.83333333, 1.66666667, 2.5, 3.33333333, 4.16666667, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)


def get_score(y_trues, y_preds):
    y_preds = get_essay_score(y_preds)
    score = cohen_kappa_score(y_trues, y_preds, weights="quadratic")
    return score
