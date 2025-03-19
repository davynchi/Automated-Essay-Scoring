import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ..common.cfg import CFG, CFG_LIST
from ..common.constants import DEVICE
from ..common.dataset import TestDataset, collate
from ..common.model import (
    CustomModel_attention,
    CustomModel_lstm,
    CustomModel_mean_pooling,
)
from ..common.modify_train_data import tokenize_text
from .data_loading import load_data
from .nelder_mead import calc_best_weights


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)

    del preds, tk0, inputs, k, v, y_preds, model
    torch.cuda.empty_cache()
    gc.collect()

    return predictions


def make_submission():
    CFG.sl = True
    test, submission = load_data()
    test = test[:300]
    tokenize_text(test, use_cfg_list=True)
    bestWght = calc_best_weights()
    predictions_list = []
    if len(test) > 10:
        for cfg in CFG_LIST:
            test_dataset = TestDataset(cfg, test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=DataCollatorWithPadding(
                    tokenizer=cfg.tokenizer, padding="longest"
                ),
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            predictions = []
            for fold in cfg.trn_fold:
                if cfg.head == "mean_pooling":
                    model = CustomModel_mean_pooling(
                        cfg, config_path=cfg.config_path, pretrained=False
                    )
                elif cfg.head == "attention":
                    model = CustomModel_attention(
                        cfg, config_path=cfg.config_path, pretrained=False
                    )
                elif cfg.head == "lstm":
                    model = CustomModel_lstm(
                        cfg, config_path=cfg.config_path, pretrained=False
                    )
                state = torch.load(
                    cfg.path + f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth",
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                model.load_state_dict(state["model"])
                prediction = inference_fn(test_loader, model, DEVICE)
                predictions.append(prediction)
                del model, state, prediction
                gc.collect()
                torch.cuda.empty_cache()
            predictions = np.mean(predictions, axis=0)
            predictions_list.append(predictions)

            del predictions, test_dataset, test_loader
            gc.collect()

        test[[f"pred_{j + 1}" for j in range(len(CFG_LIST))]] = np.array(
            predictions_list
        ).T.reshape(-1, len(CFG_LIST))
        test["pred"] = np.sum(
            bestWght * test[[f"pred_{j + 1}" for j in range(len(CFG_LIST))]], axis=1
        )
        test["score"] = (
            pd.cut(
                test["pred"].values.reshape(-1) * 5,
                [-np.inf, 0.83333333, 1.66666667, 2.5, 3.33333333, 4.16666667, np.inf],
                labels=[0, 1, 2, 3, 4, 5],
            ).astype(int)
            + 1
        )
        submission = submission.drop(columns=["score"]).merge(
            test[["essay_id"] + ["score"]], on="essay_id", how="left"
        )

    submission[["essay_id"] + ["score"]].to_csv("submission.csv", index=False)
