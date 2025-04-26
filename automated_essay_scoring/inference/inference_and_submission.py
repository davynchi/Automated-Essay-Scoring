import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ..common.constants import (
    DATA_PATH,
    DEVICE,
    SUBMISSION_FILENAME,
    SUBMISSION_PATH,
    TEST_FILENAME,
)
from ..common.dataset import LALDataset, collate
from ..common.model import create_model
from ..common.modify_train_data import modify_texts, tokenize_text
from ..common.utils import LOGGER, get_essay_score
from .nelder_mead import calc_best_weights_for_ensemble


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    for inputs in tqdm(test_loader, total=len(test_loader)):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)

    del preds, inputs, k, v, y_preds, model
    torch.cuda.empty_cache()
    gc.collect()

    return predictions


def load_test_data():
    test = pd.read_csv(DATA_PATH / TEST_FILENAME)
    modify_texts(test["full_text"])
    return test


def submit_predictions(cfg, test, predictions_list, bestWght, submission):
    test[[f"pred_{j}" for j in range(len(cfg.ensemble))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(cfg.ensemble))
    test["pred"] = np.sum(
        bestWght * test[[f"pred_{j}" for j in range(len(cfg.ensemble))]], axis=1
    )
    test["score"] = get_essay_score(test["pred"].values) + 1
    submission = test[["essay_id", "score"]]
    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH / SUBMISSION_FILENAME, index=False)
    LOGGER.info(
        f"Predicted scores are saved in the file {SUBMISSION_PATH / SUBMISSION_FILENAME}"
    )


def make_submission(cfg):
    test = load_test_data()
    tokenizer = tokenize_text(test)
    bestWght = calc_best_weights_for_ensemble(cfg)
    predictions_list = []
    for cfg_unit in cfg.ensemble.values():
        test_dataset = LALDataset(cfg_unit, test, tokenizer, is_train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.base.batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
            num_workers=cfg.base.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        predictions = []
        for fold in range(cfg.n_folds):
            model = create_model(cfg_unit, fold, load_from_existed=True)
            prediction = inference_fn(test_loader, model, DEVICE)
            predictions.append(prediction)
            del model, prediction
            gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        predictions_list.append(predictions)

        del predictions, test_dataset, test_loader
        gc.collect()

    submit_predictions(cfg, test, predictions_list, bestWght)
