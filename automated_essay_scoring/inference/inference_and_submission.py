import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from ..common.common import LOGGER
from ..common.constants import DEVICE, SUBMISSION_FILENAME, SUBMISSION_PATH
from ..common.dataset import LALDataset, collate
from ..common.model import create_model
from ..common.modify_train_data import load_test_submission_data, tokenize_text
from ..common.utils import get_essay_score
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


def make_submission(cfg):
    test, submission = load_test_submission_data()
    test = test[:300]
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
        for fold in cfg.base.trn_fold:
            model = create_model(cfg_unit, fold)
            prediction = inference_fn(test_loader, model, DEVICE)
            predictions.append(prediction)
            del model, prediction
            gc.collect()
            torch.cuda.empty_cache()
        predictions = np.mean(predictions, axis=0)
        predictions_list.append(predictions)

        del predictions, test_dataset, test_loader
        gc.collect()

    test[[f"pred_{j + 1}" for j in range(len(cfg.ensemble))]] = np.array(
        predictions_list
    ).T.reshape(-1, len(cfg.ensemble))
    test["pred"] = np.sum(
        bestWght * test[[f"pred_{j + 1}" for j in range(len(cfg.ensemble))]], axis=1
    )
    test["score"] = get_essay_score(test["pred"].values) + 1
    submission = submission.drop(columns=["score"]).merge(
        test[["essay_id", "score"]], on="essay_id", how="left"
    )

    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    submission[["essay_id", "score"]].to_csv(
        SUBMISSION_PATH / SUBMISSION_FILENAME, index=False
    )
    LOGGER.info(
        f"Predicted scores are saved in the file {SUBMISSION_PATH / SUBMISSION_FILENAME}"
    )
