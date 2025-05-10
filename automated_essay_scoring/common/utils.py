import codecs

# from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from text_unidecode import unidecode

from .constants import NAMES_OF_MODELS, OUTPUT_DIR_TRAIN


def set_torch_params():
    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True,max_split_size_mb:128"
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


def register_new_utf_errors():
    # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
    codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
    codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def modify_texts(texts):
    texts = texts.apply(lambda x: resolve_encodings_and_normalize(x))
    texts = [text.replace("\n", "[BR]") for text in texts]


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


def get_result(target_cols, oof_df, pred_col="pred"):
    labels = oof_df[target_cols].values
    preds = oof_df[pred_col].values
    score = get_score(labels, preds)
    labels = oof_df.loc[oof_df.flag == 1, target_cols].values
    preds = oof_df.loc[oof_df.flag == 1, pred_col].values
    score2 = get_score(labels, preds)
    log = logging.getLogger(__name__)
    log.info(f"Score: {score:<.4f} Score2: {score2:<.4f}")
    return score, score2


def get_model_path(cfg, fold):
    model_path = (
        Path(cfg.path)
        / f"{NAMES_OF_MODELS[cfg.model_key].replace('/', '-')}_fold{fold}_best.pth"
    )
    return model_path


def create_paths(cfg):
    for i, model_cfg in enumerate(cfg.ensemble.values()):
        model_path = OUTPUT_DIR_TRAIN / f"model_{i}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_cfg["path"] = str(model_path)
