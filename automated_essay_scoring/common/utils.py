import codecs
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from text_unidecode import unidecode

from .constants import OUTPUT_DIR_TRAIN


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


def modify_texts(texts):  # Раньше было train['text']
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


def create_paths(cfg):
    for i, model_cfg in enumerate(cfg.ensemble.values()):
        model_path = OUTPUT_DIR_TRAIN / f"model_{i}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_cfg["path"] = str(model_path)
