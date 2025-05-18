import codecs
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


def set_torch_params() -> None:
    """Configure PyTorch and environment for stable training.

    Sets environment variables for CUDA memory allocation and tokenizers,
    enables FlashAttention / SDPA for performance, and enforces deterministic
    behavior in CuDNN.

    Args:
        None

    Returns:
        None
    """
    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True,max_split_size_mb:128"
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.deterministic = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    """Error handler: re-encode problematic bytes to UTF-8.

    Args:
        error (UnicodeError): Exception raised during encoding.

    Returns:
        Tuple[bytes, int]: Replacement byte sequence and position to resume.
    """
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    """Error handler: decode problematic byte sequence with cp1252.

    Args:
        error (UnicodeError): Exception raised during decoding.

    Returns:
        Tuple[str, int]: Replacement string and position to resume.
    """
    return error.object[error.start : error.end].decode("cp1252"), error.end


def register_new_utf_errors() -> None:
    """Register custom Unicode error handlers.

    Registers 'replace_encoding_with_utf8' and 'replace_decoding_with_cp1252'
    for use in `codecs` operations to fix encoding issues.

    Args:
        None

    Returns:
        None
    """
    codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
    codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Fix mojibake and normalize Unicode text to ASCII.

    Applies a sequence of encodings/decodings with custom error handlers,
    then transliterates to ASCII using `text_unidecode`.

    Args:
        text (str): Raw text potentially containing encoding artifacts.

    Returns:
        str: Cleaned and ASCII-normalized text.
    """
    fixed = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    return unidecode(fixed)


def modify_texts(df: pd.Series, col: str = "text") -> None:
    """Normalize a Series of texts in-place.

    Applies `resolve_encodings_and_normalize` to each element of `df[col]` and replaces
    newline characters with the '[BR]' token.

    Args:
        texts (pd.Series): Series of raw text strings.

    Returns:
        None
    """
    normalized = df[col].map(resolve_encodings_and_normalize)
    df.loc[:, col] = normalized.str.replace("\n", "[BR]", regex=False)


def get_essay_score(y_preds: np.ndarray) -> pd.Series:
    """Convert continuous predictions to integer scores [0..5].

    Scales predictions (0-1) to 0-5 and bins using fixed cut points.

    Args:
        y_preds (np.ndarray): Array of prediction floats in [0,1].

    Returns:
        pd.Series: Integer scores in {0,1,2,3,4,5}.
    """
    scaled = y_preds.reshape(-1) * 5
    bins = [-np.inf, 0.83333333, 1.66666667, 2.5, 3.33333333, 4.16666667, np.inf]
    labels = list(range(6))
    return pd.cut(scaled, bins, labels=labels).astype(int)


def get_score(y_trues: np.ndarray, y_preds: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa between true and predicted scores.

    Args:
        y_trues (np.ndarray): True integer labels.
        y_preds (np.ndarray): Predicted floats in [0,1].

    Returns:
        float: QWK metric.
    """
    int_preds = get_essay_score(y_preds)
    return cohen_kappa_score(y_trues, int_preds, weights="quadratic")


def get_result(
    target_cols: list[str], oof_df: pd.DataFrame, pred_col: str = "pred"
) -> Tuple[float, float]:
    """Compute overall and unprompted QWK scores for OOF predictions.

    Logs and returns:
      - score on all examples
      - score on subset where `flag == 1` (unprompted)

    Args:
        target_cols (list[str]): Column names of true labels in `oof_df`.
        oof_df (pd.DataFrame): DataFrame with true labels, predictions, and `flag`.
        pred_col (str): Column name of predictions in `oof_df`.

    Returns:
        Tuple[float, float]: (all_texts_score, unprompted_texts_score).
    """
    y_true_all = oof_df[target_cols].values
    y_pred_all = oof_df[pred_col].values
    score_all = get_score(y_true_all, y_pred_all)

    mask = oof_df["flag"] == 1
    y_true_up = oof_df.loc[mask, target_cols].values
    y_pred_up = oof_df.loc[mask, pred_col].values
    score_up = get_score(y_true_up, y_pred_up)

    logging.getLogger(__name__).info(
        f"Score: {score_all:.4f}, Unprompted Score: {score_up:.4f}"
    )
    return score_all, score_up


def get_model_path(cfg, fold: int) -> Path:
    """Construct the path to the best model checkpoint for a fold.

    Args:
        cfg: Config object containing `path` and `model_key`.
        fold (int): Fold number.

    Returns:
        Path: Filesystem path to the best model checkpoint file.
    """
    name = NAMES_OF_MODELS[cfg.model_key].replace("/", "-")
    filename = f"{name}_fold{fold}_best.pth"
    return Path(cfg.path) / filename


def create_paths(cfg) -> None:
    """Create output directories for each ensemble model and store in config.

    For each model config in `cfg.ensemble`, creates
    `OUTPUT_DIR_TRAIN/model_i` and assigns its string path to `model_cfg['path']`.

    Args:
        cfg: Configuration object with `ensemble` mapping.

    Returns:
        None
    """
    for i, model_cfg in enumerate(cfg.ensemble.values()):
        dirpath = OUTPUT_DIR_TRAIN / f"model_{i}"
        dirpath.mkdir(parents=True, exist_ok=True)
        model_cfg["path"] = str(dirpath)
