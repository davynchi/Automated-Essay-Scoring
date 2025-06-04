import codecs
import io
import logging
import os
from pathlib import Path
from typing import Tuple

import dvc.api
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from text_unidecode import unidecode
from transformers import DebertaTokenizer


log = logging.getLogger(__name__)


def get_checkpoint_name(model_idx, fold, stage_idx):
    return f"model{model_idx}_fold{fold}_stage{stage_idx}"


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
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.deterministic = True


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


def find_latest_file(dir: Path, pattern: str) -> str:
    matches = list(dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No ckpt matches {pattern}")
    # Return the newest checkpoint in case there are multiple
    return max(matches, key=lambda p: p.stat().st_mtime).as_posix()


def remove_files(dir: Path, pattern: str) -> None:
    """Remove all files from a directory which match provided pattern.

    Args:
        dir (Path): Path to the directory containing files.
        pattern (str): pattern for files to match to be deleted.
    """
    for ckpt in dir.glob(pattern):
        try:
            ckpt.unlink()
        except FileNotFoundError:
            pass


def create_tokenizer(path: str) -> DebertaTokenizer:
    """Load a DeBERTa tokenizer and add a custom '[BR]' token.

    Args:
        path (str): Pretrained tokenizer path or identifier.

    Returns:
        DebertaTokenizer: Tokenizer with added special token.
    """
    tokenizer = DebertaTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[BR]"]})
    return tokenizer


def read_dataset(path: Path | str):
    with dvc.api.open(
        path=str(path),
        repo=".",
        rev="HEAD",
        mode="rb",
    ) as raw_fd:
        text_fd = io.TextIOWrapper(raw_fd, encoding="utf-8", errors="replace")
        df = pd.read_csv(text_fd, engine="python")
    log.info(f"File {path} is loaded")
    return df
