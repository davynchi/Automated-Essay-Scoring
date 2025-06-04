from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from ..common.constants import (
    PROMPTED_DATA_FILENAME,
    RAW_DATA_PATH,
    TRAIN_FILENAME,
    TRAIN_PICKLE_PATH,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from ..common.utils import modify_texts, read_dataset


def divide_train_into_folds(train: pd.DataFrame, n_splits: int) -> None:
    """Assign fold numbers to the training DataFrame in-place using stratification.

    Uses `MultilabelStratifiedKFold` on `prompt_name` and `score` to assign
    a `fold` column with values in [0, n_splits-1].

    Args:
        train (pd.DataFrame): DataFrame containing `prompt_name` and `score` columns.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        None
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train["fold"] = -1
    train.loc[train["prompt_name"].isna(), "prompt_name"] = "Unknown prompt name"

    for fold, (_, val_idx) in enumerate(
        mskf.split(train, train[["prompt_name", "score"]])
    ):
        train.loc[val_idx, "fold"] = fold


def set_flag_using_prompted_data(train: pd.DataFrame) -> pd.DataFrame:
    """Mark essays with a flag indicating presence of explicit prompt.

    Reads a CSV of prompted data and merges on full text; sets `flag`:
      - 0 if `prompt_name` present (prompted)
      - 1 if `prompt_name` missing (unprompted)

    Args:
        train (pd.DataFrame): DataFrame with `text` column.

    Returns:
        pd.DataFrame: Merged DataFrame including `flag` column.
    """
    prompted_data = read_dataset(RAW_DATA_PATH / "train" / PROMPTED_DATA_FILENAME)

    merged = pd.merge(
        train, prompted_data, left_on="text", right_on="full_text", how="left"
    )
    merged["flag"] = 0
    merged.loc[merged["prompt_name"].isna(), "flag"] = 1
    return merged


def write_data_into_pickle(data: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to a pickle file with specific column names.

    Renames `text`→`full_text` and `id`→`essay_id` before saving.

    Args:
        data (pd.DataFrame): Input DataFrame with `id` and `text` columns.
        file_path (Path): Destination pickle file path.

    Returns:
        None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = data.rename(columns={"text": "full_text", "id": "essay_id"})
    to_save.to_pickle(file_path)


def divide_train_into_train_and_val_by_fold(train: pd.DataFrame) -> tuple[str, str]:
    """Create contiguous text corpora for language model fine-tuning.

    Constructs two large strings:
      - `train_text`: all essays where `fold != 0`
      - `val_text`: all essays where `fold == 0`
    Joined by newline characters.

    Args:
        train (pd.DataFrame): DataFrame with `text` and `fold` columns.

    Returns:
        Tuple[str, str]: `(train_text, val_text)` corpora.
    """
    train_text = "\n".join(train.loc[train["fold"] != 0, "text"])
    val_text = "\n".join(train.loc[train["fold"] == 0, "text"])
    return train_text, val_text


def write_train_and_val(train_text: str, val_text: str) -> None:
    """Write training and validation text corpora to files.

    Args:
        train_text (str): Training corpus, one document per line.
        val_text (str): Validation corpus, one document per line.

    Returns:
        None
    """
    with open(TRAIN_TEXT_PATH, "w") as tf:
        tf.write(train_text)
    with open(VAL_TEXT_PATH, "w") as vf:
        vf.write(val_text)


def modify_train_data(cfg) -> None:
    """Full training-data preprocessing pipeline.

    Steps:
      1. Read and trim original train CSV.
      2. Flag prompting usage (`set_flag_using_prompted_data`).
      3. Stratified fold split (`divide_train_into_folds`).
      4. Save to pickle (`write_data_into_pickle`).
      5. Create and write LM corpora (`divide_train_into_train_and_val_by_fold`,
         `write_train_and_val`).

    Args:
        cfg: Configuration object with attribute `n_folds`.

    Returns:
        None
    """
    train = read_dataset(RAW_DATA_PATH / "train" / TRAIN_FILENAME)[:144]
    train.columns = ["id", "text", "score"]
    modify_texts(train, "text")
    train = set_flag_using_prompted_data(train)
    divide_train_into_folds(train, n_splits=cfg.n_folds)
    train = train[["id", "text", "score", "flag", "fold"]]
    write_data_into_pickle(train, TRAIN_PICKLE_PATH)
    train_text, val_text = divide_train_into_train_and_val_by_fold(train)
    write_train_and_val(train_text, val_text)
