from glob import glob
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import DebertaTokenizer

from .constants import (
    DATA_PATH,
    PATH_TO_TOKENIZER,
    PROMPTED_DATA_FILENAME,
    TRAIN_FILENAME,
    TRAIN_PICKLE_PATH,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from .utils import modify_texts


def load_pickle_data(cfg_unit, load_from_existed_pickle: bool) -> pd.DataFrame:
    """Load or blend training DataFrame for a given ensemble member.

    In Stage-1 (load_from_existed_pickle=False), adds column `score_s` as `score / 5`.
    In Stage-2 (load_from_existed_pickle=True), merges model's OOF predictions and
    blends with the true scores using `cfg_unit.base.sl_rate` for self-learning.

    Args:
        cfg_unit: Configuration object for the ensemble member, must have `path`,
            `base.target_cols`, `base.modif_target_cols`, and `base.sl_rate`.
        load_from_existed_pickle (bool): Whether to perform Stage-2 blending.

    Returns:
        pd.DataFrame: DataFrame containing columns `essay_id`, `text`, `score`,
            `flag`, `fold`, and `score_s` (modified target column).

    Raises:
        FileNotFoundError: If expected OOF pickle files are not found in Stage-2.
    """
    train = pd.read_pickle(TRAIN_PICKLE_PATH)

    if load_from_existed_pickle:
        oof_paths = sorted(glob(str(Path(cfg_unit.path) / "oof_fold*.pkl")))
        if not oof_paths:
            # No OOF files: fallback to Stage-1 behavior
            train[cfg_unit.base.modif_target_cols[0]] = (
                train[cfg_unit.base.target_cols[0]].values / 5
            )
            return train

        oof_list = [pd.read_pickle(p) for p in oof_paths]
        oof = pd.concat(oof_list, ignore_index=True)

        # Merge and blend
        train = train.merge(oof[["essay_id", "pred"]], on="essay_id", how="left")
        train[cfg_unit.base.modif_target_cols[0]] = (
            (train[cfg_unit.base.target_cols[0]].values / 5) * (1 - cfg_unit.base.sl_rate)
        ) + (train["pred"].fillna(0).values * cfg_unit.base.sl_rate)
    else:
        train[cfg_unit.base.modif_target_cols[0]] = (
            train[cfg_unit.base.target_cols[0]].values / 5
        )

    return train


def read_train_dataset() -> pd.DataFrame:
    """Read and rename columns of the original training CSV.

    Reads `DATA_PATH/TRAIN_FILENAME`, renames columns to `id`, `text`, `score`.

    Returns:
        pd.DataFrame: DataFrame with columns `id`, `text`, `score`.
    """
    train = pd.read_csv(DATA_PATH / TRAIN_FILENAME)
    train.columns = ["id", "text", "score"]
    return train


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
    prompted_data = pd.read_csv(DATA_PATH / PROMPTED_DATA_FILENAME)
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


def tokenize_text(
    data: pd.DataFrame, path_to_tokenizer: str = PATH_TO_TOKENIZER
) -> DebertaTokenizer:
    """Compute token lengths and sort DataFrame by length for efficient padding.

    Adds a `length` column and sorts `data` in-place by ascending token count.

    Args:
        data (pd.DataFrame): DataFrame with `full_text` column.
        path_to_tokenizer (str): Tokenizer path or identifier.

    Returns:
        DebertaTokenizer: Tokenizer used for encoding.
    """
    tokenizer = create_tokenizer(path=path_to_tokenizer)

    def _encode(text: str) -> int:
        return len(tokenizer.encode(text))

    data["length"] = data["full_text"].map(_encode)
    data.sort_values("length", ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return tokenizer


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
    train = read_train_dataset()[:144]
    modify_texts(train, "text")
    train = set_flag_using_prompted_data(train)
    divide_train_into_folds(train, n_splits=cfg.n_folds)
    train = train[["id", "text", "score", "flag", "fold"]]
    write_data_into_pickle(train, TRAIN_PICKLE_PATH)
    train_text, val_text = divide_train_into_train_and_val_by_fold(train)
    write_train_and_val(train_text, val_text)
