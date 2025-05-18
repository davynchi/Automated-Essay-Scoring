import gc
import shutil

import datasets
import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
)

from ..common.constants import (
    BEST_CHECKPOINT_POSTFIX,
    CHECKPOINT_POSTFIX,
    NAMES_OF_MODELS,
    OUTPUT_DIR_FINETUNED,
    PATH_TO_TOKENIZER,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from ..common.utils import create_tokenizer


def load_model(model_name: str) -> DebertaV2ForMaskedLM:
    """Load a pretrained DeBERTa-v3 model for Masked Language Modeling.

    Args:
        model_name (str): HuggingFace model identifier,
            e.g., `"microsoft/deberta-v3-base"`.

    Returns:
        DebertaV2ForMaskedLM: The loaded model ready for fine-tuning.
    """
    return DebertaV2ForMaskedLM.from_pretrained(model_name)


def load_datasets() -> tuple[datasets.DatasetDict, datasets.DatasetDict]:
    """Load raw text datasets for pretraining and validation.

    Reads two text files (TRAIN_TEXT_PATH, VAL_TEXT_PATH) produced by
    `modify_train_data.create_train_val_files`, using the HuggingFace `datasets` library.

    Returns:
        Tuple[datasets.DatasetDict, datasets.DatasetDict]:
            - raw_train_dataset: DatasetDict with a `"train"` split from TRAIN_TEXT_PATH.
            - raw_valid_dataset: DatasetDict with a `"train"` split from VAL_TEXT_PATH.
    """
    raw_train_dataset = load_dataset("text", data_files={"train": str(TRAIN_TEXT_PATH)})
    raw_valid_dataset = load_dataset("text", data_files={"train": str(VAL_TEXT_PATH)})
    return raw_train_dataset, raw_valid_dataset


def tokenize_datasets(
    raw_train_dataset: datasets.DatasetDict,
    raw_valid_dataset: datasets.DatasetDict,
    tokenizer,
    block_size: int,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Tokenize raw text datasets into fixed-length blocks.

    Applies `tokenizer` to each split, truncating or padding
    sequences to `block_size` tokens and removing the raw `"text"` column.

    Args:
        raw_train_dataset (datasets.DatasetDict): DatasetDict with a `"train"` split.
        raw_valid_dataset (datasets.DatasetDict): DatasetDict with a `"train"` split.
        tokenizer: A tokenizer implementing `__call__(text, truncation, max_length)`.
        block_size (int): Maximum sequence length for tokenization.

    Returns:
        Tuple[datasets.Dataset, datasets.Dataset]:
            - tokenized_train_dataset: Tokenized `raw_train_dataset["train"]`.
            - tokenized_valid_dataset: Tokenized `raw_valid_dataset["train"]`.
    """

    def tokenize_function(examples, tokenizer=tokenizer, block_size=block_size):
        return tokenizer(examples["text"], truncation=True, max_length=block_size)

    tokenized_train_dataset = raw_train_dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_valid_dataset = raw_valid_dataset["train"].map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    return tokenized_train_dataset, tokenized_valid_dataset


def finetune_model(cfg) -> None:
    """Fine-tune multiple Masked-LM models listed in configuration.

    For each model in `NAMES_OF_MODELS`:
      1. Creates a fresh tokenizer.
      2. Loads a pretrained DeBERTa-v3 via `load_model`.
      3. Loads and tokenizes train/validation datasets.
      4. Sets up a `Trainer` with `TrainingArguments` from `cfg.pretrain`.
      5. Runs `trainer.train()`, saves the best checkpoint by eval_loss.
      6. Cleans up intermediate checkpoints and frees CUDA memory.

    Args:
        cfg: Configuration object containing:
            - `pretrain.line_by_line_text_dataset.block_size` (int)
            - `pretrain.collator` (dict of DataCollatorForLanguageModeling kwargs)
            - `pretrain.training_arguments` (dict of TrainingArguments kwargs)
    """
    for model_name in NAMES_OF_MODELS.values():
        tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        model = load_model(model_name)

        block_size = cfg.pretrain.line_by_line_text_dataset.block_size

        raw_train_dataset, raw_valid_dataset = load_datasets()
        tokenized_train, tokenized_valid = tokenize_datasets(
            raw_train_dataset, raw_valid_dataset, tokenizer, block_size
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, **cfg.pretrain.collator
        )

        save_dir = OUTPUT_DIR_FINETUNED / (
            model_name.replace("/", "-") + CHECKPOINT_POSTFIX
        )
        best_dir = OUTPUT_DIR_FINETUNED / (
            model_name.replace("/", "-") + BEST_CHECKPOINT_POSTFIX
        )

        training_args = TrainingArguments(
            output_dir=save_dir, **cfg.pretrain.training_arguments
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
        )

        trainer.train()
        trainer.save_model(best_dir)

        shutil.rmtree(save_dir)
        torch.cuda.empty_cache()
        gc.collect()
