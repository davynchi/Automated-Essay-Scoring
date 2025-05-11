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

from .constants import (
    BEST_CHECKPOINT_POSTFIX,
    CHECKPOINT_POSTFIX,
    NAMES_OF_MODELS,
    OUTPUT_DIR_FINETUNED,
    PATH_TO_TOKENIZER,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from .modify_train_data import create_tokenizer


def load_model(model_name: str) -> "DebertaV2ForMaskedLM":
    """
    Загружает предобученную DeBERTa‑v3 для задачи Masked‑LM.

    Параметры
    ---------
    model_name : str
        HuggingFace‑идентификатор модели (пример: 'microsoft/deberta-v3-base').

    Возврат
    -------
    DebertaV2ForMaskedLM
        Инициализированная модель.
    """
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    return model


def load_datasets() -> tuple["datasets.DatasetDict", "datasets.DatasetDict"]:
    """
    Читает два текстовых файла, сформированных функцией
    ``modify_train_data`` (train/val), через библиотеку *datasets*.

    Возврат
    -------
    (train_ds, valid_ds)
        Сырые текстовые датасеты без токенизации.
    """
    # Load raw text datasets using the datasets library
    raw_train_dataset = load_dataset("text", data_files={"train": str(TRAIN_TEXT_PATH)})
    raw_valid_dataset = load_dataset("text", data_files={"train": str(VAL_TEXT_PATH)})
    return raw_train_dataset, raw_valid_dataset


def tokenize_datasets(
    raw_train_dataset,
    raw_valid_dataset,
    tokenizer,
    block_size: int,
) -> tuple["datasets.Dataset", "datasets.Dataset"]:
    """
    Токенизирует датасеты, нарезая тексты фиксированными блоками
    длиной ``block_size``.

    Возврат
    -------
    tokenized_train_ds, tokenized_valid_ds
    """

    # Tokenize datasets with loop variables bound as default arguments
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
    """
    Запускает дообучение Masked‑LM **для каждой** модели,
    перечисленной в ``NAMES_OF_MODELS``.

    * Создаёт каталоги ``OUTPUT_DIR_FINETUNED``.
    * Сохраняет лучший чек‑пойнт по ``eval_loss`` в подпапку
      ``*_final`` и удаляет промежуточные.
    * Очищает CUDA‑память между циклами.
    """
    for model_name in NAMES_OF_MODELS.values():
        tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        model = load_model(model_name)

        block_size = cfg.pretrain.line_by_line_text_dataset.block_size

        raw_train_dataset, raw_valid_dataset = load_datasets()

        tokenized_train_dataset, tokenized_valid_dataset = tokenize_datasets(
            raw_train_dataset, raw_valid_dataset, tokenizer, block_size
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, **cfg.pretrain.collator
        )

        PATH_TO_SAVE_MODEL = OUTPUT_DIR_FINETUNED / (
            model_name.replace("/", "-") + CHECKPOINT_POSTFIX
        )
        PATH_TO_SAVE_BEST_MODEL = OUTPUT_DIR_FINETUNED / (
            model_name.replace("/", "-") + BEST_CHECKPOINT_POSTFIX
        )

        training_args = TrainingArguments(
            output_dir=PATH_TO_SAVE_MODEL, **cfg.pretrain.training_arguments
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_valid_dataset,
        )

        trainer.train()
        trainer.save_model(PATH_TO_SAVE_BEST_MODEL)

        shutil.rmtree(PATH_TO_SAVE_MODEL)
        torch.cuda.empty_cache()
        gc.collect()
