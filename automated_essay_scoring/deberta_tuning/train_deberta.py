import gc

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
)

from ..common.constants import (
    CHECKPOINT_POSTFIX,
    NAMES_OF_MODELS,
    OUTPUT_DIR_FINETUNED,
    PATH_TO_TOKENIZER,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from ..common.modify_train_data import create_tokenizer


def load_model(model_name):
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    return model


def load_datasets():
    # Load raw text datasets using the datasets library
    raw_train_dataset = load_dataset("text", data_files={"train": str(TRAIN_TEXT_PATH)})
    raw_valid_dataset = load_dataset("text", data_files={"train": str(VAL_TEXT_PATH)})
    return raw_train_dataset, raw_valid_dataset


def tokenize_datasets(raw_train_dataset, raw_valid_dataset, tokenizer, block_size):
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


def finetune_model(cfg):
    checkpoints_names = {}
    for model_key, model_name in NAMES_OF_MODELS.items():
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

        checkpoints_names[model_key] = f"checkpoint-{trainer.state.global_step}"

        torch.cuda.empty_cache()
        gc.collect()

    return checkpoints_names, tokenizer
