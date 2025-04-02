import gc

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    LineByLineTextDataset,
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


def finetune_and_save_existing_model(cfg):
    checkpoints_names = {}
    for model_key, model_name in NAMES_OF_MODELS.items():
        tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        model = load_model(model_name)

        block_size = cfg.pretrain.line_by_line_text_dataset.block_size
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=TRAIN_TEXT_PATH,  # mention train text file here
            block_size=block_size,
        )

        valid_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=VAL_TEXT_PATH,  # mention valid text file here
            block_size=block_size,
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
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        trainer.train()
        # trainer.save_model(PATH_TO_SAVE_MODEL)

        checkpoints_names[model_key] = f"checkpoint-{trainer.state.global_step}"

        torch.cuda.empty_cache()
        gc.collect()

    return checkpoints_names, tokenizer
