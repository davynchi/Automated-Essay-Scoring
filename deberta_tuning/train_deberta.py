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
    BASE_PATH_TO_SAVE_FINETUNED,
    BLOCK_SIZE,
    CHECKPOINT_POSTFIX,
    CHECKPOINTS_NAMES,
    NAMES_OF_MODEL_TO_FINETUNE,
    PATH_TO_TOKENIZER,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)
from ..common.modify_train_data import create_tokenizer


def load_model(model_name):
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    return model


def finetune_and_save_existing_model():
    for model_key, model_name in NAMES_OF_MODEL_TO_FINETUNE.items():
        tokenizer = create_tokenizer(path=PATH_TO_TOKENIZER)
        model = load_model(model_name)

        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=TRAIN_TEXT_PATH,  # mention train text file here
            block_size=BLOCK_SIZE,
        )

        valid_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=VAL_TEXT_PATH,  # mention valid text file here
            block_size=BLOCK_SIZE,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        PATH_TO_SAVE_MODEL = BASE_PATH_TO_SAVE_FINETUNED / (
            model_name.replace("/", "-") + CHECKPOINT_POSTFIX
        )

        training_args = TrainingArguments(
            output_dir=PATH_TO_SAVE_MODEL,
            overwrite_output_dir=True,
            num_train_epochs=2,  # 8,  -- Было столько
            per_device_train_batch_size=1,
            evaluation_strategy="steps",
            save_total_limit=0,
            save_strategy="steps",
            save_steps=3614,  # 14456,-- Было столько
            eval_steps=1807,  # 7228, -- Было столько
            fp16=True,  # Этой опции не было изначально
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            prediction_loss_only=True,
            report_to="none",
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

        global CHECKPOINTS_NAMES
        CHECKPOINTS_NAMES[model_key] = f"checkpoint-{trainer.state.global_step}"

        torch.cuda.empty_cache()
        gc.collect()
