import pandas as pd
from sklearn.model_selection import GroupKFold
from transformers import (
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    LineByLineTextDataset,
    Trainer,
    TrainingArguments,
)

from ..common.basic_definitions import create_tokenizer, modify_texts
from ..common.constants import (
    NAME_OF_MODEL_TO_FINETUNE,
    PATH_TO_SAVE_FINETUNED,
    TRAIN_DATA_PATH,
    TRAIN_TEXT_PATH,
    VAL_TEXT_PATH,
)


def read_train_dataset():
    train = pd.read_csv(TRAIN_DATA_PATH)
    train = train[["essay_id", "full_text"]]
    train.columns = ["id", "text"]
    return train


def divide_train_into_folds(train, n_splits=20):
    gkf = GroupKFold(n_splits=n_splits)
    train["fold"] = -1

    for fold, (_, val_) in enumerate(gkf.split(train, train, train["id"])):
        train.loc[val_, "fold"] = fold


def divide_train_into_train_and_val(train):
    train_text = "\n".join(train.loc[train["fold"] != 0, "text"].tolist())
    val_text = "\n".join(train.loc[train["fold"] == 0, "text"].tolist())
    return train_text, val_text


def write_train_and_val(train_text, val_text):
    with open(TRAIN_TEXT_PATH, "w") as f:
        f.write(train_text)
    with open(VAL_TEXT_PATH, "w") as f:
        f.write(val_text)


def modify_train_data():
    train = read_train_dataset()
    modify_texts(train["text"])
    divide_train_into_folds(train)
    train_text, val_text = divide_train_into_train_and_val(train)
    write_train_and_val(train_text, val_text)


def load_model(model_name=NAME_OF_MODEL_TO_FINETUNE):
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    return model


def finetune_and_save_existing_model():
    modify_train_data()
    tokenizer = create_tokenizer()
    model = load_model()

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_TEXT_PATH,  # mention train text file here
        block_size=512,
    )

    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=VAL_TEXT_PATH,  # mention valid text file here
        block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=PATH_TO_SAVE_FINETUNED + "_chk",  # select model path for checkpoint
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=4,
        evaluation_strategy="steps",
        save_total_limit=0,
        save_strategy="steps",
        save_steps=14456,
        eval_steps=7228,
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
    trainer.save_model(PATH_TO_SAVE_FINETUNED)
