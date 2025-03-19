import fire

from .common.modify_train_data import modify_train_data
from .deberta_tuning import finetune_and_save_existing_model
from .inference import make_submission
from .train import train_and_save_main_model


def train_and_submit_model(path_to_finetuned_mlm=None):
    modify_train_data()
    # if path_to_finetuned_mlm is None:
    finetune_and_save_existing_model()
    train_and_save_main_model()
    make_submission()


if __name__ == "__main__":
    fire.Fire(train_and_submit_model)
