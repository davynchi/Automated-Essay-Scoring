import fire
from hydra import compose, initialize
from omegaconf import OmegaConf

from .common.modify_train_data import modify_train_data
from .common.utils import create_paths, register_new_utf_errors, seed_everything
from .deberta_tuning import finetune_and_save_existing_model
from .inference import make_submission
from .train import train_and_save_main_model


def train_and_submit_model():
    seed_everything()
    register_new_utf_errors()

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="defaults")
    OmegaConf.set_struct(cfg, False)
    create_paths(cfg)
    # print(f"Final Configurations: \n{OmegaConf.to_yaml(cfg)}")

    modify_train_data(cfg)

    checkpoints_names, tokenizer = finetune_and_save_existing_model(cfg)
    train_and_save_main_model(cfg, checkpoints_names, tokenizer)
    make_submission(cfg)


if __name__ == "__main__":
    fire.Fire(train_and_submit_model)
