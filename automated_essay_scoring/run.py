import logging

import fire
import mlflow
import mlflow.pytorch
import mlflow.transformers
from hydra import compose, initialize
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from .common.constants import SUBMISSION_FILENAME, SUBMISSION_PATH
from .common.finetune_model import finetune_model
from .common.inference_lightning import make_submission_lightning
from .common.logging_config import configure_logging
from .common.modify_train_data import modify_train_data
from .common.train_lightning import train_model_lightning
from .common.utils import create_paths, register_new_utf_errors, set_torch_params


def train_and_submit_model():
    configure_logging(level="INFO")
    log = logging.getLogger(__name__)
    log.info("🚀 Starting pipeline …")

    mlflow.set_experiment("essay-scoring-pipeline")
    mlflow.autolog()
    mlflow.pytorch.autolog()
    mlflow.transformers.autolog()

    with mlflow.start_run(run_name="full_pipeline"):
        with initialize(version_base=None, config_path="conf"):
            cfg = compose(config_name="defaults")
        OmegaConf.set_struct(cfg, False)
        create_paths(cfg)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")
        # print(f"Final Configurations: \n{OmegaConf.to_yaml(cfg)}")

        seed_everything(cfg.seed, workers=True)
        set_torch_params()
        register_new_utf_errors()

        mlflow.set_tag("seed", cfg.seed)

        # ───────────────── data ────────────────── #
        mlflow.set_tag("stage", "data_preprocessing")
        modify_train_data(cfg)

        # ───────────────── LM fine-tune ─────────── #
        mlflow.set_tag("stage", "finetune_MLM")
        checkpoints_names, tokenizer = finetune_model(cfg)

        # ───────────────── main training ────────── #
        mlflow.set_tag("stage", "main_training")
        train_model_lightning(cfg, checkpoints_names, tokenizer)

        # ───────────────── inference / submit ───── #
        mlflow.set_tag("stage", "inference")
        make_submission_lightning(cfg, tokenizer)

        mlflow.log_artifact(str(SUBMISSION_PATH / SUBMISSION_FILENAME))


if __name__ == "__main__":
    fire.Fire(train_and_submit_model)
