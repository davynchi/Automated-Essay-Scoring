import fire
import mlflow
import mlflow.pytorch
import mlflow.transformers
from hydra import compose, initialize
from omegaconf import OmegaConf

from .common.constants import SUBMISSION_FILENAME, SUBMISSION_PATH
from .common.modify_train_data import modify_train_data
from .common.utils import create_paths, register_new_utf_errors, seed_everything
from .deberta_tuning import finetune_model
from .inference import make_submission
from .train import train_model


def train_and_submit_model():
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

        seed_everything(cfg.seed)
        register_new_utf_errors()

        mlflow.set_tag("seed", cfg.seed)

        mlflow.set_tag("stage", "data_preprocessing")
        modify_train_data(cfg)

        mlflow.set_tag("stage", "finetune")
        checkpoints_names, tokenizer = finetune_model(cfg)

        mlflow.set_tag("stage", "main_training")
        train_model(cfg, checkpoints_names, tokenizer)

        mlflow.set_tag("stage", "inference")
        make_submission(cfg)

        mlflow.log_artifact(str(SUBMISSION_PATH / SUBMISSION_FILENAME))


if __name__ == "__main__":
    fire.Fire(train_and_submit_model)
