import logging

import fire
import mlflow
import mlflow.pytorch
import mlflow.transformers
import omegaconf
from hydra import compose, initialize
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from .common.calc_ensemble_weights import calc_best_weights_for_ensemble
from .common.finetune_model import finetune_model
from .common.inference_lightning import make_submission_lightning
from .common.logging_config import configure_logging
from .common.modify_train_data import modify_train_data
from .common.train_lightning import train_model_lightning
from .common.utils import create_paths, register_new_utf_errors, set_torch_params


def start_logging() -> logging.Logger:
    """
    Настраивает корневой логгер через ``configure_logging`` и
    возвращает объект‑логгер текущего модуля.

    Возврат
    -------
    logging.Logger
        Конфигурированный логгер, которым пользуется run‑pipeline.
    """
    configure_logging(level="INFO")
    log = logging.getLogger(__name__)
    log.info("🚀 Starting pipeline …")
    return log


def start_mlflow() -> None:
    """
    Инициализирует глобальные настройки MLflow:
    * Задаёт (или создаёт) эксперимент *essay‑scoring‑pipeline*.
    * Включает autolog для MLflow, PyTorch и HuggingFace Transformers.
    """
    mlflow.set_experiment("essay-scoring-pipeline")
    mlflow.autolog()
    mlflow.pytorch.autolog()
    mlflow.transformers.autolog()


def load_config() -> "omegaconf.DictConfig":
    """
    Собирает конфигурацию Hydra из каталога ``conf`` и
    переводит её в нестрогий режим (``set_struct(False)``).

    Возврат
    -------
    DictConfig
        Финальная конфигурация проекта.
    """
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="defaults")
    OmegaConf.set_struct(cfg, False)
    return cfg


def train_and_submit_model(
    path_to_finetuned_models: str | None = None,
    skip_train_phase: bool = False,
    skip_best_ensemble: bool = False,
) -> None:
    """
    Полный конвейер: подготовка данных → (опц.) дообучение MLM →
    (опц.) основное обучение → (опц.) расчёт весов ансамбля →
    инференс и генерация submission.

    Параметры
    ---------
    path_to_finetuned_models : str | None
        Путь к уже дообученным весам DeBERTa; если ``None``, запустится
        ``finetune_model``.
    skip_train_phase : bool
        Пропустить стадию обучения (использовать сохранённые чек‑пойнты).
    skip_best_ensemble : bool
        Пропустить поиск оптимальных весов ансамбля (использовать
        `best_ensemble_weights.npy`, если он уже существует).
    """
    log = start_logging()
    start_mlflow()
    cfg = load_config()
    create_paths(cfg)
    seed_everything(cfg.seed, workers=True)
    set_torch_params()
    register_new_utf_errors()

    with mlflow.start_run(run_name="full_pipeline"):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")
        mlflow.set_tag("seed", cfg.seed)
        # print(f"Final Configurations: \n{OmegaConf.to_yaml(cfg)}")

        # ───────────────── data ────────────────── #
        mlflow.set_tag("stage", "data_preprocessing")
        modify_train_data(cfg)

        # ───────────────── LM fine-tune ─────────── #
        mlflow.set_tag("stage", "finetune_MLM")
        if path_to_finetuned_models is None and not skip_train_phase:
            finetune_model(cfg)
        else:
            log.info(f"Using pre-trained model from {path_to_finetuned_models}")

        # ───────────────── main training ────────── #
        mlflow.set_tag("stage", "main_training")
        if not skip_train_phase:
            train_model_lightning(cfg, path_to_finetuned_models)
        else:
            log.info("Skipping training phase")

        # ───────────────── ensemble weights ────────── #
        mlflow.set_tag("stage", "ensemble_weights")
        if not skip_best_ensemble:
            calc_best_weights_for_ensemble(cfg)
        else:
            log.info("Skipping ensemble weights calculation")

        # ───────────────── inference / submit ───── #
        mlflow.set_tag("stage", "inference")
        make_submission_lightning(cfg)


if __name__ == "__main__":
    fire.Fire(train_and_submit_model)
