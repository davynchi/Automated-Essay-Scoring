import logging

import fire
import mlflow
import mlflow.pytorch
import mlflow.transformers
import omegaconf
from dvc.repo import Repo
from hydra import compose, initialize
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from .common.constants import ALL_DATA_FILENAMES, OUTPUT_DIR_TRAIN, RAW_DATA_PATH
from .common.logging_config import configure_logging
from .common.utils import register_new_utf_errors, set_torch_params
from .finetune.finetune_model import finetune_model
from .inference.inference_lightning import make_submission_lightning
from .preprocessing.modify_train_data import modify_train_data
from .train.calc_ensemble_weights import calc_best_weights_for_ensemble
from .train.train_lightning import train_model_lightning


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
    with initialize(version_base=None, config_path="./../configs"):
        cfg = compose(config_name="defaults")
    OmegaConf.set_struct(cfg, False)
    return cfg


def create_paths(cfg) -> None:
    """Create output directories for each ensemble model and store in config.

    For each model config in `cfg.ensemble`, creates
    `OUTPUT_DIR_TRAIN/model_i` and assigns its string path to `model_cfg['path']`.

    Args:
        cfg: Configuration object with `ensemble` mapping.

    Returns:
        None
    """
    for i, model_cfg in enumerate(cfg.ensemble.values()):
        dirpath = OUTPUT_DIR_TRAIN / f"model_{i}"
        dirpath.mkdir(parents=True, exist_ok=True)
        model_cfg["path"] = str(dirpath)


def ensure_data() -> None:
    """Гарантирует наличие CSV-файлов локально, иначе качает из Cloud.ru."""
    missing = [f for f in ALL_DATA_FILENAMES if not (RAW_DATA_PATH / f).is_file()]
    if missing:
        logging.info(
            "Локальных файлов %s нет — качаю их из удалённого хранилища…",
            ", ".join(missing),
        )
        Repo(".").pull(targets=[str(RAW_DATA_PATH)], remote="storage-cloud")

        # проверяем, что после pull всё появилось
        still_missing = [f for f in missing if not (RAW_DATA_PATH / f).is_file()]
        if still_missing:
            raise FileNotFoundError(
                f"После pull отсутствуют файлы: {', '.join(still_missing)}"
            )
    else:
        logging.info("Все исходные CSV уже есть локально.")


def train_and_submit_model(
    skip_pretrain_phase: bool = False,
    skip_train_phase: bool = False,
    skip_best_ensemble: bool = False,
    skip_converting_to_tensorrt: bool = False,
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

        mlflow.set_tag("stage", "load_data")
        ensure_data()

        # ───────────────── data ────────────────── #
        mlflow.set_tag("stage", "data_preprocessing")
        modify_train_data(cfg)

        # ───────────────── LM fine-tune ─────────── #
        mlflow.set_tag("stage", "finetune_MLM")
        if not skip_pretrain_phase and not skip_train_phase:
            finetune_model(cfg)
        else:
            log.info("Skipping pretraining phase")

        # ───────────────── main training ────────── #
        mlflow.set_tag("stage", "main_training")
        if not skip_train_phase:
            train_model_lightning(cfg, skip_converting_to_tensorrt)
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
