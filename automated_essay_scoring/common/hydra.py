import omegaconf
from hydra import compose, initialize
from omegaconf import OmegaConf

from .constants import HYDRA_CONFIG_NAME, HYDRA_CONFIG_PATH, OUTPUT_DIR_TRAIN


def load_config() -> "omegaconf.DictConfig":
    """
    Собирает конфигурацию Hydra из каталога ``conf`` и
    переводит её в нестрогий режим (``set_struct(False)``).

    Возврат
    -------
    DictConfig
        Финальная конфигурация проекта.
    """
    with initialize(version_base=None, config_path=str(HYDRA_CONFIG_PATH)):
        cfg = compose(config_name=HYDRA_CONFIG_NAME)
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
