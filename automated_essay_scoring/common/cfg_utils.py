from pathlib import Path

from .constants import OUTPUT_DIR


def create_paths(cfg):
    base_path = Path(OUTPUT_DIR)
    # print("cfg.ensemble type:", type(cfg.ensemble))
    # print("cfg.ensemble contents:", cfg.ensemble)

    for i, model_cfg in enumerate(cfg.ensemble.values()):
        # print("model_cfg type:", type(model_cfg))
        model_path = base_path / f"model_{i}"
        model_path.mkdir(parents=True, exist_ok=True)
        model_cfg["path"] = str(model_path)
