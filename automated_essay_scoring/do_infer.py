import fire
from lightning.pytorch import seed_everything

from .common.hydra import load_config
from .common.logging_config import start_logging
from .common.utils import register_new_utf_errors, set_torch_params
from .inference.client_side import make_submission_triton


def infer_model():
    log = start_logging()
    log.info("Starting the inference")
    cfg = load_config()
    seed_everything(cfg.seed, workers=True)
    set_torch_params()
    register_new_utf_errors()

    make_submission_triton(cfg)


if __name__ == "__main__":
    fire.Fire(infer_model)
