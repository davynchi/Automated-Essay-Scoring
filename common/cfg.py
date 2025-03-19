from pathlib import Path

from .constants import NAMES_OF_MODEL_TO_FINETUNE, OUTPUT_DIR, SEED


class CFG:
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    # model = NAME_OF_MODEL_TO_FINETUNE
    scheduler = "cosine"  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 1  # Было 5
    encoder_lr = 1e-5
    decoder_lr = 1e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 2
    fc_dropout = 0.0
    model_config = {
        "attention_dropout": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "hidden_dropout": 0.0,
        "hidden_dropout_prob": 0.0,
    }
    target_size = 1
    # target_cols = ["0", "1", "2", "3", "4", "5"] Было, но нигде не использовалось
    target_cols = "score"
    target_cols2 = ["score"]
    target_cols3 = ["score_s"]
    max_len = 1024
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = SEED
    # n_fold = 6 -- так было
    n_fold = 2
    trn_fold = [0]
    freeze_layer = 9
    head = "mean_pooling"  # 'mean_pooling' 'attention' 'lstm'
    sl = False
    sl_rate = 0.2
    train = True
    flag = 0
    trn_fold = [0, 1, 2, 3, 4, 5]
    pickle_name = "oof_df.pkl"


class CFGModified(CFG):
    batch_size = 4


class CFGBase(CFGModified):
    model = NAMES_OF_MODEL_TO_FINETUNE["base"]
    model_key = "base"
    freeze_layer = 9


class CFG1(CFGBase):
    max_len = 1024
    head = "mean_pooling"


class CFG2(CFGBase):
    max_len = 1024
    head = "attention"


class CFG3(CFGBase):
    max_len = 1024
    head = "lstm"


class CFG4(CFGBase):
    max_len = 1536
    head = "mean_pooling"


class CFG5(CFGBase):
    max_len = 1536
    head = "attention"


class CFG6(CFGBase):
    max_len = 1536
    head = "lstm"


class CFGLarge(CFGModified):
    model = NAMES_OF_MODEL_TO_FINETUNE["large"]
    model_key = "large"
    freeze_layer = 6


class CFG7(CFGLarge):
    max_len = 1024
    head = "mean_pooling"


class CFG8(CFGLarge):
    max_len = 1536
    head = "mean_pooling"


class CFG9(CFGLarge):
    max_len = 1024
    head = "attention"


class CFG10(CFGLarge):
    max_len = 1536
    head = "attention"


CFG_LIST = [CFG1, CFG2, CFG3, CFG4, CFG5, CFG6, CFG7, CFG8, CFG9, CFG10]


def create_paths_to_save():
    base_path = Path(OUTPUT_DIR)
    for i, cfg in enumerate(CFG_LIST):
        cfg.path = base_path / f"model_{i}"
        cfg.oof_path = cfg.path
        cfg.path.mkdir(parents=True, exist_ok=True)
        cfg.config_path = cfg.oof_path / "config.pth"


create_paths_to_save()
