from .constants import NAME_OF_MODEL_TO_FINETUNE, SEED


class CFG:
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    model = NAME_OF_MODEL_TO_FINETUNE
    scheduler = "cosine"  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
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
    target_cols = ["0", "1", "2", "3", "4", "5"]
    target_cols2 = ["score"]
    target_cols3 = ["score_s"]
    max_len = 1024
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = SEED
    n_fold = 6
    # trn_fold=[0, 1, 2, 3]
    trn_fold = [0]
    freeze_layer = 9
    head = "mean_pooling"  # 'mean_pooling' 'attention' 'lstm'
    sl = False
    sl_rate = 0.2
    train = True
    flag = 0
    trn_fold = [0, 1, 2, 3, 4, 5]
    oof_path = ""
    config_path = oof_path + "config.pth"
