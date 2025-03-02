from ..common.cfg import CFG


class CFGModified(CFG):
    batch_size = 4
    target_cols = "score"


class CFGBase(CFGModified):
    path = "/kaggle/input/lal-deberta-base-"
    oof_path = path


class CFG1(CFGBase):
    path = CFGBase.path + "v111/"
    max_len = 1024
    head = "mean_pooling"


class CFG2(CFGBase):
    path = CFGBase.path + "v115/"
    max_len = 1024
    head = "attention"


class CFG3(CFGBase):
    path = CFGBase.path + "v117/"
    max_len = 1024
    head = "lstm"


class CFG4(CFGBase):
    path = CFGBase.path + "v118/"
    max_len = 1536
    head = "mean_pooling"


class CFG5(CFGBase):
    path = CFGBase.path + "v119/"
    max_len = 1536
    head = "attention"


class CFG6(CFGBase):
    path = CFGBase.path + "v120/"
    max_len = 1536
    head = "lstm"


class CFGLarge(CFGModified):
    model = "microsoft/deberta-v3-large"
    path = "/kaggle/input/lal-deberta-large-"
    oof_path = path


class CFG7(CFGLarge):
    path = CFGLarge.path + "v051/"
    max_len = 1024
    head = "mean_pooling"


class CFG8(CFGLarge):
    path = CFGLarge.path + "v052/"
    max_len = 1536
    head = "mean_pooling"


class CFG9(CFGLarge):
    path = CFGLarge.path + "v054/"
    max_len = 1024
    head = "attention"


class CFG10(CFGLarge):
    path = CFGLarge.path + "v055/"
    max_len = 1536
    head = "attention"


CFG_LIST = [CFG1, CFG2, CFG3, CFG4, CFG5, CFG6, CFG7, CFG8, CFG9, CFG10]
