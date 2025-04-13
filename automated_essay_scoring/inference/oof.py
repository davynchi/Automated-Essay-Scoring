import gc
from pathlib import Path

import pandas as pd

from ..common.constants import PICKLE_NAME
from ..common.utils import LOGGER, get_result


def get_oof_preds(cfg):
    for i, cfg_unit in enumerate(cfg.ensemble.values()):
        oof_df = pd.read_pickle(Path(cfg_unit.path) / PICKLE_NAME)

        LOGGER.info(f"pred_{i + 1} {cfg_unit.path}")
        get_result(cfg_unit.base.target_cols, oof_df)

        if i == 0:
            df_oof = oof_df[
                ["essay_id", "fold", "flag"] + [cfg_unit.base.target_cols] + ["pred"]
            ].rename(columns={"pred": f"pred_{i + 1}"})
        else:
            oof_df = oof_df[["essay_id", "pred"]].rename(
                columns={"pred": f"pred_{i + 1}"}
            )
            df_oof = pd.merge(df_oof, oof_df, on="essay_id", how="left").reset_index(
                drop=True
            )
    del oof_df
    gc.collect()

    df_oof = df_oof.dropna().reset_index(drop=True)
    return df_oof
