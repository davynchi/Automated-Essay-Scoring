import gc

import pandas as pd

from ..common.cfg import CFG_LIST
from ..common.model_utils import get_score


def get_oof_preds():
    for i, cfg in enumerate(CFG_LIST):
        oof_df = pd.read_pickle(cfg.oof_path / "oof_df.pkl")
        labels = oof_df[cfg.target_cols].values
        preds = oof_df["pred"].values
        score = get_score(labels, preds)
        labels = oof_df.loc[oof_df.flag == 1, cfg.target_cols].values
        preds = oof_df.loc[oof_df.flag == 1, "pred"].values
        score2 = get_score(labels, preds)
        print(f"pred_{i + 1}", cfg.oof_path)
        print(f"Score: {score:<.4f} Score2: {score2:<.4f}")
        if i == 0:
            df_oof = oof_df[["essay_id", "fold", "flag"] + [cfg.target_cols] + ["pred"]]
            feature_list = ["essay_id", "fold", "flag"] + [cfg.target_cols]
            feature_list += [f"pred_{i + 1}"]
            df_oof.columns = feature_list
        else:
            oof_df = oof_df[["essay_id"] + ["pred"]]
            feature_list = ["essay_id"]
            feature_list += [f"pred_{i + 1}"]
            oof_df.columns = feature_list
            df_oof = pd.merge(df_oof, oof_df, on="essay_id", how="left").reset_index(
                drop=True
            )
    del oof_df, labels, preds, score
    gc.collect()

    df_oof = df_oof.dropna().reset_index(drop=True)
    return df_oof
