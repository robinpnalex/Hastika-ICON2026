"""The fixed 15% Task A holdout on train + released labelled validation.

Task A's counterpart of hastika.task_b.combined_holdout: binary_train.csv plus
hastika_binary_validation.csv, duplicate ids dropped, duplicate texts collapsed by
dedupe_index, then `train_test_split(test_size=0.15, stratify=y,
random_state=SPLIT_SEED)`. The CSVs are committed, so a notebook reads the same rows
whatever library versions Kaggle has.

No cross-task label derivation and no transductive rows: the released test file is
never read here.

    python -m hastika.task_a.combined_holdout --out data/derived/task_a_combined_holdout
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import dedupe_index
from hastika.models.baseline_svm import SPLIT_SEED
from hastika.task_b.combined_holdout import fingerprint

LABELS = ["Non-Hate", "Hate"]       # column order of every Task A probability matrix


def load_combined():
    df = pd.concat([pd.read_csv(RAW_DATA_DIR / "binary_train.csv"),
                    pd.read_csv(RAW_DATA_DIR / "hastika_binary_validation.csv")],
                   ignore_index=True)
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    keep = dedupe_index(df["Comment"].tolist(), df["Label"].tolist(), "task A combined")
    return df.iloc[keep].reset_index(drop=True)


def split(df=None):
    df = load_combined() if df is None else df
    y = (df["Label"] == "Hate").astype(int).to_numpy()
    tr_i, ho_i = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y,
                                  random_state=SPLIT_SEED)
    return df, tr_i, ho_i


def write(out):
    df, tr_i, ho_i = split()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all.csv", index=False)
    df.iloc[tr_i].to_csv(out / "train.csv", index=False)
    df.iloc[ho_i].to_csv(out / "holdout.csv", index=False)
    y = (df["Label"] == "Hate").to_numpy()
    print(f"combined {len(df)} rows -> train {len(tr_i)} ({y[tr_i].mean():.1%} Hate), "
          f"holdout {len(ho_i)} ({y[ho_i].mean():.1%} Hate); split seed {SPLIT_SEED}, "
          f"holdout fingerprint {fingerprint(df['id'].iloc[ho_i])}")
    return df, tr_i, ho_i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/task_a_combined_holdout")
    write(ap.parse_args().out)
