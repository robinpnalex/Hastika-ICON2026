"""Materialize the fixed Task B holdout as two labelled CSVs.

`muril_b.py --folds 0` already splits with `train_test_split(test_size=0.15,
stratify=y, random_state=SPLIT_SEED)`. SPLIT_SEED is 42 and never varies, so
every holdout arm ever run was scored on the *same* rows -- the comparisons in
docs/EXPERIMENTS.md are like-for-like already.

This writes those rows out so they can be read, counted and sanity-checked
instead of being trusted. The labels are the organizers' own, taken from
multiclass_train.csv; nothing here invents a label.

The official multiclass_validation_inputs.csv has no label column and its gold
labels are never released, so it cannot be scored locally by any means.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import dedupe_index
from hastika.models.baseline_svm import SPLIT_SEED
from hastika.models.ensemble import ROOT

LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]


def main():
    df = pd.read_csv(RAW_DATA_DIR / "multiclass_train.csv")
    df = df.iloc[dedupe_index(df["Comment"].tolist(), df["Hate Category"].tolist(),
                              "task B")].reset_index(drop=True)
    y = df["Hate Category"].map(LABELS.index).values
    tr_i, va_i = train_test_split(np.arange(len(y)), test_size=0.15,
                                  stratify=y, random_state=SPLIT_SEED)
    out = ROOT / "data" / "derived" / "task_b_holdout"
    out.mkdir(parents=True, exist_ok=True)
    df.iloc[tr_i].to_csv(out / "train.csv", index=False)
    df.iloc[va_i].to_csv(out / "val.csv", index=False)
    df.iloc[va_i][["id", "Comment"]].to_csv(out / "val_inputs.csv", index=False)
    df.iloc[va_i][["id", "Hate Category"]].rename(
        columns={"Hate Category": "label"}).to_csv(out / "val_gold.csv", index=False)

    print(f"wrote {out}/  (split seed {SPLIT_SEED}, the one every holdout arm used)")
    print(f"  train.csv       {len(tr_i):5d} rows, labelled")
    print(f"  val.csv         {len(va_i):5d} rows, labelled")
    print(f"  val_inputs.csv  {len(va_i):5d} rows, id+Comment only")
    print(f"  val_gold.csv    {len(va_i):5d} rows, id+label -- score against this")
    print(f"\n  {'class':16s} {'train':>7s} {'val':>7s}")
    for i, l in enumerate(LABELS):
        print(f"  {l:16s} {(y[tr_i]==i).sum():7d} {(y[va_i]==i).sum():7d}")


if __name__ == "__main__":
    main()
