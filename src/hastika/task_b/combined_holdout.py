"""The fixed 15% Task B holdout on train + released labelled validation.

Same convention as split_holdout.py -- dedupe_index, then
`train_test_split(test_size=0.15, stratify=y, random_state=SPLIT_SEED)` -- applied to
multiclass_train.csv plus hastika_multiclass_validation.csv. Every notebook that imports
this scores the same rows, so arms compare like-for-like and a paired bootstrap applies.

No cross-task label derivation, no test-set text: the released test file is only ever
predicted, never read for training, TAPT or model screening.

    python -m hastika.task_b.combined_holdout --out data/derived/task_b_combined_holdout
"""
import argparse
import hashlib
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from hastika.common.paths import RAW_DATA_DIR
from hastika.common.preprocessing import dedupe_index
from hastika.models.baseline_svm import SPLIT_SEED

LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]
TEST_FILE = RAW_DATA_DIR / "hastika_multiclass_test.csv"


def load_combined():
    """3,159 train + 395 released validation rows, duplicates by id dropped, then
    duplicate texts collapsed (conflicting-label groups dropped whole)."""
    df = pd.concat([pd.read_csv(RAW_DATA_DIR / "multiclass_train.csv"),
                    pd.read_csv(RAW_DATA_DIR / "hastika_multiclass_validation.csv")],
                   ignore_index=True)
    df = df.drop_duplicates("id", keep="first").reset_index(drop=True)
    keep = dedupe_index(df["Comment"].tolist(), df["Hate Category"].tolist(), "task B combined")
    return df.iloc[keep].reset_index(drop=True)


def split(df=None):
    df = load_combined() if df is None else df
    y = df["Hate Category"].map(LABELS.index).to_numpy()
    tr_i, ho_i = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y,
                                  random_state=SPLIT_SEED)
    return df, tr_i, ho_i


def fingerprint(ids):
    return hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()[:10]


def write(out):
    df, tr_i, ho_i = split()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all.csv", index=False)
    df.iloc[tr_i].to_csv(out / "train.csv", index=False)
    df.iloc[ho_i].to_csv(out / "holdout.csv", index=False)
    y = df["Hate Category"].map(LABELS.index).to_numpy()
    print(f"combined {len(df)} rows -> train {len(tr_i)}, holdout {len(ho_i)} "
          f"(split seed {SPLIT_SEED}, holdout fingerprint {fingerprint(df['id'].iloc[ho_i])})")
    print(f"  {'class':16s} {'train':>6s} {'holdout':>8s}")
    for i, l in enumerate(LABELS):
        print(f"  {l:16s} {(y[tr_i] == i).sum():6d} {(y[ho_i] == i).sum():8d}")
    return df, tr_i, ho_i


def report(y, probs, name=""):
    pred = probs.argmax(1)
    per = f1_score(y, pred, average=None, labels=range(len(LABELS)), zero_division=0)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    print(f"{name:28s} macro-F1 {macro:.4f}  acc {accuracy_score(y, pred):.4f}  | "
          + "  ".join(f"{l[:5]} {f:.3f}" for l, f in zip(LABELS, per)))
    return macro


def paired_bootstrap(y, probs_a, probs_b, n=5000, seed=0):
    """Macro-F1(a) - macro-F1(b) over the same resampled rows. Pairing removes the
    row-sampling variance both arms share, which dominates at 533 rows."""
    rng = np.random.default_rng(seed)
    pa, pb = probs_a.argmax(1), probs_b.argmax(1)
    diffs = np.empty(n)
    for k in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs[k] = (f1_score(y[i], pa[i], average="macro", zero_division=0)
                    - f1_score(y[i], pb[i], average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")),
            "ci95": [float(lo), float(hi)], "p_better": float((diffs > 0).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived/task_b_combined_holdout")
    write(ap.parse_args().out)
