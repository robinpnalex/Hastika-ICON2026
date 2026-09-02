"""Two questions about a synthetic batch, before anyone trains on it.

1. DISCRIMINABILITY -- can a cheap classifier tell synthetic from real? If yes,
   the synthetic distribution is not the real one, and a model trained on the mix
   will learn the artifact rather than the task. This is the decisive test, and
   unlike an augmentation A/B it works with only a few hundred synthetic rows.

2. AUGMENTATION -- does adding synthetic rows to the TRAINING folds improve
   out-of-fold macro-F1 on REAL data? Synthetic rows never enter a validation
   fold; otherwise you measure the ability to fit your own generations.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from prep import clean

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYN = ROOT / "work" / "synthetic"


def model():
    return make_pipeline(
        make_union(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        ),
        LinearSVC(C=0.5, class_weight="balanced"),
    )


def main(path):
    syn = [json.loads(l) for l in open(path, encoding="utf-8")]
    Xs = np.array([clean(r["text"]) for r in syn])
    ys = np.array([1 if r["hate_label"] == "HATE" else 0 for r in syn])

    real = pd.read_csv(ROOT / "data" / "binary_train.csv")
    Xr = real["Comment"].map(clean).values
    yr = (real["Label"] == "Hate").astype(int).values

    # ---- 1. discriminability -------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xr), size=len(Xs), replace=False)   # balanced comparison
    Xd = np.concatenate([Xs, Xr[idx]])
    yd = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xs))])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    dec = cross_val_predict(model(), Xd, yd, cv=skf, method="decision_function")
    auc = roc_auc_score(yd, dec)
    print(f"1. real-vs-synthetic AUC  {auc:.3f}   "
          f"({'indistinguishable' if auc < 0.65 else 'trivially separable' if auc > 0.85 else 'distinguishable'})")
    print("   0.50 = synthetic is drawn from the real distribution")
    print("   1.00 = a linear model separates them perfectly; transfer will be poor")

    # ---- 2. augmentation -----------------------------------------------------
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    base_oof = np.zeros(len(yr), dtype=int)
    aug_oof = np.zeros(len(yr), dtype=int)
    for tr, va in skf.split(Xr, yr):
        base_oof[va] = model().fit(Xr[tr], yr[tr]).predict(Xr[va])
        # synthetic joins the TRAINING side only
        aug_oof[va] = model().fit(np.concatenate([Xr[tr], Xs]),
                                  np.concatenate([yr[tr], ys])).predict(Xr[va])
    b = f1_score(yr, base_oof, average="macro")
    a = f1_score(yr, aug_oof, average="macro")
    print(f"\n2. OOF macro-F1 on REAL data")
    print(f"   baseline            {b:.4f}")
    print(f"   + {len(Xs):>4} synthetic     {a:.4f}   ({a-b:+.4f})")
    print(f"   synthetic is {100*len(Xs)/len(Xr):.1f}% of the training set -- an effect "
          f"smaller than ~0.005 is not measurable at this size")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SYN / "batch1.jsonl")
