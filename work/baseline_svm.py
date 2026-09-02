"""TF-IDF char n-gram + LinearSVC floor for Task A.

Emits calibrated probabilities on the SAME CV splits the transformers use
(StratifiedKFold, n_splits=5, shuffle=True, random_state=42), so the OOF
matrices line up row-for-row and can be blended in ensemble.py.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from prep import clean

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT_SEED = 42          # must match train_xlmr.py --seed for OOF alignment
N_SPLITS = 5


def build(C=0.5):
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )
    # LinearSVC has no predict_proba; calibrate so the outputs can be blended.
    clf = CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced"), cv=3)
    return make_pipeline(feats, clf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--tag", default="svm")
    ap.add_argument("--demojize", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(ROOT / "data" / "binary_train.csv")
    test = pd.read_csv(ROOT / "data" / "binary_validation_inputs.csv")
    X = train["Comment"].map(lambda x: clean(x, demojize=args.demojize)).values
    y = (train["Label"] == "Hate").astype(int).values
    X_test = test["Comment"].map(lambda x: clean(x, demojize=args.demojize)).values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    oof = np.zeros((len(y), 2))
    for tr, va in skf.split(X, y):
        oof[va] = build(args.C).fit(X[tr], y[tr]).predict_proba(X[va])

    pred = oof.argmax(1)
    print(f"5-fold OOF  macro-F1 {f1_score(y, pred, average='macro'):.4f}"
          f"  acc {accuracy_score(y, pred):.4f}")

    # refit on everything for the submission predictions
    full = build(args.C).fit(X, y)
    test_probs = full.predict_proba(X_test)

    run = ROOT / "work" / "runs" / args.tag
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "oof_probs.npy", oof)
    np.save(run / "test_probs.npy", test_probs)
    pd.DataFrame({"id": test["id"],
                  "label": np.where(test_probs.argmax(1) == 1, "Hate", "Non-Hate")}
                 ).to_csv(run / "predictions.csv", index=False)
    print(f"wrote {run}/")


if __name__ == "__main__":
    main()
