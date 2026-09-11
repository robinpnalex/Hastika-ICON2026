"""TF-IDF char n-gram + LinearSVC floor for Task A or Task B.

Emits probabilities on the SAME CV splits the transformers use
(StratifiedKFold, n_splits=5, shuffle=True, random_state=42), so the OOF
matrices line up row-for-row and can be blended in ensemble.py.

--task b reads the multiclass files and predicts the six hate categories. It
exists because Task B had no reproducible floor: the 0.5948 quoted in
RUN_TASK_B.md came from an uncommitted script, so nothing in the repo could
regenerate it or produce an OOF matrix to blend against MuRIL.

--proba softmax is the default for Task B and calibrated for Task A, which is
the only behaviour difference between them. Measured 5-fold on
multiclass_train, calibrating a balanced LinearSVC costs three points of
macro-F1 (0.5948 -> 0.5628): sklearn's sigmoid calibration is fitted per class
on inner folds and drags the scores back toward the class priors, undoing
exactly the class_weight="balanced" correction the tail classes depend on.
Softmax over the decision function leaves the argmax untouched, so the floor is
the floor, and still gives ensemble.py something to blend.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from prep import clean, dedupe_index

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT_SEED = 42          # must match muril.py / muril_b.py for OOF alignment
N_SPLITS = 5

TASKS = {
    "a": {"train": "binary_train.csv", "test": "binary_validation_inputs.csv",
          "label": "Label", "classes": ["Non-Hate", "Hate"], "proba": "calibrated"},
    "b": {"train": "multiclass_train.csv", "test": "multiclass_validation_inputs.csv",
          "label": "Hate Category", "proba": "softmax",
          "classes": ["Gender", "Geo-political", "Others", "Political",
                      "Religion", "Violence"]},
}


class SoftmaxSVC(LinearSVC):
    """LinearSVC with a predict_proba that is just a softmax of the margins.

    Not calibrated probabilities and not claimed to be: the point is a blendable
    score matrix whose argmax still equals the SVM's own decision.
    """

    def predict_proba(self, X):
        d = self.decision_function(X)
        if d.ndim == 1:                       # binary: one margin, two columns
            d = np.column_stack([-d, d])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def build(C=0.5, proba="calibrated"):
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )
    if proba == "calibrated":
        clf = CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced"), cv=3)
    else:
        clf = SoftmaxSVC(C=C, class_weight="balanced")
    return make_pipeline(feats, clf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["a", "b"], default="a")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--tag", default=None, help="default: svm for task a, svm_b for task b")
    ap.add_argument("--demojize", action="store_true")
    ap.add_argument("--proba", choices=["calibrated", "softmax"], default=None,
                    help="default: calibrated for task a, softmax for task b; see module docstring")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="must match the setting used by every other run in the ensemble")
    args = ap.parse_args()
    spec = TASKS[args.task]
    tag = args.tag or ("svm" if args.task == "a" else "svm_b")
    proba = args.proba or spec["proba"]
    classes = spec["classes"]

    train = pd.read_csv(ROOT / "data" / spec["train"])
    test = pd.read_csv(ROOT / "data" / spec["test"])
    if not args.no_dedupe:
        train = train.iloc[dedupe_index(train["Comment"].tolist(),
                                        train[spec["label"]].tolist(),
                                        f"task {args.task.upper()}")].reset_index(drop=True)
    X = train["Comment"].map(lambda x: clean(x, demojize=args.demojize)).values
    y = train[spec["label"]].map(classes.index).values
    X_test = test["Comment"].map(lambda x: clean(x, demojize=args.demojize)).values
    assert not pd.isna(y).any(), f"unmapped label in {spec['train']}"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    oof = np.zeros((len(y), len(classes)))
    for tr, va in skf.split(X, y):
        oof[va] = build(args.C, proba).fit(X[tr], y[tr]).predict_proba(X[va])

    pred = oof.argmax(1)
    print(f"task {args.task.upper()}  {N_SPLITS}-fold OOF  "
          f"macro-F1 {f1_score(y, pred, average='macro'):.4f}  "
          f"acc {accuracy_score(y, pred):.4f}  (proba={proba})")
    print(classification_report(y, pred, target_names=classes, digits=3))

    # refit on everything for the submission predictions
    test_probs = build(args.C, proba).fit(X, y).predict_proba(X_test)

    run = ROOT / "work" / "runs" / tag
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "oof_probs.npy", oof)
    np.save(run / "test_probs.npy", test_probs)
    labels = [classes[i] for i in test_probs.argmax(1)]
    pd.DataFrame({"id": test["id"], "label": labels}).to_csv(run / "predictions.csv", index=False)
    print(f"wrote {run}/  ({pd.Series(labels).value_counts().to_dict()})")


if __name__ == "__main__":
    main()
