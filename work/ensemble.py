"""Blend model runs by weight-searching on the out-of-fold predictions.

Every run in work/runs/<tag>/ that has both oof_probs.npy and test_probs.npy is
eligible. All OOF matrices must come from the same CV splits (StratifiedKFold,
n_splits=5, shuffle=True, random_state=42) or the blend weights are meaningless
-- baseline_svm.py, muril.py and muril_b.py agree on this by design.

--task b blends the six-class runs. Before that flag existed this script was
binary-only in three places at once: it read binary_train.csv for the labels,
sliced column 1 out of every matrix as p(Hate), and thresholded at 0.5. A Task B
run left in work/runs/ would therefore be loaded, compared against the wrong
labels, and crash on the shape mismatch. Runs whose OOF does not match the
task's row and class count are now skipped by name, so both tasks can share the
directory.

The blend is a weighted sum of probability matrices decoded by argmax, which
generalizes the old p(Hate) > 0.5 rule: for two classes the two are the same
decision.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from prep import dedupe_index

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"

TASKS = {
    "a": {"train": "binary_train.csv", "test": "binary_validation_inputs.csv",
          "label": "Label", "classes": ["Non-Hate", "Hate"]},
    "b": {"train": "multiclass_train.csv", "test": "multiclass_validation_inputs.csv",
          "label": "Hate Category",
          "classes": ["Gender", "Geo-political", "Others", "Political",
                      "Religion", "Violence"]},
}


def macro_f1(y, pred, k):
    """Macro-F1 without sklearn's per-call overhead (called ~20k times)."""
    f = 0.0
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f += 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / k


def load(tags, n_rows, n_test, k):
    """Load every run whose matrices have this task's shape; skip the rest.

    The shape check is what keeps Task A and Task B runs in one directory: a
    six-class OOF is (3143, 6) and a binary one is (6446, 2), so neither can be
    mistaken for the other, and a run from a different dedupe setting is caught
    by the row count too.
    """
    oof, test = {}, {}
    for t in tags:
        d = RUNS / t
        if not (d / "oof_probs.npy").exists():
            print(f"  skip {t}: no oof_probs.npy (holdout run? needs --folds 5)")
            continue
        o, e = np.load(d / "oof_probs.npy"), np.load(d / "test_probs.npy")
        if o.shape != (n_rows, k) or e.shape != (n_test, k):
            print(f"  skip {t}: shape {o.shape}/{e.shape}, this task needs "
                  f"{(n_rows, k)}/{(n_test, k)}")
            continue
        oof[t], test[t] = o, e
    return oof, test


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["a", "b"], default="a")
    ap.add_argument("--tags", nargs="*", default=None, help="default: every run with OOF")
    ap.add_argument("--samples", type=int, default=20000,
                    help="random Dirichlet weight samples; a full grid is combinatorial "
                         "(5 models at 0.05 resolution is 4M points)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: ensemble / ensemble_b")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="must match the runs being blended")
    a = ap.parse_args()
    spec = TASKS[a.task]
    out = a.out or ("ensemble" if a.task == "a" else "ensemble_b")
    classes = spec["classes"]
    k = len(classes)

    train = pd.read_csv(ROOT / "data" / spec["train"])
    test_df = pd.read_csv(ROOT / "data" / spec["test"])
    # the runs being blended dropped these rows, so their OOF arrays are this long
    if not a.no_dedupe:
        train = train.iloc[dedupe_index(train["Comment"].tolist(),
                                        train[spec["label"]].tolist(),
                                        f"task {a.task.upper()}")].reset_index(drop=True)
    y = train[spec["label"]].map(classes.index).values

    # exclude our own output dir so a re-run does not try to blend the previous blend
    tags = a.tags or sorted(p.name for p in RUNS.iterdir()
                            if p.is_dir() and p.name != out)
    oof, test = load(tags, len(y), len(test_df), k)
    if not oof:
        sys.exit(f"no task {a.task.upper()} runs with OOF predictions found "
                 f"-- run with --folds 5")

    names = sorted(oof)
    print("\nindividual OOF macro-F1:")
    for n in names:
        print(f"  {n:16s} {f1_score(y, oof[n].argmax(1), average='macro'):.4f}")

    # Search the weight simplex by random Dirichlet sampling -- scales to any
    # number of models, unlike a full grid. Seeded with the single-model corners
    # and the equal-weight point so the blend can never lose to its best member.
    m = len(names)
    P = np.stack([oof[n] for n in names])                 # (m, rows, k)
    T = np.stack([test[n] for n in names])
    rng = np.random.default_rng(a.seed)
    W = np.vstack([np.eye(m), np.full((1, m), 1.0 / m),
                   rng.dirichlet(np.ones(m), size=a.samples)])

    scores = np.array([macro_f1(y, np.einsum("i,ijk->jk", w, P).argmax(1), k) for w in W])
    i = int(scores.argmax())
    f1, w = float(scores[i]), W[i]
    print(f"\nbest blend OOF macro-F1 {f1:.4f}  acc "
          f"{accuracy_score(y, np.einsum('i,ijk->jk', w, P).argmax(1)):.4f}")
    for n, wi in zip(names, w):
        print(f"  {n:16s} {wi:.2f}")

    blend_test = np.einsum("i,ijk->jk", w, T)
    run = RUNS / out
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "test_probs.npy", blend_test)
    labels = [classes[j] for j in blend_test.argmax(1)]
    sub = pd.DataFrame({"id": test_df["id"], "label": labels})
    sub.to_csv(run / "predictions.csv", index=False)
    print(f"\nwrote {run/'predictions.csv'}  ({sub['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
