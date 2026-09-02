"""Blend model runs by weight-searching on the out-of-fold predictions.

Every run in work/runs/<tag>/ that has both oof_probs.npy and test_probs.npy is
eligible. All OOF matrices must come from the same CV splits (StratifiedKFold,
n_splits=5, shuffle=True, random_state=42) or the blend weights are meaningless
-- baseline_svm.py and train_xlmr.py --folds 5 --seed 42 agree on this by design.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from prep import clean  # noqa: F401  (keeps the module import surface consistent)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"


def _macro_f1(y, pred):
    """Binary macro-F1 without sklearn's per-call overhead (called ~20k times)."""
    f = 0.0
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f += 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f / 2


def load(tags):
    oof, test = {}, {}
    for t in tags:
        d = RUNS / t
        if not (d / "oof_probs.npy").exists():
            print(f"  skip {t}: no oof_probs.npy (holdout run? needs --folds 5)")
            continue
        oof[t] = np.load(d / "oof_probs.npy")
        test[t] = np.load(d / "test_probs.npy")
    return oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None, help="default: every run with OOF")
    ap.add_argument("--samples", type=int, default=20000,
                    help="random Dirichlet weight samples; a full grid is combinatorial "
                         "(5 models at 0.05 resolution is 4M points)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ensemble")
    a = ap.parse_args()

    train = pd.read_csv(ROOT / "data" / "binary_train.csv")
    test_df = pd.read_csv(ROOT / "data" / "binary_validation_inputs.csv")
    y = (train["Label"] == "Hate").astype(int).values

    # exclude our own output dir so a re-run does not try to blend the previous blend
    tags = a.tags or sorted(p.name for p in RUNS.iterdir()
                            if p.is_dir() and p.name != a.out)
    oof, test = load(tags)
    if not oof:
        sys.exit("no runs with OOF predictions found -- run with --folds 5")

    names = sorted(oof)
    print("\nindividual OOF macro-F1:")
    for n in names:
        print(f"  {n:14s} {f1_score(y, oof[n].argmax(1), average='macro'):.4f}")

    # Search the weight simplex by random Dirichlet sampling -- scales to any
    # number of models, unlike a full grid. Seeded with the single-model corners
    # and the equal-weight point so the blend can never lose to its best member.
    k = len(names)
    P = np.stack([oof[n][:, 1] for n in names])          # (k, n_rows)
    T = np.stack([test[n][:, 1] for n in names])
    rng = np.random.default_rng(a.seed)
    W = np.vstack([np.eye(k), np.full((1, k), 1.0 / k),
                   rng.dirichlet(np.ones(k), size=a.samples)])

    scores = np.array([_macro_f1(y, (w @ P) > 0.5) for w in W])
    i = int(scores.argmax())
    f1, w = float(scores[i]), W[i]
    print(f"\nbest blend OOF macro-F1 {f1:.4f}")
    for n, wi in zip(names, w):
        print(f"  {n:14s} {wi:.2f}")

    p1 = w @ T
    blend_test = np.stack([1 - p1, p1], axis=1)
    run = RUNS / a.out
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "test_probs.npy", blend_test)
    sub = pd.DataFrame({"id": test_df["id"],
                        "label": np.where(blend_test.argmax(1) == 1, "Hate", "Non-Hate")})
    sub.to_csv(run / "predictions.csv", index=False)
    print(f"\nwrote {run/'predictions.csv'}  ({sub['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
