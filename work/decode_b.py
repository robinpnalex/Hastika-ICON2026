"""Fit a macro-F1-optimal decode rule on the out-of-fold probabilities.

WHY THIS IS NOT THE THING THAT WAS ALREADY MEASURED AND REJECTED
---------------------------------------------------------------
muril_b.py's docstring records that per-class score offsets moved macro-F1 by
+0.002 on the TF-IDF OOF. That search had one free parameter: a prior exponent,
sweeping every class's offset together along a single line. This one fits six
independent per-class multipliers by coordinate ascent, which can trade one
class against another rather than tilting all six at once.

The reason to expect anything at all is that argmax maximizes accuracy, and
accuracy is not what Task B is scored on. Under macro-F1 a class carries 1/6 of
the score whatever its support, so the decision rule that maximizes it is not
the one that maximizes the posterior: it is worth losing several Gender rows to
win one Violence row, and argmax will never make that trade.

The measured symptom this targets: on the 5-fold SVM the same model calls
Violence on 4.93% of training rows but only 1.52% of test rows, and Violence is
the weakest class at F1 0.346.

HONESTY
-------
Weights fitted on the same rows they are scored on will always look better.
Everything below is therefore reported three ways -- plain argmax, in-sample
tuned, and a nested estimate that fits the weights on 4 folds and scores the
5th. Believe the nested number. If it does not beat plain argmax, this script
says so and writes nothing.
"""
import argparse
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# imported from the torch-free modules so this script runs on a CPU box
from baseline_svm import SPLIT_SEED
from ensemble import ROOT, RUNS, macro_f1
from prep import dedupe_index

LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]


def truth():
    """Rebuild y exactly as muril_b.py does, so the OOF rows line up."""
    train = pd.read_csv(ROOT / "data" / "multiclass_train.csv")
    train = train.iloc[dedupe_index(train["Comment"].tolist(),
                                    train["Hate Category"].tolist(),
                                    "task B")].reset_index(drop=True)
    return train["Hate Category"].map(LABELS.index).values


def fit_weights(P, y, grid, rounds=6):
    """Coordinate ascent on per-class multipliers. Ties broken toward w=1."""
    k = P.shape[1]
    w = np.ones(k)
    best = macro_f1(y, P.argmax(1), k)
    for _ in range(rounds):
        moved = False
        for c in range(k):
            keep, local = w[c], best
            for g in grid:
                w[c] = g
                f = macro_f1(y, (P * w).argmax(1), k)
                if f > local + 1e-9:
                    local, keep, moved = f, g, True
            w[c] = keep
            best = local
        if not moved:
            break
    return w, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a 5-fold run tag under work/runs/")
    ap.add_argument("--out", default=None, help="tag to write; default <run>_decoded")
    ap.add_argument("--zip", default=None, help="also write a CodaBench zip here")
    ap.add_argument("--folds", type=int, default=5, help="folds for the nested check")
    ap.add_argument("--span", type=float, default=1.5,
                    help="weights are searched over exp(-span..span)")
    ap.add_argument("--points", type=int, default=31)
    ap.add_argument("--force", action="store_true",
                    help="write even when the nested estimate says it does not help")
    a = ap.parse_args()

    src = RUNS / a.run
    oof = np.load(src / "oof_probs.npy")
    test = np.load(src / "test_probs.npy")
    y = truth()
    k = oof.shape[1]
    if oof.shape != (len(y), len(LABELS)):
        # a Task A run sharing work/runs/; the sweep hands us every directory
        print(f"skip {a.run}: OOF shape {oof.shape}, Task B needs "
              f"{(len(y), len(LABELS))}")
        return 1
    grid = np.exp(np.linspace(-a.span, a.span, a.points))

    base = macro_f1(y, oof.argmax(1), k)
    w, insample = fit_weights(oof, y, grid)

    # Nested: the weights never see the rows they are scored on.
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=SPLIT_SEED)
    pred = np.zeros(len(y), dtype=int)
    for tr, va in skf.split(oof, y):
        w_i, _ = fit_weights(oof[tr], y[tr], grid)
        pred[va] = (oof[va] * w_i).argmax(1)
    nested = macro_f1(y, pred, k)

    print(f"run {a.run}")
    print(f"  plain argmax   macro-F1 {base:.4f}")
    print(f"  tuned in-sample         {insample:.4f}   (optimistic, do not believe it)")
    print(f"  tuned nested            {nested:.4f}   <- the honest one")
    print(f"  nested gain             {nested - base:+.4f}")
    print("\n  weights:", {l: round(float(v), 3) for l, v in zip(LABELS, w)})

    before = np.bincount(test.argmax(1), minlength=k)
    after = np.bincount((test * w).argmax(1), minlength=k)
    print(f"\n  {'class':15s} {'argmax':>7s} {'tuned':>7s}   test rows")
    for i, l in enumerate(LABELS):
        print(f"  {l:15s} {before[i]:7d} {after[i]:7d}")

    if nested <= base and not a.force:
        print(f"\nnested estimate does not beat plain argmax ({nested:.4f} vs "
              f"{base:.4f}); writing nothing. Submit {a.run} as it is.")
        return 1

    out = RUNS / (a.out or f"{a.run}_decoded")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_probs.npy", test * w)
    ids = pd.read_csv(ROOT / "data" / "multiclass_validation_inputs.csv")["id"]
    sub = pd.DataFrame({"id": ids, "label": [LABELS[i] for i in (test * w).argmax(1)]})
    sub.to_csv(out / "predictions.csv", index=False)
    (out / "decode.json").write_text(json.dumps(
        {"source": a.run, "weights": dict(zip(LABELS, w.tolist())),
         "argmax": base, "insample": insample, "nested": nested}, indent=2))
    print(f"\nwrote {out/'predictions.csv'}")

    if a.zip:
        z = pathlib.Path(a.zip)
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            f.write(out / "predictions.csv", "predictions.csv")
        print(f"wrote {z}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
