"""How does a holdout ensemble grow as models are added? Run locally on downloaded
holdout probabilities (any number of runs and sessions).

Each file is a `holdout_probs.npy` with its `holdout_ids.csv` beside it, as every Gemma
notebook writes under `runs/<name>/`. The script checks that each file covers the
committed holdout rows, then reports:

  * every single model's macro-F1
  * the mean macro-F1 of random k-model averages, k = 1 .. N (the growth curve)
  * all N averaged, with a paired bootstrap against the best single group you name

    python -m hastika.common.ensemble_curve --task a \
        --probs dl/a39_t1/runs/*/holdout_probs.npy dl/a39_t2/runs/*/holdout_probs.npy \
        --reference dl/a35/runs/tapt_s4*/holdout_probs.npy
"""
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from hastika.common.paths import PROJECT_ROOT
from hastika.task_b.combined_holdout import paired_bootstrap

TASKS = {"a": ("task_a_combined_holdout", "Label", ["Non-Hate", "Hate"]),
         "b": ("task_b_combined_holdout", "Hate Category",
               ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"])}


def load(paths, ho_ids):
    out = {}
    for p in map(pathlib.Path, paths):
        ids = pd.read_csv(p.parent / "holdout_ids.csv")["id"].tolist()
        if ids != ho_ids:
            print(f"skip {p}: not this task's holdout rows")
            continue
        probs = np.load(p)
        share = np.bincount(probs.argmax(1), minlength=probs.shape[1]).max() / len(probs)
        if share > 0.9:
            print(f"skip {p}: collapsed ({share:.0%} one class)")
            continue
        key = "/".join(p.parts[-5:-1])          # session folder / output / runs / model
        out[key if key not in out else str(p)] = probs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--reference", nargs="*", default=[], help="the current best ensemble's files")
    ap.add_argument("--samples", type=int, default=200, help="random subsets per k")
    args = ap.parse_args()

    split, col, labels = TASKS[args.task]
    ho = pd.read_csv(PROJECT_ROOT / "data/derived" / split / "holdout.csv")
    y = ho[col].map(labels.index).to_numpy()
    models = load(args.probs, ho["id"].tolist())
    assert models, "no usable holdout files"
    f1 = lambda p: f1_score(y, p.argmax(1), average="macro")

    print(f"{len(models)} models on {len(y)} holdout rows")
    for k, p in sorted(models.items()):
        print(f"  {f1(p):.4f}  {k}")
    keys, rng = list(models), np.random.default_rng(0)
    print("\nk-model average (mean over random subsets):")
    for k in range(1, len(keys) + 1):
        combos = list(itertools.combinations(keys, k))
        if len(combos) > args.samples:
            combos = [tuple(rng.choice(keys, k, replace=False)) for _ in range(args.samples)]
        scores = [f1(np.mean([models[c] for c in combo], 0)) for combo in combos]
        print(f"  k={k:2d}  {np.mean(scores):.4f}  (min {min(scores):.4f}, max {max(scores):.4f})")
    full = np.mean(list(models.values()), 0)
    print(f"\nall {len(keys)} averaged: {f1(full):.4f}")
    if args.reference:
        ref = load(args.reference, ho["id"].tolist())
        if ref:
            r = np.mean(list(ref.values()), 0)
            b = paired_bootstrap(y, full, r)
            print(f"reference ({len(ref)} models): {f1(r):.4f}")
            print(f"all - reference: {b['diff']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, "
                  f"{b['ci95'][1]:+.4f}]  P(better) {b['p_better']:.2f}")


if __name__ == "__main__":
    main()
