"""Average test-probability matrices from several runs into one submission ZIP.

Every Gemma notebook saves `<name>_test_probs.npy` in its submissions folder, rows in
released-test-file order and columns in the task's label order:
    Task A  [Non-Hate, Hate]
    Task B  [Gemma's LABELS order: Gender, Geo-political, Others, Political, Religion, Violence]

Each file is weighted by the number of models it already averages, so 2 + 4 models gives
a true 6-model mean rather than half-and-half:

    python -m hastika.common.combine_probs --task a \
        --probs a28_gemma_test_probs.npy:2 a33_gemma_test_probs.npy:2 \
        --out submissions/task_a/final/final_task_a.zip
"""
import argparse
import pathlib
import zipfile

import numpy as np
import pandas as pd

from hastika.common.paths import RAW_DATA_DIR

TASKS = {
    "a": (["Non-Hate", "Hate"], "hastika_binary_test.csv"),
    "b": (["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"],
          "hastika_multiclass_test.csv"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--probs", nargs="+", required=True,
                    help="path[:n_models] -- n_models defaults to 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels, test_file = TASKS[args.task]
    ids = pd.read_csv(RAW_DATA_DIR / test_file)["id"]
    total, weight = None, 0
    for spec in args.probs:
        path, _, n = spec.partition(":")
        n = int(n or 1)
        p = np.load(path)
        assert p.shape == (len(ids), len(labels)), f"{path}: shape {p.shape}"
        assert np.allclose(p.sum(1), 1, atol=1e-3), f"{path}: rows do not sum to 1"
        share = np.bincount(p.argmax(1), minlength=len(labels)) / len(p)
        assert share.max() <= 0.9, f"{path}: collapsed, {share.max():.0%} in one class"
        print(f"{path}: {n} model(s), predicted shares "
              + " ".join(f"{l[:5]} {s:.1%}" for l, s in zip(labels, share)))
        total = p * n if total is None else total + p * n
        weight += n
    probs = total / weight

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame({"id": ids, "label": [labels[i] for i in probs.argmax(1)]})
    csv = out.with_name("predictions.csv")
    pred.to_csv(csv, index=False)
    np.save(out.with_name(out.stem + "_test_probs.npy"), probs)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "predictions.csv")
    counts = pred["label"].value_counts(normalize=True)
    print(f"{weight} models -> {out}: "
          + " ".join(f"{l[:5]} {counts.get(l, 0):.1%}" for l in labels))


if __name__ == "__main__":
    main()
