"""Validate a predictions.csv against the task rules and zip it for CodaBench.

Takes either a predictions CSV (--pred) or a raw probability matrix (--probs).
The second exists because a run that saved only test_probs.npy still has everything
needed for a submission: muril.py and baseline_svm.py both write column order
[Non-Hate, Hate] for task a and the LABELS order for task b, and the label is the
argmax, or for task a `p(Hate) > --threshold` so a tuned threshold can be applied
after the fact without retraining.
"""
import argparse
import pathlib
import zipfile

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT, RAW_DATA_DIR

VALID = {"a": {"Hate", "Non-Hate"},
         "b": {"Gender", "Political", "Religion", "Geo-political", "Violence", "Others"}}
INPUTS = {"a": "binary_validation_inputs.csv", "b": "multiclass_validation_inputs.csv"}
# column order written by muril.py / baseline_svm.py, matching their label encoders
CLASSES = {"a": ["Non-Hate", "Hate"],
           "b": ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="a predictions.csv with columns id,label")
    ap.add_argument("--probs", help="a test_probs.npy instead; labels are taken from it")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="task a only, with --probs: label Hate when p(Hate) > this")
    ap.add_argument("--task", choices=["a", "b"], default="a")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inputs", default=None, help="override the reference input CSV")
    a = ap.parse_args()
    if bool(a.pred) == bool(a.probs):
        ap.error("give exactly one of --pred or --probs")

    ref = pd.read_csv(a.inputs or RAW_DATA_DIR / INPUTS[a.task])
    if a.probs:
        probs_path = pathlib.Path(a.probs)
        if not probs_path.is_absolute():
            probs_path = PROJECT_ROOT / probs_path
        P = np.load(probs_path)
        assert len(P) == len(ref), f"{len(P)} rows of probabilities vs {len(ref)} inputs"
        if a.task == "a":
            hate = CLASSES["a"].index("Hate")
            labels = np.where(P[:, hate] > a.threshold, "Hate", "Non-Hate")
        else:
            labels = [CLASSES["b"][i] for i in P.argmax(1)]
        pred = pd.DataFrame({"id": ref["id"], "label": labels})
        pred_path = probs_path
        print(f"labels from {probs_path.name}"
              + (f" at p(Hate) > {a.threshold}" if a.task == "a" else " by argmax"))
    else:
        pred_path = pathlib.Path(a.pred)
        if not pred_path.is_absolute():
            pred_path = PROJECT_ROOT / pred_path
        pred = pd.read_csv(pred_path)

    assert list(pred.columns) == ["id", "label"], f"header must be id,label -- got {list(pred.columns)}"
    assert not pred["id"].duplicated().any(), "duplicate ids"
    bad = set(pred["label"]) - VALID[a.task]
    assert not bad, f"invalid labels: {bad}"
    missing = set(ref["id"]) - set(pred["id"])
    extra = set(pred["id"]) - set(ref["id"])
    assert not missing, f"{len(missing)} ids missing from predictions"
    assert not extra, f"{len(extra)} ids not in the input file"

    out = pathlib.Path(a.out or pred_path.with_suffix(".zip"))
    if a.probs:
        # the CSV inside the ZIP has to exist on disk to be added to the archive
        tmp = out.with_name(out.stem + "_predictions.csv")
        pred.to_csv(tmp, index=False)
        pred_path = tmp
    # arcname must be a bare predictions.csv -- no enclosing folder
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.csv")
    print(f"OK: {len(pred)} rows, {pred['label'].value_counts().to_dict()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
