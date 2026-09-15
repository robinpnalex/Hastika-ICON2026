"""Validate a predictions.csv against the task rules and zip it for CodaBench."""
import argparse
import pathlib
import zipfile

import pandas as pd

from .paths import PROJECT_ROOT, RAW_DATA_DIR

VALID = {"a": {"Hate", "Non-Hate"},
         "b": {"Gender", "Political", "Religion", "Geo-political", "Violence", "Others"}}
INPUTS = {"a": "binary_validation_inputs.csv", "b": "multiclass_validation_inputs.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--task", choices=["a", "b"], default="a")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inputs", default=None, help="override the reference input CSV")
    a = ap.parse_args()

    pred_path = pathlib.Path(a.pred)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / pred_path
    pred = pd.read_csv(pred_path)
    ref = pd.read_csv(a.inputs or RAW_DATA_DIR / INPUTS[a.task])

    assert list(pred.columns) == ["id", "label"], f"header must be id,label -- got {list(pred.columns)}"
    assert not pred["id"].duplicated().any(), "duplicate ids"
    bad = set(pred["label"]) - VALID[a.task]
    assert not bad, f"invalid labels: {bad}"
    missing = set(ref["id"]) - set(pred["id"])
    extra = set(pred["id"]) - set(ref["id"])
    assert not missing, f"{len(missing)} ids missing from predictions"
    assert not extra, f"{len(extra)} ids not in the input file"

    out = pathlib.Path(a.out or pred_path.with_suffix(".zip"))
    # arcname must be a bare predictions.csv -- no enclosing folder
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.csv")
    print(f"OK: {len(pred)} rows, {pred['label'].value_counts().to_dict()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
