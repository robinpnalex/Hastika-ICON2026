"""Build Run 21's test-set ZIPs from its saved output, without retraining.

Run 21's first execution trained both Gemma-4-12B full fits but its packaging cell looked
for `test_probs.npy`, while llm_classifier.py writes `<input stem>_probs.npy`
(`test_inputs_probs.npy`). The probabilities were saved; only the ZIP was not written.
This reads them back:

    runs/<model>_full_s<seed>/test_inputs_probs.npy + test_inputs_ids.csv   (LLM seeds)
    submissions/b21_muril_test_probs.npy                                    (MuRIL, 5 seeds)
    result.json                                                             (holdout decision)

and writes b21_llm.zip, b21_ensemble.zip and b21_muril.zip, marking the arm the holdout
rule chose as RECOMMENDED_. Each ZIP holds one bare predictions.csv (id,label).

    python -m hastika.task_b.package_b21 --b21 path/to/b21_outputs --out submissions/task_b/run21
"""
import argparse
import glob
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd

from hastika.task_b.combined_holdout import LABELS, TEST_FILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b21", required=True, help="Run 21's b21_outputs directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b21, out = pathlib.Path(args.b21), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test_ids = pd.read_csv(TEST_FILE)["id"].tolist()

    seeds = []
    for p in sorted(glob.glob(str(b21 / "runs" / "*_full_s*" / "test_inputs_probs.npy"))):
        p = pathlib.Path(p)
        ids = pd.read_csv(p.parent / "test_inputs_ids.csv")["id"].tolist()
        assert ids == test_ids, f"{p.parent.name}: ids differ from {TEST_FILE.name}"
        seeds.append(np.load(p))
        print(f"LLM seed  {p.parent.name}  {seeds[-1].shape}")
    full = {}
    if seeds:
        full["llm"] = np.mean(seeds, 0)
    mp = b21 / "submissions" / "b21_muril_test_probs.npy"
    if mp.exists():
        full["muril"] = np.load(mp)
        print(f"MuRIL     {mp.name}  {full['muril'].shape}")
    if "llm" in full and "muril" in full:
        full["ensemble"] = (full["llm"] + full["muril"]) / 2
    assert full, f"nothing to package under {b21}"

    rj = b21 / "result.json"
    recommended = json.load(open(rj)).get("recommended") if rj.exists() else None
    print(f"holdout rule recommended: {recommended}")
    prior = pd.concat([pd.read_csv(TEST_FILE.parent / "multiclass_train.csv"),
                       pd.read_csv(TEST_FILE.parent / "hastika_multiclass_validation.csv")]
                      )["Hate Category"].value_counts(normalize=True)
    for name, probs in full.items():
        assert probs.shape == (len(test_ids), len(LABELS)), probs.shape
        pred = pd.DataFrame({"id": test_ids, "label": [LABELS[i] for i in probs.argmax(1)]})
        assert pred["id"].is_unique and set(pred["label"]) <= set(LABELS)
        np.save(out / f"b21_{name}_test_probs.npy", probs)
        csv = out / f"predictions_{name}.csv"
        pred.to_csv(csv, index=False)
        tag = "RECOMMENDED_" if name == recommended else ""
        with zipfile.ZipFile(out / f"{tag}b21_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(csv, "predictions.csv")
        counts = pred["label"].value_counts(normalize=True)
        print(f"{tag}b21_{name}.zip  predicted/prior: "
              + "  ".join(f"{l[:5]} {counts.get(l, 0):.1%}/{prior[l]:.1%}" for l in LABELS))
    print("wrote", out)


if __name__ == "__main__":
    main()
