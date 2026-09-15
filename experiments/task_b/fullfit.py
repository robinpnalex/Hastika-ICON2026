"""Five-seed full fits of every Task B idea that was not measurably bad.

Each arm trains five models on ALL 3,143 labelled rows, no held-out split, and
averages their probabilities. Five models each seeing 100% of the data, which is
strictly more data per model than the five-fold runs got while keeping the
averaging that made those runs beat a single 85% model.

WHAT YOU DO NOT GET, AND WHY
----------------------------
A score. Every labelled row is in training, so there is nothing left to measure
against, and the official validation file has no labels and never will. That is
not a limitation of this script; it is arithmetic.

So the ranking column here is the **holdout** score each idea already earned on
the fixed 472-row split, which is the best available evidence of which arm is
worth submitting. The full fits are better versions of those same ideas, not
new evidence about them.

In place of a score, each arm reports two diagnostics that need no labels:

  * predicted distribution against the training prior. A model that has
    collapsed onto Gender shows it here and nowhere else.
  * agreement with the already-submitted predictions, which scored 0.5922 on
    CodaBench. An arm that agrees with it on 95% of rows will score near it; one
    that agrees on 70% is a genuinely different bet.

DROPPED
-------
b_xlmr (0.5561) and b_mdeberta (0.1002). The first lost to stock MuRIL by four
points, the second collapsed to roughly one class because 3e-5 is too high for
it. Neither is worth 1.5 h of GPU.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
LOGS = ROOT / "artifacts" / "logs"
MURIL = "google/muril-base-cased"
ABUSIVE = "Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL"
LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]
REFERENCE = ROOT / "submissions" / "b_tapt_5f" / "predictions.csv"

# tag, flags, holdout score already measured, note. Priority order: the budget
# guard cuts from the bottom, so the weakest idea is the one that gets dropped.
ARMS = [
    ("f_tapt",    ["--model", "artifacts/runs/tapt-muril"], 0.5972, "MLM-adapted MuRIL"),
    ("f_base",    ["--model", MURIL],                     0.5948, "stock MuRIL"),
    ("f_rdrop",   ["--model", MURIL, "--rdrop", "0.5"],   0.5845, "R-Drop 0.5"),
    ("f_focal",   ["--model", MURIL, "--loss", "focal"],  0.5779, "focal loss, gamma 2"),
    ("f_abusive", ["--model", ABUSIVE],                   0.5718, "abusive-tuned warm start"),
    ("f_hing",    ["--model", "l3cube-pune/hing-roberta"], 0.5678, "HingRoBERTa"),
]


def sh(cmd, log=None, required=False):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    fh = open(log, "w") if log else None
    p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        if fh:
            fh.write(line)
    p.wait()
    if fh:
        fh.close()
    if p.returncode:
        if required:
            raise RuntimeError(f"exit {p.returncode}: {cmd}")
        print(f"\n{'!'*70}\nARM FAILED (exit {p.returncode}) -- continuing\n{'!'*70}",
              flush=True)
    return p.returncode


def diagnostics(tag, prior):
    """Everything that can be said about a full fit without any labels."""
    pred = RUNS / tag / "predictions.csv"
    if not pred.exists():
        return None
    df = pd.read_csv(pred)
    dist = df["label"].value_counts(normalize=True)
    out = {"dist": {l: round(100 * float(dist.get(l, 0.0)), 1) for l in LABELS}}
    out["max_drift"] = round(max(abs(out["dist"][l] - prior[l]) for l in LABELS), 1)
    if REFERENCE.exists():
        ref = pd.read_csv(REFERENCE).set_index("id")["label"]
        j = df.set_index("id")["label"].reindex(ref.index)
        out["agree_with_submitted"] = round(100 * float((j == ref).mean()), 1)
    return out


def write_results(res, out, prior, spent):
    L = ["# Task B -- five-seed full fits", "",
         f"Elapsed {spent/3600:.2f} h. Every arm trained 5 models on all 3,143 rows "
         "and averaged them.", "",
         "**There are no scores here and there cannot be.** Every labelled row was used "
         "for training.",
         "The `holdout` column is what each idea already scored on the fixed 472-row "
         "split, and it is",
         "the only evidence of which arm is better. Rank on it.", "",
         "`agree` is how often the arm's 395 predictions match the submission that "
         "scored 0.5922 on",
         "CodaBench. High agreement means expect a similar score. `drift` is the largest "
         "gap between",
         "the arm's predicted class rates and the training prior, in points: a big number "
         "means the",
         "model has skewed toward some class and is worth distrusting.", "",
         "| arm | holdout | agree | drift | zip | note |", "|---|---|---|---|---|---|"]
    for tag, r in sorted(res.items(), key=lambda kv: -(kv[1].get("holdout") or 0)):
        d = r.get("diag") or {}
        L.append(f"| `{tag}` | {r.get('holdout', '-')} | "
                 f"{d.get('agree_with_submitted', '-')}% | {d.get('max_drift', '-')} | "
                 f"`{pathlib.Path(r['zip']).name if r.get('zip') else '-'}` | "
                 f"{r.get('note', '')} |")
    L += ["", "## Predicted class rates (%), against the training prior", "",
          "| arm | " + " | ".join(l[:6] for l in LABELS) + " |",
          "|---" * (len(LABELS) + 1) + "|",
          "| *prior* | " + " | ".join(f"{prior[l]:.1f}" for l in LABELS) + " |"]
    for tag, r in sorted(res.items(), key=lambda kv: -(kv[1].get("holdout") or 0)):
        d = (r.get("diag") or {}).get("dist")
        if d:
            L.append(f"| `{tag}` | " + " | ".join(f"{d[l]:.1f}" for l in LABELS) + " |")
    pathlib.Path(out, "RESULTS.md").write_text("\n".join(L) + "\n")
    pathlib.Path(out, "results.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=10.5)
    ap.add_argument("--reserve-min", type=float, default=15)
    ap.add_argument("--seeds", default="42 43 44 45 46")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-tapt", action="store_true",
                    help="assume artifacts/runs/tapt-muril exists or drop the f_tapt arm")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subs = out / "subs"; subs.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    left = lambda: a.budget_hours * 3600 - (time.time() - t0) - a.reserve_min * 60

    train = pd.read_csv(ROOT / "data" / "raw" / "multiclass_train.csv")
    pc = train["Hate Category"].value_counts(normalize=True)
    prior = {l: 100 * float(pc.get(l, 0.0)) for l in LABELS}

    # The adaptation pass is GPU work like any other and has to clear the budget,
    # or a short run starts a 25-minute job it can never finish.
    if not a.skip_tapt and not (RUNS / "tapt-muril").exists():
        if left() > 1800:
            sh([sys.executable, "-u", "-m", "hastika.task_b.tapt", "--out",
                "artifacts/runs/tapt-muril"], log=LOGS / "tapt.log")
        else:
            print(f"skipping tapt: {left()/60:.0f} min left, needs ~25", flush=True)

    res, times = {}, []
    arms = [x for x in ARMS if a.only is None or x[0] in a.only]
    for tag, flags, hold, note in arms:
        model = flags[flags.index("--model") + 1]
        if model.startswith("artifacts/") and not (ROOT / model).is_dir():
            print(f"skipping {tag}: {model} missing", flush=True)
            res[tag] = {"holdout": hold, "note": note + " (skipped: no checkpoint)"}
            continue
        est = np.median(times) if times else 5600
        if left() < est:
            print(f"skipping {tag}: {left()/60:.0f} min left, needs ~{est/60:.0f}",
                  flush=True)
            res[tag] = {"holdout": hold, "note": note + " (skipped: out of budget)"}
            write_results(res, out, prior, time.time() - t0)
            continue
        t = time.time()
        log = LOGS / f"{tag}.log"
        rc = sh([sys.executable, "-u", "-m", "hastika.task_b.train", "--tag", tag,
                 "--folds", "1",
                 "--seeds", *a.seeds.split(), "--epochs", str(a.epochs), *flags], log=log)
        if rc == 0:
            times.append(time.time() - t)
        print(f"[{tag}] {(time.time()-t)/60:.0f} min, {left()/60:.0f} min left", flush=True)

        res[tag] = {"holdout": hold, "note": note}
        if rc != 0:
            res[tag]["note"] += " (failed; no artifact packaged)"
            write_results(res, out, prior, time.time() - t0)
            continue
        z = subs / f"{tag}.zip"
        if sh([sys.executable, "-m", "hastika.common.submission", "--task", "b",
               "--pred", str(RUNS / tag / "predictions.csv"), "--out", str(z)]) == 0:
            res[tag]["zip"] = str(z)
        res[tag]["diag"] = diagnostics(tag, prior)
        write_results(res, out, prior, time.time() - t0)

    for f in LOGS.glob("*.log"):
        sh(["cp", str(f), str(out)])
    write_results(res, out, prior, time.time() - t0)
    print(f"\nDONE in {(time.time()-t0)/3600:.2f} h. Zips:", flush=True)
    for z in sorted(subs.glob("*.zip")):
        print(f"  {z}")


if __name__ == "__main__":
    main()
