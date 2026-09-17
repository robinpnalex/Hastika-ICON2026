"""Stack one change at a time on the Run 9 recipe, full data, one zip per arm.

THE RECIPE, UNCHANGED
---------------------
Every arm is byte-for-byte `b_reinit1_rdrop_full`, which scored 0.6410 on
CodaBench, except for the single flag named in its row:

    tapt.py  --corpus multiclass_train + offenseval_kn --val-frac 0
             --min-words 1 --no-dedupe --epochs 8
    train.py --folds 1 --no-dedupe --reinit-layers 1 --rdrop 0.5
             --aux-weight 0 --seeds 42 43 44 45 46 --epochs 6

`--folds 1` means every seed trains on all 3,159 rows. So, as in Run 9, THERE IS
NO LOCAL SCORE and there cannot be one: nothing is held out and the official
validation labels are never released. Each arm's zip goes to CodaBench and the
leaderboard is the readout. Two diagnostics that need no labels are printed
instead, and neither is a score:

  * predicted class distribution against the training prior. An arm that has
    collapsed onto Gender shows it here and nowhere else.
  * agreement with a reference submission whose CodaBench score you know. An arm
    agreeing on 97% of 395 rows will land near it; one agreeing on 70% is a
    different bet and worth a submission slot.

READ THE RESULTS WITH THIS IN MIND
----------------------------------
The 395-row validation set puts ~23 rows in Geo-political and ~28 in Violence,
and macro-F1 weights those equally with Gender's ~170. Two rows changing in a
23-row class moves macro-F1 by about a point. Rank arms by the leaderboard, but
do not read a 1-point gap as a result.

stopwords and stem additionally break the match with the text TAPT adapted to;
see normalize.py. Measured on this corpus, the 60-word frequency stoplist
contains bjp, congress, dagar and desha -- the Political and Gender signal --
so that arm is expected to lose badly, and that expectation is the test.
"""
import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
LOGS = ROOT / "artifacts" / "logs"
LABELS = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]
TAPT_OUT = "artifacts/runs/tapt-stack"
POLARITY = "artifacts/runs/polarity_map.json"

SEEDS = ["42", "43", "44", "45", "46"]

# Ordered by how much each is worth, because the budget guard skips from the
# bottom. Everything below the default list stays runnable via --arms; it is off
# by default because the evidence already available says it will lose:
#
#   s_stopwords  the 60-word frequency stoplist contains bjp, congress, dagar and
#                desha, which is the Political and Gender signal. Nothing to learn.
#   s_stem       char normalization measured -0.003 and transliteration-lite
#                +0.001 on this corpus. Wordpiece already splits madthare into
#                stem and suffixes, so this does that job worse and breaks the
#                match with the text TAPT adapted to.
#   s_polarity   the same shape as b_abusive, an abusive-tuned warm start, which
#                scored 0.5718 against stock MuRIL's 0.5948. Also the only arm
#                carrying a rules question, since it reads external LABELS.
#   s_control    reruns the recipe unchanged; only needed to supply
#                test_probs.npy for the TF-IDF blend.
#
#   tag              extra train.py flags                      est. minutes
ARMS = [
    ("s_seeds10",   ["--seeds", *SEEDS, "47", "48", "49", "50", "51"],  226),
    ("s_epochs10",  ["--epochs-override", "10"],                        188),
    ("s_nofgm",     ["--no-fgm"],                                        73),
    ("s_tags",      ["--tags"],                                         113),
    ("s_stopwords", ["--text-transform", "stopwords"],                  113),
    ("s_stem",      ["--text-transform", "stem"],                       113),
    ("s_polarity",  ["--text-transform", "polarity",
                     "--polarity-map", POLARITY],                       113),
    ("s_control",   [],                                                 113),
]
BASE = ["--folds", "1", "--no-dedupe", "--reinit-layers", "1", "--rdrop", "0.5",
        "--aux-weight", "0", "--seeds", *SEEDS, "--epochs", "6"]


def base_for(flags):
    """BASE with --seeds / --epochs replaced when an arm overrides them.

    argparse would take the last value anyway, but a command line carrying two
    --seeds is unreadable in a log, and these logs are the record of what ran.
    """
    out, flags = list(BASE), list(flags)
    if "--epochs-override" in flags:
        i = flags.index("--epochs-override")
        out[out.index("--epochs") + 1] = flags[i + 1]
        del flags[i:i + 2]
    if "--seeds" in flags:
        i = flags.index("--seeds")
        j = i + 1
        while j < len(flags) and not flags[j].startswith("--"):
            j += 1
        seeds = flags[i + 1:j]
        k = out.index("--seeds")
        e = k + 1
        while e < len(out) and not out[e].startswith("--"):
            e += 1
        out[k + 1:e] = seeds
        del flags[i:j]
    return out, flags


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
    if p.returncode and required:
        raise RuntimeError(f"exit {p.returncode}: {cmd}")
    if p.returncode:
        print(f"\n{'!'*70}\nARM FAILED (exit {p.returncode}) -- continuing\n{'!'*70}",
              flush=True)
    return p.returncode


def diagnostics(tag, reference):
    """Label-free checks. Neither is a score."""
    pred = RUNS / tag / "predictions.csv"
    if not pred.exists():
        return None
    p = pd.read_csv(pred)
    train = pd.read_csv(ROOT / "data" / "raw" / "multiclass_train.csv")
    prior = train["Hate Category"].value_counts(normalize=True)
    rate = p["label"].value_counts(normalize=True)
    drift = sum(abs(rate.get(c, 0.0) - prior.get(c, 0.0)) for c in LABELS) * 100
    agree = None
    if reference.exists():
        r = pd.read_csv(reference)
        m = p.merge(r, on="id", suffixes=("", "_ref"))
        agree = float((m["label"] == m["label_ref"]).mean())
    return {"tag": tag, "drift": drift, "agree": agree,
            "rates": {c: float(rate.get(c, 0.0)) for c in LABELS}}


def write_results(rows, out, spent, stage):
    L = ["# Task B -- one change stacked on the 0.6410 recipe", "",
         f"Elapsed {spent/3600:.2f} h. Stage: {stage}.", "",
         "Every arm is the Run 9 recipe with ONE flag added, trained on all 3,159",
         "rows with five seeds. There is no local score and there cannot be one:",
         "nothing is held out. Upload each zip and rank on CodaBench.", "",
         "`agree` is the share of the 395 predictions matching the reference",
         "submission. `drift` is the total absolute gap between the arm's predicted",
         "class rates and the training prior, in points; a large number means the",
         "arm has collapsed onto a class and is worth checking before submitting.", "",
         "| arm | agree | drift | " + " | ".join(l[:6] for l in LABELS) + " |",
         "|---|---|---|" + "---|" * len(LABELS)]
    for r in rows:
        if r is None:
            continue
        a = "-" if r["agree"] is None else f"{r['agree']:.3f}"
        L.append(f"| `{r['tag']}` | {a} | {r['drift']:.1f} | "
                 + " | ".join(f"{r['rates'][c]:.3f}" for c in LABELS) + " |")
    train = pd.read_csv(ROOT / "data" / "raw" / "multiclass_train.csv")
    prior = train["Hate Category"].value_counts(normalize=True)
    L += ["| *training prior* | | | "
          + " | ".join(f"{prior.get(c, 0.0):.3f}" for c in LABELS) + " |", "",
          "Reminder: 23 of the 395 validation rows are Geo-political and 28 are",
          "Violence, and macro-F1 weights them like Gender's 170. A 1-point gap on",
          "that set is two rows and is not a result."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=10.5)
    ap.add_argument("--reserve-min", type=float, default=20)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--arms", nargs="*",
                    default=["s_seeds10", "s_epochs10", "s_nofgm", "s_tags"],
                    help="see the ARMS table for what is off by default and why. "
                         "s_control reruns Run 9 unchanged; include it only if you want "
                         "its test_probs.npy for the TF-IDF blend, at 113 min")
    ap.add_argument("--reference", default="submissions/b_tapt_5f/predictions.csv",
                    help="submission whose CodaBench score you know, for the agreement "
                         "column. Point this at Run 9's predictions.csv if you kept it")
    ap.add_argument("--blend-weight", type=float, default=0.25,
                    help="SVM share in the extra TF-IDF blend zip; 0 disables it")
    ap.add_argument("--skip-tapt", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subs = out / "subs"; subs.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    py, t0 = sys.executable, time.time()
    left = lambda: a.budget_hours * 3600 - (time.time() - t0) - a.reserve_min * 60
    reference = ROOT / a.reference
    rows = []

    # ---- stage 0: the encoder, built once at Run 9's settings ---------------
    if not a.skip_tapt and not (ROOT / TAPT_OUT).is_dir():
        sh([py, "-u", "-m", "hastika.task_b.tapt",
            "--corpus", "data/raw/multiclass_train.csv", "data/external/offenseval_kn.csv",
            "--val-frac", "0", "--min-words", "1", "--no-dedupe", "--epochs", "8",
            "--out", TAPT_OUT], log=LOGS / "stack_tapt.log", required=True)

    # ---- stage 0b: the polarity tagger, only if that arm is selected --------
    if "s_polarity" in a.arms and not (ROOT / POLARITY).exists():
        if left() > 1800:
            sh([py, "-u", "experiments/task_b/polarity_tagger.py", "--out", POLARITY],
               log=LOGS / "stack_polarity.log")
        else:
            print("skip polarity tagger: out of budget", flush=True)

    # ---- stage 1: one arm per stacked change -------------------------------
    for tag, flags, est in ARMS:
        if tag not in a.arms:
            continue
        if tag == "s_polarity" and not (ROOT / POLARITY).exists():
            print(f"skip {tag}: no polarity map", flush=True)
            continue
        if left() < est * 60:
            print(f"skip {tag}: {left()/60:.0f} min left, needs ~{est}", flush=True)
            continue
        base, extra = base_for(flags)
        rc = sh([py, "-u", "-m", "hastika.task_b.train", "--tag", tag,
                 "--model", TAPT_OUT, *base, *extra], log=LOGS / f"{tag}.log")
        if rc == 0:
            sh([py, "-m", "hastika.common.submission", "--task", "b",
                "--pred", str(RUNS / tag / "predictions.csv"),
                "--out", str(subs / f"{tag}.zip")])
        rows.append(diagnostics(tag, reference))
        write_results(rows, out, time.time() - t0, "1 (arms)")

    # ---- stage 2: the TF-IDF blend, no GPU ---------------------------------
    # Needs an unstacked neural run's test_probs. Use s_control if it ran, else
    # drop in Run 9's test_probs.npy at artifacts/runs/s_control/.
    ctrl = RUNS / "s_control" / "test_probs.npy"
    if a.blend_weight > 0 and ctrl.exists():
        sh([py, "-u", "-m", "hastika.models.baseline_svm", "--task", "b",
            "--demojize", "--tag", "svm_b_stack"], log=LOGS / "stack_svm.log")
        svm = RUNS / "svm_b_stack" / "test_probs.npy"
        if svm.exists():
            n = np.load(ctrl)
            s = np.load(svm)
            blend = (1 - a.blend_weight) * n + a.blend_weight * s
            tag = "s_blend"
            (RUNS / tag).mkdir(parents=True, exist_ok=True)
            ids = pd.read_csv(ROOT / "data" / "raw" / "multiclass_validation_inputs.csv")["id"]
            pd.DataFrame({"id": ids,
                          "label": [LABELS[i] for i in blend.argmax(1)]}
                         ).to_csv(RUNS / tag / "predictions.csv", index=False)
            sh([py, "-m", "hastika.common.submission", "--task", "b",
                "--pred", str(RUNS / tag / "predictions.csv"),
                "--out", str(subs / f"{tag}.zip")])
            rows.append(diagnostics(tag, reference))
    elif a.blend_weight > 0:
        print("skip blend: no unstacked test_probs.npy; run with --arms ... s_control",
              flush=True)

    for f in LOGS.glob("*.log"):
        sh(["cp", str(f), str(out)])
    write_results(rows, out, time.time() - t0, "done")
    print(f"\nDONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
