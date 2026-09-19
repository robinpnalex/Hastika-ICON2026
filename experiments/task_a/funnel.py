"""Task A: screen every component-level idea cheaply, confirm the leaders, then ship one.

THE PROBLEM THIS SOLVES
-----------------------
There are eight or so ideas queued for Task A and each is worth a point or two. The
CodaBench validation set is 806 rows and one score there carries about 1.5 points of
standard deviation, so a sequence of full-data submissions cannot separate them: most
gaps come back inside noise and nothing is learned. Running each idea as its own
five-fold notebook does separate them, but costs ~160 min per arm, about 30 hours for the
queue, and still never tests any two together.

THE FUNNEL
----------
Three stages, each answering a different question, cheapest first.

  stage 1  every arm on the fixed 15% holdout          ~33 min per arm
  stage 2  the leaders re-run on five folds            ~160 min per arm
  stage 3  the winner's blend weight and threshold
           fitted on its OOF, then a full-data refit   ~35 min

Every arm is also packaged as a CodaBench ZIP under subs/, because muril.py writes
predictions.csv whatever --folds is. A five-fold arm's predictions are the average of its
five fold models, which is a legitimate submission -- Task B's b_tapt_5f was exactly that.
A holdout arm's predictions come from one model trained on 85% of the rows, so it is
weaker, but it is there if you want it.

Task B ran this exact funnel and recorded the result: "for all four promoted arms the
five-fold order matched the holdout order exactly, while the level dropped by roughly
0.022 in three of the four cases." So the holdout RANKS correctly and READS high. Stage 1
is therefore used to order arms, never to report a number.

WHY THE ARMS ARE WHAT THEY ARE
------------------------------
Every arm changes the MuRIL component only. The TF-IDF/SVM half of the blend is fixed and
its OOF is computed once, because nothing queued touches it.

Arms are component-level because that is the dependency order: a blend weight depends on
how strong its components are, so it cannot be fitted until the component is settled.
Fitting the weight first and then changing the encoder would invalidate the weight.

Skipped deliberately, all measured and dead -- see docs/IDEAS.md: gazetteer and profanity
features, stemming, stopword removal, focal loss, class weighting, frozen embeddings.

RESUMABLE, AND SHARED WITH RUN 11
---------------------------------
Every stage skips work whose output already exists, so a session that dies partway can be
re-run and will continue rather than restart. RESULTS.md is rewritten after every arm.

The five-fold control is tagged `f_control`, which is the same tag and the same
configuration that notebooks/task_a/11_reinit1_full_data.ipynb uses for its MuRIL arm.
Whichever runs second finds the file and skips the 160 minutes. Nothing else is shared:
each arm trains exactly the one configuration it is testing, and the control is trained
once for all seven arms rather than once per arm.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "artifacts" / "runs"
LOGS = ROOT / "artifacts" / "logs"
MURIL = "google/muril-base-cased"
TAPT_OUT = "artifacts/runs/tapt-task-a"

# Shared by every MuRIL arm. --select last because `best` picks each fold's checkpoint
# using the rows it then reports, which flatters every arm equally but is still a lie.
COMMON = ["--epochs", "6", "--bs", "8", "--grad-accum", "2", "--eval-bs", "32",
          "--select", "last", "--seeds", "42", "--reinit-layers", "1"]

#   name            flags that differ from COMMON                      holdout  5fold  needs
ARMS = [
    ("control",     ["--model", MURIL],                                    33,   160, None),
    ("reinit2",     ["--model", MURIL, "--reinit-layers", "2"],            33,   160, None),
    ("tapt",        ["--model", TAPT_OUT],                                 33,   160, "tapt"),
    ("ext_mix",     ["--model", MURIL, "--external",
                     "--external-mode", "mix"],                            48,   240, None),
    ("ext_stage",   ["--model", MURIL, "--external",
                     "--external-mode", "stage"],                          40,   195, None),
    ("epochs10",    ["--model", MURIL, "--epochs", "10"],                  55,   265, None),
    ("large",       ["--model", "google/muril-large-cased", "--no-fgm",
                     "--lr", "1e-5", "--bs", "4", "--grad-accum", "4",
                     "--eval-bs", "16"],                                  100,   480, None),
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
    if p.returncode and required:
        raise RuntimeError(f"exit {p.returncode}: {cmd}")
    if p.returncode:
        print(f"\n{'!'*70}\nARM FAILED (exit {p.returncode}) -- continuing\n{'!'*70}",
              flush=True)
    return p.returncode


def dedup_flags(base, extra):
    """Later flags win. argparse would do this anyway, but a command line carrying
    --epochs twice is unreadable and these logs are the record of what ran."""
    out = list(base)
    i = 0
    while i < len(extra):
        if extra[i].startswith("--") and extra[i] in out:
            j = out.index(extra[i])
            k = j + 1
            while k < len(out) and not out[k].startswith("--"):
                k += 1
            del out[j:k]
        i += 1
    return out + extra


def holdout_score(tag):
    """The unbiased holdout number: with --select last, the plain line is it."""
    p = LOGS / f"a_{tag}.log"
    if not p.exists():
        return None
    t = p.read_text()
    m = (re.search(r"holdout macro-F1 ([\d.]+) with the last checkpoint", t)
         or re.search(r"holdout macro-F1 ([\d.]+)\s+acc", t)
         # fallback: muril.py printed no holdout score before 2026-09-20, but
         # train_fold has always logged the selected checkpoint's own number
         or re.search(r"\[s\d+holdout\] best [\d.]+\s+last ([\d.]+)", t))
    return float(m.group(1)) if m else None


def oof_score(tag, y):
    from sklearn.metrics import f1_score
    p = RUNS / f"f_{tag}" / "oof_probs.npy"
    if not p.exists():
        return None
    return float(f1_score(y, np.load(p).argmax(1), average="macro"))


def healthy(tag):
    """Predicted class balance, the only failure detectable without labels.

    Task A is 49% Hate. An arm calling one class on over 90% of rows has collapsed,
    which is what mDeBERTa did on Task B at a learning rate that suited the base model.
    """
    p = RUNS / f"a_{tag}" / "holdout_probs.npy"
    if not p.exists():
        return None
    hp = np.load(p)
    m = hp.sum(1) > 0
    return float(hp[m].argmax(1).mean()) if m.any() else None


def write_results(res, out, spent, stage):
    L = ["# Task A -- funnel over the component-level ideas", "",
         f"Elapsed {spent/3600:.2f} h. Stage: {stage}.", "",
         "Every arm changes the MuRIL component only; the TF-IDF/SVM half of the blend is",
         "fixed. `holdout` is the fixed 15% split, 960 rows: it RANKS arms correctly but",
         "reads about two points high, measured on Task B. `5-fold` is 6,401 out-of-fold",
         "rows, noise about 0.6 points, and is the number to report.", "",
         "`hate_rate` is the predicted positive rate on the holdout; the training prior is",
         "0.491 and anything outside 0.10-0.90 has collapsed.", "",
         "| arm | holdout | 5-fold | hate_rate | vs control |", "|---|---|---|---|---|"]
    ctrl5 = res.get("control", {}).get("fold")
    ctrlh = res.get("control", {}).get("holdout")
    for tag, r in sorted(res.items(), key=lambda kv: -(kv[1].get("fold")
                                                       or kv[1].get("holdout") or -1)):
        h = "-" if r.get("holdout") is None else f"{r['holdout']:.4f}"
        f = "-" if r.get("fold") is None else f"{r['fold']:.4f}"
        hr = "-" if r.get("hate_rate") is None else f"{r['hate_rate']:.2f}"
        if r.get("fold") is not None and ctrl5 is not None:
            d = f"{r['fold'] - ctrl5:+.4f} (5-fold)"
        elif r.get("holdout") is not None and ctrlh is not None:
            d = f"{r['holdout'] - ctrlh:+.4f} (holdout)"
        else:
            d = "-"
        L.append(f"| `{tag}` | {h} | {f} | {hr} | {d} |")
    L += ["", "Reference points on the same data: TF-IDF + LinearSVC floor 0.8073 five-fold,",
          "current best CodaBench submission 0.8187 (a blend, not comparable to a single",
          "component's OOF).", "",
          "A holdout gap under about 0.013 is inside that split's noise; a five-fold gap",
          "under about 0.006 is inside its own. Promote on rank, conclude on five folds."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=10.5)
    ap.add_argument("--reserve-min", type=float, default=20)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--stage", choices=["screen", "confirm", "both"], default="both")
    ap.add_argument("--promote", type=int, default=2,
                    help="how many leaders go to five folds")
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    py, t0 = sys.executable, time.time()
    left = lambda: a.budget_hours * 3600 - (time.time() - t0) - a.reserve_min * 60

    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from hastika.common.preprocessing import dedupe_index
    tr = pd.read_csv(ROOT / "data" / "raw" / "binary_train.csv")
    keep = dedupe_index(tr["Comment"].tolist(), tr["Label"].tolist(), "task A")
    y = (tr.iloc[keep]["Label"] == "Hate").astype(int).values

    arms = [x for x in ARMS if a.arms is None or x[0] in a.arms]
    res = {}

    # ---- the shared TAPT checkpoint, built only if a TAPT arm is selected ----
    if any(x[3] == "tapt" for x in arms) and not (ROOT / TAPT_OUT / "config.json").exists():
        if left() > 70 * 60:
            sh([py, "-u", "-m", "hastika.task_b.tapt",
                "--corpus", "data/raw/binary_train.csv", "data/external/offenseval_kn.csv",
                "--val-frac", "0", "--min-words", "1", "--no-dedupe", "--epochs", "8",
                "--allow-transductive", "--out", TAPT_OUT], log=LOGS / "a_tapt_build.log")
        else:
            print("skip TAPT build: out of budget", flush=True)

    # ---- stage 1: rank everything on the cheap split ------------------------
    if a.stage in ("screen", "both"):
        for tag, flags, est, _, needs in arms:
            if needs == "tapt" and not (ROOT / TAPT_OUT / "config.json").exists():
                print(f"skip {tag}: no TAPT checkpoint", flush=True); continue
            if holdout_score(tag) is None:
                if left() < est * 60:
                    print(f"skip {tag}: {left()/60:.0f} min left, needs ~{est}", flush=True)
                    continue
                sh([py, "-u", "-m", "hastika.models.muril", "--tag", f"a_{tag}",
                    "--folds", "0", *dedup_flags(COMMON, flags)], log=LOGS / f"a_{tag}.log")
            res[tag] = {"holdout": holdout_score(tag), "hate_rate": healthy(tag),
                        "fold": None, "flags": flags}
            write_results(res, out, time.time() - t0, "1 (screening)")

    # ---- stage 2: confirm the leaders on five folds -------------------------
    if a.stage in ("confirm", "both"):
        ranked = [t for t, r in sorted(res.items(), key=lambda kv: -(kv[1]["holdout"] or -1))
                  if r["holdout"] is not None
                  and (r["hate_rate"] is None or 0.10 < r["hate_rate"] < 0.90)]
        # the control is always confirmed: without it the promoted numbers have no reference
        promote = ranked[:a.promote]
        if "control" in res and "control" not in promote:
            promote.append("control")
        for tag in promote:
            est = dict((x[0], x[3]) for x in ARMS)[tag]
            if oof_score(tag, y) is not None:
                print(f"reusing existing five-fold {tag}"
                      + (" (from Run 11)" if tag == "control" else ""), flush=True)
            else:
                if left() < est * 60:
                    print(f"skip 5-fold {tag}: {left()/60:.0f} min left, needs ~{est}",
                          flush=True)
                    continue
                flags = dict((x[0], x[1]) for x in ARMS)[tag]
                sh([py, "-u", "-m", "hastika.models.muril", "--tag", f"f_{tag}",
                    "--folds", "5", *dedup_flags(COMMON, flags)],
                   log=LOGS / f"f_{tag}.log")
            res.setdefault(tag, {})["fold"] = oof_score(tag, y)
            write_results(res, out, time.time() - t0, "2 (confirming)")

    # ---- package every arm that produced predictions -----------------------
    # muril.py writes predictions.csv whatever --folds is: for 5 it is the average
    # over the fold models, for 0 it is the single 85% model. Both are valid
    # submissions, the holdout one just saw less data. Zipping them costs nothing
    # and means a screening session still hands you something uploadable.
    subs = out / "subs"; subs.mkdir(exist_ok=True)
    for d in sorted(RUNS.glob("[af]_*")):
        pred = d / "predictions.csv"
        if pred.exists():
            sh([py, "-m", "hastika.common.submission", "--task", "a",
                "--pred", str(pred), "--out", str(subs / f"{d.name}.zip")])
    print(f"\npackaged {len(list(subs.glob('*.zip')))} submission ZIPs in {subs}", flush=True)

    for f in LOGS.glob("*.log"):
        sh(["cp", str(f), str(out)])
    write_results(res, out, time.time() - t0, "done")
    print(f"\nDONE in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
