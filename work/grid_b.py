"""Full factorial over the data-processing levers, measured on a real holdout.

THE DESIGN
----------
Two data-processing factors, crossed, giving four encoders, plus stock MuRIL as
the control that uses neither:

    corpus   D0 = Kannada only (6,362 comments, the current default)
             D1 = + Tamil and Malayalam OffensEval-Dravidian (~58,000).
                  Romanized Dravidian in the same script convention, carrying no
                  HASTIKA labels, so it is legal external data for masked-LM use.
    vocab    V0 = MuRIL's tokenizer as shipped
             V1 = extended with this corpus's frequent word-forms

Crossed again with the training objective (none / auxiliary act head), giving
ten configurations. Each is a 15% holdout run on the fixed split, so unlike a
full fit **every cell here has a real score** and the question "which processing
helps" gets an actual answer.

Stage 2 re-runs the leaders with more seeds, because at 3,143 rows a single seed
resolves about one point and several cells will land inside that.

WHY THESE TWO FACTORS AND NOT OTHERS
------------------------------------
Measured on this corpus, all inside the one-point fold noise and therefore dead:
per-class decode weights (-0.009 nested), an OOV spelling resolver (+0.003),
explicit conjunction features (-0.003). Feature-level work does not move this
problem. Adaptation data does, at +2.9, and it works on the representation.
Both factors here are representation-level for that reason.
"""
import argparse
import itertools
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
MURIL = "google/muril-base-cased"
SMALL = ["data/multiclass_train.csv", "data/external/offenseval_kn.csv"]
BIG = SMALL + ["data/external/offenseval_ta.csv", "data/external/offenseval_ma.csv"]


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
        print(f"\n{'!'*70}\nSTEP FAILED (exit {p.returncode}) -- continuing\n{'!'*70}",
              flush=True)
    return p.returncode


def score(log):
    """The unbiased holdout number: --select best means the 'last' line is it."""
    p = pathlib.Path(log)
    if not p.exists():
        return None
    t = p.read_text()
    alt = re.search(r"holdout macro-F1 ([\d.]+) with the last checkpoint", t)
    if alt:
        return float(alt.group(1))
    m = re.search(r"holdout macro-F1 ([\d.]+)\s+acc", t)
    return float(m.group(1)) if m else None


def write_results(res, out, spent, stage):
    rows = sorted(res.items(), key=lambda kv: -(kv[1].get("f1") or -1))
    L = ["# Task B -- data-processing factorial", "",
         f"Elapsed {spent/3600:.2f} h. Stage: {stage}.", "",
         "Every row is a 15% holdout run on the fixed 472-row split, so these are real",
         "scores on rows no model trained on. Rank on `macro-F1`.", "",
         "`corpus` D0 = Kannada only, D1 = + Tamil and Malayalam (~58k comments).",
         "`vocab` V0 = MuRIL as shipped, V1 = extended with frequent word-forms.",
         "`aux` = auxiliary head on the act axis.", "",
         "| config | corpus | vocab | aux | seeds | macro-F1 | vs control |",
         "|---|---|---|---|---|---|---|"]
    base = res.get("stock_V0_noaux", {}).get("f1")
    for tag, r in rows:
        f1 = f"{r['f1']:.4f}" if r.get("f1") is not None else "-"
        d = (f"{r['f1']-base:+.4f}" if r.get("f1") is not None and base else "-")
        L.append(f"| `{tag}` | {r.get('corpus','-')} | {r.get('vocab','-')} | "
                 f"{r.get('aux','-')} | {r.get('seeds',1)} | {f1} | {d} |")
    L += ["", "Reference points measured earlier on this same split: stock MuRIL 0.5948,",
          "MLM-adapted MuRIL 0.5972, the TF-IDF floor 0.5948 (5-fold OOF).", ""]
    pathlib.Path(out, "RESULTS.md").write_text("\n".join(L) + "\n")
    pathlib.Path(out, "results.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=10.5)
    ap.add_argument("--reserve-min", type=float, default=15)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--confirm-top", type=int, default=2,
                    help="leaders re-run with --confirm-seeds")
    ap.add_argument("--confirm-seeds", default="43 44")
    ap.add_argument("--final-full-fit", action="store_true",
                    help="after ranking, 5-seed full fit of the winner for submission")
    ap.add_argument("--tapt-epochs-small", type=int, default=8)
    ap.add_argument("--tapt-epochs-big", type=int, default=1,
                    help="the big corpus is ~9x the small one, so ONE epoch over it is "
                         "about the same number of optimizer steps as eight over the "
                         "small one. That makes D0 vs D1 an equal-compute comparison: "
                         "the only thing that differs is how varied the text is.")
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subs = out / "subs"; subs.mkdir(exist_ok=True)
    logs = ROOT / "work"
    t0 = time.time()
    left = lambda: a.budget_hours*3600 - (time.time()-t0) - a.reserve_min*60
    res, times = {}, []
    py = sys.executable

    # ---- stage 0: build the four encoders -----------------------------------
    encoders = {}
    for d, v in itertools.product(["D0", "D1"], ["V0", "V1"]):
        corpus = SMALL if d == "D0" else BIG
        ep = a.tapt_epochs_small if d == "D0" else a.tapt_epochs_big
        base = MURIL
        if v == "V1":
            ex = f"work/runs/vocab-{d}"
            if not (ROOT / ex).is_dir():
                if left() < 900:
                    print(f"skip vocab-{d}: out of budget", flush=True); continue
                sh([py, "-u", "work/extend_vocab.py", "--out", ex,
                    "--corpus", *corpus], log=logs/f"vocab_{d}.log")
            if not (ROOT / ex).is_dir():
                continue
            base = ex
        tag = f"tapt-{d}{v}"
        path = f"work/runs/{tag}"
        if not (ROOT / path).is_dir():
            est = 1800 if d == "D0" else 2400
            if left() < est:
                print(f"skip {tag}: {left()/60:.0f} min left, needs ~{est/60:.0f}",
                      flush=True); continue
            sh([py, "-u", "work/tapt.py", "--model", base, "--out", path,
                "--corpus", *corpus, "--epochs", str(ep)], log=logs/f"{tag}.log")
        if (ROOT / path).is_dir():
            encoders[f"{d}{v}"] = path

    # the control uses neither lever
    configs = [("stock_V0_noaux", MURIL, "-", "V0", "no")]
    for key, path in encoders.items():
        d, v = key[:2], key[2:]
        configs.append((f"{d}_{v}_noaux", path, d, v, "no"))
    for key, path in encoders.items():
        d, v = key[:2], key[2:]
        configs.append((f"{d}_{v}_aux", path, d, v, "yes"))
    configs.append(("stock_V0_aux", MURIL, "-", "V0", "yes"))

    # ---- stage 1: one seed per cell -----------------------------------------
    for tag, model, d, v, aux in configs:
        est = np.median(times) if times else 1250
        if left() < est:
            print(f"skip {tag}: out of budget", flush=True)
            res[tag] = {"corpus": d, "vocab": v, "aux": aux, "f1": None}
            write_results(res, out, time.time()-t0, "1 (screening)")
            continue
        t = time.time()
        log = logs / f"g_{tag}.log"
        cmd = [py, "-u", "work/muril_b.py", "--tag", f"g_{tag}", "--folds", "0",
               "--seeds", "42", "--epochs", str(a.epochs), "--model", model]
        if aux == "yes":
            cmd += ["--aux-weight", str(a.aux_weight)]
        if sh(cmd, log=log) == 0:
            times.append(time.time()-t)
        res[tag] = {"corpus": d, "vocab": v, "aux": aux, "seeds": 1,
                    "f1": score(log), "model": model}
        write_results(res, out, time.time()-t0, "1 (screening)")

    # ---- stage 2: more seeds for the leaders --------------------------------
    ranked = [t for t, r in sorted(res.items(), key=lambda kv: -(kv[1].get("f1") or -1))
              if r.get("f1") is not None][:a.confirm_top]
    for tag in ranked:
        est = (np.median(times) if times else 1250) * len(a.confirm_seeds.split())
        if left() < est:
            print(f"skip confirm {tag}: out of budget", flush=True); continue
        r = res[tag]
        log = logs / f"c_{tag}.log"
        cmd = [py, "-u", "work/muril_b.py", "--tag", f"c_{tag}", "--folds", "0",
               "--seeds", "42", *a.confirm_seeds.split(), "--epochs", str(a.epochs),
               "--model", r["model"]]
        if r["aux"] == "yes":
            cmd += ["--aux-weight", str(a.aux_weight)]
        if sh(cmd, log=log) == 0:
            f1 = score(log)
            res[tag + "_multiseed"] = dict(r, seeds=1+len(a.confirm_seeds.split()), f1=f1)
        write_results(res, out, time.time()-t0, "2 (confirming leaders)")

    # ---- stage 3: the winner, full fit, for submission -----------------------
    if a.final_full_fit and ranked:
        win = res[ranked[0]]
        est = (np.median(times) if times else 1250) * 5 * 1.2
        if left() >= est:
            cmd = [py, "-u", "work/muril_b.py", "--tag", "g_winner_full", "--folds", "1",
                   "--seeds", "42", "43", "44", "45", "46", "--epochs", str(a.epochs),
                   "--model", win["model"]]
            if win["aux"] == "yes":
                cmd += ["--aux-weight", str(a.aux_weight)]
            sh(cmd, log=logs/"g_winner_full.log")
            pred = RUNS / "g_winner_full" / "predictions.csv"
            if pred.exists():
                sh([py, "work/make_submission.py", "--task", "b", "--pred", str(pred),
                    "--out", str(subs / "winner_full.zip")])
        else:
            print(f"skip final full fit: {left()/60:.0f} min left", flush=True)

    # package every holdout run too, for inspection
    for tag in list(res):
        p = RUNS / f"g_{tag}" / "predictions.csv"
        if p.exists():
            sh([py, "work/make_submission.py", "--task", "b", "--pred", str(p),
                "--out", str(subs / f"{tag}.zip")])
    for f in logs.glob("*.log"):
        sh(["cp", str(f), str(out)])
    write_results(res, out, time.time()-t0, "done")
    print(f"\nDONE in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
