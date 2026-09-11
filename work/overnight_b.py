"""Overnight Task B sweep: rank ideas on a holdout, then 5-fold the survivors.

RULES THIS OBEYS
----------------
1. Every stage that produces predictions gets packaged into a CodaBench zip the
   moment it finishes, so a session killed at hour nine still leaves everything
   it completed. Nothing is left to a final cell that may never run.
2. A wall-clock budget. No new GPU work starts once the remaining time cannot
   fit the next stage, so the run stops on its own rather than being cut off
   mid-fold by Kaggle's 12-hour limit.
3. RESULTS.md is rewritten after every single arm, so the table is complete
   whenever you read it.

THE IDEAS, AND WHY EACH ONE IS HERE
-----------------------------------
The measured failure mode is that Task B is target identification wearing a
slur-detection costume: Violence loses 103 of its 221 rows to Gender because
gendered slurs get used as generic insults and the label follows the target.
Macro-F1 then charges 1/6 of the score for that one class. Tuning learning
rates does not touch any of that, so none of these arms is a learning rate.

  b_tapt      MuRIL never saw this register -- 2.28 wordpieces per whitespace
              word. A masked-LM pass over in-domain text is the standard fix.
  b_xlmr      The corpus is 0% Kannada script. MuRIL spends a 197k vocab on
              Indic scripts that never appear, so its headline advantage is
              dead weight here and a Latin-heavy encoder may simply fit better.
  b_mdeberta  Same bet as b_xlmr with the strongest multilingual base encoder.
  b_hing      Trained on romanized Hindi-English social media, which is the
              closest public register to romanized Kannada-English.
  b_focal     Down-weights rows the model already gets right, which is the
              Gender mass, and spends capacity on the tail macro-F1 pays for.
  b_rdrop     Two dropout passes with a KL tie between them. At 3,143 rows the
              binding constraint is variance, not capacity.
  b_large     MuRIL-large, if the budget survives that far.

Blending across tokenizer families is the real prize: ensemble.py weight-searches
on the OOF and reports a nested estimate, so a blend that only looks good
in-sample is caught before it is submitted.
"""
import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "runs"
MURIL = "google/muril-base-cased"

# tag, extra flags, why. Order is priority: the budget cuts from the bottom.
STAGE1 = [
    ("b_tapt",     ["--model", "work/runs/tapt-muril"],        "MLM-adapted MuRIL"),
    ("b_xlmr",     ["--model", "xlm-roberta-base"],            "Latin-heavy encoder"),
    ("b_mdeberta", ["--model", "microsoft/mdeberta-v3-base"],  "strongest multilingual base"),
    ("b_hing",     ["--model", "l3cube-pune/hing-roberta"],    "romanized Indic social media"),
    ("b_focal",    ["--model", MURIL, "--loss", "focal"],      "spend capacity on the tail"),
    ("b_rdrop",    ["--model", MURIL, "--rdrop", "0.5"],       "variance, not capacity"),
    # --no-fgm because FGM clones the embedding table every step and large's is
    # 197285 x 1024; --lr below base's 3e-5 because large diverges at it.
    ("b_large",    ["--model", "google/muril-large-cased", "--bs", "8",
                    "--grad-accum", "2", "--no-fgm", "--lr", "1.5e-5"],
                                                               "more capacity, 3x the time"),
]

# Measured on this corpus: b_base took 1055s for a holdout on a T4.
HOLDOUT_S = 1150.0
FULL_MULT = 4.7          # 5 folds at 80% vs one holdout at 85%


class Budget:
    def __init__(self, hours, reserve_min):
        self.t0 = time.time()
        self.total = hours * 3600
        self.reserve = reserve_min * 60
        self.gpu = 0.0

    @property
    def left(self):
        return self.total - (time.time() - self.t0)

    def fits(self, cost):
        return self.left - cost > self.reserve

    def spent(self):
        return time.time() - self.t0


def sh(cmd, log=None, required=False):
    """Run, stream, tee, and return the exit code. Never silently succeeds."""
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    fh = open(log, "w") if log else None
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         shell=isinstance(cmd, str))
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
        print(f"\n{'!' * 70}\nARM FAILED (exit {p.returncode}) -- continuing\n{'!' * 70}",
              flush=True)
    return p.returncode


def parse_score(log):
    """Return (selected, unbiased). --select best means the 'last' line is unbiased."""
    p = pathlib.Path(log)
    if not p.exists():
        return None, None
    t = p.read_text()
    sel = re.search(r"(?:OOF|holdout) macro-F1 ([\d.]+)\s+acc", t)
    alt = re.search(r"(?:OOF|holdout) macro-F1 ([\d.]+) with the last checkpoint", t)
    return (float(sel.group(1)) if sel else None,
            float(alt.group(1)) if alt else (float(sel.group(1)) if sel else None))


def package(tag, subs, results, kind):
    """Zip a run's predictions if it has any. Records the zip path in results."""
    pred = RUNS / tag / "predictions.csv"
    if not pred.exists():
        return None
    z = subs / f"{tag}.zip"
    rc = sh([sys.executable, "work/make_submission.py", "--task", "b",
             "--pred", str(pred), "--out", str(z)])
    if rc == 0:
        results[tag]["zip"] = str(z)
        results[tag]["kind"] = kind
    return z if rc == 0 else None


def write_results(results, out, budget):
    rows = sorted(results.items(),
                  key=lambda kv: (kv[1].get("unbiased") is None,
                                  -(kv[1].get("unbiased") or 0)))
    lines = ["# Task B overnight sweep", "",
             f"Elapsed {budget.spent()/3600:.2f} h. "
             f"Ranked by the unbiased macro-F1 (the `last` checkpoint).", "",
             "Submit the top row that has a zip. Five-fold rows are worth more than",
             "holdout rows at equal score, because a holdout trained on 15% less data.",
             "",
             "| rank | arm | kind | unbiased macro-F1 | selected | zip | note |",
             "|---|---|---|---|---|---|---|"]
    for i, (tag, r) in enumerate(rows, 1):
        u = f"{r['unbiased']:.4f}" if r.get("unbiased") is not None else "-"
        s = f"{r['selected']:.4f}" if r.get("selected") is not None else "-"
        z = pathlib.Path(r["zip"]).name if r.get("zip") else "-"
        lines.append(f"| {i} | `{tag}` | {r.get('kind','-')} | {u} | {s} | "
                     f"`{z}` | {r.get('note','')} |")
    lines += ["", "## Baselines already known", "",
              "| arm | macro-F1 | note |", "|---|---|---|",
              "| `svm_b` | 0.5948 | TF-IDF + LinearSVC, 5-fold OOF |",
              "| `b_base` | 0.5948 | stock MuRIL, 15% holdout |",
              "| `b_abusive` | 0.5718 | abusive-tuned MuRIL, 15% holdout |", ""]
    pathlib.Path(out, "RESULTS.md").write_text("\n".join(lines))
    pathlib.Path(out, "results.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines[6:]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=10.5)
    ap.add_argument("--reserve-min", type=float, default=20,
                    help="held back for blending and packaging, which are CPU-only")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--stage2-max", type=int, default=4,
                    help="how many stage-1 winners to promote to a 5-fold run")
    ap.add_argument("--skip-tapt", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="run only these stage-1 tags")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subs = out / "subs"
    subs.mkdir(exist_ok=True)
    logs = ROOT / "work"
    budget = Budget(a.budget_hours, a.reserve_min)
    results = {}
    py = sys.executable

    def record(tag, log, note, kind):
        sel, unb = parse_score(log)
        results.setdefault(tag, {})
        results[tag].update({"selected": sel, "unbiased": unb, "note": note, "log": str(log)})
        package(tag, subs, results, kind)
        write_results(results, out, budget)

    print(f"budget {a.budget_hours} h, reserve {a.reserve_min} min, out {out}", flush=True)

    # ---- stage 0: domain adaptation, which b_tapt depends on -------------------
    if not a.skip_tapt and not (RUNS / "tapt-muril").exists():
        if budget.fits(1200):
            sh([py, "-u", "work/tapt.py", "--out", "work/runs/tapt-muril"],
               log=logs / "tapt.log")
        else:
            print("skipping tapt: out of budget", flush=True)

    # ---- stage 1: rank ideas on a 15% holdout ---------------------------------
    times = []
    arms = [x for x in STAGE1 if a.only is None or x[0] in a.only]
    for tag, flags, note in arms:
        model = flags[flags.index("--model") + 1]
        if model.startswith("work/") and not (ROOT / model).is_dir():
            print(f"skipping {tag}: {model} was never written", flush=True)
            results[tag] = {"note": note + " (skipped: no checkpoint)", "kind": "skipped"}
            write_results(results, out, budget)
            continue
        est = statistics.median(times) if times else HOLDOUT_S
        if tag == "b_large":
            est *= 3
        if not budget.fits(est):
            print(f"skipping {tag}: {budget.left/60:.0f} min left, needs ~{est/60:.0f}",
                  flush=True)
            results[tag] = {"note": note + " (skipped: out of budget)", "kind": "skipped"}
            write_results(results, out, budget)
            continue
        t = time.time()
        log = logs / f"{tag}.log"
        rc = sh([py, "-u", "work/muril_b.py", "--tag", tag, "--folds", "0",
                 "--seeds", *a.seeds.split(), "--epochs", str(a.epochs), *flags], log=log)
        took = time.time() - t
        if rc == 0 and tag != "b_large":
            times.append(took)
        print(f"[{tag}] {took/60:.1f} min, {budget.left/60:.0f} min left", flush=True)
        record(tag, log, note, "holdout")

    # ---- stage 2: 5-fold the winners ------------------------------------------
    ranked = [t for t, r in sorted(results.items(),
                                   key=lambda kv: -(kv[1].get("unbiased") or 0))
              if r.get("unbiased") is not None]
    # stock MuRIL has no 5-fold run yet and is the reference every blend needs
    promote = ["b_base"] + [t for t in ranked if t != "b_base"]
    promote = promote[:a.stage2_max]
    full_est = (statistics.median(times) if times else HOLDOUT_S) * FULL_MULT
    print(f"\nstage 2 candidates: {promote} (~{full_est/3600:.1f} h each)", flush=True)

    flags_of = {t: f for t, f, _ in STAGE1}
    flags_of["b_base"] = ["--model", MURIL]
    notes_of = {t: n for t, _, n in STAGE1}
    notes_of["b_base"] = "stock MuRIL, the reference"

    for tag in promote:
        if not budget.fits(full_est):
            print(f"stopping stage 2: {budget.left/60:.0f} min left, "
                  f"a 5-fold run needs ~{full_est/60:.0f}", flush=True)
            break
        full = f"{tag}_5f"
        t = time.time()
        log = logs / f"{full}.log"
        rc = sh([py, "-u", "work/muril_b.py", "--tag", full, "--folds", "5",
                 "--seeds", *a.seeds.split(), "--epochs", str(a.epochs),
                 *flags_of[tag]], log=log)
        print(f"[{full}] {(time.time()-t)/60:.1f} min, {budget.left/60:.0f} min left",
              flush=True)
        record(full, log, notes_of.get(tag, "") + " (5-fold)", "5-fold")

    # ---- stage 3: blend, decode, package (CPU only, inside the reserve) --------
    print("\n=== blending every 5-fold run ===", flush=True)
    sh([py, "work/ensemble.py", "--task", "b"], log=logs / "ensemble_b.log")
    if (RUNS / "ensemble_b" / "predictions.csv").exists():
        sel, unb = parse_score(logs / "ensemble_b.log")
        blend = re.search(r"nested estimate\s+([\d.]+)",
                          (logs / "ensemble_b.log").read_text())
        results["ensemble_b"] = {
            "selected": None,
            "unbiased": float(blend.group(1)) if blend else None,
            "note": "weight-searched blend, nested estimate", "kind": "blend"}
        package("ensemble_b", subs, results, "blend")
        write_results(results, out, budget)

    print("\n=== macro-F1-optimal decode on every 5-fold run ===", flush=True)
    # every 5-fold run on disk, including svm_b and runs from earlier sessions
    for d in sorted(RUNS.iterdir()):
        tag = d.name
        if tag.endswith("_decoded") or not (d / "oof_probs.npy").exists():
            continue
        dec = f"{tag}_decoded"
        rc = sh([py, "work/decode_b.py", "--run", tag, "--out", dec,
                 "--zip", str(subs / f"{dec}.zip")], log=logs / f"{dec}.log")
        if rc == 0 and (subs / f"{dec}.zip").exists():
            m = re.search(r"tuned nested\s+([\d.]+)", (logs / f"{dec}.log").read_text())
            results[dec] = {"selected": None,
                            "unbiased": float(m.group(1)) if m else None,
                            "note": f"macro-F1 decode over {tag}", "kind": "decode",
                            "zip": str(subs / f"{dec}.zip")}
            write_results(results, out, budget)

    for f in logs.glob("*.log"):
        sh(["cp", str(f), str(out)])
    write_results(results, out, budget)
    print(f"\nDONE in {budget.spent()/3600:.2f} h. Zips in {subs}", flush=True)
    for z in sorted(subs.glob("*.zip")):
        print(f"  {z}")


if __name__ == "__main__":
    main()
