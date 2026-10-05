"""Generates the self-contained Gemma-4-12B notebooks.

    python experiments/build_gemma_notebooks.py

Every notebook runs standalone on Kaggle (GPU T4 x2, Internet on, no token). When the
output of an earlier run is attached it is found by searching every input, following
symlinks, and reused. Otherwise the notebook trains what it needs instead of failing.

Writes:
    notebooks/task_b/22_package_run21.ipynb          Run 21's 2-seed LLM submission
    notebooks/task_b/23_gemma_epochs_seeds_final.ipynb 4 vs 3 epochs, 4-model submission
    notebooks/task_b/24_gemma_muril_lessons.ipynb    TAPT and R-Drop on the holdout
    notebooks/task_b/25_gemma_final_recipe.ipynb     configurable final full-data fit
    notebooks/task_a/28_gemma_holdout_final.ipynb    Gemma on Task A, holdout + submission
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def notebook(cells, rel):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {},
                "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    nb = {"cells": out, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / rel, "w"), indent=1, ensure_ascii=False)
    print("wrote", rel, len(out), "cells")


TASK_CFG = {
    "b": dict(split="task_b_combined_holdout", fp="f85f4f049b", n_all=3532, n_ho=530,
              classes='["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]',
              col="Hate Category", test="hastika_multiclass_test.csv"),
    "a": dict(split="task_a_combined_holdout", fp="815110ff24", n_all=7193, n_ho=1079,
              classes='["Non-Hate", "Hate"]', col="Label", test="hastika_binary_test.csv"),
}


def setup(task, out_name):
    c = TASK_CFG[task]
    return r'''
import json, os, pathlib, subprocess, sys, time
from fnmatch import fnmatch
T_START = time.time()
HARD_STOP = T_START + 11 * 3600   # Kaggle kills at 12 h and then saves nothing

REPO = "https://github.com/robinpnalex/Hastika-ICON2026.git"
BRANCH = "task-b"
WORK = "/tmp/hastika"             # outside /kaggle/working: the output holds results only


def git(*args):
    return subprocess.run(["git", "-C", WORK, *args], check=True,
                          capture_output=True, text=True).stdout.strip()


try:
    remote = subprocess.run(["git", "ls-remote", "--exit-code", REPO, f"refs/heads/{BRANCH}"],
                            check=True, capture_output=True, text=True,
                            timeout=60).stdout.split()[0]
except Exception as e:
    raise SystemExit("Cannot reach GitHub. Turn on Settings -> Internet, then restart.") from e
if os.path.isdir(WORK + "/.git"):
    git("fetch", "-q", "--depth", "1", "origin", BRANCH)
    git("reset", "-q", "--hard", "FETCH_HEAD")
else:
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, "--depth", "1", REPO, WORK], check=True)
head = git("rev-parse", "HEAD")
assert head == remote, f"clone is at {head[:8]} but {BRANCH} is at {remote[:8]}"
os.chdir(WORK)
os.environ["PYTHONPATH"] = os.path.join(WORK, "src")
sys.path.insert(0, os.path.join(WORK, "src"))
print("repo at", git("log", "-1", "--oneline"))
subprocess.run('pip install -q emoji ftfy sentencepiece protobuf "transformers>=4.45,<6"',
               shell=True, check=True)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass                              # Gemma-4 is not gated; no token is needed

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

assert torch.cuda.is_available(), "select GPU T4 x2"
N_GPU = torch.cuda.device_count()
print("gpus:", [torch.cuda.get_device_name(i) for i in range(N_GPU)])
OUT = pathlib.Path("/kaggle/working/''' + out_name + r'''")
RUNS, LOGS, SUB = OUT / "runs", OUT / "logs", OUT / "submissions"
for p in (RUNS, LOGS, SUB):
    p.mkdir(parents=True, exist_ok=True)
HF_CACHE = "/tmp/hf"
os.environ["HF_HUB_CACHE"] = HF_CACHE

from hastika.common.gpu_queue import run_queue
from hastika.task_b.combined_holdout import fingerprint, paired_bootstrap

TASK = "''' + task + r'''"
CLASSES = ''' + c["classes"] + r'''
LABEL_COL = "''' + c["col"] + r'''"
SPLIT = pathlib.Path(WORK) / "data/derived/''' + c["split"] + r'''"
df = pd.read_csv(SPLIT / "all.csv")
holdout = pd.read_csv(SPLIT / "holdout.csv")
assert (len(df), len(holdout)) == (''' + str(c["n_all"]) + ", " + str(c["n_ho"]) + r'''), (len(df), len(holdout))
HOLDOUT_FP = fingerprint(holdout["id"])
assert HOLDOUT_FP == "''' + c["fp"] + r'''", HOLDOUT_FP
y_ho = holdout[LABEL_COL].map(CLASSES.index).to_numpy()
TEST_CSV = pathlib.Path("/tmp/test_inputs.csv")
pd.read_csv(pathlib.Path(WORK) / "data/raw/''' + c["test"] + r'''")[["id", "Comment"]].to_csv(TEST_CSV, index=False)
test = pd.read_csv(TEST_CSV)
EXTERNAL = pathlib.Path(WORK) / "data/external/offenseval_kn.csv"
print(f"task {TASK.upper()}: train {len(df) - len(holdout)}, holdout {len(holdout)} ({HOLDOUT_FP}), "
      f"all {len(df)}, test {len(test)}")


def report(y, probs, name):
    pred = probs.argmax(1)
    per = f1_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    macro = float(f1_score(y, pred, average="macro"))
    print(f"{name:34s} macro-F1 {macro:.4f}  acc {accuracy_score(y, pred):.4f}  | "
          + "  ".join(f"{c[:5]} {f:.3f}" for c, f in zip(CLASSES, per)))
    return macro


def boot_line(label, a, b):
    r = paired_bootstrap(y_ho, a, b)
    print(f"{label:28s} {r['diff']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
          f"P(better) {r['p_better']:.2f}")
    return r


# Earlier runs' outputs are reused when attached (Add Input -> Notebook Output), found by
# walking every input with symlinks followed. Nothing below *requires* them.
def find_prior(pattern):
    hits = []
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        for f in files:
            p = os.path.join(root, f)
            if fnmatch(p, pattern):
                hits.append(pathlib.Path(p))
    return sorted(hits)


def show_inputs():
    if not os.path.isdir("/kaggle/input"):
        print("no /kaggle/input")
        return
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        depth = root[len("/kaggle/input"):].count(os.sep)
        if depth <= 4:
            print("  " * depth + (os.path.basename(root) or "/kaggle/input") + f"/  ({len(files)} files)")


print("attached inputs:")
show_inputs()
'''


LLM_ENV = r'''
# LLM stack in its own environment (transformers >= 5.12 for Gemma-4); Kaggle's torch and
# CUDA are reused via --system-site-packages. The repo's MuRIL/SVM code stays on the
# system environment.
LLM_PY = "/tmp/llmenv/bin/python"
if not os.path.exists(LLM_PY):
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip",
                    "/tmp/llmenv"], check=True)
subprocess.run(f'{LLM_PY} -m pip install -q "transformers>=5.12,<6" "peft>=0.17" '
               f'"bitsandbytes>=0.46" "accelerate>=1.0" emoji ftfy sentencepiece protobuf',
               shell=True, check=True)
subprocess.run([LLM_PY, "-c", "import transformers, peft, bitsandbytes; print('LLM env: "
                "transformers', transformers.__version__, '| peft', peft.__version__)"], check=True)

MODEL = "google/gemma-4-12B"
S = "gemma-4-12b"


def download(model=MODEL):
    from huggingface_hub import snapshot_download
    snapshot_download(model, cache_dir=HF_CACHE, token=os.environ.get("HF_TOKEN") or None,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                      "tokenizer*", "*.jinja"])


def llm_job(name, train, predict, seed, evalf="", extra=""):
    ev = f"--eval {evalf}" if evalf else ""
    return {"name": name, "log": str(LOGS / f"{name}.log"),
            "cmd": (f"{LLM_PY} -u -m hastika.task_b.llm_classifier --task {TASK} --model {MODEL} "
                    f"--train {train} {ev} --predict {' '.join(map(str, predict))} "
                    f"--out {RUNS / name} --seed {seed} --cache {HF_CACHE} {extra}")}


def tapt_job(name, texts):
    """Gemma TAPT: LoRA next-token training on unlabelled comments (the MuRIL +2.9 lesson)."""
    return {"name": name, "log": str(LOGS / f"{name}.log"),
            "cmd": (f"{LLM_PY} -u -m hastika.task_b.llm_tapt --model {MODEL} "
                    f"--texts {' '.join(map(str, texts))} --out {RUNS / name} --cache {HF_CACHE}")}


def load_probs(run_dir, csv):
    """<input stem>_probs.npy, with the saved ids checked against the input file."""
    run_dir, csv = pathlib.Path(run_dir), pathlib.Path(csv)
    p = run_dir / f"{csv.stem}_probs.npy"
    if not p.exists():
        return None
    ids = pd.read_csv(run_dir / f"{csv.stem}_ids.csv")["id"].tolist()
    assert ids == pd.read_csv(csv)["id"].tolist(), f"{run_dir}: ids differ from {csv.name}"
    return np.load(p)


def write_zip(name, probs, recommended=False):
    import zipfile
    assert probs.shape == (len(test), len(CLASSES)), probs.shape
    pred = pd.DataFrame({"id": test["id"], "label": [CLASSES[i] for i in probs.argmax(1)]})
    assert pred["id"].is_unique and set(pred["label"]) <= set(CLASSES)
    np.save(SUB / f"{name}_test_probs.npy", probs)
    csv = SUB / f"predictions_{name}.csv"
    pred.to_csv(csv, index=False)
    path = SUB / f"{'RECOMMENDED_' if recommended else ''}{name}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "predictions.csv")
    prior = df[LABEL_COL].value_counts(normalize=True)
    counts = pred["label"].value_counts(normalize=True)
    print(f"{path.name}  predicted/prior: "
          + "  ".join(f"{c[:5]} {counts.get(c, 0):.1%}/{prior[c]:.1%}" for c in CLASSES))
    return path


download()
'''

SETTINGS = """**Settings:** GPU T4 x2, Internet on. No token and no attached input are needed.
Save & Run All. A hard stop at 11 h means the notebook always finishes and packages
what it has."""

# ---------------------------------------------------------------------------- B22
b22 = [("markdown", r'''
# Task B Run 22 -- Run 21's Gemma-4-12B submission, standalone

Run 21 put Gemma-4-12B (QLoRA, 3 epochs, 2 seeds) at **0.6792** macro-F1 on the 530-row
holdout, against MuRIL's 0.6151 (+0.064, CI [+0.024, +0.105]). Its packaging cell then
missed the full fits' files. This notebook produces that submission:

- **If Run 21's output is attached,** its two full-fit probability files are found and
  packaged in under a minute.
- **If not,** the two full fits (3 epochs, seeds 42 and 43, all 3,532 rows) are trained, one
  per T4, in about 1.5 h, then packaged.

The output is `RECOMMENDED_b22_gemma_3ep_2seeds.zip`.

''' + SETTINGS), ("code", setup("b", "b22_outputs")), ("code", LLM_ENV), ("code", r'''
members, jobs = {}, []
for s in (42, 43):
    hits = find_prior(f"*/runs/{S}_full_s{s}/test_inputs_probs.npy")
    p = load_probs(hits[0].parent, TEST_CSV) if hits else None
    if p is not None:
        print(f"seed {s}: reusing {hits[0]}")
        members[f"run21_s{s}"] = p
    else:
        print(f"seed {s}: not attached, training it")
        jobs.append(llm_job(f"{S}_full_s{s}", SPLIT / "all.csv", [TEST_CSV], s))
if jobs:
    print(run_queue(jobs, N_GPU, raise_on_fail=False, stop_at=HARD_STOP))
    for j in jobs:
        p = load_probs(RUNS / j["name"], TEST_CSV)
        if p is not None:
            members[j["name"]] = p
assert members, "no full fit available -- see the logs"
print("averaging:", list(members))
write_zip("b22_gemma_3ep_2seeds", np.mean(list(members.values()), 0), recommended=True)
''')]

# ---------------------------------------------------------------------------- B23
b23 = [("markdown", r'''
# Task B Run 23 -- Gemma-4-12B: four epochs vs three, then a four-model submission

Both of Run 21's seeds were still rising at the last epoch (0.597, 0.629, 0.657 and 0.561,
0.630, 0.661), and averaging the two seeds added about 2 points.

**Stage 1: four epochs on the holdout.** Seeds 42 and 43 run in parallel, about 1.6 h.
- **Without an attached Run 21 output,** the comparison is with Run 21's recorded 3-epoch
  two-seed score, **0.6792**, on the same rows and seeds.
- **With it attached,** Run 21's 3-epoch probabilities are also loaded. That allows a
  paired bootstrap and a third candidate, `mix`, which averages all four models.

**Stage 2: four full-data models of the winner, averaged.** Two GPUs run two at a time, so
four 4-epoch seeds take about 4 h. Run 21's 3-epoch full fits are reused when attached and
relevant. The output is `RECOMMENDED_b23_gemma.zip`.

''' + SETTINGS), ("code", setup("b", "b23_outputs")), ("code", LLM_ENV), ("code", r'''
RUN21_3EP = 0.6792                 # Run 21, 3 epochs, seeds 42+43, this holdout
ho3, full3 = {}, {}
for s in (42, 43):
    h = find_prior(f"*/runs/{S}_ho_s{s}/holdout_probs.npy")
    if h:
        ho3[s] = load_probs(h[0].parent, SPLIT / "holdout.csv")
    f = find_prior(f"*/runs/{S}_full_s{s}/test_inputs_probs.npy")
    if f:
        full3[s] = load_probs(f[0].parent, TEST_CSV)
ho3 = {k: v for k, v in ho3.items() if v is not None}
full3 = {k: v for k, v in full3.items() if v is not None}
print("reused from Run 21 -- holdout seeds:", list(ho3), "| full-fit seeds:", list(full3))

jobs = [llm_job(f"{S}_ep4_ho_s{s}", SPLIT / "train.csv", [SPLIT / "holdout.csv"], s,
                evalf=SPLIT / "holdout.csv", extra="--epochs 4") for s in (42, 43)]
codes1 = run_queue(jobs, N_GPU, raise_on_fail=False, stop_at=HARD_STOP)
ho4 = {s: load_probs(RUNS / f"{S}_ep4_ho_s{s}", SPLIT / "holdout.csv") for s in (42, 43)}
ho4 = {k: v for k, v in ho4.items() if v is not None}
for s in ho4:
    hist = json.load(open(RUNS / f"{S}_ep4_ho_s{s}" / "metrics.json"))["history"]
    print(f"4 epochs seed {s} per epoch (diagnostic):", [round(h["eval_macro_f1"], 4) for h in hist])
'''), ("code", r'''
scores = {"3ep": report(y_ho, np.mean(list(ho3.values()), 0), "3 epochs (Run 21)")
          if len(ho3) == 2 else RUN21_3EP}
if len(ho3) != 2:
    print(f"3 epochs (Run 21, recorded)          macro-F1 {RUN21_3EP:.4f}")
boots = {}
if len(ho4) == 2:
    p4 = np.mean(list(ho4.values()), 0)
    scores["4ep"] = report(y_ho, p4, "4 epochs, 2 seeds")
    if len(ho3) == 2:
        p3 = np.mean(list(ho3.values()), 0)
        pm = np.mean(list(ho3.values()) + list(ho4.values()), 0)
        scores["mix"] = report(y_ho, pm, "3 + 4 epochs, 4 models")
        boots["4ep - 3ep"] = boot_line("4ep - 3ep", p4, p3)
        boots["mix - 3ep"] = boot_line("mix - 3ep", pm, p3)
CHOICE = max(scores, key=scores.get)
print("\nchosen:", CHOICE, scores)
'''), ("code", r'''
def full(seed, ep4):
    return llm_job(f"{S}_{'ep4_' if ep4 else ''}full_s{seed}", SPLIT / "all.csv", [TEST_CSV],
                   seed, extra="--epochs 4" if ep4 else "")


if CHOICE == "3ep":
    plan = [full(s, False) for s in (42, 43, 44, 45) if s not in full3]
elif CHOICE == "4ep":
    plan = [full(s, True) for s in (42, 43, 44, 45)]
else:                               # mix: Run 21's 3-epoch fits plus two 4-epoch fits
    plan = [full(s, True) for s in (42, 43)]
print("queueing:", [j["name"] for j in plan])
codes2 = run_queue(plan, N_GPU, raise_on_fail=False, stop_at=HARD_STOP)

members = {}
if CHOICE in ("3ep", "mix"):
    members.update({f"run21_3ep_s{s}": p for s, p in full3.items()})
for j in plan:
    p = load_probs(RUNS / j["name"], TEST_CSV)
    if p is not None:
        members[j["name"]] = p
assert members, "no full fit finished -- see the logs"
print("averaging", len(members), "models:", list(members))
write_zip("b23_gemma", np.mean(list(members.values()), 0), recommended=True)
json.dump({"commit": head, "choice": CHOICE, "holdout_scores": scores, "bootstrap": boots,
           "members": list(members), "exit_codes": {**codes1, **codes2}},
          open(OUT / "result.json", "w"), indent=2)
print(f"{(time.time() - T_START) / 3600:.1f} h")
''')]

# ---------------------------------------------------------------------------- B24
b24 = [("markdown", r'''
# Task B Run 24 -- do MuRIL's lessons transfer to Gemma-4-12B?

Every arm is Gemma-4-12B QLoRA, 3 epochs, seeds 42 and 43, scored on the same 530-row
holdout. Each tests one thing that helped MuRIL:

| arm | change | what it did for MuRIL |
|---|---|---|
| `base` | Run 21's recipe, the reference | -- |
| `tapt` | LoRA next-token pretraining on the unlabelled training comments plus the external Kannada corpus, then classification starting from that adapter | +2.9 OOF on Task B, the biggest MuRIL gain |
| `rdrop` | R-Drop 0.5: two dropout passes tied by a symmetric KL | +1.1 on CodaBench; doubles the step cost |

Already settled, and not re-tested here:

| setting | why it is already settled |
|---|---|
| balanced class weights, label smoothing 0.05 | both are in the recipe |
| raw emoji | demojizing fixed MuRIL's unknown tokens; Gemma's vocabulary has emoji |
| fixed epoch count, no checkpoint selection on the holdout | kept for honesty |
| no per-class decode weights | lost every nested check |
| layer re-initialisation | is MuRIL-specific; there is nothing to re-initialise under LoRA |
| more epochs and more seeds | Run 23 |

The TAPT text is training-portion comments only, never holdout or test.

If Run 21's output is attached, `base` reuses its holdout probabilities instead of
retraining. Otherwise `base` trains too. **Time:** about 5.5 h standalone, about 4.3 h with
Run 21 attached.

Each arm is compared with `base` by paired bootstrap. The winners feed Run 25's flags.

''' + SETTINGS), ("code", setup("b", "b24_outputs")), ("code", LLM_ENV), ("code", r'''
base = {}
for s in (42, 43):
    h = find_prior(f"*/runs/{S}_ho_s{s}/holdout_probs.npy")
    p = load_probs(h[0].parent, SPLIT / "holdout.csv") if h else None
    if p is not None:
        base[s] = p
print("base reused from Run 21 for seeds:", list(base))

HO = dict(train=SPLIT / "train.csv", predict=[SPLIT / "holdout.csv"], evalf=SPLIT / "holdout.csv")
TAPT_DIR = RUNS / "tapt_train"
jobs = [tapt_job("tapt_train", [SPLIT / "train.csv", EXTERNAL]),
        llm_job("rdrop_s42", seed=42, extra="--rdrop 0.5", **HO),
        llm_job("rdrop_s43", seed=43, extra="--rdrop 0.5", **HO)]
jobs += [llm_job(f"base_s{s}", seed=s, **HO) for s in (42, 43) if s not in base]


def on_done(name, code):
    if name == "tapt_train":
        if code == 0:
            return [llm_job(f"tapt_s{s}", seed=s, extra=f"--init-adapter {TAPT_DIR}", **HO)
                    for s in (42, 43)]
        print("TAPT failed -- the tapt arm is skipped, see", LOGS / "tapt_train.log")
    return []


codes = run_queue(jobs, N_GPU, raise_on_fail=False, on_done=on_done, stop_at=HARD_STOP)
print("exit codes:", codes, f"| {(time.time() - T_START) / 3600:.1f} h")
'''), ("code", r'''
def arm(prefix):
    got = {s: load_probs(RUNS / f"{prefix}_s{s}", SPLIT / "holdout.csv") for s in (42, 43)}
    return {s: p for s, p in got.items() if p is not None}


arms = {"base": base or arm("base"), "tapt": arm("tapt"), "rdrop": arm("rdrop")}
avg, scores, boots = {}, {}, {}
for name, seeds in arms.items():
    for s, p in seeds.items():
        report(y_ho, p, f"{name} seed {s}")
    if len(seeds) == 2:
        avg[name] = np.mean(list(seeds.values()), 0)
        scores[name] = report(y_ho, avg[name], f"{name}, 2 seeds")
print()
for name in ("tapt", "rdrop"):
    if name in avg and "base" in avg:
        boots[f"{name} - base"] = boot_line(f"{name} - base", avg[name], avg["base"])
if {"tapt", "rdrop", "base"} <= set(avg):
    combo = (avg["tapt"] + avg["rdrop"]) / 2
    scores["tapt+rdrop models averaged"] = report(y_ho, combo, "tapt and rdrop models averaged")
    boots["(tapt, rdrop) avg - base"] = boot_line("(tapt, rdrop) avg - base", combo, avg["base"])
USE = {k.split(" ")[0]: v["p_better"] >= 0.7 for k, v in boots.items() if k.endswith("- base")
       and k.split(" ")[0] in ("tapt", "rdrop")}
print("\nflags for Run 25 (P(better than base) >= 0.7):", USE)
json.dump({"commit": head, "holdout_scores": scores, "bootstrap": boots, "use": USE,
           "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
''')]

# ---------------------------------------------------------------------------- B25
b25 = [("markdown", r'''
# Task B Run 25 -- the final Gemma-4-12B recipe, all 3,532 rows

This run combines whatever Runs 23 and 24 showed into four full-data models, averaged
into one submission. Set the four flags in the next cell from those runs' printed
results. The defaults are Run 21's proven recipe (0.6792 holdout) with four seeds, so the
notebook is a safe submission even with nothing changed.

| flag | default | set it to | when |
|---|---|---|---|
| `EPOCHS` | 3 | 4 | Run 23 chose `4ep` or `mix` |
| `TAPT` | False | True | Run 24 printed `tapt: True` |
| `RDROP` | 0.0 | 0.5 | Run 24 printed `rdrop: True` |
| `SEEDS` | 42-45 | -- | four models, two per GPU round |

**Time:**

| recipe | time |
|---|---|
| 3 epochs | about 3 h |
| 4 epochs | about 4 h |
| R-Drop | roughly doubles training time |
| TAPT | adds about 50 min |

The worst combination is about 8.5 h, inside the 11 h stop.

TAPT here reads all 3,532 labelled comments' text plus the external corpus; never the test
text. The output is `RECOMMENDED_b25_gemma.zip`.

''' + SETTINGS), ("code", setup("b", "b25_outputs")), ("code", LLM_ENV), ("code", r'''
EPOCHS = 3          # 4 if Run 23 chose 4ep or mix
TAPT = False        # True if Run 24 printed tapt: True
RDROP = 0.0         # 0.5 if Run 24 printed rdrop: True
SEEDS = [42, 43, 44, 45]

extra = f"--epochs {EPOCHS}" + (f" --rdrop {RDROP}" if RDROP else "")
full_jobs = lambda init: [llm_job(f"full_s{s}", SPLIT / "all.csv", [TEST_CSV], s,
                                  extra=extra + (f" --init-adapter {RUNS / 'tapt_all'}" if init else ""))
                          for s in SEEDS]
state = {"tapt_ok": False}


def on_done(name, code):
    if name == "tapt_all":
        state["tapt_ok"] = code == 0
        if code:
            print("TAPT failed -- training without it")
        return full_jobs(code == 0)
    return []


jobs = [tapt_job("tapt_all", [SPLIT / "all.csv", EXTERNAL])] if TAPT else full_jobs(False)
codes = run_queue(jobs, N_GPU, raise_on_fail=False, on_done=on_done, stop_at=HARD_STOP)
members = {s: load_probs(RUNS / f"full_s{s}", TEST_CSV) for s in SEEDS}
members = {s: p for s, p in members.items() if p is not None}
assert members, "no full fit finished -- see the logs"
print("averaging seeds", list(members))
write_zip("b25_gemma", np.mean(list(members.values()), 0), recommended=True)
json.dump({"commit": head, "epochs": EPOCHS, "tapt": TAPT and state["tapt_ok"], "rdrop": RDROP,
           "seeds": list(members), "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
print(f"{(time.time() - T_START) / 3600:.1f} h")
''')]

# ---------------------------------------------------------------------------- A28
a28 = [("markdown", r'''
# Task A Run 28 -- Gemma-4-12B on Task A: holdout check, then the submission

On Task B, Gemma-4-12B QLoRA beat the best MuRIL recipe by 6.4 points. Task A's own
lessons apply too.

**What we know on Task A.** The 1,079-row holdout of train + released validation
(`815110ff24`) gave:

| arm | macro-F1 |
|---|---|
| char SVM | 0.8117 |
| MuRIL | 0.7803 |
| SVM 0.57 + MuRIL 0.43 | 0.8155 |
| translation + hateBERT | 0.7275 |

The char SVM is Task A's strongest component, because character n-grams absorb spelling
variants. Blending it with an encoder has always helped.

**Stage 1, holdout, about 2.5 h.** Gemma-4-12B, 3 epochs, seeds 42 and 43, one per T4, on the
6,114 training rows. The char SVM is refitted here, which takes about a minute. Arms:

| arm | what |
|---|---|
| Gemma | 2 seeds averaged |
| char SVM | refitted here |
| Gemma + SVM 50/50 | fixed weight; no weight is fitted on the holdout |

The paired bootstraps are against the SVM, and between the blend and Gemma.

**Stage 2, about 3 h.** Gemma seeds 42 and 43 and the SVM on all 7,193 rows, predicting the
released test file. One ZIP each for:
- Gemma
- Gemma + SVM
- the SVM

A rule fixed in advance marks one `RECOMMENDED`:
- **the blend**, if it beats Gemma with P >= 0.7
- **Gemma**, if it beats the SVM with P >= 0.5
- **the SVM** otherwise

No cross-task label derivation and no test text in training. Task A's current best is
0.8188 on CodaBench.

''' + SETTINGS), ("code", setup("a", "a28_outputs")), ("code", LLM_ENV), ("code", r'''
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from hastika.common.preprocessing import clean


def svm_probs(train_df, pred_df):
    """Run 20's char_wb 2-5 TF-IDF + calibrated LinearSVC, demojized: 0.8117 on this holdout."""
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                      sublinear_tf=True),
                      CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3))
    X = lambda d: [clean(t, demojize=True) for t in d["Comment"]]
    m.fit(X(train_df), train_df[LABEL_COL].map(CLASSES.index))
    return m.predict_proba(X(pred_df))


train = pd.read_csv(SPLIT / "train.csv")
svm_ho = svm_probs(train, holdout)
svm_test = svm_probs(df, test)
report(y_ho, svm_ho, "char SVM (holdout)")

# holdout fits first; the full fits follow on the freed GPUs
jobs = [llm_job(f"ho_s{s}", SPLIT / "train.csv", [SPLIT / "holdout.csv"], s,
                evalf=SPLIT / "holdout.csv") for s in (42, 43)]
jobs += [llm_job(f"full_s{s}", SPLIT / "all.csv", [TEST_CSV], s) for s in (42, 43)]
codes = run_queue(jobs, N_GPU, raise_on_fail=False, stop_at=HARD_STOP)
print("exit codes:", codes, f"| {(time.time() - T_START) / 3600:.1f} h")
'''), ("code", r'''
ho = {s: load_probs(RUNS / f"ho_s{s}", SPLIT / "holdout.csv") for s in (42, 43)}
ho = {s: p for s, p in ho.items() if p is not None}
scores, boots, recommended = {"svm": report(y_ho, svm_ho, "char SVM")}, {}, "svm"
print("references on these rows: MuRIL 0.7803, SVM 0.57 + MuRIL 0.43 0.8155 (Run 20)")
if ho:
    for s, p in ho.items():
        report(y_ho, p, f"Gemma seed {s}")
    g = np.mean(list(ho.values()), 0)
    blend = (g + svm_ho) / 2
    scores["gemma"] = report(y_ho, g, f"Gemma, {len(ho)} seed(s)")
    scores["gemma+svm"] = report(y_ho, blend, "Gemma + SVM 50/50")
    boots["gemma - svm"] = boot_line("gemma - svm", g, svm_ho)
    boots["blend - gemma"] = boot_line("blend - gemma", blend, g)
    boots["blend - svm"] = boot_line("blend - svm", blend, svm_ho)
    recommended = ("gemma+svm" if boots["blend - gemma"]["p_better"] >= 0.7 else
                   "gemma" if boots["gemma - svm"]["p_better"] >= 0.5 else "svm")
print("\nrecommended:", recommended)

full = {s: load_probs(RUNS / f"full_s{s}", TEST_CSV) for s in (42, 43)}
full = [p for p in full.values() if p is not None]
zips = {"svm": svm_test}
if full:
    zips["gemma"] = np.mean(full, 0)
    zips["gemma+svm"] = (zips["gemma"] + svm_test) / 2
for name, probs in zips.items():
    write_zip(f"a28_{name.replace('+', '_')}", probs, recommended=(name == recommended))
if recommended not in zips:
    print(f"WARNING: '{recommended}' has no full fit -- submit the next best ZIP")
json.dump({"commit": head, "holdout_scores": scores, "bootstrap": boots,
           "recommended": recommended, "full_seeds": len(full), "exit_codes": codes},
          open(OUT / "result.json", "w"), indent=2)
''')]

if __name__ == "__main__":
    notebook(b22, "notebooks/task_b/22_package_run21.ipynb")
    notebook(b23, "notebooks/task_b/23_gemma_epochs_seeds_final.ipynb")
    notebook(b24, "notebooks/task_b/24_gemma_muril_lessons.ipynb")
    notebook(b25, "notebooks/task_b/25_gemma_final_recipe.ipynb")
    notebook(a28, "notebooks/task_a/28_gemma_holdout_final.ipynb")
