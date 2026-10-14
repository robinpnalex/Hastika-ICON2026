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
    notebooks/task_b/28_gemma_prompt_lr.ipynb        definitions prompt and lr 2e-4 on the holdout
    notebooks/task_a/28_gemma_holdout_final.ipynb    Gemma on Task A, holdout + submission
    notebooks/task_a/29-32_gemma_*.ipynb             Task A ablations: epochs, prompt, TAPT, lr
    notebooks/task_a/33_gemma_final_recipe.ipynb     the final Task A fit
    notebooks/task_a/34_final_holdout_check.ipynb    holdout twin of the Task A final ensemble
    notebooks/task_b/33_final_holdout_check.ipynb    holdout twin of the Task B final ensemble
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


def tapt_job(name, texts, extra=""):
    """Gemma TAPT: LoRA next-token training on unlabelled comments (the MuRIL +2.9 lesson)."""
    return {"name": name, "log": str(LOGS / f"{name}.log"),
            "cmd": (f"{LLM_PY} -u -m hastika.task_b.llm_tapt --model {MODEL} "
                    f"--texts {' '.join(map(str, texts))} --out {RUNS / name} --cache {HF_CACHE} {extra}")}


def load_probs(run_dir, csv):
    """<input stem>_probs.npy, or None when it is missing, belongs to other rows (another
    task's output found among the inputs), or is collapsed onto one class (Task A Run 28's
    seed 43 predicted Non-Hate for every row)."""
    run_dir, csv = pathlib.Path(run_dir), pathlib.Path(csv)
    p = run_dir / f"{csv.stem}_probs.npy"
    if not p.exists():
        return None
    ids = pd.read_csv(run_dir / f"{csv.stem}_ids.csv")["id"].tolist()
    if ids != pd.read_csv(csv)["id"].tolist():
        return None
    probs = np.load(p)
    share = np.bincount(probs.argmax(1), minlength=probs.shape[1]) / len(probs)
    if share.max() > 0.9:
        print(f"skipping {run_dir.name}: collapsed, {share.max():.0%} of rows in one class")
        return None
    return probs


def run_jobs(jobs, on_done=None):
    """run_queue with two automatic retries per job: exit code 4 (collapsed after epoch 1)
    reruns with seed + 1000; out of GPU memory reruns at micro-batch 4 (same effective
    batch). on_done sees each job's final outcome."""
    import re
    registry, retried = {j["name"]: j for j in jobs}, set()

    def done(name, code):
        job, nxt = registry.get(name), []
        if code and job is not None and name not in retried:
            log = open(job["log"], errors="replace").read()
            seed = re.search(r"--seed (\d+)", job["cmd"])
            if code == 4 and seed:
                print(f"{name} collapsed -- rerunning with seed {int(seed.group(1)) + 1000}")
                nxt = [{**job, "cmd": job["cmd"] + f" --seed {int(seed.group(1)) + 1000}"}]
            elif "OutOfMemoryError" in log and "llm_tapt" in job["cmd"]:
                # TAPT already runs at micro-batch 1 by default: last resort is shorter text
                print(f"{name} ran out of GPU memory -- rerunning TAPT at 64 tokens")
                nxt = [{**job, "cmd": job["cmd"] + " --bs 1 --grad-accum 16 --max-len 64 --logit-chunk 8"}]
            elif "OutOfMemoryError" in log:
                print(f"{name} ran out of GPU memory -- rerunning at micro-batch 4")
                nxt = [{**job, "cmd": job["cmd"] + " --bs 4 --grad-accum 4 --eval-bs 8"}]
            if nxt:
                retried.add(name)
        if not nxt and on_done is not None:
            nxt = list(on_done(name, code) or [])
        for j in nxt:
            registry[j["name"]] = j
        return nxt

    return run_queue(jobs, N_GPU, raise_on_fail=False, on_done=done, stop_at=HARD_STOP)


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
    print(run_jobs(jobs))
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
codes1 = run_jobs(jobs)
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
codes2 = run_jobs(plan)

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


codes = run_jobs(jobs, on_done)
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
# b25 is defined by final_tapt() in the TAPT line below.

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


PROMPT = "short"    # "defs" if Task B Run 28 showed the definitions prompt helps
train = pd.read_csv(SPLIT / "train.csv")
svm_ho = svm_probs(train, holdout)
svm_test = svm_probs(df, test)
report(y_ho, svm_ho, "char SVM (holdout)")

# holdout fits first; the full fits follow on the freed GPUs
jobs = [llm_job(f"ho_s{s}", SPLIT / "train.csv", [SPLIT / "holdout.csv"], s,
                evalf=SPLIT / "holdout.csv", extra=f"--prompt {PROMPT}") for s in (42, 43)]
jobs += [llm_job(f"full_s{s}", SPLIT / "all.csv", [TEST_CSV], s, extra=f"--prompt {PROMPT}")
         for s in (42, 43)]
codes = run_jobs(jobs)
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

# ---------------------------------------------------------------------------- B28
b28 = [("markdown", r'''
# Task B Run 28 -- Gemma-4-12B: the annotators' definitions in the prompt, and learning rate

Two untested levers that cost nothing to add to the final fit. Every arm is Gemma-4-12B
QLoRA, 3 epochs, seeds 42 and 43, on the 530-row holdout:

| arm | change | why |
|---|---|---|
| `base` | Run 21's recipe | the reference, 0.6753-0.6792 |
| `defs` | `--prompt defs`: the organisers' category definitions from the HASTIKA paper, condensed into the prompt | the short prompt gives only label names. The paper defines **Others** as including hate against sports, movies and news channels and untargeted hate, and **Geo-political** as including hatred of regional languages and interstate disputes. Those are exactly the boundaries Others (0.56) and Geo-political (0.70) lose on. About twice the prompt tokens. |
| `lr2e-4` | learning rate 2e-4 instead of 1e-4 | 2e-4 is the usual QLoRA rate, and every Gemma run so far was still improving at its last epoch |

`base` reuses the no-TAPT holdout probabilities of Run 21, 24 or 27 when attached, and is
trained here otherwise. Each arm is compared with `base` by paired bootstrap. A flag is
printed `True` for Run 25 when P(better) >= 0.7.

**Time:** about 4.5 h standalone, about 3.5 h with a base attached.

''' + SETTINGS), ("code", setup("b", "b28_outputs")), ("code", LLM_ENV), ("code", r'''
base = {}
for s in (42, 43):
    hits = (find_prior(f"*/runs/{S}_ho_s{s}/holdout_probs.npy")
            or find_prior(f"*/runs/base_s{s}/holdout_probs.npy")
            or find_prior(f"*/classifier/{S}_notapt_cls_s{s}/holdout_probs.npy"))
    p = load_probs(hits[0].parent, SPLIT / "holdout.csv") if hits else None
    if p is not None:
        base[s] = p
print("base reused for seeds:", list(base))

HO = dict(train=SPLIT / "train.csv", predict=[SPLIT / "holdout.csv"], evalf=SPLIT / "holdout.csv")
jobs = [llm_job(f"defs_s{s}", seed=s, extra="--prompt defs", **HO) for s in (42, 43)]
jobs += [llm_job(f"lr2e-4_s{s}", seed=s, extra="--lr 2e-4", **HO) for s in (42, 43)]
jobs += [llm_job(f"base_s{s}", seed=s, **HO) for s in (42, 43) if s not in base]
codes = run_jobs(jobs)
print("exit codes:", codes, f"| {(time.time() - T_START) / 3600:.1f} h")
'''), ("code", r'''
def arm(prefix):
    got = {s: load_probs(RUNS / f"{prefix}_s{s}", SPLIT / "holdout.csv") for s in (42, 43)}
    return {s: p for s, p in got.items() if p is not None}


arms = {"base": base or arm("base"), "defs": arm("defs"), "lr2e-4": arm("lr2e-4")}
avg, scores, boots = {}, {}, {}
for name, seeds in arms.items():
    for s, p in seeds.items():
        report(y_ho, p, f"{name} seed {s}")
    if len(seeds) == 2:
        avg[name] = np.mean(list(seeds.values()), 0)
        scores[name] = report(y_ho, avg[name], f"{name}, 2 seeds")
print()
for name in ("defs", "lr2e-4"):
    if name in avg and "base" in avg:
        boots[name] = boot_line(f"{name} - base", avg[name], avg["base"])
USE = {k: v["p_better"] >= 0.7 for k, v in boots.items()}
print("\nflags for Run 25 (P(better than base) >= 0.7):", USE)
json.dump({"commit": head, "holdout_scores": scores, "bootstrap": boots, "use": USE,
           "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
''')]

# ---------------------------------------------------------------------------- Task A line
A_FACTS = """**What Task A already knows** (1,079-row holdout of train + released validation,
`815110ff24`):

| arm | macro-F1 |
|---|---|
| char SVM | 0.8117 |
| SVM 0.57 + MuRIL 0.43 (the MuRIL-era best) | 0.8155 |
| **Gemma-4-12B QLoRA, 3 epochs, seed 42 (Run 28)** | **0.8563**, +0.043 over the SVM, CI [+0.021, +0.067] |
| Gemma + SVM, 50/50 | 0.8471, -0.007 against Gemma alone |

Run 28's seed 43 collapsed: every row came out Non-Hate. The classifier now checks after
epoch 1 and reruns a collapsed job with another seed. Seed 42's holdout curve was still
rising at the last epoch: 0.817, 0.846, 0.856."""

ABLATION_CODE = r'''
BASE_EXTRA = __BASE_EXTRA__     # the current best recipe's flags; every arm adds to them
base = {}
for s in (42, 43):
    for pat in [p.format(seed=s) for p in __BASE_REUSE__]:
        got = [load_probs(h.parent, SPLIT / "holdout.csv") for h in find_prior(pat)]
        got = [g for g in got if g is not None]
        if got:
            base[s] = got[0]
            break
print("base reused for seeds:", list(base))

ARMS = __ARMS__
HO = dict(train=SPLIT / "train.csv", predict=[SPLIT / "holdout.csv"], evalf=SPLIT / "holdout.csv")
TAPT_DIR = RUNS / "tapt_train"
jobs = [llm_job(f"{a}_s{s}", seed=s, extra=f"{BASE_EXTRA} {x}", **HO)
        for a, x in ARMS.items() if x for s in (42, 43)]
jobs += [llm_job(f"base_s{s}", seed=s, extra=BASE_EXTRA, **HO) for s in (42, 43) if s not in base]
USE_TAPT = __TAPT__
if USE_TAPT:
    jobs.insert(0, tapt_job("tapt_train", [SPLIT / "train.csv", EXTERNAL]))


def on_done(name, code):
    if name == "tapt_train":
        if code == 0:
            return [llm_job(f"tapt_s{s}", seed=s, extra=f"{BASE_EXTRA} --init-adapter {TAPT_DIR}", **HO)
                    for s in (42, 43)]
        print("TAPT failed -- the tapt arm is skipped, see", LOGS / "tapt_train.log")
    return []


codes = run_jobs(jobs, on_done)
print("exit codes:", codes, f"| {(time.time() - T_START) / 3600:.1f} h")
'''

ABLATION_REPORT = r'''
def arm(prefix):
    got = {s: load_probs(RUNS / f"{prefix}_s{s}", SPLIT / "holdout.csv") for s in (42, 43)}
    return {s: p for s, p in got.items() if p is not None}


results = {"base": base or arm("base"), **{a: arm(a) for a in ARMS}}
avg, scores, boots = {}, {}, {}
for name, seeds in results.items():
    for s, p in seeds.items():
        report(y_ho, p, f"{name} seed {s}")
        m = RUNS / f"{name}_s{s}" / "metrics.json"
        if m.exists():
            print("   per epoch (diagnostic):",
                  [round(h.get("eval_macro_f1", 0), 4) for h in json.load(open(m))["history"]])
    if seeds:
        avg[name] = np.mean(list(seeds.values()), 0)
        scores[name] = report(y_ho, avg[name], f"{name}, {len(seeds)} seed(s)")
print()
for a in ARMS:
    if a in avg and "base" in avg:
        boots[a] = boot_line(f"{a} - base", avg[a], avg["base"])
USE = {k: v["p_better"] >= 0.7 for k, v in boots.items()}
print("\nflags for the final recipe (P(better than base) >= 0.7):", USE)
if USE_TAPT and not results.get("tapt"):
    raise SystemExit("TAPT did not complete -- no TAPT result. See the tapt_train log above.")
json.dump({"commit": head, "holdout_scores": scores, "bootstrap": boots, "use": USE,
           "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
'''


def ablation(run, title, why, arms, tapt=False, minutes="", task="a", base_extra="",
             base_reuse=("*/runs/base_s{seed}/holdout_probs.npy", "*/runs/ho_s{seed}/holdout_probs.npy"),
             facts=None, base_desc="Run 28's recipe", final_run="Task A Run 33", n_train="6,114"):
    """Task A holdout ablation: each arm vs a no-change base, 2 seeds, paired bootstrap.
    The base is reused from an attached earlier output when found, else trained here."""
    rows = "\n".join(f"| `{n}` | `{x}` | {why[n]} |" if x else f"| `{n}` | TAPT adapter | {why[n]} |"
                     for n, x in arms.items())
    md = (f"# Task {task.upper()} Run {run} -- Gemma-4-12B: {title}\n\n{facts or A_FACTS}\n\n"
          "**This run** trains Gemma-4-12B QLoRA, seeds 42 and 43, on the "
          f"{n_train} training rows. Every arm is scored on the same holdout against `base`, "
          f"which is {base_desc}:\n\n"
          "| arm | change | why |\n|---|---|---|\n" + rows + "\n\n"
          "**How each arm is judged.** Paired bootstrap against `base`. A flag is printed "
          f"`True` for the final recipe ({final_run}) when P(better) >= 0.7.\n\n"
          "**The base is reused when available.** If an earlier output with the same base "
          "recipe and seeds is attached, its healthy holdout probabilities are reused. "
          "Otherwise the base trains here.\n\n"
          f"**Time:** {minutes}.\n\n" + SETTINGS)
    code = (ABLATION_CODE.replace("__ARMS__", repr(arms)).replace("__TAPT__", repr(tapt))
            .replace("__BASE_EXTRA__", repr(base_extra))
            .replace("__BASE_REUSE__", repr(list(base_reuse))))
    return [("markdown", md), ("code", setup(task, f"{task}{run}_outputs")), ("code", LLM_ENV),
            ("code", code), ("code", ABLATION_REPORT)]


a29 = ablation(29, "four epochs vs three", {
    "ep4": "Run 28's holdout curve was still rising at epoch 3 (0.817, 0.846, 0.856), as both "
           "Task B seeds were"},
    {"ep4": "--epochs 4"}, minutes="about 5.5 h standalone, about 3.2 h with Run 28 attached")
a30 = ablation(30, "the annotators' definitions in the prompt", {
    "defs": "the organisers' Hate / Non-hate definitions from the HASTIKA paper, in the prompt. "
            "Task A errors sit in comments with no profanity at all, where the definition of "
            "hate (prejudice, animosity, intolerance) carries the decision. About 2x prompt "
            "tokens"},
    {"defs": "--prompt defs"}, minutes="about 6 h standalone, about 3.5 h with a base attached")
a31 = ablation(31, "task-adaptive pretraining (TAPT)", {
    "tapt": "LoRA next-token pretraining on the 6,114 training comments plus the external "
            "Kannada corpus, then classification from that adapter. TAPT was +2.3 on Task A "
            "for MuRIL. Holdout and test text are never read"},
    {"tapt": ""}, tapt=True, minutes="about 6.5 h standalone, about 4 h with a base attached")
a32 = ablation(32, "learning rate 2e-4", {
    "lr2e-4": "2e-4 is the usual QLoRA rate. Every Gemma run so far, on both tasks, was still "
              "improving at its last epoch at 1e-4"},
    {"lr2e-4": "--lr 2e-4"}, minutes="about 5 h standalone, about 2.5 h with a base attached")

# a33 is defined by final_tapt() in the TAPT line below.

# ---------------------------------------------------------------------------- holdout twins
TWIN_CODE = r'''
# Each member: (tag, seed, classifier flags, holdout files of earlier runs that are the
# same recipe and seed -- reused when attached). Edit to mirror the full-data fit exactly.
MEMBERS = __MEMBERS__
GROUPS = __GROUPS__            # group name -> member keys "<tag>_s<seed>"
FINAL, REFERENCE = __FINAL__, __REFERENCE__
VERDICT = __VERDICT__          # (candidate, baseline) groups that decide the verdict
# holdout models also predict the test set: extra members for the final average
HO = dict(train=SPLIT / "train.csv", predict=[SPLIT / "holdout.csv", TEST_CSV], evalf=SPLIT / "holdout.csv")
TAPT_EXTRA = __TAPT_EXTRA__      # e.g. "--epochs 2"

probs, jobs, after_tapt = {}, [], []
TAPT_DIR = RUNS / "tapt_train"
for tag, seed, extra, reuse in MEMBERS:
    key = f"{tag}_s{seed}"
    for pat in reuse:
        got = [load_probs(h.parent, SPLIT / "holdout.csv") for h in find_prior(pat.format(seed=seed))]
        got = [g for g in got if g is not None]
        if got:
            probs[key] = got[0]
            print(f"{key}: reused from an attached run ({pat.format(seed=seed)})")
            break
    else:
        if "{tapt}" in extra:      # a TAPT member: classifier continues from the TAPT adapter
            after_tapt.append(llm_job(key, seed=seed, extra=extra.format(tapt=f"--init-adapter {TAPT_DIR}"), **HO))
        else:
            jobs.append(llm_job(key, seed=seed, extra=extra, **HO))
state = {"tapt_ok": not after_tapt}
if after_tapt:   # TAPT on training-split text + external corpus only, never holdout or test
    jobs.insert(0, tapt_job("tapt_train", [SPLIT / "train.csv", EXTERNAL], extra=TAPT_EXTRA))


def on_done(name, code):
    if name == "tapt_train":
        state["tapt_ok"] = code == 0
        if code == 0:
            return after_tapt
        print(f"TAPT FAILED (exit {code}) -- no TAPT member is trained; see {LOGS / 'tapt_train.log'}")
    return []


print("training on the 85% split:", [j["name"] for j in jobs + after_tapt])
codes = run_jobs(jobs, on_done)
for j in jobs + after_tapt:
    p = load_probs(RUNS / j["name"], SPLIT / "holdout.csv")
    if p is not None:
        probs[j["name"]] = p
print("exit codes:", codes, f"| {(time.time() - T_START) / 3600:.1f} h")
'''

TWIN_REPORT = r'''
for key in sorted(probs):
    report(y_ho, probs[key], key)
print()
avg, scores = {}, {}
for name, keys in GROUPS.items():
    have = [k for k in keys if k in probs]
    if have:
        avg[name] = np.mean([probs[k] for k in have], 0)
        scores[name] = report(y_ho, avg[name], f"{name} ({len(have)}/{len(keys)})")
print()
boots = {}
pairs = [(FINAL, o) for o in avg if o != FINAL] + ([VERDICT] if VERDICT[0] != FINAL else [])
for a, b_ in pairs:
    if a in avg and b_ in avg:
        boots[f"{a} - {b_}"] = boot_line(f"{a[:13]} - {b_[:13]}", avg[a], avg[b_])
if f"{VERDICT[0]} - {VERDICT[1]}" in boots:
    b = boots[f"{VERDICT[0]} - {VERDICT[1]}"]
    verdict = (f"{VERDICT[0]} BEATS {VERDICT[1]}" if b["ci95"][0] > 0 else
               f"{VERDICT[0]} is probably better" if b["p_better"] >= 0.7 else
               "no clear gain over the current best" if b["p_better"] >= 0.3 else
               "probably WORSE -- submit the current best")
    print(f"\nverdict ({VERDICT[0]} vs {VERDICT[1]}): {verdict} (P(better) {b['p_better']:.2f})")
if not state["tapt_ok"]:
    raise SystemExit("TAPT did not complete -- no TAPT result. See the tapt_train log above.")
# the FINAL group's holdout models were also asked to predict the test set
test_members = {k: load_probs(RUNS / k, TEST_CSV) for k in GROUPS[FINAL]}
test_members = {k: v for k, v in test_members.items() if v is not None}
if test_members:
    print(f"\n{len(test_members)} holdout models predicted the test set -- extra members for the final average:")
    write_zip(f"{OUT.name.replace('_outputs', '')}_holdout_models", np.mean(list(test_members.values()), 0))
json.dump({"commit": head, "holdout_fingerprint": HOLDOUT_FP, "scores": scores,
           "bootstrap": boots, "members": sorted(probs), "test_members": sorted(test_members),
           "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
'''


def holdout_twin(task, run, title, intro, members, groups, final, reference, minutes, verdict=None,
                 tapt_extra=""):
    md = (f"# Task {task.upper()} Run {run} -- holdout check: {title}\n\n" + intro + """

**How it works.** Every member of the full-data ensemble is trained with the same flags
and seed on the 85% training split, then scored on the fixed holdout. The members are
averaged exactly as the final fit averages them, and the final combination is compared
with the current best by paired bootstrap. That makes this the evidence for or against
submitting the full-data version.

**Reuse.** A member whose identical recipe and seed already ran on the holdout is reused
when that run's output is attached. Everything else is trained here. The `MEMBERS`
list in the next cell mirrors the full-data fit; if you change the final fit's flags,
change them here too.

""" + f"**Time:** {minutes}.\n\n" + SETTINGS)
    code = (TWIN_CODE.replace("__MEMBERS__", "[\n" + "".join(f"    {m!r},\n" for m in members) + "]")
            .replace("__GROUPS__", "{\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in groups.items()) + "}")
            .replace("__FINAL__", repr(final)).replace("__REFERENCE__", repr(reference))
            .replace("__VERDICT__", repr(verdict or (final, reference)))
            .replace("__TAPT_EXTRA__", repr(tapt_extra)))
    return [("markdown", md), ("code", setup(task, f"{task}{run}_outputs")), ("code", LLM_ENV),
            ("code", code), ("code", TWIN_REPORT)]


A_REUSE_3EP = ["*/runs/ho_s{seed}/holdout_probs.npy", "*/runs/base_s{seed}/holdout_probs.npy"]
a34 = holdout_twin(
    "a", 34, "Run 28 + Run 33 (4 epochs, seeds 44/45), the Task A final",
    """Today's Task A final averages four full-data Gemma-4-12B models:

| source | models |
|---|---|
| Run 28 | 3 epochs, seeds 42 and 43 |
| Run 33, run with `EPOCHS = 4`, `SEEDS = [44, 45]` | 4 epochs, seeds 44 and 45 |

Run 28's seed-42 holdout score is 0.8563; its seed 43 collapsed and is retrained here.
This notebook measures the four-model combination against **Run 28's recipe**, the
current best, on the 1,079-row holdout.""",
    [("3ep", 42, "", A_REUSE_3EP), ("3ep", 43, "", A_REUSE_3EP),
     ("4ep", 44, "--epochs 4", ["*/runs/ep4_s{seed}/holdout_probs.npy"]),
     ("4ep", 45, "--epochs 4", ["*/runs/ep4_s{seed}/holdout_probs.npy"])],
    {"current best: Run 28 recipe, 3 epochs x2": ["3ep_s42", "3ep_s43"],
     "Run 33 today: 4 epochs x2": ["4ep_s44", "4ep_s45"],
     "FINAL: all four": ["3ep_s42", "3ep_s43", "4ep_s44", "4ep_s45"]},
    "FINAL: all four", "current best: Run 28 recipe, 3 epochs x2",
    "about 5.6 h standalone (four fits, two per GPU), about 3.2 h with Run 28 attached")

B_REUSE_SHORT3 = ["*/runs/short_s{seed}/holdout_probs.npy", "*/runs/gemma-4-12b_ho_s{seed}/holdout_probs.npy",
                  "*/runs/base_s{seed}/holdout_probs.npy"]
B_REUSE_EP4 = ["*/runs/gemma-4-12b_ep4_ho_s{seed}/holdout_probs.npy"]
b33 = holdout_twin(
    "b", 33, "Run 23 + Run 32 (prompt ensemble), the Task B final",
    """Today's Task B final averages eight full-data Gemma-4-12B models:

| source | models |
|---|---|
| Run 23 | 4 epochs, standard prompt, seeds 42-45 |
| Run 32 (Robin) | 3 epochs, standard and definitions prompts, seeds 42 and 43 each |

This notebook measures the eight-model combination against **Run 23's recipe**, the
current best (`RECOMMENDED_b23_gemma.zip`), on the 530-row holdout. Members reused when
attached:
- Run 23's 4-epoch seeds 42 and 43
- Run 30's prompt-ensemble members, the same recipe as Run 32
- Run 21 or 24's 3-epoch standard seeds""",
    [("ep4", s, "--epochs 4", B_REUSE_EP4) for s in (42, 43, 44, 45)]
    + [("short3", s, "--prompt short", B_REUSE_SHORT3) for s in (42, 43)]
    + [("defs3", s, "--prompt defs", ["*/runs/defs_s{seed}/holdout_probs.npy"]) for s in (42, 43)],
    {"current best: Run 23 recipe, 4 epochs x4": [f"ep4_s{s}" for s in (42, 43, 44, 45)],
     "Run 32: prompt ensemble x4": ["short3_s42", "short3_s43", "defs3_s42", "defs3_s43"],
     "FINAL: all eight": [f"ep4_s{s}" for s in (42, 43, 44, 45)]
                         + ["short3_s42", "short3_s43", "defs3_s42", "defs3_s43"]},
    "FINAL: all eight", "current best: Run 23 recipe, 4 epochs x4",
    "about 6.3 h standalone (eight fits), about 3 h with Runs 23 and 30 attached")

B_FACTS = """**What Task B already knows** (530-row holdout of train + released validation,
`f85f4f049b`):

| recipe | macro-F1 |
|---|---|
| MuRIL, Run 9 recipe | 0.6151 |
| Gemma-4-12B, 3 epochs, 2 seeds (Run 21) | 0.6792 |
| **Gemma-4-12B, 4 epochs, 2 seeds (Run 23): the current best recipe** | **0.6872** |

For MuRIL, TAPT was the single largest gain on this task: +2.9 OOF. On Gemma it has never
completed. Three bugs stopped it, all now fixed: the logits ran out of memory, every chunk
ran its own decoder backward, and the out-of-memory retry used a flag TAPT did not
accept."""

b34 = ablation(34, "TAPT on the current best recipe (4 epochs)", {
    "tapt": "LoRA next-token pretraining on the 3,002 training comments plus the external "
            "Kannada corpus, one epoch, then the 4-epoch classifier continues from that "
            "adapter. Holdout and test text are never read"},
    {"tapt": ""}, tapt=True, task="b", base_extra="--epochs 4",
    base_reuse=("*/runs/gemma-4-12b_ep4_ho_s{seed}/holdout_probs.npy",),
    facts=B_FACTS, base_desc="Run 23's recipe, 4 epochs (the current best)",
    final_run="Task B Run 25", n_train="3,002",
    minutes="about 5 h standalone, about 3 h with Run 23's output attached")


# ---------------------------------------------------------------------------- TAPT line (8 notebooks)
# Generic full-data TAPT fit: TAPT once, then N classifier seeds from the adapter, averaged.
FINAL_TAPT_CODE = r'''
EPOCHS = __EPOCHS__          # classifier epochs (the task's best measured recipe)
TAPT_EPOCHS = __TAPT_EPOCHS__     # LoRA next-token pretraining epochs before classification
RDROP = __RDROP__           # R-Drop weight; 0 = off. Doubles the cost of each step
SEEDS = __SEEDS__
PROMPT, LR = "short", 1e-4

extra = f"--epochs {EPOCHS} --prompt {PROMPT} --lr {LR}" + (f" --rdrop {RDROP}" if RDROP else "")
full_jobs = [llm_job(f"full_s{s}", SPLIT / "all.csv", [TEST_CSV], s,
                     extra=f"{extra} --init-adapter {RUNS / 'tapt_all'}") for s in SEEDS]
state = {"tapt_ok": False}


def on_done(name, code):
    if name == "tapt_all":
        state["tapt_ok"] = code == 0
        if code:   # TAPT-only: never a silent switch to non-TAPT models
            print(f"TAPT FAILED (exit {code}) -- no models are trained; see {LOGS / 'tapt_all.log'}")
            return []
        return full_jobs
    return []


# TAPT reads the text of every labelled row plus the external corpus -- never the test text
codes = run_jobs([tapt_job("tapt_all", [SPLIT / "all.csv", EXTERNAL], extra=f"--epochs {TAPT_EPOCHS}")],
                 on_done)
if not state["tapt_ok"]:
    raise SystemExit("TAPT did not complete -- no ZIP written (TAPT-only run). See the TAPT log above.")
members = {s: load_probs(RUNS / f"full_s{s}", TEST_CSV) for s in SEEDS}
members = {s: p for s, p in members.items() if p is not None}
assert members, "no full fit finished -- see the logs"
print("averaging seeds", list(members))
write_zip("__ZIP__", np.mean(list(members.values()), 0), recommended=True)
json.dump({"commit": head, "epochs": EPOCHS, "tapt_epochs": TAPT_EPOCHS, "rdrop": RDROP,
           "seeds": list(members), "exit_codes": codes}, open(OUT / "result.json", "w"), indent=2)
print(f"{(time.time() - T_START) / 3600:.1f} h")
'''


def final_tapt(task, run, epochs, tapt_epochs, rdrop, seeds, twin, evidence, minutes):
    n = {"a": "7,193", "b": "3,532"}[task]
    test = {"a": "hastika_binary_test.csv (806 rows)", "b": "hastika_multiclass_test.csv (396 rows)"}[task]
    zip_name = f"{task}{run}_gemma_tapt" + ("_rdrop" if rdrop else "")
    recipe = "TAPT + R-Drop" if rdrop else "TAPT"
    md = (f"# Task {task.upper()} Run {run} -- {recipe} Gemma-4-12B, final submission on all {n} rows\n\n"
          f"**This runs the {recipe} recipe only.**\n\n"
          f"1. **TAPT.** {tapt_epochs} epoch{'s' if tapt_epochs > 1 else ''} of LoRA next-token "
          f"pretraining on the text of all {n} labelled comments plus the external Kannada corpus. "
          "Test text is never read.\n"
          f"2. **Classifiers.** {len(seeds)} Gemma-4-12B QLoRA classifiers (seeds "
          f"{', '.join(map(str, seeds))}; {epochs} epochs"
          + (f"; R-Drop {rdrop}" if rdrop else "") +
          f") continue from that adapter, trained on all {n} labelled rows, two per GPU round.\n"
          f"3. **Output.** Their test probabilities are averaged into `RECOMMENDED_{zip_name}.zip` "
          f"for `{test}`.\n\n"
          "**If TAPT fails, the notebook stops with an error and writes no ZIP.** A non-TAPT model "
          "is never substituted.\n\n"
          f"**Its holdout check is `task_{task}/{twin}`.** It runs the same recipe on the 85% "
          "split, so run it at the same time.\n\n" + evidence +
          "\n\nTAPT is memory-safe on a T4: micro-batch 1 x 16 accumulation, 128 tokens, chunked "
          "logits, expandable CUDA segments, and one retry at 64 tokens. It measured a peak of 10.4 "
          "of 14.6 GB in Task A.\n\n"
          f"**Time:** {minutes}. The 11 h stop packages whichever seeds finished.\n\n" + SETTINGS)
    code = (FINAL_TAPT_CODE.replace("__EPOCHS__", str(epochs)).replace("__TAPT_EPOCHS__", str(tapt_epochs))
            .replace("__RDROP__", str(rdrop)).replace("__SEEDS__", repr(list(seeds)))
            .replace("__ZIP__", zip_name))
    return [("markdown", md), ("code", setup(task, f"{task}{run}_outputs")), ("code", LLM_ENV),
            ("code", code)]


A_TAPT_EVIDENCE = """**Evidence** (Task A holdout, 1,079 rows, `815110ff24`):

| ensemble | macro-F1 |
|---|---|
| TAPT, 1 epoch + 3-epoch classifiers, 4 seeds | **0.8702**, the best Task A holdout so far |
| no-TAPT ensembles | 0.8582 (3 seeds) to 0.8610 (2 seeds) |

Single TAPT seeds score 0.848-0.857, about level with single base seeds. TAPT's gain comes
through averaging: its models disagree more usefully."""

B_TAPT_EVIDENCE = """**Evidence:** the current best Task B recipe is Gemma-4-12B, 4 epochs, which scored
0.6872 on the 530-row holdout (Run 23). TAPT gave Task A its best holdout result (0.8702).
For Task B, TAPT now runs **2 epochs**: the corpus is small enough to afford it, the Task A
LM loss was still falling at the end of epoch 1, and MuRIL's TAPT used 8 epochs."""

a33 = final_tapt("a", 33, 3, 1, 0.0, [42, 43, 44, 45], "35_tapt_final_holdout_twin", A_TAPT_EVIDENCE,
                 "about 7.5 h (TAPT about 2 h, then two rounds of two 3-epoch fits)")
a37 = final_tapt("a", 37, 3, 1, 0.5, [42, 43], "36_tapt_rdrop_holdout_twin", A_TAPT_EVIDENCE + """

**R-Drop** (two dropout passes tied by a symmetric KL) was in MuRIL's best Task B recipe.
On Gemma Task B without TAPT, it improved every single seed (+2 points) but not their
average. Two seeds keep this inside 11 h, since R-Drop doubles each fit's cost. To average
more models, combine its `a37_gemma_tapt_rdrop_test_probs.npy` with Run 33's four TAPT
models using `python -m hastika.common.combine_probs`.""",
                 "about 7 h (TAPT about 2 h, then two 3-epoch R-Drop fits in parallel)")
b25 = final_tapt("b", 25, 4, 2, 0.0, [42, 43, 44, 45], "35_tapt_final_holdout_twin", B_TAPT_EVIDENCE,
                 "about 7 h (TAPT 2 epochs about 3 h, then two rounds of two 4-epoch fits)")
b37 = final_tapt("b", 37, 4, 2, 0.5, [42, 43], "36_tapt_rdrop_holdout_twin", B_TAPT_EVIDENCE + """

**R-Drop** was in MuRIL's best Task B recipe (0.6410). On Gemma without TAPT it improved both
single seeds by about 2 points, but not their average. Two seeds keep this inside 11 h. Combine
its `b37_gemma_tapt_rdrop_test_probs.npy` with Run 25's four TAPT models using
`python -m hastika.common.combine_probs` to average more models.""",
                 "about 6.5 h (TAPT 2 epochs about 3 h, then two 4-epoch R-Drop fits in parallel)")

A_TAPT_REUSE = ["*/runs/tapt_s{seed}/holdout_probs.npy"]
a35 = holdout_twin(
    "a", 35, "exact twin of the TAPT submission (Task A Run 33)",
    """Task A Run 33 does three things: TAPT (1 epoch) on comment text plus the external corpus,
then four 3-epoch Gemma-4-12B classifiers (seeds 42-45) from that adapter, then averages
them. **This runs the same recipe on the 85% split:** TAPT reads only the 6,114 training
comments, and everything is scored on the 1,079-row holdout.

**What it reports:**

| line | what it is |
|---|---|
| verdict | a fair test: TAPT x2 vs Run 28's no-TAPT recipe x2 |
| TAPT x4 | the expected level of the submission. A modified run of this notebook measured **0.8702** |

**New: every holdout model also predicts the test set.** The four TAPT models are written
as `a35_holdout_models_test_probs.npy` plus a ZIP. They are trained on 85% of the rows, so
each is slightly weaker than a full fit. Averaging is what made TAPT win, so combine them
with Run 33's four full-data models using `python -m hastika.common.combine_probs`.""",
    [("tapt", s, "{tapt}", A_TAPT_REUSE) for s in (42, 43, 44, 45)]
    + [("base", s, "", A_REUSE_3EP) for s in (42, 43)],
    {"current best: Run 28 recipe x2": ["base_s42", "base_s43"],
     "TAPT x2": ["tapt_s42", "tapt_s43"],
     "FINAL: TAPT x4 (= Run 33)": [f"tapt_s{s}" for s in (42, 43, 44, 45)]},
    "FINAL: TAPT x4 (= Run 33)", "current best: Run 28 recipe x2",
    "about 9 h standalone (TAPT, then six 3-epoch fits two at a time), about 6.5 h with Run 28 attached",
    verdict=("TAPT x2", "current best: Run 28 recipe x2"))
a36 = holdout_twin(
    "a", 36, "TAPT + R-Drop vs TAPT (twin of Task A Run 37)",
    """Task A Run 37 is TAPT (1 epoch) plus R-Drop 0.5, with 3-epoch classifiers. **This runs the
same recipe on the 85% split.** It compares TAPT + R-Drop with TAPT alone, Run 33's recipe and
the current best at 0.8702, both at seeds 42 and 43 from the same TAPT adapter. The test is
whether R-Drop adds anything on top of TAPT.

| line | what it is |
|---|---|
| verdict | TAPT + R-Drop x2 vs TAPT x2 |
| all four | the two recipes averaged together, as the combined submission would be |

The TAPT-only control is reused from an attached Run 35 output when available. The holdout
models also predict the test set, so they can be added to the final average.""",
    [("tapt", s, "{tapt}", A_TAPT_REUSE) for s in (42, 43)]
    + [("tapt_rdrop", s, "--rdrop 0.5 {tapt}", []) for s in (42, 43)],
    {"TAPT x2 (Run 33 recipe)": ["tapt_s42", "tapt_s43"],
     "TAPT + R-Drop x2": ["tapt_rdrop_s42", "tapt_rdrop_s43"],
     "FINAL: all four": ["tapt_s42", "tapt_s43", "tapt_rdrop_s42", "tapt_rdrop_s43"]},
    "FINAL: all four", "TAPT x2 (Run 33 recipe)",
    "about 8.5 h standalone, about 6.5 h with Run 35 attached",
    verdict=("TAPT + R-Drop x2", "TAPT x2 (Run 33 recipe)"))

B_TAPT_REUSE = ["*/runs/tapt_s{seed}/holdout_probs.npy"]
b35 = holdout_twin(
    "b", 35, "exact twin of the TAPT submission (Task B Run 25)",
    """Task B Run 25 does three things: TAPT (**2 epochs**) on comment text plus the external
corpus, then four 4-epoch Gemma-4-12B classifiers (seeds 42-45) from that adapter, then
averages them. **This runs the same recipe on the 85% split:** TAPT reads only the 3,002
training comments, and everything is scored on the 530-row holdout.

| line | what it is |
|---|---|
| verdict | a fair test: TAPT x2 vs Run 23's recipe (4 epochs, no TAPT) x2 |
| TAPT x4 | the expected level of the submission |

Run 23's 4-epoch holdout probabilities are reused when attached. **New: every holdout model
also predicts the test set** (`b35_holdout_models_test_probs.npy` plus a ZIP), as extra
members to average with Run 25.""",
    [("tapt", s, "--epochs 4 {tapt}", []) for s in (42, 43, 44, 45)]
    + [("base", s, "--epochs 4", B_REUSE_EP4) for s in (42, 43)],
    {"current best: Run 23 recipe x2": ["base_s42", "base_s43"],
     "TAPT x2": ["tapt_s42", "tapt_s43"],
     "FINAL: TAPT x4 (= Run 25)": [f"tapt_s{s}" for s in (42, 43, 44, 45)]},
    "FINAL: TAPT x4 (= Run 25)", "current best: Run 23 recipe x2",
    "about 8 h standalone (TAPT 2 epochs, then six 4-epoch fits two at a time), about 6 h with Run 23 attached",
    verdict=("TAPT x2", "current best: Run 23 recipe x2"), tapt_extra="--epochs 2")
b36 = holdout_twin(
    "b", 36, "TAPT + R-Drop vs TAPT (twin of Task B Run 37)",
    """Task B Run 37 is TAPT (2 epochs) plus R-Drop 0.5, with 4-epoch classifiers. **This runs the
same recipe on the 85% split.** It compares TAPT + R-Drop with TAPT alone (Run 25's recipe),
both at seeds 42 and 43 from the same TAPT adapter.

| line | what it is |
|---|---|
| verdict | TAPT + R-Drop x2 vs TAPT x2 |
| all four | the two recipes averaged together |

R-Drop was in MuRIL's best Task B recipe. On Gemma without TAPT it improved single seeds but
not their average. The TAPT-only control is reused from an attached Run 35 output when
available. The holdout models also predict the test set.""",
    [("tapt", s, "--epochs 4 {tapt}", B_TAPT_REUSE) for s in (42, 43)]
    + [("tapt_rdrop", s, "--epochs 4 --rdrop 0.5 {tapt}", []) for s in (42, 43)],
    {"TAPT x2 (Run 25 recipe)": ["tapt_s42", "tapt_s43"],
     "TAPT + R-Drop x2": ["tapt_rdrop_s42", "tapt_rdrop_s43"],
     "FINAL: all four": ["tapt_s42", "tapt_s43", "tapt_rdrop_s42", "tapt_rdrop_s43"]},
    "FINAL: all four", "TAPT x2 (Run 25 recipe)",
    "about 8 h standalone, about 6.5 h with Run 35 attached",
    verdict=("TAPT + R-Drop x2", "TAPT x2 (Run 25 recipe)"), tapt_extra="--epochs 2")

if __name__ == "__main__":
    notebook(b22, "notebooks/task_b/22_package_run21.ipynb")
    notebook(b23, "notebooks/task_b/23_gemma_epochs_seeds_final.ipynb")
    notebook(b24, "notebooks/task_b/24_gemma_muril_lessons.ipynb")
    notebook(b25, "notebooks/task_b/25_gemma_final_recipe.ipynb")
    notebook(b28, "notebooks/task_b/28_gemma_prompt_lr.ipynb")
    notebook(a28, "notebooks/task_a/28_gemma_holdout_final.ipynb")
    notebook(a29, "notebooks/task_a/29_gemma_epochs.ipynb")
    notebook(a30, "notebooks/task_a/30_gemma_definitions_prompt.ipynb")
    notebook(a31, "notebooks/task_a/31_gemma_tapt.ipynb")
    notebook(a32, "notebooks/task_a/32_gemma_lr.ipynb")
    notebook(a33, "notebooks/task_a/33_gemma_final_recipe.ipynb")
    notebook(a34, "notebooks/task_a/34_final_holdout_check.ipynb")
    notebook(b33, "notebooks/task_b/33_final_holdout_check.ipynb")
    notebook(b34, "notebooks/task_b/34_gemma_tapt_4ep_holdout.ipynb")
    notebook(a35, "notebooks/task_a/35_tapt_final_holdout_twin.ipynb")
    notebook(b35, "notebooks/task_b/35_tapt_final_holdout_twin.ipynb")
    notebook(a36, "notebooks/task_a/36_tapt_rdrop_holdout_twin.ipynb")
    notebook(a37, "notebooks/task_a/37_tapt_rdrop_final.ipynb")
    notebook(b36, "notebooks/task_b/36_tapt_rdrop_holdout_twin.ipynb")
    notebook(b37, "notebooks/task_b/37_tapt_rdrop_final.ipynb")
