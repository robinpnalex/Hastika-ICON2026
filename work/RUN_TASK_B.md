# Task B: run and submit

Copy-paste guide for a fresh GPU machine. Task B is the 6-way hate category
(`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`),
scored on macro-F1, so the two rare classes count as much as `Gender`.

## 1. Set up

```bash
git clone https://github.com/robinpnalex/Hastika-ICON2026.git
cd Hastika-ICON2026
git checkout task-b

uv sync --extra cu128        # --extra cu130 for a newer driver, --extra cpu for no GPU
```

Confirm the GPU is visible before spending an hour on it:

```bash
uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

MuRIL's weights are ~1GB and checkpoints add more. Check you have several GB
free; `uv cache prune` clears stale entries safely if you do not.

## 2. Smoke test

Two minutes. Exercises both new code paths (`--tags`, `--loss focal`) and tells
you whether the batch size fits, before the real run.

```bash
uv run --extra cu128 python work/muril_b.py \
  --tag b_smoke --folds 0 --epochs 1 --limit 300 --tags --loss focal
```

## 3. Train

Five folds, six epochs, roughly 30-40 minutes on an RTX 3070.

```bash
uv run --extra cu128 python work/muril_b.py --tag muril_b 2>&1 | tee work/muril_b.log
```

Writes `work/runs/muril_b/` containing `predictions.csv`, `oof_probs.npy` and
`test_probs.npy`. The `.npy` files are what `ensemble.py` blends, so keep them.

The run prints a per-class report and a confusion matrix. Look at `Violence`
and `Others` first: on the TF-IDF SVM they score 0.35 and 0.43 against a 0.5948
macro-F1 overall, and they are where the leaderboard is decided.

## 4. Submit

```bash
uv run --extra cu128 python work/make_submission.py \
  --task b --pred work/runs/muril_b/predictions.csv \
  --out work/runs/muril_b/task_b_predictions.zip
```

`make_submission.py` refuses to write unless the header is `id,label`, every id
in `data/multiclass_validation_inputs.csv` is present exactly once, and every
label is one of the six. Upload the zip to the Task B phase on CodaBench.

## 5. Decide whether the new features help

`--tags` and `--loss focal` both ship OFF: on the TF-IDF SVM the tags moved
5-fold macro-F1 by -0.007, inside noise, and neither has been measured on MuRIL.
This runs the four arms on shared folds and seed, so each line differs by one
flag.

```bash
bash work/sweep_b.sh              # 4 arms, 5-fold each, ~4x a single run
FOLDS=0 bash work/sweep_b.sh      # 15% holdout, ~4x faster, first look only
SEEDS="42 43 44" bash work/sweep_b.sh
```

Use the multi-seed form before believing any gap under about one point.
Fine-tuning variance at 3,143 rows is roughly that size, so a single run per arm
cannot resolve the effect either way.

## Out of memory

In order of how much they buy:

| Flag | Effect |
|---|---|
| `--trim-vocab` | 293M params -> 93M by dropping the 95% of MuRIL's 197k wordpieces this corpus never emits |
| `--bs 8 --grad-accum 2` | same effective batch, half the activations |
| `--no-fgm` | stops cloning the embedding table every step |
| `--max-len 96` | p99 is 121 tokens, so this truncates ~1% of rows |

## Useful flags

| Flag | Default | Why |
|---|---|---|
| `--folds` | 5 | `0` gives a 15% holdout; `ensemble.py` needs 5 |
| `--seeds` | 42 | space-separated list averages several runs |
| `--tags` | off | gazetteer / mood / address tags, see `features.py` |
| `--tag-min-z` | 3.0 | gazetteer threshold; swept, 3.0 is the knee |
| `--loss` | ce | `focal` down-weights already-correct rows |
| `--class-weight` | balanced | leave on; composes with focal |
| `--no-dedupe` | off | must match every other run you intend to blend |

Everything else is inherited from the tuned Task A recipe in `work/muril.py`;
see its module docstring for what each choice is justified by.
