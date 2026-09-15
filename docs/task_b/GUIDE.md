# Task B: run and submit

Copy-paste guide for Kaggle or a fresh GPU box. Task B is the 6-way hate
category (`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`,
`Others`), scored on macro-F1, so the two rare classes count as much as
`Gender`, which is 43% of the rows.

## What Task B may train on

Only `data/raw/multiclass_train.csv` and external corpora. **Task A's files are off
limits**, and not as a matter of taste:

| overlap | rows |
|---|---|
| `multiclass_validation_inputs.csv` ids also in `binary_train.csv` | 319 of 395 |
| `binary_validation_inputs.csv` ids also in `multiclass_train.csv` | 324 |

81% of the Task B test comments sit in Task A's training file. The two tasks
share a comment pool and are otherwise unrelated, so anything that reads Task A
text -- a warm start, a blend, even an unlabelled masked-LM pass -- is fitting
the text you are about to be scored on. `src/hastika/task_b/tapt.py` refuses those files by
name rather than trusting a flag.

External data is fine and is the only way to add rows:
`data/external/offenseval_kn.csv` (OffensEval-Dravidian Kannada, CC BY 4.0, via
`src/hastika/task_b/fetch_external.py`) and the abusive-tuned MuRIL checkpoint below. Neither
contains HASTIKA labels.

## Measured facts this guide relies on

| quantity | value |
|---|---|
| TF-IDF + LinearSVC floor, 5-fold OOF macro-F1 | 0.5948 |
| same, best of a 14-point hyperparameter sweep | 0.5979 |
| training rows after dedupe | 3,143 |
| optimizer steps per fold at `--bs 16 --epochs 6` | 942 |
| MuRIL wordpieces per whitespace word | 2.28 |
| rows truncated at `--max-len 128` (test inputs) | 2.03% |
| rows truncated at `--max-len 192` (test inputs) | 0.51% |
| corpus written in Kannada script | 0% |

Everything here is romanized. Back-transliteration was measured and rejected in
`docs/task_a/RESEARCH.md`; light spelling normalization (`prep.clean(normalize=...)`)
moves the SVM floor by +0.002, which is noise, so it stays off. The SVM's own
hyperparameters are at their plateau too -- the full sweep is in
`src/hastika/models/baseline_svm.py`'s docstring, and the spread across fourteen settings is
smaller than fold noise. The floor is not where the remaining points are.

The failure mode is worth knowing before you tune anything. On the floor's
confusion matrix, `Violence` loses 103 of its 221 rows to `Gender` and `Others`
loses 282 of 447. Reading those rows explains why: gendered slurs are used as
generic insults against TV channels and politicians, so the lexical cue points at
`Gender` while the label follows the *target*. That is also why the gazetteer
tags in `features.py` measured as noise. Task B is target identification wearing
a slur-detection costume.

## Current recipe (start here)

Three Kaggle runs settled the recipe; the results are in `docs/EXPERIMENTS.md` and the
summary is in the README's Task B section. It is **`D0_V0_noaux`**: TAPT on Kannada text
only (D0), MuRIL's tokenizer as shipped (V0), a plain six-way head (no aux). What remains
is to train it on everything and let CodaBench score it.

On Kaggle, upload `notebooks/task_b/04_full_data_fit.ipynb` and Save & Run All (~2 h). By hand:

```bash
# 1. TAPT on all 6,406 comments: nothing held out, one-word and repeated comments kept
python -u -m hastika.task_b.tapt --model google/muril-base-cased \
    --corpus data/raw/multiclass_train.csv data/external/offenseval_kn.csv --epochs 8 \
    --val-frac 0 --min-words 1 --no-dedupe --out artifacts/runs/tapt-d0v0-100

# 2. five seeds, each on all 3,159 rows, no auxiliary head
python -u -m hastika.task_b.train --tag b_d0v0_noaux_full --model artifacts/runs/tapt-d0v0-100 \
    --folds 1 --no-dedupe --aux-weight 0 --seeds 42 43 44 45 46 --epochs 6

# 3. package, then upload to the Task B validation phase
python -m hastika.common.submission --task b \
    --pred artifacts/runs/b_d0v0_noaux_full/predictions.csv --out d0v0_noaux_full.zip
```

Check the logs before uploading. TAPT should print `MLM trains on 6406 of 6406 comments,
0 held out`, and each seed `FULL FIT, 3159 rows, no validation`. There is no local F1:
the CodaBench validation score is the result. Record it in `docs/EXPERIMENTS.md`, and
read any gap under about three points against earlier submissions as noise, since that is
the standard deviation of macro-F1 on 395 rows.

Everything below is how the recipe was chosen: the same pipeline, one stage at a time,
with the holdout and fold modes that give local scores.

## 0. The overnight sweep

`experiments/task_b/overnight.py` is the whole of this guide as one unattended run. Upload
`notebooks/task_b/01_baseline_sweep.ipynb`, Save & Run All, read `RESULTS.md` in the Output tab
the next morning, submit the zip its top row names.

```bash
python -u -m experiments.task_b.overnight --budget-hours 10.5 --out /kaggle/working
```

It ranks seven ideas on a 15% holdout, promotes the best four to 5-fold runs,
blends everything with `ensemble.py`, fits a macro-F1-optimal decode on each
run with `decode_b.py`, and writes a CodaBench zip per arm into
`<out>/subs/`. `RESULTS.md` is rewritten after every arm, and `--budget-hours`
is a wall clock: no new GPU work starts once the remaining time cannot fit the
next stage, so Kaggle's 12-hour limit never cuts a fold in half.

The arms are in `STAGE1` in that file, each with the reason it is there. The
short version: the corpus is 0% Kannada script, so MuRIL's 197k Indic vocab is
mostly dead weight and a Latin-heavy encoder is a real alternative rather than a
tweak; and macro-F1 charges 1/6 of the score for `Violence`, which the floor
scores 0.346 on, so the arms that matter are the ones that move the tail.

The rest of this guide is the same pipeline run by hand, one stage at a time.

## 1. Kaggle

`notebooks/task_b/01_baseline_sweep.ipynb` is this whole guide as a notebook: upload it, flip the
switches in the second cell, Run All. The rest of this section is what it does.

Notebook settings: **Accelerator** GPU T4 x2 or P100, **Internet** on. The
training script uses one GPU whichever you pick. Neither card supports bfloat16,
so `--amp auto` selects fp16 by itself.

Do **not** run `uv sync` here. Kaggle already ships a driver-matched torch, and
the `cu128` extra would download a second copy for nothing.

```python
!git clone -q -b task-b --depth 1 https://github.com/robinpnalex/Hastika-ICON2026.git /kaggle/working/hastika
%cd /kaggle/working/hastika
!pip install -q emoji ftfy "transformers>=4.45,<6"
!python -c "import torch,transformers as t; print(torch.__version__, t.__version__, torch.cuda.get_device_name(0))"
```

Only `/kaggle/working` survives into the notebook's output, which is why the
clone goes there. Model weights cache outside it and do not count against the
20 GB output cap. For anything past half an hour use **Save Version -> Save &
Run All**; an interactive session dies with the browser tab. Budget: 12 h per
session, 30 GPU-hours a week.

### Local GPU instead

```bash
git clone -b task-b https://github.com/robinpnalex/Hastika-ICON2026.git
cd Hastika-ICON2026
uv sync --extra cu128        # --extra cu130 for a newer driver, --extra cpu for no GPU
uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available())"
```

Below, `python` means `uv run --extra cu128 python` on a local box.

## 2. Floor, then smoke test

The SVM floor takes seconds on CPU and is what every later number has to beat:

```bash
python -m hastika.models.baseline_svm --task b --demojize
```

Then two minutes of GPU, exercising the paths a full run will use:

```bash
python -u -m hastika.task_b.train --tag b_smoke --folds 0 --epochs 1 --limit 300 --tags --loss focal
```

## 3. Domain-adapt MuRIL (~25 min)

MuRIL splits these comments into 2.28 wordpieces per word because it never saw
this register. A masked-LM pass over the 6.4k in-domain comments it *is* allowed
to see was worth +2.9 points over stock MuRIL on identical folds, the only gain
this task has produced. It is not optional.

```bash
python -u -m hastika.task_b.tapt --out artifacts/runs/tapt-muril 2>&1 | tee artifacts/logs/tapt.log
```

Watch the held-out perplexity line. It starts near 2709 and should fall
substantially. With `--val-frac 0` nothing is held out, so there is no
perplexity line and the epoch lines show training loss only.

This used to OOM on a T4 and the failure was invisible, because `cmd | tee`
exits with `tee`'s status, which is always 0. The cause is structural rather
than incidental: MuRIL's MLM head emits a `bs x seq x 197285` logits tensor and
cross-entropy upcasts it to fp32, so `--bs 16 --max-len 192` is a single 2.26
GiB allocation. `--bs` now defaults to 4 with `--grad-accum 4`, which holds the
effective batch at 16. Lower `--bs` and raise `--grad-accum` by the same factor
on a smaller card.

## 4. Choose the encoder

Already decided: TAPT MuRIL beat every alternative, including XLM-R, HingRoBERTa,
mDeBERTa and the abusive checkpoint below. Kept for reproducing that comparison.

Three candidates, one flag apart. The second is a MuRIL already fine-tuned on
code-mixed Kannada abusive speech (Das et al., ACM HT 2022) whose vocabulary is
byte-identical to MuRIL's, so it is a drop-in warm start:

```bash
python -u -m hastika.task_b.train --tag b_base    --folds 0
python -u -m hastika.task_b.train --tag b_abusive --folds 0 --model Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL
python -u -m hastika.task_b.train --tag b_tapt    --folds 0 --model artifacts/runs/tapt-muril
```

`--folds 0` is a 15% holdout, about a fifth of the cost of a full run. Use it to
rank the arms, not to pick a submission.

## 5. Train and submit

Five folds, six epochs, with whichever encoder won:

```bash
python -u -m hastika.task_b.train --tag muril_b --model <winner> 2>&1 | tee artifacts/logs/muril_b.log
python -m hastika.common.submission --task b \
  --pred artifacts/runs/muril_b/predictions.csv --out /kaggle/working/task_b_predictions.zip
```

Every run prints **two** OOF scores. The `best` number picks each fold's
checkpoint by macro-F1 on that fold's own validation rows and then reports those
same rows, so it is optimistic; the `last` number takes the final checkpoint and
is unbiased. Believe `last`, and expect CodaBench to land near it. `--select
last` makes the unbiased one drive the submission too.

Blend the SVM in if both have 5-fold OOF:

```bash
python -m hastika.models.ensemble --task b
python -m hastika.common.submission --task b \
  --pred artifacts/runs/ensemble_b/predictions.csv --out /kaggle/working/task_b_predictions.zip
```

`make_submission.py` refuses to write unless the header is `id,label`, every id
in `data/raw/multiclass_validation_inputs.csv` appears exactly once, and every label
is one of the six. Upload the zip to the Task B phase on CodaBench.

## 6. Ablations

```bash
PY="python -u" FOLDS=0 bash experiments/task_b/sweep.sh   # holdout, first look
PY="python -u" bash experiments/task_b/sweep.sh           # 5-fold
SEEDS="42 43 44" PY="python -u" bash experiments/task_b/sweep.sh
```

Use the multi-seed form before believing any gap under about one point.
Fine-tuning variance at 3,143 rows is roughly that size, so one run per arm
cannot resolve the effect either way.

## Timings

Scaled from 30-40 min for a 5-fold run on an RTX 3070; a T4 is roughly two to
three times slower.

| run | T4 estimate |
|---|---|
| the whole `overnight_b.py` sweep | 8-10 h |
| smoke test | 2 min |
| `src/hastika/task_b/tapt.py` | ~25 min |
| one `--folds 0` arm | 15-25 min |
| full 5-fold run | 1.5-2 h |
| `--folds 1`, five seeds | ~95 min |
| `sweep_b.sh`, 5 core arms | 7-10 h |

## Out of memory

In order of how much they buy:

| Flag | Effect |
|---|---|
| `--bs 8 --grad-accum 2` | same effective batch, half the activations |
| `--no-fgm` | stops cloning the embedding table every step, and is ~45% of runtime |
| `--max-len 128` | truncates 2% of test rows instead of 0.5% |
| `--trim-vocab` | 293M params -> 93M; built for a 6 GB CPU box, unnecessary on a 16 GB GPU |

## Useful flags

| Flag | Default | Why |
|---|---|---|
| `--model` | `google/muril-base-cased` | also takes the abusive checkpoint or a `tapt.py` output |
| `--max-len` | 192 | not muril.py's 128; Task B's comments are longer, see the table above |
| `--select` | best | `last` gives an unbiased OOF; both are always printed |
| `--folds` | 5 | `0` gives a 15% holdout, `1` trains on every row with no local score; `ensemble.py` needs 5 |
| `--seeds` | 42 | space-separated list averages several runs |
| `--ema-decay` | 0.999 | now bias-corrected; see `EMA` in `src/hastika/models/muril.py` |
| `--no-ema-bias-correct` | off | reproduces pre-fix runs only |
| `--tags` | off | gazetteer / mood / address tags, see `features.py` |
| `--loss` | ce | `focal` down-weights already-correct rows |
| `--class-weight` | balanced | leave on. It is also why per-class logit offsets do nothing: tuned against a balanced model the best prior exponent is 0.00 |
| `--no-dedupe` | off | keeps all 3,159 rows; must match every other run you intend to blend |
| `--aux-weight` | 0 | auxiliary violent-act head; 0.3 measured as no gain in the grid |

`src/hastika/task_b/tapt.py` flags for using every comment, all off by default:

| Flag | Default | Why |
|---|---|---|
| `--val-frac` | 0.05 | `0` holds nothing back and skips the perplexity report |
| `--min-words` | 2 | `1` keeps one-word comments |
| `--no-dedupe` | off | keeps repeated comments |
| `--corpus` | Task B train + OffensEval Kannada | the D0 corpus; adding Tamil and Malayalam (D1) measured as no gain |

Everything else is inherited from the tuned Task A recipe in `src/hastika/models/muril.py`;
see its module docstring for what each choice is justified by.
