# Task B: run and submit

Copy-paste guide for Kaggle or a fresh GPU box. Task B is the 6-way hate
category (`Gender`, `Political`, `Religion`, `Geo-political`, `Violence`,
`Others`), scored on macro-F1, so the two rare classes count as much as
`Gender`, which is 43% of the rows.

## What Task B may train on

Only `data/multiclass_train.csv` and external corpora. **Task A's files are off
limits**, and not as a matter of taste:

| overlap | rows |
|---|---|
| `multiclass_validation_inputs.csv` ids also in `binary_train.csv` | 319 of 395 |
| `binary_validation_inputs.csv` ids also in `multiclass_train.csv` | 324 |

81% of the Task B test comments sit in Task A's training file. The two tasks
share a comment pool and are otherwise unrelated, so anything that reads Task A
text -- a warm start, a blend, even an unlabelled masked-LM pass -- is fitting
the text you are about to be scored on. `work/tapt.py` refuses those files by
name rather than trusting a flag.

External data is fine and is the only way to add rows:
`data/external/offenseval_kn.csv` (OffensEval-Dravidian Kannada, CC BY 4.0, via
`work/fetch_external.py`) and the abusive-tuned MuRIL checkpoint below. Neither
contains HASTIKA labels.

## Measured facts this guide relies on

| quantity | value |
|---|---|
| TF-IDF + LinearSVC floor, 5-fold OOF macro-F1 | 0.5948 |
| training rows after dedupe | 3,143 |
| optimizer steps per fold at `--bs 16 --epochs 6` | 942 |
| MuRIL wordpieces per whitespace word | 2.28 |
| rows truncated at `--max-len 128` (test inputs) | 2.03% |
| rows truncated at `--max-len 192` (test inputs) | 0.51% |
| corpus written in Kannada script | 0% |

Everything here is romanized. Back-transliteration was measured and rejected in
`work/SETUP.md`; light spelling normalization (`prep.clean(normalize=...)`)
moves the SVM floor by +0.002, which is noise, so it stays off.

## 1. Kaggle

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
python work/baseline_svm.py --task b --demojize
```

Then two minutes of GPU, exercising the paths a full run will use:

```bash
python -u work/muril_b.py --tag b_smoke --folds 0 --epochs 1 --limit 300 --tags --loss focal
```

## 3. Domain-adapt MuRIL (optional, ~10 min)

MuRIL splits these comments into 2.28 wordpieces per word because it never saw
this register. A masked-LM pass over the 6.4k in-domain comments it *is* allowed
to see usually buys a point or two on small code-mixed sets.

```bash
python -u work/tapt.py --out work/runs/tapt-muril 2>&1 | tee work/tapt.log
```

Watch the held-out perplexity line. If it does not fall substantially, skip the
`--model work/runs/tapt-muril` arm below.

## 4. Choose the encoder

Three candidates, one flag apart. The second is a MuRIL already fine-tuned on
code-mixed Kannada abusive speech (Das et al., ACM HT 2022) whose vocabulary is
byte-identical to MuRIL's, so it is a drop-in warm start:

```bash
python -u work/muril_b.py --tag b_base    --folds 0
python -u work/muril_b.py --tag b_abusive --folds 0 --model Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL
python -u work/muril_b.py --tag b_tapt    --folds 0 --model work/runs/tapt-muril
```

`--folds 0` is a 15% holdout, about a fifth of the cost of a full run. Use it to
rank the arms, not to pick a submission.

## 5. Train and submit

Five folds, six epochs, with whichever encoder won:

```bash
python -u work/muril_b.py --tag muril_b --model <winner> 2>&1 | tee work/muril_b.log
python work/make_submission.py --task b \
  --pred work/runs/muril_b/predictions.csv --out /kaggle/working/task_b_predictions.zip
```

Every run prints **two** OOF scores. The `best` number picks each fold's
checkpoint by macro-F1 on that fold's own validation rows and then reports those
same rows, so it is optimistic; the `last` number takes the final checkpoint and
is unbiased. Believe `last`, and expect CodaBench to land near it. `--select
last` makes the unbiased one drive the submission too.

Blend the SVM in if both have 5-fold OOF:

```bash
python work/ensemble.py --task b
python work/make_submission.py --task b \
  --pred work/runs/ensemble_b/predictions.csv --out /kaggle/working/task_b_predictions.zip
```

`make_submission.py` refuses to write unless the header is `id,label`, every id
in `data/multiclass_validation_inputs.csv` appears exactly once, and every label
is one of the six. Upload the zip to the Task B phase on CodaBench.

## 6. Ablations

```bash
PY="python -u" FOLDS=0 bash work/sweep_b.sh   # holdout, first look
PY="python -u" bash work/sweep_b.sh           # 5-fold
SEEDS="42 43 44" PY="python -u" bash work/sweep_b.sh
```

Use the multi-seed form before believing any gap under about one point.
Fine-tuning variance at 3,143 rows is roughly that size, so one run per arm
cannot resolve the effect either way.

## Timings

Scaled from 30-40 min for a 5-fold run on an RTX 3070; a T4 is roughly two to
three times slower.

| run | T4 estimate |
|---|---|
| smoke test | 2 min |
| `work/tapt.py` | 10-15 min |
| one `--folds 0` arm | 15-25 min |
| full 5-fold run | 1.5-2 h |
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
| `--folds` | 5 | `0` gives a 15% holdout; `ensemble.py` needs 5 |
| `--seeds` | 42 | space-separated list averages several runs |
| `--ema-decay` | 0.999 | now bias-corrected; see `EMA` in `work/muril.py` |
| `--no-ema-bias-correct` | off | reproduces pre-fix runs only |
| `--tags` | off | gazetteer / mood / address tags, see `features.py` |
| `--loss` | ce | `focal` down-weights already-correct rows |
| `--class-weight` | balanced | leave on. It is also why per-class logit offsets do nothing: tuned against a balanced model the best prior exponent is 0.00 |
| `--no-dedupe` | off | must match every other run you intend to blend |

Everything else is inherited from the tuned Task A recipe in `work/muril.py`;
see its module docstring for what each choice is justified by.
