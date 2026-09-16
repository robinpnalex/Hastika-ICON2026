# HASTIKA @ ICON-2026

Training and experiment code for Kannada-English code-mixed hate-speech
classification.

- **Task A:** binary `Hate` / `Non-Hate` classification.
- **Task B:** six-way target classification: `Gender`, `Geo-political`, `Others`,
  `Political`, `Religion`, and `Violence`.
- **Primary metric:** macro-averaged F1.

## Repository map

```text
src/hastika/             reusable, installable Python package
  common/                paths, preprocessing, submission validation
  models/                shared SVM, MuRIL, XLM-R and ensemble code
  task_a/                Task A training and prediction commands
  task_b/                Task B training, TAPT and analysis commands
experiments/              sweep and full-fit orchestration
notebooks/task_b/         ordered Kaggle entry points
data/raw/                 organizer-provided data
data/external/            permitted external corpora
data/derived/             deterministic generated datasets
artifacts/                ignored checkpoints, probabilities and logs
results/                  small historical outputs kept for comparison
submissions/              exact files sent to CodaBench
docs/                     guides, experiment history and project decisions
tests/                    fast structural and data-contract checks
```

See [the restructure guide](docs/RESTRUCTURE.md) for the conventions, migration
map, and reasoning behind this layout.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Choose the PyTorch build for
the machine:

```bash
uv sync --extra cu128  # NVIDIA CUDA 12.8
uv sync --extra cu130  # NVIDIA CUDA 13.0
uv sync --extra cpu    # CPU-only
```

Commands below assume they are run from the repository root.

## Task A

```bash
uv run --extra cu128 hastika-task-a-train \
  --model google/muril-base-cased \
  --output-dir artifacts/checkpoints/muril_task_a \
  --fp16

uv run --extra cu128 hastika-task-a-predict \
  --model-dir artifacts/checkpoints/muril_task_a \
  --input-csv data/raw/binary_validation_inputs.csv \
  --output artifacts/runs/task_a/predictions.csv \
  --fp16

uv run --extra cu128 hastika-submit \
  --task a --pred artifacts/runs/task_a/predictions.csv
```

The complete walkthrough is in [docs/task_a/GUIDE.md](docs/task_a/GUIDE.md).

## Task B

The strongest observed Task B recipe is task-adaptive pretraining of MuRIL followed by
six-way fine-tuning with only the final encoder layer reinitialized. It scored `0.6102`
five-fold OOF macro-F1 at seed 42; confirmation at seeds 43 and 44 is pending. The
provisional full-data commands are:

```bash
uv run --extra cu128 hastika-task-b-tapt \
  --model google/muril-base-cased \
  --corpus data/raw/multiclass_train.csv data/external/offenseval_kn.csv \
  --val-frac 0 --min-words 1 --no-dedupe \
  --out artifacts/runs/tapt-d0v0-reinit1-100

uv run --extra cu128 hastika-task-b-train \
  --tag b_reinit1_full \
  --model artifacts/runs/tapt-d0v0-reinit1-100 \
  --folds 1 --no-dedupe --reinit-layers 1 --aux-weight 0 \
  --seeds 42 43 44 45 46 --epochs 6

uv run --extra cu128 hastika-submit \
  --task b \
  --pred artifacts/runs/b_reinit1_full/predictions.csv \
  --out artifacts/runs/b_reinit1_full/submission.zip
```

There is no local score for a full-data fit because every labelled row is used
for training. Use the CodaBench validation phase to score its predictions.

The complete experiment history and ordered next steps are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The seed-confirmation notebooks should be
run before treating one-layer reinitialization as final.

- Commands and methodology: [docs/task_b/GUIDE.md](docs/task_b/GUIDE.md)
- Notebook index: [docs/task_b/NOTEBOOKS.md](docs/task_b/NOTEBOOKS.md)
- Recorded results: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- Submission contract: [docs/project/SUBMISSION_FORMAT.md](docs/project/SUBMISSION_FORMAT.md)

## Verification

Run the fast checks before starting a costly GPU experiment:

```bash
uv run --extra cpu python -m unittest discover -s tests -v
python3 -m compileall -q src experiments
```

Model downloads and GPU training are intentionally outside the fast test suite.
