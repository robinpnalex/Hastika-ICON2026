# HASTIKA @ ICON-2026

Kannada-English code-mixed hate-speech classification for the HASTIKA shared
task. The current implementation work is focused on **Task A**, the binary
`Hate` / `Non-Hate` classification problem.

## Start here

The supported Task A workflow is deliberately small:

1. Create the environment with `uv`.
2. Fine-tune MuRIL on a stratified split of the labeled training data.
3. Generate predictions for the unlabeled validation or test CSV.
4. Validate and zip the resulting submission.

For copy-paste commands and troubleshooting, use
[`docs/TASK_A_GUIDE.md`](docs/TASK_A_GUIDE.md).

## Repository layout

```text
.
├── data/                         # Organizer-provided Task A and Task B CSVs
├── scripts/
│   ├── make_submission.py        # Validate and zip a prediction CSV
│   └── task_a/
│       ├── finetune.py           # Main MuRIL/XLM-R fine-tuning entry point
│       ├── finetune_demojized.py # Demojized Task A experiment
│       └── predict.py            # Inference on any compatible input split
├── docs/
│   ├── TASK_A_GUIDE.md           # Complete run guide
│   ├── TRAINING_RESULTS.md       # Recorded experiment results
│   ├── FORMAT.md                 # Submission format reference
│   └── LICENSE_NOTE.md           # Dataset usage note
├── work/                         # Advanced CV, SVM, and ensemble experiments
├── baseline/                     # Original baseline supplied with the task
├── starting_kit/                 # Organizer examples
├── references/                   # Dataset paper and background material
├── submissions/                  # Generated submission archives
├── pyproject.toml                # Python dependencies and uv configuration
└── uv.lock                       # Reproducible dependency lockfile
```

The files under `scripts/task_a/` are the recommended starting point. The
`work/` directory contains research experiments and is not required for a basic
fine-tune-and-submit run. Generated checkpoints belong in `checkpoints/` and
are ignored by Git.

## Data

| File | Rows | Purpose |
|---|---:|---|
| `data/binary_train.csv` | 6,446 | Task A labeled training data |
| `data/binary_validation_inputs.csv` | 806 | Task A unlabeled prediction input |
| `data/multiclass_train.csv` | 3,159 | Task B labeled training data |
| `data/multiclass_validation_inputs.csv` | 395 | Task B unlabeled prediction input |

Task A uses the labels `Hate` and `Non-Hate`. Task B uses `Gender`,
`Political`, `Religion`, `Geo-political`, `Violence`, and `Others`. Preserve
the supplied `id` values exactly.

## Environment setup

Run everything from the repository root. For an NVIDIA GPU with CUDA 12.8:

```bash
uv sync --extra cu128
uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Alternatives are `--extra cu130` for CUDA 13.0 and `--extra cpu` for a CPU-only
environment. Use the same extra for `uv sync` and subsequent `uv run` commands.

## Train Task A

Original-text MuRIL:

```bash
uv run --extra cu128 python scripts/task_a/finetune.py \
  --model google/muril-base-cased \
  --output-dir checkpoints/muril_task_a \
  --fp16
```

Demojized MuRIL, saved separately:

```bash
uv run --extra cu128 python scripts/task_a/finetune_demojized.py \
  --model google/muril-base-cased \
  --output-dir checkpoints/muril_task_a_demojized \
  --fp16
```

Each run saves the best model, tokenizer, validation history, preprocessing
settings, and best macro-F1 in its selected checkpoint directory.

## Predict and package

```bash
uv run --extra cu128 python scripts/task_a/predict.py \
  --model-dir checkpoints/muril_task_a_demojized \
  --input-csv data/binary_validation_inputs.csv \
  --output predictions.csv \
  --fp16

uv run --extra cu128 python scripts/make_submission.py \
  --pred predictions.csv \
  --task a \
  --out submissions/task_a_predictions.zip
```

`predict.py` reads `training_metadata.json`, so it automatically applies the
same cleaning, emoji handling, and maximum sequence length used during
training. It works with a future test split as long as the CSV contains an ID
column and a `Comment` column; pass that file through `--input-csv`.

The archive contains one file named `predictions.csv` with exactly these
columns:

```csv
id,label
7417,Non-Hate
958,Hate
```

Macro-F1 is the primary model-selection metric. Accuracy is useful as a
secondary diagnostic, but it should not determine the winning checkpoint.

## Results

Recorded Task A experiments are in
[`docs/TRAINING_RESULTS.md`](docs/TRAINING_RESULTS.md). The best currently
recorded score is the demojized TF-IDF + LinearSVC experiment at **0.8103
five-fold CV macro-F1**. The demojized MuRIL run reached **0.8023 holdout
macro-F1**. These values use different evaluation protocols and therefore are
not a direct model ranking; compare future candidates on identical folds.

## Dataset citation

```bibtex
@article{kavatagi2025hastika,
  title={HASTIKA: hate speech and target identification in Kannada-English code-mixed text: S. Kavatagi, R. Rachh},
  author={Kavatagi, Sanjana and Rachh, Rashmi},
  journal={Language Resources and Evaluation},
  volume={59},
  number={3},
  pages={2811--2856},
  year={2025},
  publisher={Springer}
}
```

See [`docs/LICENSE_NOTE.md`](docs/LICENSE_NOTE.md) before redistributing the
dataset or derived artifacts.
