# Task A: Training and Prediction Guide

This guide covers the complete MuRIL/XLM-R workflow for HASTIKA Task A:

1. create the Python environment;
2. verify GPU access;
3. fine-tune the classifier;
4. inspect the saved result;
5. generate `predictions.csv`;
6. validate and zip the submission.

Run all commands from the repository root.

## 1. Install uv

Check whether `uv` is already installed:

```bash
uv --version
```

If the command is unavailable, install `uv` by following the instructions at
<https://docs.astral.sh/uv/getting-started/installation/>.

## 2. Create the environment

Choose one accelerator. CUDA 12.8 is the recommended starting point for an
NVIDIA GPU:

```bash
uv sync --extra cu128
```

Alternatives:

```bash
# Newer NVIDIA driver/CUDA 13.0
uv sync --extra cu130

# CPU-only environment
uv sync --extra cpu
```

Use the same extra in every later `uv run` command. This guide uses `cu128`.

## 3. Verify the GPU

```bash
uv run --extra cu128 python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output includes `CUDA: True` and your GPU name. Do not pass `--fp16`
when using the CPU environment.

## 4. Fine-tune MuRIL

```bash
uv run --extra cu128 hastika-task-a-train \
  --model google/muril-base-cased \
  --output-dir artifacts/checkpoints/muril_task_a \
  --fp16
```

The training script:

- reads `data/raw/binary_train.csv`;
- creates a stratified 80/20 train-validation split;
- cleans malformed Unicode and HTML by default;
- trains for up to six epochs;
- evaluates macro-F1 and accuracy after every epoch;
- stops early after two epochs without improvement;
- saves the checkpoint with the highest validation macro-F1.

The last epoch is not necessarily the saved epoch. Look for this final message:

```text
Saved best model to artifacts/checkpoints/muril_task_a
```

### Optional XLM-R comparison

```bash
uv run --extra cu128 hastika-task-a-train \
  --model xlm-roberta-base \
  --output-dir artifacts/checkpoints/xlmr_task_a \
  --fp16
```

Keep different models in different output directories.

### Optional demojized MuRIL comparison

Run the separate demojized entry point with the same seed and hyperparameters as
the original MuRIL run:

```bash
uv run --extra cu128 python -m hastika.task_a.train_demojized \
  --model google/muril-base-cased \
  --output-dir artifacts/checkpoints/muril_task_a_demojized \
  --fp16
```

This converts emoji such as `😂` into English descriptions such as
`face with tears of joy`. Results are saved under
`artifacts/checkpoints/muril_task_a_demojized/`, leaving the original checkpoint untouched.
Compare the two runs' `best_macro_f1` values before choosing a model.

## 5. Check the saved result

The MuRIL output directory should contain:

```text
artifacts/checkpoints/muril_task_a/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── training_metadata.json
└── tokenizer vocabulary files
```

Display the training history and best macro-F1:

```bash
uv run --extra cu128 python -m json.tool artifacts/checkpoints/muril_task_a/training_metadata.json
```

The important field is:

```json
"best_macro_f1": 0.7938
```

Your value will depend on the model, seed, and hyperparameters.

## 6. Generate predictions

Use the saved checkpoint with the unlabeled Task A validation inputs:

```bash
uv run --extra cu128 hastika-task-a-predict \
  --model-dir artifacts/checkpoints/muril_task_a \
  --input-csv data/raw/binary_validation_inputs.csv \
  --output artifacts/runs/task_a/predictions.csv \
  --fp16
```

The prediction script automatically reads `training_metadata.json` and reuses
the training run's text-cleaning mode, emoji mode, and maximum sequence length.
It prints the number of rows and the predicted class distribution before
finishing. To predict with the demojized model, change `--model-dir` to
`artifacts/checkpoints/muril_task_a_demojized`; no additional emoji flag is required.

Expected final output resembles:

```text
Prediction distribution: {'Non-Hate': ..., 'Hate': ...}
Wrote 806 predictions to artifacts/runs/task_a/predictions.csv
```

## 7. Check the submission

The CSV must contain 807 lines: one header plus 806 predictions.

```bash
wc -l artifacts/runs/task_a/predictions.csv
head artifacts/runs/task_a/predictions.csv
```

The first lines must follow this format:

```csv
id,label
4186,Non-Hate
5693,Hate
```

Only `Hate` and `Non-Hate` are valid Task A labels. The inference script also
checks that it produced exactly one prediction for every input row.

## 8. Zip and submit

### R-Drop experiment on an RTX 3070

Use these commands on the GPU computer. They assume an NVIDIA driver with CUDA
support and run from a clean checkout of the `task-b` branch:

```bash
# 1. Get the experiment code
git clone -b task-b https://github.com/robinpnalex/Hastika-ICON2026.git
cd Hastika-ICON2026

# 2. Install the CUDA PyTorch environment
uv sync --extra cu128
source .venv/bin/activate

# 3. Confirm that Python can see the RTX 3070
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 4. Run the matched control and R-Drop experiment
PY=python FOLDS=5 EPOCHS=6 SEEDS=42 \
  bash experiments/task_a/run_rdrop.sh
```

If the repository is already cloned, update it instead of cloning again:

```bash
cd /path/to/Hastika-ICON2026
git switch task-b
git pull --ff-only origin task-b
uv sync --extra cu128
source .venv/bin/activate
PY=python FOLDS=5 EPOCHS=6 SEEDS=42 \
  bash experiments/task_a/run_rdrop.sh
```

`FOLDS=5` is recommended because it gives a local OOF score and predictions
from five models. `SEEDS=42` keeps this run small enough for a 3070; an optional
two-seed run can be started later with `SEEDS="42 1337"`.

The script uses demojized MuRIL, two reinitialized encoder layers, FGM, EMA,
micro-batch 8 with gradient accumulation 2, and `--select last`. It trains the
control with `--rdrop 0` and the experiment with `--rdrop 0.5`, then packages both
prediction files against `binary_validation_inputs.csv`:

```text
artifacts/runs/task_a_muril_control/submission.zip
artifacts/runs/task_a_muril_rdrop/submission.zip
```

Each ZIP contains exactly `predictions.csv` with `id,label` and the valid Task A
labels `Hate` or `Non-Hate`. Check the results with:

```bash
grep -H "OOF macro-F1" artifacts/logs/task_a_muril_*.log
unzip -l artifacts/runs/task_a_muril_rdrop/submission.zip
head artifacts/runs/task_a_muril_rdrop/predictions.csv
```

Compare the OOF scores before choosing which ZIP to upload. The R-Drop arm costs
more than the control because it performs a second stochastic forward pass during
training; expect roughly 2--4 hours for one seed on a 3070, depending on the driver
and batch throughput.

```bash
uv run --extra cu128 hastika-submit --task a \
  --pred artifacts/runs/task_a/predictions.csv \
  --out artifacts/runs/task_a/submission.zip
unzip -l artifacts/runs/task_a/submission.zip
```

The archive should contain only `predictions.csv`. Upload
`artifacts/runs/task_a/submission.zip` to the matching Task A phase on CodaBench.

## Useful training options

```bash
# Reduce GPU memory usage
--batch-size 8

# Use shorter sequences
--max-length 128

# Apply balanced loss weights
--class-weight balanced

# Disable text cleaning
--raw-text

# Change the reproducible split and training seed
--seed 123
```

Example for a smaller GPU:

```bash
uv run --extra cu128 hastika-task-a-train \
  --model google/muril-base-cased \
  --output-dir artifacts/checkpoints/muril_task_a_batch8 \
  --batch-size 8 \
  --max-length 128 \
  --fp16
```

## Troubleshooting

### CUDA out of memory

Reduce `--batch-size` from 16 to 8 or 4. If needed, reduce `--max-length` from
160 to 128.

### `CUDA: False`

Confirm that the NVIDIA driver is installed, then try the CUDA extra compatible
with that driver (`cu128` or `cu130`). Re-run the GPU verification command before
training.

### Checkpoint directory does not exist

Confirm that training ended with `Saved best model to ...` and pass the same path
to `hastika-task-a-predict --model-dir`.

### Re-running an experiment

Use a new output directory for each model or seed so previous checkpoints are not
overwritten.
