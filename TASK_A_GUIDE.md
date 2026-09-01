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
uv run --extra cu128 python finetune_task_a.py \
  --model google/muril-base-cased \
  --output-dir checkpoints/muril_task_a \
  --fp16
```

The training script:

- reads `data/binary_train.csv`;
- creates a stratified 80/20 train-validation split;
- cleans malformed Unicode and HTML by default;
- trains for up to six epochs;
- evaluates macro-F1 and accuracy after every epoch;
- stops early after two epochs without improvement;
- saves the checkpoint with the highest validation macro-F1.

The last epoch is not necessarily the saved epoch. Look for this final message:

```text
Saved best model to checkpoints/muril_task_a
```

### Optional XLM-R comparison

```bash
uv run --extra cu128 python finetune_task_a.py \
  --model xlm-roberta-base \
  --output-dir checkpoints/xlmr_task_a \
  --fp16
```

Keep different models in different output directories.

## 5. Check the saved result

The MuRIL output directory should contain:

```text
checkpoints/muril_task_a/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── training_metadata.json
└── tokenizer vocabulary files
```

Display the training history and best macro-F1:

```bash
uv run --extra cu128 python -m json.tool checkpoints/muril_task_a/training_metadata.json
```

The important field is:

```json
"best_macro_f1": 0.7938
```

Your value will depend on the model, seed, and hyperparameters.

## 6. Generate predictions

Use the saved checkpoint with the unlabeled Task A validation inputs:

```bash
uv run --extra cu128 python predict_task_a.py \
  --model-dir checkpoints/muril_task_a \
  --input-csv data/binary_validation_inputs.csv \
  --output predictions.csv \
  --fp16
```

The prediction script automatically reads `training_metadata.json` and reuses
the training run's text-cleaning mode and maximum sequence length. It prints the
number of rows and the predicted class distribution before finishing.

Expected final output resembles:

```text
Prediction distribution: {'Non-Hate': ..., 'Hate': ...}
Wrote 806 predictions to predictions.csv
```

## 7. Check the submission

The CSV must contain 807 lines: one header plus 806 predictions.

```bash
wc -l predictions.csv
head predictions.csv
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

```bash
zip task_a_predictions.zip predictions.csv
unzip -l task_a_predictions.zip
```

The archive should contain only `predictions.csv`. Upload
`task_a_predictions.zip` to the matching Task A phase on CodaBench.

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
uv run --extra cu128 python finetune_task_a.py \
  --model google/muril-base-cased \
  --output-dir checkpoints/muril_task_a_batch8 \
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
to `predict_task_a.py --model-dir`.

### Re-running an experiment

Use a new output directory for each model or seed so previous checkpoints are not
overwritten.
