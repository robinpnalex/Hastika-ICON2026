# HASTIKA @ ICON-2026 — Starting Kit

**Hate Speech and Target Category Identification in Kannada-English Code-Mixed Text**

This repository contains the **training-phase** data, submission format, a baseline, and
sample submissions for the HASTIKA shared task. Registration, submission, and the
leaderboards are on CodaBench:

➡️ **CodaBench:** https://www.codabench.org/competitions/17784/

---

## Overview

Social media in India is heavily **code-mixed** — native languages blended with English in a
single utterance. HASTIKA targets hate speech detection in low-resource **Kannada-English
(Kanglish)** text, using **8,058 manually annotated** YouTube comments.

The task has two independent sub-tasks (**separate leaderboards** — you may enter either or both):

- **Task A — Binary Hate Speech Detection:** classify a comment as `Hate` or `Non-Hate`.
- **Task B — Fine-Grained Hate Speech Classification:** classify a **hate** comment into one of
  six target categories: `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`.

---

## Data

This repo provides the **training-phase** files (released 20 Aug). Test inputs are released
on 20 Sep; test gold labels are never released — scoring happens on CodaBench.

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `data/binary_train.csv` | 6,446 | `id, Comment, Label` | Task A training data (with labels) |
| `data/binary_validation_inputs.csv` | 806 | `id, Comment` | Task A validation inputs (**no labels** — predict & submit) |
| `data/multiclass_train.csv` | 3,159 | `id, Comment, Hate Category` | Task B training data (with labels) |
| `data/multiclass_validation_inputs.csv` | 395 | `id, Comment` | Task B validation inputs (**no labels** — predict & submit) |

**Labels**
- Task A `Label`: `Hate` / `Non-Hate`
- Task B `Hate Category`: `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`

> Note: keep the `id` column from the provided file **unchanged** in your submission — it is how
> predictions are matched to the gold labels.

---

## Submission format

Submit a single **`predictions.csv`** (zipped) to the matching phase/task on CodaBench.

**Task A — `predictions.csv`**
```
id,label
7417,Non-Hate
958,Hate
```
Accepted label values: `Hate`, `Non-Hate`.

**Task B — `predictions.csv`**
```
id,label
958,Political
4204,Political
```
Accepted label values: `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`.

**Rules**
- One row per `id` from the input file.
- Header must be `id,label`.
- Zip **only** `predictions.csv` (no enclosing folder) before uploading. If your tool nests it in a
  folder, that is handled, but a flat zip is safest.
- UTF-8 encoding.

See `starting_kit/` for ready-made sample submissions.

---

## Evaluation

- **Macro-averaged F1** — the **primary ranking metric** (weights every class equally, so rare
  categories such as *Geo-political* count as much as frequent ones).
- **Accuracy** — reported alongside.

Task A is scored over the two binary classes; Task B over the six categories.

---

## Task A Transformer Fine-Tuning

`finetune_task_a.py` fine-tunes a Hugging Face sequence-classification model for
binary hate-speech detection. It creates a stratified validation split from
`data/binary_train.csv`, selects the checkpoint with the best validation macro-F1,
and saves the model and tokenizer. The supplied
`data/binary_validation_inputs.csv` file has no labels, so it is not used for
early stopping or hyperparameter tuning.

### Installation

The project uses [uv](https://docs.astral.sh/uv/) for Python, virtual-environment,
and dependency management. Choose one PyTorch accelerator extra and run it from
the repository root. CUDA 12.8 is the recommended starting point for an NVIDIA
GPU:

```bash
# NVIDIA GPU using CUDA 12.8
uv sync --extra cu128

# Alternatives: newer CUDA 13.0 or CPU-only
uv sync --extra cu130
uv sync --extra cpu
```

`uv` reads `pyproject.toml` and `uv.lock`, creates `.venv`, installs the locked
dependencies, and uses the Python version in `.python-version`. You do not need
to activate the environment when commands are launched with `uv run`. Use the
same accelerator extra for `uv sync` and every `uv run` command.

Before training, confirm that PyTorch can see your GPU:

```bash
uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Train with MuRIL

MuRIL is the default model and is a good first choice for Kannada-English text.
The command below enables FP16 mixed precision on a supported NVIDIA GPU.

```bash
uv run --extra cu128 python finetune_task_a.py \
  --model google/muril-base-cased \
  --output-dir checkpoints/muril_task_a \
  --fp16
```

### Train with XLM-R

```bash
uv run --extra cu128 python finetune_task_a.py \
  --model xlm-roberta-base \
  --output-dir checkpoints/xlmr_task_a \
  --fp16
```

The first run downloads the selected model from Hugging Face. Training progress
reports loss, macro-F1, and accuracy for each epoch. The best model is written to
the selected output directory, together with `training_metadata.json`.

### Useful options

```bash
# Reduce GPU memory use
--batch-size 8

# Change sequence length
--max-length 128

# Try balanced loss weighting
--class-weight balanced

# Use the original comments without HTML/encoding cleanup
--raw-text

# Reproduce a different training split/seed
--seed 123
```

For a reliable comparison, train MuRIL and XLM-R with the same seed and settings,
then repeat the best configuration with several seeds. The competition’s primary
metric is macro-F1, so use it—not accuracy—to choose checkpoints.


## Important dates

| Date | Milestone |
|------|-----------|
| 25 Aug | Training data released (this repo) |
| 20 Sep | Test inputs released |
| 01 Oct | Final submission deadline |
| 04 Oct | Results & rankings |
| 25 Oct | System paper deadline |
| 10 Dec | Camera-ready working notes |

*Tentative; deadlines 23:59 AoE unless noted. See CodaBench for the authoritative schedule.*

---

## Citation

If you use this data or take part, please cite the HASTIKA dataset paper

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
and the shared task overview paper.

## Contact

shankar.biradar@manipal.edu · sanjana.kavatagi@manipal.edu
