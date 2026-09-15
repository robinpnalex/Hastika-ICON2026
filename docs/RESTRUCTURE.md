# Repository restructure

This document records the September 2026 repository reorganization. It is both
a migration guide for old commands and the ownership contract for new files.

## Why it changed

The original `work/` directory accumulated reusable model code, experiment
runners, Kaggle notebooks, generated checkpoints, logs, and documentation. That
made it hard to answer three basic questions:

1. Which code is reusable and expected to remain stable?
2. Which file should be run for a particular experiment?
3. Which outputs belong in Git?

The new layout gives each category one home. The restructure is intended to
change organization, not the established modeling recipes or recorded scores.

## Directory ownership

| path | purpose | committed? |
|---|---|---|
| `src/hastika/common/` | shared paths, text preparation, submission validation | yes |
| `src/hastika/models/` | reusable SVM, transformer, MuRIL and ensemble implementations | yes |
| `src/hastika/task_a/` | supported Task A training and prediction entry points | yes |
| `src/hastika/task_b/` | supported Task B TAPT, training and analysis entry points | yes |
| `experiments/` | multi-run orchestration and ablation scripts | yes |
| `notebooks/task_b/` | ordered, self-contained Kaggle entry points | yes |
| `data/raw/` | files released by the organizers, unchanged | yes |
| `data/external/` | documented external corpora allowed by the task rules | when licensing permits |
| `data/derived/` | deterministic derivatives such as the fixed holdout split | selectively |
| `artifacts/` | checkpoints, logs, probability matrices and ordinary predictions | no |
| `results/` | small historical outputs retained as experimental evidence | yes |
| `submissions/` | exact prediction CSV/ZIP pairs actually submitted | yes |
| `docs/` | guides, decisions and experiment history | yes |
| `tests/` | fast contracts that do not download models or require a GPU | yes |

The key distinction is `artifacts/` versus `results/` versus `submissions/`:

- An **artifact** is regenerable output from an ordinary run.
- A **result** is a small historical record used to support a conclusion.
- A **submission** is the exact immutable payload sent to CodaBench, accompanied
  by enough metadata to identify its recipe and score.

Do not commit model checkpoints or full run directories. If a checkpoint must
be shared, publish it through an artifact/model store and record its identifier
and checksum in the corresponding result or submission README.

## Migration map

| old location | new location |
|---|---|
| `work/prep.py` | `src/hastika/common/preprocessing.py` |
| `src/hastika/common/submission.py` | `src/hastika/common/submission.py` |
| `src/hastika/models/muril.py` | `src/hastika/models/muril.py` |
| `src/hastika/models/baseline_svm.py` | `src/hastika/models/baseline_svm.py` |
| `work/train_xlmr.py` | `src/hastika/models/train_xlmr.py` |
| `src/hastika/models/ensemble.py` | `src/hastika/models/ensemble.py` |
| `src/hastika/task_b/train.py` | `src/hastika/task_b/train.py` |
| `src/hastika/task_b/tapt.py` | `src/hastika/task_b/tapt.py` |
| `work/{axes,features,corpora}.py` | `src/hastika/task_b/` |
| `work/{fullfit_b,grid_b,overnight_b}.py` | `experiments/task_b/` |
| `work/kaggle_*.ipynb` | `notebooks/task_b/`, ordered by experiment stage |
| `work/runs/` | `artifacts/runs/` |
| `work/*.log` | `artifacts/logs/` |
| `data/*.csv` | `data/raw/*.csv` for organizer-provided inputs |
| `data/holdout/` | `data/derived/task_b_holdout/` |
| root guides and results | `docs/` |
| diagnostic log directories | `results/task_b/diagnostics/` |

## Command migration

The project is now an installable `src` package. Prefer the console commands:

```bash
hastika-task-a-train
hastika-task-a-predict
hastika-task-b-tapt
hastika-task-b-train
hastika-submit
```

Every command can also be invoked as a module, which is convenient in notebooks
and experiment runners:

```bash
python -m hastika.models.baseline_svm --task b --demojize
python -m hastika.task_b.tapt --out artifacts/runs/tapt-muril
python -m hastika.task_b.train --tag b_tapt_5f \
  --model artifacts/runs/tapt-muril --folds 5
python -m hastika.models.ensemble --task b
python -m hastika.common.submission --task b \
  --pred artifacts/runs/b_tapt_5f/predictions.csv
```

Run these through `uv run --extra <accelerator>` locally. Kaggle notebooks add
`src/` to `PYTHONPATH` after cloning and then use the same module commands.

## Data rules

- Files in `data/raw/` are treated as immutable. Cleaning occurs in memory.
- External sources must document provenance, license, label mapping, and overlap
  removal.
- Task B test comments remain gated in `hastika.task_b.corpora`; transductive use
  requires the explicit `--allow-transductive` flag.
- Deterministic derivatives belong in `data/derived/` and must have a generating
  command. The fixed Task B split is generated with:

```bash
python -m hastika.task_b.split_holdout
```

## Experiment and notebook rules

- Reusable training behavior belongs under `src/`; orchestration belongs under
  `experiments/`.
- An experiment runner may call package modules but should not duplicate their
  training implementation.
- New notebooks use numeric prefixes, run top-to-bottom, raise on failed
  subprocesses, and write downloadable outputs outside the cloned repository on
  Kaggle.
- Expensive runs should log the Git commit, command, seed, row counts, model, and
  elapsed time.
- Full-data fits must not report copied holdout scores as if they measured the
  full fit. Their only score comes from CodaBench.

## Documentation ownership

- `README.md` is the short entry point and current recommended commands.
- `docs/task_a/GUIDE.md` and `docs/task_b/GUIDE.md` are operational guides.
- `docs/EXPERIMENTS.md` is the historical ledger of measured results.
- `docs/task_b/NOTEBOOKS.md` explains which notebook to run.
- A submission-specific README records the exact recipe and external score.

When a command or path changes, update the relevant guide and notebook in the
same commit. Do not add a second competing guide at the repository root.

## Verification after the move

The restructure is checked without downloading transformer weights:

```bash
python3 -m compileall -q src experiments
uv run --extra cpu python -m unittest discover -s tests -v
git diff --check
```

The checks cover package paths, preprocessing behavior, organizer data presence,
the deterministic Task B split, notebook JSON, and submission label contracts.
Actual model training still requires a GPU smoke run before an expensive sweep.
