# Kaggle notebooks

Upload one of these to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and
**Internet** on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch
itself, so the notebook file is all you upload. Never run these interactively: the
session dies with the browser tab and they all run for hours.

## Run this one

| notebook | what it does | time | score |
|---|---|---|---|
| `kaggle_d0v0_noaux_full.ipynb` | The current recipe, `D0_V0_noaux`, on 100% of the data. TAPT on all 6,406 comments with nothing held out, then five seeds on all 3,159 rows with no deduplication and no auxiliary head. Reads its own logs and stops if either count is short. | ~2 h | CodaBench only |

`D0_V0_noaux` means TAPT on Kannada text only (D0), MuRIL's tokenizer as shipped (V0),
and a plain six-way head (no aux). It led the data-processing grid and is the recipe
behind both CodaBench submissions so far.

Nothing is held out, so the notebook prints no F1. Upload `d0v0_noaux_full.zip` to the
Task B validation phase on CodaBench; the macro-F1 it returns is the run's score. Record
it in `TRAINING_RESULTS.md` and `submissions/README.md`.

## Earlier runs, kept for reproducibility

| notebook | what it answered | time | result |
|---|---|---|---|
| `kaggle_task_b.ipynb` | Run 1, 12 Sep. Which encoder and loss: holdout ranking, then 5-fold on the leaders. | ~8.5 h | TAPT MuRIL 0.6013 OOF, the only arm above the floor |
| `kaggle_rebuild_winner.ipynb` | Rebuilds only `b_tapt_5f` from Run 1. | ~2 h | 0.6013 OOF, 0.5922 on CodaBench |
| `kaggle_fullfit_sweep.ipynb` | Run 2, 13 Sep. Six ideas, five seeds each, on all 3,143 deduplicated rows. TAPT still held 5% of its text back. | ~9.5 h | no local score; `f_tapt` 0.6007 on CodaBench (inferred) |
| `kaggle_fullfit_single.ipynb` | One full fit, one seed. The minimal version of Run 2. | ~45 min | no recorded run |
| `kaggle_grid.ipynb` | Run 3, 13-14 Sep. Factorial over TAPT corpus, vocabulary and an auxiliary head, on the 472-row holdout. | ~8.5 h | no lever helped; vocabulary extension cost 3-8 points |

Every number above is in `TRAINING_RESULTS.md`, with the tables.

## Holdout runs decide, full fits build

A holdout or fold run keeps rows back so it can score itself; that is how the recipe was
chosen. A full fit trains on every labelled row, so nothing is left to measure it with,
and the official validation file has no labels. Once the recipe is fixed, the full fit is
what gets submitted and CodaBench is where it gets scored. Do not read a full fit's
missing local score as a good sign, and do not copy a holdout number onto it.

## Outputs

Everything lands in `/kaggle/working`. Download the zip, the logs and any
`*_test_probs.npy` individually rather than using Download All, because the TAPT
checkpoints in there are about a gigabyte each.
