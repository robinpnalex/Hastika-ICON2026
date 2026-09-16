# Kaggle notebooks

Upload one of these to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and
**Internet** on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch
itself, so the notebook file is all you upload. Never run these interactively: the
session dies with the browser tab and they all run for hours.

## Run this one

| notebook | what it does | time | score |
|---|---|---|---|
| `08_full_data_fit_reinit1.ipynb` | Provisional final fit using one-layer reinitialization: TAPT on all 6,406 comments, five seeds on all 3,159 labelled rows, inference on the 395 official validation inputs, and ZIP packaging. | ~2 h | CodaBench only |
| `04_full_data_fit.ipynb` | The current recipe, `D0_V0_noaux`, on 100% of the data. TAPT on all 6,406 comments with nothing held out, then five seeds on all 3,159 rows with no deduplication and no auxiliary head. Reads its own logs and stops if either count is short. | ~2 h | CodaBench only |

`D0_V0_noaux` means TAPT on Kannada text only (D0), MuRIL's tokenizer as shipped (V0),
and a plain six-way head (no aux). It led the data-processing grid. The current
provisional candidate adds `--reinit-layers 1`, based on Run 5's `0.6102` OOF result.

Nothing is held out in either full-data notebook, so neither prints a local F1. Upload the
ZIP from the selected candidate to the Task B validation phase on CodaBench; the macro-F1
it returns is the run's score. Record it in `docs/EXPERIMENTS.md` and
`submissions/README.md`.

The `08_full_data_fit_reinit1.ipynb` candidate uses the same full-data protocol but sets
`--reinit-layers 1`, based on the observed 0.6102 OOF result. Run it after the seed
confirmation if you want the confirmed recipe; its ZIP is `b_reinit1_full.zip`.

## Current ablation

| notebook | what it does | time | score |
|---|---|---|---|
| `05_reinit_one_layer.ipynb` | Run 5. Reuses one TAPT checkpoint and compares the established top-two-layer reset with a top-one-layer reset using identical five-fold settings. | ~3--3.5 h | 0.6013 vs **0.6102** |
| `06_reinit_confirmation.ipynb` | Run 6. Repeats both settings at model seeds 43 and 44, then compares individual and probability-averaged OOF scores. | ~6--7 h | pending |
| `07_no_reinit_ablation.ipynb` | Run 7. Compares the current one-layer setting with no encoder-layer reinitialization at seeds 43 and 44. | ~6--7 h | pending |

This is a local controlled experiment, not a submission notebook. It runs the TAPT pass
once, then trains `b_tapt_reinit2_5f` and `b_tapt_reinit1_5f` with seed 42, five folds,
six epochs and `--select last`. The completed run scored 0.6013 for two layers and
0.6102 for one layer; the one-layer variant improved macro-F1 by 0.0089 and was
particularly better on `Violence` and `Others`. The outputs belong under `artifacts/`;
download the logs and probability matrices from Kaggle for review.

Run 6 is the confirmation before changing the recommended recipe. It must show the
one-layer setting winning at both seeds, or in the averaged-probability comparison,
before it is used for a full-data fit.

Run 7 asks the next nested question: after keeping layer 11, should we keep layer 12 as
well? It compares `--reinit-layers 1` against `--reinit-layers 0` under the same protocol.

## Earlier runs, kept for reproducibility

| notebook | what it answered | time | result |
|---|---|---|---|
| `01_baseline_sweep.ipynb` | Run 1, 12 Sep. Which encoder and loss: holdout ranking, then 5-fold on the leaders. | ~8.5 h | TAPT MuRIL 0.6013 OOF, the only arm above the floor |
| `01_rebuild_winner.ipynb` | Rebuilds only `b_tapt_5f` from Run 1. | ~2 h | 0.6013 OOF, 0.5922 on CodaBench |
| `02_fullfit_sweep.ipynb` | Run 2, 13 Sep. Six ideas, five seeds each, on all 3,143 deduplicated rows. TAPT still held 5% of its text back. | ~9.5 h | no local score; `f_tapt` 0.6007 on CodaBench (inferred) |
| `02_fullfit_single.ipynb` | One full fit, one seed. The minimal version of Run 2. | ~45 min | no recorded run |
| `03_factorial_grid.ipynb` | Run 3, 13-14 Sep. Factorial over TAPT corpus, vocabulary and an auxiliary head, on the 472-row holdout. | ~8.5 h | no lever helped; vocabulary extension cost 3-8 points |

Every number above is in `docs/EXPERIMENTS.md`, with the tables.

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
