# Kaggle notebooks

Upload one of these to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and
**Internet** on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch
itself, so the notebook file is all you upload. Never run these interactively: the
session dies with the browser tab and they all run for hours.

## Queued, not yet run

| notebook | what it answers | time | score |
|---|---|---|---|
| `10_context_decode.ipynb` | Run 10. Does a violent action word mean Violence only when no target group is named? Five folds of the current recipe, then a per-cell prior correction fitted and nested on the OOF. Writes a ZIP only if the nested check wins. | ~2.6 h | yes, 5-fold OOF |
| `11_context_tags.ipynb` | Run 11. Compare the winning TAPT + one-layer + R-Drop recipe with the same recipe augmented by fold-safe topic, mood and address tags. Reports matched OOF macro-F1 and preserves both probability matrices. | ~3.5--4 h | yes, matched 5-fold OOF |

A violent word's meaning depends on its company. Among the 420 training comments
containing one: with media context present, Others is 0.27 and Violence 0.18; with no
target named, Violence is 0.34 and Others 0.06. `decode.py` fits one weight per class and
has lost its nested check every time, because the right correction for Violence flips
sign with context. `hastika.task_b.context_decode` fits a lift per cell instead, and on
the calibrated TF-IDF SVM that is worth +2.9 points nested, 0.5630 to 0.5823, with
Violence F1 going 0.24 to about 0.35. Whether it helps MuRIL is open: a transformer can
represent the interaction and a bag of n-grams cannot, so the model may already know it.

Run 10 also leaves behind `oof_probs.npy` for the current recipe, which nothing else in
the repo has. With that file any future decode idea can be tested in seconds rather than
in 108 minutes of GPU.

## Run this one

| notebook | what it does | time | score |
|---|---|---|---|
| `16_full_data_rdrop_augmented.ipynb` | Full-data one-layer R-Drop fit with multiclass-augmented binary data: merges `multiclass_train.csv` (3,159 gold rows) with `hate_only_compiled_multiclass.csv` (779 new rows), TAPT on combined corpus, five seeds on all 3,938 rows, and ZIP packaging. | ~3--5 h | CodaBench pending |
| `09_rdrop_one_layer.ipynb` | Full-data one-layer R-Drop fit: TAPT on all 6,406 comments, five seeds on all 3,159 labelled rows, averaged validation predictions, and ZIP packaging. | ~2.7 h | **0.6410 CodaBench** |
| `12_full_data_context_tags.ipynb` | Full-data context-tagged fit: TAPT on all 6,406 comments, five tagged models on all 3,159 labelled rows, averaged validation predictions, and ZIP packaging. The uploaded log shows training completed, but the final packaging assertion failed because the trainer log did not print the expected R-Drop string. | ~2--3 h | training complete; recover artifacts or rerun packaging |
| `14_full_data_tapt_taska_text.ipynb` | Full-data expanded-TAPT fit: TAPT on Task B text, OffensEval and permitted Task A training comments, then five Task B models on all 3,159 labelled rows and ZIP packaging. Task A labels are not read. | ~3--4 h | CodaBench pending |
| `15_full_data_muril_large.ipynb` | Full-data MuRIL-large fit: separate large TAPT checkpoint, memory-safe classifier settings, five seeds on all 3,159 rows, and ZIP packaging. | ~5--8 h | CodaBench pending |
| `08_full_data_fit_reinit1.ipynb` | Full-data one-layer fit: TAPT on all 6,406 comments, five seeds on all 3,159 labelled rows, inference on the 395 official validation inputs, and ZIP packaging. | ~2 h | **0.6299 CodaBench** |
| `04_full_data_fit.ipynb` | The current recipe, `D0_V0_noaux`, on 100% of the data. TAPT on all 6,406 comments with nothing held out, then five seeds on all 3,159 rows with no deduplication and no auxiliary head. Reads its own logs and stops if either count is short. | ~2 h | CodaBench only |

`D0_V0_noaux` means TAPT on Kannada text only (D0), MuRIL's tokenizer as shipped (V0),
and a plain six-way head (no aux). It led the data-processing grid. The current
provisional candidate adds `--reinit-layers 1` and `--rdrop 0.5`, based on the Run 5/6
OOF results and Run 9's CodaBench improvement.

Nothing is held out in either full-data notebook, so neither prints a local F1. Upload the
ZIP from the selected candidate to the Task B validation phase on CodaBench; the macro-F1
it returns is the run's score. Record it in `docs/EXPERIMENTS.md` and
`submissions/README.md`.

The `08_full_data_fit_reinit1.ipynb` candidate uses the same full-data protocol but sets
`--reinit-layers 1`, based on the observed 0.6102 OOF result. It scored **0.6299** on
CodaBench; its ZIP is `b_reinit1_full.zip`. Run 9 adds R-Drop and is now the preferred
candidate, with ZIP `b_reinit1_rdrop_full.zip` and CodaBench macro-F1 **0.6410**.

## Current ablation

| notebook | what it does | time | score |
|---|---|---|---|
| `05_reinit_one_layer.ipynb` | Run 5. Reuses one TAPT checkpoint and compares the established top-two-layer reset with a top-one-layer reset using identical five-fold settings. | ~3--3.5 h | 0.6013 vs **0.6102** |
| `06_reinit_confirmation.ipynb` | Run 6. Repeats both settings at model seeds 43 and 44, then compares individual and probability-averaged OOF scores. | ~6--7 h | one layer **0.614** vs two layers 0.612 (averaged OOF) |
| `07_no_reinit_ablation.ipynb` | Run 7. Compares the current one-layer setting with no encoder-layer reinitialization at seeds 43 and 44. | ~6--7 h | one layer **0.614** vs no layers 0.584 (averaged OOF) |

This is a local controlled experiment, not a submission notebook. It runs the TAPT pass
once, then trains `b_tapt_reinit2_5f` and `b_tapt_reinit1_5f` with seed 42, five folds,
six epochs and `--select last`. The completed run scored 0.6013 for two layers and
0.6102 for one layer; the one-layer variant improved macro-F1 by 0.0089 and was
particularly better on `Violence` and `Others`. The outputs belong under `artifacts/`;
download the logs and probability matrices from Kaggle for review.

Run 6 confirmed that the one-layer setting was stronger in the averaged-probability
comparison: 0.614 versus 0.612 for two-layer reinitialization. Run 7 then showed that
removing reinitialization entirely was substantially worse: 0.584 versus 0.614.

Run 9 is the current preferred Task B candidate: full-data TAPT plus one-layer
reinitialization and R-Drop (`--rdrop 0.5`) across five seeds. It scored **0.6410
macro-F1** and **0.6937 accuracy** on CodaBench, improving on Run 8's 0.6299 macro-F1
and 0.6886 accuracy. Further GPU budget should now prioritize Task A experiments.

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
