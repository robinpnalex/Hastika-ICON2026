# Kaggle notebooks

Upload one of these to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and
**Internet** on, then **Save Version -> Save & Run All**. Each clones this branch
itself, so the notebook file is all you upload. Never run these interactively: the
session dies with the browser tab and they all run for hours.

| notebook | what it does | time | gives you a score? |
|---|---|---|---|
| `kaggle_tapt_full.ipynb` | `f_tapt` with TAPT on all 6,406 comments instead of 95% of them, then five seeds on all 3,143 rows. Checks both counts from the logs. | ~2 h | no |
| `kaggle_grid.ipynb` | Factorial over both data-processing levers, ten cells on the real holdout, then a five-seed full fit of the winner. | ~8.5 h | **yes**, every cell |
| `kaggle_fullfit_sweep.ipynb` | Six ideas, five seeds each, all 3,143 rows per model. | ~9.5 h | no |
| `kaggle_fullfit_single.ipynb` | One idea, all rows, one seed. The minimal full fit. | ~45 min | no |
| `kaggle_rebuild_winner.ipynb` | Rebuilds only `b_tapt_5f`, the submitted model. | ~2 h | yes, 5-fold OOF |
| `kaggle_task_b.ipynb` | The original overnight sweep: holdout ranking, then 5-fold on the winners. | ~8.5 h | yes |

## Which to run

Use `kaggle_grid.ipynb`. It is the only one that answers which data processing helps,
because holdout runs hold rows back and full fits cannot.

A full fit trains on every labelled row, so nothing is left to measure it with, and the
official validation file has no labels. Full fits are for *building* a submission once
you already know what to build. Holdout runs are for *deciding*. Do not read a full fit's
absence of a score as a good sign.

## Outputs

Everything lands in `/kaggle/working`. Download `RESULTS.md` and the zips in `subs/`
individually rather than using Download All, because the adaptation checkpoints in there
are about a gigabyte each.
