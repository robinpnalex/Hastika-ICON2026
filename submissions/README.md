# Submissions

One directory per CodaBench submission, each holding the exact zip that was
uploaded plus its `predictions.csv` unpacked so the contents are diffable in
git. The zip is the artifact of record; the CSV is there to be read.

## Task B

| rank | arm | macro-F1 | measured on | submitted |
|---|---|---|---|---|
| 1 | `b_tapt_5f` | **0.6013** | 5-fold OOF, 3,143 rows | yes |
| 2 | `ensemble_b` | 0.6002 | nested estimate | not yet |
| 3 | `b_tapt` | 0.5972 | 15% holdout, 472 rows | not yet |

Scores are the **unbiased** number, meaning the `last` checkpoint rather than
the `best` one. `best` picks each fold's checkpoint by macro-F1 on the rows it
then reports, so it flatters every run. `work/muril_b.py` always prints both.

Reference points: the TF-IDF + LinearSVC floor scores 0.5948 on the same folds,
and stock MuRIL scores 0.5727 over five folds. Only the domain-adapted run beats
the floor.

## Task A

`task_a_predictions.zip` at the repo root, kept where it has always been.

## Format

CodaBench takes a zip containing one bare `predictions.csv` at the top level,
header `id,label`, one row per id in the matching `*_validation_inputs.csv`.
`work/make_submission.py` refuses to write unless all of that holds, so build
zips with it rather than by hand -- desktop zip tools wrap the file in a folder
and the scorer then cannot find it.
