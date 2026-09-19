# Submissions

One directory per CodaBench submission, each holding the exact zip that was
uploaded plus its `predictions.csv` unpacked so the contents are diffable in
git. The zip is the artifact of record; the CSV is there to be read.

## Task B

| submission | recipe | local macro-F1 | CodaBench validation | in this folder |
|---|---|---|---|---|
| `f_tapt` | TAPT MuRIL, 5 seeds on all 3,143 deduplicated rows | none, full fit | **0.6007**¹ | no |
| `b_tapt_5f` | TAPT MuRIL, five fold models averaged | 0.6013, 5-fold OOF | 0.5922 | yes |
| `d0v0_noaux_full` | TAPT on every comment, 5 seeds on all 3,159 rows | none, full fit | not run yet | no |
| `b_reinit1_full` | TAPT MuRIL, one-layer reinit, 5 seeds on all 3,159 rows | none, full fit | **0.6299** | no (Kaggle output) |
| `b_reinit1_rdrop_full` | TAPT MuRIL, one-layer reinit + R-Drop 0.5, 5 seeds on all 3,159 rows | none, full fit | **0.6410** | no (Kaggle output) |

¹ Inferred, not confirmed: the `scoring_result.zip` reading 0.6007 was downloaded a few
minutes after `f_tapt.zip`. Check the CodaBench submission list, then add
`f_tapt/` here with its zip.

All three are the `D0_V0_noaux` recipe. The local number is the unbiased `last`
checkpoint, never `best`, which picks each fold's checkpoint on the rows it then
reports. A full fit trains on every row and has no local number; its only score
is CodaBench's.

395 validation rows give macro-F1 a standard deviation of about three points, so
the gap between 0.6007 and 0.5922 does not show that one recipe is better.

Reference points: the TF-IDF + LinearSVC floor scores 0.5948 on the same folds,
and stock MuRIL scores 0.5727 over five folds.

When a new submission is scored, add a directory named after it holding the
uploaded zip, its `predictions.csv` and `test_probs.npy`, plus a README with the
exact commands and the CodaBench score, and add its row above.

## Task A

`submissions/task_a/task_a_predictions.zip` is the preserved Task A upload.

| rank | arm | macro-F1 | measured on | submitted |
|---|---|---|---|---|
| 1 | one-layer reinit + refitted blend, Run 11 | **0.8188** | CodaBench, 806 rows | yes |
| 1= | full-data MuRIL + TF-IDF ensemble, Run 9 | **0.8187** | CodaBench, 806 rows | yes |
| 2 | MuRIL, flags unrecorded | 0.8163 | CodaBench | yes |
| 3 | demojized TF-IDF + LinearSVC | 0.8103 | 5-fold CV | — |
| — | `task_a_embeddings_svm` | pending | — | built, not yet scored |

Runs 9 and 11 are tied: `0.8187` and `0.8188` macro-F1, both `0.8189` accuracy. The gap is
under one row of 806, so either can serve as the base recipe. Run 11 changed three things
at once — one reinitialized layer, a refitted blend weight, a fitted threshold — and
together they moved nothing. Its ZIP and unpacked CSV were uploaded from external Kaggle output
and are not currently stored in this repository.

`task_a_embeddings_svm/` holds Run 10, an RBF SVM on frozen MuRIL embeddings. The artifact
is validated and preserved, but its holdout macro-F1 was not captured and it has not been
scored on CodaBench. It agrees with the older preserved Task A submission on only 77.2% of
rows, which makes it a candidate ensemble member regardless of its own score.

## Format

CodaBench takes a zip containing one bare `predictions.csv` at the top level,
header `id,label`, one row per id in the matching `*_validation_inputs.csv`.
`src/hastika/common/submission.py` refuses to write unless all of that holds, so build
zips with it rather than by hand -- desktop zip tools wrap the file in a folder
and the scorer then cannot find it.
