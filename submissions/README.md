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

## Format

CodaBench takes a zip containing one bare `predictions.csv` at the top level,
header `id,label`, one row per id in the matching `*_validation_inputs.csv`.
`src/hastika/common/submission.py` refuses to write unless all of that holds, so build
zips with it rather than by hand -- desktop zip tools wrap the file in a folder
and the scorer then cannot find it.
