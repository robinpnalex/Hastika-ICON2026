# Submissions

Layout: `submissions/task_a/<name>/` and `submissions/task_b/<name>/`. Each directory holds
the exact ZIP plus its `predictions.csv` unpacked, so the contents are diffable in git. The
ZIP is the artifact of record; the CSV is there to be read.

## The final submission (from 2026-10-05)

There is no more CodaBench scoring. **One final submission is made for Task A and Task B
together**, from models fine-tuned on all labelled rows: train plus the released labelled
validation. Every choice behind it is made on the fixed 15% holdouts:

| task | holdout rows | fingerprint |
|---|---|---|
| A | 1,079 | `815110ff24` |
| B | 530 | `f85f4f049b` |

The notebooks that produce it are listed in [`../docs/FINAL.md`](../docs/FINAL.md). When
the final ZIPs exist, store them as `task_a/final/` and `task_b/final/`. Each gets the ZIP,
its `predictions.csv` and a README naming the notebook, the commit and the holdout evidence.

The current candidates, both in Kaggle outputs and not yet copied into this folder:

| task | ZIP | what it is | holdout |
|---|---|---|---|
| B | `RECOMMENDED_b22_gemma_3ep_2seeds.zip` (Run 22) | Gemma-4-12B, 3 epochs, 2 seeds, all 3,532 rows | 0.6792 |
| A | **submitted:** `task_a/final/Project MANAS_taskA.csv`, from `RECOMMENDED_a28_gemma.zip` (Task A Run 28) | Gemma-4-12B, 3 epochs, 2 seeds, all 7,193 rows | 0.8563 |

The tables below are the CodaBench history of the MuRIL era, kept as a record.

## Task B

| submission | recipe | local macro-F1 | CodaBench validation | in this folder |
|---|---|---|---|---|
| `f_tapt` | TAPT MuRIL, 5 seeds on all 3,143 deduplicated rows | none, full fit | **0.6007**¹ | no |
| `b_tapt_5f` | TAPT MuRIL, five fold models averaged | 0.6013, 5-fold OOF | 0.5922 | yes |
| `d0v0_noaux_full` | TAPT on every comment, 5 seeds on all 3,159 rows | none, full fit | not run yet | no |
| `b_reinit1_full` | TAPT MuRIL, one-layer reinit, 5 seeds on all 3,159 rows | none, full fit | **0.6299** | no (Kaggle output) |
| `b_reinit1_rdrop_full` | TAPT MuRIL, one-layer reinit + R-Drop 0.5, 5 seeds on all 3,159 rows | none, full fit | **0.6410** | no (Kaggle output) |
| `run12_context_tags_full` | TAPT MuRIL with context tags, five full-data fits | none, full fit | pending | recovered under `submissions/task_b/run12_context_tags_full/` |
| `run14_tapt_taska_full` | Task A comments added as unlabelled TAPT text | none, full fit | pending | recovered under `submissions/task_b/run14_tapt_taska_full/` |

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

Directories: `task_b/b_tapt_5f/`, `task_b/fast_svm/`, `task_b/fast_svm_calibrated/`,
`task_b/run12_context_tags_full/` and `task_b/run14_tapt_taska_full/`.

## Task A

`task_a/task_a_predictions.zip` is the preserved Task A upload. `task_a/task_a_predictions_2026-09-10_main.zip` is a
second, different early MuRIL-era Task A upload. It was preserved from `main`'s commit
`fe18b8a` when `main` and `task-b` were reconciled on 2026-10-06: 806 rows, 462 Hate,
91.9% agreement with `task_a_predictions.zip`. `task_a/fast_svm/` and
`task_a/embeddings_svm/` hold the SVM baselines.

| rank | arm | macro-F1 | measured on | submitted |
|---|---|---|---|---|
| 1 | one-layer reinit + refitted blend, Run 11 | **0.8188** | CodaBench, 806 rows | yes |
| 1= | full-data MuRIL + TF-IDF ensemble, Run 9 | **0.8187** | CodaBench, 806 rows | yes |
| 2 | MuRIL, flags unrecorded | 0.8163 | CodaBench | yes |
| 3 | demojized TF-IDF + LinearSVC | 0.8103 | 5-fold CV | — |
| — | `task_a/embeddings_svm` | pending | — | built, not yet scored |

Runs 9 and 11 are tied: `0.8187` and `0.8188` macro-F1, both `0.8189` accuracy. The gap is
under one row of 806, so either can serve as the base recipe. Run 11 changed three things
at once — one reinitialized layer, a refitted blend weight, a fitted threshold — and
together they moved nothing. Its ZIP and unpacked CSV were uploaded from external Kaggle output
and are not currently stored in this repository.

`task_a/embeddings_svm/` holds Run 10, an RBF SVM on frozen MuRIL embeddings. The artifact
is validated and preserved, but its holdout macro-F1 was not captured and it has not been
scored on CodaBench. It agrees with the older preserved Task A submission on only 77.2% of
rows, which makes it a candidate ensemble member regardless of its own score.

## Format

CodaBench takes a zip containing one bare `predictions.csv` at the top level,
header `id,label`, one row per id in the matching `*_validation_inputs.csv`.
`src/hastika/common/submission.py` refuses to write unless all of that holds, so build
zips with it rather than by hand -- desktop zip tools wrap the file in a folder
and the scorer then cannot find it.
