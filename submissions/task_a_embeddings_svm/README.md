# task_a_embeddings_svm

Task A Run 10: an RBF SVM on **frozen** MuRIL embeddings. No fine-tuning of the
encoder at all — MuRIL is used as a feature extractor and only the SVM is trained.

| | |
|---|---|
| local holdout macro-F1 | **not recorded** — see below |
| CodaBench macro-F1 | **pending** |
| encoder | `google/muril-base-cased`, frozen, meanmax pooling |
| classifier | `SVC(kernel="rbf", C=2.0, class_weight="balanced")` |
| fitted on | all 6,401 deduplicated rows, after the holdout measurement |
| produced by | `notebooks/task_a/03_muril_embeddings_svm.ipynb`, 2026-09-19 |

## Validation of the artifact

Checked on 2026-09-19 against `data/raw/binary_validation_inputs.csv`:

* 806 rows, one per released id, no duplicates
* id set matches the validation inputs exactly
* header is `id,label`; both labels are from the allowed set
* the ZIP contains one bare `predictions.csv` at its top level

## Predicted distribution

| | this run | training prior | previous Task A submission |
|---|---|---|---|
| Hate | 55.2% | 49.1% | 49.3% |
| Non-Hate | 44.8% | 50.9% | 50.7% |

It leans about six points more toward `Hate` than either the training prior or the
previous submission. Not a collapse, but worth watching: on a balanced task, a skewed
prior costs macro-F1 on whichever class it under-calls.

## Why it is interesting even if its score is lower

It agrees with the preserved Task A submission on only **77.2%** of the 806 rows. That is
a lot of disagreement for two systems on the same task, and it is exactly the property
that makes a useful ensemble member. Blending helps when members fail differently, not
when the added member is strong. A frozen encoder with an RBF head has a genuinely
different inductive bias from a fine-tuned transformer, and this number says so.

If its own CodaBench score is anywhere near the current best, it belongs in the blend
search in `notebooks/task_a/08_third_member_blend.ipynb` alongside XLM-R.

## What is missing

The notebook measures the method on the repository's fixed 85/15 holdout before refitting
on all rows, and writes `config.json`, `holdout_idx.npy` and `holdout_decision.npy` into
`task_a_muril_embeddings_svm_outputs` on Kaggle. Only the ZIP was downloaded, so the
**holdout macro-F1 is not recorded here**. Retrieve it from the notebook output or its
log and add it to this file and to `docs/EXPERIMENTS.md`; without it there is no local
number to compare against the 0.8073 TF-IDF floor.

## Reproduce

Upload `notebooks/task_a/03_muril_embeddings_svm.ipynb` to Kaggle with GPU and Internet
on. About 20 to 45 minutes.
