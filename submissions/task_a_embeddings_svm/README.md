# task_a_embeddings_svm

Task A Run 10: an RBF SVM on **frozen** MuRIL embeddings. No fine-tuning of the
encoder at all — MuRIL is used as a feature extractor and only the SVM is trained.

| | |
|---|---|
| CodaBench macro-F1 | **0.71**, accuracy **0.70** — rejected |
| local holdout macro-F1 | not recorded; see below |
| encoder | `google/muril-base-cased`, frozen, meanmax pooling |
| classifier | `SVC(kernel="rbf", C=2.0, class_weight="balanced")` |
| fitted on | all 6,401 deduplicated rows, after the holdout measurement |
| produced by | `notebooks/task_a/10_frozen_embeddings_svm.ipynb`, 2026-09-19 |

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

## Rejected, and the arithmetic that rejects it

At `0.71` it is 11 points below the current best (`0.8187`) and 10 below the TF-IDF floor
(`0.8103`). Freezing the encoder costs far more than the RBF head recovers.

The open question was whether it could still earn a place in the ensemble, since it
disagrees with the best submission on 22.8% of rows and blending pays when members fail
differently. Two CodaBench scores plus the two prediction files answer that **without
needing the hidden labels**. On a binary task, when two systems disagree exactly one of
them matches gold, so on the disagreement rows the two hit-rates must sum to 1:

    accuracy_best = agree x c + disagree x p_best
    accuracy_this = agree x c + disagree x p_this        with p_best + p_this = 1

Solving with agreement 0.772, accuracy 0.8189 and 0.70:

| quantity | value |
|---|---|
| rows where they disagree | 184 of 806 |
| on those rows, best submission is right | 76.0% |
| on those rows, this model is right | **24.0%** |
| on rows where they agree, both right | 83.6% |

A blend can only help by overruling the stronger member somewhere, and here it would be
wrong three times out of four when it did. There is no weight that gains. **Do not add it
to the blend search in `notebooks/task_a/14_third_member_blend.ipynb`.**

One by-product is worth keeping: on the 622 rows where the two agree, they are still wrong
16.4% of the time. Two systems with very different inductive biases failing together on
the same rows is consistent with the Task A error analysis, which found 92% of errors in
comments carrying no profanity at all. That shared residue, not member diversity, is where
the remaining points are.

## What is missing

The notebook measures the method on the repository's fixed 85/15 holdout before refitting
on all rows, and writes `config.json`, `holdout_idx.npy` and `holdout_decision.npy` into
`task_a_muril_embeddings_svm_outputs` on Kaggle. Only the ZIP was downloaded, so the
**holdout macro-F1 is not recorded here**. Retrieving it is now optional: the CodaBench
score already settles the method, and a holdout number would only refine by how much.

## Reproduce

Upload `notebooks/task_a/10_frozen_embeddings_svm.ipynb` to Kaggle with GPU and Internet
on. About 20 to 45 minutes.
