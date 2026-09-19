# Kaggle notebooks — Task A

Upload one of these to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and
**Internet** on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch
itself, so the notebook file is all you upload. Never run these interactively: the
session dies with the browser tab and they all run for hours.

Notebook numbers are the order they were written, not the Run numbers in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md). The mapping is in the tables below.

## Queued, not yet run

| notebook | run | what it answers | time | local score? |
|---|---|---|---|---|
| `05_reinit_ensemble_weight.ipynb` | 11 | does one-layer reinitialization help Task A, and what is the right blend weight now? Five-fold OOF for both settings, nested weight and threshold search, then a full-data fit of the winner. | ~8.2 h | yes, 5-fold OOF |
| `04_rdrop_full_data.ipynb` | 5 | does R-Drop improve the recipe? Two arms on all 6,401 rows, five seeds each, one flag apart. | ~6.5 h | no |
| `01_tapt_demojized_muril.ipynb` | 7 | does Task-A-domain TAPT help before fine-tuning? Control against TAPT on the fixed 85/15 split. | ~2--4 h | yes, 15% holdout |
| `03_muril_embeddings_svm.ipynb` | 10 | can an RBF SVM on frozen MuRIL embeddings beat fine-tuning? | — | yes, 85/15 holdout |

### Run 11 in detail

Run 9's blend weight of 57/43 was fitted in Run 8 against a **two-layer** MuRIL. A better
MuRIL component should earn more weight, so this notebook refits it rather than assuming
it. Reinitialization itself cannot interfere with the ensemble: it touches only MuRIL's
top layers before fine-tuning and the TF-IDF/SVM is untouched. What it changes is the
optimal mixing weight, which depends on the relative strength of the two components.

Stage 1 runs five-fold OOF for the SVM and for MuRIL at one and two reinitialized layers,
all on split seed 42 so the matrices line up row for row. Stage 2 compares the two
reinitialization settings on 6,401 out-of-fold rows, where noise is about 0.6 points --
Task B could never resolve a gap this small. Stage 3 sweeps the blend weight and the
decision threshold together and reports a **nested** estimate, with both chosen on an
inner split of each fold's training rows. Stage 4 fits the winner on all rows with five
seeds and applies the stage-3 values unchanged.

The threshold is swept because it is free and it works: on the TF-IDF floor, 0.5 scores
`0.8073` and a nested threshold scores `0.8108`. `hastika.models.muril` already does this
for a single model, but the ensemble notebook hard-codes `> 0.5`.

The budget guard runs the one-layer arm first, so a short session still answers the weight
question and still produces a submission.

### Run 5 in detail

Both arms are identical except `--rdrop`: demojized MuRIL, `--folds 1` so every seed
trains on all 6,401 deduplicated rows, two-layer reinitialization, 6 epochs, effective
batch 16, `--select last`, seeds 42 through 46 averaged.

A matched control is trained rather than reusing Run 9's `0.8187`, because that
submission's MuRIL component used a **single seed**. Comparing a five-seed R-Drop arm
against it would change two things at once, and 806 validation rows cannot separate them.

Four ZIPs come out: each MuRIL arm alone, and each blended with a full-fit TF-IDF/SVM at
Run 8's fixed 57/43 weights so the result is comparable with the current best. Those
weights are carried over unchanged, because a final fit has no OOF pass to refit them on
honestly and the hidden labels must not influence them.

**Upload `task_a_control_blend.zip` first.** Its score against `0.8187` shows how far the
806-row set moves on a near-rerun, and only then is the control-versus-R-Drop gap
readable. Record every score, including the losing arm.

Neither arm has a local macro-F1 and neither can have one: every labelled row is in
training. `experiments/task_a/run_rdrop.sh` is the five-fold version of the same
comparison and does produce an OOF score, at roughly five times the cost.

## Completed

| notebook | run | what it did | result |
|---|---|---|---|
| `02_muril_tfidf_ensemble.ipynb` | 8 | OOF blend of demojized MuRIL and TF-IDF/SVM, weights nested-checked | 0.7890, rejected |
| `02_muril_tfidf_ensemble.ipynb` | 9 | the same two components, each fitted on all 6,401 rows, fixed 57/43 weights | **0.8187**, current best |

Runs 1 to 4 and 6 predate the notebooks and were run from the Task A training scripts.
Their numbers are in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

## Holdout runs decide, full fits build

A holdout or fold run keeps rows back so it can score itself; that is how a recipe gets
chosen. A full fit trains on every labelled row, so nothing is left to measure it with,
and the official validation file has no labels. Once the recipe is fixed, the full fit is
what gets submitted and CodaBench is where it gets scored. Do not read a full fit's
missing local score as a good sign, and do not copy a holdout number onto it.

Task A's validation set is 806 rows, roughly twice Task B's 395, so its scores are less
noisy — but a one-point gap is still about eight rows. Treat small differences carefully.

## Outputs

Everything lands in `/kaggle/working`. Download the ZIPs, the logs and any
`*_test_probs.npy` individually rather than using Download All, because a TAPT checkpoint
is about a gigabyte. Keep the probability matrices: they are the only way to rebuild a
blend later without retraining.
