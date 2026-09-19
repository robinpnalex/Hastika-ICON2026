# Kaggle notebooks — Task A

**The notebook number is the Run number.** `11_reinit1_full_data.ipynb` is Run 11. Task B
uses the same convention. Two runs share one notebook only where one notebook produced
both: `08_muril_tfidf_ensemble.ipynb` produced Runs 8 and 9.

Upload one to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and **Internet**
on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch itself, so the
notebook file is all you upload. Never run these interactively: the session dies with the
browser tab and most of them run for hours.

## Completed

| notebook | run | what it did | result |
|---|---|---|---|
| `08_muril_tfidf_ensemble.ipynb` | 8 | OOF blend of demojized MuRIL and TF-IDF/SVM, weights nested-checked | 0.7890, rejected |
| `08_muril_tfidf_ensemble.ipynb` | 9 | the same two components each fitted on all 6,401 rows, fixed 57/43 weights | **0.8187 — current best** |
| `10_frozen_embeddings_svm.ipynb` | 10 | RBF SVM on frozen MuRIL embeddings, no fine-tuning | 0.71, rejected; also rejected as a blend member |

Runs 1 to 4 and 6 predate the notebooks and were run from the Task A training scripts.
Their numbers are in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

Run 10's artifact is preserved at `submissions/task_a_embeddings_svm/`. It is rejected as a
blend member too: it disagrees with the best submission on 184 of 806 rows and is right on
only 24% of them, so no weight gains. The derivation is in that directory's README.

## Queued — full data, one change on the current best

These follow the standing constraint: stacked on the `0.8187` recipe, `--folds 1` on all
6,401 rows, no holdout, CodaBench as the only readout. Each writes a submission ZIP.

| notebook | run | the one change | time |
|---|---|---|---|
| `11_reinit1_full_data.ipynb` | 11 | `--reinit-layers 1` instead of 2 | ~40 min |
| `05_rdrop_full_data.ipynb` | 5 | `--rdrop 0.5`, against a matched control, 5 seeds each | ~6.5 h |

**Run 11 first.** It is 40 minutes and it decides the base for everything after it. If one
reinitialized layer beats `0.8187`, the queued experiments stack on one layer; if it loses
by more than about 3 points, they keep two.

### Run 11 in detail

Run 9's MuRIL component reinitializes the top **two** encoder layers, a default inherited
rather than chosen. Task B measured one layer against none at **+3.0** and one against two
at +0.5. Task A has tested neither.

Everything else is held fixed, including the 57/43 blend weights and the 0.5 threshold.
Those weights were fitted in Run 8 against a two-layer MuRIL so they are arguably no longer
optimal, but refitting them would be a second change and there is no holdout to refit them
on honestly. One change at a time is what makes a CodaBench gap attributable.

### Run 5 in detail

Two arms, both `--folds 1` with five seeds averaged, differing only by `--rdrop`. A matched
control is trained rather than comparing against `0.8187` directly, because Run 9's MuRIL
used a **single seed** — a five-seed R-Drop arm against it would change two things at once.
Four ZIPs: each arm alone and each blended with the SVM. Upload the control blend first.

## Queued — measurement, five-fold

These hold rows back so they can score themselves, which is why they exist despite the
no-holdout rule: they decide *what* to build before a submission slot is spent. None writes
a submission.

| notebook | run | what it answers | time |
|---|---|---|---|
| `12_tapt_oof.ipynb` | 12 | does task-adaptive pretraining help Task A? | ~6.3 h |
| `13_external_labels.ipynb` | 13 | can the external corpus's **labels** be trained on? | ~9.9 h |
| `14_third_member_blend.ipynb` | 14 | does XLM-R add anything as a third ensemble member? | ~6 h |
| `15_capacity_and_schedule.ipynb` | 15 | is the recipe underfitting? MuRIL-large and 10 epochs | ~8 h |
| `07_tapt_holdout.ipynb` | 7 | the leak-free version of Run 12, on the fixed 85/15 split | ~2--4 h |

**Run 12** tests the largest measured effect in the project: TAPT was worth +2.9 on Task B.
It states its own bias plainly — the MLM stage reads every Task A training comment, so the
TAPT arm has seen the wording of its own out-of-fold rows and the control has not. Run 7 is
the leak-free version on a 960-row holdout; read them together.

**Run 13** is the one opening Task A has that Task B did not: the external corpus carries
`Hate`/`Non-Hate`, exactly Task A's label space, where Task B's six-way taxonomy made those
labels unusable. It uses external **labels**, which is a rules question to settle before
submitting anything built on it.

**Run 14** is downgraded after Run 10. A frozen MuRIL disagreed with the best submission on
23% of rows and still added nothing, which lowers the prior on member diversity as a lever.

**Run 15** is motivated by the existing Task A logs: averaged over 14 fold-runs, validation
macro-F1 is still rising at epoch 6, gaining +0.0038 in the final epoch. MuRIL-large runs
behind a one-fold collapse check, because mDeBERTa collapsed outright on Task B at a
learning rate that suited the base model.

## Holdout runs decide, full fits build

A holdout or fold run keeps rows back so it can score itself; that is how a recipe gets
chosen. A full fit trains on every labelled row, so nothing is left to measure it with, and
the official validation file has no labels. Once the recipe is fixed, the full fit is what
gets submitted and CodaBench is where it gets scored. Do not read a full fit's missing local
score as a good sign, and do not copy a holdout number onto it.

Task A's validation set is 806 rows, roughly twice Task B's 395, so its scores are less
noisy — but one score still carries about 1.5 points of standard deviation, and a one-point
gap is about eight rows. Treat small differences carefully.

## Outputs

Everything lands in `/kaggle/working`. Download the ZIPs, the logs and any `*_probs.npy`
individually rather than using Download All, because a TAPT checkpoint is about a gigabyte.
Keep the probability matrices: they are the only way to rebuild a blend or retune a
threshold later without retraining.
