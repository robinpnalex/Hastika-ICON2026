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
| `11_reinit1_full_data.ipynb` | 11 | `--reinit-layers 1`, **and the blend weight refitted to match** | ~3.4 h |
| `05_rdrop_full_data.ipynb` | 5 | `--rdrop 0.5`, against a matched control, 5 seeds each | ~6.5 h |

**Run 11 first.** It is 40 minutes and it decides the base for everything after it. If one
reinitialized layer beats `0.8187`, the queued experiments stack on one layer; if it loses
by more than about 3 points, they keep two.

### Run 11 in detail — run this first

Run 9's MuRIL reinitializes the top **two** encoder layers, a default inherited rather than
chosen, and its 57/43 blend weights were fitted in Run 8 **against that two-layer
component**. Change the component and the weight it deserves changes with it: a stronger
MuRIL should take more of the blend. So Run 11 does both, one layer and a refitted weight.

Refitting needs out-of-fold probabilities, so stage 1 runs five folds for both components,
stage 2 fits the weight and threshold with a nested check, and stage 3 refits both on all
6,401 rows and applies those values. About 3.4 hours, not the eight a two-way comparison
would cost, because folds are run only for the configuration being shipped.

Three things come out besides the submission: MuRIL's own out-of-fold score against the
`0.8073` floor, a fitted threshold worth a measured `+0.0035`, and **`oof_probs.npy` for
both components — which the repository has never had for Task A**. With those stored, every
later blend weight, threshold or decode idea costs seconds of CPU instead of a GPU session.

Its five-fold MuRIL arm is the same configuration as the funnel's `control`, so running it
first also supplies the funnel's reference point.

### Run 5 in detail

Two arms, both `--folds 1` with five seeds averaged, differing only by `--rdrop`. A matched
control is trained rather than comparing against `0.8187` directly, because Run 9's MuRIL
used a **single seed** — a five-seed R-Drop arm against it would change two things at once.
Four ZIPs: each arm alone and each blended with the SVM. Upload the control blend first.

## Queued — the funnel, which supersedes Runs 12 to 15

| notebook | run | what it does | time |
|---|---|---|---|
| `16_funnel.ipynb` | 16 | screens seven component-level arms on the cheap split, promotes the leaders to five folds | ~6 h stage 1 |

**Run this instead of Runs 12 to 15 individually.** Those test one factor each against
their own control, cost about 30 hours in total, and never put the factors on one table.
The funnel screens all seven at ~33 min each, then spends five-fold time only on the
leaders. Task B ran this exact funnel and recorded that the holdout ranked all four
promoted arms in the same order five folds did, reading about two points high — so stage 1
orders arms and never reports a number.

Arms: `control` (one layer, 6 epochs), `reinit2`, `tapt`, `ext_mix`, `ext_stage`,
`epochs10`, `large`. Every one changes the MuRIL component only; the SVM half of the blend
is fixed because nothing queued touches it. Each arm's predicted class balance is checked
against Task A's 0.491 prior, and a collapsed arm is not promoted.

Runs 12 to 15 are kept as standalone notebooks for anyone who wants one factor in
isolation, and Run 7 remains the leak-free TAPT comparison on the fixed holdout.

| notebook | run | kept for | time |
|---|---|---|---|
| `12_tapt_oof.ipynb` | 12 | TAPT alone, five-fold | ~6.3 h |
| `13_external_labels.ipynb` | 13 | external labels alone, three arms | ~9.9 h |
| `14_third_member_blend.ipynb` | 14 | XLM-R as a third member; downgraded after Run 10 | ~6 h |
| `15_capacity_and_schedule.ipynb` | 15 | MuRIL-large and 10 epochs alone | ~8 h |
| `07_tapt_holdout.ipynb` | 7 | the leak-free TAPT comparison | ~2--4 h |

**Run 12**, if run standalone, tests the largest measured effect in the project: TAPT was worth +2.9 on Task B.
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
