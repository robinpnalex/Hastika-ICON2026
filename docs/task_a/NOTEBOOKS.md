# Kaggle notebooks — Task A

**The notebook number is the Run number.** `11_reinit1_full_data.ipynb` is Run 11. Task B
uses the same convention. Two runs share one notebook only where one notebook produced
both: `08_muril_tfidf_ensemble.ipynb` produced Runs 8 and 9.

Upload one to Kaggle, set **Accelerator** to `GPU T4 x2` or `GPU P100` and **Internet**
on, then **Save Version -> Save & Run All**. Each clones the `task-b` branch itself, so the
notebook file is all you upload. Never run these interactively: the session dies with the
browser tab and most of them run for hours.

## Master index

This is the authoritative status view. `OOF` and holdout notebooks produce a local score;
`full fit` notebooks train on every labelled row and need CodaBench for evaluation. A
notebook marked **no ZIP** is diagnostic-only.

| notebook | status | purpose/result |
|---|---|---|
| `05_rdrop_full_data.ipynb` | ready; not run | full-data R-Drop versus matched control; ZIPs, no local score |
| `07_tapt_holdout.ipynb` | trained; reporting assertion failed | leak-free TAPT holdout comparison; score not recorded |
| `08_muril_tfidf_ensemble.ipynb` | completed | Run 8 OOF blend rejected at `0.7890`; Run 9 full fit scored `0.8187` CodaBench |
| `10_frozen_embeddings_svm.ipynb` | completed; rejected | frozen MuRIL embeddings + RBF SVM, about `0.71` macro-F1 |
| `11_reinit1_full_data.ipynb` | completed | one-layer MuRIL blend scored `0.8188` CodaBench |
| `12_tapt_oof.ipynb` | completed | TAPT MuRIL scored `0.8128` five-fold OOF |
| `13_external_labels.ipynb` | ready; not run | external labelled-data ablation; rules-sensitive |
| `14_third_member_blend.ipynb` | ready; not run | XLM-R third-member blend; lower priority after Run 10 |
| `15_capacity_and_schedule.ipynb` | ready; not run | MuRIL-large and ten-epoch screen; superseded by Run 16 stage 1 |
| `16_funnel.ipynb` | stage 1 completed | epochs10, MuRIL-large and reinit2 ranked highest on the screen |
| `17_final_submission.ipynb` | ready; not run | full-data TAPT/10-epoch/reinit2 candidate; CodaBench pending |
| `18_rdrop_short.ipynb` | ready; not run | short Task A R-Drop control comparison |
| `19_optimized_submission.ipynb` | ready; not run | full-data TAPT + 10 epochs + reinit2 + R-Drop candidate |
| `20_translate_to_english.ipynb` | setup fixed; rerun pending | translation pipeline; first attempt failed during dependency installation |
| `21_train_validation_test.ipynb` | setup failed | released-test final fit; clone lacked the labelled validation CSV |
| `22_run9_oof_evaluation.ipynb` | completed; no ZIP | Run 9 combined-corpus OOF: `0.8213` / `0.8215` |
| `23_oof_compare_run11_run9.ipynb` | completed; no ZIP | same-fold comparison: Run 11 `0.8204`, Run 9 `0.8213` |
| `24_tapt_epochs10_reinit2_oof.ipynb` | ready; no ZIP | TAPT + 10 epochs + two-layer reinitialization |
| `25_char_svm_third_member_oof.ipynb` | ready; no ZIP | character-only SVM as a third ensemble member |
| `26_transductive_derivable_labels_oof.ipynb` | ready; no ZIP | certain cross-task-derived labels, with honest/non-derived diagnostics |

The detailed sections below retain the reasoning and commands for each group; this table
is the authoritative run-status summary.

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

## Queued — evaluation-only next experiments

These three notebooks deliberately produce five-fold out-of-fold probabilities and a
nested diagnostic instead of a submission ZIP. They use the original labels plus the
released labelled validation rows where stated, so their scores are local rankings and
not CodaBench results.

| notebook | run | question | time |
|---|---|---|---|
| `24_tapt_epochs10_reinit2_oof.ipynb` | 24 | does the strongest combination—TAPT, 10 epochs and two reinitialized layers—beat the current recipe? | ~6–10 h |
| `25_char_svm_third_member_oof.ipynb` | 25 | does character-only TF-IDF add complementary errors to word+char SVM plus MuRIL? | ~3–5 h |
| `26_transductive_derivable_labels_oof.ipynb` | 26 | do the 365 certain labels derivable from Task B overlap improve Task A? | ~3–4 h |

Run 24 uses text-only TAPT over the staged combined corpus and the permitted external
Kannada text. Run 25 trains all three members on identical folds and lets `ensemble.py`
fit the weights, with a nested estimate to expose weight-search optimism. Run 26 excludes
uncertain labels and hidden-test rows; its derived validation slice is not an honest score
because those labels enter training, so the non-derived slice is the primary readout. Any
submission built from Run 26 must disclose the transductive labels.

## Full-data candidates and pending runs

These are stacked on the Task A submission recipes and use `--folds 1` on all 6,401 rows.
They write submission ZIPs and have no honest local F1. Run 11 is already completed;
Runs 5 and 20 remain pending.

| notebook | run | the one change | time |
|---|---|---|---|
| `11_reinit1_full_data.ipynb` | 11 | `--reinit-layers 1`, **and the blend weight refitted to match** | ~3.4 h |
| `05_rdrop_full_data.ipynb` | 5 | `--rdrop 0.5`, against a matched control, 5 seeds each | ~6.5 h |
| `20_translate_to_english.ipynb` | 20 | spelling-robust slur glossing, best-of-two translator (indic-translate / IndicTrans2 1B), fine-tune hateBERT; needs T4 x2 | ~3 h |

Run 20's first Kaggle attempt failed during dependency installation: current pip rejected
the legacy `omegaconf` metadata pulled by IndicXlit/Fairseq. The notebook now pins pip
24.0 before installing `ai4bharat-transliteration`; the experiment is fixed and pending a
rerun. The failed setup log is retained under `results/task_a/logs/`.

Run 11 is historical rather than queued. Its `0.8188` CodaBench result is effectively
tied with Run 9's `0.8187`; the newer OOF comparison in Run 23 slightly favors the
two-layer Run 9 recipe.

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

Its five-fold MuRIL arm is tagged **`f_control`**, the same tag and configuration the
funnel uses for its own control. Whichever notebook runs second finds the file and skips
the 160 minutes, so the control is trained once across both. Run 11 first and the funnel
inherits its reference point free.

### Run 5 in detail

Two arms, both `--folds 1` with five seeds averaged, differing only by `--rdrop`. A matched
control is trained rather than comparing against `0.8187` directly, because Run 9's MuRIL
used a **single seed** — a five-seed R-Drop arm against it would change two things at once.
Four ZIPs: each arm alone and each blended with the SVM. Upload the control blend first.

## Queued — the funnel, which supersedes Runs 12 to 15

| notebook | run | what it does | time |
|---|---|---|---|
| `16_funnel.ipynb` | 16 | screens six component-level arms on the cheap split, promotes the leaders to five folds | ~5.2 h stage 1 |

**Run this instead of Runs 12 to 15 individually.** Those test one factor each against
their own control, cost about 30 hours in total, and never put the factors on one table.
The funnel screens all seven at ~33 min each, then spends five-fold time only on the
leaders. Task B ran this exact funnel and recorded that the holdout ranked all four
promoted arms in the same order five folds did, reading about two points high — so stage 1
orders arms and never reports a number.

Arms as shipped: `control` (one layer, 6 epochs), `reinit2`, `ext_mix`, `ext_stage`,
`epochs10`, `large`. **`tapt` is deliberately excluded** because Run 12 answers it at five
folds, which beats a holdout screen; add it back with `--arms` if Run 12 was never run. Every one changes the MuRIL component only; the SVM half of the blend
is fixed because nothing queued touches it. Each arm's predicted class balance is checked
against Task A's 0.491 prior, and a collapsed arm is not promoted.

**Runs 12 to 15 each train their own control**, which is why they are wasteful once the
funnel exists: the funnel trains one control for all seven arms, and shares it with Run 11
on top of that. They are kept only for anyone wanting a single factor in isolation. Run 5
is superseded the same way — it is a 6.5 hour full-data pair testing what a 33 minute
funnel arm would answer. Run 7 remains the leak-free TAPT comparison and is worth running
only if `tapt` wins the funnel.

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

## Released test-data fit

`21_train_validation_test.ipynb` combines the original labelled training corpus with
the released labelled validation corpus, refits the Run 11 one-layer MuRIL + TF-IDF
blend, and packages predictions for `hastika_binary_test.csv`. It is a final fit, so
its output cannot be scored locally unless labels for that test file are released.

The first released-test attempt stopped during setup, before any training, because the
Kaggle clone did not contain `data/raw/hastika_binary_validation.csv`. Run 22 was then
repurposed as an evaluation-only OOF notebook, so the released-test final fit remains
separate from the OOF comparison. The failed attempt is retained in
`results/task_a/logs/run21_train_validation_test_failed_missing_validation.log`.

## Evaluation-only OOF comparison

`22_run9_oof_evaluation.ipynb` is now evaluation-only. It measures the
second-best Run 9 recipe—demojized MuRIL with two-layer reinitialization plus demojized
TF-IDF/SVM—on five matched folds of the combined labelled corpus, using the fixed 57/43
blend and threshold 0.50. The completed run scored **0.8213 macro-F1 / 0.8215 accuracy**
on 7,193 deduplicated rows. It does not produce a submission ZIP.

`23_oof_compare_run11_run9.ipynb` combines the original labels with the released
labelled validation rows, trains the SVM once and both MuRIL variants over the same
five folds, then reports the individual scores and the locked Run 11 and Run 9 ensemble
scores. The completed run used 7,252 combined rows, deduplicated to 7,193, and reported
0.8204 macro-F1 for Run 11 versus 0.8213 for Run 9. It does not produce a ZIP. These are
local OOF scores, not CodaBench scores, and should not be compared as though they were
the official 806-row result.

Run 23 evaluates the same Run 9 configuration while also comparing it with Run 11; its
Run 9 result matches Run 22, providing a consistency check across the two logs.
