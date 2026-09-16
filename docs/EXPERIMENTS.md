# Training Results Log

This is the experiment ledger for the project. It separates completed measurements,
provisional findings, and planned runs. Local Task B scores are unbiased OOF macro-F1
unless explicitly marked as a holdout or full fit; full-data fits have no local score.

* **Task A** — binary `Hate` / `Non-Hate` classification.
* **Task B** — six-way target classification: `Gender`, `Geo-political`, `Others`,
  `Political`, `Religion`, and `Violence`.

## Current status — 2026-09-17

| item | current state |
|---|---|
| Best Task A result | MuRIL, **`0.8163`** macro-F1 and `0.8164` accuracy on validation A |
| Best Task B local result | One-layer reinitialization, `0.6102` five-fold OOF macro-F1, Run 5 |
| Best recorded Task B CodaBench result | `b_reinit1_full`, **`0.6299`** — confirmed Run 8 |
| Current Task B candidate | `b_reinit1_full`, one-layer reinitialization, five seeds on all data |
| Next Task A experiment | TAPT + demojized MuRIL notebook, created and ready to run |
| Current branch | `task-b` |
| Official validation size | 395 rows with hidden labels; score via CodaBench |

Run 8 is now the strongest confirmed Task B submission: the one-layer full-data fit scored
`0.6299` on CodaBench. Runs 6 and 7 remain useful scientific checks of the reinitialization
choice, but they no longer block using the Run 8 recipe as the current candidate.

## Experiment roadmap

| run | date/status | notebook or runner | question | result/status |
|---|---|---|---|---|
| Task A 1--4 | completed | Task A training scripts | baseline, demojization, and TF-IDF comparisons | best: `0.8103` |
| Task A 5 | pending | `experiments/task_a/run_rdrop.sh` | test R-Drop against the matched demojized MuRIL control | local OOF and validation submission |
| Task A 6 | 2026-09-17, completed | MuRIL validation submission | compare MuRIL with the TF-IDF result on validation A | **0.8163 macro-F1, 0.8164 accuracy** |
| Task A 7 | 2026-09-17, created; not run | `01_tapt_demojized_muril.ipynb` | test Task-A-domain TAPT before demojized MuRIL fine-tuning | local holdout and validation submissions |
| Task B 1 | 2026-09-12, completed | `01_baseline_sweep.ipynb` | which encoder/loss is useful? | TAPT MuRIL `0.6013` OOF; submitted `0.5922` |
| Task B 2 | 2026-09-13, completed | `02_fullfit_sweep.ipynb` | full-data five-seed versions | `f_tapt` scored `0.6007` on CodaBench, inferred |
| Task B 3 | 2026-09-13--14, completed | `03_factorial_grid.ipynb` | more TAPT text, vocabulary extension, auxiliary head | `D0_V0_noaux` remained best |
| Task B 4 | not run | `04_full_data_fit.ipynb` | original two-layer full-data fit | superseded by the one-layer candidate |
| Task B 5 | 2026-09-16, completed | `05_reinit_one_layer.ipynb` | one versus two reinitialized layers | one layer: `0.6102` vs `0.6013` |
| Task B 6 | pending | `06_reinit_confirmation.ipynb` | confirm one layer at seeds 43/44 | not run |
| Task B 7 | pending | `07_no_reinit_ablation.ipynb` | compare one layer with no reinitialization | not run |
| Task B 8 | 2026-09-17, completed | `08_full_data_fit_reinit1.ipynb` | train current candidate on all data and infer validation | **0.6299 CodaBench** |

## Ordered next steps

1. Run [Run 6](../notebooks/task_b/06_reinit_confirmation.ipynb) and compare one-layer
   versus two-layer reinitialization at model seeds 43 and 44.
2. Run [Run 7](../notebooks/task_b/07_no_reinit_ablation.ipynb) and compare one-layer
   versus zero-layer reinitialization at the same seeds.
3. Select the reinitialization setting using the individual seed scores and the
   probability-averaged OOF score. Prefer a change only if the advantage is consistent.
4. If either ablation is consistently better, train that alternative on all data and
   submit it as a controlled follow-up to Run 8; otherwise retain `b_reinit1_full`.
5. Preserve the Run 8 score, exact recipe, commit, and output metadata in
   `submissions/README.md` and compare future submissions against `0.6299`.

Do not compare a full-data fit to a local OOF score as though they were the same
measurement. A full-data fit has no local F1; its only evaluation is CodaBench. Also treat
small differences on the 395-row official validation set cautiously.

Task A and Task B macro-F1 are **not comparable**: Task A is two classes and Task B is six, so 0.80 on one is not better than 0.60 on the other.

## Task A

## Experiment 1: Baseline MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **Epochs:** 6
* **Best Epoch:** 6
* **Best Macro F1:** 0.7937
* **Output Directory:** `artifacts/checkpoints/muril_task_a`

## Experiment 2: Demojized MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **Epochs:** 6
* **Best Epoch:** 5
* **Best Macro F1:** 0.8023
* **Output Directory:** `artifacts/checkpoints/muril_task_a_demojized`
* **Notes:** Converting emojis to their text equivalents yielded an improvement of ~0.9% in the macro F1 score.

## Experiment 3: TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **5-Fold CV Macro F1:** 0.8039
* **Output Directory:** `artifacts/runs/svm/`
* **Notes:** Surprisingly outperformed Baseline MuRIL, establishing a strong floor.

## Experiment 4: Demojized TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **5-Fold CV Macro F1:** 0.8103
* **Output Directory:** `artifacts/runs/svm_demojize/`
* **Notes:** Previously best performing model. It has now been surpassed by the MuRIL validation submission recorded below.

## Experiment 6: MuRIL validation submission, 2026-09-17
* **Model:** MuRIL
* **Evaluation:** Task A validation A on CodaBench
* **Macro F1:** **0.8163**
* **Accuracy:** **0.8164**
* **Notes:** Best Task A result recorded so far, improving on demojized TF-IDF + LinearSVC (`0.8103`) by `0.0060` macro-F1. The exact flags for the submitted artifact were not provided, so this entry does not assume whether it used R-Drop or another variant.

## Task A follow-up

Run 5 tests whether R-Drop improves the demojized MuRIL model. It uses one seed by
default because R-Drop adds a second stochastic forward pass, but trains both a
matched `--rdrop 0` control and a `--rdrop 0.5` variant under identical five-fold
splits. The script packages both validation predictions; only the stronger candidate
should be submitted after comparing the local OOF scores. This is a model experiment,
not a replacement for the confirmed MuRIL validation result (`0.8163`) until the local
comparison identifies whether it improves the same underlying recipe.

## Experiment 7: Task-A-domain TAPT + demojized MuRIL, notebook created 2026-09-17

[The Kaggle notebook](../notebooks/task_a/01_tapt_demojized_muril.ipynb) compares two
matched Task A arms:

| arm | encoder initialization | classifier fine-tuning |
|---|---|---|
| control | stock `google/muril-base-cased` | demojized MuRIL, two-layer reinitialization |
| TAPT | MuRIL adapted with masked-LM training | the same demojized MuRIL recipe |

The notebook recreates the fixed deduplicate-first, stratified 85/15 split with seed 42.
TAPT sees only the 85% classifier-training-side comments plus permitted OffensEval Kannada
text; the Task A holdout, Task A validation inputs, labels, and Task B files are excluded.
Both arms use seed 42, six classifier epochs, effective batch size 16, FGM, EMA,
`--select last`, and produce validated Task A submission ZIPs. The TAPT stage runs for
eight MLM epochs with a 5% held-out perplexity check. Expected runtime is approximately
2--4 hours on a T4 or RTX 3070.

Outputs are `task_a_muril_control.zip`, `task_a_muril_tapt.zip`, logs, holdout
probabilities, and `task_a_tapt_summary.csv` in the Kaggle Output tab. The experiment is
created and code-validated, but has not been run yet. Its purpose is to decide whether
TAPT is worth a later full-data submission against the current `0.8163` MuRIL result; it
does not assume that the Task B TAPT gain transfers to Task A.

---

## Task B

The completed Kaggle runs and follow-up ablations are listed below. Local figures are the
unbiased `last`-checkpoint number; CodaBench figures are from the 395-row validation phase
with hidden labels. The current status and ordered roadmap are at the top of this document.

### CodaBench validation scores

| submission | run | recipe | macro-F1 |
|---|---|---|---|
| `f_tapt` | 2 | TAPT MuRIL, 5 seeds on all 3,143 deduplicated rows | 0.6007¹ |
| `b_tapt_5f` | 1 | TAPT MuRIL, five fold models averaged | 0.5922 |
| `d0v0_noaux_full` | 4 | TAPT MuRIL, 5 seeds on all 3,159 rows, TAPT on every comment | not run yet |
| `b_reinit1_full` | 8 | TAPT MuRIL, one-layer reinit, 5 seeds on all 3,159 rows | **0.6299** |

¹ Inferred, not confirmed: the `scoring_result.zip` reading 0.6007 was downloaded a few
minutes after `f_tapt.zip`. Check the CodaBench submission list.

One CodaBench score carries about 3.0 points of standard deviation (bootstrap of
macro-F1 at 395 rows), so the 0.85-point gap between these two is not evidence that
either recipe is better.

### Run 1 -- overnight sweep, 2026-09-12

`experiments/task_b/overnight.py` via `01_baseline_sweep.ipynb`, 8.44 h. Encoder and loss ideas ranked
against stock MuRIL on a 15% holdout, the best four promoted to five folds.

Five-fold runs, on all 3,143 rows:

| run | macro-F1 | note |
|---|---|---|
| `b_tapt_5f` | **0.6013** | domain-adapted MuRIL -- **submitted**, 0.5922 on CodaBench |
| `ensemble_b` | 0.6002 | weight-searched blend, nested estimate |
| `svm_b` | 0.5948 | TF-IDF + LinearSVC floor |
| `b_base_5f` | 0.5727 | stock MuRIL, the reference |
| `b_rdrop_5f` | 0.5614 | R-Drop 0.5 |
| `b_focal_5f` | 0.5572 | focal loss, gamma 2.0 |

Holdout arms, 15% of rows (472):

| run | macro-F1 | note |
|---|---|---|
| `b_tapt` | 0.5972 | domain-adapted MuRIL |
| `b_base` | 0.5948 | stock MuRIL |
| `b_rdrop` | 0.5845 | R-Drop |
| `b_focal` | 0.5779 | focal loss |
| `b_abusive` | 0.5718 | abusive-tuned MuRIL warm start |
| `b_hing` | 0.5678 | HingRoBERTa |
| `b_xlmr` | 0.5561 | XLM-R base |
| `b_mdeberta` | 0.1002 | collapsed; 3e-5 is too high for mDeBERTa |
| `b_large` | -- | MuRIL-large, cut by the budget guard |

* **Only task-adaptive pretraining helped.** +2.9 points over stock MuRIL on
  identical folds. Nothing else beat the reference.
* **Holdout ranks correctly but reads about two points high.** For all four
  promoted arms the five-fold order matched the holdout order exactly, while
  the level dropped by roughly 0.022 in three of the four cases.
* **Blending did not pay.** The nested estimate landed below the best single
  model, so `ensemble.py` recommended the single run and that is what was sent.
* **The macro-F1-optimal decode did not pay either.** `src/hastika/task_b/decode.py` fits
  per-class decision weights on the OOF and validates them nested; it lost on
  every run and wrote nothing. On the floor it scored 0.5856 against 0.5948 for
  plain argmax.
* **Prediction distribution is the sanity check that matters.** The floor called
  `Violence` on 6 of 395 test rows against a 7.0% training rate, which under
  macro-F1 forfeits most of a class. The submitted run calls it 24 times.

### Run 2 -- five-seed full fits, 2026-09-13

`experiments/task_b/fullfit.py` via `02_fullfit_sweep.ipynb`, 9.35 h. Six ideas, each trained
five times on all 3,143 deduplicated rows with the probabilities averaged. TAPT still
held 5% of its text back and dropped one-word comments, so it used 6,044 of 6,406.

**No local score exists for these and none can.** The `holdout` column is the Run 1
holdout score of the same idea, copied in for ranking, not a measurement of these models.

| arm | holdout (Run 1) | agrees with `b_tapt_5f` | largest drift from prior | note |
|---|---|---|---|---|
| `f_tapt` | 0.5972 | 91.1% | 3.2 points | MLM-adapted MuRIL -- submitted, 0.6007 on CodaBench¹ |
| `f_base` | 0.5948 | 87.1% | 2.7 | stock MuRIL |
| `f_rdrop` | 0.5845 | 85.8% | 3.0 | R-Drop 0.5 |
| `f_focal` | 0.5779 | 85.1% | 4.0 | focal loss, gamma 2 |
| `f_abusive` | 0.5718 | 81.3% | 3.0 | abusive-tuned warm start |
| `f_hing` | 0.5678 | 81.0% | 3.9 | HingRoBERTa |

### Run 3 -- data-processing grid, 2026-09-13 to 14

`experiments/task_b/grid.py` via `03_factorial_grid.ipynb`, 8.87 h. Every cell is a 15% holdout run on
the same 472 rows, so these are real scores.

* `corpus` D0 = TAPT on Kannada text only; D1 = plus Tamil and Malayalam
  OffensEval-Dravidian, about 58,000 comments, one epoch so the compute matches.
* `vocab` V0 = MuRIL's tokenizer as shipped; V1 = extended with frequent word-forms.
* `aux` = auxiliary head predicting violent language, weight 0.3.

| config | corpus | vocab | aux | seeds | macro-F1 | vs stock MuRIL |
|---|---|---|---|---|---|---|
| `D0_V0_noaux` | D0 | V0 | no | 1 | **0.5972** | +0.0024 |
| `stock_V0_noaux` | - | V0 | no | 1 | 0.5948 | +0.0000 |
| `D0_V0_noaux_multiseed` | D0 | V0 | no | 3 | 0.5917 | -0.0031 |
| `D0_V0_aux` | D0 | V0 | yes | 1 | 0.5916 | -0.0032 |
| `D1_V0_aux` | D1 | V0 | yes | 1 | 0.5875 | -0.0073 |
| `D1_V0_noaux` | D1 | V0 | no | 1 | 0.5865 | -0.0083 |
| `stock_V0_noaux_multiseed` | - | V0 | no | 3 | 0.5820 | -0.0128 |
| `D0_V1_noaux` | D0 | V1 | no | 1 | 0.5656 | -0.0292 |
| `D0_V1_aux` | D0 | V1 | yes | 1 | 0.5546 | -0.0402 |
| `stock_V0_aux` | - | V0 | yes | 1 | 0.5495 | -0.0453 |
| `D1_V1_noaux` | D1 | V1 | no | 1 | 0.5117 | -0.0831 |
| `D1_V1_aux` | D1 | V1 | yes | 1 | 0.5071 | -0.0877 |

* **None of the three levers helped.** `D0_V0_noaux` is the same recipe as `b_tapt`
  and reproduced its 0.5972 exactly.
* **The top seven cells are one result.** The holdout's macro-F1 has a bootstrap
  standard deviation of 2.7 points and those cells span 1.5.
* **Vocabulary extension is the only real effect, and it hurts:** 3 to 8 points below
  the same cell with V0. New embeddings, initialized as the mean of their pieces, do
  not get enough training from 3k comments.
* **Three seeds scored below seed 42 alone** for both stock and TAPT, so seed 42 was
  a lucky draw on this split.
* **The grid is tilted toward TAPT.** TAPT read every training comment's text,
  including the 472 holdout rows (never their labels). The case for TAPT rests on
  Run 1's five-fold comparison instead.
* **`subs/D0_V0_noaux.zip` is not a submission candidate.** It is the single-seed
  holdout model trained on 85% of the rows. The grid's own five-seed full fit only
  starts with about two hours of budget left and the run ended at 8.87 of 10.25 h, so
  it most likely skipped; check the Kaggle output for `subs/winner_full.zip`.

### Run 4 -- `D0_V0_noaux` on 100% of the data, not run yet

`notebooks/task_b/04_full_data_fit.ipynb`, about 2 h. The Run 3 winner with nothing held back
at either stage:

| stage | `f_tapt` (Run 2) | Run 4 |
|---|---|---|
| TAPT text | 6,044 of 6,406 (94.3%) | **6,406 of 6,406** |
| classifier rows per model | 3,143 of 3,159 (deduplicated) | **3,159 of 3,159** |
| seeds | 42-46 | 42-46 |

Its only score will be CodaBench's. Record it in the table at the top of this section.

### Run 5 -- one-layer reinitialization ablation, 2026-09-16

`notebooks/task_b/05_reinit_one_layer.ipynb` tests whether reinitializing only the final
MuRIL encoder layer is better than the established two-layer reset. The notebook builds
one shared TAPT checkpoint, then runs both classifier variants with the same deduplicated
Task B data, five-fold split, split seed 42, model seed 42, six epochs, all existing
regularization, no auxiliary head, and `--select last`. This isolates the
`--reinit-layers` choice:

| run | reinitialized layers | status |
|---|---:|---|
| `b_tapt_reinit2_5f` | 2 (control) | 0.6013 OOF macro-F1 |
| `b_tapt_reinit1_5f` | 1 (ablation) | **0.6102 OOF macro-F1** |

The run used one shared TAPT checkpoint (`6044` of `6362` cleaned comments, with `318`
held out for MLM perplexity), the deduplicated 3,143-row Task B set, five folds, seed 42,
six epochs, the existing regularization, no auxiliary head, and `--select last`. The
one-layer variant improved macro-F1 by `+0.0089` and accuracy from `0.6557` to `0.6592`.
Per-class F1 changed as follows:

| class | two layers | one layer | delta |
|---|---:|---:|---:|
| Gender | 0.735 | 0.730 | -0.005 |
| Geo-political | 0.592 | 0.590 | -0.002 |
| Others | 0.454 | 0.476 | +0.022 |
| Political | 0.783 | 0.797 | +0.014 |
| Religion | 0.716 | 0.717 | +0.001 |
| Violence | 0.328 | 0.352 | +0.024 |

This is encouraging but is still one five-fold run with model seed 42. The notebook's
independent comparison cell failed after training because the live kernel did not refresh
`PYTHONPATH`; the training logs produced the scores above. The source notebook now inserts
`src/` into the live kernel path. Confirm the one-layer result with seeds 43 and 44 before
changing the recommended recipe. There is no CodaBench submission in this experiment.

### Run 6 -- confirm one-layer reinitialization, not run yet

`notebooks/task_b/06_reinit_confirmation.ipynb` repeats Run 5 at model seeds 43 and 44.
It builds one shared TAPT checkpoint and runs four separate five-fold comparisons:

| run | reinitialized layers | model seed | status |
|---|---:|---:|---|
| `b_tapt_reinit2_s43` | 2 (control) | 43 | not run yet |
| `b_tapt_reinit1_s43` | 1 (ablation) | 43 | not run yet |
| `b_tapt_reinit2_s44` | 2 (control) | 44 | not run yet |
| `b_tapt_reinit1_s44` | 1 (ablation) | 44 | not run yet |

All settings other than `--reinit-layers` match Run 5. The notebook reports each seed's
OOF macro-F1, the mean of the two seed scores, and macro-F1 after averaging the two seeds'
OOF probabilities. It also prints per-class reports. There is no submission in this run;
the one-layer recipe should be promoted to the final full-data experiment only if its
advantage survives this confirmation.

### Run 7 -- no encoder-layer reinitialization ablation, not run yet

`notebooks/task_b/07_no_reinit_ablation.ipynb` tests whether preserving the final encoder
layer as well is better than the current one-layer setting. `--reinit-layers 1` is the
control; `--reinit-layers 0` keeps both layers 11 and 12 from the TAPT checkpoint:

| run | reinitialized layers | model seed | status |
|---|---:|---:|---|
| `b_tapt_reinit1_vs0_s43` | 1 (control) | 43 | not run yet |
| `b_tapt_reinit0_s43` | 0 (ablation) | 43 | not run yet |
| `b_tapt_reinit1_vs0_s44` | 1 (control) | 44 | not run yet |
| `b_tapt_reinit0_s44` | 0 (ablation) | 44 | not run yet |

All other settings match Runs 5 and 6: one shared TAPT checkpoint, deduplicated Task B
data, fixed five-fold split with split seed 42, six epochs, existing regularization, no
auxiliary head, and `--select last`. The notebook reports each seed, the mean seed score,
the score after averaging OOF probabilities, and per-class F1. There is no submission in
this run; use `--reinit-layers 0` for the final fit only if it consistently beats the
one-layer control.

### Run 8 -- full-data one-layer fit and official validation inference, 2026-09-17

`notebooks/task_b/08_full_data_fit_reinit1.ipynb` is the provisional final-fit notebook
for the currently best observed recipe. It uses `--reinit-layers 1`, TAPT on all 6,406
allowed comments, and five classifier seeds (`42--46`) trained on all 3,159 labelled Task
B rows. It then averages the five validation probability matrices and predicts all 395
rows in `data/raw/multiclass_validation_inputs.csv`.

| output | purpose | status |
|---|---|---|
| `b_reinit1_full` | five-seed full-data predictions and probabilities | completed |
| `b_reinit1_full.zip` | validated `id,label` payload for CodaBench | **0.6299** |

The uploaded log `fulldatafit08.log` confirms TAPT on all 6,406 comments and five classifier
seeds (`42--46`) on all 3,159 labelled rows with `--reinit-layers 1`. The submission helper
validated 395 rows and a ZIP containing one bare `predictions.csv`. CodaBench returned
macro-F1 `0.6299` and accuracy `0.6886`, making this the best confirmed Task B output so far.

The notebook's final diagnostic cell raised `NameError: pd is not defined` after the ZIP had
already been written. This affected only the distribution-reporting cell, not training,
prediction generation, ZIP validation, or the submitted result; the canonical notebook now
imports pandas before that cell.
