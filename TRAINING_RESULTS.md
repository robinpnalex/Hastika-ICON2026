# Training Results Log

This document tracks the experimental runs and their results.

* **Task A** — Binary Classification: Hate vs Non-Hate. Experiments 1-4 below.
* **Task B** — Fine-Grained six-way target category. Four Kaggle runs and the CodaBench scores, below.

Task A and Task B macro-F1 are **not comparable**: Task A is two classes and Task B is six, so 0.80 on one is not better than 0.60 on the other.

## Task A

## Experiment 1: Baseline MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **Epochs:** 6
* **Best Epoch:** 6
* **Best Macro F1:** 0.7937
* **Output Directory:** `checkpoints/muril_task_a`

## Experiment 2: Demojized MuRIL
* **Model:** `google/muril-base-cased`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **Epochs:** 6
* **Best Epoch:** 5
* **Best Macro F1:** 0.8023
* **Output Directory:** `checkpoints/muril_task_a_demojized`
* **Notes:** Converting emojis to their text equivalents yielded an improvement of ~0.9% in the macro F1 score.

## Experiment 3: TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default (HTML cleaning, mojibake repair)
* **5-Fold CV Macro F1:** 0.8039
* **Output Directory:** `work/runs/svm/`
* **Notes:** Surprisingly outperformed Baseline MuRIL, establishing a strong floor.

## Experiment 4: Demojized TF-IDF + LinearSVC Baseline
* **Model:** `TfidfVectorizer` (Word + Char n-grams) + `LinearSVC`
* **Preprocessing:** Default + Demojized (Emojis converted to English text descriptions)
* **5-Fold CV Macro F1:** 0.8103
* **Output Directory:** `work/runs/svm_demojize/`
* **Notes:** Best performing model so far, outperforming both standard TF-IDF and Demojized MuRIL.

## Next Planned Experiments
* **Experiment 5:** XLM-RoBERTa Base (`xlm-roberta-base`) with Demojized text.

---

## Task B

Four Kaggle runs on a T4. Local figures are the unbiased `last`-checkpoint number; the
CodaBench figures are the validation phase, 395 rows with hidden labels. The pipeline
and the plan are in the README's Task B section.

### CodaBench validation scores

| submission | run | recipe | macro-F1 |
|---|---|---|---|
| `f_tapt` | 2 | TAPT MuRIL, 5 seeds on all 3,143 deduplicated rows | 0.6007¹ |
| `b_tapt_5f` | 1 | TAPT MuRIL, five fold models averaged | 0.5922 |
| `d0v0_noaux_full` | 4 | TAPT MuRIL, 5 seeds on all 3,159 rows, TAPT on every comment | not run yet |

¹ Inferred, not confirmed: the `scoring_result.zip` reading 0.6007 was downloaded a few
minutes after `f_tapt.zip`. Check the CodaBench submission list.

One CodaBench score carries about 3.0 points of standard deviation (bootstrap of
macro-F1 at 395 rows), so the 0.85-point gap between these two is not evidence that
either recipe is better.

### Run 1 -- overnight sweep, 2026-09-12

`work/overnight_b.py` via `kaggle_task_b.ipynb`, 8.44 h. Encoder and loss ideas ranked
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
* **The macro-F1-optimal decode did not pay either.** `work/decode_b.py` fits
  per-class decision weights on the OOF and validates them nested; it lost on
  every run and wrote nothing. On the floor it scored 0.5856 against 0.5948 for
  plain argmax.
* **Prediction distribution is the sanity check that matters.** The floor called
  `Violence` on 6 of 395 test rows against a 7.0% training rate, which under
  macro-F1 forfeits most of a class. The submitted run calls it 24 times.

### Run 2 -- five-seed full fits, 2026-09-13

`work/fullfit_b.py` via `kaggle_fullfit_sweep.ipynb`, 9.35 h. Six ideas, each trained
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

`work/grid_b.py` via `kaggle_grid.ipynb`, 8.87 h. Every cell is a 15% holdout run on
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

`work/kaggle_d0v0_noaux_full.ipynb`, about 2 h. The Run 3 winner with nothing held back
at either stage:

| stage | `f_tapt` (Run 2) | Run 4 |
|---|---|---|
| TAPT text | 6,044 of 6,406 (94.3%) | **6,406 of 6,406** |
| classifier rows per model | 3,143 of 3,159 (deduplicated) | **3,159 of 3,159** |
| seeds | 42-46 | 42-46 |

Its only score will be CodaBench's. Record it in the table at the top of this section.
