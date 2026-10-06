# Final-submission notebook status

This is the compact checklist for the strongest Task A and Task B candidates. It records
the best reported evaluation for each distinct recipe, the notebook that measured it, and
whether a full-data submission notebook exists and has actually been run.

Scores are not all directly comparable: some are fixed-holdout scores, some are five-fold
OOF scores, and older results are official CodaBench scores. The protocol is shown in the
score column deliberately.

Status meanings:

- **Exists / run:** the notebook is in the repository and its execution completed.
- **Exists / not run:** the notebook is present, but no completed execution is recorded.
- **No dedicated notebook:** the result is recorded, but the current repository does not
  identify a separate final-submission notebook for that candidate.

## Task A — top five candidates

| rank | candidate | strongest reported result | evaluation notebook | final-submission notebook | status | ZIP status |
|---:|---|---|---|---|---|---|
| 1 | Gemma-4-12B QLoRA, Run 28 | **0.8563 macro-F1** on the fixed 1,079-row holdout | `task_a/28_gemma_holdout_final.ipynb` | `task_a/28_gemma_holdout_final.ipynb` | Exists / run | **Yes:** `RECOMMENDED_a28_gemma.zip` |
| 2 | Run 9 demojized MuRIL + TF-IDF ensemble | **0.8213 / 0.8215** five-fold OOF; **0.8187 / 0.8189** CodaBench | `task_a/22_run9_oof_evaluation.ipynb`, `23_oof_compare_run11_run9.ipynb` | `task_a/08_muril_tfidf_ensemble.ipynb` | Exists / run | **Yes:** notebook produced `task_a_full_ensemble.zip`; not stored in the current tracked submissions folder |
| 3 | Run 11 one-layer MuRIL + TF-IDF ensemble | **0.8204 / 0.8204** five-fold OOF; **0.8188 / 0.8189** CodaBench | `task_a/23_oof_compare_run11_run9.ipynb` | `task_a/11_reinit1_full_data.ipynb` | Exists / run | **Yes:** notebook builds `task_a_reinit1_refit_blend.zip`; artifact not currently tracked |
| 4 | MuRIL validation submission | **0.8163 / 0.8164** CodaBench | recorded as Task A Run 6; no named evaluation notebook identified | No dedicated notebook identified | Result recorded; notebook status unknown | Not tracked |
| 5 | Task-A-domain TAPT + demojized MuRIL | **0.8128** five-fold OOF | `task_a/12_tapt_oof.ipynb` | None; this notebook is evaluation-only | Exists / run; no final-fit notebook | No submission ZIP from this OOF run |

The current Task A final-submission artifact is therefore Run 28's Gemma ZIP. Task A
Run 33 is a newer planned four-seed full-data recipe, but it has not been run yet.

## Task B — top five candidates

| rank | candidate | strongest reported result | evaluation notebook | final-submission notebook | status | ZIP status |
|---:|---|---|---|---|---|---|
| 1 | Run 30 four-model Gemma prompt ensemble: standard + definitions prompts, two seeds each | **0.6882 / 0.7321** on the fixed 530-row holdout | `task_b/30_gemma_prompt_ensemble_holdout.ipynb` | `task_b/32_gemma_prompt_ensemble_full.ipynb` | Evaluation exists / run; final notebook exists / **not run** | No ZIP yet; Run 32 will create `RECOMMENDED_b32_gemma_prompt_ensemble.zip` |
| 2 | Run 23 Gemma-4-12B, 4 epochs, four full-data models | **0.6872** on the fixed holdout | `task_b/23_gemma_epochs_seeds_final.ipynb` | Same notebook | Exists / run | **Yes:** `RECOMMENDED_b23_gemma.zip` |
| 3 | Run 31 Gemma with balanced class weights | **0.6864 / 0.7283** on the fixed holdout | `task_b/31_gemma_class_weight_holdout.ipynb` | `task_b/25_gemma_final_recipe.ipynb` is the compatible full-data recipe | Evaluation exists / run; final notebook exists / **not run** | No ZIP for Run 31 |
| 4 | Run 21 50/50 Gemma + MuRIL ensemble | **0.6825 / 0.730** on the fixed holdout | `task_b/21_llm_confirm_final.ipynb` | Same notebook | Exists / run | Ensemble packaging was attempted, but no retained recommended ensemble ZIP |
| 5 | Run 21 two-seed Gemma-4-12B | **0.6792 / 0.728** on the fixed holdout | `task_b/21_llm_confirm_final.ipynb` | `task_b/22_package_run21.ipynb` | Exists / run | **Yes:** `RECOMMENDED_b22_gemma_3ep_2seeds.zip` |

Run 24's R-Drop model scored 0.6734, and Run 31's square-root-weight variant scored
0.6822; neither replaces the candidates above. Run 25 remains an unrun four-seed
single-prompt full-data recipe.

## Immediate final-submission choices

| task | safest already-generated ZIP | highest-scoring unsubmitted candidate |
|---|---|---|
| A | Run 28: `RECOMMENDED_a28_gemma.zip` | Run 33, once run |
| B | Run 23: `RECOMMENDED_b23_gemma.zip` | Run 32, the Run 30 prompt ensemble full-data fit |

The Task B Run 30 score is only modestly above Run 23 and its bootstrap comparison was
inconclusive (P(better) = 0.67), so Run 23 is the available fallback while Run 32 is the
optional higher-scoring final prediction run.
