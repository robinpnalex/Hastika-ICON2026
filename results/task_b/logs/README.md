# Task B run logs

These are the captured Kaggle logs for the completed Task B experiments:

| file | experiment |
|---|---|
| `run06_reinit_confirmation.log` | Run 6: one versus two reinitialized layers at seeds 43 and 44 |
| `run07_no_reinit_ablation.log` | Run 7: one versus zero reinitialized layers at seeds 43 and 44 |
| `run08_full_data_reinit1.log` | Run 8: five-seed full-data one-layer fit submitted to CodaBench |
| `run09_rdrop_full.log` | Run 9: five-seed full-data one-layer R-Drop fit, scored 0.6410 on CodaBench |
| `run12_context_tags_full.log` | Run 12: all five context-tagged full-data fits completed; packaging stopped on an incorrect post-training R-Drop log assertion |
| `run17_muril_large_oof_oom.log` | Run 17: MuRIL-large five-fold OOF evaluation | TAPT completed, but classifier fold 1 stopped with CUDA out-of-memory; no OOF score |
| `run18_train_validation_test_failed_clone.log` | Run 18: train-plus-validation final fit for released Task B test data | stopped before training because Kaggle could not clone the GitHub repository; no ZIP was produced |
| `run20_llm_screen_holdout.log` | Run 20: LLM screen + MuRIL baseline on the 530-row holdout | MuRIL 0.6151; Sarvam-1 0.5567; Gemma-4-12B screened best but its fine-tune crashed on peft/torchao |
| `run21_gemma12b_holdout_full.log` | Run 21: Gemma-4-12B QLoRA holdout + full fits | Gemma 2 seeds 0.6792 vs MuRIL 0.6151 (+0.064, CI [+0.024, +0.105]); packaging cell missed the LLM files |
| `run22_gemma_3ep_2seeds_full.log` | Run 22: standalone Gemma-4-12B full fits, 3 epochs, seeds 42/43, on all 3,532 rows | both fits completed (77 and 93 min); `RECOMMENDED_b22_gemma_3ep_2seeds.zip` written |
| `run24_gemma_rdrop_tapt_holdout.log` | Run 24: MuRIL lessons on Gemma-4-12B, holdout | base 0.6753, R-Drop 0.6734 (-0.002, CI [-0.027, +0.023]); TAPT arm OOM in the LM-head logits, not measured |
| `run23_gemma_4ep_holdout_full.log` | Run 23: Gemma-4-12B 4 epochs on the holdout, then four 4-epoch full fits | 4 epochs 0.6872 vs 3 epochs 0.6792 (2 seeds each); seed 43 near-collapsed through epoch 2, recovered; `RECOMMENDED_b23_gemma.zip` (4 models) |
| `run19_oof_best_recipe.log` | Run 19: MuRIL best recipe OOF on train + released validation | see `docs/EXPERIMENTS.md` |
| `run30_gemma_prompt_ensemble_holdout.log` | Run 30 (Robin): Gemma prompt ensemble on the holdout | see `docs/EXPERIMENTS.md` |
| `run31_gemma_class_weight_holdout.log` | Run 31 (Robin): square-root vs balanced class weights | see `docs/EXPERIMENTS.md` |
| `run01_baseline_sweep_holdout_arms.log` | Run 1 (2026-09-12): holdout arms of the overnight sweep | b_base 0.5948, b_abusive 0.5718 (as recorded) |
| `run15_muril_large_full.log` | Run 15 (2026-09-20): MuRIL-large full fit on all 3,159 rows | completed; `b_muril_large_full.zip` (validation phase, never scored) |
| `run16_rdrop_augmented_full.log` | Run 16 (2026-09-20): R-Drop recipe on 3,159 gold + 779 multiclass-labelled binary rows | completed; `b_reinit1_rdrop_augmented_full.zip` (validation phase, never scored; uses cross-task labels) |
| `run36_tapt_rdrop_holdout.log` | Run 36: TAPT (2 epochs) + R-Drop vs TAPT, one adapter, seeds 42/43 | TAPT x2 0.6741; TAPT + R-Drop x2 **0.7026**; all four 0.6966 |
| `run37_tapt_rdrop_full.log` | Run 37: TAPT + R-Drop, two 4-epoch full fits | completed in 6.5 h; `RECOMMENDED_b37_gemma_tapt_rdrop.zip` |
| `run38_tapt_bag_full_t1.log` | Run 38, bag session 1, full data | completed in 4.3 h; `RECOMMENDED_b38_bag_t1.zip` |
| `run39_tapt_bag_holdout_t1.log` | Run 39, bag session 1, holdout | pair **0.7056**; seed 43 collapsed (all Violence) and its rerun as 1043 trained normally |
| `run38_tapt_bag_full_t2.log` | Run 38, bag session 2 (TAPT seed 2, seeds 44/45), full data | completed in 4.4 h; `RECOMMENDED_b38_bag_t2.zip` |
| `run38_tapt_bag_full_t3.log` | Run 38, bag session 3 (TAPT seed 3, seeds 46/47), full data | completed in 4.5 h; `RECOMMENDED_b38_bag_t3.zip` |

The logs are retained as raw captured output. The corresponding experiment descriptions
and results are recorded in `docs/EXPERIMENTS.md`.
