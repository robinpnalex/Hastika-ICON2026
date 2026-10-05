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

The logs are retained as raw captured output. The corresponding experiment descriptions
and results are recorded in `docs/EXPERIMENTS.md`.
