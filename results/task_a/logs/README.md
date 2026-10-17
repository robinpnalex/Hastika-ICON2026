# Task A execution logs

These are uploaded Kaggle logs, renamed by experiment so they can be found without
depending on the original upload filenames.

| file | experiment | outcome |
|---|---|---|
| `run07_tapt_holdout.log` | Task-A-domain TAPT holdout notebook | training reached both arms; the notebook's final score-summary assertion failed, so no holdout score is recorded |
| `run09_task_a_ensemble.log` | full-data MuRIL + TF-IDF ensemble | ZIPs were written; the preferred ensemble scored 0.8187 macro-F1 on CodaBench |
| `run10_muril_embeddings_svm.log` | frozen MuRIL embeddings + RBF SVM | ZIP was written; the 0.71 CodaBench result was rejected |
| `legacy_full_data_svm_failed.log` | obsolete full-data SVM attempt | failed because the runner did not accept the `--full-fit` argument |
| `run21_train_validation_test_failed_missing_validation.log` | Task A Run 21 train-plus-validation final-fit notebook | stopped before training because the released validation CSV was missing from the Kaggle clone; no ZIP was produced |
| `run22_run9_oof_evaluation.log` | Task A Run 22: focused five-fold OOF evaluation of the Run 9 ensemble | completed; Run 9 locked ensemble 0.8213 macro-F1 and 0.8215 accuracy on 7,193 deduplicated rows |
| `run23_oof_compare_run11_run9.log` | Task A Run 23: five-fold OOF comparison of Run 11 and Run 9 on the combined labelled corpus | completed; Run 11 locked ensemble 0.8204 macro-F1, Run 9 locked ensemble 0.8213 macro-F1 |
| `run20_translate_to_english_failed_ai4bharat.log` | Task A Run 20 translation setup, previous attempt | failed before training because current pip rejected IndicXlit's legacy Fairseq metadata; the notebook was subsequently patched to pin pip 24.0; no model or ZIP from this attempt |
| `run20_translate_hatebert_holdout.log` | Task A Run 20: Sarvam-Translate -> hateBERT on the 1,079-row holdout | hateBERT 0.7275 vs MuRIL 0.7803 and SVM+MuRIL 0.8155; translation line closed |
| `run28_gemma_holdout_full.log` | Task A Run 28: Gemma-4-12B QLoRA on the 1,079-row holdout, then full fits | seed 42 **0.8563** vs char SVM 0.8117 (+0.043, CI [+0.021, +0.067]); seed 43 collapsed (all Non-Hate); both full fits healthy; `RECOMMENDED_a28_gemma.zip` |
| `run29_gemma_epochs.log` | Task A Run 29: three versus four Gemma epochs on the fixed holdout | completed; three-epoch two-seed blend **0.8610 / 0.8610**, four-epoch blend 0.8452; no final test artifact from this notebook |
| `run31_gemma_tapt_holdout.log` | Task A Run 31: TAPT plus Gemma classification on the fixed holdout | TAPT failed with CUDA OOM; baseline completed at **0.8610 / 0.8610**; no TAPT score or ZIP |
| `run33_gemma_full_4seed.log` | Uploaded legacy Task A Run 33 full-data Gemma fit | four non-TAPT seeds completed and `RECOMMENDED_a33_gemma.zip` was written in Kaggle; this log predates the current TAPT version of notebook 33 and was not selected for final submission |
| `run25_char_svm_third_member_oof.log` | Task A Run 25: char-only TF-IDF as a third member, OOF | see `docs/EXPERIMENTS.md` |
| `run26_transductive_derivable_labels_oof.log` | Task A Run 26: derivable-label diagnostic, OOF | see `docs/EXPERIMENTS.md` |
| `run27_run25_validation_and_test.log` | Task A Run 27: Run 25 recipe, validation and test predictions | see `docs/EXPERIMENTS.md` |
| `run34_final_holdout_check.log` | Task A Run 34: holdout twin of Run 28 + a 4-epoch Run 33 | 3 epochs x2 **0.8610** (best Task A holdout); 4 epochs x2 0.8313; all four 0.8563 (-0.005 vs 3 epochs x2); seed 43 collapsed, auto-rerun as 1043, 0.8573 |
| `run35_tapt_holdout_4x4_modified.log` | Task A Run 35, a modified copy (4 base + 4 TAPT seeds), commit `d138f23` | **TAPT x4 0.8702**; base x3 0.8582 (base seed 43 cut at the 11 h stop); TAPT single seeds 0.848-0.857 |
| `run33_tapt_full.log` | Task A Run 33, TAPT (1 epoch) + four 3-epoch full fits, commit `de2fc07` | completed in 7.6 h; TAPT peak 10.4 GB; `RECOMMENDED_a33_gemma.zip` (TAPT), 50.2% Hate |
| `run33_notapt_4seed_full.log` | Task A Run 33, no TAPT, four 3-epoch full fits, commit `a99bd69` | completed in 5.6 h; `RECOMMENDED_a33_gemma.zip` (no TAPT), 51.5% Hate |
| `run11_reinit1_full_data.log` | Task A Run 11: one reinit layer + refitted blend | OOF nested blend 0.8164 (as recorded); CodaBench 0.8188 |
| `run12_tapt_oof.log` | Task A Run 12: TAPT vs stock MuRIL, five-fold | stock 0.7894; TAPT 0.8128 (as recorded) |
| `run16_funnel.log` | Task A Run 16: the funnel, stage 1 | epochs10 +0.028, large +0.021, external arms dead (as recorded) |
| `run17_final_submission_transductive.log` | Task A Run 17 (2026-09-20): full-corpus TAPT, 10 epochs, 3 seeds, **transductive**, SVM blend | completed; `task_a_final.zip` (validation phase, never scored; uses cross-task derived labels, so excluded from the final) |
| `run37_tapt_rdrop_full.log` | Task A Run 37: TAPT (1 epoch) + R-Drop 0.5, two 3-epoch full fits, commit `360c131` | completed in 7.5 h; `RECOMMENDED_a37_gemma_tapt_rdrop.zip` (51.5% Hate); unvalidated until Run 36 |
| `run38_tapt_bag_full_t1.log` | Task A Run 38, bag session 1 (`TAPT_SEED` 1, seeds 42/43), full data | completed in 4.7 h; `RECOMMENDED_a38_bag_t1.zip` (51.9% Hate) |
| `run39_tapt_bag_holdout_t1.log` | Task A Run 39, bag session 1, holdout | singles 0.8535 / 0.8610, **pair 0.8675** |
| `run36_tapt_rdrop_holdout.log` | Task A Run 36: TAPT + R-Drop vs TAPT, one adapter, seeds 42/43 | TAPT x2 0.8619; TAPT + R-Drop x2 0.8647 (+0.003, P 0.64); **all four 0.8721**, the best Task A holdout |
| `run38_tapt_bag_full_t2.log` | Task A Run 38, bag session 2 (TAPT seed 2, seeds 44/45), full data | completed in 4.4 h; `RECOMMENDED_a38_bag_t2.zip` (49.8% Hate) |
| `run38_tapt_bag_full_t3.log` | Task A Run 38, bag session 3 (TAPT seed 3, seeds 46/47), full data | completed in 4.8 h; `RECOMMENDED_a38_bag_t3.zip` (52.5% Hate) |
