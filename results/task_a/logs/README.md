# Task A execution logs

These are uploaded Kaggle logs, renamed by experiment so they can be found without
depending on the original upload filenames.

| file | experiment | outcome |
|---|---|---|
| `run07_tapt_holdout.log` | Task-A-domain TAPT holdout notebook | training reached both arms; the notebook's final score-summary assertion failed, so no holdout score is recorded |
| `run09_task_a_ensemble.log` | full-data MuRIL + TF-IDF ensemble | ZIPs were written; the preferred ensemble scored 0.8187 macro-F1 on CodaBench |
| `run10_muril_embeddings_svm.log` | frozen MuRIL embeddings + RBF SVM | ZIP was written; the 0.71 CodaBench result was rejected |
| `task_a_full_data_svm_failed.log` | obsolete full-data SVM attempt | failed because the runner did not accept the `--full-fit` argument |
| `run21_train_validation_test_failed_missing_validation.log` | Task A Run 21 train-plus-validation final-fit notebook | stopped before training because the released validation CSV was missing from the Kaggle clone; no ZIP was produced |
| `run22_run9_oof_evaluation.log` | Task A Run 22: focused five-fold OOF evaluation of the Run 9 ensemble | completed; Run 9 locked ensemble 0.8213 macro-F1 and 0.8215 accuracy on 7,193 deduplicated rows |
| `run23_oof_compare_run11_run9.log` | Task A Run 23: five-fold OOF comparison of Run 11 and Run 9 on the combined labelled corpus | completed; Run 11 locked ensemble 0.8204 macro-F1, Run 9 locked ensemble 0.8213 macro-F1 |
| `run20_translate_to_english_failed_ai4bharat.log` | Task A Run 20 translation setup, previous attempt | failed before training because current pip rejected IndicXlit's legacy Fairseq metadata; the notebook was subsequently patched to pin pip 24.0; no model or ZIP from this attempt |
