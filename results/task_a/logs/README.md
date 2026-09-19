# Task A execution logs

These are uploaded Kaggle logs, renamed by experiment so they can be found without
depending on the original upload filenames.

| file | experiment | outcome |
|---|---|---|
| `run07_tapt_holdout.log` | Task-A-domain TAPT holdout notebook | training reached both arms; the notebook's final score-summary assertion failed, so no holdout score is recorded |
| `run09_task_a_ensemble.log` | full-data MuRIL + TF-IDF ensemble | ZIPs were written; the preferred ensemble scored 0.8187 macro-F1 on CodaBench |
| `run10_muril_embeddings_svm.log` | frozen MuRIL embeddings + RBF SVM | ZIP was written; the 0.71 CodaBench result was rejected |
| `task_a_full_data_svm_failed.log` | obsolete full-data SVM attempt | failed because the runner did not accept the `--full-fit` argument |
